#include "Scene.h"

#include "Pipelines.h"
#include "gpu/JiData.h"
#include "metal/Buffer.h"
#include "metal/Context.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <semaphore>
#include <stdexcept>
#include <string>
#include <tuple>

namespace ji {
namespace {

Vec3 ToVec3(rbp::float3 v) { return {v.x, v.y, v.z}; }
rbp::float3 ToFloat3(Vec3 v) { return {float(v[0]), float(v[1]), float(v[2])}; }
Vec3 Scale(Vec3 v, double s) { return {v[0] * s, v[1] * s, v[2] * s}; }
Vec3 Add(Vec3 a, Vec3 b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }

std::array<Vec3, 3> ContactAxes(Vec3 normal) {
    const auto cross = [](Vec3 a, Vec3 b) -> Vec3 {
        return {a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]};
    };
    const auto normalize = [](Vec3 v) {
        const double length = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
        if (!(length > 0) || !std::isfinite(length)) throw std::invalid_argument("Invalid contact normal");
        return Scale(v, 1 / length);
    };
    normal = normalize(normal);
    const Vec3 seed = std::abs(normal[0]) < 0.8 ? Vec3{1, 0, 0} : Vec3{0, 1, 0};
    const Vec3 tangent = normalize(cross(normal, seed));
    return {normal, tangent, cross(normal, tangent)};
}

void CheckFloat(double value) {
    if (!std::isfinite(float(value)))
        throw std::invalid_argument("Ji scene input exceeds finite FP32 range");
}
void CheckFloat(Vec3 value) {
    for (double component : value) CheckFloat(component);
}

NS::SharedPtr<MTL4::ArgumentTable> NewTable(const rbp::mtl::Context &context, uint32_t bindings) {
    NS::Error *error{};
    auto descriptor = rbp::mtl::Make<MTL4::ArgumentTableDescriptor>();
    descriptor->setMaxBufferBindCount(bindings);
    auto table = NS::TransferPtr(context.Device->newArgumentTable(descriptor.get(), &error));
    if (!table) throw std::runtime_error("Cannot create Ji scene argument table");
    return table;
}

ContactKey Key(const rbp::World &world, const rbp::GeometryContact &raw) {
    return {world.IdOf(raw.BodyA), world.IdOf(raw.BodyB),
            world.BodyShapes[raw.BodyA], world.BodyShapes[raw.BodyB],
            rbp::OwnChild(raw.Children), rbp::OtherChild(raw.Children),
            raw.SubShapeA, raw.SubShapeB, raw.Feature};
}

} // namespace

bool ContactKey::operator<(const ContactKey &other) const {
    return std::tie(a.Slot, a.Spawn, b.Slot, b.Spawn, shape_a, shape_b, child_a, child_b,
                    subshape_a, subshape_b, feature) <
           std::tie(other.a.Slot, other.a.Spawn, other.b.Slot, other.b.Spawn, other.shape_a,
                    other.shape_b, other.child_a, other.child_b, other.subshape_a, other.subshape_b,
                    other.feature);
}

struct Scene::Workspace {
    enum Stage : uint32_t {
        BuildBodies, BuildContacts, BuildJoints, SetCount, ResetIncidence, CountIncidence,
        PrefixIncidence, FillIncidence, SortIncidence, ClearWarm, BuildWarmIndex, SeedContacts,
        ValidateState, FinalizeStep, CacheContacts,
        PublishCache, Integrate, StageCount
    };
    const rbp::mtl::Context &Context;
    MetalPasses Admm;
    std::array<NS::SharedPtr<MTL::ComputePipelineState>, StageCount> Pipelines;
    NS::SharedPtr<MTL::ResidencySet> Residency;
    NS::SharedPtr<MTL4::CounterHeap> ProfileHeap;
    uint32_t ProfileCapacity = 0;
    NS::SharedPtr<MTL4::CommandAllocator> Allocator;
    NS::SharedPtr<MTL4::CommandBuffer> Commands;
    NS::SharedPtr<MTL4::ArgumentTable> BodyTable, ContactTable, JointTable, IncidenceTable;
    NS::SharedPtr<MTL4::ArgumentTable> SolveTable, WarmTable, IntegrateTable, ValidateTable, SnapshotTable;
    rbp::mtl::Buffer<uint32_t> BodyMap, BodySlots, BodyComponents, ComponentContacts;
    rbp::mtl::Buffer<uint32_t> Spawns, SceneCounts, IncidenceCounts;
    rbp::mtl::Buffer<uint32_t> Offsets, Cursor, Incidence, ContributionSlots;
    rbp::mtl::Buffer<uint32_t> Claims, WarmCount, WarmHeads, WarmNext;
    rbp::mtl::Buffer<uint32_t> CacheCounts[2];
    rbp::mtl::Buffer<JiWrenchData> Wrenches;
    rbp::mtl::Buffer<JiBodyData> Bodies;
    rbp::mtl::Buffer<JiConstraintTile> Constraints;
    rbp::mtl::Buffer<JiBodyData> SnapshotBodies;
    rbp::mtl::Buffer<JiConstraintTile> SnapshotConstraints;
    rbp::mtl::Buffer<JiContactLink> ContactLinks;
    rbp::mtl::Buffer<JiJointLink> JointLinks;
    rbp::mtl::Buffer<JiWarmContact> Cache[2];
    rbp::mtl::Buffer<JiResidualData> BodyResiduals, ConstraintResiduals, PartialResiduals;
    rbp::mtl::Buffer<JiContributionData> Contributions;
    rbp::mtl::Buffer<float> BodyMatrix, BodyInverse;
    rbp::mtl::Buffer<JiSceneParams> SceneParams;
    rbp::mtl::Buffer<JiParams> SolveParams;
    rbp::mtl::Buffer<JiStatus> Status;
    const MTL::ResidencySet *PreviousWorld = nullptr;
    uint32_t Previous = 0;

    explicit Workspace(const rbp::mtl::Context &context)
        : Context(context), Admm(context) {
        constexpr rbp::shaders::Pass passes[]{
            rbp::shaders::JiBuildBodiesPass, rbp::shaders::JiBuildContactsPass,
            rbp::shaders::JiBuildJointsPass, rbp::shaders::JiSetConstraintCountPass,
            rbp::shaders::JiResetIncidencePass, rbp::shaders::JiCountIncidencePass,
            rbp::shaders::JiPrefixIncidencePass, rbp::shaders::JiFillIncidencePass,
            rbp::shaders::JiSortIncidencePass,
            rbp::shaders::JiClearWarmIndexPass, rbp::shaders::JiBuildWarmIndexPass,
            rbp::shaders::JiSeedContactsPass,
            rbp::shaders::JiValidateStatePass, rbp::shaders::JiFinalizeStepPass,
            rbp::shaders::JiCacheContactsPass,
            rbp::shaders::JiPublishCacheCountPass, rbp::shaders::JiIntegrateBodiesPass};
        static_assert(std::size(passes) == StageCount);
        for (uint32_t i = 0; i < StageCount; ++i)
            Pipelines[i] = context.Pipeline(rbp::shaders::PipelineIndices[0][passes[i]][0]);
        NS::Error *error{};
        Residency = NS::TransferPtr(context.Device->newResidencySet(
            rbp::mtl::Make<MTL::ResidencySetDescriptor>().get(), &error));
        if (!Residency) throw std::runtime_error("Cannot create Ji scene residency set");
        Allocator = NS::TransferPtr(context.Device->newCommandAllocator());
        Commands = NS::TransferPtr(context.Device->newCommandBuffer());
        BodyTable = NewTable(context, 7);
        ContactTable = NewTable(context, 11);
        JointTable = NewTable(context, 8);
        IncidenceTable = NewTable(context, 18);
        SolveTable = NewTable(context, 13);
        WarmTable = NewTable(context, 19);
        IntegrateTable = NewTable(context, 7);
        ValidateTable = NewTable(context, 8);
        SnapshotTable = NewTable(context, 5);
        bool changed = false;
        Ensure(SceneCounts, 10, changed);
        Ensure(SceneParams, 1, changed);
        Ensure(SolveParams, 1, changed);
        Ensure(Status, 1, changed);
        Ensure(WarmCount, 1, changed);
        Ensure(CacheCounts[0], 1, changed);
        Ensure(CacheCounts[1], 1, changed);
        CacheCounts[0][0] = CacheCounts[1][0] = 0;
        Residency->commit();
        Residency->requestResidency();
        Context.Queue->addResidencySet(Residency.get());
    }

    ~Workspace() {
        Context.Queue->removeResidencySet(Residency.get());
    }

    template<class T> void Ensure(rbp::mtl::Buffer<T> &buffer, uint32_t needed, bool &changed) {
        if (buffer.Capacity >= needed) return;
        if (buffer.Handle) Residency->removeAllocation(buffer.Handle.get());
        buffer = {Context.Device.get(), needed};
        Residency->addAllocation(buffer.Handle.get());
        changed = true;
    }

    void Reserve(uint32_t slots, uint32_t moving, uint32_t contacts, uint32_t joints) {
        const uint32_t constraints = contacts + 7 * joints;
        bool changed = false;
        Ensure(BodyMap, slots, changed);
        Ensure(BodySlots, moving, changed);
        Ensure(BodyComponents, moving, changed);
        Ensure(ComponentContacts, moving, changed);
        Ensure(Spawns, slots, changed);
        Ensure(Wrenches, slots, changed);
        Ensure(Bodies, moving, changed);
        Ensure(Constraints, (constraints + JI_CONSTRAINT_LANES - 1) / JI_CONSTRAINT_LANES, changed);
        Ensure(ContactLinks, contacts, changed);
        Ensure(JointLinks, std::max(1u, 7 * joints), changed);
        Ensure(IncidenceCounts, moving, changed);
        Ensure(Offsets, moving + 1, changed);
        Ensure(Cursor, moving, changed);
        Ensure(Incidence, 2 * constraints, changed);
        Ensure(ContributionSlots, 2 * constraints, changed);
        Ensure(Claims, contacts, changed);
        Ensure(BodyResiduals, moving, changed);
        Ensure(ConstraintResiduals, constraints, changed);
        Ensure(PartialResiduals, (std::max(moving, constraints) + 63) / 64, changed);
        Ensure(Contributions, 2 * constraints, changed);
        Ensure(BodyMatrix, 42 * moving, changed);
        Ensure(BodyInverse, 42 * moving, changed);
        for (uint32_t i=0;i<2;++i) {
            if (Cache[i].Capacity >= contacts) continue;
            const uint32_t count=CacheCounts[i][0];
            rbp::mtl::Buffer<JiWarmContact> replacement(Context.Device.get(),contacts);
            if (count>Cache[i].Capacity || count>replacement.Capacity)
                throw std::length_error("Ji warm cache count exceeds capacity");
            if (count) std::copy_n(Cache[i].Data(),count,replacement.Data());
            Residency->addAllocation(replacement.Handle.get());
            if (Cache[i].Handle) Residency->removeAllocation(Cache[i].Handle.get());
            Cache[i]=std::move(replacement); changed=true;
        }
        const uint32_t warm_entries = std::max(contacts, Cache[Previous].Capacity);
        if (warm_entries > (1u << 30))
            throw std::length_error("Ji warm index exceeds supported capacity");
        uint32_t buckets = 1;
        while (buckets < 2 * warm_entries) buckets <<= 1;
        Ensure(WarmHeads, buckets, changed);
        Ensure(WarmNext, warm_entries, changed);
        if (changed) {
            Residency->commit();
            Residency->requestResidency();
        }
    }

    void ReserveSnapshot(uint32_t moving, uint32_t constraints) {
        bool changed = false;
        Ensure(SnapshotBodies, moving, changed);
        Ensure(SnapshotConstraints, (constraints + JI_CONSTRAINT_LANES - 1) / JI_CONSTRAINT_LANES, changed);
        if (changed) {
            Residency->commit();
            Residency->requestResidency();
        }
    }

    void Dispatch(MTL4::ComputeCommandEncoder *encoder, Stage stage,
                  MTL4::ArgumentTable *table, uint32_t count) const {
        if (!count) return;
        auto *pipeline = Pipelines[stage].get();
        encoder->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch,
                                           MTL4::VisibilityOptionDevice);
        encoder->setArgumentTable(table);
        encoder->setComputePipelineState(pipeline);
        const uint32_t width = std::min(64u, uint32_t(pipeline->maxTotalThreadsPerThreadgroup()));
        encoder->dispatchThreadgroups({(count + width - 1) / width, 1, 1}, {width, 1, 1});
    }

    void EnsureProfileHeap(uint32_t count) {
        if (ProfileCapacity >= count) return;
        auto descriptor = rbp::mtl::Make<MTL4::CounterHeapDescriptor>();
        descriptor->setType(MTL4::CounterHeapTypeTimestamp);
        descriptor->setCount(count);
        NS::Error *error{};
        ProfileHeap = NS::TransferPtr(Context.Device->newCounterHeap(descriptor.get(), &error));
        if (!ProfileHeap) throw std::runtime_error("Cannot create Ji GPU timestamp heap");
        ProfileCapacity = count;
    }
};

Scene::Scene(const rbp::mtl::Context &context)
    : Context(context), Collision(context), Work(std::make_unique<Workspace>(context)) {}
Scene::~Scene() = default;

SceneResult Scene::Step(rbp::World &world, const SceneSettings &settings,
                        std::span<const Wrench> external) {
    return Run(world, settings, external, true);
}

SceneAdvance Scene::Advance(rbp::World &world, const SceneSettings &settings,
                            std::span<const Wrench> external) {
    SceneResult result = Run(world, settings, external, false);
    return {result.step, result.warm_started, result.geometry_features,
            result.static_only_features, result.solve.residual, result.solve.iterations,
            result.gpu_stages, result.cpu_stages};
}

SceneResult Scene::Run(rbp::World &world, const SceneSettings &settings,
                       std::span<const Wrench> external, bool diagnostics) {
    const auto cpu_start = std::chrono::steady_clock::now();
    const auto elapsed_ms = [](auto begin, auto end) {
        return std::chrono::duration<double, std::milli>(end - begin).count();
    };
    SceneResult::CpuStages cpu_stages{};
    if (!(settings.dt > 0) || !std::isfinite(settings.dt) || !settings.contact_capacity ||
        !(settings.contact_error_reduction >= 0) || !std::isfinite(settings.contact_error_reduction) ||
        !(settings.joint_error_reduction >= 0) || !std::isfinite(settings.joint_error_reduction) ||
        !settings.solve.iterations || !(settings.solve.alpha >= 1) ||
        !(settings.solve.tolerance >= 0) || !(settings.penalty_mass >= 0) ||
        !(settings.penalty_offset > 0))
        throw std::invalid_argument("Invalid Ji scene settings");
    CheckFloat(settings.dt);
    CheckFloat(settings.gravity);
    CheckFloat(settings.contact_error_reduction);
    CheckFloat(settings.joint_error_reduction);
    CheckFloat(settings.penalty_mass);
    CheckFloat(settings.penalty_offset);
    CheckFloat(settings.solve.alpha);
    CheckFloat(settings.solve.tolerance);
    CheckFloat(settings.collision_margin);
    if (std::isnan(settings.max_contact_reach) || !(float(settings.dt) > 0) ||
        !(float(settings.penalty_offset) > 0))
        throw std::invalid_argument("Invalid Ji scene FP32 settings");
    if (!external.empty() && external.size() != world.BodyCount())
        throw std::invalid_argument("Ji external wrenches must match world body slots");
    if (settings.capture_initial_system && !diagnostics)
        throw std::invalid_argument("Initial Ji system capture requires an audited Step");
    if (world.Queue.get() != Context.Queue.get())
        throw std::invalid_argument("Ji scene world and solver must share a Metal context");
    if (world.TrackContacts || world.TrackSensors)
        throw std::invalid_argument("Ji scene reports its own contacts; disable AVBD event tracking");
    auto &w = *Work;
    if (w.PreviousWorld != world.Residency.get()) {
        w.CacheCounts[w.Previous][0] = 0;
        w.PreviousWorld = world.Residency.get();
    }
    if (!world.BodyCount()) {
        w.CacheCounts[w.Previous][0] = 0;
        world.OnStepped(float(settings.dt));
        SceneResult result;
        result.step = ++Steps;
        return result;
    }
    const rbp::StepSettings collision_settings{
        .Gravity = ToFloat3(settings.gravity), .DeltaTime = float(settings.dt),
        .ContactMargin = float(settings.collision_margin),
        .MaxContactReach = float(settings.max_contact_reach)};

    uint32_t moving = 0;
    for (rbp::Index slot = 0; slot < world.BodyCount(); ++slot) {
        if (!world.Alive(slot)) continue;
        const auto &mass = world.Masses[slot];
        if (!(mass.InvMass >= 0) || !std::isfinite(mass.InvMass))
            throw std::invalid_argument("Ji scene requires nonnegative finite inverse mass");
        if (mass.InvMass == 0) continue;
        if (!(mass.InvInertiaLocal.x > 0 && mass.InvInertiaLocal.y > 0 && mass.InvInertiaLocal.z > 0) ||
            !std::isfinite(mass.InvInertiaLocal.x) ||
            !std::isfinite(mass.InvInertiaLocal.y) || !std::isfinite(mass.InvInertiaLocal.z))
            throw std::invalid_argument("Ji scene requires positive three-axis inertia on moving bodies");
        if (mass.LinearDamping != 0 || mass.AngularDamping != 0)
            throw std::invalid_argument("Ji scene has no authored AVBD damping conversion yet");
        ++moving;
    }
    if (!moving) {
        const auto geometry = Collision.CollectGeometry(world, collision_settings, settings.contact_capacity);
        w.CacheCounts[w.Previous][0] = 0;
        world.OnStepped(float(settings.dt));
        SceneResult result;
        result.step = ++Steps;
        result.geometry_features = result.static_only_features = uint32_t(geometry.size());
        return result;
    }
    if (settings.contact_capacity > std::numeric_limits<uint32_t>::max() / 2 ||
        world.JointCount() > (std::numeric_limits<uint32_t>::max() / 2 - settings.contact_capacity) / 7)
        throw std::length_error("Ji scene constraint capacity overflows");
    const uint32_t capacity = settings.contact_capacity + 7 * world.JointCount();
    w.Reserve(world.BodyCount(), moving, settings.contact_capacity, world.JointCount());
    const bool snapshot_requested=settings.capture_initial_system;
    if (snapshot_requested) w.ReserveSnapshot(moving, capacity);
    uint32_t next_body = 0;
    for (rbp::Index slot = 0; slot < world.BodyCount(); ++slot) {
        w.BodyMap[slot] = rbp::NoIndex;
        w.Spawns[slot] = world.IdOf(slot).Spawn;
        w.Wrenches[slot] = {};
        if (!external.empty()) {
            CheckFloat(external[slot].force);
            CheckFloat(external[slot].torque);
            for (uint32_t k = 0; k < 3; ++k) {
                w.Wrenches[slot].force[k] = float(external[slot].force[k]);
                w.Wrenches[slot].torque[k] = float(external[slot].torque[k]);
            }
        }
        if (!world.Alive(slot) || !(world.Masses[slot].InvMass > 0)) continue;
        w.BodyMap[slot] = next_body;
        w.BodySlots[next_body++] = slot;
    }
    // Joint topology is CPU-owned. The GPU marks which joint-connected groups
    // touch contacts, so unrelated resting bodies do not suppress a free
    // articulated group's initial A^-1 b velocity.
    std::vector<uint32_t> parent(moving);
    for (uint32_t body = 0; body < moving; ++body) parent[body] = body;
    const auto root = [&](uint32_t body) {
        while (parent[body] != body) body = parent[body];
        return body;
    };
    for (uint32_t j = 0; j < world.JointCount(); ++j) {
        const auto &joint = world.Joints[j];
        if (!joint.Active || joint.BodyA >= world.BodyCount() || joint.BodyB >= world.BodyCount())
            continue;
        bool constrained = false;
        for (uint32_t axis = 0; axis < 3; ++axis)
            constrained |= ((joint.LinearModes >> (3 * axis)) & 7u) != rbp::AxisFree ||
                           ((joint.AngularModes >> (3 * axis)) & 7u) != rbp::AxisFree;
        if (!constrained) continue;
        const uint32_t a = w.BodyMap[joint.BodyA], b = w.BodyMap[joint.BodyB];
        if (a != rbp::NoIndex && b != rbp::NoIndex)
            parent[root(a)] = root(b);
    }
    for (uint32_t body = 0; body < moving; ++body) {
        w.BodyComponents[body] = root(body);
        w.ComponentContacts[body] = 0;
    }
    w.SceneCounts[0] = w.SceneCounts[1] = w.SceneCounts[2] = w.SceneCounts[3] = 0;
    w.SceneCounts[4] = w.SceneCounts[5] = w.SceneCounts[8] = 0;
    w.SceneCounts[6] = w.SceneCounts[7] = w.SceneCounts[9] = std::numeric_limits<uint32_t>::max();
    w.WarmCount[0] = 0;
    w.SceneParams[0] = {
        .max_features = settings.contact_capacity, .body_count = world.BodyCount(),
        .moving_bodies = moving, .joint_count = world.JointCount(),
        .constraint_capacity = capacity, .broad_phase_root = rbp::BroadPhaseRoot(world.BodyCount()),
        .warm_bucket_mask = w.WarmHeads.Capacity - 1,
        .dt = float(settings.dt),
        .contact_error_reduction = float(settings.contact_error_reduction),
        .joint_error_reduction = float(settings.joint_error_reduction),
        .gravity = {float(settings.gravity[0]), float(settings.gravity[1]), float(settings.gravity[2])}};
    w.SolveParams[0] = {moving, 0, settings.solve.penalty_period, settings.solve.iterations,
                        float(settings.penalty_mass), float(settings.penalty_offset),
                        float(settings.solve.alpha), float(settings.solve.tolerance)};
    w.Status[0] = {1, 0, 0, 0, 0};
    const uint32_t next = 1 - w.Previous;
    w.CacheCounts[next][0] = 0;
    uint32_t profile_count = 0;
    if (settings.profile_gpu) {
        profile_count = 6 + 3 * std::min(5u, settings.solve.iterations);
        w.EnsureProfileHeap(profile_count);
        w.ProfileHeap->invalidateCounterRange({0, profile_count});
    }
    const double timestamp_frequency=settings.profile_gpu ? double(Context.Device->queryTimestampFrequency()) : 0;
    if (settings.profile_gpu && (!(timestamp_frequency>0) || !std::isfinite(timestamp_frequency)))
        throw std::runtime_error("Ji GPU timestamp frequency unavailable");
    const auto read_timestamps=[&](uint32_t first,uint32_t count) {
        const NS::Data *data=w.ProfileHeap->resolveCounterRange({first,count});
        if (!data || data->length()!=count*sizeof(MTL4::TimestampHeapEntry))
            throw std::runtime_error("Ji GPU timestamps unavailable");
        std::vector<uint64_t> times(count);
        for (uint32_t i=0;i<count;++i)
            std::memcpy(&times[i],static_cast<const char *>(data->bytes())+
                i*sizeof(MTL4::TimestampHeapEntry),sizeof(uint64_t));
        return times;
    };
    const auto gpu_interval=[&](std::span<const uint64_t> times,uint32_t i) {
        if (!times[i] || !times[i+1] || times[i+1]<times[i])
            throw std::runtime_error("Ji GPU timestamps invalid or out of order: slot="+
                std::to_string(i)+" start="+std::to_string(times[i])+" end="+
                std::to_string(times[i+1])+" frequency="+std::to_string(timestamp_frequency));
        return 1000*double(times[i+1]-times[i])/timestamp_frequency;
    };
    std::vector<uint64_t> initial_timestamps;
    std::span<const rbp::GeometryContact> geometry;

    const auto setup_end = std::chrono::steady_clock::now();
    cpu_stages.setup_ms = elapsed_ms(cpu_start, setup_end);
    w.Allocator->reset();
    w.Commands->beginCommandBuffer(w.Allocator.get());
    auto *encoder = w.Commands->computeCommandEncoder();
    rbp::Solver::GeometryOutput output{};
    std::chrono::steady_clock::time_point complete_time;
    try {
        const auto stamp = [&](uint32_t index) {
            if (settings.profile_gpu)
                encoder->writeTimestamp(MTL4::TimestampGranularityPrecise,
                                        w.ProfileHeap.get(), index);
        };
        stamp(0);
        output = Collision.EncodeGeometry(encoder, world, collision_settings, settings.contact_capacity);
        stamp(1);
        const auto bind = [](auto &table, std::initializer_list<uint64_t> addresses) {
            uint32_t slot = 0;
            for (uint64_t address : addresses) table->setAddress(address, slot++);
        };
        bind(w.BodyTable, {
            world.Poses.Address(), world.Velocities.Address(), world.Masses.Address(),
            w.BodySlots.Address(), w.Wrenches.Address(), w.SceneParams.Address(),
            w.Bodies.Address()});
        bind(w.ContactTable, {
            output.ContactsAddress, output.CountAddress, world.Poses.Address(),
            world.Velocities.Address(), w.BodyMap.Address(), w.Constraints.Address(),
            w.SceneCounts.Address(), w.SceneParams.Address(), w.ContactLinks.Address(),
            w.BodyComponents.Address(), w.ComponentContacts.Address()});
        bind(w.JointTable, {
            world.Joints.Address(), world.Poses.Address(), w.BodyMap.Address(),
            w.Constraints.Address(), w.SceneCounts.Address(), w.SceneParams.Address(),
            w.JointLinks.Address(), world.Velocities.Address()});
        bind(w.IncidenceTable, {
            w.Constraints.Address(), w.SolveParams.Address(), w.IncidenceCounts.Address(),
            w.Offsets.Address(), w.Cursor.Address(), w.Incidence.Address(),
            w.SceneCounts.Address(), w.Status.Address(), w.SceneParams.Address(),
            output.CountAddress, world.BroadPhaseNodes.Address(), w.Bodies.Address(),
            w.BodyComponents.Address(), w.ComponentContacts.Address(), w.ContributionSlots.Address(),
            output.ContactsAddress, w.ContactLinks.Address(), w.JointLinks.Address()});
        bind(w.SolveTable, {
            w.Bodies.Address(), w.Constraints.Address(), w.BodyResiduals.Address(),
            w.ConstraintResiduals.Address(), w.SolveParams.Address(), w.Status.Address(),
            w.Offsets.Address(), w.Incidence.Address(), w.Contributions.Address(),
            w.BodyMatrix.Address(), w.BodyInverse.Address(), w.PartialResiduals.Address(),
            w.ContributionSlots.Address()});
        bind(w.WarmTable, {
            output.ContactsAddress, w.ContactLinks.Address(), w.SceneCounts.Address(),
            world.Poses.Address(), world.BodyShapes.Address(), w.Spawns.Address(),
            w.Cache[w.Previous].Address(), w.CacheCounts[w.Previous].Address(), w.Claims.Address(),
            w.Constraints.Address(), w.Bodies.Address(), w.WarmCount.Address(),
            w.WarmHeads.Address(), w.Cache[next].Address(), w.Status.Address(),
            w.SolveParams.Address(), w.CacheCounts[next].Address(), w.WarmNext.Address(),
            w.SceneParams.Address()});
        bind(w.IntegrateTable, {
            world.Poses.Address(), world.Velocities.Address(), w.BodySlots.Address(),
            w.Bodies.Address(), w.SceneParams.Address(), w.Status.Address(),
            w.SolveParams.Address()});
        bind(w.ValidateTable, {
            w.Bodies.Address(), w.Constraints.Address(), w.SolveParams.Address(),
            w.Status.Address(), w.SceneCounts.Address(), w.BodyInverse.Address(),
            w.ContactLinks.Address(), output.CountAddress});
        if (snapshot_requested) {
            bind(w.SnapshotTable, {
                w.Bodies.Address(), w.Constraints.Address(), w.SnapshotBodies.Address(),
                w.SnapshotConstraints.Address(), w.SolveParams.Address()});
        }

        w.Dispatch(encoder, Workspace::BuildBodies, w.BodyTable.get(), moving);
        w.Dispatch(encoder, Workspace::BuildContacts, w.ContactTable.get(), settings.contact_capacity);
        w.Dispatch(encoder, Workspace::BuildJoints, w.JointTable.get(), world.JointCount());
        stamp(2);
        w.Dispatch(encoder, Workspace::ClearWarm, w.WarmTable.get(), w.WarmHeads.Capacity);
        w.Dispatch(encoder, Workspace::BuildWarmIndex, w.WarmTable.get(),
                   w.CacheCounts[w.Previous][0]);
        w.Dispatch(encoder, Workspace::SeedContacts, w.WarmTable.get(), settings.contact_capacity);
        w.Dispatch(encoder, Workspace::SetCount, w.IncidenceTable.get(), 1);
        w.Dispatch(encoder, Workspace::ResetIncidence, w.IncidenceTable.get(), moving);
        w.Dispatch(encoder, Workspace::CountIncidence, w.IncidenceTable.get(), capacity);
        w.Dispatch(encoder, Workspace::PrefixIncidence, w.IncidenceTable.get(), 1);
        w.Dispatch(encoder, Workspace::FillIncidence, w.IncidenceTable.get(), capacity);
        w.Dispatch(encoder, Workspace::SortIncidence, w.IncidenceTable.get(), moving);
        stamp(3);
        w.Admm.Encode(encoder, w.SolveTable.get(), settings.solve, moving, capacity, true,
                      settings.profile_gpu ? w.ProfileHeap.get() : nullptr,
                      settings.capture_initial_system ? w.SnapshotTable.get() : nullptr);
        stamp(4);
        w.Dispatch(encoder, Workspace::ValidateState, w.ValidateTable.get(), capacity);
        w.Dispatch(encoder, Workspace::FinalizeStep, w.ValidateTable.get(), 1);
        w.Dispatch(encoder, Workspace::CacheContacts, w.WarmTable.get(), settings.contact_capacity);
        w.Dispatch(encoder, Workspace::PublishCache, w.WarmTable.get(), 1);
        w.Dispatch(encoder, Workspace::Integrate, w.IntegrateTable.get(), moving);
        stamp(5);
        encoder->endEncoding(); w.Commands->endCommandBuffer();
        std::binary_semaphore done{0};
        std::string gpu_error;
        auto options=rbp::mtl::Make<MTL4::CommitOptions>();
        options->addFeedbackHandler([&](MTL4::CommitFeedback *feedback) {
            if (NS::Error *error=feedback->error())
                gpu_error=error->localizedDescription()->utf8String();
            done.release();
        });
        const MTL4::CommandBuffer *list[]{w.Commands.get()};
        const auto before_commit=std::chrono::steady_clock::now();
        cpu_stages.encode_ms+=elapsed_ms(setup_end,before_commit);
        Context.Queue->commit(list,1,options.get());
        const auto after_commit=std::chrono::steady_clock::now();
        cpu_stages.commit_ms+=elapsed_ms(before_commit,after_commit);
        done.acquire();
        complete_time=std::chrono::steady_clock::now();
        cpu_stages.wait_ms+=elapsed_ms(after_commit,complete_time);
        if (!gpu_error.empty()) throw std::runtime_error("Ji scene Metal command failed: "+gpu_error);
        if (settings.profile_gpu) initial_timestamps=read_timestamps(0,profile_count);
    } catch (...) {
        Collision.CancelGeometry();
        throw;
    }
    geometry = Collision.FinishGeometry(world);
    const auto geometry_end = std::chrono::steady_clock::now();
    cpu_stages.geometry_ms = elapsed_ms(complete_time, geometry_end);
    if (w.SceneCounts[2] || w.SceneCounts[0] + w.SceneCounts[1] != geometry.size() ||
        w.SolveParams[0].constraints != w.SceneCounts[0] + w.SceneCounts[3] ||
        !w.Status[0].iterations)
        throw std::runtime_error("Ji scene rejected GPU step: geometry=" +
            std::to_string(geometry.size()) + " contacts=" + std::to_string(w.SceneCounts[0]) +
            " static=" + std::to_string(w.SceneCounts[1]) +
            " invalid=" + std::to_string(w.SceneCounts[2]) +
            " invalid_bodies=" + std::to_string(w.SceneCounts[4]) +
            " first_body=" + std::to_string(w.SceneCounts[6]) +
            " invalid_constraints=" + std::to_string(w.SceneCounts[5]) +
            " first_constraint=" + std::to_string(w.SceneCounts[7]) +
            " invalid_factors=" + std::to_string(w.SceneCounts[8]) +
            " first_factor=" + std::to_string(w.SceneCounts[9]) +
            " joints=" + std::to_string(w.SceneCounts[3]) +
            " constraints=" + std::to_string(w.SolveParams[0].constraints) +
            " iterations=" + std::to_string(w.Status[0].iterations));
    if (settings.solve.tolerance > 0 && w.Status[0].active)
        throw std::runtime_error("Ji scene solve reached its iteration limit before convergence");
    GpuResult solved;
    std::optional<System> initial_system;
    if (diagnostics) {
        solved = DecodeGpu({}, w.Bodies.All().first(moving),
            w.Constraints.All(), w.SolveParams[0].constraints, w.Status[0]);
        if (settings.capture_initial_system) {
            const uint32_t constraints = w.SolveParams[0].constraints;
            initial_system = DecodeGpu({}, w.SnapshotBodies.All().first(moving),
                w.SnapshotConstraints.All(), constraints, JiStatus{1, 0, 0, 0, 0}).system;
        }
    } else {
        solved.residual = {w.Status[0].primal, w.Status[0].dual, w.Status[0].dynamics,
                           w.Status[0].contact_law};
        solved.iterations = w.Status[0].iterations;
        solved.converged = !w.Status[0].active;
    }
    const auto decode_end = std::chrono::steady_clock::now();
    cpu_stages.decode_ms = elapsed_ms(geometry_end, decode_end);
    const MechanicalAudit audit = diagnostics ? AuditFinal(solved.system) : MechanicalAudit{};
    const auto audit_end = std::chrono::steady_clock::now();
    cpu_stages.audit_ms = elapsed_ms(decode_end, audit_end);
    std::optional<SceneResult::GpuStages> gpu_stages;
    if (settings.profile_gpu) {
        const auto &times=initial_timestamps;
        const auto elapsed=[&](uint32_t i) { return gpu_interval(times,i); };
        double body_ms = 0, constraint_ms = 0;
        const uint32_t samples = std::min(5u, settings.solve.iterations);
        for (uint32_t i = 0; i < samples; ++i) {
            body_ms += elapsed(6 + 3 * i);
            constraint_ms += elapsed(7 + 3 * i);
        }
        if (samples) {
            body_ms /= samples;
            constraint_ms /= samples;
        }
        const double solve_ms=elapsed(3);
        const double finalize_ms=elapsed(4);
        gpu_stages = SceneResult::GpuStages{elapsed(0), elapsed(1), elapsed(2),
                                            solve_ms, finalize_ms, body_ms, constraint_ms};
    }
    std::vector<SceneContact> reports;
    if (diagnostics) {
        reports.reserve(w.SceneCounts[0]);
        for (uint32_t i = 0; i < w.SceneCounts[0]; ++i) {
            const JiContactLink link = w.ContactLinks[i];
            if (link.geometry_index >= geometry.size())
                throw std::runtime_error("Ji scene contact link escaped geometry output");
            const rbp::GeometryContact &raw = geometry[link.geometry_index];
            const auto axes = ContactAxes(Scale(ToVec3(raw.Normal), link.flipped ? -1 : 1));
            Vec3 impulse{};
            for (uint32_t row = 0; row < 3; ++row)
                impulse = Add(impulse, Scale(axes[row], settings.dt * solved.system.constraints[i].impulse[row]));
            if (link.flipped) impulse = Scale(impulse, -1);
            reports.push_back({Key(world, raw), ToVec3(raw.PointA), ToVec3(raw.PointB),
                               ToVec3(raw.Normal), impulse,
                               Scale(impulse, 1 / settings.dt),
                               raw.Gap, raw.Friction});
        }
    }
    world.OnStepped(float(settings.dt));
    w.Previous = next;
    cpu_stages.report_ms = elapsed_ms(audit_end, std::chrono::steady_clock::now());
    return {++Steps, w.WarmCount[0], uint32_t(geometry.size()), w.SceneCounts[1],
            std::move(reports), std::move(solved), std::move(initial_system), audit, gpu_stages,
            cpu_stages};
}

} // namespace ji
