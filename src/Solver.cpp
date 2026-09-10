#include "Solver.h"

#include "GpuSource.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <exception>
#include <numeric>
#include <string>
#include <string_view>

namespace rbp {

namespace {
// Collision and primal passes use this binding at separate times.
constexpr uint32_t BroadPhaseOrCursorAt = 14;
constexpr uint32_t BindingCount = 31;
constexpr uint32_t BatchSteps = 16;
constexpr uint64_t BatchBytes = 32 * 1024 * 1024;

// Sweeps of the restitution pass over its contacts.
// The pass is Jacobi: every contact computes an impulse from one velocity snapshot, and a gather applies them.
// Each point of a manifold therefore takes the whole approach speed on the first sweep, and later sweeps divide it between them.
// Four is convergence, with eight and sixteen bit-identical.
constexpr uint32_t RestitutionPasses = 4;

// The number of colors a sweep dispatches.
// Coloring is incremental, so last step's count carries over, plus one spare for the contacts a closing scene adds.
// Counted from the world rather than cached in the solver, so a step stays a pure function of the world it is given.
uint32_t ColorsNeeded(const World &world, const StepSettings &settings) {
    uint32_t used = 1;
    for (uint32_t body = 0; body < world.BodyCount(); ++body)
        if (Moves(world.Masses[body])) used = std::max(used, ColorOf(world.Colors[body]) + 1);
    return std::clamp(used + 1, 1u, std::max(1u, std::min(settings.MaxColors, MaxSupportedColors)));
}

constexpr uint32_t BoundedPlanes = 1, MeshPairs = 2, MeshQueries = 4, RestitutionMaterials = 8;

uint32_t ColliderFeatures(const World &world) {
    uint32_t features = 0;
    bool mesh_seen = false;
    for (Index body = 0; body < world.BodyCount(); ++body) {
        const Index root = world.BodyShapes[body];
        if (root == NoIndex) continue;
        const Shape &shape = world.Shapes[root];
        if (world.Materials[body].Restitution != 0 || (shape.HasMaterial && shape.Surface.Restitution != 0)) features |= RestitutionMaterials;
        bool mesh = shape.Kind == ShapeMesh;
        features |= IsBoundedPlane(shape) ? BoundedPlanes : 0u;
        if (shape.Kind == ShapeCompound)
            for (uint32_t leaf = 0; leaf < shape.VertexCount; ++leaf) {
                const Shape &child = world.Shapes[world.Child(root, leaf)];
                if (child.HasMaterial && child.Surface.Restitution != 0) features |= RestitutionMaterials;
                features |= IsBoundedPlane(child) ? BoundedPlanes : 0u;
                mesh |= child.Kind == ShapeMesh;
            }
        if (mesh && mesh_seen) features |= MeshPairs;
        if (mesh) features |= MeshQueries;
        mesh_seen |= mesh;
        if (features == (BoundedPlanes | MeshPairs | MeshQueries | RestitutionMaterials)) break;
    }
    return features;
}

// The kernel each pass runs, in the order of Solver::Pass.
// A prefix supplies a #define when one kernel text is compiled more than one way.
constexpr struct {
    const char *Name;
    std::string_view Prefix;
} Kernels[]{
    {"ReduceBodyBounds"},
    {"ReduceSceneBounds"},
    {"MakeMortonKeys"},
    {"RadixHistogram"},
    {"RadixOffsets"},
    {"RadixScatter"},
    {"BuildRadixTree"},
    {"RefitRadixTree"},
    {"RefreshRadixLeaves"},
    {"BuildSmallBroadPhase"},
    {"BuildBodyBounds"},
    {"BuildBodyBounds", "#define SENSOR_PASS 1"},
    {"Integrate"},
    {"CollectContacts", "#define BOUNDED_PLANES 0\n#define MESH_PAIRS 0"},
    {"ScanIncoming"},
    {"ScanIncomingBlocks"},
    {"OffsetIncoming"},
    {"FillIncoming"},
    {"SortIncoming"},
    {"PrepareJoints"},
    {"WarmStart"},
    {"UpdateColors"},
    {"PublishColors"},
    {"SolveBodies"},
    {"PublishPoses"},
    {"UpdateDuals"},
    {"UpdateJointDuals"},
    {"Finalize"},
    {"Restitution"},
    {"ApplyRestitution"},
    {"SolveBodies", "#define STABILIZE 1"},
    {"CountQuiet"},
    {"SpreadWaking"},
    {"PublishWaking"},
    {"FollowSensors"},
    {"PrepareStepColors"},
    {"CaptureStep"},
    {"CollectSensorContacts", "#define BOUNDED_PLANES 0\n#define MESH_PAIRS 0\n#define SENSOR_PASS 1\n#define CollectContacts CollectSensorContacts"},
    {"CollectContacts", "#define MESH_PAIRS 0"},
    {"CollectSensorContacts", "#define MESH_PAIRS 0\n#define SENSOR_PASS 1\n#define CollectContacts CollectSensorContacts"},
    {"CollectContacts", "#define BOUNDED_PLANES 0"},
    {"CollectSensorContacts", "#define BOUNDED_PLANES 0\n#define SENSOR_PASS 1\n#define CollectContacts CollectSensorContacts"},
    {"CollectContacts"},
    {"CollectSensorContacts", "#define SENSOR_PASS 1\n#define CollectContacts CollectSensorContacts"},
    {"ResetQueries"},
    {"EvaluateQueries", "#define COLLECT_LANES 64"},
    {"EvaluateQueries", "#define COLLECT_LANES 64\n#define SENSOR_PASS 1"},
    {"CountColorWork"},
    {"PrefixColorWork"},
    {"FillColorWork"},
    {"SolveSmallWorld", "#define FUSED_SMALL_SOLVE 1"},
    {"ResetQueryReuse"},
    {"CheckQueryInputs"},
    {"CheckQueryGeometry"},
    {"BuildBodyBounds", "#define CACHE_BOUNDS_HINT 1"},
    {"InitializeIslands"},
    {"UnionIslands"},
    {"PackIslands"},
    {"FinishIslands"},
    {"SolveIslands", "#define FUSED_SMALL_SOLVE 1\n#define FUSED_GENERAL_SOLVE 1\n#define SolveSmallWorld SolveIslands"},
    {"PrepareSmallWorld"},
};
} // namespace

Solver::Solver(const mtl::Context &context) : Context(context) {
    static_assert(std::size(Kernels) == PassCount, "one kernel per pass, in the enum's order");
    // Additional collider specializations compile lazily when a world first uses them.
    for (uint32_t pass = 0; pass < BoundedCollectPass; ++pass) {
        std::string prefix{Kernels[pass].Prefix};
        if (pass == CollectPass || pass == SensorPass) prefix += "\n#define BROAD_PHASE_MODE 1";
        Pipelines[Direct][pass][0] = context.Pipeline(pass < BoundsPass ? gpu::BroadPhaseSource : gpu::SolveSource, Kernels[pass].Name, prefix, pass == FollowPass);
    }

    NS::Error *error{};
    auto descriptor = mtl::Make<MTL4::ArgumentTableDescriptor>();
    descriptor->setMaxBufferBindCount(BindingCount);
    Table = NS::TransferPtr(context.Device->newArgumentTable(descriptor.get(), &error));
    Allocator = NS::TransferPtr(context.Device->newCommandAllocator());
    Commands = NS::TransferPtr(context.Device->newCommandBuffer());
    Done = NS::TransferPtr(context.Device->newSharedEvent());

    Params = {context.Device.get(), BatchSteps};
    ColorGroups = {context.Device.get(), BatchSteps * MaxSupportedColors * 3};
    SmallIslands = {context.Device.get(), SolveLanes};
    ColorScratch = {context.Device.get(), 1};
    OutputFlags = {context.Device.get(), 1};
    // One slot per color, holding its own index, so a color pass is selected by the slot the cursor binding points at.
    // No counting kernel is dispatched between colors.
    ColorCursor = {context.Device.get(), MaxSupportedColors + 2};
    for (uint32_t color = 0; color < MaxSupportedColors; ++color) ColorCursor[color] = color;
    Residency = NS::TransferPtr(context.Device->newResidencySet(mtl::Make<MTL::ResidencySetDescriptor>().get(), &error));
    Residency->addAllocation(Params.Handle.get());
    Residency->addAllocation(ColorGroups.Handle.get());
    Residency->addAllocation(SmallIslands.Handle.get());
    Residency->addAllocation(ColorScratch.Handle.get());
    Residency->addAllocation(OutputFlags.Handle.get());
    Residency->addAllocation(ColorCursor.Handle.get());
    Residency->commit();
    Residency->requestResidency();
    context.Queue->addResidencySet(Residency.get());
}

Solver::~Solver() {
    Context.Queue->removeResidencySet(Residency.get());
    mtl::Drain(Context.Queue.get()); // see mtl::Drain
}

void Solver::Dispatch(MTL4::ComputeCommandEncoder *encoder, Pass pass, uint32_t threads, uint32_t lanes, uint64_t indirect, CollectionMode mode) {
    const bool collector = pass == CollectPass || pass == SensorPass || (pass >= BoundedCollectPass && pass <= FullSensorPass);
    const bool primal = pass == SolvePass || pass == StabilizePass;
    const bool bounds = pass == BoundsPass || pass == SensorBoundsPass || pass == CacheBoundsPass;
    const uint32_t variant = collector ? uint32_t(threads > RadixSimdWidth) + 2 * uint32_t(lanes > 1) :
        bounds                         ? uint32_t(lanes > 1) :
                                         uint32_t(primal && threads > SolveLanes);
    auto &selected = Pipelines[mode][pass][variant];
    if (!selected) {
        std::string prefix{Kernels[pass].Prefix};
        if (mode == Prepare) prefix += "\n#define PREPARE_QUERIES 1";
        if (mode == Queued) prefix += "\n#define QUEUED_QUERIES 1";
        if (mode == Recompute) prefix += "\n#define RECOMPUTE_QUERIES 1";
        if (collector) {
            prefix += (variant & 1) ? "\n#define BROAD_PHASE_MODE 2" : "\n#define BROAD_PHASE_MODE 1";
            prefix += "\n#define COLLECT_LANES " + std::to_string(lanes);
        }
        if (primal) prefix += "\n#define SOLVE_BODIES_PER_GROUP " + std::to_string(variant ? SolveBodiesPerGroup : 1);
        if (bounds) prefix += "\n#define BOUNDS_LANES " + std::to_string(lanes);
        selected = Context.Pipeline(pass < BoundsPass ? gpu::BroadPhaseSource : gpu::SolveSource, Kernels[pass].Name, prefix, pass == FollowPass);
    }
    MTL::ComputePipelineState *pipeline = selected.get();
    // Every pass reads the previous pass's writes, so every dispatch takes a barrier.
    encoder->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
    encoder->setComputePipelineState(pipeline);
    const bool per_body = collector || pass == SolvePass || pass == StabilizePass;
    const bool radix = pass == RadixHistogramPass || pass == RadixOffsetsPass || pass == ReduceBoundsPass || pass == ReduceScenePass;
    const bool scan = pass == ScanIncomingPass || pass == ScanIncomingBlocksPass;
    const bool contact = pass == DualPass || pass == RestitutionPass;
    if ((radix || scan || pass == PrefixColorWorkPass || pass == PrepareSmallWorldPass || pass == SmallBroadPhasePass || ((collector || bounds) && lanes > 1)) && pipeline->threadExecutionWidth() != RadixSimdWidth)
        throw std::runtime_error("GPU scans require 32-lane SIMD groups");
    const uint32_t limit = scan ? RadixBlockSize : per_body ? 8u :
        contact                                             ? 64u :
                                                              pipeline->maxTotalThreadsPerThreadgroup();
    const auto group = pass == SmallBroadPhasePass ? threads : radix ? RadixBlockSize :
                                                                       std::min(threads, limit);
    if (pass == SolveIslandsPass) {
        if (pipeline->threadExecutionWidth() != SolveLanes || SmallSolveWaves * SolveLanes > pipeline->maxTotalThreadsPerThreadgroup())
            throw std::runtime_error("The island solve requires eight 32-lane SIMD groups");
        encoder->dispatchThreadgroups(indirect, {SmallSolveWaves * SolveLanes, 1, 1});
    } else if (indirect && (pass == PublishPass || pass == DualPass || pass == JointDualPass)) {
        encoder->dispatchThreadgroups(indirect, {pass == PublishPass ? 128u : pass == DualPass ? 64u :
                                                                                                 32u,
                                                 1, 1});
    } else if (pass == SolveSmallWorldPass) {
        const uint32_t width = std::min(threads, SmallSolveWaves) * SolveLanes;
        if (pipeline->threadExecutionWidth() != SolveLanes || width > pipeline->maxTotalThreadsPerThreadgroup())
            throw std::runtime_error("The small-world solve requires up to eight 32-lane SIMD groups");
        encoder->dispatchThreadgroups({threads, 1, 1}, {width, 1, 1});
    } else if (mode == Queued) encoder->dispatchThreadgroups({threads, 1, 1}, {1, 1, 1});
    else if (pass == CheckQueryGeometryPass || pass == CheckQueryInputsPass) encoder->dispatchThreadgroups(indirect, {128, 1, 1});
    else if (pass == QueryPass || pass == SensorQueryPass) encoder->dispatchThreadgroups(indirect ? indirect : QueryScratch.Address(), {32, 1, 1});
    else if (primal) {
        if (pipeline->threadExecutionWidth() != SolveLanes) throw std::runtime_error("The body solve requires 32-lane SIMD groups");
        const uint32_t bodies_per_group = variant ? SolveBodiesPerGroup : 1;
        if (indirect) encoder->dispatchThreadgroups(indirect, {SolveLanes, 1, 1});
        else encoder->dispatchThreadgroups({(threads + bodies_per_group - 1) / bodies_per_group, 1, 1}, {SolveLanes, 1, 1});
    } else if (pass == JointDualPass) encoder->dispatchThreadgroups({threads, 1, 1}, {32, 1, 1});
    else if ((collector || bounds) && lanes > 1) encoder->dispatchThreadgroups({threads * (mode == Prepare ? QueryPartitions(threads) : 1u), 1, 1}, {lanes, 1, 1});
    else encoder->dispatchThreads({threads, 1, 1}, {group, 1, 1});
}

void Solver::PrepareFollowers(World &world, std::span<const SensorFollower> followers) {
    FollowerRanges.clear();
    if (followers.empty()) return;
    if (followers.size() > world.BodyCount()) throw std::invalid_argument("more followers than bodies");
    const uint32_t count = uint32_t(followers.size());
    std::vector<uint32_t> order(count), dependency(count, NoIndex), depth(count), state(count);
    std::iota(order.begin(), order.end(), 0u);
    for (const auto &f : followers) {
        if (!world.Alive(f.Sensor) || !world.Alive(f.Owner) || !world.Filters[f.Sensor].Sensor || Moves(world.Masses[f.Sensor]))
            throw std::invalid_argument("a follower requires a live kinematic sensor and a live owner");
        bool finite = true;
        for (uint32_t i = 0; i < 3; ++i) finite &= std::isfinite(f.Local.Position[i]);
        const float norm = simd::dot(f.Local.Orientation, f.Local.Orientation);
        if (!finite || !std::isfinite(norm) || norm <= 0) throw std::invalid_argument("invalid sensor follower pose");
    }
    std::ranges::sort(order, {}, [&](uint32_t i) { return followers[i].Sensor; });
    for (uint32_t i = 0; i < count; ++i) {
        if (i && followers[order[i - 1]].Sensor == followers[order[i]].Sensor)
            throw std::invalid_argument("duplicate sensor follower");
        const auto owner = std::ranges::lower_bound(order, followers[i].Owner, {}, [&](uint32_t j) { return followers[j].Sensor; });
        if (owner != order.end() && followers[*owner].Sensor == followers[i].Owner) dependency[i] = *owner;
    }
    std::vector<uint32_t> path;
    for (uint32_t root = 0; root < count; ++root) {
        path.clear();
        uint32_t at = root;
        while (at != NoIndex && state[at] == 0) {
            state[at] = 1;
            path.push_back(at);
            at = dependency[at];
        }
        if (at != NoIndex && state[at] == 1) throw std::invalid_argument("cyclic sensor followers");
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            depth[*it] = dependency[*it] == NoIndex ? 0 : depth[dependency[*it]] + 1;
            state[*it] = 2;
        }
    }
    std::iota(order.begin(), order.end(), 0u);
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return depth[a] < depth[b]; });
    if (Followers.Capacity < count) {
        if (Followers.Handle) Residency->removeAllocation(Followers.Handle.get());
        Followers = {Context.Device.get(), count};
        Residency->addAllocation(Followers.Handle.get());
        Residency->commit();
    }
    for (uint32_t i = 0; i < count; ++i) {
        Followers[i] = followers[order[i]];
        if (i == 0 || depth[order[i]] != depth[order[i - 1]]) FollowerRanges.push_back({i, 0});
        ++FollowerRanges.back().Count;
    }
}

void Solver::Step(World &world, const StepSettings &settings) { Advance(world, settings, 1); }

void Solver::Bind(World &world, uint32_t parameter) {
    const uint64_t bindings[]{
        world.Poses.Address(),
        world.InitialPoses.Address(),
        world.InertialPoses.Address(),
        world.Velocities.Address(),
        world.Masses.Address(),
        world.Contacts.Address(),
        world.BodyShapes.Address(),
        Params.Address() + parameter * sizeof(StepParams),
        world.Shapes.Address(),
        world.PreviousVelocities.Address(),
        world.Materials.Address(),
        world.Iterates.Address(),
        world.Colors.Address(),
        world.NextColors.Address(),
        world.Bounds.Address(),
        world.CompoundChildren.Address(),
        world.Joints.Address(),
        world.Incoming.Address(),
        world.IncomingSlots.Address(),
        world.Filters.Address(),
        world.Jointed.Address(),
        world.Quiet.Address(),
        world.RestPoses.Address(),
        world.NextQuiet.Address(),
        world.ContactEvents.Address(),
        world.ContactEventCounts.Address(),
        world.ContactRefusals.Address(),
        world.ShapeVertices.Address(),
        world.Triangles.Address(),
        world.BvhNodes.Address(),
        world.HullFaces.Address(),
    };
    static_assert(sizeof(bindings) / sizeof(bindings[0]) == BindingCount, "one address per slot the table holds");
    for (uint32_t slot = 0; slot < BindingCount; ++slot) Table->setAddress(bindings[slot], slot);
}

AdvanceResult Solver::Advance(World &world, const StepSettings &settings, uint32_t substeps, std::span<const SensorFollower> followers, const std::function<void(const StepResult &)> &observer) {
    if (Advancing) throw std::logic_error("Advance cannot be reentered from its observer");
    if (substeps == 0 || world.BodyCount() == 0) return {};
    struct Scope {
        bool &Active;
        ~Scope() { Active = false; }
    } scope{Advancing};
    Advancing = true;
    ColorCursor[MaxSupportedColors] = settings.Iterations;
    ColorCursor[MaxSupportedColors + 1] = settings.ColoringPasses;
    const uint64_t island_words = IslandHeaderWords + uint64_t(world.BodyCount()) * IslandWordsPerBody;
    if (world.BodyCount() > SolveLanes && island_words > GeneralIslands.Capacity) {
        if (island_words > UINT32_MAX) throw std::length_error("island storage is too large");
        if (GeneralIslands.Handle) Residency->removeAllocation(GeneralIslands.Handle.get());
        GeneralIslands = {Context.Device.get(), uint32_t(island_words)};
        Residency->addAllocation(GeneralIslands.Handle.get());
        Residency->commit();
    }
    world.RefreshFilters();
    PrepareFollowers(world, followers);
    const uint32_t joints = world.JointCount();
    const uint32_t collider_features = ColliderFeatures(world);
    if (collider_features & MeshQueries) {
        bool residency_changed = false;
        if (!QueryScratch.Handle) {
            residency_changed = true;
            QueryScratch = {Context.Device.get(), QueryScratchBytes / sizeof(uint32_t)};
            QueryInputs = {Context.Device.get(), 1};
            QueryInputs[0] = {};
            auto &solid = *reinterpret_cast<QueryArenaHeader *>(QueryScratch.Data());
            solid = {};
            solid.Bytes = QueryScratchBytes;
            Residency->addAllocation(QueryScratch.Handle.get());
            Residency->addAllocation(QueryInputs.Handle.get());
        }
        QueryInputSpec spec{};
        const uint32_t bodies = world.BodyCount();
        const uint64_t bytes[]{uint64_t(bodies) * sizeof(Pose), uint64_t(bodies) * sizeof(Velocity), uint64_t(bodies) * sizeof(BodyMass), uint64_t(bodies) * sizeof(Index), uint64_t(world.ShapeCount()) * sizeof(Shape), uint64_t(bodies) * sizeof(Material), uint64_t(bodies) * sizeof(Filter), uint64_t(world.Jointed.Capacity) * sizeof(Index), uint64_t(world.ShapeVertices.Capacity) * sizeof(float3), uint64_t(world.HullFaces.Capacity) * sizeof(HullFace), uint64_t(world.Triangles.Capacity) * sizeof(Triangle), uint64_t(world.BvhNodes.Capacity) * sizeof(BvhNode), uint64_t(world.CompoundChildren.Capacity) * sizeof(Index), sizeof(StepParams), uint64_t(bodies) * sizeof(uint32_t), uint64_t(bodies) * ContactsPerBody * 8 * sizeof(uint32_t)};
        uint64_t words = 0;
        for (uint32_t at = 0; at < 16; ++at) {
            if (bytes[at] / 4 > UINT32_MAX - words) throw std::length_error("query input snapshot is too large");
            spec.Offsets[at] = uint32_t(words);
            words += bytes[at] / 4;
        }
        spec.Words = uint32_t(words);
        if (spec.Words > QueryInputSnapshot.Capacity) {
            residency_changed = true;
            if (QueryInputSnapshot.Handle) Residency->removeAllocation(QueryInputSnapshot.Handle.get());
            QueryInputSnapshot = {Context.Device.get(), spec.Words};
            std::fill_n(QueryInputSnapshot.Data(), spec.Words, 0u);
            Residency->addAllocation(QueryInputSnapshot.Handle.get());
            reinterpret_cast<QueryArenaHeader *>(QueryScratch.Data())->Valid = 0;
        }
        if (std::memcmp(&QueryInputs[0], &spec, sizeof(QueryInputSpec)) != 0)
            reinterpret_cast<QueryArenaHeader *>(QueryScratch.Data())->Valid = 0;
        QueryInputs[0] = spec;
        // Geometry is immutable until this Advance returns, including across submissions.
        reinterpret_cast<QueryArenaHeader *>(QueryScratch.Data())->GeometryChecked = 0;
        if (residency_changed) Residency->commit();
    }
    AdvanceResult result;
    while (result.Steps < substeps && world.BodyCount() != 0) {
        const uint32_t first_bodies = world.BodyCount();
        uint32_t later_bodies = first_bodies;
        while (later_bodies && !world.LiveBodies[later_bodies - 1]) --later_bodies;
        bool sensors = !world.Overlaps().empty();
        for (uint32_t body = 0; body < first_bodies && !sensors; ++body) sensors = world.Alive(body) && world.Filters[body].Sensor;
        if (sensors) world.EnsureSensorBuffers();
        if (sensors && (collider_features & MeshQueries) && !SensorQueries.Handle) {
            SensorQueries = {Context.Device.get(), QueryScratchBytes / sizeof(uint32_t)};
            auto &header = *reinterpret_cast<QueryArenaHeader *>(SensorQueries.Data());
            header = {};
            header.Bytes = QueryScratchBytes;
            Residency->addAllocation(SensorQueries.Handle.get());
            Residency->commit();
        }
        const bool poses = bool(observer) || world.TrackContacts;
        OutputFlags[0] = {uint32_t(poses), uint32_t(sensors)};
        Layout = {};
        const auto take = [&](uint64_t bytes) {
            Layout.Stride = (Layout.Stride + 15) & ~uint64_t(15);
            const uint64_t offset = Layout.Stride;
            Layout.Stride += bytes;
            return offset;
        };
        Layout.Completion = take(sizeof(StepCompletion));
        Layout.Counts = take(uint64_t(first_bodies) * sizeof(StepCounts));
        if (poses) {
            Layout.Poses = take(uint64_t(first_bodies) * sizeof(Pose));
            Layout.Velocities = take(uint64_t(first_bodies) * sizeof(Velocity));
        }
        if (world.TrackContacts) {
            Layout.Initial = take(uint64_t(first_bodies) * sizeof(Pose));
            Layout.Contacts = take(uint64_t(first_bodies) * ContactsPerBody * sizeof(ContactReport));
            Layout.RemovedContacts = take(uint64_t(first_bodies) * ContactsPerBody * sizeof(ContactEvent));
        }
        if (sensors) Layout.Sensors = take(uint64_t(first_bodies) * ContactsPerBody * sizeof(SensorPair));
        Layout.Stride = (Layout.Stride + 15) & ~uint64_t(15);
        const uint32_t count = later_bodies == 0 ? 1 : std::min({substeps - result.Steps, BatchSteps, uint32_t(std::max(uint64_t(1), BatchBytes / Layout.Stride))});
        const bool snapshot = count > 1;
        if (snapshot && Outputs.Capacity < count * Layout.Stride) {
            if (Outputs.Handle) Residency->removeAllocation(Outputs.Handle.get());
            Outputs = {Context.Device.get(), uint32_t(count * Layout.Stride)};
            Residency->addAllocation(Outputs.Handle.get());
            Residency->commit();
        }
        Allocator->reset();
        Commands->beginCommandBuffer(Allocator.get());
        auto *encoder = Commands->computeCommandEncoder();
        encoder->setArgumentTable(Table.get());
        for (uint32_t index = 0; index < count; ++index) {
            const uint32_t bodies = index == 0 ? first_bodies : later_bodies;
            const uint32_t colors = index == 0 ? ColorsNeeded(world, settings) : std::max(1u, std::min(settings.MaxColors, MaxSupportedColors));
            Params[index] = {
                .Gravity = settings.Gravity,
                .DeltaTime = settings.DeltaTime,
                .Beta = settings.Beta,
                .ContactBeta = settings.ContactBeta,
                .Gamma = settings.Gamma,
                .PenaltyMin = settings.PenaltyMin,
                .PenaltyMax = settings.PenaltyMax,
                .ContactMargin = settings.ContactMargin,
                .MaxContactReach = settings.MaxContactReach,
                .MaxAngularSpeed = settings.MaxAngularSpeed,
                .MinBounceSpeed = settings.BounceSpeedFactor * simd::length(settings.Gravity) * settings.DeltaTime,
                .SleepSpeed = settings.SleepSpeed,
                .SleepSteps = settings.SleepSteps,
                .SleepDrift = settings.SleepDrift,
                .BodyCount = bodies,
                .JointCount = joints,
                .MaxColors = colors,
                .ReportContacts = world.TrackContacts,
            };

            Bind(world, index);
            Encode(encoder, {.Bodies = bodies, .Joints = joints, .Iterations = settings.Iterations, .Colors = colors, .ColoringPasses = settings.ColoringPasses, .ColliderFeatures = collider_features, .Parameter = index, .GpuColors = index != 0, .Snapshot = snapshot}, world);
        }
        encoder->endEncoding();
        Commands->endCommandBuffer();
        const MTL4::CommandBuffer *list[]{Commands.get()};
        Context.Queue->commit(list, 1);
        Context.Queue->signalEvent(Done.get(), ++Signal);
        while (!Done->waitUntilSignaledValue(Signal, 1000)) {}

        std::exception_ptr observer_error;
        for (uint32_t index = 0; index < count; ++index) {
            const uint32_t bodies = index == 0 ? first_bodies : later_bodies;
            StepSnapshot saved;
            StepCompletion completion;
            StepResult step{};
            if (snapshot) {
                const uint64_t base = index * Layout.Stride;
                const auto view = [&]<typename T>(uint64_t offset, uint32_t length) {
                    return std::span<const T>{reinterpret_cast<const T *>(Outputs.Data() + base + offset), length};
                };
                completion = view.template operator()<StepCompletion>(Layout.Completion, 1)[0];
                saved.Counts = view.template operator()<StepCounts>(Layout.Counts, bodies);
                if (poses) {
                    saved.Poses = view.template operator()<Pose>(Layout.Poses, bodies);
                    saved.Velocities = view.template operator()<Velocity>(Layout.Velocities, bodies);
                }
                if (world.TrackContacts) {
                    saved.InitialPoses = view.template operator()<Pose>(Layout.Initial, bodies);
                    saved.Contacts = view.template operator()<ContactReport>(Layout.Contacts, bodies * ContactsPerBody);
                    saved.RemovedContacts = view.template operator()<ContactEvent>(Layout.RemovedContacts, bodies * ContactsPerBody);
                }
                if (sensors) saved.Sensors = view.template operator()<SensorPair>(Layout.Sensors, bodies * ContactsPerBody);
                for (const auto &counts : saved.Counts) {
                    step.ContactRefusals += counts.ContactRefusals;
                    step.SensorRefusals += counts.SensorRefusals;
                }
            } else {
                const auto root = world.BroadPhaseNodes[BroadPhaseRoot(bodies)];
                completion = {Params[index].MaxColors, root.Ready, root.Errors};
                for (uint32_t body = 0; body < bodies; ++body) {
                    step.ContactRefusals += world.ContactRefusals[body];
                    if (sensors) step.SensorRefusals += world.SensorRefusals[body];
                }
            }
            if (completion.Ready != (bodies <= RadixSimdWidth ? 0u : 2u) || completion.Errors)
                throw std::runtime_error("GPU broad phase did not complete");
            world.OnStepped(settings.DeltaTime, saved);
            ++result.Steps;
            result.ContactRefusals += step.ContactRefusals;
            result.SensorRefusals += step.SensorRefusals;
            step.Step = world.CompletedSteps;
            step.Poses = (snapshot ? saved.Poses : world.Poses.All()).first(poses ? world.BodyCount() : 0);
            step.Velocities = (snapshot ? saved.Velocities : world.Velocities.All()).first(poses ? world.BodyCount() : 0);
            if (observer && !observer_error) {
                try {
                    observer(step);
                } catch (...) { observer_error = std::current_exception(); }
            }
        }
        if (observer_error) std::rethrow_exception(observer_error);
    }
    return result;
}

void Solver::Encode(MTL4::ComputeCommandEncoder *encoder, const Recording &recording, World &world) {
    const uint32_t bodies = recording.Bodies, joints = recording.Joints;
    const bool compact = bodies > SolveLanes;
    const bool general_islands = compact && recording.Iterations;
    const uint32_t slots = bodies * ContactsPerBody;
    for (const auto range : FollowerRanges) {
        Table->setAddress(Followers.Address() + range.First * sizeof(SensorFollower), 26);
        Dispatch(encoder, FollowPass, range.Count);
    }
    const uint64_t groups = ColorGroups.Address() + recording.Parameter * MaxSupportedColors * 3 * sizeof(uint32_t);
    if (recording.GpuColors) {
        Table->setAddress(groups, 26);
        Dispatch(encoder, PrepareColorsPass, 32);
    }
    Table->setAddress(world.ContactRefusals.Address(), 26);

    // Each color reads one immutable cursor slot.
    const auto sweep = [&](Pass primal) {
        for (uint32_t color = 0; color < recording.Colors; ++color) {
            Table->setAddress(ColorCursor.Address() + color * sizeof(uint32_t), BroadPhaseOrCursorAt);
            const uint64_t work_groups = general_islands && primal == SolvePass ? GeneralIslands.Address() + (IslandColorsAt + color * 3) * sizeof(uint32_t) :
                (compact || recording.GpuColors)                                ? groups + color * 3 * sizeof(uint32_t) :
                                                                                  0;
            Dispatch(encoder, primal, bodies, 1, work_groups);
        }
        Dispatch(encoder, PublishPass, bodies, 1, general_islands && primal == SolvePass ? GeneralIslands.Address() + IslandPublishAt * sizeof(uint32_t) : 0);
    };

    const uint32_t bounds_lanes = bodies <= RadixSimdWidth ? RadixSimdWidth : 1;
    const bool queued_queries = (recording.ColliderFeatures & MeshQueries) && uint64_t(bodies) * (QueryPartitions(bodies) + 1) * sizeof(uint32_t) + sizeof(QueryArenaHeader) + 48 < QueryScratchBytes;
    if (queued_queries) {
        Table->setAddress(QueryScratch.Address(), 11);
        Table->setAddress(QueryInputs.Address(), 9);
        Table->setAddress(QueryInputSnapshot.Address(), 18);
        Dispatch(encoder, ResetQueryReusePass, 1);
        Dispatch(encoder, CacheBoundsPass, bodies, bounds_lanes);
        Table->setAddress(world.Iterates.Address(), 11);
        Table->setAddress(world.PreviousVelocities.Address(), 9);
        Table->setAddress(world.IncomingSlots.Address(), 18);
    } else Dispatch(encoder, BoundsPass, bodies, bounds_lanes);
    if (bodies <= RadixBlockSize) {
        Table->setAddress(world.BroadPhaseKeys.Address(), 11);
        Table->setAddress(world.BroadPhaseNodes.Address(), 13);
        Dispatch(encoder, SmallBroadPhasePass, std::bit_ceil(std::max(RadixSimdWidth, bodies)));
    } else {
        const uint32_t blocks = RadixBlocks(bodies);
        Table->setAddress(world.BoundsReductions.Address(), 13);
        Dispatch(encoder, ReduceBoundsPass, blocks * RadixBlockSize);
        Dispatch(encoder, ReduceScenePass, RadixBlockSize);
        Table->setAddress(world.BroadPhaseKeys.Address(), 11);
        Table->setAddress(world.BroadPhaseNodes.Address(), 10);
        Dispatch(encoder, MortonPass, bodies);
        Table->setAddress(world.BroadPhaseScratch.Address(), 9);
        for (uint32_t digit = 0; digit < 4; ++digit) {
            const uint64_t input = world.BroadPhaseKeys.Address() + (digit % 2) * bodies * sizeof(MortonKey);
            const uint64_t output = world.BroadPhaseKeys.Address() + ((digit + 1) % 2) * bodies * sizeof(MortonKey);
            Table->setAddress(input, 11);
            Table->setAddress(output, 13);
            Table->setAddress(ColorCursor.Address() + digit * sizeof(uint32_t), 25);
            Dispatch(encoder, RadixHistogramPass, blocks * RadixBlockSize);
            Dispatch(encoder, RadixOffsetsPass, RadixBlockSize);
            Dispatch(encoder, RadixScatterPass, bodies);
        }
        Table->setAddress(world.BroadPhaseKeys.Address(), 11);
        Table->setAddress(world.BroadPhaseNodes.Address(), 13);
        Dispatch(encoder, BuildTreePass, bodies);
        Dispatch(encoder, RefitTreePass, bodies);
    }
    Table->setAddress(world.BroadPhaseNodes.Address(), BroadPhaseOrCursorAt);
    Table->setAddress(world.PreviousVelocities.Address(), 9);
    Table->setAddress(world.Materials.Address(), 10);
    Table->setAddress(world.Iterates.Address(), 11);
    Table->setAddress(world.NextColors.Address(), 13);
    Table->setAddress(world.ContactEventCounts.Address(), 25);
    Dispatch(encoder, IntegratePass, bodies);
    // Collision, and with it every C0, anchor and Jacobian, is taken at the pose the step began from, before WarmStart moves the body to its starting guess.
    // This is the reference's order, and the only one that expands the Taylor series about the pose the constraint was measured at.
    constexpr Pass collectors[]{CollectPass, BoundedCollectPass, MeshCollectPass, FullCollectPass};
    const uint32_t geometry_features = recording.ColliderFeatures & (BoundedPlanes | MeshPairs);
    // Shared lanes increase parallelism for mesh queries and small worlds.
    const uint32_t collision_lanes = bodies <= RadixSimdWidth || (recording.ColliderFeatures & MeshQueries) ? CollisionLanes : 1;
    const auto collect = [&](Pass pass, bool sensor) {
        if (!(recording.ColliderFeatures & MeshQueries) || uint64_t(bodies) * (QueryPartitions(bodies) + 1) * sizeof(uint32_t) + sizeof(QueryArenaHeader) + 48 >= QueryScratchBytes) {
            Dispatch(encoder, pass, bodies, collision_lanes);
            return;
        }
        // Retain solid geometry while sensors use separate scratch storage.
        const uint64_t query_address = sensor ? SensorQueries.Address() : QueryScratch.Address();
        Table->setAddress(query_address, 11);
        if (!sensor) {
            Table->setAddress(QueryInputSnapshot.Address(), 18);
            Table->setAddress(QueryInputs.Address(), 9);
            Dispatch(encoder, CheckQueryInputsPass, 1, 1, query_address + offsetof(QueryArenaHeader, InputX));
            Dispatch(encoder, CheckQueryGeometryPass, 1, 1, query_address + offsetof(QueryArenaHeader, GeometryX));
            Table->setAddress(world.IncomingSlots.Address(), 18);
            Table->setAddress(world.PreviousVelocities.Address(), 9);
        }
        Dispatch(encoder, ResetQueryPass, bodies);
        Dispatch(encoder, pass, bodies, collision_lanes, 0, Prepare);
        Dispatch(encoder, sensor ? SensorQueryPass : QueryPass, 1, 1, query_address);
        Dispatch(encoder, pass, bodies, 1, 0, Queued);
        // Recompute queries that exceed arena capacity.
        Dispatch(encoder, pass, bodies, collision_lanes, 0, Recompute);
        Table->setAddress(world.Iterates.Address(), 11);
    };
    collect(collectors[geometry_features], false);
    if (bodies <= SolveLanes) {
        Table->setAddress(world.JointIncidence.Address(), 20);
        Table->setAddress(world.Displacements.Address(), 10);
        Table->setAddress(SmallIslands.Address(), 26);
        Table->setAddress(ColorCursor.Address() + MaxSupportedColors * sizeof(uint32_t), BroadPhaseOrCursorAt);
        Dispatch(encoder, PrepareSmallWorldPass, SolveLanes);
    } else {
        // Gather each body's contacts-as-B into a contiguous run, so the passes below do not scan the whole pool.
        Table->setAddress(world.BroadPhaseScratch.Address(), BroadPhaseOrCursorAt);
        Dispatch(encoder, ScanIncomingPass, bodies);
        if (bodies > RadixBlockSize) {
            Dispatch(encoder, ScanIncomingBlocksPass, RadixBlockSize);
            Dispatch(encoder, OffsetIncomingPass, bodies);
        }
        Dispatch(encoder, FillIncomingPass, bodies);
        Dispatch(encoder, SortIncomingPass, bodies);
        Table->setAddress(world.JointIncidence.Address(), 20);
        if (joints > 0) Dispatch(encoder, PrepareJointsPass, joints);
        Table->setAddress(world.Displacements.Address(), 10);
        Table->setAddress(ColorScratch.Address(), 26);
        Dispatch(encoder, WarmStartPass, bodies);
        for (uint32_t pass = 0; pass < recording.ColoringPasses; ++pass) {
            Dispatch(encoder, ColorPass, bodies);
            Dispatch(encoder, PublishColorPass, bodies);
        }
    }
    if (compact) {
        Table->setAddress(groups, BroadPhaseOrCursorAt);
        Dispatch(encoder, CountColorWorkPass, bodies);
        Dispatch(encoder, PrefixColorWorkPass, MaxSupportedColors);
        Dispatch(encoder, FillColorWorkPass, bodies);
    }
    if (general_islands) {
        Table->setAddress(GeneralIslands.Address(), 26);
        Dispatch(encoder, InitializeIslandsPass, bodies);
        Dispatch(encoder, UnionIslandsPass, std::max(bodies, joints));
        Dispatch(encoder, PackIslandsPass, bodies);
        Dispatch(encoder, FinishIslandsPass, MaxSupportedColors);
        Table->setAddress(ColorCursor.Address() + MaxSupportedColors * sizeof(uint32_t), BroadPhaseOrCursorAt);
        Dispatch(encoder, SolveIslandsPass, bodies, 1, GeneralIslands.Address());
        Table->setAddress(ColorScratch.Address(), 26);
    }
    if (bodies <= SolveLanes) {
        if (recording.Iterations) {
            Table->setAddress(ColorCursor.Address() + MaxSupportedColors * sizeof(uint32_t), BroadPhaseOrCursorAt);
            Table->setAddress(SmallIslands.Address(), 26);
            Dispatch(encoder, SolveSmallWorldPass, bodies);
            Table->setAddress(groups, 26);
        }
    } else {
        for (uint32_t iteration = 0; iteration < recording.Iterations; ++iteration) {
            sweep(SolvePass);
            Dispatch(encoder, DualPass, slots, 1, GeneralIslands.Address() + IslandContactDualAt * sizeof(uint32_t));
            if (joints > 0) Dispatch(encoder, JointDualPass, joints, 1, GeneralIslands.Address() + IslandJointDualAt * sizeof(uint32_t));
        }
    }
    // Velocity is taken from the motion the iterations above produced, before the stabilization sweep, so removing leftover penetration adds no velocity.
    Dispatch(encoder, FinalizePass, bodies);
    // Restitution runs as a velocity pass rather than as a row inside the solve. See Restitution for the gapped-contact case that requires it.
    // It runs before quiet counting, because a body given a rebound this step is not at rest.
    if (recording.ColliderFeatures & RestitutionMaterials)
        for (uint32_t pass = 0; pass < RestitutionPasses; ++pass) {
            Dispatch(encoder, RestitutionPass, slots);
            Dispatch(encoder, ApplyRestitutionPass, bodies);
        }
    sweep(StabilizePass);
    // Sleep state is settled at the very end of a step, after the stabilization sweep, so every kernel of a step sees one sleep state per body.
    // Published before the sweep, a body woken by the spread would run a stabilization pass for a step it slept through.
    // Asleep at Integrate its inertial target is its frozen pose with no gravity in it, while its contact duals still carry the force balancing gravity.
    // The pass then shifts it out of rest and the body above penetrates it.
    Dispatch(encoder, CountQuietPass, bodies);
    Dispatch(encoder, SpreadWakingPass, bodies);
    Dispatch(encoder, PublishWakingPass, bodies);

    if (OutputFlags[0].Sensors) {
        Table->setAddress(world.Jointed.Address(), 20);
        Table->setAddress(world.SensorContacts.Address(), 5);
        Table->setAddress(world.SensorRefusals.Address(), 26);
        Table->setAddress(world.Materials.Address(), 10);
        constexpr Pass sensor_collectors[]{SensorPass, BoundedSensorPass, MeshSensorPass, FullSensorPass};
        Table->setAddress(world.Bounds.Address(), BroadPhaseOrCursorAt);
        // Sensors query the final poses, after solving and stabilization.
        Dispatch(encoder, SensorBoundsPass, bodies, bounds_lanes);
        Table->setAddress(world.BroadPhaseNodes.Address(), 13);
        Dispatch(encoder, RefreshTreePass, bodies);
        if (bodies > RadixSimdWidth) Dispatch(encoder, RefitTreePass, bodies);
        Table->setAddress(world.BroadPhaseNodes.Address(), BroadPhaseOrCursorAt);
        Table->setAddress(world.NextColors.Address(), 13);
        collect(sensor_collectors[geometry_features], true);
    }

    if (recording.Snapshot) {
        const uint64_t output = Outputs.Address() + recording.Parameter * Layout.Stride;
        Table->setAddress(world.Contacts.Address(), 5);
        Table->setAddress(OutputFlags[0].Sensors ? world.SensorContacts.Address() : world.Contacts.Address(), 6);
        Table->setAddress(output + Layout.Contacts, 8);
        Table->setAddress(world.ContactEvents.Address(), 9);
        Table->setAddress(world.ContactEventCounts.Address(), 10);
        Table->setAddress(output + Layout.Initial, 11);
        Table->setAddress(world.BroadPhaseNodes.Address(), 13);
        Table->setAddress(OutputFlags[0].Sensors ? world.SensorRefusals.Address() : world.ContactRefusals.Address(), 14);
        Table->setAddress(output + Layout.Poses, 15);
        Table->setAddress(output + Layout.Velocities, 16);
        Table->setAddress(output + Layout.Sensors, 17);
        Table->setAddress(output + Layout.Counts, 18);
        Table->setAddress(output + Layout.Completion, 19);
        Table->setAddress(output + Layout.RemovedContacts, 21);
        Table->setAddress(OutputFlags.Address(), 20);
        Table->setAddress(world.ContactRefusals.Address(), 26);
        Dispatch(encoder, CapturePass, bodies);
    }
}

} // namespace rbp
