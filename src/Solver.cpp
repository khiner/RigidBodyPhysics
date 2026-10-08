#include "Solver.h"
#include "Mesh.h"
#include "metal/AutoreleasePool.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <exception>
#include <numeric>
#include <map>
#include <set>
#include <semaphore>
#include <string>

namespace rbp {

namespace {
// Collision and primal passes use this binding at separate times.
constexpr uint32_t BroadPhaseOrCursorAt = 14;
constexpr uint32_t BindingCount = 31;
static_assert(SolveBodiesPerGroup == 5 && CollisionLanes == 64 && RadixSimdWidth == 32);
constexpr uint32_t BatchSteps = 16;
constexpr uint64_t BatchBytes = 32 * 1024 * 1024;

// Reserve one additional color for new contacts.
uint32_t ColorsNeeded(const World &world, const StepSettings &settings) {
    uint32_t used = 1;
    for (uint32_t body = 0; body < world.BodyCount(); ++body)
        if (Moves(world.Masses[body])) used = std::max(used, ColorOf(world.Colors[body]) + 1);
    return std::clamp(used + 1, 1u, std::max(1u, std::min(settings.MaxColors, MaxSupportedColors)));
}

constexpr uint32_t BoundedPlanes = 1, MeshQueries = 2, RestitutionMaterials = 4, FullStepGeometry = 8;
constexpr uint32_t SolveCursorCount = 2 * MaxSupportedColors;
constexpr uint32_t FullStepBodyLimit = 9;
// A full SIMD group per native body improves utilization before scalar body threads fill the GPU.
constexpr uint32_t NativeCollectBodyLimit = 512;
constexpr uint32_t SerialColorBudgetBodyLimit = 512;
static_assert(FullStepBodyLimit <= SolveLanes);

uint32_t ColliderFeatures(const World &world) {
    uint32_t features = 0;
    bool full_step_geometry = true;
    for (Index body = 0; body < world.BodyCount(); ++body) {
        const Index root = world.BodyShapes[body];
        if (root == NoIndex) continue;
        const Shape &shape = world.Shapes[root];
        full_step_geometry &= FullStepShape(shape.Kind);
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
        if (mesh) features |= MeshQueries;
        if (features == (BoundedPlanes | MeshQueries | RestitutionMaterials)) break;
    }
    return features | (full_step_geometry ? FullStepGeometry : 0u);
}

} // namespace

Solver::Solver(const mtl::Context &context) : Context(context) {
    const mtl::AutoreleasePool pool;
    NS::Error *error{};
    auto descriptor = mtl::Make<MTL4::ArgumentTableDescriptor>();
    descriptor->setMaxBufferBindCount(BindingCount);
    auto table = NS::TransferPtr(context.Device->newArgumentTable(descriptor.get(), &error));
    auto allocator = NS::TransferPtr(context.Device->newCommandAllocator());
    auto commands = NS::TransferPtr(context.Device->newCommandBuffer());
    auto done = NS::TransferPtr(context.Device->newSharedEvent());

    Params = {context.Device.get(), BatchSteps};
    ColorGroups = {context.Device.get(), BatchSteps * MaxSupportedColors * 3};
    SmallIslands = {context.Device.get(), SolveLanes};
    ColorScratch = {context.Device.get(), 1};
    OutputFlags = {context.Device.get(), 1};
    FullData = {context.Device.get(), BatchSteps};
    ColorCursor = {context.Device.get(), SolveCursorCount + 2};
    for (uint32_t cursor = 0; cursor < SolveCursorCount; ++cursor) ColorCursor[cursor] = cursor;
    auto residency = NS::TransferPtr(context.Device->newResidencySet(mtl::Make<MTL::ResidencySetDescriptor>().get(), &error));
    residency->addAllocation(Params.Handle.get());
    residency->addAllocation(ColorGroups.Handle.get());
    residency->addAllocation(SmallIslands.Handle.get());
    residency->addAllocation(ColorScratch.Handle.get());
    residency->addAllocation(OutputFlags.Handle.get());
    residency->addAllocation(FullData.Handle.get());
    residency->addAllocation(ColorCursor.Handle.get());
    residency->commit();
    residency->requestResidency();
    context.Queue->addResidencySet(residency.get());
    Table = std::move(table);
    Allocator = std::move(allocator);
    Commands = std::move(commands);
    Done = std::move(done);
    Residency = std::move(residency);
}

Solver::~Solver() {
    const mtl::AutoreleasePool pool;
    Context.Queue->removeResidencySet(Residency.get());
    mtl::Drain(Context.Queue.get());
    mtl::AutoreleasePool::Release(SolveCommands, Residency, Done, Commands, Allocator, Table, Pipelines);
}

MTL::ComputePipelineState *Solver::Pipeline(uint32_t index) {
    auto &pipeline = Pipelines.at(index);
    if (!pipeline) {
        auto loaded = Context.Pipeline(index);
        if (index == shaders::FullStep && (loaded->threadExecutionWidth() != SolveLanes || loaded->maxTotalThreadsPerThreadgroup() < SmallSolveWaves * SolveLanes))
            throw std::runtime_error("The complete step requires eight 32-lane SIMD groups");
        if (index == shaders::EncodeSolveCommands && loaded->threadExecutionWidth() != SolveLanes)
            throw std::runtime_error("Solve command encoding requires 32 lanes");
        if (shaders::Pipelines[index].Indirect) {
            Residency->addAllocation(loaded.get());
            Residency->commit();
        }
        pipeline = std::move(loaded);
    }
    return pipeline.get();
}

void Solver::Dispatch(MTL4::ComputeCommandEncoder *encoder, Pass pass, uint32_t threads, uint32_t lanes, uint64_t indirect, CollectionMode mode) {
    const bool native = pass == NativeCollectPass;
    const bool collector = pass == CollectPass || pass == SensorPass || pass == ScalarCollectPass || native;
    const bool primal = pass == SolvePass;
    const bool bounds = pass == BoundsPass || pass == GeometryBoundsPass || pass == SensorBoundsPass;
    const uint32_t variant = native && threads <= RadixSimdWidth ? 0 :
        collector                                                ? uint32_t(native) + 2 * uint32_t(lanes > 1) :
        bounds                                                   ? uint32_t(lanes > 1) :
        primal                                                   ? uint32_t(threads > WideSolveBodyLimit) :
                                                                   0;
    auto *pipeline = Pipeline(shaders::PipelineIndices[mode][pass][variant]);
    // Each dispatch requires visibility of the preceding dispatch's writes.
    encoder->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
    encoder->setComputePipelineState(pipeline);
    // Small threadgroups distribute divergent body work across GPU cores.
    const bool per_body = collector || primal;
    const bool radix = pass == RadixHistogramPass || pass == RadixOffsetsPass || pass == ReduceBoundsPass || pass == ReduceScenePass;
    const bool scan = pass == ScanIncomingPass || pass == ScanIncomingBlocksPass;
    const bool contact = pass == DualPass || pass == RestitutionPass;
    if ((radix || scan || pass == PrefixColorWorkPass || pass == ReduceStepColorsPass || pass == FinishStepColorsPass || pass == PrepareSmallWorldPass || pass == SmallBroadPhasePass || ((collector || bounds) && lanes > 1)) && pipeline->threadExecutionWidth() != RadixSimdWidth)
        throw std::runtime_error("GPU scans require 32-lane SIMD groups");
    const uint32_t limit = scan ? RadixBlockSize : per_body ? 8u :
        contact                                             ? 64u :
                                                              pipeline->maxTotalThreadsPerThreadgroup();
    const auto group = pass == SmallBroadPhasePass ? threads : radix ? RadixBlockSize :
                                                                       std::min(threads, limit);
    if (pass == ReduceStepColorsPass) {
        encoder->dispatchThreadgroups({(threads + 127) / 128, 1, 1}, {128, 1, 1});
    } else if (pass == SolveIslandsPass) {
        const uint32_t width = std::min(threads, SmallSolveWaves) * SolveLanes;
        if (pipeline->threadExecutionWidth() != SolveLanes || width > pipeline->maxTotalThreadsPerThreadgroup())
            throw std::runtime_error("The island solve requires up to eight 32-lane SIMD groups");
        if (indirect) encoder->dispatchThreadgroups(indirect, {width, 1, 1});
        else encoder->dispatchThreadgroups({threads, 1, 1}, {width, 1, 1});
    } else if (indirect && (pass == PublishPass || pass == DualPass || pass == JointDualPass)) {
        encoder->dispatchThreadgroups(indirect, {pass == PublishPass ? 128u : pass == DualPass ? 64u :
                                                                                                 32u,
                                                 1, 1});
    } else if (mode == Queued) encoder->dispatchThreadgroups({threads, 1, 1}, {1, 1, 1});
    else if (pass == CheckQueryGeometryPass || pass == CheckQueryInputsPass) encoder->dispatchThreadgroups(indirect, {128, 1, 1});
    else if (pass == QueryPass) encoder->dispatchThreadgroups(indirect ? indirect : QueryScratch.Address(), {32, 1, 1});
    else if (primal) {
        if (pipeline->threadExecutionWidth() != SolveLanes) throw std::runtime_error("The body solve requires 32-lane SIMD groups");
        const uint32_t bodies_per_group = FallbackBodiesPerGroup(threads);
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

void Solver::PrepareQueries(World &world, uint32_t collider_features) {
    if (!(collider_features & MeshQueries)) return;
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
    // Geometry is immutable until this call returns, including across submissions.
    reinterpret_cast<QueryArenaHeader *>(QueryScratch.Data())->GeometryChecked = 0;
    if (residency_changed) Residency->commit();
}

struct Solver::GeometryEpoch::Impl {
    Solver *Owner;
    const World *Source;
    uint64_t Epoch, Bytes = 0;
    uint32_t Bodies, Shapes, Joints, Features, Capacity, Encoded = 0;
    uint32_t VertexCapacity, TriangleCapacity, NodeCapacity;
    bool Failed = false;
    mtl::Buffer<BodyBounds> Bounds, Reductions;
    mtl::Buffer<BroadPhaseNode> Nodes;
    mtl::Buffer<MortonKey> Keys;
    mtl::Buffer<uint32_t> Scratch, QueryScratch, QuerySnapshot, Counts, Digits, Refusals;
    mtl::Buffer<QueryInputSpec> QueryInputs;
    mtl::Buffer<StepParams> Parameters;
    mtl::Buffer<GeometryContact> Contacts;
    mtl::Buffer<GeometryQueryStatus> Status;
    mtl::Buffer<BodyId> Ids;
    mtl::Buffer<MeshRefitTriangle> RefitTriangles;
    mtl::Buffer<MeshRefitNode> RefitNodes;
    mtl::Buffer<uint32_t> RefitVertices, RefitFarVertices, RefitDepths;
    mtl::Buffer<MeshRefitCounts> RefitCounts;
    mtl::Buffer<float3> RefitNormals;
    mtl::Buffer<std::array<uint32_t,2>> RefitFlags;
    mtl::Buffer<BodyBounds> RefitBounds;
    uint32_t RefitMaximumDepth = 0;
    mtl::Buffer<Filter> Filters;
    std::array<uint64_t, BindingCount> Bindings{};
    std::array<NS::SharedPtr<MTL::Buffer>, BindingCount> ModelBuffers;
    std::vector<GeometryView> Views;
    std::vector<uint8_t> Bound, Live;
    std::vector<Index> Roots;
    std::vector<NS::SharedPtr<MTL4::ArgumentTable>> Tables;
    NS::SharedPtr<MTL::ResidencySet> Residency;

    Impl(Solver &owner, const World &world, uint32_t capacity, uint32_t queries, uint64_t epoch)
        : Owner(&owner), Source(&world), Epoch(epoch), Bodies(world.BodyCount()), Shapes(world.ShapeCount()), Joints(world.JointCount()),
          Features(ColliderFeatures(world)), Capacity(capacity),
          VertexCapacity(world.ShapeVertices.Capacity), TriangleCapacity(world.Triangles.Capacity), NodeCapacity(world.BvhNodes.Capacity),
          Views(queries), Bound(queries), Live(Bodies), Roots(Bodies), Tables(queries) {}
    void CheckWorld() const {
        if (Source->BodyCount() != Bodies || Source->ShapeCount() != Shapes || Source->JointCount() != Joints)
            throw std::logic_error("Geometry epoch topology changed");
        for (uint32_t i = 0; i < Bodies; ++i)
            if (Source->IdOf(i) != Ids[i] || Source->Alive(i) != bool(Live[i]) || Source->BodyShapes[i] != Roots[i])
                throw std::logic_error("Geometry epoch body generation or shape changed");
    }
    CollisionBuffers Storage() const {
        const auto address = [](const auto &b) { return b.Handle ? b.Address() : uint64_t(0); };
        return {Bounds.Address(), Reductions.Address(), Nodes.Address(), Keys.Address(), Scratch.Address(),
            Bindings[11], Bindings[9], Bindings[18], Bindings[10], Bindings[13], Bindings[25],
            address(QueryScratch), address(QuerySnapshot), address(QueryInputs), Digits.Address()};
    }
};

Solver::GeometryEpoch::GeometryEpoch(std::unique_ptr<Impl> data) : Data(std::move(data)) {}
Solver::GeometryEpoch::GeometryEpoch(GeometryEpoch &&) noexcept = default;
Solver::GeometryEpoch &Solver::GeometryEpoch::operator=(GeometryEpoch &&) noexcept = default;
Solver::GeometryEpoch::~GeometryEpoch() = default;
MTL::ResidencySet *Solver::GeometryEpoch::ResidencySet() const { return Data ? Data->Residency.get() : nullptr; }
uint64_t Solver::GeometryEpoch::StorageBytes() const { return Data ? Data->Bytes : 0; }
uint32_t Solver::GeometryEpoch::EncodedQueries() const { return Data ? Data->Encoded : 0; }
std::span<const BodyId> Solver::GeometryEpoch::BodyIds() const { return Data ? std::span<const BodyId>(Data->Ids.All()) : std::span<const BodyId>{}; }

GeometryQueryStatus Solver::GeometryEpoch::StatusAfterCompletion(uint32_t slot) const {
    if (!Data || slot >= Data->Encoded) throw std::out_of_range("Geometry status slot is not encoded");
    const auto result = Data->Status[slot];
    if (!result.Complete) throw std::logic_error("Geometry query has not completed");
    return result;
}
std::span<const GeometryContact> Solver::GeometryEpoch::ContactsAfterCompletion() const {
    if (!Data || !Data->Encoded) throw std::logic_error("Geometry epoch has no query");
    const auto status = StatusAfterCompletion(Data->Encoded - 1);
    if (status.Errors || status.Count > Data->Capacity) throw std::runtime_error("Geometry query failed; no contacts published");
    return Data->Contacts.All().first(status.Count);
}
void Solver::GeometryEpoch::ResetAfterCompletion() {
    if (!Data) throw std::logic_error("Moved geometry epoch");
    Data->CheckWorld();
    Data->Encoded = 0; Data->Failed = false;
}

Solver::GeometryEpoch Solver::PrepareGeometryEpoch(const World &world, const StepSettings &settings,
        uint32_t capacity, uint32_t maximum_queries, uint64_t epoch, std::span<const Index> refit_shapes) {
    const mtl::AutoreleasePool pool;
    if (Advancing || !world.BodyCount() || !capacity || !maximum_queries || !epoch)
        throw std::invalid_argument("Geometry epoch needs an idle solver and nonzero capacities/identity");
    if (world.Queue.get() != Context.Queue.get()) throw std::invalid_argument("Geometry epoch requires the shared queue");
    auto data = std::make_unique<GeometryEpoch::Impl>(*this, world, capacity, maximum_queries, epoch);
    NS::Error *error{};
    data->Residency = NS::TransferPtr(Context.Device->newResidencySet(mtl::Make<MTL::ResidencySetDescriptor>().get(), &error));
    if (!data->Residency) throw std::runtime_error("Geometry epoch residency allocation failed");
    const auto make = [&](auto &buffer, uint64_t count) {
        if (!count || count > UINT32_MAX) throw std::length_error("Geometry epoch capacity overflow");
        buffer = {Context.Device.get(), uint32_t(count)};
        if (!buffer.Handle) throw std::bad_alloc();
        data->Bytes += buffer.Handle->length();
        data->Residency->addAllocation(buffer.Handle.get());
    };
    const uint32_t bodies = data->Bodies;
    make(data->Bounds, bodies); make(data->Reductions, RadixBlocks(bodies) + 1);
    make(data->Nodes, uint64_t(2) * bodies); make(data->Keys, uint64_t(2) * bodies);
    make(data->Scratch, uint64_t(bodies) + uint64_t(RadixBlocks(bodies)) * RadixBins);
    make(data->Contacts, capacity); make(data->Counts, maximum_queries); make(data->Status, maximum_queries);
    make(data->Ids, bodies); make(data->Filters, bodies); make(data->Refusals, bodies); make(data->Parameters, maximum_queries); make(data->Digits, 4);
    for (uint32_t i = 0; i < bodies; ++i) { data->Ids[i] = world.IdOf(i); data->Filters[i] = world.Filters[i]; data->Live[i] = world.Alive(i); data->Roots[i] = world.BodyShapes[i]; }
    world.RefreshFilters(data->Filters.All());
    for (uint32_t i = 0; i < 4; ++i) data->Digits[i] = i;
    std::fill(data->Status.All().begin(), data->Status.All().end(), GeometryQueryStatus{});
    std::fill(data->Counts.All().begin(), data->Counts.All().end(), 0u);
    data->Parameters[0] = {.Gravity = settings.Gravity, .DeltaTime = settings.DeltaTime,
        .ContactMargin = settings.ContactMargin, .MaxContactReach = settings.MaxContactReach,
        .BodyCount = bodies, .JointCount = world.JointCount(),
        .QueuedQueries = bool(data->Features & MeshQueries) && QueuedQueriesFit(bodies),
        .GeometryOnly = 1, .GeometryCapacity = capacity};
    for (uint32_t i = 0; i < maximum_queries; ++i) {
        data->Parameters[i] = data->Parameters[0];
        data->Parameters[i].GeometryStatusAddress = data->Status.Address()+uint64_t(i)*sizeof(GeometryQueryStatus);
    }
    if (!refit_shapes.empty()) {
        std::vector<MeshRefitTriangle> triangle_map;
        std::vector<MeshRefitNode> node_map;
        std::vector<uint32_t> vertex_map, far_vertices;
        std::set<Index> selected_shapes, selected_vertices, selected_triangles, selected_nodes;
        for (const Index index : refit_shapes) {
            if (index >= world.ShapeCount() || !world.LiveShapes[index] || !selected_shapes.insert(index).second)
                throw std::invalid_argument("Refit selection requires unique live mesh shapes");
            const Shape &shape = world.Shapes[index];
            if (shape.Kind != ShapeMesh || uint64_t(shape.FirstVertex)+shape.VertexCount > world.ShapeVertices.Capacity ||
                uint64_t(shape.FirstTriangle)+shape.TriangleCount > world.Triangles.Capacity ||
                uint64_t(shape.RootNode)+shape.NodeCount > world.BvhNodes.Capacity)
                throw std::invalid_argument("Invalid geometry epoch refit ranges");
            const auto points = world.ShapeVertices.All().subspan(shape.FirstVertex,shape.VertexCount);
            const auto source_triangles = world.Triangles.All().subspan(shape.FirstTriangle,shape.TriangleCount);
            const auto source_nodes = world.BvhNodes.All().subspan(shape.RootNode,shape.NodeCount);
            std::vector<Triangle> local(source_triangles.begin(),source_triangles.end());
            std::vector<BvhNode> checked_nodes(source_nodes.begin(),source_nodes.end());
            for (auto &t : local) {
                for (Index *v : {&t.A,&t.B,&t.C}) {
                    if (*v < shape.FirstVertex || *v-shape.FirstVertex >= shape.VertexCount)
                        throw std::invalid_argument("Invalid refit triangle vertex");
                    *v -= shape.FirstVertex;
                }
            }
            // Use the authoritative CPU validator at immutable-epoch creation,
            // never as a per-query geometry update or numerical fallback.
            if (!RefitMesh(points,local,checked_nodes)) throw std::invalid_argument("Invalid cooked refit topology");
            for (uint32_t v=0;v<shape.VertexCount;++v) {
                const Index at=shape.FirstVertex+v;
                if (!selected_vertices.insert(at).second) throw std::invalid_argument("Overlapping refit mesh vertex runs");
                vertex_map.push_back(at);
            }
            std::map<std::pair<Index,Index>,std::vector<uint32_t>> edges;
            for (uint32_t t=0;t<shape.TriangleCount;++t) {
                const auto &x=source_triangles[t];const Index c[]{x.A,x.B,x.C};
                for(uint32_t e=0;e<3;++e)edges[{std::min(c[e],c[(e+1)%3]),std::max(c[e],c[(e+1)%3])}].push_back(t);
            }
            for (uint32_t t=0;t<shape.TriangleCount;++t) {
                MeshRefitTriangle out{};out.Original=source_triangles[t];out.Index=shape.FirstTriangle+t;
                if (!selected_triangles.insert(out.Index).second) throw std::invalid_argument("Overlapping refit triangle runs");
                const Index c[]{out.Original.A,out.Original.B,out.Original.C};
                for(uint32_t e=0;e<3;++e) {
                    const Index from=c[e],to=c[(e+1)%3];const auto &shared=edges.at({std::min(from,to),std::max(from,to)});
                    if (far_vertices.size()>UINT32_MAX || shared.size()>UINT32_MAX) throw std::length_error("Refit incidence exceeds device indices");
                    out.EdgeOffset[e]=uint32_t(far_vertices.size());out.SharedCount[e]=uint32_t(shared.size());
                    for(const uint32_t other:shared)if(other!=t) {
                        const auto &neighbor=source_triangles[other];
                        for(const Index far:{neighbor.A,neighbor.B,neighbor.C})if(far!=from&&far!=to)far_vertices.push_back(far);
                    }
                    if (far_vertices.size()>UINT32_MAX) throw std::length_error("Refit incidence exceeds device indices");
                    out.EdgeCount[e]=uint32_t(far_vertices.size())-out.EdgeOffset[e];
                }
                triangle_map.push_back(out);
            }
            if (node_map.size()+shape.NodeCount>UINT32_MAX) throw std::length_error("Refit node map overflow");
            const uint32_t base=uint32_t(node_map.size());node_map.resize(node_map.size()+shape.NodeCount);
            const auto visit=[&](auto &&self,uint32_t at,uint32_t depth)->void {
                const auto &node=source_nodes[at];const Index absolute=shape.RootNode+at;
                if (!selected_nodes.insert(absolute).second) throw std::invalid_argument("Overlapping refit node runs");
                node_map[base+at]={absolute,node.First,node.Count,shape.FirstTriangle,
                    node.Count?NoIndex:base+at+1,node.Count?NoIndex:base+node.First,depth};
                data->RefitMaximumDepth=std::max(data->RefitMaximumDepth,depth);
                if(!node.Count){self(self,at+1,depth+1);self(self,node.First,depth+1);}
            };visit(visit,0,0);
        }
        if(vertex_map.size()>UINT32_MAX||triangle_map.size()>UINT32_MAX)throw std::length_error("Refit work map overflow");
        const auto upload=[&](auto &buffer,const auto &values) {
            make(buffer,std::max(std::size_t(1),values.size()));
            std::copy(values.begin(),values.end(),buffer.Data());
        };
        upload(data->RefitVertices,vertex_map);upload(data->RefitTriangles,triangle_map);
        upload(data->RefitFarVertices,far_vertices);upload(data->RefitNodes,node_map);
        make(data->RefitCounts,1);data->RefitCounts[0]={uint32_t(vertex_map.size()),uint32_t(triangle_map.size()),uint32_t(node_map.size()),0};
        make(data->RefitNormals,triangle_map.size());make(data->RefitFlags,triangle_map.size());make(data->RefitBounds,node_map.size());
        make(data->RefitDepths,data->RefitMaximumDepth+1);
        for(uint32_t d=0;d<=data->RefitMaximumDepth;++d)data->RefitDepths[d]=d;
    }
    const auto retain = [&](uint32_t slot, const auto &buffer) {
        data->ModelBuffers[slot] = buffer.Handle;
        data->Bindings[slot] = buffer.Handle ? buffer.Address() : 0;
        if (buffer.Handle) data->Residency->addAllocation(buffer.Handle.get());
    };
    retain(0, world.Poses); retain(1, world.InitialPoses); retain(2, world.InertialPoses);
    retain(3, world.Velocities); retain(4, world.Masses); retain(5, world.Contacts);
    retain(6, world.BodyShapes); retain(8, world.Shapes); retain(9, world.PreviousVelocities);
    retain(10, world.Materials); retain(11, world.Iterates); retain(13, world.NextColors);
    retain(15, world.CompoundChildren); retain(17, world.Incoming); retain(18, world.IncomingSlots);
    retain(20, world.Jointed); retain(21, world.Quiet); retain(22, world.RestPoses); retain(23, world.NextQuiet);
    retain(24, world.ContactEvents); retain(25, world.ContactEventCounts); retain(26, world.ContactRefusals);
    retain(27, world.ShapeVertices); retain(28, world.Triangles); retain(29, world.BvhNodes); retain(30, world.HullFaces);
    data->Bindings[7] = data->Parameters.Address(); data->Bindings[12] = data->Contacts.Address();
    data->Bindings[14] = data->Bounds.Address(); data->Bindings[19] = data->Filters.Address();
    data->Bindings[26] = data->Refusals.Address();
    if (data->Parameters[0].QueuedQueries) {
        make(data->QueryScratch, QueryScratchBytes / sizeof(uint32_t)); make(data->QueryInputs, 1);
        auto &header = *reinterpret_cast<QueryArenaHeader *>(data->QueryScratch.Data());
        header = {}; header.Bytes = QueryScratchBytes;
        QueryInputSpec spec{};
        const uint64_t bytes[]{uint64_t(bodies)*sizeof(Pose), uint64_t(bodies)*sizeof(Velocity), uint64_t(bodies)*sizeof(BodyMass),
            uint64_t(bodies)*sizeof(Index), uint64_t(world.ShapeCount())*sizeof(Shape), uint64_t(bodies)*sizeof(Material),
            uint64_t(bodies)*sizeof(Filter), uint64_t(world.Jointed.Capacity)*sizeof(Index),
            uint64_t(world.ShapeVertices.Capacity)*sizeof(float3), uint64_t(world.HullFaces.Capacity)*sizeof(HullFace),
            uint64_t(world.Triangles.Capacity)*sizeof(Triangle), uint64_t(world.BvhNodes.Capacity)*sizeof(BvhNode),
            uint64_t(world.CompoundChildren.Capacity)*sizeof(Index), sizeof(StepParams), uint64_t(bodies)*sizeof(uint32_t),
            uint64_t(bodies)*ContactsPerBody*8*sizeof(uint32_t)};
        uint64_t words = 0;
        for (uint32_t i = 0; i < 16; ++i) {
            if (bytes[i]/4 > UINT32_MAX-words) throw std::length_error("Geometry epoch query snapshot overflow");
            spec.Offsets[i] = uint32_t(words); words += bytes[i]/4;
        }
        spec.Words = uint32_t(words); data->QueryInputs[0] = spec;
        make(data->QuerySnapshot, spec.Words);
        std::fill(data->QuerySnapshot.All().begin(), data->QuerySnapshot.All().end(), 0u);
    }
    for (auto &table : data->Tables) {
        auto desc = mtl::Make<MTL4::ArgumentTableDescriptor>(); desc->setMaxBufferBindCount(BindingCount);
        table = NS::TransferPtr(Context.Device->newArgumentTable(desc.get(), &error));
        if (!table) throw std::runtime_error("Geometry epoch argument table allocation failed");
    }
    // Pipeline creation is preparation, never an interior numerical operation.
    const auto warm = [&](Pass pass, uint32_t variant = 0, CollectionMode mode = Direct) {
        Pipeline(shaders::PipelineIndices[mode][pass][variant]);
    };
    warm(ResetGeometryOutputPass); warm(FinishGeometryOutputPass);
    if(data->RefitCounts.Handle)for(Pass pass:{MeshRefitValidatePass,MeshRefitLeavesPass,MeshRefitLevelPass,MeshRefitCommitPass})warm(pass);
    warm(GeometryBoundsPass, bodies <= RadixSimdWidth ? 1 : 0);
    if (bodies <= RadixBlockSize) warm(SmallBroadPhasePass);
    else for (Pass pass : {ReduceBoundsPass, ReduceScenePass, MortonPass, RadixHistogramPass,
                           RadixOffsetsPass, RadixScatterPass, BuildTreePass, RefitTreePass}) warm(pass);
    const bool mesh = data->Features & MeshQueries;
    const Pass collect = mesh ? CollectPass : bodies > NativeCollectBodyLimit && !(data->Features & BoundedPlanes) ? ScalarCollectPass : NativeCollectPass;
    const uint32_t lanes = bodies <= RadixSimdWidth || mesh ? CollisionLanes : bodies <= NativeCollectBodyLimit ? 32 : 1;
    const auto variant = [&](uint32_t width) { return collect == NativeCollectPass && bodies <= RadixSimdWidth ? 0u : uint32_t(collect == NativeCollectPass)+2*uint32_t(width>1); };
    warm(collect, variant(lanes));
    if (data->Parameters[0].QueuedQueries) {
        for (Pass pass : {ResetQueryReusePass, CheckQueryInputsPass, CheckQueryGeometryPass, ResetQueryPass, QueryPass}) warm(pass);
        warm(collect, variant(lanes), Prepare); warm(collect, variant(1), Queued);
    }
    data->Residency->commit(); data->Residency->requestResidency();
    return GeometryEpoch(std::move(data));
}

void Solver::GeometryEpoch::SetView(uint32_t slot, const GeometryView &view) {
    if (!Data || Data->Encoded || Data->Failed) throw std::logic_error("Geometry view must be bound before recording");
    auto &d = *Data; d.CheckWorld();
    if (slot >= d.Views.size()) throw std::out_of_range("Geometry epoch view slot");
    const auto validate = [&](const auto &b, uint32_t capacity) {
        if (!b.Handle || b.Capacity < capacity || b.Handle->device() != d.Owner->Context.Device.get())
            throw std::invalid_argument("Geometry staged buffer capacity/device mismatch");
    };
    validate(view.Poses, d.Bodies); validate(view.Velocities, d.Bodies);
    validate(view.Vertices, d.VertexCapacity); validate(view.Triangles, d.TriangleCapacity); validate(view.Nodes, d.NodeCapacity);
    if (d.Bound[slot]) throw std::logic_error("Geometry epoch slot is already bound");
    d.Views[slot] = view;
    for (auto *b : {view.Poses.Handle.get(), view.Velocities.Handle.get(), view.Vertices.Handle.get(), view.Triangles.Handle.get(), view.Nodes.Handle.get()})
        d.Residency->addAllocation(b);
    d.Residency->commit(); d.Bound[slot] = 1;
}

Solver::GeometryEpoch::Output Solver::GeometryEpoch::Encode(MTL4::ComputeCommandEncoder *encoder, uint32_t slot, uint64_t trial, bool refit) {
    if (!Data || !encoder || Data->Failed) throw std::invalid_argument("Invalid geometry epoch recording");
    auto &d = *Data; d.CheckWorld();
    if (slot != d.Encoded || slot >= d.Views.size() || !d.Bound[slot]) throw std::logic_error("Geometry query slots must be bound and encoded in order");
    if(refit&&!d.RefitCounts.Handle)throw std::invalid_argument("Geometry epoch has no prepared refit topology");
    auto *table = d.Tables[slot].get(); const auto &v = d.Views[slot];
    if(refit&&(v.Triangles.Handle.get()==d.ModelBuffers[28].get()||v.Nodes.Handle.get()==d.ModelBuffers[29].get()))
        throw std::invalid_argument("GPU refit requires private staged triangle and BVH buffers");
    const auto count = d.Counts.Address()+uint64_t(slot)*sizeof(uint32_t);
    const auto status = d.Status.Address()+uint64_t(slot)*sizeof(GeometryQueryStatus);
    d.Status[slot] = {.Epoch=d.Epoch,.Trial=trial};
    for (uint32_t i = 0; i < BindingCount; ++i) table->setAddress(d.Bindings[i], i);
    table->setAddress(v.Poses.Address(),0); table->setAddress(v.Velocities.Address(),3);
    table->setAddress(v.Vertices.Address(),27); table->setAddress(v.Triangles.Address(),28); table->setAddress(v.Nodes.Address(),29);
    table->setAddress(status,1); table->setAddress(count,16);
    table->setAddress(d.Parameters.Address()+uint64_t(slot)*sizeof(StepParams),7);
    table->setAddress(d.QueryScratch.Handle ? d.QueryScratch.Address() : 0,11);
    encoder->setArgumentTable(table);
    try {
        d.Owner->Dispatch(encoder, ResetGeometryOutputPass, 1);
        if(refit) {
            const auto n=d.RefitCounts[0];
            table->setAddress(d.RefitTriangles.Address(),2);table->setAddress(d.RefitFarVertices.Address(),4);
            table->setAddress(d.RefitVertices.Address(),6);table->setAddress(d.RefitNodes.Address(),8);
            table->setAddress(d.RefitFlags.Address(),10);table->setAddress(d.RefitBounds.Address(),11);
            table->setAddress(d.RefitNormals.Address(),12);table->setAddress(d.RefitCounts.Address(),13);
            d.Owner->Dispatch(encoder,MeshRefitValidatePass,std::max({n.Vertices,n.Triangles,n.Nodes}));
            d.Owner->Dispatch(encoder,MeshRefitLeavesPass,std::max(n.Triangles,n.Nodes));
            for(uint32_t depth=d.RefitMaximumDepth;depth-->0;) {
                table->setAddress(d.RefitDepths.Address()+uint64_t(depth)*sizeof(uint32_t),17);
                d.Owner->Dispatch(encoder,MeshRefitLevelPass,n.Nodes);
            }
            d.Owner->Dispatch(encoder,MeshRefitCommitPass,std::max(n.Triangles,n.Nodes));
            for(uint32_t binding:{2,4,6,8,10,11,12,13,17})table->setAddress(d.Bindings[binding],binding);
        }
        d.Owner->EncodeCollision(encoder, {.Bodies=d.Bodies,.ColliderFeatures=d.Features}, d.Storage(), table, d.Parameters[slot]);
        d.Owner->Dispatch(encoder, FinishGeometryOutputPass, 1);
    } catch (...) { d.Failed = true; throw; }
    ++d.Encoded;
    return {d.Contacts.Address(), count, status, d.Ids.Address(), d.Epoch, trial, d.Capacity, d.Bodies};
}

Solver::GeometryOutput Solver::EncodeGeometry(MTL4::ComputeCommandEncoder *encoder, World &world, const StepSettings &settings, uint32_t capacity) {
    if (Advancing) throw std::logic_error("Collision collection cannot reenter this solver");
    if (!encoder) throw std::invalid_argument("Geometry collection requires a compute encoder");
    if (!capacity) throw std::invalid_argument("Geometry contact capacity must be positive");
    if (!world.BodyCount()) throw std::invalid_argument("Geometry encoding requires at least one body");
    Advancing = true;
    try {
        world.RefreshFilters();
        const uint32_t features = ColliderFeatures(world);
        PrepareQueries(world, features);
        if (GeometryContacts.Capacity < capacity) {
            if (GeometryContacts.Handle) Residency->removeAllocation(GeometryContacts.Handle.get());
            GeometryContacts = {Context.Device.get(), capacity};
            Residency->addAllocation(GeometryContacts.Handle.get());
            Residency->commit();
        }
        if (!GeometryCount.Handle) {
            GeometryCount = {Context.Device.get(), 1};
            Residency->addAllocation(GeometryCount.Handle.get());
            Residency->commit();
        }
        GeometryCount[0] = 0;
        const uint32_t bodies = world.BodyCount();
        Params[0] = {
            .Gravity = settings.Gravity,
            .DeltaTime = settings.DeltaTime,
            .ContactMargin = settings.ContactMargin,
            .MaxContactReach = settings.MaxContactReach,
            .BodyCount = bodies,
            .JointCount = world.JointCount(),
            .QueuedQueries = bool(features & MeshQueries) && QueuedQueriesFit(bodies),
            .GeometryOnly = 1,
            .GeometryCapacity = capacity,
        };
        Bind(world, 0);
        Table->setAddress(GeometryContacts.Address(), 12);
        Table->setAddress(GeometryCount.Address(), 16);
        encoder->setArgumentTable(Table.get());
        EncodeCollision(encoder, {.Bodies = bodies, .ColliderFeatures = features}, world);
        PendingGeometry = &world;
        PendingGeometryBodies = bodies;
        PendingGeometryCapacity = capacity;
        return {GeometryContacts.Address(), GeometryCount.Address(), capacity};
    } catch (...) {
        CancelGeometry();
        throw;
    }
}

std::span<const GeometryContact> Solver::FinishGeometry(World &world) {
    if (!Advancing || PendingGeometry != &world || world.BodyCount() != PendingGeometryBodies)
        throw std::logic_error("Geometry completion does not match its encoding");
    const uint32_t bodies = PendingGeometryBodies;
    const uint32_t capacity = PendingGeometryCapacity;
    CancelGeometry();
    const BroadPhaseNode root = world.BroadPhaseNodes[BroadPhaseRoot(bodies)];
    if (root.Ready != (bodies <= RadixSimdWidth ? 0u : 2u) || root.Errors)
        throw std::runtime_error("GPU broad phase did not complete");
    if (GeometryCount[0] > capacity)
        throw std::overflow_error("Geometry contact capacity exceeded; no contacts were accepted for Ji stepping");
    return GeometryContacts.All().first(GeometryCount[0]);
}

void Solver::CancelGeometry() {
    PendingGeometry = nullptr;
    PendingGeometryBodies = 0;
    PendingGeometryCapacity = 0;
    Advancing = false;
}

std::span<const GeometryContact> Solver::CollectGeometry(World &world, const StepSettings &settings, uint32_t capacity) {
    if (Advancing) throw std::logic_error("Collision collection cannot reenter this solver");
    if (!capacity) throw std::invalid_argument("Geometry contact capacity must be positive");
    if (!world.BodyCount()) return {};
    Allocator->reset();
    Commands->beginCommandBuffer(Allocator.get());
    auto *encoder = Commands->computeCommandEncoder();
    try {
        EncodeGeometry(encoder, world, settings, capacity);
    } catch (...) {
        encoder->endEncoding();
        Commands->endCommandBuffer();
        throw;
    }
    encoder->endEncoding();
    Commands->endCommandBuffer();
    std::binary_semaphore done{0};
    std::string gpu_error;
    auto options = mtl::Make<MTL4::CommitOptions>();
    options->addFeedbackHandler([&](MTL4::CommitFeedback *feedback) {
        if (NS::Error *error = feedback->error()) gpu_error = error->localizedDescription()->utf8String();
        done.release();
    });
    const MTL4::CommandBuffer *list[]{Commands.get()};
    try {
        Context.Queue->commit(list, 1, options.get());
    } catch (...) {
        CancelGeometry();
        throw;
    }
    done.acquire();
    if (!gpu_error.empty()) {
        CancelGeometry();
        throw std::runtime_error("Geometry collection failed: " + gpu_error);
    }
    return FinishGeometry(world);
}

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
    const mtl::AutoreleasePool pool;
    if (Advancing) throw std::logic_error("Advance cannot be reentered from its observer");
    if (substeps == 0 || world.BodyCount() == 0) return {};
    struct Scope {
        bool &Active;
        ~Scope() { Active = false; }
    } scope{Advancing};
    Advancing = true;
    ColorCursor[SolveCursorCount] = settings.Iterations;
    ColorCursor[SolveCursorCount + 1] = settings.ColoringPasses;
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
    PrepareQueries(world, collider_features);
    AdvanceResult result;
    while (result.Steps < substeps && world.BodyCount() != 0) {
        const mtl::AutoreleasePool batch_pool;
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
        struct Recording {
            MTL4::ComputeCommandEncoder *Encoder;
            MTL4::CommandBuffer *Commands;
            void End() {
                if (!Encoder) return;
                std::exchange(Encoder, nullptr)->endEncoding();
                Commands->endCommandBuffer();
            }
            ~Recording() { End(); }
        } recording{encoder, Commands.get()};
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
                .QueuedQueries = bool(collider_features & MeshQueries) && QueuedQueriesFit(bodies),
            };

            Encode(encoder, {.Bodies = bodies, .Joints = joints, .Iterations = settings.Iterations, .Colors = colors, .ColoringPasses = settings.ColoringPasses, .ColliderFeatures = collider_features, .Parameter = index, .GpuColors = index != 0, .Snapshot = snapshot}, world);
        }
        recording.End();
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

void Solver::EncodeSolveCommands(MTL4::ComputeCommandEncoder *encoder, const Recording &recording, World &world) {
    if (!SolveCommands) {
        auto descriptor = mtl::Make<MTL::IndirectCommandBufferDescriptor>();
        descriptor->setCommandTypes(MTL::IndirectCommandTypeConcurrentDispatch);
        descriptor->setInheritBuffers(false);
        descriptor->setInheritPipelineState(false);
        descriptor->setMaxKernelBufferBindCount(27);
        SolveCommands = NS::TransferPtr(Context.Device->newIndirectCommandBuffer(descriptor.get(), BatchSteps * CommandIterationLimit * (MaxSupportedColors + 3), MTL::ResourceStorageModeShared));
        if (!SolveCommands) throw std::runtime_error("Cannot allocate solve commands");
        SolveCommandDataBuffer = {Context.Device.get(), BatchSteps};
        SolveCommandRanges = {Context.Device.get(), BatchSteps * 2};
        Residency->addAllocation(SolveCommands.get());
        Residency->addAllocation(SolveCommandDataBuffer.Handle.get());
        Residency->addAllocation(SolveCommandRanges.Handle.get());
        Residency->commit();
    }
    const uint32_t indices[]{
        shaders::PipelineIndices[Direct][SolvePass][recording.Bodies > WideSolveBodyLimit],
        shaders::PipelineIndices[Direct][PublishPass][0],
        shaders::PipelineIndices[Direct][DualPass][0],
        shaders::PipelineIndices[Direct][JointDualPass][0],
    };
    auto &data = SolveCommandDataBuffer[recording.Parameter];
    data = {};
    data.Commands = SolveCommands->gpuResourceID()._impl;
    for (uint32_t i = 0; i < std::size(indices); ++i) {
        auto *pipeline = Pipeline(indices[i]);
        if (i == 0 && pipeline->threadExecutionWidth() != SolveLanes) throw std::runtime_error("Primal command execution requires 32 lanes");
        data.Pipelines[i] = pipeline->gpuResourceID()._impl;
    }
    data.Buffers[0] = world.Poses.Address();
    data.Buffers[1] = world.InitialPoses.Address();
    data.Buffers[2] = world.InertialPoses.Address();
    data.Buffers[4] = world.Masses.Address();
    data.Buffers[5] = world.Contacts.Address();
    data.Buffers[7] = Params.Address() + recording.Parameter * sizeof(StepParams);
    data.Buffers[10] = world.Displacements.Address();
    data.Buffers[11] = world.Iterates.Address();
    data.Buffers[12] = world.Colors.Address();
    data.Buffers[13] = GeneralIslands.Address() + (IslandHeaderWords + (3 + IslandBodyLimit) * uint64_t(recording.Bodies)) * sizeof(uint32_t);
    data.Buffers[14] = ColorCursor.Address();
    data.Buffers[16] = world.Joints.Address();
    data.Buffers[17] = world.Incoming.Address();
    data.Buffers[18] = world.IncomingSlots.Address();
    data.Buffers[20] = world.JointIncidence.Address();
    data.Buffers[21] = world.Quiet.Address();
    data.Buffers[25] = GeneralIslands.Address();
    data.Buffers[26] = ColorScratch.Address();
    data.Groups = GeneralIslands.Address();
    data.Range = SolveCommandRanges.Address() + recording.Parameter * 2 * sizeof(uint32_t);
    data.FirstCommand = recording.Parameter * CommandIterationLimit * (MaxSupportedColors + 3);
    data.Iterations = recording.Iterations;
    data.Colors = recording.Colors;
    Table->setAddress(SolveCommandDataBuffer.Address() + recording.Parameter * sizeof(SolveCommandData), 0);
    encoder->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
    encoder->setComputePipelineState(Pipeline(shaders::EncodeSolveCommands));
    encoder->dispatchThreadgroups({recording.Iterations, 1, 1}, {SolveLanes, 1, 1});
    Table->setAddress(world.Poses.Address(), 0);
}

Solver::CollisionBuffers Solver::CollisionStorage(const World &world, bool sensor) const {
    const auto address = [](const auto &buffer) { return buffer.Handle ? buffer.Address() : uint64_t(0); };
    return {world.Bounds.Address(), world.BoundsReductions.Address(), world.BroadPhaseNodes.Address(),
        world.BroadPhaseKeys.Address(), world.BroadPhaseScratch.Address(), world.Iterates.Address(),
        world.PreviousVelocities.Address(), world.IncomingSlots.Address(), world.Materials.Address(),
        world.NextColors.Address(), world.ContactEventCounts.Address(),
        address(sensor ? SensorQueries : QueryScratch), address(QueryInputSnapshot), address(QueryInputs), ColorCursor.Address()};
}
void Solver::EncodeCollect(MTL4::ComputeCommandEncoder *encoder, const Recording &recording, World &world, bool sensor) {
    EncodeCollect(encoder, recording, CollisionStorage(world, sensor), Table.get(), Params[recording.Parameter], sensor);
}
void Solver::EncodeCollision(MTL4::ComputeCommandEncoder *encoder, const Recording &recording, World &world) {
    EncodeCollision(encoder, recording, CollisionStorage(world), Table.get(), Params[recording.Parameter]);
}

void Solver::EncodeCollect(MTL4::ComputeCommandEncoder *encoder, const Recording &recording, const CollisionBuffers &buffers, MTL4::ArgumentTable *table, const StepParams &params, bool sensor) {
    const uint32_t bodies = recording.Bodies;
    const bool queued_queries = params.QueuedQueries;
    const uint32_t collision_lanes = bodies <= RadixSimdWidth || (recording.ColliderFeatures & MeshQueries) ? CollisionLanes :
                                     bodies <= NativeCollectBodyLimit ? 32 : 1;
    Pass pass = sensor ? SensorPass : CollectPass;
    if (!sensor && !(recording.ColliderFeatures & MeshQueries))
        pass = bodies > NativeCollectBodyLimit && !(recording.ColliderFeatures & BoundedPlanes) ? ScalarCollectPass : NativeCollectPass;
    const uint32_t lanes = sensor ? CollisionLanes : collision_lanes;
    if (!queued_queries) {
        Dispatch(encoder, pass, bodies, lanes);
        return;
    }
    // Retain solid geometry while sensors use separate scratch storage.
    const uint64_t query_address = buffers.QueryScratch;
    table->setAddress(query_address, 11);
    if (!sensor) {
        table->setAddress(buffers.QueryInputSnapshot, 18);
        table->setAddress(buffers.QueryInputs, 9);
        Dispatch(encoder, CheckQueryInputsPass, 1, 1, query_address + offsetof(QueryArenaHeader, InputX));
        Dispatch(encoder, CheckQueryGeometryPass, 1, 1, query_address + offsetof(QueryArenaHeader, GeometryX));
        table->setAddress(buffers.IncomingSlots, 18);
        table->setAddress(buffers.PreviousVelocities, 9);
    }
    Dispatch(encoder, ResetQueryPass, bodies);
    Dispatch(encoder, pass, bodies, lanes, 0, Prepare);
    Dispatch(encoder, QueryPass, 1, 1, query_address);
    Dispatch(encoder, pass, bodies, 1, 0, Queued);
    // Recompute queries that exceed arena capacity.
    Dispatch(encoder, pass, bodies, lanes);
    table->setAddress(buffers.Iterates, 11);
}

void Solver::EncodeCollision(MTL4::ComputeCommandEncoder *encoder, const Recording &recording, const CollisionBuffers &buffers, MTL4::ArgumentTable *table, const StepParams &params) {
    const uint32_t bodies = recording.Bodies;
    const uint32_t bounds_lanes = bodies <= RadixSimdWidth ? RadixSimdWidth : 1;
    const bool queued_queries = params.QueuedQueries;
    if (queued_queries) {
        table->setAddress(buffers.QueryScratch, 11);
        table->setAddress(buffers.QueryInputs, 9);
        table->setAddress(buffers.QueryInputSnapshot, 18);
        Dispatch(encoder, ResetQueryReusePass, 1);
    }
    Dispatch(encoder, params.GeometryOnly ? GeometryBoundsPass : BoundsPass, bodies, bounds_lanes);
    if (queued_queries) {
        table->setAddress(buffers.Iterates, 11);
        table->setAddress(buffers.PreviousVelocities, 9);
        table->setAddress(buffers.IncomingSlots, 18);
    }
    if (bodies <= RadixBlockSize) {
        table->setAddress(buffers.BroadPhaseKeys, 11);
        table->setAddress(buffers.BroadPhaseNodes, 13);
        Dispatch(encoder, SmallBroadPhasePass, std::bit_ceil(std::max(RadixSimdWidth, bodies)));
    } else if (recording.Parameter && Params[recording.Parameter - 1].BodyCount == bodies) {
        // Rebuild at each submission boundary; within it the same body slots retain a valid topology.
        table->setAddress(buffers.BroadPhaseNodes, 13);
        Dispatch(encoder, RefreshTreePass, bodies);
        Dispatch(encoder, RefitTreePass, bodies);
    } else {
        const uint32_t blocks = RadixBlocks(bodies);
        table->setAddress(buffers.BoundsReductions, 13);
        Dispatch(encoder, ReduceBoundsPass, blocks * RadixBlockSize);
        Dispatch(encoder, ReduceScenePass, RadixBlockSize);
        table->setAddress(buffers.BroadPhaseKeys, 11);
        table->setAddress(buffers.BroadPhaseNodes, 10);
        Dispatch(encoder, MortonPass, bodies);
        table->setAddress(buffers.BroadPhaseScratch, 9);
        for (uint32_t digit = 0; digit < 4; ++digit) {
            const uint64_t input = buffers.BroadPhaseKeys + (digit % 2) * bodies * sizeof(MortonKey);
            const uint64_t output = buffers.BroadPhaseKeys + ((digit + 1) % 2) * bodies * sizeof(MortonKey);
            table->setAddress(input, 11);
            table->setAddress(output, 13);
            table->setAddress(buffers.ColorCursor + digit * sizeof(uint32_t), 25);
            Dispatch(encoder, RadixHistogramPass, blocks * RadixBlockSize);
            Dispatch(encoder, RadixOffsetsPass, RadixBlockSize);
            Dispatch(encoder, RadixScatterPass, bodies);
        }
        table->setAddress(buffers.BroadPhaseKeys, 11);
        table->setAddress(buffers.BroadPhaseNodes, 13);
        Dispatch(encoder, BuildTreePass, bodies);
        Dispatch(encoder, RefitTreePass, bodies);
    }
    table->setAddress(buffers.BroadPhaseNodes, BroadPhaseOrCursorAt);
    table->setAddress(buffers.PreviousVelocities, 9);
    table->setAddress(buffers.Materials, 10);
    table->setAddress(buffers.Iterates, 11);
    table->setAddress(buffers.NextColors, 13);
    table->setAddress(buffers.ContactEventCounts, 25);
    // Measure collision geometry before warm starting so C0 and Jacobians use the initial pose.
    // Shared lanes increase parallelism for mesh queries and small worlds.
    EncodeCollect(encoder, recording, buffers, table, params, false);
}

void Solver::Encode(MTL4::ComputeCommandEncoder *encoder, const Recording &recording, World &world) {
    const uint32_t bodies = recording.Bodies, joints = recording.Joints;
    const bool compact = bodies > SolveLanes;
    const bool general_islands = compact && recording.Iterations;
    const bool complete = bodies <= FullStepBodyLimit && !joints && (recording.ColliderFeatures & FullStepGeometry) && !(recording.ColliderFeatures & BoundedPlanes);
    if (!complete) Bind(world, recording.Parameter);
    else if (!FollowerRanges.empty()) {
        Table->setAddress(world.Poses.Address(), 0);
        Table->setAddress(world.Velocities.Address(), 3);
    }
    const uint32_t slots = bodies * ContactsPerBody;
    for (const auto range : FollowerRanges) {
        Table->setAddress(Followers.Address() + range.First * sizeof(SensorFollower), 26);
        Dispatch(encoder, FollowPass, range.Count);
    }
    if (complete) {
        const uint64_t output = Outputs.Address() + recording.Parameter * Layout.Stride;
        FullData[recording.Parameter] = {
            .poses = world.Poses.Address(),
            .initial = world.InitialPoses.Address(),
            .inertial = world.InertialPoses.Address(),
            .velocities = world.Velocities.Address(),
            .masses = world.Masses.Address(),
            .contacts = world.Contacts.Address(),
            .body_shapes = world.BodyShapes.Address(),
            .shapes = world.Shapes.Address(),
            .previous = world.PreviousVelocities.Address(),
            .displacements = world.Displacements.Address(),
            .iterates = world.Iterates.Address(),
            .colors = world.Colors.Address(),
            .next = world.NextColors.Address(),
            .bounds = world.Bounds.Address(),
            .nodes = world.BroadPhaseNodes.Address(),
            .materials = world.Materials.Address(),
            .compound_children = world.CompoundChildren.Address(),
            .joints = world.Joints.Address(),
            .incoming = world.Incoming.Address(),
            .incoming_slots = world.IncomingSlots.Address(),
            .filters = world.Filters.Address(),
            .jointed_to = world.Jointed.Address(),
            .joint_incidence = world.JointIncidence.Address(),
            .quiet = world.Quiet.Address(),
            .rest = world.RestPoses.Address(),
            .next_quiet = world.NextQuiet.Address(),
            .contact_events = world.ContactEvents.Address(),
            .contact_event_counts = world.ContactEventCounts.Address(),
            .contact_refusals = world.ContactRefusals.Address(),
            .hull_vertices = world.ShapeVertices.Address(),
            .mesh_triangles = world.Triangles.Address(),
            .bvh_nodes = world.BvhNodes.Address(),
            .hull_faces = world.HullFaces.Address(),
            .sensors = world.SensorContacts.Address(),
            .sensor_refusals = world.SensorRefusals.Address(),
            .budgets = ColorCursor.Address() + SolveCursorCount * sizeof(uint32_t),
            .islands = SmallIslands.Address(),
            .params = Params.Address() + recording.Parameter * sizeof(StepParams),
            .out_contacts = output + Layout.Contacts,
            .out_initial = output + Layout.Initial,
            .out_poses = output + Layout.Poses,
            .out_velocities = output + Layout.Velocities,
            .out_sensors = output + Layout.Sensors,
            .out_counts = output + Layout.Counts,
            .completion = output + Layout.Completion,
            .out_removed = output + Layout.RemovedContacts,
            .snapshot = recording.Snapshot,
            .gpu_colors = recording.GpuColors,
            .output_poses = OutputFlags[0].Poses,
            .output_sensors = OutputFlags[0].Sensors,
            .restitution = bool(recording.ColliderFeatures & RestitutionMaterials),
        };
        Table->setAddress(FullData.Address() + recording.Parameter * sizeof(FullStepData), 0);
        encoder->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
        encoder->setComputePipelineState(Pipeline(shaders::FullStep));
        encoder->dispatchThreadgroups({1, 1, 1}, {std::min(bodies, SmallSolveWaves) * SolveLanes, 1, 1});
        return;
    }
    const uint64_t groups = ColorGroups.Address() + recording.Parameter * MaxSupportedColors * 3 * sizeof(uint32_t);
    if (recording.GpuColors) {
        Table->setAddress(groups, 26);
        if (bodies > SerialColorBudgetBodyLimit) {
            // Each batch slot is independent and all CPU initialization precedes submission.
            ColorGroups[recording.Parameter * MaxSupportedColors * 3] = 1;
            Dispatch(encoder, ReduceStepColorsPass, bodies);
            Dispatch(encoder, FinishStepColorsPass, MaxSupportedColors);
        } else Dispatch(encoder, PrepareColorsPass, MaxSupportedColors);
    }
    Table->setAddress(world.ContactRefusals.Address(), 26);

    // Each stage and color reads one immutable cursor slot.
    const auto sweep = [&](bool stabilize) {
        for (uint32_t color = 0; color < recording.Colors; ++color) {
            Table->setAddress(ColorCursor.Address() + (color + uint32_t(stabilize) * MaxSupportedColors) * sizeof(uint32_t), BroadPhaseOrCursorAt);
            const uint64_t work_groups = general_islands && !stabilize ? GeneralIslands.Address() + (IslandColorsAt + color * 3) * sizeof(uint32_t) :
                (compact || recording.GpuColors)                       ? groups + color * 3 * sizeof(uint32_t) :
                                                                         0;
            Dispatch(encoder, SolvePass, bodies, 1, work_groups);
        }
        Dispatch(encoder, stabilize ? FinishPosesPass : PublishPass, bodies, 1, general_islands && !stabilize ? GeneralIslands.Address() + IslandPublishAt * sizeof(uint32_t) : 0);
    };

    EncodeCollision(encoder, recording, world);
    if (bodies <= SolveLanes) {
        Table->setAddress(world.JointIncidence.Address(), 20);
        Table->setAddress(world.Displacements.Address(), 10);
        Table->setAddress(SmallIslands.Address(), 26);
        Table->setAddress(ColorCursor.Address() + SolveCursorCount * sizeof(uint32_t), BroadPhaseOrCursorAt);
        Dispatch(encoder, PrepareSmallWorldPass, SolveLanes);
    } else {
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
        Table->setAddress(ColorScratch.Address(), 25);
        Dispatch(encoder, FinishIslandsPass, bodies);
        Table->setAddress(GeneralIslands.Address(), 25);
        Table->setAddress(ColorCursor.Address() + SolveCursorCount * sizeof(uint32_t), BroadPhaseOrCursorAt);
        Dispatch(encoder, SolveIslandsPass, bodies, 1, GeneralIslands.Address());
        Table->setAddress(GeneralIslands.Address() + (IslandHeaderWords + (3 + IslandBodyLimit) * uint64_t(bodies)) * sizeof(uint32_t), 13);
        Table->setAddress(ColorScratch.Address(), 26);
    }
    if (bodies <= SolveLanes) {
        if (recording.Iterations) {
            Table->setAddress(ColorCursor.Address() + SolveCursorCount * sizeof(uint32_t), BroadPhaseOrCursorAt);
            Table->setAddress(SmallIslands.Address(), 26);
            Dispatch(encoder, SolveIslandsPass, bodies);
            Table->setAddress(groups, 26);
        }
    } else if (general_islands && recording.Iterations <= CommandIterationLimit) {
        EncodeSolveCommands(encoder, recording, world);
        encoder->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
        encoder->executeCommandsInBuffer(SolveCommands.get(), SolveCommandRanges.Address() + recording.Parameter * 2 * sizeof(uint32_t));
    } else {
        for (uint32_t iteration = 0; iteration < recording.Iterations; ++iteration) {
            sweep(false);
            Dispatch(encoder, DualPass, slots, 1, GeneralIslands.Address() + IslandContactDualAt * sizeof(uint32_t));
            if (joints > 0) Dispatch(encoder, JointDualPass, joints, 1, GeneralIslands.Address() + IslandJointDualAt * sizeof(uint32_t));
        }
    }
    if (general_islands) Table->setAddress(world.NextColors.Address(), 13);
    Table->setAddress(world.ContactEventCounts.Address(), 25);
    // Measure velocity before stabilization to exclude penetration correction.
    Dispatch(encoder, FinalizePass, bodies);
    if (recording.ColliderFeatures & RestitutionMaterials)
        for (uint32_t pass = 0; pass < RestitutionPasses; ++pass) {
            Dispatch(encoder, RestitutionPass, slots);
            Dispatch(encoder, ApplyRestitutionPass, bodies);
        }
    sweep(true);
    // Publish quiet counts before neighboring bodies read them.
    // Capture performs this publication when the sensor pass is skipped.
    if (!recording.Snapshot || OutputFlags[0].Sensors) Dispatch(encoder, FinishWakingPass, bodies);

    if (OutputFlags[0].Sensors) {
        Table->setAddress(world.Jointed.Address(), 20);
        Table->setAddress(world.SensorContacts.Address(), 5);
        Table->setAddress(world.SensorRefusals.Address(), 26);
        Table->setAddress(world.Materials.Address(), 10);
        Table->setAddress(world.Bounds.Address(), BroadPhaseOrCursorAt);
        // Sensors query the final poses, after solving and stabilization.
        Dispatch(encoder, SensorBoundsPass, bodies, bodies <= RadixSimdWidth ? RadixSimdWidth : 1);
        Table->setAddress(world.BroadPhaseNodes.Address(), 13);
        Dispatch(encoder, RefreshTreePass, bodies);
        if (bodies > RadixSimdWidth) Dispatch(encoder, RefitTreePass, bodies);
        Table->setAddress(world.BroadPhaseNodes.Address(), BroadPhaseOrCursorAt);
        Table->setAddress(world.NextColors.Address(), 13);
        EncodeCollect(encoder, recording, world, true);
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
        if (!OutputFlags[0].Sensors) {
            Table->setAddress(world.Quiet.Address(), 22);
            Table->setAddress(world.Joints.Address(), 27);
            Table->setAddress(world.Incoming.Address(), 28);
            Table->setAddress(world.IncomingSlots.Address(), 29);
            Table->setAddress(world.JointIncidence.Address(), 30);
        }
        Dispatch(encoder, CapturePass, bodies);
    }
}

} // namespace rbp
