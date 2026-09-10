#include "Solver.h"

#include "GpuSource.h"

#include <algorithm>
#include <bit>
#include <string>
#include <string_view>

namespace rbp {

namespace {
// Collision and primal passes use this binding at separate times.
constexpr uint32_t BroadPhaseOrCursorAt = 14;
constexpr uint32_t BindingCount = 31;

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
    return std::clamp(used + 1, 1u, std::min(settings.MaxColors, MaxSupportedColors));
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
    {"CollectSensorContacts", "#define BOUNDED_PLANES 0\n#define MESH_PAIRS 0\n#define SENSOR_PASS 1\n#define CollectContacts CollectSensorContacts"},
    {"CollectContacts", "#define MESH_PAIRS 0"},
    {"CollectSensorContacts", "#define MESH_PAIRS 0\n#define SENSOR_PASS 1\n#define CollectContacts CollectSensorContacts"},
    {"CollectContacts", "#define BOUNDED_PLANES 0"},
    {"CollectSensorContacts", "#define BOUNDED_PLANES 0\n#define SENSOR_PASS 1\n#define CollectContacts CollectSensorContacts"},
    {"CollectContacts"},
    {"CollectSensorContacts", "#define SENSOR_PASS 1\n#define CollectContacts CollectSensorContacts"},
};
} // namespace

Solver::Solver(const mtl::Context &context) : Context(context) {
    static_assert(std::size(Kernels) == PassCount, "one kernel per pass, in the enum's order");
    // Additional collider specializations compile lazily when a world first uses them.
    for (uint32_t pass = 0; pass < BoundedCollectPass; ++pass) {
        std::string prefix{Kernels[pass].Prefix};
        if (pass == CollectPass || pass == SensorPass) prefix += "\n#define BROAD_PHASE_MODE 1";
        Pipelines[pass][0] = context.Pipeline(pass < BoundsPass ? gpu::BroadPhaseSource : gpu::SolveSource, Kernels[pass].Name, prefix);
    }

    NS::Error *error{};
    auto descriptor = mtl::Make<MTL4::ArgumentTableDescriptor>();
    descriptor->setMaxBufferBindCount(BindingCount);
    Table = NS::TransferPtr(context.Device->newArgumentTable(descriptor.get(), &error));
    Allocator = NS::TransferPtr(context.Device->newCommandAllocator());
    Commands = NS::TransferPtr(context.Device->newCommandBuffer());
    Done = NS::TransferPtr(context.Device->newSharedEvent());

    Params = {context.Device.get(), 1};
    // One slot per color, holding its own index, so a color pass is selected by the slot the cursor binding points at.
    // No counting kernel is dispatched between colors.
    ColorCursor = {context.Device.get(), MaxSupportedColors};
    for (uint32_t color = 0; color < MaxSupportedColors; ++color) ColorCursor[color] = color;
    Residency = NS::TransferPtr(context.Device->newResidencySet(mtl::Make<MTL::ResidencySetDescriptor>().get(), &error));
    Residency->addAllocation(Params.Handle.get());
    Residency->addAllocation(ColorCursor.Handle.get());
    Residency->commit();
    Residency->requestResidency();
    context.Queue->addResidencySet(Residency.get());
}

Solver::~Solver() {
    Context.Queue->removeResidencySet(Residency.get());
    mtl::Drain(Context.Queue.get()); // see mtl::Drain
}

void Solver::Dispatch(MTL4::ComputeCommandEncoder *encoder, Pass pass, uint32_t threads, uint32_t lanes) {
    const bool collector = pass == CollectPass || pass == SensorPass || pass >= BoundedCollectPass;
    const bool primal = pass == SolvePass || pass == StabilizePass;
    const bool bounds = pass == BoundsPass || pass == SensorBoundsPass;
    const uint32_t variant = collector ? uint32_t(threads > RadixSimdWidth) + 2 * uint32_t(lanes > 1) :
        bounds                         ? uint32_t(lanes > 1) :
                                         uint32_t(primal && threads > SolveLanes);
    if (!Pipelines[pass][variant]) {
        std::string prefix{Kernels[pass].Prefix};
        if (collector) {
            prefix += (variant & 1) ? "\n#define BROAD_PHASE_MODE 2" : "\n#define BROAD_PHASE_MODE 1";
            prefix += "\n#define COLLECT_LANES " + std::to_string(lanes);
        }
        if (primal) prefix += "\n#define SOLVE_BODIES_PER_GROUP " + std::to_string(variant ? SolveBodiesPerGroup : 1);
        if (bounds) prefix += "\n#define BOUNDS_LANES " + std::to_string(lanes);
        Pipelines[pass][variant] = Context.Pipeline(pass < BoundsPass ? gpu::BroadPhaseSource : gpu::SolveSource, Kernels[pass].Name, prefix);
    }
    MTL::ComputePipelineState *pipeline = Pipelines[pass][variant].get();
    // Every pass reads the previous pass's writes, so every dispatch takes a barrier.
    encoder->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
    encoder->setComputePipelineState(pipeline);
    const bool per_body = collector || pass == SolvePass || pass == StabilizePass;
    const bool radix = pass == RadixHistogramPass || pass == RadixOffsetsPass || pass == ReduceBoundsPass || pass == ReduceScenePass;
    const bool scan = pass == ScanIncomingPass || pass == ScanIncomingBlocksPass;
    const bool contact = pass == DualPass || pass == RestitutionPass;
    if ((radix || scan || pass == SmallBroadPhasePass || ((collector || bounds) && lanes > 1)) && pipeline->threadExecutionWidth() != RadixSimdWidth)
        throw std::runtime_error("GPU scans require 32-lane SIMD groups");
    const uint32_t limit = scan ? RadixBlockSize : per_body ? 8u :
        contact                                             ? 64u :
                                                              pipeline->maxTotalThreadsPerThreadgroup();
    const auto group = pass == SmallBroadPhasePass ? threads : radix ? RadixBlockSize :
                                                                       std::min(threads, limit);
    if (primal) {
        if (pipeline->threadExecutionWidth() != SolveLanes) throw std::runtime_error("The body solve requires 32-lane SIMD groups");
        const uint32_t bodies_per_group = variant ? SolveBodiesPerGroup : 1;
        encoder->dispatchThreadgroups({(threads + bodies_per_group - 1) / bodies_per_group, 1, 1}, {SolveLanes, 1, 1});
    } else if (pass == JointDualPass) encoder->dispatchThreadgroups({threads, 1, 1}, {32, 1, 1});
    else if ((collector || bounds) && lanes > 1) encoder->dispatchThreadgroups({threads, 1, 1}, {lanes, 1, 1});
    else encoder->dispatchThreads({threads, 1, 1}, {group, 1, 1});
}

void Solver::Step(World &world, const StepSettings &settings) {
    const uint32_t bodies = world.BodyCount();
    if (bodies == 0) return;
    world.RefreshFilters();
    const uint32_t joints = world.JointCount();
    // A scene using two colors would otherwise spend most of a step's dispatches on six empty color passes.
    const uint32_t colors = ColorsNeeded(world, settings);
    const uint32_t collider_features = ColliderFeatures(world);

    Params[0] = {
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
    // Every buffer the kernels declare, at its declared slot.
    // This list is the argument table's layout, not a copy kept alongside one.
    const uint64_t bindings[]{
        world.Poses.Address(), // 0
        world.InitialPoses.Address(), // 1
        world.InertialPoses.Address(), // 2
        world.Velocities.Address(), // 3
        world.Masses.Address(), // 4
        world.Contacts.Address(), // 5
        world.BodyShapes.Address(), // 6
        Params.Address(), // 7
        world.Shapes.Address(), // 8
        world.PreviousVelocities.Address(), // 9
        world.Materials.Address(), // 10
        world.Iterates.Address(), // 11
        world.Colors.Address(), // 12
        world.NextColors.Address(), // 13
        world.Bounds.Address(), // 14
        world.CompoundChildren.Address(), // 15
        world.Joints.Address(), // 16
        world.Incoming.Address(), // 17
        world.IncomingSlots.Address(), // 18
        world.Filters.Address(), // 19
        world.Jointed.Address(), // 20
        world.Quiet.Address(), // 21
        world.RestPoses.Address(), // 22
        world.NextQuiet.Address(), // 23
        world.ContactEvents.Address(), // 24
        world.ContactEventCounts.Address(), // 25
        world.ContactRefusals.Address(), // 26
        world.ShapeVertices.Address(), // 27
        world.Triangles.Address(), // 28
        world.BvhNodes.Address(), // 29
        world.HullFaces.Address(), // 30
    };
    static_assert(sizeof(bindings) / sizeof(bindings[0]) == BindingCount, "one address per slot the table holds");
    for (uint32_t slot = 0; slot < BindingCount; ++slot) Table->setAddress(bindings[slot], slot);

    Encode({.Bodies = bodies, .Joints = joints, .Iterations = settings.Iterations, .Colors = colors, .ColoringPasses = settings.ColoringPasses, .ColliderFeatures = collider_features}, world);

    // Queue signalling publishes the GPU's writes to the host safely, per Architecture.md.
    const MTL4::CommandBuffer *list[]{Commands.get()};
    Context.Queue->commit(list, 1);
    Context.Queue->signalEvent(Done.get(), ++Signal);
    while (!Done->waitUntilSignaledValue(Signal, 1000)) {}
    // The GPU is done with the world, so a removal deferred during the step applies now. See World::OnStepped.
    const auto root = world.BroadPhaseNodes[BroadPhaseRoot(bodies)];
    if (root.Ready != (bodies <= RadixSimdWidth ? 0u : 2u) || root.Errors != 0) throw std::runtime_error("GPU broad phase did not complete");
    world.OnStepped(settings.DeltaTime);
}

void Solver::Encode(const Recording &recording, World &world) {
    const uint32_t bodies = recording.Bodies, joints = recording.Joints;
    const uint32_t slots = bodies * ContactsPerBody;
    // Recycling the previous recording's memory is safe because every step waits for its own completion.
    Allocator->reset();
    Commands->beginCommandBuffer(Allocator.get());
    auto *encoder = Commands->computeCommandEncoder();
    encoder->setArgumentTable(Table.get());

    // Each color reads one immutable cursor slot.
    const auto sweep = [&](Pass primal) {
        for (uint32_t color = 0; color < recording.Colors; ++color) {
            Table->setAddress(ColorCursor.Address() + color * sizeof(uint32_t), BroadPhaseOrCursorAt);
            Dispatch(encoder, primal, bodies);
        }
        Dispatch(encoder, PublishPass, bodies);
    };

    const uint32_t bounds_lanes = bodies <= RadixSimdWidth ? RadixSimdWidth : 1;
    Dispatch(encoder, BoundsPass, bodies, bounds_lanes);
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
    Dispatch(encoder, collectors[geometry_features], bodies, collision_lanes);
    // Gather each body's contacts-as-B into a contiguous run, so the passes below do not scan the whole pool.
    Table->setAddress(world.BroadPhaseScratch.Address(), BroadPhaseOrCursorAt);
    Dispatch(encoder, ScanIncomingPass, bodies);
    if (bodies > RadixBlockSize) {
        Dispatch(encoder, ScanIncomingBlocksPass, RadixBlockSize);
        Dispatch(encoder, OffsetIncomingPass, bodies);
    }
    Dispatch(encoder, FillIncomingPass, bodies);
    Dispatch(encoder, SortIncomingPass, bodies);
    if (joints > 0) Dispatch(encoder, PrepareJointsPass, joints);
    Table->setAddress(world.Displacements.Address(), 10);
    Dispatch(encoder, WarmStartPass, bodies);
    for (uint32_t pass = 0; pass < recording.ColoringPasses; ++pass) {
        Dispatch(encoder, ColorPass, bodies);
        Dispatch(encoder, PublishColorPass, bodies);
    }
    for (uint32_t iteration = 0; iteration < recording.Iterations; ++iteration) {
        sweep(SolvePass);
        Dispatch(encoder, DualPass, slots);
        if (joints > 0) Dispatch(encoder, JointDualPass, joints);
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

    bool sensors = !world.Overlaps().empty();
    for (Index body = 0; body < bodies && !sensors; ++body) sensors = world.Alive(body) && world.Filters[body].Sensor;
    if (sensors) {
        world.EnsureSensorBuffers();
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
        Dispatch(encoder, sensor_collectors[geometry_features], bodies, collision_lanes);
    }

    encoder->endEncoding();
    Commands->endCommandBuffer();
}

} // namespace rbp
