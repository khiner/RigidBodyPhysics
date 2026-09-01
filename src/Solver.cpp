#include "Solver.h"

#include "GpuSource.h"

#include <algorithm>
#include <string_view>

namespace rbp {

namespace {
// The one argument-table slot a step rebinds, and the total slot count.
// Every other slot is bound once from the list in Step, the only place the slot numbers appear outside Solve.metal's [[buffer(n)]] indices.
constexpr uint32_t CursorAt = 14;
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

// The kernel each pass runs, in the order of Solver::Pass.
// A prefix supplies a #define when one kernel text is compiled more than one way.
constexpr struct {
    const char *Name;
    std::string_view Prefix;
} Kernels[]{
    {"Integrate"},
    {"CollectContacts"},
    {"CountIncoming"},
    {"ScanIncoming"},
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
};
} // namespace

Solver::Solver(const mtl::Context &context) : Context(context) {
    static_assert(std::size(Kernels) == PassCount, "one kernel per pass, in the enum's order");
    for (uint32_t pass = 0; pass < PassCount; ++pass)
        Pipelines[pass] = context.Pipeline(gpu::SolveSource, Kernels[pass].Name, Kernels[pass].Prefix);

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

void Solver::Dispatch(MTL4::ComputeCommandEncoder *encoder, Pass pass, uint32_t threads) const {
    MTL::ComputePipelineState *pipeline = Pipelines[pass].get();
    // Every pass reads the previous pass's writes, so every dispatch takes a barrier.
    encoder->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
    encoder->setComputePipelineState(pipeline);
    const auto group = std::min<uint32_t>(threads, pipeline->maxTotalThreadsPerThreadgroup());
    encoder->dispatchThreads({threads, 1, 1}, {group, 1, 1});
}

void Solver::Step(World &world, const StepSettings &settings) {
    const uint32_t bodies = world.BodyCount();
    if (bodies == 0) return;
    const uint32_t joints = world.JointCount();
    // A scene using two colors would otherwise spend most of a step's dispatches on six empty color passes.
    const uint32_t colors = ColorsNeeded(world, settings);

    Params[0] = {
        .Gravity = settings.Gravity,
        .DeltaTime = settings.DeltaTime,
        .Beta = settings.Beta,
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
        world.Frictions.Address(), // 10
        world.SolvedPoses.Address(), // 11
        world.Colors.Address(), // 12
        world.NextColors.Address(), // 13
        ColorCursor.Address(), // 14
        world.Restitutions.Address(), // 15
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

    Encode({.Bodies = bodies, .Joints = joints, .Iterations = settings.Iterations, .Colors = colors, .ColoringPasses = settings.ColoringPasses});

    // Queue signalling publishes the GPU's writes to the host safely, per Architecture.md.
    const MTL4::CommandBuffer *list[]{Commands.get()};
    Context.Queue->commit(list, 1);
    Context.Queue->signalEvent(Done.get(), ++Signal);
    while (!Done->waitUntilSignaledValue(Signal, 1000)) {}
    // The GPU is done with the world, so a removal deferred during the step applies now. See World::OnStepped.
    world.OnStepped();
}

void Solver::Encode(const Recording &recording) {
    const uint32_t bodies = recording.Bodies, joints = recording.Joints;
    const uint32_t slots = bodies * ContactsPerBody;
    // Recycling the previous recording's memory is safe because every step waits for its own completion.
    Allocator->reset();
    Commands->beginCommandBuffer(Allocator.get());
    auto *encoder = Commands->computeCommandEncoder();
    encoder->setArgumentTable(Table.get());

    // One pass per color, so the cursor returns to color zero at the end of every sweep with no reset.
    const auto sweep = [&](Pass primal) {
        for (uint32_t color = 0; color < recording.Colors; ++color) {
            Table->setAddress(ColorCursor.Address() + color * sizeof(uint32_t), CursorAt);
            Dispatch(encoder, primal, bodies);
            Dispatch(encoder, PublishPass, bodies);
        }
    };

    Dispatch(encoder, IntegratePass, bodies);
    // Collision, and with it every C0, anchor and Jacobian, is taken at the pose the step began from, before WarmStart moves the body to its starting guess.
    // This is the reference's order, and the only one that expands the Taylor series about the pose the constraint was measured at.
    Dispatch(encoder, CollectPass, bodies);
    // Gather each body's contacts-as-B into a contiguous run, so the passes below do not scan the whole pool.
    Dispatch(encoder, CountIncomingPass, slots);
    Dispatch(encoder, ScanIncomingPass, 1);
    Dispatch(encoder, FillIncomingPass, slots);
    Dispatch(encoder, SortIncomingPass, bodies);
    if (joints > 0) Dispatch(encoder, PrepareJointsPass, joints);
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

    encoder->endEncoding();
    Commands->endCommandBuffer();
}

} // namespace rbp
