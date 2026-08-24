#include "Solver.h"

#include "GpuSource.h"

#include <algorithm>

namespace {
// The argument table slots the kernels declare. One table, bound once, since every pass reads from
// the same world.
enum Binding : uint32_t { PosesAt,
                          InitialAt,
                          InertialAt,
                          VelocitiesAt,
                          MassesAt,
                          ContactsAt,
                          BodyShapesAt,
                          ParamsAt,
                          ShapesAt,
                          PreviousVelocitiesAt,
                          FrictionsAt,
                          SolvedAt,
                          ColorsAt,
                          NextColorsAt,
                          CursorAt,
                          RestitutionsAt,
                          JointsAt,
                          IncomingAt,
                          IncomingSlotsAt,
                          FiltersAt,
                          JointedAt,
                          QuietAt,
                          RestPosesAt,
                          NextQuietAt,
                          ContactEventsAt,
                          ContactEventCountsAt,
                          ContactRefusalsAt,
                          ShapeVerticesAt,
                          TrianglesAt,
                          BvhNodesAt,
                          HullFacesAt,
                          BindingCount };

// How many times the restitution pass sweeps its contacts. It is Jacobi - every contact computes an
// impulse from one velocity snapshot and a gather applies them - so a manifold's points each answer for
// the whole approach on the first pass, and the ones after share it out between them.
//
// Four because that is where it stops moving. A box dropped flat comes out the same to five figures at
// one pass and at sixteen, its four points being symmetric, and dropped tilted onto a corner it leaves
// 0.114 of its arrival speed after one pass, 0.0972 after two and 0.0976 after four, with eight and
// sixteen bit-identical to four.
constexpr uint32_t RestitutionPasses = 4;

// How many colour passes a sweep needs. Colouring is incremental, so what it settled on last step is
// what it will want this one, and a spare on top is what it grows into as the scene closes up - one a
// step, which is as fast as an incremental colouring moves anyway.
//
// Read off the world rather than remembered here, so a step stays a pure function of the world it is
// handed and two runs of a scene still agree to the bit.
uint32_t ColorsNeeded(const World &world, const StepSettings &settings) {
    uint32_t used = 1;
    for (uint32_t body = 0; body < world.BodyCount(); ++body)
        if (world.Masses[body].InvMass > 0) used = std::max(used, ColorOf(world.Colors[body]) + 1);
    return std::clamp(used + 1, 1u, std::min(settings.MaxColors, MaxSupportedColors));
}
} // namespace

Solver::Solver(const mtl::Context &context) : Context(context) {
    IntegratePipeline = context.Pipeline(gpu::SolveSource, "Integrate");
    CollectPipeline = context.Pipeline(gpu::SolveSource, "CollectContacts");
    WarmStartPipeline = context.Pipeline(gpu::SolveSource, "WarmStart");
    PrepareJointsPipeline = context.Pipeline(gpu::SolveSource, "PrepareJoints");
    SpreadWakingPipeline = context.Pipeline(gpu::SolveSource, "SpreadWaking");
    PublishWakingPipeline = context.Pipeline(gpu::SolveSource, "PublishWaking");
    CountIncomingPipeline = context.Pipeline(gpu::SolveSource, "CountIncoming");
    ScanIncomingPipeline = context.Pipeline(gpu::SolveSource, "ScanIncoming");
    FillIncomingPipeline = context.Pipeline(gpu::SolveSource, "FillIncoming");
    SortIncomingPipeline = context.Pipeline(gpu::SolveSource, "SortIncoming");
    JointDualPipeline = context.Pipeline(gpu::SolveSource, "UpdateJointDuals");
    SolvePipeline = context.Pipeline(gpu::SolveSource, "SolveBodies");
    DualPipeline = context.Pipeline(gpu::SolveSource, "UpdateDuals");
    FinalizePipeline = context.Pipeline(gpu::SolveSource, "Finalize");
    RestitutionPipeline = context.Pipeline(gpu::SolveSource, "Restitution");
    ApplyRestitutionPipeline = context.Pipeline(gpu::SolveSource, "ApplyRestitution");
    CountQuietPipeline = context.Pipeline(gpu::SolveSource, "CountQuiet");
    // The same text again, with the C0 term kept, for the stabilization pass at the end of a step.
    StabilizePipeline = context.Pipeline(gpu::SolveSource, "SolveBodies", "#define STABILIZE 1");
    PublishPipeline = context.Pipeline(gpu::SolveSource, "PublishPoses");
    ColorPipeline = context.Pipeline(gpu::SolveSource, "UpdateColors");
    PublishColorPipeline = context.Pipeline(gpu::SolveSource, "PublishColors");

    NS::Error *error{};
    auto descriptor = mtl::Make<MTL4::ArgumentTableDescriptor>();
    descriptor->setMaxBufferBindCount(BindingCount);
    Table = NS::TransferPtr(context.Device->newArgumentTable(descriptor.get(), &error));
    Allocator = NS::TransferPtr(context.Device->newCommandAllocator());
    Commands = NS::TransferPtr(context.Device->newCommandBuffer());
    Done = NS::TransferPtr(context.Device->newSharedEvent());

    Params = {context.Device.get(), 1};
    // One slot per colour, holding its own index, so a colour pass is selected by which slot the
    // cursor binding points at rather than by a kernel dispatched between colours to count.
    ColorCursor = {context.Device.get(), MaxSupportedColors};
    for (uint32_t color = 0; color < MaxSupportedColors; ++color) ColorCursor[color] = color;
    Residency = NS::TransferPtr(context.Device->newResidencySet(mtl::Make<MTL::ResidencySetDescriptor>().get(), &error));
    Residency->addAllocation(Params.Handle.get());
    Residency->addAllocation(ColorCursor.Handle.get());
    Residency->commit();
    Residency->requestResidency();
    context.Queue->addResidencySet(Residency.get());
}

Solver::~Solver() { Context.Queue->removeResidencySet(Residency.get()); }

void Solver::Dispatch(MTL4::ComputeCommandEncoder *encoder, MTL::ComputePipelineState *pipeline, uint32_t threads) const {
    // Every pass reads what the one before it wrote, so every one of them earns its barrier.
    encoder->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
    encoder->setComputePipelineState(pipeline);
    const auto group = std::min<uint32_t>(threads, pipeline->maxTotalThreadsPerThreadgroup());
    encoder->dispatchThreads({threads, 1, 1}, {group, 1, 1});
}

void Solver::Step(World &world, const StepSettings &settings) {
    const uint32_t bodies = world.BodyCount();
    if (bodies == 0) return;
    const uint32_t joints = world.JointCount();
    // Eight colour passes for a scene that only ever uses two is most of a step's dispatches spent on
    // nothing. A stack is a chain, so it needs two.
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
    Table->setAddress(world.Poses.Address(), PosesAt);
    Table->setAddress(world.InitialPoses.Address(), InitialAt);
    Table->setAddress(world.InertialPoses.Address(), InertialAt);
    Table->setAddress(world.Velocities.Address(), VelocitiesAt);
    Table->setAddress(world.Masses.Address(), MassesAt);
    Table->setAddress(world.Contacts.Address(), ContactsAt);
    Table->setAddress(world.BodyShapes.Address(), BodyShapesAt);
    Table->setAddress(Params.Address(), ParamsAt);
    Table->setAddress(world.Shapes.Address(), ShapesAt);
    Table->setAddress(world.PreviousVelocities.Address(), PreviousVelocitiesAt);
    Table->setAddress(world.Frictions.Address(), FrictionsAt);
    Table->setAddress(world.SolvedPoses.Address(), SolvedAt);
    Table->setAddress(world.Colors.Address(), ColorsAt);
    Table->setAddress(world.NextColors.Address(), NextColorsAt);
    Table->setAddress(world.Restitutions.Address(), RestitutionsAt);
    Table->setAddress(world.Joints.Address(), JointsAt);
    Table->setAddress(world.Incoming.Address(), IncomingAt);
    Table->setAddress(world.IncomingSlots.Address(), IncomingSlotsAt);
    Table->setAddress(world.Filters.Address(), FiltersAt);
    Table->setAddress(world.Jointed.Address(), JointedAt);
    Table->setAddress(world.Quiet.Address(), QuietAt);
    Table->setAddress(world.RestPoses.Address(), RestPosesAt);
    Table->setAddress(world.NextQuiet.Address(), NextQuietAt);
    Table->setAddress(world.ContactEvents.Address(), ContactEventsAt);
    Table->setAddress(world.ContactEventCounts.Address(), ContactEventCountsAt);
    Table->setAddress(world.ContactRefusals.Address(), ContactRefusalsAt);
    Table->setAddress(world.ShapeVertices.Address(), ShapeVerticesAt);
    Table->setAddress(world.Triangles.Address(), TrianglesAt);
    Table->setAddress(world.BvhNodes.Address(), BvhNodesAt);
    Table->setAddress(world.HullFaces.Address(), HullFacesAt);

    Encode({.Bodies = bodies,
            .Joints = joints,
            .Iterations = settings.Iterations,
            .Colors = colors,
            .ColoringPasses = settings.ColoringPasses});

    // Queue signalling is what safely publishes the GPU's writes to the host, per Architecture.md.
    const MTL4::CommandBuffer *list[]{Commands.get()};
    Context.Queue->commit(list, 1);
    Context.Queue->signalEvent(Done.get(), ++Signal);
    while (!Done->waitUntilSignaledValue(Signal, 1000)) {}
    // The step is over and the world is the host's again, which is what a body removed during the last
    // one has been waiting for. See World::OnStepped.
    world.OnStepped();
}

void Solver::Encode(const Recording &recording) {
    const uint32_t bodies = recording.Bodies, joints = recording.Joints;
    const uint32_t slots = bodies * ContactsPerBody;
    // Safe to recycle the memory the last recording lived in: every step waits for its own completion,
    // so nothing the GPU still holds is in there.
    Allocator->reset();
    Commands->beginCommandBuffer(Allocator.get());
    auto *encoder = Commands->computeCommandEncoder();
    encoder->setArgumentTable(Table.get());

    // A colour pass per colour, so the cursor comes back in phase at the end of every sweep and the
    // step always starts on colour zero without anyone having to reset it.
    const auto sweep = [&](MTL::ComputePipelineState *primal) {
        for (uint32_t color = 0; color < recording.Colors; ++color) {
            Table->setAddress(ColorCursor.Address() + color * sizeof(uint32_t), CursorAt);
            Dispatch(encoder, primal, bodies);
            Dispatch(encoder, PublishPipeline.get(), bodies);
        }
    };

    Dispatch(encoder, IntegratePipeline.get(), bodies);
    // Collision, and with it every C0, anchor and Jacobian, is taken at the pose the step began from,
    // and only then does WarmStart move the body to its starting guess - which is the order the
    // reference solves in and the only one where the Taylor series is expanded about the pose it is
    // measured at.
    Dispatch(encoder, CollectPipeline.get(), bodies);
    // Gather each body's contacts-as-B into a run of its own, so nothing below has to sweep the pool.
    Dispatch(encoder, CountIncomingPipeline.get(), slots);
    Dispatch(encoder, ScanIncomingPipeline.get(), 1);
    Dispatch(encoder, FillIncomingPipeline.get(), slots);
    Dispatch(encoder, SortIncomingPipeline.get(), bodies);
    if (joints > 0) Dispatch(encoder, PrepareJointsPipeline.get(), joints);
    Dispatch(encoder, WarmStartPipeline.get(), bodies);
    for (uint32_t pass = 0; pass < recording.ColoringPasses; ++pass) {
        Dispatch(encoder, ColorPipeline.get(), bodies);
        Dispatch(encoder, PublishColorPipeline.get(), bodies);
    }
    for (uint32_t iteration = 0; iteration < recording.Iterations; ++iteration) {
        sweep(SolvePipeline.get());
        Dispatch(encoder, DualPipeline.get(), slots);
        if (joints > 0) Dispatch(encoder, JointDualPipeline.get(), joints);
    }
    // Velocity comes from the motion the iterations above produced, and is taken before the pass
    // below, so removing leftover penetration does not hand the body the energy that motion implies.
    Dispatch(encoder, FinalizePipeline.get(), bodies);
    // Restitution, which is a velocity pass on that velocity rather than a row inside the solve - see
    // Restitution for why a gapped contact leaves it no choice. It runs before sleeping is counted,
    // since a body just handed a rebound is not a body that has come to rest.
    for (uint32_t pass = 0; pass < RestitutionPasses; ++pass) {
        Dispatch(encoder, RestitutionPipeline.get(), slots);
        Dispatch(encoder, ApplyRestitutionPipeline.get(), bodies);
    }
    sweep(StabilizePipeline.get());
    // Whose turn it is to sleep is settled at the very end of a step, after the stabilization sweep, so
    // a body's sleep state is one answer for every kernel of a step. Published before the sweep
    // instead, a body the spread wakes runs a stabilization pass for a step it slept through: it was
    // asleep at Integrate, so its inertial target is its frozen pose with no gravity in it, while its
    // contact duals still carry the upward force balancing gravity - so the pass moves it M g / H out
    // of its rest pose with no velocity, and the body above inherits that as penetration at full
    // penalty. A settled ten-coin stack collapsed from nothing else.
    Dispatch(encoder, CountQuietPipeline.get(), bodies);
    Dispatch(encoder, SpreadWakingPipeline.get(), bodies);
    Dispatch(encoder, PublishWakingPipeline.get(), bodies);

    encoder->endEncoding();
    Commands->endCommandBuffer();
}
