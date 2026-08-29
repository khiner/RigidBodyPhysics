#pragma once

#include "World.h"

// The paper's parameters, at the reference implementation's defaults except for Beta, which the mass
// scale here moves. Penalties are absolute rather than scaled to a body's mass, because Beta ramps
// them adaptively from PenaltyMin in proportion to the violation actually seen - they find their own
// scale rather than being told one.
struct StepSettings {
    float3 Gravity{0, -9.81f, 0};
    float DeltaTime = 1.f / 60;
    uint32_t Iterations = 10;
    // Penalty ramp per unit of violation. The reference warns outright that the right number depends
    // on the length and mass scales in use, and it does: its own 1e5 suits roughly unit-mass boxes,
    // while a cubic metre of water is a thousand times heavier. Holding a penalty at a body's inertial
    // stiffness m/h^2 costs a standing error of |C| = (m/h^2)(1 - Gamma)/(Iterations Beta), so a
    // thousandfold heavier body wants a hundredfold larger ramp to sag by the same amount.
    float Beta = 1e7f;
    float Gamma = 0.99f; // how much of the penalty survives into the next step
    float PenaltyMin = 1;
    float PenaltyMax = 1e9f;
    float ContactMargin = 5e-4f;
    // How far past that margin a pair's contact reach may grow with the speed it is closing at.
    // Unclamped by default: this engine has no sweep behind it and defers one on purpose, so the reach
    // has to cover a step of motion or a fast body crosses what it should have hit. Clamping it trades
    // ghost collisions at speed for that crossing, which is the trade every engine with a sweep behind
    // it takes - see CollectContacts.
    float MaxContactReach = INFINITY;
    float MaxAngularSpeed = 50;
    // Restitution below a closing speed of this many times what one step of free flight adds is
    // ignored, so a resting body does not bounce on its own settling jitter. `Integrate` guesses a body
    // forward to x + h v + h^2 g before anything constrains it, which is h |g| of approach speed a body
    // merely settling can show for having been integrated - so the threshold belongs to the step and
    // the gravity in use rather than being a flat speed. Two of them is the XPBD rigid-body paper's
    // choice, for the same reason, and it holds for any solver that guesses before it constrains.
    float BounceSpeedFactor = 2;
    // A body this slow for this long stops being solved until something disturbs it. Half a second of
    // quiet, which is Jolt's, and a short stack settles four orders below the speed.
    float SleepSpeed = 0.02f;
    uint32_t SleepSteps = 30;
    // And how far it may have got over those steps and still count as having gone nowhere. Slowness
    // alone cannot tell a body jittering about a point from one creeping towards it, and a settling
    // stack creeps, so sleeping on speed alone freezes it wherever it had got to.
    float SleepDrift = 1e-3f;
    // Bodies of one color touch nothing in common, so a color solves in parallel and the colors in
    // sequence give Gauss-Seidel propagation. Eight is generous for boxes, which touch few neighbours.
    uint32_t MaxColors = 8;
    uint32_t ColoringPasses = 4; // the paper's incremental coloring settles in a few
};

// The most colours a step may be dispatched with, which is the size of the cursor table the encoder
// walks. Settings can ask for fewer and never for more.
inline constexpr uint32_t MaxSupportedColors = 32;

// The fixed step, encoded as one command buffer with one compute encoder and a barrier between passes
// that truly depend on each other, which Architecture.md records as much the cheapest way to say it.
//
// Nothing here is atomic and no ordering depends on which thread finished first, so a step is a pure
// function of the world it was handed: two runs of the same scene produce bit-identical state.
struct Solver {
    explicit Solver(const mtl::Context &);
    ~Solver(); // to give its residency set back to the queue, for the reason World's destructor carries

    void Step(World &, const StepSettings & = {});

private:
    // Everything a step's encoded commands depend on. Nothing else about a step changes what is
    // recorded, only what the buffers behind it contain.
    struct Recording {
        uint32_t Bodies{}, Joints{}, Iterations{}, Colors{}, ColoringPasses{};
    };

    // The kernels a step is made of. Named here, spelled once in the table Solver.cpp builds them from,
    // and dispatched by name below - so a pass exists in two places rather than in three.
    enum Pass : uint32_t {
        IntegratePass,
        CollectPass,
        CountIncomingPass,
        ScanIncomingPass,
        FillIncomingPass,
        SortIncomingPass,
        PrepareJointsPass,
        WarmStartPass,
        ColorPass,
        PublishColorPass,
        SolvePass,
        PublishPass,
        DualPass,
        JointDualPass,
        FinalizePass,
        RestitutionPass,
        ApplyRestitutionPass,
        StabilizePass, // SolveBodies again, compiled with the C0 term kept
        CountQuietPass,
        SpreadWakingPass,
        PublishWakingPass,
        PassCount,
    };

    void Encode(const Recording &); // which is the whole of what it reads, as above
    void Dispatch(MTL4::ComputeCommandEncoder *, Pass, uint32_t threads) const;

    const mtl::Context &Context;
    NS::SharedPtr<MTL::ComputePipelineState> Pipelines[PassCount];
    NS::SharedPtr<MTL4::ArgumentTable> Table;
    NS::SharedPtr<MTL4::CommandAllocator> Allocator;
    NS::SharedPtr<MTL4::CommandBuffer> Commands;
    NS::SharedPtr<MTL::SharedEvent> Done;
    NS::SharedPtr<MTL::ResidencySet> Residency;
    mtl::Buffer<StepParams> Params;
    mtl::Buffer<uint32_t> ColorCursor;
    uint64_t Signal{};
};
