#pragma once

#include "World.h"

#include <functional>

namespace rbp {

struct StepSettings {
    float3 Gravity{0, -9.81f, 0};
    float DeltaTime = 1.f / 60;
    uint32_t Iterations = 10;
    // Absolute joint penalty ramp per unit of linear or angular violation.
    float Beta = 1e7f;
    // Contact ramp per metre of violation, scaled by inertial stiffness and sustained load.
    float ContactBeta = 10;
    float Gamma = 0.99f; // the fraction of a penalty carried into the next step
    float PenaltyMin = 1;
    float PenaltyMax = 1e9f;
    float ContactMargin = 5e-4f;
    // The limit on how far past ContactMargin a pair's contact reach grows with its closing speed.
    // Unclamped by default: the engine has no sweep, so the reach must cover a step of motion or a fast body passes through a body it should have hit.
    // Clamping accepts that crossing in exchange for fewer ghost collisions at speed. See CollectContacts.
    float MaxContactReach = INFINITY;
    float MaxAngularSpeed = 50;
    // Restitution is ignored below a closing speed of this many times h |g|, the approach one step of free fall adds.
    // `Integrate` predicts a body forward to x + h v + h^2 g, so a settling body shows h |g| of approach from the integration alone.
    // The threshold therefore scales with the step and the gravity in use, and a resting body does not bounce on its own jitter.
    // A factor of two is the XPBD rigid-body paper's choice.
    float BounceSpeedFactor = 2;
    // A body below this speed for this many steps sleeps until it is woken.
    // Thirty steps is half a second, matching Jolt, and a short stack settles four orders of magnitude below SleepSpeed.
    float SleepSpeed = 0.02f;
    uint32_t SleepSteps = 30;
    // The most a body may move over those steps and still sleep.
    // Speed alone does not separate a body jittering about a point from one creeping in a direction, and a settling stack creeps.
    float SleepDrift = 1e-3f;
    // Bodies of one color share no constraint, so a color solves in parallel, and the colors run in sequence for Gauss-Seidel propagation.
    // Eight is ample for boxes, which contact few neighbours.
    uint32_t MaxColors = 8;
    uint32_t ColoringPasses = 4; // incremental coloring converges in a few passes
};

// The size of the encoder's color cursor table, and so the most colors a step may be dispatched with.
// StepSettings::MaxColors is clamped to this.
inline constexpr uint32_t MaxSupportedColors = 32;

struct StepResult {
    uint64_t Step, ContactRefusals, SensorRefusals;
    std::span<const Pose> Poses;
    std::span<const Velocity> Velocities;
};

struct AdvanceResult {
    uint32_t Steps = 0;
    uint64_t ContactRefusals = 0, SensorRefusals = 0;
};

// The fixed step, encoded as one command buffer with one compute encoder and a barrier between passes that depend on each other.
// Architecture.md measures that as the cheapest encoding.
//
// Two runs of the same scene produce bit-identical state.
struct Solver {
    explicit Solver(const mtl::Context &);
    ~Solver(); // returns the residency set to the queue, then drains. See mtl::Drain.

    void Step(World &, const StepSettings & = {});
    AdvanceResult Advance(World &, const StepSettings &, uint32_t substeps, std::span<const SensorFollower> followers = {}, const std::function<void(const StepResult &)> &observer = {});

private:
    // The inputs the encoded commands depend on.
    // Any other change to a step changes only the contents of the buffers it binds.
    struct Recording {
        uint32_t Bodies{}, Joints{}, Iterations{}, Colors{}, ColoringPasses{};
        uint32_t ColliderFeatures{}, Parameter{};
        bool GpuColors = false, Snapshot = false;
    };

    // The kernels of a step, in the order of the pipeline table in Solver.cpp.
    enum Pass : uint32_t {
        ReduceBoundsPass,
        ReduceScenePass,
        MortonPass,
        RadixHistogramPass,
        RadixOffsetsPass,
        RadixScatterPass,
        BuildTreePass,
        RefitTreePass,
        RefreshTreePass,
        SmallBroadPhasePass,
        BoundsPass,
        SensorBoundsPass,
        IntegratePass,
        CollectPass,
        ScanIncomingPass,
        ScanIncomingBlocksPass,
        OffsetIncomingPass,
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
        FollowPass,
        PrepareColorsPass,
        CapturePass,
        SensorPass,
        BoundedCollectPass,
        BoundedSensorPass,
        MeshCollectPass,
        MeshSensorPass,
        FullCollectPass,
        FullSensorPass,
        ResetQueryPass,
        QueryPass,
        SensorQueryPass,
        PassCount,
    };

    struct OutputLayout {
        uint64_t Initial{}, Poses{}, Velocities{}, Contacts{}, RemovedContacts{}, Sensors{}, Counts{}, Completion{}, Stride{};
    };
    struct FollowerRange {
        uint32_t First, Count;
    };
    void PrepareFollowers(World &, std::span<const SensorFollower>);
    void Bind(World &, uint32_t parameter);
    void Encode(MTL4::ComputeCommandEncoder *, const Recording &, World &);
    enum CollectionMode : uint32_t {
        Direct,
        Prepare,
        Queued,
        Recompute,
        CollectionModeCount
    };
    void Dispatch(MTL4::ComputeCommandEncoder *, Pass, uint32_t threads, uint32_t lanes = 1, uint64_t indirect = 0, CollectionMode = Direct);

    const mtl::Context &Context;
    NS::SharedPtr<MTL::ComputePipelineState> Pipelines[CollectionModeCount][PassCount][4];
    mtl::Buffer<uint32_t> QueryScratch;
    NS::SharedPtr<MTL4::ArgumentTable> Table;
    NS::SharedPtr<MTL4::CommandAllocator> Allocator;
    NS::SharedPtr<MTL4::CommandBuffer> Commands;
    NS::SharedPtr<MTL::SharedEvent> Done;
    NS::SharedPtr<MTL::ResidencySet> Residency;
    mtl::Buffer<StepParams> Params;
    mtl::Buffer<uint32_t> ColorCursor;
    mtl::Buffer<SensorFollower> Followers;
    std::vector<FollowerRange> FollowerRanges;
    mtl::Buffer<uint32_t> ColorGroups;
    mtl::Buffer<StepOutputFlags> OutputFlags;
    mtl::Buffer<uint8_t> Outputs;
    OutputLayout Layout;
    uint64_t Signal{};
    bool Advancing = false;
};

} // namespace rbp
