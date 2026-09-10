#pragma once

#include "World.h"
#include "gpu/FullStepData.h"
#include "gpu/SolveCommandData.h"

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
    float Gamma = 0.99f; // Fraction of the penalty retained for the next step.
    float PenaltyMin = 1;
    float PenaltyMax = 1e9f;
    float ContactMargin = 5e-4f;
    // Maximum speculative reach beyond ContactMargin; INFINITY disables the cap.
    float MaxContactReach = INFINITY;
    float MaxAngularSpeed = 50;
    // Ignore restitution below this multiple of DeltaTime * |Gravity|.
    float BounceSpeedFactor = 2;
    // Sleep after SleepSteps consecutive steps below SleepSpeed.
    float SleepSpeed = 0.02f;
    uint32_t SleepSteps = 30;
    // Maximum displacement over the quiet interval before sleeping.
    float SleepDrift = 1e-3f;
    // Color cap; unresolved conflicts use snapshot reads and relaxed updates.
    uint32_t MaxColors = 8;
    uint32_t ColoringPasses = 4;
};

struct StepResult {
    uint64_t Step, ContactRefusals, SensorRefusals;
    std::span<const Pose> Poses;
    std::span<const Velocity> Velocities;
};

struct AdvanceResult {
    uint32_t Steps = 0;
    uint64_t ContactRefusals = 0, SensorRefusals = 0;
};

// Step and Advance block until GPU execution and reporting complete.
// The solver and world must share a live context.
struct Solver {
    explicit Solver(const mtl::Context &);
    ~Solver();

    void Step(World &, const StepSettings & = {});
    // DeltaTime applies to each substep.
    // Followers update before each substep.
    // Observer spans are immutable and valid only during the callback.
    // Observers require an unchanged world and exclusive use of the solver.
    AdvanceResult Advance(World &, const StepSettings &, uint32_t substeps, std::span<const SensorFollower> followers = {}, const std::function<void(const StepResult &)> &observer = {});

private:
    struct Recording {
        uint32_t Bodies{}, Joints{}, Iterations{}, Colors{}, ColoringPasses{};
        uint32_t ColliderFeatures{}, Parameter{};
        bool GpuColors = false, Snapshot = false;
    };

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
        StabilizePass,
        FinishPosesPass,
        FinishWakingPass,
        FollowPass,
        PrepareColorsPass,
        ReduceStepColorsPass,
        FinishStepColorsPass,
        CapturePass,
        FinishCapturePass,
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
        CountColorWorkPass,
        PrefixColorWorkPass,
        FillColorWorkPass,
        SolveSmallWorldPass,
        ResetQueryReusePass,
        CheckQueryInputsPass,
        CheckQueryGeometryPass,
        CacheBoundsPass,
        InitializeIslandsPass,
        UnionIslandsPass,
        PackIslandsPass,
        FinishIslandsPass,
        SolveIslandsPass,
        PrepareSmallWorldPass,
        NativeCollectPass,
        NativeSensorPass,
        NativeBoundedCollectPass,
        NativeBoundedSensorPass,
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
    void EncodeSolveCommands(MTL4::ComputeCommandEncoder *, const Recording &, World &);
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
    mtl::Buffer<uint32_t> SensorQueries, QueryScratch, QueryInputSnapshot;
    mtl::Buffer<QueryInputSpec> QueryInputs;
    NS::SharedPtr<MTL4::ArgumentTable> Table;
    NS::SharedPtr<MTL4::CommandAllocator> Allocator;
    NS::SharedPtr<MTL4::CommandBuffer> Commands;
    NS::SharedPtr<MTL::SharedEvent> Done;
    NS::SharedPtr<MTL::ResidencySet> Residency;
    mtl::Buffer<StepParams> Params;
    mtl::Buffer<uint32_t> ColorCursor;
    mtl::Buffer<SensorFollower> Followers;
    std::vector<FollowerRange> FollowerRanges;
    mtl::Buffer<uint32_t> ColorGroups, SmallIslands, GeneralIslands;
    mtl::Buffer<ColorWork> ColorScratch;
    mtl::Buffer<StepOutputFlags> OutputFlags;
    mtl::Buffer<uint8_t> Outputs;
    mtl::Buffer<FullStepData> FullData;
    NS::SharedPtr<MTL::ComputePipelineState> FullPipeline;
    NS::SharedPtr<MTL::IndirectCommandBuffer> SolveCommands;
    NS::SharedPtr<MTL::ComputePipelineState> SolveCommandEncoder, CommandPrimalPipelines[2], CommandAuxiliaryPipelines[3];
    mtl::Buffer<SolveCommandData> SolveCommandDataBuffer;
    mtl::Buffer<uint32_t> SolveCommandRanges;
    OutputLayout Layout;
    uint64_t Signal{};
    bool Advancing = false;
};

} // namespace rbp
