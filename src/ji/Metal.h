#pragma once

#include "Oracle.h"
#include "gpu/JiData.h"

#include <Metal/Metal.hpp>

#include <array>
#include <span>

namespace rbp::mtl { struct Context; }

namespace ji {

struct GpuResult {
    System system;
    Residual residual;
    uint32_t iterations = 0;
    bool converged = false;
};

// Retains pipeline objects through command completion. The same ADMM schedule
// serves frozen-system tests and the scene encoder; the caller owns all buffers.
struct MetalPasses {
    explicit MetalPasses(const rbp::mtl::Context &);
    void Encode(MTL4::ComputeCommandEncoder *, MTL4::ArgumentTable *, Settings,
                uint32_t bodies, uint32_t constraint_capacity, bool initialize_penalties,
                MTL4::CounterHeap *profile_heap = nullptr,
                MTL4::ArgumentTable *snapshot_table = nullptr) const;

private:
    std::array<NS::SharedPtr<MTL::ComputePipelineState>, 11> Pipelines;
};

GpuResult DecodeGpu(System, std::span<const JiBodyData>, std::span<const JiConstraintTile>, uint32_t,
                    JiStatus);

// Runs fixed iterations on frozen Jacobians in one Metal command buffer.
// This is the equation baseline; it does not perform collision or stepping.
GpuResult RunGpu(const rbp::mtl::Context &, System, Settings, bool initialize_penalties = false);

} // namespace ji
