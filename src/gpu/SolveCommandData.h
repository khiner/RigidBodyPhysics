#pragma once
#ifdef __METAL_VERSION__
#include <metal_command_buffer>
#define COMMAND_BUFFER_TYPE command_buffer
#define COMPUTE_PIPELINE_TYPE compute_pipeline_state
#define COMMAND_ADDRESS(Type) device Type *
#else
#include "gpu/Shared.h"
#define COMMAND_BUFFER_TYPE uint64_t
#define COMPUTE_PIPELINE_TYPE uint64_t
#define COMMAND_ADDRESS(Type) uint64_t
#endif
namespace rbp {
// Bound retained command storage; larger iteration counts use the ordinary GPU schedule.
GPU_CONSTANT uint CommandIterationLimit = 64;
struct SolveCommandData {
    COMMAND_BUFFER_TYPE Commands;
    COMPUTE_PIPELINE_TYPE Pipelines[4];
    COMMAND_ADDRESS(char)
    Buffers[27];
    COMMAND_ADDRESS(uint)
    Groups;
    COMMAND_ADDRESS(uint)
    Range;
    uint FirstCommand, Colors, Iterations, Padding;
};
static_assert(sizeof(SolveCommandData) == 288);
} // namespace rbp
#undef COMMAND_BUFFER_TYPE
#undef COMPUTE_PIPELINE_TYPE
#undef COMMAND_ADDRESS
