__attribute__((noinline)) static void EncodeIterationCommand(constant SolveCommandData &s, uint at, uint kind, uint groups, uint color) {
    compute_command command(s.Commands, at);
    command.reset();
    command.set_barrier();
    command.set_compute_pipeline_state(s.Pipelines[kind]);
    if (kind == 0) {
        const uint bindings[]{0, 1, 2, 4, 5, 7, 10, 11, 12, 13, 16, 17, 18, 20, 21, 26};
#pragma unroll
        for (uint i = 0; i < 16; ++i) command.set_kernel_buffer(s.Buffers[bindings[i]], bindings[i]);
        command.set_kernel_buffer(s.Buffers[14] + color * sizeof(uint), 14);
    } else if (kind == 1) {
        const uint bindings[]{0, 11, 10, 4, 21, 7};
#pragma unroll
        for (uint i = 0; i < 6; ++i) command.set_kernel_buffer(s.Buffers[bindings[i]], bindings[i]);
    } else if (kind == 2) {
        const uint bindings[]{5, 10, 1, 4, 21, 17, 18, 7, 25};
#pragma unroll
        for (uint i = 0; i < 9; ++i) command.set_kernel_buffer(s.Buffers[bindings[i]], bindings[i]);
    } else {
        const uint bindings[]{16, 0, 1, 7, 25, 4, 21};
#pragma unroll
        for (uint i = 0; i < 7; ++i) command.set_kernel_buffer(s.Buffers[bindings[i]], bindings[i]);
    }
    command.concurrent_dispatch_threadgroups(uint3(groups, 1, 1), uint3(kind == 1 ? 128 : kind == 2 ? 64 :
                                                                                                      32,
                                                                        1, 1));
}

kernel void EncodeSolveCommands(constant SolveCommandData &s [[buffer(0)]], uint lane [[thread_index_in_simdgroup]], uint iteration [[threadgroup_position_in_grid]]) {
    // Iterations write disjoint command records in parallel; execution retains their serial order.
    const uint primal_groups = s.Groups[IslandColorsAt + lane * 3];
    const bool active = lane < s.Colors && primal_groups != 0;
    const uint index = simd_prefix_exclusive_sum(uint(active));
    const uint count = simd_sum(uint(active));
    const uint auxiliary_groups = lane < 3 ? s.Groups[3 + lane * 3] : 0;
    const uint auxiliary_index = simd_prefix_exclusive_sum(uint(auxiliary_groups != 0));
    const uint total = count + simd_sum(uint(auxiliary_groups != 0));
    if (lane == 0 && iteration == 0) {
        s.Range[0] = s.FirstCommand;
        s.Range[1] = total * s.Iterations;
    }
    const uint first = s.FirstCommand + iteration * total;
    if (active) EncodeIterationCommand(s, first + index, 0, primal_groups, lane);
    if (auxiliary_groups) EncodeIterationCommand(s, first + count + auxiliary_index, lane + 1, auxiliary_groups, 0);
}
