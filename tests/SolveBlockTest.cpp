#include "Gpu.h"
#include "GpuSource.h"
#include "gpu/Shared.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <string>

#include <doctest/doctest.h>

using namespace rbp;

TEST_CASE("numerics: coupled block solves match independent solutions") {
    constexpr uint32_t Count = 17, Rows = 6, Stride = Rows + 1;
    constexpr std::array<double, Rows> Solution{0.5, -1, 2, -3, 0, 4};
    std::array<std::array<double, Rows>, Count> expected;
    const mtl::Context context;
    mtl::Buffer<simd::float2> input{context.Device.get(), Count * Rows * Stride};
    mtl::Buffer<float> output{context.Device.get(), Count * Rows + 32};
    std::ranges::fill(output.All(), 12345.f);
    for (uint32_t matrix = 0; matrix < Count; ++matrix) {
        expected[matrix] = Solution;
        // Values just above a float rounding boundary expose discarded compensation.
        if (matrix == 14) expected[matrix].fill(1 + std::ldexp(1., -24) + std::ldexp(1., -28));
        std::array<std::array<double, Rows>, Rows> coefficients{};
        const double scale = std::ldexp(1., (int(matrix) % 3 - 1) * 36);
        for (uint32_t row = 0; row < Rows; ++row) {
            coefficients[row][row] = matrix % 2 == 0 ? std::ldexp(1., int(row) * 19 - 48) : 4 * scale;
            coefficients[row][row] *= 1 + std::ldexp(1., -30);
        }
        if (matrix % 2 == 1) {
            coefficients[1][4] = coefficients[4][1] = scale / 4;
        }
        if (matrix == Count - 1) {
            for (uint32_t row = 0; row < Rows; ++row)
                for (uint32_t column = 0; column < Rows; ++column)
                    coefficients[row][column] = 1024. + (row == column ? std::ldexp(1., -10) : 0);
        }
        for (uint32_t row = 0; row < Rows; ++row) {
            double gradient = 0;
            for (uint32_t column = 0; column <= Rows; ++column) {
                const double value = column < Rows ? coefficients[row][column] : gradient;
                const float high = float(value);
                input[(matrix * Rows + row) * Stride + column] = {high, float(value - high)};
                if (column < Rows) gradient -= value * expected[matrix][column];
            }
        }
    }

    const std::string source = std::string(gpu::SolveSource) + R"(
kernel void ProbeSolveBlock(device const float2 *input [[buffer(0)]], device float *output [[buffer(1)]],
                           uint lane [[thread_index_in_simdgroup]], uint group [[threadgroup_position_in_grid]]) {
    if (lane >= 30) return;
    const uint matrix = group * 5 + lane / 6;
    if (matrix >= 17) return;
    const uint row = lane % 6, start = (matrix * 6 + row) * 7;
    float2 H[Dof];
    for (uint column = 0; column < Dof; ++column) H[column] = input[start + column];
    output[matrix * 6 + row] = SolveBlock(H, input[start + 6], row, lane - row);
}
)";
    auto pipeline = context.Pipeline(source, "ProbeSolveBlock");
    RunGpu(context, pipeline.get(), {{0, input.Handle.get()}, {1, output.Handle.get()}}, (Count + 4) / 5);

    for (uint32_t matrix = 0; matrix < Count; ++matrix) {
        CAPTURE(matrix);
        for (uint32_t row = 0; row < Rows; ++row) {
            CAPTURE(row);
            const float result = output[matrix * Rows + row];
            REQUIRE(std::isfinite(result));
            CHECK(std::abs(result - expected[matrix][row]) <= 2e-6);
            if (matrix % 2 == 0 && matrix != Count - 1) CHECK(result == float(expected[matrix][row]));
            const auto gradient = input[(matrix * Rows + row) * Stride + Rows];
            double residual = double(gradient.x) + gradient.y, magnitude = std::abs(residual);
            for (uint32_t column = 0; column < Rows; ++column) {
                const auto coefficient = input[(matrix * Rows + row) * Stride + column];
                const double term = (double(coefficient.x) + coefficient.y) * output[matrix * Rows + column];
                residual += term;
                magnitude += std::abs(term);
            }
            CHECK(std::abs(residual) <= 2e-7 * magnitude);
        }
    }
    for (uint32_t i = Count * Rows; i < output.Capacity; ++i) CHECK(output[i] == 12345.f);
}

TEST_CASE("numerics: stiff assembly preserves unconstrained axes") {
    bool batched = false;
    SUBCASE("ordered rows") {}
    SUBCASE("parallel contact blocks") { batched = true; }
    constexpr uint32_t Count = 24, Width = 6, Stride = 7;
    const mtl::Context context;
    mtl::Buffer<simd::float2> input{context.Device.get(), Count};
    mtl::Buffer<simd::float2> output{context.Device.get(), Count * Width * Stride};

    const uint32_t lengths[]{1, 7, 32, 128, 512, 1024, 2048, 4096};
    const float stiffness[]{1024.f, 100000000.f, 1000000000.f};
    for (uint32_t i = 0; i < Count; ++i) input[i] = {float(lengths[i % 8]), stiffness[i / 8]};
    const std::string source = std::string(gpu::SolveSource) + R"(
kernel void ProbeAssembly(device const float2 *input [[buffer(0)]], device float2 *output [[buffer(1)]],
                          uint lane [[thread_index_in_simdgroup]], uint matrix [[threadgroup_position_in_grid]]) {
#if !BATCHED_ASSEMBLY
    if (lane >= 6) return;
#endif
    const uint row = lane % 6;
    float2 H[Dof];
    for (uint j = 0; j < Dof; ++j) H[j] = float2(j == row ? 1.f : 0.f, 0);
    const float solution[6] = {1, -1, 2, -2, 3, -3};
    float2 g = float2(-solution[row], 0);
#if BATCHED_ASSEMBLY
    const uint count = uint(input[matrix].x);
    for (uint first = 0; first < count; first += 15) {
        float2 partial[Dof], partial_g = 0;
        for (uint j = 0; j < Dof; ++j) partial[j] = 0;
        if (lane < 30)
            for (uint r = 0; r < 3 && first + (lane / 6) * 3 + r < count; ++r)
                AddRow(partial, partial_g, row, float3(1), float3(1), 1, 0, input[matrix].y);
        for (uint tile = 0; tile < min(5u, (count - first + 2) / 3); ++tile)
            MergePreparedBlock(H, g, partial, partial_g, tile * 6 + row);
    }
    if (lane >= 6) return;
#else
    for (uint i = 0; i < uint(input[matrix].x); ++i)
        AddRow(H, g, lane, float3(1), float3(1), 1, 0, input[matrix].y);
#endif
    for (uint j = 0; j < Dof; ++j) {
        H[j] = WideSum(H[j].x, H[j].y);
        output[(matrix * 6 + lane) * 7 + j] = H[j];
    }
    output[(matrix * 6 + lane) * 7 + 6] = float2(SolveBlock(H, g, lane, 0), 0);
}
)";
    auto pipeline = context.Pipeline(source, "ProbeAssembly", batched ? "#define BATCHED_ASSEMBLY 1" : "#define BATCHED_ASSEMBLY 0");
    RunGpu(context, pipeline.get(), {{0, input.Handle.get()}, {1, output.Handle.get()}}, Count);
    const double solution[]{1, -1, 2, -2, 3, -3};
    for (uint32_t matrix = 0; matrix < Count; ++matrix) {
        CAPTURE(matrix);
        CAPTURE(input[matrix].x);
        CAPTURE(input[matrix].y);
        for (uint32_t row = 0; row < 6; ++row) {
            for (uint32_t j = 0; j < 6; ++j) {
                const auto actual = output[(matrix * 6 + row) * 7 + j];
                const double expected = double(input[matrix].x) * input[matrix].y + double(j == row);
                CHECK(double(actual.x) + actual.y == expected);
            }
            const float result = output[(matrix * 6 + row) * 7 + 6].x;
            REQUIRE(std::isfinite(result));
            CHECK(std::abs(double(result) - solution[row]) <= 2e-6);
        }
    }
}
