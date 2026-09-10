#include "GpuSource.h"
#include "gpu/Shared.h"
#include "metal/Buffer.h"
#include "metal/Context.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <string>

#include <doctest/doctest.h>

using namespace rbp;

TEST_CASE("SIMD block solves retain independent solutions with mixed diagonal and coupled matrices") {
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
    NS::Error *error{};
    auto residency = NS::TransferPtr(context.Device->newResidencySet(mtl::Make<MTL::ResidencySetDescriptor>().get(), &error));
    residency->addAllocation(input.Handle.get());
    residency->addAllocation(output.Handle.get());
    residency->commit();
    residency->requestResidency();
    context.Queue->addResidencySet(residency.get());
    auto desc = mtl::Make<MTL4::ArgumentTableDescriptor>();
    desc->setMaxBufferBindCount(2);
    auto table = NS::TransferPtr(context.Device->newArgumentTable(desc.get(), &error));
    table->setAddress(input.Address(), 0);
    table->setAddress(output.Address(), 1);
    auto allocator = NS::TransferPtr(context.Device->newCommandAllocator());
    auto commands = NS::TransferPtr(context.Device->newCommandBuffer());
    commands->beginCommandBuffer(allocator.get());
    auto *encoder = commands->computeCommandEncoder();
    encoder->setArgumentTable(table.get());
    encoder->setComputePipelineState(pipeline.get());
    encoder->dispatchThreadgroups({(Count + 4) / 5, 1, 1}, {32, 1, 1});
    encoder->endEncoding();
    commands->endCommandBuffer();
    auto done = NS::TransferPtr(context.Device->newSharedEvent());
    const MTL4::CommandBuffer *list[]{commands.get()};
    context.Queue->commit(list, 1);
    context.Queue->signalEvent(done.get(), 1);
    while (!done->waitUntilSignaledValue(1, 1000)) {}

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
    context.Queue->removeResidencySet(residency.get());
}
