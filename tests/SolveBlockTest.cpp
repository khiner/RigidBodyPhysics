#include "GpuSource.h"
#include "gpu/Shared.h"
#include "metal/Buffer.h"
#include "metal/Context.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <string>

#include <doctest/doctest.h>

using namespace rbp;

TEST_CASE("compensated sums preserve normalization across signs, scales and cancellation") {
    constexpr uint32_t Count = 65536, EdgeCases = 128;
    const mtl::Context context;
    mtl::Buffer<simd::float2> input{context.Device.get(), 2 * (Count + EdgeCases)};
    mtl::Buffer<simd::float2> output{context.Device.get(), 2 * (Count + EdgeCases) + 32};
    std::ranges::fill(output.All(), simd::float2{12345.f, 12345.f});
    uint32_t random = 0x68bc21eb;
    const auto next = [&] { return random = 1664525u * random + 1013904223u; };
    for (uint32_t i = 0; i < Count; ++i) {
        for (uint32_t operand = 0; operand < 2; ++operand) {
            const int exponent = int(next() % 161) - 60;
            const uint32_t bits = (next() & 0x807fffffu) | (uint32_t(exponent + 127) << 23);
            input[2 * i + operand] = {std::bit_cast<float>(bits), std::ldexp(float(double(std::bit_cast<int32_t>(next())) / 4294967296.), exponent - 23)};
        }
        if (i % 4 == 0) input[2 * i + 1] = -input[2 * i];
        if (i % 4 == 1) input[2 * i + 1] = {-input[2 * i].x, input[2 * i].y};
        if (i % 4 == 2) input[2 * i + 1] = {-std::nextafter(input[2 * i].x, 0.f), -input[2 * i].y};
    }
    const std::array<float, 8> edge{
        0, -0.f, std::numeric_limits<float>::denorm_min(), std::numeric_limits<float>::min(),
        std::nextafter(std::numeric_limits<float>::min(), 1.f), std::ldexp(1.f, -124),
        std::ldexp(1.f, 124), std::numeric_limits<float>::max()
    };
    for (uint32_t i = 0; i < EdgeCases; ++i) {
        const float sign = i / 64 ? -1.f : 1.f;
        input[2 * (Count + i)] = {sign * edge[i % edge.size()], 0};
        input[2 * (Count + i) + 1] = {-sign * edge[i / 8 % edge.size()], 0};
    }
    const std::string source = std::string(gpu::SolveSource) + R"(
static float2 FullAdd(float2 a, float2 b) {
#pragma clang fp reassociate(off) contract(off)
    const float2 sum = WideSum(a.x, b.x);
    return WideSum(sum.x, sum.y + a.y + b.y);
}
kernel void ProbeSums(device const float2 *input [[buffer(0)]], device float2 *output [[buffer(1)]],
                     uint id [[thread_position_in_grid]]) {
    const float2 a = WideSum(input[2 * id].x, input[2 * id].y);
    const float2 b = WideSum(input[2 * id + 1].x, input[2 * id + 1].y);
    output[2 * id] = WideAdd(a, b);
    output[2 * id + 1] = FullAdd(a, b);
}
)";
    auto pipeline = context.Pipeline(source, "ProbeSums");
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
    encoder->dispatchThreads({Count + EdgeCases, 1, 1}, {32, 1, 1});
    encoder->endEncoding();
    commands->endCommandBuffer();
    auto done = NS::TransferPtr(context.Device->newSharedEvent());
    const MTL4::CommandBuffer *list[]{commands.get()};
    context.Queue->commit(list, 1);
    context.Queue->signalEvent(done.get(), 1);
    while (!done->waitUntilSignaledValue(1, 1000)) {}
    for (uint32_t i = 0; i < Count + EdgeCases; ++i) {
        CAPTURE(i);
        const auto sum = output[2 * i], reference = output[2 * i + 1];
        REQUIRE(std::isfinite(sum.x));
        REQUIRE(std::isfinite(sum.y));
        CHECK(sum.x == reference.x);
        CHECK(sum.y == reference.y);
        if (i >= Count) continue;
        const auto a = input[2 * i], b = input[2 * i + 1];
        const double av = double(a.x) + a.y, bv = double(b.x) + b.y;
        const double expected = av + bv, actual = double(sum.x) + sum.y;
        CHECK(std::abs(actual - expected) <= 4e-14 * (std::abs(av) + std::abs(bv)));
    }
    for (uint32_t i = 2 * (Count + EdgeCases); i < output.Capacity; ++i) {
        CHECK(output[i].x == 12345.f);
        CHECK(output[i].y == 12345.f);
    }
    context.Queue->removeResidencySet(residency.get());
}

TEST_CASE("compensated products retain roundoff across signs, scales and cancellation") {
    constexpr uint32_t Count = 16384, EdgeCases = 32;
    const mtl::Context context;
    mtl::Buffer<simd::float2> input{context.Device.get(), 2 * (Count + EdgeCases)};
    mtl::Buffer<simd::float2> output{context.Device.get(), 3 * (Count + EdgeCases) + 32};
    std::ranges::fill(output.All(), simd::float2{12345.f, 12345.f});
    uint32_t random = 0x68bc21eb;
    const auto next = [&] { return random = 1664525u * random + 1013904223u; };
    for (uint32_t i = 0; i < Count; ++i) {
        for (uint32_t operand = 0; operand < 2; ++operand) {
            const int exponent = int(next() % 81) - 40;
            const uint32_t bits = (next() & 0x807fffffu) | (uint32_t(exponent + 127) << 23);
            input[2 * i + operand] = {std::bit_cast<float>(bits), std::ldexp(float(double(std::bit_cast<int32_t>(next())) / 4294967296.), exponent - 23)};
        }
        if (i % 4 == 0) {
            input[2 * i + 1] = input[2 * i];
            input[2 * i + 1].y = -input[2 * i].y;
        }
    }
    const std::array<float, 8> edge{
        0, -0.f, std::numeric_limits<float>::denorm_min(), std::numeric_limits<float>::min(),
        std::nextafter(std::numeric_limits<float>::min(), 1.f), std::ldexp(1.f, -124),
        std::ldexp(1.f, 124), std::numeric_limits<float>::max()
    };
    for (uint32_t i = 0; i < EdgeCases; ++i) {
        input[2 * (Count + i)] = {edge[i % edge.size()], 0};
        input[2 * (Count + i) + 1] = {i / 8 % 2 ? -0.5f : 0.5f, i / 16 ? std::ldexp(1.f, -26) : 0};
    }
    const std::string source = std::string(gpu::SolveSource) + R"(
static float2 FullProductSum(float2 a, float2 b) {
#pragma clang fp reassociate(off) contract(off)
    const float product = a.x * b.x;
    const float error = fma(a.x, b.x, -product) + a.x * b.y + a.y * b.x + a.y * b.y;
    return WideSum(product, error);
}
kernel void ProbeProducts(device const float2 *input [[buffer(0)]], device float2 *output [[buffer(1)]],
                          uint id [[thread_position_in_grid]]) {
    const float2 a = input[2 * id], b = input[2 * id + 1];
    output[3 * id] = WideMul(a, b);
    output[3 * id + 1] = FullProductSum(a, b);
    if (id < 16384) output[3 * id + 2] = WideAdd(WideMul(a, a), -WideMul(b, b));
}
)";
    auto pipeline = context.Pipeline(source, "ProbeProducts");
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
    encoder->dispatchThreads({Count + EdgeCases, 1, 1}, {32, 1, 1});
    encoder->endEncoding();
    commands->endCommandBuffer();
    auto done = NS::TransferPtr(context.Device->newSharedEvent());
    const MTL4::CommandBuffer *list[]{commands.get()};
    context.Queue->commit(list, 1);
    context.Queue->signalEvent(done.get(), 1);
    while (!done->waitUntilSignaledValue(1, 1000)) {}
    for (uint32_t i = 0; i < Count + EdgeCases; ++i) {
        CAPTURE(i);
        const auto product = output[3 * i], reference = output[3 * i + 1];
        REQUIRE(std::isfinite(product.x));
        REQUIRE(std::isfinite(product.y));
        CHECK(product.x == reference.x);
        CHECK(product.y == reference.y);
        if (i >= Count) continue;
        const auto a = input[2 * i], b = input[2 * i + 1];
        const double av = double(a.x) + a.y, bv = double(b.x) + b.y;
        const double expected = av * bv, actual = double(product.x) + product.y;
        CHECK(std::abs(actual - expected) <= 4e-14 * std::abs(expected));
        const auto difference = output[3 * i + 2];
        CHECK(std::abs((double(difference.x) + difference.y) - (av * av - bv * bv)) <= 4e-14 * (av * av + bv * bv));
    }
    for (uint32_t i = 3 * (Count + EdgeCases); i < output.Capacity; ++i) {
        CHECK(output[i].x == 12345.f);
        CHECK(output[i].y == 12345.f);
    }
    context.Queue->removeResidencySet(residency.get());
}

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

TEST_CASE("compensated assembly retains free axes beside thousands of stiff rows") {
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
    if (lane >= 6) return;
    float2 H[Dof];
    for (uint j = 0; j < Dof; ++j) H[j] = float2(j == lane ? 1.f : 0.f, 0);
    const float solution[6] = {1, -1, 2, -2, 3, -3};
    float2 g = float2(-solution[lane], 0);
    for (uint i = 0; i < uint(input[matrix].x); ++i)
        AddRow(H, g, lane, float3(1), float3(1), 1, 0, input[matrix].y);
    for (uint j = 0; j < Dof; ++j) {
        H[j] = WideSum(H[j].x, H[j].y);
        output[(matrix * 6 + lane) * 7 + j] = H[j];
    }
    output[(matrix * 6 + lane) * 7 + 6] = float2(SolveBlock(H, g, lane, 0), 0);
}
)";
    auto pipeline = context.Pipeline(source, "ProbeAssembly");
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
    encoder->dispatchThreadgroups({Count, 1, 1}, {32, 1, 1});
    encoder->endEncoding();
    commands->endCommandBuffer();
    auto done = NS::TransferPtr(context.Device->newSharedEvent());
    const MTL4::CommandBuffer *list[]{commands.get()};
    context.Queue->commit(list, 1);
    context.Queue->signalEvent(done.get(), 1);
    while (!done->waitUntilSignaledValue(1, 1000)) {}
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
    context.Queue->removeResidencySet(residency.get());
}
