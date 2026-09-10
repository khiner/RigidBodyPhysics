#include "Gpu.h"
#include "GpuSource.h"
#include "gpu/Shared.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <random>
#include <string>
#include <vector>

#include <doctest/doctest.h>

using namespace rbp;

TEST_CASE("hull distance agrees with an independent prism distance under rotation") {
    constexpr uint32_t Count = 4096;
    struct Query {
        float3 Position;
        float4 Orientation;
    };
    std::array<float3, 64> vertices;
    for (uint32_t i = 0; i < 32; ++i) {
        const double angle = 2 * std::numbers::pi * i / 32;
        for (uint32_t end = 0; end < 2; ++end)
            vertices[2 * i + end] = {float(0.5 * std::cos(angle)), end ? 0.7f : -0.7f, float(0.5 * std::sin(angle))};
    }
    std::vector<Query> queries(Count);
    std::vector<double> expected(Count);
    std::mt19937 random(70553);
    std::uniform_real_distribution<float> offset(-1.7f, 1.7f), unit(-1.f, 1.f);
    for (uint32_t i = 0; i < Count; ++i) {
        auto &query = queries[i];
        query.Position = {offset(random), offset(random), offset(random)};
        query.Orientation = simd::normalize(float4{unit(random), unit(random), unit(random), unit(random)});
        constexpr float3 aligned[]{{0, 0, 0}, {1, 0, 0}, {1.01f, 0, 0}, {0, 1.4f, 0}, {0, 1.41f, 0}, {0.99f, 1.399f, 0}};
        if (i < std::size(aligned)) {
            query.Position = aligned[i];
            query.Orientation = {0, 0, 0, 1};
        }

        const double x = query.Position.x, z = query.Position.z;
        double nearest = INFINITY;
        bool inside = true;
        for (uint32_t edge = 0; edge < 32; ++edge) {
            const auto from = vertices[edge * 2], to = vertices[((edge + 1) % 32) * 2];
            const double ax = 2.0 * from.x, az = 2.0 * from.z;
            const double dx = 2.0 * to.x - ax, dz = 2.0 * to.z - az;
            inside &= dx * (z - az) - dz * (x - ax) >= 0;
            const double t = std::clamp(((x - ax) * dx + (z - az) * dz) / (dx * dx + dz * dz), 0.0, 1.0);
            nearest = std::min(nearest, std::hypot(x - ax - t * dx, z - az - t * dz));
        }
        const double axial = std::max(0.0, std::abs(double(query.Position.y)) - double(2 * 0.7f));
        expected[i] = std::hypot(inside ? 0.0 : nearest, axial);
    }

    const mtl::Context context;
    mtl::Buffer<float3> points{context.Device.get(), uint32_t(vertices.size())};
    mtl::Buffer<Query> input{context.Device.get(), Count};
    mtl::Buffer<float4> output{context.Device.get(), Count + 1};
    std::ranges::copy(vertices, points.Data());
    std::ranges::copy(queries, input.Data());
    const std::string source = std::string(gpu::SolveSource) + R"(
struct DistanceQuery { float3 Position; float4 Orientation; };
kernel void ProbeHullDistance(device const float3 *points [[buffer(0)]], device const DistanceQuery *input [[buffer(1)]],
    device float4 *output [[buffer(2)]], uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    const DistanceQuery q = input[group];
    Poly a{}, b{};
    a.Kind = b.Kind = ShapeHull;
    a.Count = b.Count = 64;
    a.Orientation = b.Orientation = q.Orientation;
    b.Center = Rotate(q.Orientation, q.Position);
    Mink simplex[4]; uint count = 0; float3 direction; float distance;
    const bool inside = Gjk(a, b, points, simplex, count, direction, distance, COLLECT_LANES == 1 ? NoIndex : lane);
    if (lane == 0) output[group] = float4(direction, inside ? 0.f : distance);
}
)";
    for (uint32_t lanes : {1u, 32u}) {
        CAPTURE(lanes);
        constexpr float Guard = -99999;
        std::ranges::fill(output.All(), float4{Guard, Guard, Guard, Guard});
        auto pipeline = context.Pipeline(source, "ProbeHullDistance", "#define COLLECT_LANES " + std::to_string(lanes));
        RunGpu(context, pipeline.get(), {{0, points.Handle.get()}, {1, input.Handle.get()}, {2, output.Handle.get()}}, Count, lanes);
        double maximum = 0;
        uint32_t worst = 0;
        for (uint32_t i = 0; i < Count; ++i) {
            const auto value = output[i];
            REQUIRE(std::isfinite(value.w));
            REQUIRE(value.w >= 0);
            CHECK(simd::length(float3{value.x, value.y, value.z}) == doctest::Approx(1).epsilon(1e-5));
            const double error = std::abs(double(value.w) - expected[i]);
            if (error > maximum) {
                maximum = error;
                worst = i;
            }
        }
        CAPTURE(worst);
        CAPTURE(maximum);
        CAPTURE(expected[worst]);
        CAPTURE(output[worst].w);
        CHECK(maximum < 1e-4);
        CHECK(simd::all(output[Count] == float4{Guard, Guard, Guard, Guard}));
    }
}
