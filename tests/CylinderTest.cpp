#include "Gpu.h"
#include "GpuSource.h"
#include "Solver.h"
#include "World.h"
#include "fixtures/CylinderQueries.h"
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

namespace {
template<class Query, class Check>
void QueryCylinders(std::span<const Query> queries, std::string_view kernel, Check check) {
    const uint32_t Count = uint32_t(queries.size());
    const mtl::Context context;
    mtl::Buffer<float3> points{context.Device.get(), 1};
    mtl::Buffer<Query> input{context.Device.get(), Count};
    mtl::Buffer<float4> output{context.Device.get(), Count * 2 + 1};
    std::ranges::copy(queries, input.Data());
    const std::string source = std::string(gpu::SolveSource) + std::string(kernel);
    for (uint32_t lanes : {1u, 32u}) {
        CAPTURE(lanes);
        constexpr float Guard = -99999;
        std::ranges::fill(output.All(), float4{Guard, Guard, Guard, Guard});
        auto pipeline = context.Pipeline(source, "ProbeCylinderDistance", "#define COLLECT_LANES " + std::to_string(lanes));
        RunGpu(context, pipeline.get(), {{0, points.Handle.get()}, {1, input.Handle.get()}, {2, output.Handle.get()}}, Count, lanes);
        for (uint32_t i = 0; i < Count; ++i) {
            CAPTURE(i);
            const auto value = output[2 * i], contact = output[2 * i + 1];
            REQUIRE(std::isfinite(value.w));
            CHECK(simd::length(float3{value.x, value.y, value.z}) == doctest::Approx(1).epsilon(1e-5));
            if (contact.x > 0) CHECK(contact.z < 1e-4f);
            check(i, value, contact);
        }
        CHECK(simd::all(output[Count * 2] == float4{Guard, Guard, Guard, Guard}));
    }
}
} // namespace

TEST_CASE("collision: cylinder distances match analytic and independent reference geometry") {
    SUBCASE("native cylinders agree with analytic distances and surface contacts under rotation") {
        constexpr uint32_t Count = 4096;
        struct Query {
            float3 Position;
            float4 Orientation;
        };
        std::vector<Query> queries(Count);
        std::vector<double> expected(Count);
        std::mt19937 random(70553);
        std::uniform_real_distribution<float> offset(-1.7f, 1.7f), unit(-1.f, 1.f);
        for (uint32_t i = 0; i < Count; ++i) {
            auto &query = queries[i];
            query.Position = {offset(random), offset(random), offset(random)};
            query.Orientation = simd::normalize(float4{unit(random), unit(random), unit(random), unit(random)});
            constexpr float3 aligned[]{{0, 0, 0}, {1, 0, 0}, {1.01f, 0, 0}, {0, 1.4f, 0}, {0, 1.41f, 0}, {0.99f, 1.399f, 0}, {0.999f * 0.9238795325f, 1.3999f, 0.999f * 0.3826834324f}};
            if (i < std::size(aligned)) {
                query.Position = aligned[i];
                query.Orientation = {0, 0, 0, 1};
            }
            const double radial = std::hypot(double(query.Position.x), double(query.Position.z)) - 1;
            const double axial = std::abs(double(query.Position.y)) - double(2 * 0.7f);
            expected[i] = radial > 0 && axial > 0 ? std::hypot(radial, axial) : std::max(radial, axial);
        }

        QueryCylinders(std::span<const Query>(queries), R"(
struct DistanceQuery { float3 Position; float4 Orientation; };
kernel void ProbeCylinderDistance(device const float3 *points [[buffer(0)]], device const DistanceQuery *input [[buffer(1)]],
    device float4 *output [[buffer(2)]], uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    const DistanceQuery q = input[group];
    Poly a{}, b{};
    a.Kind = b.Kind = ShapeCylinder;
    a.Half = b.Half = float3(0.5f,0.7f,0);
    a.Orientation = b.Orientation = q.Orientation;
    b.Center = Rotate(q.Orientation, q.Position);
    float3 direction; float distance;
    const uint query_lane = COLLECT_LANES == 1 ? NoIndex : lane;
    const bool valid = ConvexSeparation(a,b,points,direction,distance,query_lane);
    float3 here[MaxClipPoints],there[MaxClipPoints],normal; uint names[MaxClipPoints];
    const uint found = ConvexManifold(a,b,points,nullptr,0.0005f,float3(0),here,there,names,normal,query_lane);
    float deepest=INFINITY, surface_error=0;
    for(uint i=0;i<found;++i) {
        deepest=min(deepest,dot(here[i]-there[i],normal));
        for(uint side=0;side<2;++side) {
            const Poly shape=side==0?a:b;
            const float3 point=Rotate(QuatConjugate(shape.Orientation),(side==0?here[i]:there[i])-shape.Center);
            const float radial=length(point.xz)-0.5f, axial=abs(point.y)-0.7f;
            surface_error=max(surface_error,max(max(radial,axial),min(abs(radial),abs(axial))));
        }
    }
    if (lane == 0) {
        output[group*2] = float4(direction,valid?distance:INFINITY);
        output[group*2+1] = float4(float(found),deepest,surface_error,0);
    }
}
)",
                       [&](uint32_t i, float4 value, float4 contact) {
                           CHECK(std::abs(value.w - expected[i]) < 1e-4);
                           if (expected[i] < -1e-5) CHECK(contact.x > 0);
                           if (expected[i] > 0.0006) CHECK(contact.x == 0);
                           if (contact.x > 0) CHECK(std::abs(contact.y - expected[i]) < 1e-3);
                       });
    }
    SUBCASE("native cylinder mixed queries agree with frozen Jolt geometry") {
        QueryCylinders(std::span<const CylinderOracleCase>(CylinderOracle), R"(
struct DistanceQuery { float3 Position; float4 Orientation; uint Kind; float Separation; };
kernel void ProbeCylinderDistance(device const float3 *points [[buffer(0)]], device const DistanceQuery *input [[buffer(1)]],
    device float4 *output [[buffer(2)]], uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    const DistanceQuery q = input[group];
    Poly a{}, b{};
    a.Kind = ShapeCylinder;
    a.Half = float3(0.5f,0.7f,0);
    a.Orientation = float4(0,0,0,1);
    b.Kind = q.Kind==0 ? ShapeCylinder : q.Kind==1 ? ShapeBox : q.Kind==2 ? ShapeSphere : ShapeCapsule;
    b.Half = q.Kind==0 ? float3(0.35f,0.6f,0) : q.Kind==1 ? float3(0.4f,0.6f,0.3f) : q.Kind==3 ? float3(0,0.4f,0) : float3(0);
    b.Radius = q.Kind>=2 ? 0.3f : 0;
    b.Orientation = q.Orientation;
    b.Center = q.Position;
    float3 direction; float distance;
    const uint query_lane = COLLECT_LANES == 1 ? NoIndex : lane;
    const bool valid = ConvexSeparation(a,b,points,direction,distance,query_lane);
    float3 here[MaxClipPoints],there[MaxClipPoints],normal; uint names[MaxClipPoints];
    const uint found = ConvexManifold(a,b,points,nullptr,0.0005f,float3(0),here,there,names,normal,query_lane);
    float deepest=INFINITY, surface_error=0;
    for(uint i=0;i<found;++i) {
        deepest=min(deepest,dot(here[i]-there[i],normal));
        for(uint side=0;side<2;++side) {
            const Poly shape=side==0?a:b;
            const float3 point=Rotate(QuatConjugate(shape.Orientation),(side==0?here[i]:there[i])-shape.Center);
            float error;
            if(shape.Kind==ShapeCylinder) {
                const float radial=length(point.xz)-shape.Half.x, axial=abs(point.y)-shape.Half.y;
                error=max(max(radial,axial),min(abs(radial),abs(axial)));
            } else if(shape.Kind==ShapeBox) {
                const float3 gap=abs(point)-shape.Half;
                error=abs(length(max(gap,0.f))+min(max(gap.x,max(gap.y,gap.z)),0.f));
            } else {
                float3 closest=point;
                closest.y-=clamp(point.y,-shape.Half.y,shape.Half.y);
                error=abs(length(closest)-shape.Radius);
            }
            surface_error=max(surface_error,error);
        }
    }
    if (lane == 0) {
        output[group*2] = float4(direction,valid?distance-b.Radius:INFINITY);
        output[group*2+1] = float4(float(found),deepest,surface_error,0);
    }
}
)",
                       [&](uint32_t i, float4 value, float4 contact) {
                           const float expected = CylinderOracle[i].Separation;
                           CHECK(std::abs(value.w - expected) < 1e-4f);
                           if (expected < -1e-5f) CHECK(contact.x > 0);
                           if (expected > 0.0006f) CHECK(contact.x == 0);
                           if (contact.x > 0) CHECK(std::abs(contact.y - expected) < 0.002f);
                       });
    }
}

TEST_CASE("collision: cylinder caps require complete unfiltered coverage") {
    SUBCASE("native cylinder compound caps require complete circular coverage") {
        const mtl::Context context;
        World world{context};
        for (float half_width : {0.48f, 0.5f, 0.6f}) {
            const auto cylinder = world.AddShape({.HalfExtents = {0.5f, 0.7f, 0}, .Kind = ShapeCylinder});
            const auto box = world.AddShape({.HalfExtents = {half_width, 0.1f, 0.6f}, .Kind = ShapeBox, .Local = At(float3{0, 0.8f, 0})});
            const Index children[]{cylinder, box};
            const auto compound = world.AddCompound(children);
            REQUIRE(compound != NoIndex);
            const auto child = ChildOf(world.Shapes[compound], 0, world.CompoundChildren.Data());
            CHECK(InternalFaces(world.Shapes[child]) == (half_width >= 0.5f ? 2u : 0u));
        }
    }
    SUBCASE("native cylinder caps remain solid when a covering sibling is filtered out") {
        const mtl::Context context;
        Solver solver{context};
        for (bool moving_cylinder : {false, true}) {
            CAPTURE(moving_cylinder);
            World world{context};
            const auto cylinder = world.AddShape({.HalfExtents = {0.5f, 0.7f, 0}, .Kind = ShapeCylinder, .Mask = {1, 2}, .HasFilter = 1});
            const auto slab = world.AddShape({.HalfExtents = {0.6f, 0.1f, 0.6f}, .Kind = ShapeBox, .Local = At(float3{0, 0.8f, 0}), .Mask = {4, 8}, .HasFilter = 1});
            const Index children[]{cylinder, slab};
            Pose frame;
            const auto compound = world.AddCompound(children, &frame);
            REQUIRE(compound != NoIndex);
            REQUIRE(InternalFaces(world.Shapes[world.Child(compound, 0)]) == 2);
            const auto body = world.AddBody({.Pose = frame, .Shape = compound, .Density = moving_cylinder ? 1.f : 0.f});
            const auto ball = world.AddBody({.Pose = At(float3{0, 0.75f, 0}), .Shape = world.AddShape({.Radius = 0.1f, .Kind = ShapeSphere}), .Density = moving_cylinder ? 0.f : 1.f, .Layer = 2, .CollidesWith = 1});
            for (uint32_t step = 0; step < 30; ++step) solver.Step(world, {.Gravity = {0, 0, 0}});
            CHECK(world.Filters[body].Mixed == 1);
            if (moving_cylinder) CHECK(world.Poses[body].Position.y < frame.Position.y - 0.045f);
            else CHECK(world.Poses[ball].Position.y > 0.795f);
        }
    }
}
