#include "Gpu.h"
#include "GpuSource.h"
#include "Shapes.h"
#include "Solver.h"

#include <algorithm>
#include <numeric>
#include <vector>

#include <doctest/doctest.h>

using namespace rbp;

namespace {
std::vector<uint32_t> ReportPairs(const mtl::Context &context, World &world, bool batched) {
    const uint32_t bodies = world.BodyCount();
    mtl::Buffer<uint32_t> pairs{context.Device.get(), bodies * bodies}, count{context.Device.get(), 1};
    std::ranges::fill(pairs.All(), 0u);
    count[0] = bodies;
    auto pipeline = context.Pipeline(gpu::LayoutSource, batched ? "ReportBatchedBroadPhasePairs" : "ReportBroadPhasePairs");
    RunGpu(context, pipeline.get(), {{0, pairs.Handle.get()}, {7, count.Handle.get()}, {13, world.BroadPhaseNodes.Handle.get()}}, bodies, std::min(bodies, 32u), false);
    std::vector<uint32_t> result(pairs.All().begin(), pairs.All().end());
    return result;
}

void CheckPairs(const mtl::Context &context, World &world) {
    for (bool batched : {false, true}) {
        CAPTURE(batched);
        const auto actual = ReportPairs(context, world, batched);
        const uint32_t bodies = world.BodyCount();
        CHECK(world.BroadPhaseNodes[BroadPhaseRoot(bodies)].Ready == (bodies <= RadixSimdWidth ? 0u : 2u));
        CHECK(world.BroadPhaseNodes[BroadPhaseRoot(bodies)].Errors == 0);
        for (uint32_t a = 0; a < bodies; ++a) {
            std::vector<uint32_t> wanted, found, order;
            for (uint32_t b = 0; b < bodies; ++b) {
                bool overlap = a != b;
                for (uint32_t axis = 0; axis < 3; ++axis) {
                    const auto ba = world.Bounds[a], bb = world.Bounds[b];
                    overlap &= ba.Low[axis] <= ba.High[axis] && bb.Low[axis] <= bb.High[axis] &&
                        ba.Low[axis] <= bb.High[axis] && bb.Low[axis] <= ba.High[axis];
                }
                if (overlap) wanted.push_back(b);
                if (actual[a * bodies + b]) {
                    found.push_back(b);
                    order.push_back(actual[a * bodies + b]);
                }
            }
            CAPTURE(a);
            CHECK(found == wanted);
            std::vector<uint32_t> expected_order(wanted.size());
            std::iota(expected_order.begin(), expected_order.end(), 1u);
            CHECK(order == expected_order);
        }
    }
}
} // namespace

TEST_CASE("collision: broad phase matches brute force across size and motion boundaries") {
    SUBCASE("GPU broad phase matches brute force across radix block boundaries") {
        const mtl::Context context;
        Solver solver{context};
        uint32_t random = 0xabc937u;
        const auto position = [&]() {
            float3 result{};
            for (uint32_t axis = 0; axis < 3; ++axis) {
                random = random * 1664525u + 1013904223u;
                result[axis] = float(int32_t(random >> 24) - 128) * 0.03125f;
            }
            return result;
        };
        for (uint32_t count : {1u, 2u, 31u, 32u, 33u, 255u, 256u, 257u, 769u}) {
            CAPTURE(count);
            World world{context, {.Bodies = count}};
            const auto sphere = world.AddShape({.Radius = 0.2f, .Kind = ShapeSphere});
            const auto plane = world.AddShape({.Normal = {0, 1, 0}, .Kind = ShapePlane});
            for (uint32_t body = 0; body < count; ++body)
                REQUIRE(world.AddBody({.Pose = At(position()), .Shape = body == 0 ? plane : sphere, .Density = 0}) == body);
            for (uint32_t step = 0; step < 3; ++step) {
                if (step == 1 && count > 16) REQUIRE(world.RemoveBody(5));
                if (step == 2)
                    for (uint32_t body = 1; body < count; ++body) world.Poses[body] = At(position());
                solver.Step(world);
                if (step == 0 && count > RadixSimdWidth) CHECK(world.BroadPhaseKeys[0].Code != world.BroadPhaseKeys[count - 2].Code);
                CheckPairs(context, world);
            }
        }
    }
    SUBCASE("GPU broad phase preserves coincident and unbounded pairs") {
        const mtl::Context context;
        Solver solver{context};
        bool planes = false;
        SUBCASE("coincident finite bounds") {}
        SUBCASE("all bodies unbounded") { planes = true; }
        const uint32_t count = planes ? 257 : 65;
        World world{context, {.Bodies = count}};
        const auto shape = world.AddShape(planes ? Shape{.Normal = {0, 1, 0}, .Kind = ShapePlane} : Shape{.Radius = 1, .Kind = ShapeSphere});
        for (uint32_t i = 0; i < count; ++i) REQUIRE(world.AddBody({.Shape = shape, .Density = 0}) == i);
        solver.Step(world);
        CheckPairs(context, world);
    }
    SUBCASE("GPU broad phase refits moving bounds against brute force across batched steps") {
        const mtl::Context context;
        Solver solver{context};
        const StepSettings settings{.Gravity = {0, 0, 0}, .DeltaTime = 1.f / 64, .SleepSteps = ~0u};
        for (uint32_t count : {257u, 769u})
            for (bool sensors : {false, true}) {
                CAPTURE(count);
                CAPTURE(sensors);
                World world{context, {.Bodies = count}};
                const auto sphere = world.AddShape({.Radius = .125f, .Kind = ShapeSphere});
                REQUIRE(world.AddBody({.Shape = world.AddShape(GroundPlane), .Density = 0}) == 0);
                for (uint32_t body = 1; body < count; ++body)
                    REQUIRE(world.AddBody({.Pose = At(float3{float((body * 37) % 64) * .125f, 2.f + float(body % 32), float(body / 32)}), .Velocity = {.Linear = {body % 2 ? 128.f : -128.f, 0, 0}}, .Shape = sphere, .Density = 0, .Sensor = sensors && body == 1}) == body);
                for (uint32_t steps : {2u, 10u, 17u}) {
                    const auto result = solver.Advance(world, settings, steps);
                    REQUIRE(result.Steps == steps);
                    CHECK(result.ContactRefusals == 0);
                    CHECK(result.SensorRefusals == 0);
                    CheckPairs(context, world);
                }
            }
    }
}

TEST_CASE("collision: bounds include incoming speed before damping and waking") {
    const mtl::Context context;
    Solver solver{context};
    const StepSettings settings{.Gravity = {0, 0, 0}, .DeltaTime = .125f, .Iterations = 0, .ContactMargin = 0};
    for (uint32_t count : {32u, 33u})
        for (bool mesh : {false, true})
            for (bool sleeping : {false, true}) {
                CAPTURE(count);
                CAPTURE(mesh);
                CAPTURE(sleeping);
                World world{context, {.Bodies = count}};
                const Index sphere = world.AddShape({.Radius = 1, .Kind = ShapeSphere});
                REQUIRE(world.AddBody({.Velocity = {.Linear = {4, 0, 0}}, .Shape = sphere, .LinearDamping = 4}) == 0);
                if (mesh) REQUIRE(world.AddBody({.Pose = At(float3{100, 0, 0}), .Shape = BoxMesh(world, .5f), .Density = 0}) == 1);
                while (world.BodyCount() < count) REQUIRE(world.AddBody({.Density = 0}) != NoIndex);
                if (sleeping) world.Quiet[0] = settings.SleepSteps;
                solver.Step(world, settings);
                // Radius 1, roundoff 1e-5, and incoming speculative reach h |v| = .5.
                // Search bounds retain the initial speculative reach after damping halves velocity.
                for (uint32_t axis = 0; axis < 3; ++axis) {
                    CHECK(std::abs(world.Bounds[0].Low[axis] + 1.50001f) < 2e-6f);
                    CHECK(std::abs(world.Bounds[0].High[axis] - 1.50001f) < 2e-6f);
                }
                CHECK(world.InitialPoses[0].Position.x == 0);
                CHECK(world.Poses[0].Position.x == .25f);
                CHECK(world.Velocities[0].Linear.x == 2);
                CHECK(world.Quiet[0] == 0);
            }
}
