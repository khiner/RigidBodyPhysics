#include "GpuSource.h"
#include "Shapes.h"
#include "Solver.h"

#include <algorithm>
#include <vector>

#include <doctest/doctest.h>

using namespace rbp;

namespace {
std::vector<uint32_t> ReportPairs(const mtl::Context &context, World &world) {
    const uint32_t bodies = world.BodyCount();
    mtl::Buffer<uint32_t> pairs{context.Device.get(), bodies * bodies}, count{context.Device.get(), 1};
    std::ranges::fill(pairs.All(), 0u);
    count[0] = bodies;
    NS::Error *error{};
    auto residency = NS::TransferPtr(context.Device->newResidencySet(mtl::Make<MTL::ResidencySetDescriptor>().get(), &error));
    residency->addAllocation(pairs.Handle.get());
    residency->addAllocation(count.Handle.get());
    residency->commit();
    residency->requestResidency();
    context.Queue->addResidencySet(residency.get());
    auto desc = mtl::Make<MTL4::ArgumentTableDescriptor>();
    desc->setMaxBufferBindCount(14);
    auto table = NS::TransferPtr(context.Device->newArgumentTable(desc.get(), &error));
    table->setAddress(pairs.Address(), 0);
    table->setAddress(count.Address(), 7);
    table->setAddress(world.BroadPhaseNodes.Address(), 13);
    auto pipeline = context.Pipeline(gpu::LayoutSource, "ReportBroadPhasePairs");
    auto allocator = NS::TransferPtr(context.Device->newCommandAllocator());
    auto commands = NS::TransferPtr(context.Device->newCommandBuffer());
    commands->beginCommandBuffer(allocator.get());
    auto *encoder = commands->computeCommandEncoder();
    encoder->setArgumentTable(table.get());
    encoder->setComputePipelineState(pipeline.get());
    encoder->dispatchThreads({bodies, 1, 1}, {std::min(bodies, 32u), 1, 1});
    encoder->endEncoding();
    commands->endCommandBuffer();
    auto done = NS::TransferPtr(context.Device->newSharedEvent());
    const MTL4::CommandBuffer *list[]{commands.get()};
    context.Queue->commit(list, 1);
    context.Queue->signalEvent(done.get(), 1);
    while (!done->waitUntilSignaledValue(1, 1000)) {}
    std::vector<uint32_t> result(pairs.All().begin(), pairs.All().end());
    context.Queue->removeResidencySet(residency.get());
    return result;
}

void CheckTree(const World &world) {
    const uint32_t bodies = world.BodyCount(), root = BroadPhaseRoot(bodies);
    CHECK(world.BroadPhaseNodes[root].Parent == NoIndex);
    if (bodies <= RadixSimdWidth) {
        for (uint32_t body = 0; body < bodies; ++body)
            for (uint32_t axis = 0; axis < 3; ++axis) {
                CHECK((world.BroadPhaseNodes[body].Bounds.Low[axis] == world.Bounds[body].Low[axis]));
                CHECK((world.BroadPhaseNodes[body].Bounds.High[axis] == world.Bounds[body].High[axis]));
            }
        return;
    }
    std::vector<uint32_t> stack{root};
    std::vector<bool> visited(2 * bodies - 1), keyed(bodies);
    for (uint32_t i = 0; i < bodies; ++i) {
        const auto key = world.BroadPhaseKeys[i];
        REQUIRE(key.Body < bodies);
        CHECK_FALSE(keyed[key.Body]);
        keyed[key.Body] = true;
        if (i) {
            const auto previous = world.BroadPhaseKeys[i - 1];
            CHECK((previous.Code < key.Code || (previous.Code == key.Code && previous.Body < key.Body)));
        }
    }
    while (!stack.empty()) {
        const uint32_t at = stack.back();
        stack.pop_back();
        REQUIRE(at < visited.size());
        REQUIRE_FALSE(visited[at]);
        visited[at] = true;
        const auto node = world.BroadPhaseNodes[at];
        if (at < bodies) {
            CHECK((node.Left == NoIndex && node.Right == NoIndex && node.MinBody == at && node.MaxBody == at));
            for (uint32_t axis = 0; axis < 3; ++axis) {
                CHECK((node.Bounds.Low[axis] == world.Bounds[at].Low[axis]));
                CHECK((node.Bounds.High[axis] == world.Bounds[at].High[axis]));
            }
            continue;
        }
        REQUIRE(node.Left < visited.size());
        REQUIRE(node.Right < visited.size());
        const auto a = world.BroadPhaseNodes[node.Left], b = world.BroadPhaseNodes[node.Right];
        CHECK((a.Parent == at && b.Parent == at && node.Ready == 2));
        CHECK((node.MinBody == std::min(a.MinBody, b.MinBody) && node.MaxBody == std::max(a.MaxBody, b.MaxBody)));
        CHECK(a.MinBody < b.MinBody);
        for (uint32_t axis = 0; axis < 3; ++axis) {
            CHECK(node.Bounds.Low[axis] == std::min(a.Bounds.Low[axis], b.Bounds.Low[axis]));
            CHECK(node.Bounds.High[axis] == std::max(a.Bounds.High[axis], b.Bounds.High[axis]));
        }
        stack.push_back(node.Left);
        stack.push_back(node.Right);
    }
    CHECK(std::ranges::all_of(visited, [](bool value) { return value; }));
}

void CheckPairs(const mtl::Context &context, World &world) {
    const auto actual = ReportPairs(context, world);
    const uint32_t bodies = world.BodyCount();
    CHECK(world.BroadPhaseNodes[BroadPhaseRoot(bodies)].Ready == (bodies <= RadixSimdWidth ? 0u : 2u));
    CHECK(world.BroadPhaseNodes[BroadPhaseRoot(bodies)].Errors == 0);
    for (uint32_t a = 0; a < bodies; ++a) {
        std::vector<uint32_t> wanted, found;
        for (uint32_t b = 0; b < bodies; ++b) {
            bool overlap = a != b;
            for (uint32_t axis = 0; axis < 3; ++axis) {
                const auto ba = world.Bounds[a], bb = world.Bounds[b];
                overlap &= ba.Low[axis] <= ba.High[axis] && bb.Low[axis] <= bb.High[axis] &&
                    ba.Low[axis] <= bb.High[axis] && bb.Low[axis] <= ba.High[axis];
            }
            if (overlap) wanted.push_back(b);
            if (actual[a * bodies + b]) found.push_back(b);
        }
        CAPTURE(a);
        CHECK(found == wanted);
    }
}
} // namespace

TEST_CASE("GPU broad phase matches brute force across radix block boundaries") {
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
            CheckTree(world);
            CheckPairs(context, world);
        }
    }
}

TEST_CASE("GPU broad phase preserves coincident and unbounded pairs") {
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
    CheckTree(world);
    CheckPairs(context, world);
}

TEST_CASE("GPU broad phase changes build mode with body lifetime and sensor refits") {
    const mtl::Context context;
    Solver solver{context};
    World world{context, {.Bodies = 300}};
    const auto shape = world.AddShape({.Radius = 0.2f, .Kind = ShapeSphere});
    for (uint32_t count : {32u, 33u, 256u, 257u}) {
        CAPTURE(count);
        while (world.BodyCount() < count) {
            const auto body = world.BodyCount();
            REQUIRE(world.AddBody({.Pose = At(float3{float(body % 8) * 0.3f, float(body / 8) * 0.3f, 0}), .Shape = shape, .Density = 0, .Sensor = body == 0}) == body);
        }
        solver.Step(world);
        CheckTree(world);
        CheckPairs(context, world);
    }
    for (uint32_t count : {256u, 33u, 32u}) {
        CAPTURE(count);
        for (uint32_t body = count; body < world.BodyCount(); ++body) REQUIRE(world.RemoveBody(body));
        solver.Step(world);
        REQUIRE(world.BodyCount() == count);
        solver.Step(world);
        CheckTree(world);
        CheckPairs(context, world);
    }
}

TEST_CASE("cooperative body bounds match scalar bounds through geometry and sensor changes") {
    const mtl::Context context;
    Solver solver{context};
    World cooperative{context, {.Bodies = 40}}, scalar{context, {.Bodies = 40}};
    Index hull = NoIndex;
    for (World *world : {&cooperative, &scalar}) {
        std::vector<Index> shapes{world->AddShape(UnitBox), world->AddShape({.Radius = 0.4f, .Kind = ShapeSphere}), world->AddShape({.HalfExtents = {0, 0.5f, 0}, .Radius = 0.2f, .Kind = ShapeCapsule})};
        hull = world->AddHull(PrismPoints(32, 0.75f, 0.3f), nullptr, At(float3{0.2f, -0.3f, 0.1f}, QuatFromRotationVector(float3{0.2f, 0.4f, -0.1f})));
        REQUIRE(hull != NoIndex);
        shapes.push_back(hull);
        shapes.push_back(BoxMesh(*world, 0.6f));
        shapes.push_back(world->AddCompound(std::vector<Index>{shapes[1], hull}));
        shapes.push_back(world->AddShape(GroundPlane));
        shapes.push_back(NoIndex);
        for (uint32_t body = 0; body < 32; ++body)
            REQUIRE(world->AddBody({.Pose = At(float3{4.f * body, 3, -0.25f * body}, QuatFromRotationVector(float3{0.03f * body, 0.07f, -0.02f})), .Velocity = {.Linear = body == 1 ? float3{0.3f, 0.2f, 0.1f} : float3{0, 0, 0}}, .Shape = shapes[body % shapes.size()], .Density = 0}) == body);
    }
    REQUIRE(scalar.AddBody({.Density = 0}) == 32);
    for (uint32_t step = 0; step < 5; ++step) {
        CAPTURE(step);
        for (World *world : {&cooperative, &scalar}) {
            if (step == 1) REQUIRE(world->RemoveBody(5));
            if (step == 2) REQUIRE(world->AddBody({.Pose = At(float3{25, -2, 4}), .Shape = hull, .Density = 0}) == 5);
            world->Filters[0].Sensor = step == 3;
            if (step == 4) world->Shapes[hull].Local.Position += float3{0.1f, 0.2f, -0.1f};
            solver.Step(*world);
        }
        REQUIRE(cooperative.BodyCount() == 32);
        REQUIRE(scalar.BodyCount() == 33);
        for (uint32_t body = 0; body < 32; ++body) {
            CAPTURE(body);
            for (uint32_t axis = 0; axis < 3; ++axis) {
                CHECK((cooperative.Bounds[body].Low[axis] == scalar.Bounds[body].Low[axis]));
                CHECK((cooperative.Bounds[body].High[axis] == scalar.Bounds[body].High[axis]));
            }
        }
    }
}
