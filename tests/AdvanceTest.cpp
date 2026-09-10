#include "Shapes.h"
#include "Solver.h"

#include <bit>
#include <stdexcept>
#include <vector>

#include <doctest/doctest.h>

using namespace rbp;

namespace {
using Record = std::vector<uint64_t>;
void Put(Record &r, uint64_t v) { r.push_back(v); }
void Put(Record &r, float v) {
    CHECK(std::isfinite(v));
    r.push_back(std::bit_cast<uint32_t>(v));
}
void Put(Record &r, float3 v) {
    Put(r, v.x);
    Put(r, v.y);
    Put(r, v.z);
}
void Put(Record &r, Pose p) {
    Put(r, p.Position);
    for (uint32_t i = 0; i < 4; ++i) Put(r, p.Orientation[i]);
}
void Put(Record &r, Velocity v) {
    Put(r, v.Linear);
    Put(r, v.Angular);
}
void Put(Record &r, BodyId id) {
    Put(r, uint64_t(id.Slot));
    Put(r, uint64_t(id.Spawn));
}
void Put(Record &r, const ContactSide &v) {
    Put(r, v.InitialPose);
    Put(r, v.Pose);
    Put(r, v.Velocity);
    Put(r, v.Point);
    Put(r, v.Anchor);
    Put(r, v.UserData);
    Put(r, v.InvMass);
}
void Put(Record &r, const ContactChange &v) {
    Put(r, v.A);
    Put(r, v.B);
    Put(r, uint64_t(v.Feature));
    Put(r, uint64_t(v.SubShape));
    Put(r, uint64_t(v.SubShapeA));
    Put(r, v.Children);
    Put(r, uint64_t(v.Kind));
    Put(r, v.Lambda);
    Put(r, v.Approach);
    Put(r, v.BounceImpulse);
    Put(r, v.Step);
    Put(r, v.DeltaTime);
    Put(r, v.SideA);
    Put(r, v.SideB);
    Put(r, v.Normal);
    Put(r, v.Friction);
    Put(r, v.Restitution);
    Put(r, v.NominalArea);
    Put(r, v.NominalExtent);
}
struct Frame {
    Record Bodies, Contacts, Sensors;
    uint64_t Step, ContactRefusals, SensorRefusals;
};
Frame Capture(World &world, const StepResult &result, bool ignore_user_data = false) {
    Frame frame{.Step = result.Step, .ContactRefusals = result.ContactRefusals, .SensorRefusals = result.SensorRefusals};
    for (uint32_t i = 0; i < result.Poses.size(); ++i) {
        Put(frame.Bodies, result.Poses[i]);
        Put(frame.Bodies, result.Velocities[i]);
    }
    for (auto event : world.TakeContactChanges()) {
        if (ignore_user_data) event.SideA.UserData = event.SideB.UserData = 0;
        Put(frame.Contacts, event);
    }
    for (const auto &event : world.TakeSensorChanges()) {
        Put(frame.Sensors, event.Pair.A);
        Put(frame.Sensors, event.Pair.B);
        Put(frame.Sensors, event.Pair.Children);
        Put(frame.Sensors, uint64_t(event.Entered));
    }
    return frame;
}
constexpr WorldLimits Limits{.Bodies = 64, .Shapes = 16, .Joints = 8, .ShapeVertices = 64, .HullFaces = 64, .Triangles = 64, .BvhNodes = 128, .CompoundChildren = 32};
struct Comparison {
    mtl::Context context;
    Solver solver{context};
    World serial, batched;
    uint32_t ContactFrames = 0, SensorFrames = 0;
    explicit Comparison(WorldLimits limits = Limits) : serial(context, limits), batched(context, limits) {}
    AdvanceResult Check(uint32_t count, const StepSettings &settings = {}, std::span<const SensorFollower> followers = {}) {
        std::vector<Frame> expected;
        AdvanceResult ordinary;
        for (uint32_t i = 0; i < count; ++i) {
            const auto result = solver.Advance(serial, settings, 1, followers, [&](const StepResult &step) { expected.push_back(Capture(serial, step));
                ContactFrames += !expected.back().Contacts.empty();
                SensorFrames += !expected.back().Sensors.empty(); });
            ordinary.Steps += result.Steps;
            ordinary.ContactRefusals += result.ContactRefusals;
            ordinary.SensorRefusals += result.SensorRefusals;
        }
        uint32_t seen = 0;
        const auto result = solver.Advance(batched, settings, count, followers, [&](const StepResult &step) {
            REQUIRE(seen < expected.size());
            const auto actual = Capture(batched, step);
            const auto &frame = expected[seen++];
            CAPTURE(seen);
            CHECK(actual.Step == frame.Step);
            CHECK(actual.Bodies == frame.Bodies);
            CHECK(actual.Contacts == frame.Contacts);
            CHECK(actual.Sensors == frame.Sensors);
            CHECK(actual.ContactRefusals == frame.ContactRefusals);
            CHECK(actual.SensorRefusals == frame.SensorRefusals);
        });
        CHECK(seen == expected.size());
        CHECK(result.Steps == ordinary.Steps);
        CHECK(result.ContactRefusals == ordinary.ContactRefusals);
        CHECK(result.SensorRefusals == ordinary.SensorRefusals);
        CHECK(batched.BodyCount() == serial.BodyCount());
        return result;
    }
};
void Scene(World &world) {
    world.TrackContacts = world.TrackSensors = true;
    const auto shape = world.AddShape(UnitBox);
    world.AddBody({.Shape = world.AddShape(GroundPlane), .Density = 0});
    world.AddBody({.Pose = At(float3{0, 0.52f, 0}), .Velocity = {.Linear = {0.1f, -0.2f, 0}}, .Shape = shape, .Restitution = 0.4f});
    world.AddBody({.Pose = At(float3{0.05f, 1.55f, 0}), .Shape = shape});
    const auto pivot = world.AddBody({.Pose = At(float3{2, 2, 0}), .Density = 0});
    const auto arm = world.AddBody({.Pose = At(float3{2, 1, 0}), .Velocity = {.Angular = {0, 0, 0.5f}}, .Shape = shape});
    world.AddJoint({.BodyA = arm, .BodyB = pivot, .At = {2, 2, 0}});
    world.AddBody({.Shape = world.AddShape({.Radius = 0.75f, .Kind = ShapeSphere}), .Density = 0, .Sensor = true});
}
} // namespace

TEST_CASE("advance: complete states and reports match serial execution across schedule boundaries") {
    SUBCASE("advance: complete states and reports survive multiple submissions and a partial batch") {
        Comparison pair;
        Scene(pair.serial);
        Scene(pair.batched);
        const SensorFollower follower{5, 4, IdentityPose};
        const auto result = pair.Check(35, {.DeltaTime = 1.f / 120, .SleepSteps = ~0u}, std::span{&follower, 1});
        CHECK(result.Steps == 35);
        CHECK(result.ContactRefusals == 0);
        CHECK(result.SensorRefusals == 0);
        CHECK_FALSE(pair.batched.Overlaps().empty());
        CHECK(pair.ContactFrames > 0);
        CHECK(pair.SensorFrames > 0);
        pair.serial.TrackContacts = pair.batched.TrackContacts = false;
        pair.Check(7, {.DeltaTime = 1.f / 90, .Iterations = 3, .MaxColors = 1}, std::span{&follower, 1});
        pair.serial.TrackContacts = pair.batched.TrackContacts = true;
        pair.Check(5, {.DeltaTime = 1.f / 240, .Iterations = 0, .MaxColors = 32}, std::span{&follower, 1});
    }
    SUBCASE("advance: small-world schedule changes preserve complete contact and sensor reports") {
        bool hull = false, bounded = false;
        SUBCASE("box on plane") {}
        SUBCASE("hull on plane") { hull = true; }
        SUBCASE("box on bounded plane") { bounded = true; }
        SUBCASE("hull on bounded plane") { hull = bounded = true; }
        WorldLimits limits = Limits;
        limits.Bodies = 520;
        Comparison pair{limits};
        for (World *world : {&pair.serial, &pair.batched}) {
            world->TrackContacts = world->TrackSensors = true;
            Shape plane = GroundPlane;
            if (bounded) plane.HalfExtents = {3, 0, 3};
            world->AddBody({.Shape = world->AddShape(plane), .Density = 0});
            const Index shape = hull ? world->AddHull(PrismPoints(8, Half, Half)) : world->AddShape(UnitBox);
            REQUIRE(shape != NoIndex);
            world->AddBody({.Pose = At(float3{0, 0.55f, 0}), .Velocity = {.Linear = {0.125f, -0.25f, 0}}, .Shape = shape, .Restitution = 0.4f});
            world->AddBody({.Pose = At(float3{0, 0.5f, 0}), .Shape = world->AddShape({.Radius = 2, .Kind = ShapeSphere}), .Density = 0, .Sensor = true});
        }
        const auto pad = [](World &world, uint32_t count) {
            while (world.BodyCount() < count)
                REQUIRE(world.AddBody({.Pose = At(float3{100, 100, 100}), .Density = 0}) != NoIndex);
        };

        pad(pair.serial, 513);
        const auto capture = [](World &world, StepResult step) {
            step.Poses = step.Poses.first(3);
            step.Velocities = step.Velocities.first(3);
            return Capture(world, step);
        };
        uint32_t phase = 0;
        for (uint32_t count : {3u, 9u, 10u, 17u, 32u, 33u, 511u, 512u, 513u, 3u}) {
            CAPTURE(count);
            pad(pair.batched, count);
            for (Index body = pair.batched.BodyCount(); body-- > count;)
                REQUIRE(pair.batched.RemoveBody(body));
            StepSettings settings{.DeltaTime = 1.f / 120, .SleepSteps = 8};
            if (phase++ == 3) {
                for (World *world : {&pair.serial, &pair.batched}) {
                    world->Wake(1);
                    world->Velocities[1].Linear = {0.2f, 0.3f, 0};
                }
            }
            std::vector<Frame> expected;
            for (uint32_t i = 0; i < 17; ++i)
                pair.solver.Advance(pair.serial, settings, 1, {}, [&](const StepResult &step) { expected.push_back(capture(pair.serial, step)); });
            uint32_t seen = 0;
            const auto result = pair.solver.Advance(pair.batched, settings, 17, {}, [&](const StepResult &step) {
                REQUIRE(seen < expected.size());
                const auto actual = capture(pair.batched, step);
                const auto &frame = expected[seen++];
                CHECK(actual.Step == frame.Step);
                CHECK(actual.Bodies == frame.Bodies);
                CHECK(actual.Contacts == frame.Contacts);
                CHECK(actual.Sensors == frame.Sensors);
                CHECK(actual.ContactRefusals == 0);
                CHECK(actual.SensorRefusals == 0);
            });
            CHECK(result.Steps == 17);
            CHECK(seen == 17);
            CHECK(pair.batched.BodyCount() == count);
        }
    }
}

TEST_CASE("advance: geometry edits and retirement invalidate cached queries") {
    mtl::Context context;
    Solver cached_solver{context}, fresh_solver{context};
    World cached{context, Limits}, fresh{context, Limits};
    const float3 points[]{{-2, 0, -2}, {-2, 0, 2}, {2, 0, 2}, {2, 0, -2}};
    const uint32_t indices[]{0, 1, 2, 0, 2, 3};
    for (World *world : {&cached, &fresh}) {
        world->TrackContacts = world->TrackSensors = true;
        const Index mesh = world->AddMesh(points, indices);
        REQUIRE(mesh == 0);
        world->AddBody({.Shape = mesh, .Density = 0});
        const Index box = world->AddShape(UnitBox);
        world->AddBody({.Pose = At(float3{0, Half, 0}), .Shape = box});
        world->AddBody({.Pose = At(float3{0, Half, 0}), .Shape = box, .Density = 0, .Sensor = true});
        for (uint32_t node = 0; node < world->Shapes[mesh].NodeCount; ++node)
            world->BvhNodes[world->Shapes[mesh].RootNode + node].Low.y = -1;
    }
    StepSettings settings{.Gravity = {0, 0, 0}, .SleepSteps = ~0u};
    for (uint32_t step = 0; step < 96; ++step) {
        CAPTURE(step);
        for (World *world : {&cached, &fresh}) {
            if (step == 24) world->Materials[1].StaticFriction = world->Materials[1].DynamicFriction = 0.9f;
            if (step == 32 || step == 40)
                for (uint32_t i = 0; i < 4; ++i) world->ShapeVertices[world->Shapes[0].FirstVertex + i].y = step == 32 ? -0.001f : 0;
            if (step == 64) world->Velocities[1].Linear.y = -1;
            if (step == 72) world->Filters[0].Collides = 0;
            if (step == 80) world->Filters[0].Collides = ~0u;
        }
        if (step == 48) settings.Gravity.y = -9.81f;
        if (step == 88) settings.ContactMargin = 0.001f;

        fresh.Shapes[0].UserData = step + 1;
        Frame actual, expected;
        cached_solver.Advance(cached, settings, 1, {}, [&](const StepResult &result) { actual = Capture(cached, result, true); });
        fresh_solver.Advance(fresh, settings, 1, {}, [&](const StepResult &result) { expected = Capture(fresh, result, true); });
        CHECK(actual.Bodies == expected.Bodies);
        CHECK(actual.Contacts == expected.Contacts);
        CHECK(actual.Sensors == expected.Sensors);
        CHECK(actual.Step == expected.Step);
        CHECK(actual.ContactRefusals == 0);
        CHECK(expected.ContactRefusals == 0);
        CHECK(actual.SensorRefusals == 0);
        CHECK(expected.SensorRefusals == 0);
    }
}

TEST_CASE("advance: transient overlaps and follower ordering remain observable") {
    SUBCASE("advance: an overlap that enters and exits within a batch remains observable") {
        mtl::Context context;
        Solver solver{context};
        World world{context, Limits};
        world.TrackSensors = true;
        const auto shape = world.AddShape({.Radius = 0.25f, .Kind = ShapeSphere});
        world.AddBody({.Pose = At(float3{-2, 0, 0}), .Velocity = {.Linear = {10, 0, 0}}, .Shape = shape});
        world.AddBody({.Shape = shape, .Density = 0, .Sensor = true});
        std::vector<std::pair<uint64_t, bool>> changes;
        solver.Advance(world, {.Gravity = {0, 0, 0}, .DeltaTime = 0.1f}, 4, {}, [&](const StepResult &step) {
            CHECK(step.Poses[0].Position.x == doctest::Approx(-2.f + float(step.Step)));
            for (const auto &event : world.TakeSensorChanges()) changes.emplace_back(step.Step, event.Entered);
        });
        REQUIRE(changes.size() == 2);
        CHECK(changes[0] == std::pair<uint64_t, bool>{2, true});
        CHECK(changes[1] == std::pair<uint64_t, bool>{3, false});
        CHECK(world.Overlaps().empty());
    }
    SUBCASE("advance: follower dependencies are ordered and invalid graphs leave the solver reusable") {
        mtl::Context context;
        Solver solver{context};
        World world{context, Limits};
        const auto shape = world.AddShape(UnitBox);
        const auto owner = world.AddBody({.Velocity = {.Linear = {1, 2, 0}}, .Shape = shape, .CollidesWith = 0});
        const auto first = world.AddBody({.Shape = shape, .Density = 0, .CollidesWith = 0, .Sensor = true});
        const auto second = world.AddBody({.Shape = shape, .Density = 0, .CollidesWith = 0, .Sensor = true});
        std::vector<SensorFollower> followers{{second, first, At(float3{0, 0, 3})}, {first, owner, At(float3{2, 0, 0})}};
        solver.Advance(world, {.Gravity = {0, 0, 0}}, 5, followers, [&](const StepResult &step) {
            CHECK(simd::length(step.Poses[first].Position - step.Poses[owner].Position - float3{2, 0, 0}) < 1e-6f);
            CHECK(simd::length(step.Poses[second].Position - step.Poses[first].Position - float3{0, 0, 3}) < 1e-6f);
        });
        followers[1].Owner = second;
        CHECK_THROWS_AS(solver.Advance(world, {}, 2, followers), std::invalid_argument);
        followers[1] = followers[0];
        CHECK_THROWS_AS(solver.Advance(world, {}, 2, followers), std::invalid_argument);
        followers.resize(1);
        followers[0].Sensor = owner;
        CHECK_THROWS_AS(solver.Advance(world, {}, 2, followers), std::invalid_argument);
        followers[0] = {second, NoIndex, IdentityPose};
        CHECK_THROWS_AS(solver.Advance(world, {}, 2, followers), std::invalid_argument);
        CHECK(solver.Advance(world, {}, 0, followers).Steps == 0);
        CHECK(solver.Advance(world, {.Gravity = {0, 0, 0}}, 1).Steps == 1);
    }
}

TEST_CASE("advance: refusal and memory limits preserve every completed step") {
    SUBCASE("advance: refusals are accumulated across substeps with and without sensors") {
        bool sensors = false;
        SUBCASE("solid contacts") {}
        SUBCASE("sensor pairs") { sensors = true; }
        Comparison pair;
        for (World *world : {&pair.serial, &pair.batched}) {
            world->TrackContacts = world->TrackSensors = true;
            const auto shape = world->AddShape(UnitBox);
            world->AddBody({.Shape = shape, .Density = sensors ? 0.f : 1000.f, .Sensor = sensors});
            for (uint32_t i = 0; i < 50; ++i) world->AddBody({.Pose = At(float3{0, -0.999f, 0}), .Shape = shape, .Density = 0});
        }
        const auto result = pair.Check(3, {.Gravity = {0, 0, 0}, .Iterations = 0});
        CHECK((sensors ? result.SensorRefusals : result.ContactRefusals) > 0);
        CHECK((sensors ? result.ContactRefusals : result.SensorRefusals) == 0);
    }
    SUBCASE("advance: reporting above the batch memory budget preserves every step") {
        auto limits = Limits;
        limits.Bodies = 16 * 1024 * 1024 / (ContactsPerBody * (sizeof(ContactReport) + sizeof(ContactEvent))) + 2;
        Comparison pair{limits};
        for (World *world : {&pair.serial, &pair.batched}) {
            world->TrackContacts = true;
            world->AddBody({.Shape = world->AddShape(GroundPlane), .Density = 0});
            world->AddBody({.Pose = At(float3{0, 0.5f, 0}), .Shape = world->AddShape(UnitBox)});
            while (world->BodyCount() < limits.Bodies) world->AddBody({.Density = 0});
        }
        CHECK(pair.Check(3).Steps == 3);
    }
}

TEST_CASE("advance: observer failures drain completed work and reject reentry") {
    mtl::Context context;
    Solver solver{context};
    World world{context, Limits};
    world.AddBody({.Velocity = {.Linear = {1, 0, 0}}, .Shape = world.AddShape(UnitBox)});
    CHECK_THROWS_AS(solver.Advance(world, {.Gravity = {0, 0, 0}}, 3, {}, [&](const StepResult &) { solver.Step(world); }), std::logic_error);
    uint64_t completed = 0;
    solver.Advance(world, {.Gravity = {0, 0, 0}}, 1, {}, [&](const StepResult &step) { completed = step.Step; });
    CHECK(completed == 4);
    CHECK(world.Poses[0].Position.x == doctest::Approx(4.f / 60));
    REQUIRE(world.RemoveBody(0));
    CHECK(solver.Advance(world, {}, 5).Steps == 1);
    CHECK(world.BodyCount() == 0);
    CHECK(solver.Advance(world, {}, 5).Steps == 0);
}

TEST_CASE("advance: joint topology and waking survive batched execution") {
    Comparison pair{{.Bodies = 64, .Shapes = 1, .Joints = 64}};
    // Binary-exact gravity and timestep isolate joint-solver roundoff in the analytic control.
    const StepSettings settings{.Gravity = {0, -8, 0}, .DeltaTime = 1.f / 64, .SleepSteps = ~0u};
    for (World *world : {&pair.serial, &pair.batched}) {
        const Index shape = world->AddShape(UnitBox);
        for (Index body = 0; body < 64; ++body) {
            REQUIRE(world->AddBody({.Pose = At(float3{float(body) * 2, 10, 0}), .Shape = shape}) == body);
            if (body % 8)
                REQUIRE(world->AddJoint({.BodyA = body - 1, .BodyB = body, .At = {float(body) * 2 - 1, 10, 0}}) != NoIndex);
        }
    }
    uint32_t steps = 0;
    const auto check = [&](uint32_t count, uint32_t iterations = 10) {
        auto selected = settings;
        selected.Iterations = iterations;
        const auto result = pair.Check(count, selected);
        steps += count;
        REQUIRE(result.ContactRefusals == 0);
        REQUIRE(result.SensorRefusals == 0);
        const float speed = settings.Gravity.y * settings.DeltaTime * steps;
        const float height = 10 + settings.Gravity.y * settings.DeltaTime * settings.DeltaTime * float(steps * (steps + 1)) / 2;
        for (Index body = 0; body < pair.serial.BodyCount(); ++body) {
            // Equal free fall preserves joint displacement independently of component scheduling.
            CHECK(std::abs(pair.serial.Poses[body].Position.x - (float(body) * 2)) < 2e-4f);
            CHECK(std::abs(pair.serial.Poses[body].Position.y - (height)) < 2e-4f);
            CHECK(std::abs(pair.serial.Velocities[body].Linear.y - (speed)) < 2e-4f);
            CHECK(simd::length(pair.serial.Velocities[body].Angular) < 2e-4f);
        }
    };
    check(17);
    std::vector<Index> bridges;
    for (World *world : {&pair.serial, &pair.batched}) {
        for (Index body : {8u, 16u, 24u, 32u}) {
            const float3 midpoint = (world->Poses[body - 1].Position + world->Poses[body].Position) * .5f;
            const Index joint = world->AddJoint({.BodyA = body - 1, .BodyB = body, .At = midpoint});
            REQUIRE(joint != NoIndex);
            if (world == &pair.serial) bridges.push_back(joint);
        }
    }
    check(19);
    check(3, CommandIterationLimit);
    check(3, CommandIterationLimit + 1);
    check(3);
    for (World *world : {&pair.serial, &pair.batched})
        for (Index joint : bridges) REQUIRE(world->RemoveJoint(joint));
    check(3);
    for (World *world : {&pair.serial, &pair.batched})
        for (Index body = 64; body-- > 32;) REQUIRE(world->RemoveBody(body));
    check(19);
    CHECK(pair.serial.BodyCount() == 32);
}
