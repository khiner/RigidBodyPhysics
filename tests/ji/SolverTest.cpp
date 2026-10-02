#include "Scene.h"
#include "../Shapes.h"
#include "Pipelines.h"
#include "metal/Buffer.h"

#include <doctest/doctest.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <semaphore>

using namespace rbp;
using namespace ji;

namespace {

struct OnDevice {
    const mtl::Context context;
    World world{context};
    Scene scene{context};
    SceneSettings settings;

    OnDevice() {
        settings.dt = 0.005;
        settings.contact_capacity = 128;
        settings.contact_error_reduction = settings.joint_error_reduction = 0;
        settings.solve = {.iterations = 120, .penalty_period = 30};
    }
    Index Floor(float friction = 0.6f) {
        return world.AddBody({.Shape = world.AddShape(GroundPlane), .Friction = friction});
    }
    Index Ball(float3 at = {0, Half, 0}, float mass = 1) {
        return world.AddBody({.Pose = At(at),
            .Shape = world.AddShape({.Radius = Half, .Kind = ShapeSphere}),
            .Mass = AuthoredMass{mass, {0.1f * mass, 0.1f * mass, 0.1f * mass}}, .Friction = 0});
    }
};

float3 Vector(Vec3 value) { return {float(value[0]), float(value[1]), float(value[2])}; }

float3 Impulse(const SceneContact &contact, Index body) {
    return Vector(contact.impulse_on_a) * (contact.key.a.Slot == body ? 1.f : -1.f);
}

void CheckState(const World &a, const World &b, float tolerance = 0) {
    REQUIRE(a.BodyCount() == b.BodyCount());
    for (Index body = 0; body < a.BodyCount(); ++body) {
        CAPTURE(body);
        CHECK(length(a.Poses[body].Position - b.Poses[body].Position) <= tolerance);
        CHECK(length(a.Poses[body].Orientation - b.Poses[body].Orientation) <= tolerance);
        CHECK(length(a.Velocities[body].Linear - b.Velocities[body].Linear) <= tolerance);
        CHECK(length(a.Velocities[body].Angular - b.Velocities[body].Angular) <= tolerance);
    }
}

Body UnitBody(Vec rhs = {}) {
    Body body;
    body.a.fill(1);
    body.b = rhs;
    body.beta = 1;
    return body;
}

System MixedSystem() {
    System system{{UnitBody({-1, 2, 0, 0.3}), UnitBody({0.3, 0, 0, -0.2})}, {}};
    Constraint contact{.kind = Kind::Contact, .rows = 3};
    contact.side[0].body = 0;
    for (uint32_t row = 0; row < 3; ++row) contact.side[0].j[7 * row] = 1;
    contact.friction = 0.5;
    system.constraints.push_back(contact);
    Constraint joint{.kind = Kind::Hard, .rows = 6};
    joint.side[0].body = 0;
    joint.side[1].body = 1;
    for (uint32_t row : {0u, 3u}) {
        joint.side[0].j[7 * row] = 1;
        joint.side[1].j[7 * row] = -1;
    }
    system.constraints.push_back(joint);
    Constraint spring{.kind = Kind::Soft, .rows = 6};
    spring.side[0].body = 0;
    spring.side[1].body = 1;
    spring.side[0].j[7] = 1;
    spring.side[1].j[7] = -1;
    spring.error[1] = 0.03;
    spring.stiffness = 2;
    spring.damping = 0.1;
    system.constraints.push_back(spring);
    return system;
}

} // namespace

TEST_CASE_FIXTURE(OnDevice, "Ji: free motion obeys six-DOF discrete dynamics") {
    const float4 rotation = QuatFromRotationVector(float3{0.2f, -0.1f, 0.3f});
    const float3 inertia{0.2f, 0.3f, 0.4f}, omega{0.3f, 0.2f, -0.1f};
    const float3 linear{1, -0.4f, 0.25f}, position{0.5f, 1, -0.2f};
    const Index body = world.AddBody({.Pose = At(position, rotation),
        .Velocity = {.Linear = Rotate(rotation, linear), .Angular = Rotate(rotation, omega)},
        .Shape = world.AddShape(UnitBox), .Mass = AuthoredMass{2, inertia}, .GravityScale = 0.5f});
    settings.dt = 0.01;
    settings.capture_initial_system = true;
    settings.solve.iterations = 2;
    const Wrench wrench{{1, 2, -3}, {0.2, -0.1, 0.4}};
    const float3 force = Rotate(QuatConjugate(rotation), Vector(wrench.force) + float3{0, -9.81f, 0});
    const float3 torque = Rotate(QuatConjugate(rotation), Vector(wrench.torque));
    // Body-coordinate translation includes the rotating-frame momentum term.
    const float3 expected_v = linear + float(settings.dt) * (force / 2 - cross(omega, linear));
    const float3 expected_w = omega + float(settings.dt) * (torque - cross(omega, inertia * omega)) / inertia;
    const auto result = scene.Step(world, settings, std::span{&wrench, 1});
    REQUIRE(result.initial_system.has_value());
    CHECK(result.contacts.empty());
    const auto &source = result.initial_system->bodies.at(0);
    for (uint32_t axis = 0; axis < 3; ++axis) {
        CHECK(source.a[axis] == doctest::Approx(2));
        CHECK(source.a[axis + 3] == doctest::Approx(inertia[axis]).epsilon(2e-6));
        CHECK(source.b[axis] == doctest::Approx(2 * expected_v[axis]).epsilon(2e-6));
        CHECK(source.b[axis + 3] == doctest::Approx(inertia[axis] * expected_w[axis]).epsilon(2e-6));
    }
    const float4 next_rotation = normalize(QuatMul(rotation, QuatFromRotationVector(float(settings.dt) * expected_w)));
    CHECK(length(world.Poses[body].Position - position - float(settings.dt) * Rotate(rotation, expected_v)) < 2e-7f);
    CHECK(length(world.Poses[body].Orientation - next_rotation) < 2e-7f);
    CHECK(length(world.Velocities[body].Linear - Rotate(next_rotation, expected_v)) < 2e-6f);
    CHECK(length(world.Velocities[body].Angular - Rotate(next_rotation, expected_w)) < 2e-6f);
}

TEST_CASE_FIXTURE(OnDevice, "Ji: contact separates, supports, sticks and slides") {
    SUBCASE("normal impulse respects separation, look-ahead gap and stabilization") {
        Floor(0);
        const Index ball = Ball();
        settings.collision_margin = 0.002;
        settings.max_contact_reach = 0.05;
        // Expected post-step normal speeds in m/s. Positive gap permits free fall
        // until the next step would cross the plane; penetration uses 10% repair.
        struct Trial { float gap, speed; double repair, expected; };
        for (const Trial trial : {Trial{0, 0.1f, 0, 0.1 - 9.81 * 0.005},
             {0.001f, 0, 0, -9.81 * 0.005}, {0.0001f, 0, 0, -0.02},
             {-0.001f, 0, 0.1, 0.02}}) {
            CAPTURE(trial.gap);
            CAPTURE(trial.speed);
            Scene fresh{context};
            world.Poses[ball] = At(float3{0, Half + trial.gap, 0});
            world.Velocities[ball] = {.Linear = {0, trial.speed, 0}};
            settings.contact_error_reduction = trial.repair;
            const auto result = fresh.Step(world, settings);
            REQUIRE(result.contacts.size() == 1);
            const double expected_impulse = trial.expected - trial.speed + 9.81 * settings.dt;
            CHECK(std::abs(world.Velocities[ball].Linear.y - trial.expected) < 2e-5);
            CHECK(std::abs(Impulse(result.contacts[0], ball).y - expected_impulse) < 2e-5);
        }
    }
    SUBCASE("Coulomb friction follows applied load and dissipates sliding energy") {
        Floor();
        const Index box = world.AddBody({.Pose = At(float3{0, 0.2f, 0}),
            .Shape = world.AddShape({.HalfExtents = {0.2f, 0.2f, 0.2f}, .Kind = ShapeBox}),
            .Mass = AuthoredMass{1, {0.02666667f, 0.02666667f, 0.02666667f}}, .Friction = 0.6f});
        // Under load: static friction, boundary, and established sliding.
        for (const auto [speed, drive] : {std::pair{0.f, 2.f}, {0.f, 5.886f}, {0.4f, 0.f}}) {
            CAPTURE(speed);
            CAPTURE(drive);
            Scene fresh{context};
            world.Poses[box] = At(float3{0, 0.2f, 0});
            world.Velocities[box] = {.Linear = {speed, 0, 0}};
            std::vector<Wrench> loads(world.BodyCount());
            loads[box].force[0] = drive;
            const double expected_x = std::max(0.0, speed + settings.dt * (drive - 0.6 * 9.81));
            const auto result = fresh.Step(world, settings, loads);
            REQUIRE(result.contacts.size() == 4);
            float3 impulse{};
            for (const auto &contact : result.contacts) {
                const float3 value = Impulse(contact, box);
                CHECK(std::hypot(value.x, value.z) <= 0.6 * value.y + 2e-5);
                impulse += value;
            }
            CHECK(std::abs(impulse.y - settings.dt * 9.81) < 5e-4); // N s
            CHECK(std::abs(world.Velocities[box].Linear.x - expected_x) < 5e-4); // m/s
            CHECK(std::abs(impulse.x - (expected_x - speed - settings.dt * drive)) < 5e-4);
            if (speed > 0) {
                for (uint32_t step = 0; step < 5; ++step) fresh.Advance(world, settings);
                CHECK(world.Velocities[box].Linear.x < expected_x - 0.1);
                CHECK(std::abs(world.Poses[box].Position.y - 0.2f) < 1e-4f);
            }
        }
    }
}

TEST_CASE_FIXTURE(OnDevice, "Ji: contact impulses account for linear and angular momentum") {
    settings.gravity = {};
    settings.penalty_mass = 0;
    settings.penalty_offset = 1 / (settings.dt * settings.dt);
    settings.solve = {.iterations = 200};
    SUBCASE("two moving bodies exchange equal and opposite impulses") {
        const Index a = Ball({-Half, 0, 0}, 2), b = Ball({Half, 0, 0}, 3);
        world.Velocities[a].Linear = {1, 0, 0};
        world.Velocities[b].Linear = {-1, 0, 0};
        const auto result = scene.Step(world, settings);
        REQUIRE(result.contacts.size() == 1);
        const double va = world.Velocities[a].Linear.x, vb = world.Velocities[b].Linear.x;
        CHECK(std::abs(2 * va + 3 * vb + 1) < 1e-3);
        CHECK(std::abs(va + 0.2) < 4e-3);
        CHECK(std::abs(vb + 0.2) < 4e-3);
        CHECK(std::abs(Impulse(result.contacts[0], a).x - 2 * (va - 1)) < 1e-3);
        CHECK(std::abs(Impulse(result.contacts[0], b).x - 3 * (vb + 1)) < 1e-3);
        CHECK(2 * va * va + 3 * vb * vb <= 5);
    }
    SUBCASE("off-center impulse has the reported moment arm") {
        Floor(0);
        const float4 rotation = QuatFromRotationVector(float3{0.2f, 0, 0.3f});
        float lowest = 0;
        for (uint32_t i = 0; i < 8; ++i)
            lowest = std::min(lowest, Rotate(rotation, Half * CornerSign(i)).y);
        const float3 center{0, -lowest, 0};
        const Index box = world.AddBody({.Pose = At(center, rotation),
            .Velocity = {.Linear = {0, -1, 0}}, .Shape = world.AddShape(UnitBox),
            .Mass = AuthoredMass{2, {1, 1, 1}}, .Friction = 0});
        settings.collision_margin = 1e-5;
        settings.capture_initial_system = true;
        const auto result = scene.Step(world, settings);
        REQUIRE(!result.contacts.empty());
        float3 impulse{}, moment{};
        for (const auto &contact : result.contacts) {
            const float3 applied = Impulse(contact, box);
            impulse += applied;
            moment += cross(Vector(contact.key.a.Slot == box ? contact.point_a : contact.point_b) - center, applied);
            CHECK(length(Vector(contact.force_on_a) * float(settings.dt) - Vector(contact.impulse_on_a)) < 1e-6f);
        }
        REQUIRE(length(moment) > 0.01f);
        // The solver updates body-frame velocities; integration rotates the stored
        // world velocity into the new pose. Audit the impulse at the source pose.
        const auto &v = result.solve.system.bodies[0].v;
        const float3 linear = Rotate(rotation, float3{float(v[0]), float(v[1]), float(v[2])});
        const float3 angular = Rotate(rotation, float3{float(v[3]), float(v[4]), float(v[5])});
        CHECK(length(2 * (linear - float3{0, -1, 0}) - impulse) < 2e-4f);
        CHECK(length(angular - moment) < 2e-4f);
        CHECK(2 * dot(linear, linear) + dot(angular, angular) <= 2.0001f);
    }
}

TEST_CASE_FIXTURE(OnDevice, "Ji: support loads and large contact incidence remain complete") {
    SUBCASE("a small stack carries the weight above it") {
        Floor(0);
        const Index shape = world.AddShape(UnitBox);
        std::array<Index, 3> boxes;
        for (uint32_t i = 0; i < boxes.size(); ++i)
            boxes[i] = world.AddBody({.Pose = At(float3{0, Half + i, 0}), .Shape = shape,
                .Mass = AuthoredMass{1, {1, 1, 1}}, .Friction = 0});
        for (uint32_t step = 0; step < 8; ++step) {
            const auto result = scene.Step(world, settings);
            float floor_impulse = 0;
            for (const auto &contact : result.contacts)
                if (contact.key.a.Slot == 0 || contact.key.b.Slot == 0)
                    floor_impulse -= Impulse(contact, 0).y;
            CHECK(std::abs(floor_impulse - 3 * 9.81 * settings.dt) < 2e-3);
            for (uint32_t i = 0; i < boxes.size(); ++i) {
                CHECK(std::abs(world.Poses[boxes[i]].Position.y - (Half + i)) < 2e-4f);
                CHECK(length(world.Velocities[boxes[i]].Linear) < 2e-3f);
            }
        }
    }
    SUBCASE("68 features cross 32-lane tiles, 64-row reductions and per-body limits") {
        for (uint32_t i = 0; i < 17; ++i) Floor(0);
        const Index box = world.AddBody({.Pose = At(float3{0, Half, 0}),
            .Shape = world.AddShape(UnitBox), .Mass = AuthoredMass{1, {1, 1, 1}}});
        settings.contact_capacity = 64;
        CHECK_THROWS_AS(scene.Step(world, settings), std::overflow_error);
        CHECK(world.Poses[box].Position.y == Half);
        settings.contact_capacity = 128;
        for (uint32_t step = 0; step < 2; ++step) {
            const auto result = scene.Step(world, settings);
            CHECK(result.step == step + 1);
            REQUIRE(result.contacts.size() == 68);
            CHECK(result.geometry_features == 68);
            CHECK(result.warm_started == (step ? 68 : 0));
            float total = 0;
            for (const auto &contact : result.contacts) total += Impulse(contact, box).y;
            CHECK(std::abs(total - 9.81 * settings.dt) < 5e-4);
            CHECK(std::abs(world.Velocities[box].Linear.y) < 5e-4f);
        }
    }
}

TEST_CASE_FIXTURE(OnDevice, "Ji: joint masks and frames preserve the intended freedoms") {
    SUBCASE("six joint masks constrain axes in a rotated frame") {
        const float4 rotation = QuatFromRotationVector(float3{0.25f, -0.2f, 0.3f});
        const float3 linear{1, 2, 3}, angular{0.3f, 0.4f, 0.5f};
        for (const uint32_t mask : {0x3fu, 0x3eu, 0x1cu, 0x38u, 0x07u, 0x1eu}) {
            CAPTURE(mask);
            World trial{context};
            Scene solver{context};
            const Index pivot = trial.AddBody({.Pose = At(float3{}, rotation)});
            const Index body = trial.AddBody({.Pose = At(float3{}, rotation),
                .Velocity = {.Linear = Rotate(rotation, linear), .Angular = Rotate(rotation, angular)},
                .Shape = trial.AddShape(UnitBox), .Mass = AuthoredMass{1, {1, 1, 1}}});
            JointDesc joint;
            joint.BodyA = body; joint.BodyB = pivot; joint.Frame = rotation;
            for (uint32_t axis = 0; axis < 3; ++axis) {
                joint.Linear[axis] = mask & (1u << axis) ? AxisLocked : AxisFree;
                joint.Angular[axis] = mask & (1u << (axis + 3)) ? AxisLocked : AxisFree;
            }
            REQUIRE(trial.AddJoint(joint) != NoIndex);
            settings.gravity = {};
            settings.penalty_mass = 0;
            settings.penalty_offset = 1 / (settings.dt * settings.dt);
            settings.solve = {.iterations = 200};
            settings.capture_initial_system = true;
            const auto result = solver.Step(trial, settings);
            REQUIRE(result.initial_system.has_value());
            const auto &rows = result.initial_system->constraints.at(0);
            const auto &v = result.solve.system.bodies[0].v;
            const float3 free_linear = linear - float(settings.dt) * cross(angular, linear);
            for (uint32_t row = 0; row < 6; ++row) {
                const double initial = row < 3 ? free_linear[row] : angular[row - 3];
                CHECK(std::abs(v[row] - ((mask & (1u << row)) ? 0 : initial)) < 2e-3);
                CHECK(std::abs(rows.side[0].j[7 * row] - ((mask & (1u << row)) ? settings.dt : 0)) < 1e-7);
            }
        }
    }
    SUBCASE("an unrelated supported body does not pin a free articulated group") {
        Floor(0);
        Ball({2, Half, 0});
        const Index a = Ball({0, 2, 0}), b = Ball({0, 3.2f, 0});
        REQUIRE(world.AddJoint({.BodyA = a, .BodyB = b, .At = {0, 2.6f, 0}}) != NoIndex);
        for (uint32_t step = 1; step <= 5; ++step) {
            const auto result = scene.Step(world, settings);
            REQUIRE(!result.contacts.empty());
            CHECK(result.audit.hard < 1e-5);
            for (Index body : {a, b})
                CHECK(std::abs(world.Velocities[body].Linear.y + step * 9.81 * settings.dt) < 2e-5);
        }
    }
}

TEST_CASE_FIXTURE(OnDevice, "Ji: soft rows obey spring-damper laws alongside hard rows") {
    // Eight prescribed-endpoint controls cover all gain branches and dt scaling.
    for (const auto [stiffness, damping] : {std::pair{0., 3.}, {6., 0.}, {6., 3.}, {0., 0.}})
        for (const double dt : {0.01, 0.04}) {
            CAPTURE(stiffness);
            CAPTURE(damping);
            CAPTURE(dt);
            World trial{context};
            Scene solver{context};
            const Index pivot = trial.AddBody({.Velocity = {.Linear = {-0.4f, 0.25f, 0}}});
            const Index body = trial.AddBody({.Velocity = {.Linear = {1.2f, 0.7f, 0}},
                .Shape = trial.AddShape(UnitBox), .Mass = AuthoredMass{2, {1, 1, 1}}});
            JointDesc joint;
            joint.BodyA = body; joint.BodyB = pivot;
            joint.Linear[2] = AxisFree;
            joint.LinearStiffness.x = float(stiffness);
            joint.LinearDamping.x = float(damping);
            const Index id = trial.AddJoint(joint);
            REQUIRE(id != NoIndex);
            trial.Poses[body].Position.x = 0.03f;
            settings.dt = dt; settings.gravity = {};
            settings.solve = {.iterations = 300};
            settings.penalty_mass = 0; settings.penalty_offset = 1 / (dt * dt);
            const auto result = solver.Step(trial, settings);
            const double expected = (2 * 1.2 - dt * stiffness * 0.03 - dt * damping * 0.4) / (2 + dt * damping);
            const double force = -stiffness * 0.03 - damping * (expected + 0.4);
            REQUIRE(result.solve.system.constraints.size() == 2);
            const auto &soft = result.solve.system.constraints[0];
            CHECK(soft.kind == Kind::Soft);
            CHECK(std::abs(soft.error[0] - dt * 0.03) < 1e-8);
            CHECK(std::abs(soft.velocity_bias[0] - dt * 0.4) < 1e-8);
            CHECK(std::abs(soft.impulse[0] - force) < 2e-5); // N
            CHECK(std::abs(2 * (trial.Velocities[body].Linear.x - 1.2) - dt * force) < 2e-6); // N s
            CHECK(std::abs(trial.Velocities[body].Linear.y - 0.25) < 2e-6);
            CHECK(result.audit.hard < 2e-6);
            CHECK(result.audit.soft < 2e-5);
            // Removing the row must immediately release its force and constraints.
            trial.Joints[id].Active = false;
            const auto before = trial.Velocities[body];
            const auto released = solver.Step(trial, settings);
            CHECK(released.solve.system.constraints.empty());
            CHECK(length(trial.Velocities[body].Linear - before.Linear) < 2e-6f);
        }

    // Coupled moving endpoints must exchange equal/opposite reactions even
    // while normal contact supplies support. Their soft axis is frictionless.
    Floor(0);
    const Index a = Ball({0, Half, 0}, 2), b = Ball({2, Half, 0}, 4);
    world.Velocities[a] = {.Linear = {1.2f, -0.3f, 0}};
    world.Velocities[b] = {.Linear = {-0.4f, -0.3f, 0}};
    JointDesc joint;
    joint.BodyA = a; joint.BodyB = b; joint.At = {1, Half, 0};
    joint.Linear[1] = AxisFree;
    for (auto &axis : joint.Angular) axis = AxisFree;
    joint.LinearStiffness.x = 6; joint.LinearDamping.x = 3;
    REQUIRE(world.AddJoint(joint) != NoIndex);
    world.Poses[a].Position.x = 0.03f;
    settings.dt = 0.02; settings.gravity = {0, -9.81, 0};
    settings.penalty_offset = 1 / (settings.dt * settings.dt);
    const double relative = (1.6 - settings.dt * 6 * 0.03 * 0.75) / (1 + settings.dt * 3 * 0.75);
    const double force = -6 * 0.03 - 3 * relative;
    const auto result = scene.Step(world, settings);
    REQUIRE(result.contacts.size() == 2);
    CHECK(result.solve.system.constraints.size() == 4); // two contacts, soft x, hard z
    CHECK(std::abs(world.Velocities[a].Linear.x - (1.2 + settings.dt * force / 2)) < 2e-5);
    CHECK(std::abs(world.Velocities[b].Linear.x - (-0.4 - settings.dt * force / 4)) < 2e-5);
    CHECK(std::abs(2 * world.Velocities[a].Linear.x + 4 * world.Velocities[b].Linear.x - 0.8) < 2e-5);
    CHECK(result.audit.soft < 2e-5);
    CHECK(result.audit.hard < 2e-6);
    CHECK(result.audit.dynamics < 2e-6);
    for (Index body : {a, b}) CHECK(std::abs(world.Velocities[body].Linear.y) < 2e-5);
}

TEST_CASE_FIXTURE(OnDevice, "Ji: warm contacts follow geometry and endpoint lifetimes") {
    SUBCASE("rotation, lift, geometric return and reuse preserve cache identity") {
        const Index floor = Floor(0), ball = Ball();
        settings.dt = 0.02;
        settings.max_contact_reach = 0.001;
        const auto supported = scene.Step(world, settings);
        REQUIRE(supported.contacts.size() == 1);
        CHECK(supported.warm_started == 0);
        settings.contact_capacity *= 2;
        world.Poses[floor].Orientation = QuatFromRotationVector(float3{0, 0, 0.1f});
        const auto rotated = scene.Step(world, settings);
        REQUIRE(rotated.contacts.size() == 1);
        CHECK(rotated.warm_started == 1);
        CHECK(rotated.contacts[0].normal[0] < -0.09);
        world.Poses[floor].Orientation = {0, 0, 0, 1};
        world.Poses[ball] = At(float3{0, Half, 0});
        world.Velocities[ball] = {};
        std::vector<Wrench> lift(world.BodyCount());
        lift[ball].force = {0, 40, 0};
        const auto departing = scene.Step(world, settings, lift);
        REQUIRE(departing.contacts.size() == 1);
        CHECK(length(Impulse(departing.contacts[0], ball)) < 1e-5f);
        bool absent = false, returned = false;
        for (uint32_t step = 0; step < 16 && !returned; ++step) {
            const auto before = world.Velocities[ball];
            const auto result = scene.Step(world, settings);
            if (result.contacts.empty()) { absent = true; continue; }
            if (!absent) continue;
            REQUIRE(result.contacts.size() == 1);
            CHECK(result.warm_started == 0);
            CHECK(before.Linear.y < 0);
            CHECK(Impulse(result.contacts[0], ball).y > 0.001f);
            CHECK(std::abs(Impulse(result.contacts[0], ball).y -
                (world.Velocities[ball].Linear.y - before.Linear.y + settings.dt * 9.81)) < 2e-5);
            returned = true;
        }
        REQUIRE(absent); REQUIRE(returned);
        const Index replacement_shape = world.AddShape({.Radius = Half, .Kind = ShapeSphere});
        REQUIRE(world.SetBodyShape(ball, replacement_shape, 1000, AuthoredMass{1, {0.1f, 0.1f, 0.1f}}));
        CHECK(scene.Step(world, settings).warm_started == 0);
        const auto former = world.IdOf(ball);
        REQUIRE(world.RemoveBody(ball));
        CHECK(scene.Step(world, settings).contacts.empty());
        const Index replacement = Ball();
        REQUIRE(replacement == ball);
        CHECK(world.IdOf(replacement) != former);
        const auto result = scene.Step(world, settings);
        REQUIRE(result.contacts.size() == 1);
        CHECK(result.warm_started == 0);
    }
    SUBCASE("changing dynamic sides invalidates the same geometric contact") {
        const Index a = Ball({-Half, 0, 0}, 2), b = Ball({Half, 0, 0}, 3);
        const BodyMass masses[]{world.Masses[a], world.Masses[b]};
        settings.gravity = {};
        settings.penalty_mass = 0; settings.penalty_offset = 1;
        settings.solve = {.iterations = 200};
        settings.capture_initial_system = true;
        uint32_t previous = 0;
        ContactKey previous_key{};
        for (uint32_t mask : {3u, 3u, 1u, 1u, 2u, 2u}) {
            CAPTURE(mask);
            world.Masses[a] = mask & 1u ? masses[0] : StaticMass;
            world.Masses[b] = mask & 2u ? masses[1] : StaticMass;
            world.Poses[a] = At(float3{-Half, 0, 0});
            world.Poses[b] = At(float3{Half, 0, 0});
            world.Velocities[a] = {.Linear = {1, 0, 0}};
            world.Velocities[b] = {.Linear = {-1, 0, 0}};
            const auto result = scene.Step(world, settings);
            REQUIRE(result.contacts.size() == 1);
            CHECK(result.warm_started == uint32_t(mask == previous));
            REQUIRE(result.initial_system.has_value());
            const double seed = result.initial_system->constraints.at(0).impulse[0];
            if (mask != previous) CHECK(seed == 0);
            else CHECK(seed > 0);
            if (previous == 3 && mask == 1) {
                CHECK_FALSE(result.contacts[0].key < previous_key);
                CHECK_FALSE(previous_key < result.contacts[0].key);
            }
            previous = mask;
            previous_key = result.contacts[0].key;
        }
    }
}

TEST_CASE_FIXTURE(OnDevice, "Ji: stepping, observation and failed-step recovery agree") {
    World control{context};
    Scene compact{context};
    const auto populate = [](World &target) {
        target.AddBody({.Shape = target.AddShape(GroundPlane)});
        target.AddBody({.Pose = At(float3{0, Half, 0}), .Velocity = {.Linear = {0.2f, 0, 0}},
            .Shape = target.AddShape({.Radius = Half, .Kind = ShapeSphere}),
            .Mass = AuthoredMass{1, {0.1f, 0.1f, 0.1f}}});
    };
    populate(world); populate(control);
    JointDesc unsupported;
    unsupported.BodyA = 1; unsupported.BodyB = 0; unsupported.Collide = true;
    unsupported.Drives[0].Enabled = 1;
    const Index joint = world.AddJoint(unsupported), control_joint = control.AddJoint(unsupported);
    REQUIRE(joint != NoIndex);
    REQUIRE(control_joint != NoIndex);
    world.Joints[joint].Active = control.Joints[control_joint].Active = false;
    for (uint32_t step = 0; step < 8; ++step) {
        CAPTURE(step);
        if (step == 2 || step == 4) {
            auto denied = settings;
            if (step == 2) denied.dt = std::numeric_limits<double>::quiet_NaN();
            else denied.solve = {.iterations = 1, .tolerance = 1e-12};
            CHECK_THROWS(scene.Step(world, denied));
            CHECK_THROWS(scene.Advance(world, denied));
            CheckState(world, control);
        }
        if (step == 5) {
            world.Joints[joint].Active = true;
            CHECK_THROWS_AS(scene.Step(world, settings), std::runtime_error);
            CheckState(world, control);
            world.Joints[joint].Active = false;
        }
        // Profiled and plain submissions must publish the same physical state.
        settings.profile_gpu = step >= 6;
        const auto full = scene.Step(world, settings);
        auto plain = settings;
        plain.profile_gpu = false;
        const auto light = compact.Advance(control, plain);
        CHECK(full.step == step + 1);
        CHECK(light.step == full.step);
        CHECK(light.warm_started == full.warm_started);
        CHECK(light.geometry_features == full.geometry_features);
        CHECK(light.iterations == full.solve.iterations);
        CheckState(world, control);
        if (settings.profile_gpu) {
            REQUIRE(full.gpu_stages.has_value());
            CHECK(std::isfinite(full.gpu_stages->solve_ms));
            CHECK(full.gpu_stages->solve_ms > 0);
        }
    }
}

TEST_CASE_FIXTURE(OnDevice, "Ji: frozen equations retain the essential numerical safeguards") {
    SUBCASE("mixed rows satisfy each ADMM equation and preserve continuation state") {
        System cpu = MixedSystem(), gpu = cpu;
        for (uint32_t pass = 0; pass < 3; ++pass) {
            const System before = gpu;
            Iterate(cpu);
            const auto result = RunGpu(context, gpu, {.iterations = 1});
            gpu = result.system;
            const auto audit = AuditIteration(before, gpu, result.residual);
            // Dimensionless unit fixture: each pass field has an independent equation check.
            for (double error : {audit.body, audit.x, audit.z, audit.impulse, audit.dual_update,
                                 audit.primal_residual, audit.dual_residual, audit.dynamics_residual})
                CHECK(error < 2e-5);
            for (uint32_t body = 0; body < cpu.bodies.size(); ++body)
                for (uint32_t k = 0; k < 6; ++k)
                    CHECK(std::abs(cpu.bodies[body].v[k] - gpu.bodies[body].v[k]) < 2e-5);
        }
        const auto full = RunGpu(context, MixedSystem(), {.iterations = 3});
        for (uint32_t c = 0; c < gpu.constraints.size(); ++c)
            for (uint32_t side = 0; side < 2; ++side)
                for (uint32_t k = 0; k < 6; ++k) {
                    CHECK(std::abs(gpu.constraints[c].side[side].z[k] - full.system.constraints[c].side[side].z[k]) < 2e-5);
                    CHECK(std::abs(gpu.constraints[c].side[side].u[k] - full.system.constraints[c].side[side].u[k]) < 2e-5);
                }
        cpu = MixedSystem();
        InitializePenalties(cpu);
        const double initial_beta = cpu.bodies[0].beta;
        const auto initialized = RunGpu(context, MixedSystem(), {.iterations = 1}, true);
        CHECK(initialized.system.bodies[0].beta == doctest::Approx(initial_beta).epsilon(1e-5));
        Solve(cpu, {.iterations = 2, .penalty_period = 1});
        const auto adaptive = RunGpu(context, MixedSystem(), {.iterations = 2, .penalty_period = 1}, true);
        CHECK(cpu.bodies[0].beta != initial_beta);
        for (uint32_t body = 0; body < cpu.bodies.size(); ++body)
            CHECK(adaptive.system.bodies[body].beta == doctest::Approx(cpu.bodies[body].beta).epsilon(1e-4));
        const auto converged = RunGpu(context, MixedSystem(), {.iterations = 250, .tolerance = 1e-5});
        REQUIRE(converged.converged);
        CHECK(converged.iterations < 250);
        const auto audit = AuditFinal(converged.system);
        CHECK(audit.dynamics < 2e-5);
        CHECK(audit.hard < 2e-5);
        CHECK(audit.soft < 2e-5);
        CHECK(audit.contact_law < 2e-5);
    }
    SUBCASE("a 10000:1 mass ratio retains momentum and the strict dynamics tolerance") {
        System input{{UnitBody({-1000}), UnitBody({0.1})}, {}};
        input.bodies[0].a.fill(1000); input.bodies[1].a.fill(0.1);
        input.bodies[0].beta = 1000; input.bodies[1].beta = 0.1;
        Constraint contact{.kind = Kind::Contact, .rows = 3};
        contact.side[0].body = 0; contact.side[1].body = 1;
        contact.side[0].j[0] = 1; contact.side[1].j[0] = -1;
        input.constraints.push_back(contact);
        const auto result = RunGpu(context, input, {.iterations = 200, .tolerance = 1e-5});
        REQUIRE(result.converged);
        CHECK(result.iterations < 200);
        CHECK(result.residual.primal <= 1e-5);
        CHECK(result.residual.dual <= 1e-5);
        CHECK(result.residual.dynamics < 1e-5);
        const double force = 2 / (1 / 1000. + 1 / 0.1);
        CHECK(result.system.constraints[0].impulse[0] == doctest::Approx(force).epsilon(1e-4));
        CHECK(std::abs(result.system.bodies[0].v[0] * 1000 + result.system.bodies[1].v[0] * 0.1 + 999.9) < 1e-4);
    }
    SUBCASE("the contact audit distinguishes the strict normal law and gap bias") {
        const auto projected = StrictContactProjection({1, 3, 4}, 0.5);
        CHECK(projected[0] == 1);
        CHECK(projected[1] == doctest::Approx(0.3));
        CHECK(projected[2] == doctest::Approx(0.4));
        System input{{UnitBody()}, {}};
        input.bodies[0].a[0] = 2;
        input.bodies[0].v[0] = 0.001; input.bodies[0].v[1] = 0.3;
        Constraint contact{.kind = Kind::Contact, .rows = 3};
        contact.side[0].body = 0;
        contact.side[0].j[0] = 0.1; contact.side[0].j[7] = 0.2;
        contact.friction = 0.5; contact.impulse[0] = 0.2;
        input.constraints.push_back(contact);
        CHECK(AuditFinal(input).contact_law == doctest::Approx(0.09));
        input.constraints[0].error[0] = 0.001;
        CHECK(AuditFinal(input).contact_law == doctest::Approx(0.2));
        input.bodies[0].a[0] = std::numeric_limits<double>::quiet_NaN();
        CHECK_THROWS_AS(RunGpu(context, input, {.iterations = 1}), std::invalid_argument);
    }
}

TEST_CASE_FIXTURE(OnDevice, "Ji: a nonfinite cache export prevents publication and integration") {
    // Only this failure requires direct kernels. One allocation holds the minimal
    // input to cache export, publication and integration; no solver is reassembled.
    struct Storage {
        GeometryContact geometry;
        JiContactLink link;
        uint32_t counts[3], shapes[2], spawns[2], slots[1], cache_count;
        rbp::Pose poses[2];
        Velocity velocities[2];
        JiBodyData body;
        JiConstraintTile constraint;
        JiWarmContact cache;
        JiStatus status;
        JiParams solve;
        JiSceneParams params;
    };
    mtl::Buffer<Storage> storage{context.Device.get(), 1};
    auto &s = storage[0];
    s = {};
    s.geometry.BodyA = 0; s.geometry.BodyB = 1;
    s.geometry.Normal = {0, 1, 0};
    s.constraint.side[0].body[0] = 0;
    s.constraint.side[1].body[0] = NoBody;
    s.body.v[0] = 1;
    s.params.moving_bodies = 1; s.params.body_count = 2; s.params.dt = 2;
    s.solve.bodies = s.solve.constraints = 1;
    NS::Error *error{};
    auto residency = NS::TransferPtr(context.Device->newResidencySet(mtl::Make<MTL::ResidencySetDescriptor>().get(), &error));
    REQUIRE(residency);
    residency->addAllocation(storage.Handle.get());
    residency->commit(); residency->requestResidency();
    context.Queue->addResidencySet(residency.get());
    auto descriptor = mtl::Make<MTL4::ArgumentTableDescriptor>();
    descriptor->setMaxBufferBindCount(19);
    auto cache_table = NS::TransferPtr(context.Device->newArgumentTable(descriptor.get(), &error));
    auto integrate_table = NS::TransferPtr(context.Device->newArgumentTable(descriptor.get(), &error));
    REQUIRE(cache_table); REQUIRE(integrate_table);
    const auto bind = [&](MTL4::ArgumentTable *table, uint32_t slot, size_t offset) {
        table->setAddress(storage.Address() + offset, slot);
    };
    bind(cache_table.get(), 0, offsetof(Storage, geometry));
    bind(cache_table.get(), 1, offsetof(Storage, link));
    bind(cache_table.get(), 2, offsetof(Storage, counts));
    bind(cache_table.get(), 3, offsetof(Storage, poses));
    bind(cache_table.get(), 4, offsetof(Storage, shapes));
    bind(cache_table.get(), 5, offsetof(Storage, spawns));
    bind(cache_table.get(), 9, offsetof(Storage, constraint));
    bind(cache_table.get(), 13, offsetof(Storage, cache));
    bind(cache_table.get(), 14, offsetof(Storage, status));
    bind(cache_table.get(), 15, offsetof(Storage, solve));
    bind(cache_table.get(), 16, offsetof(Storage, cache_count));
    bind(cache_table.get(), 18, offsetof(Storage, params));
    bind(integrate_table.get(), 0, offsetof(Storage, poses));
    bind(integrate_table.get(), 1, offsetof(Storage, velocities));
    bind(integrate_table.get(), 2, offsetof(Storage, slots));
    bind(integrate_table.get(), 3, offsetof(Storage, body));
    bind(integrate_table.get(), 4, offsetof(Storage, params));
    bind(integrate_table.get(), 5, offsetof(Storage, status));
    bind(integrate_table.get(), 6, offsetof(Storage, solve));
    const std::array passes{shaders::JiCacheContactsPass, shaders::JiPublishCacheCountPass, shaders::JiIntegrateBodiesPass};
    std::array<NS::SharedPtr<MTL::ComputePipelineState>, 3> pipelines;
    for (size_t i = 0; i < passes.size(); ++i)
        pipelines[i] = context.Pipeline(shaders::PipelineIndices[0][passes[i]][0]);
    for (float force : {1.f, std::numeric_limits<float>::max(), 1.f}) {
        const bool valid = force == 1;
        CAPTURE(valid);
        s.poses[0] = At(float3{0, Half, 0}); s.poses[1] = At(float3{});
        s.velocities[0] = {};
        s.counts[0] = 1; s.counts[2] = 0; s.cache_count = 99;
        s.status = {1, 1, 0, 0, 0, 0};
        s.constraint.impulse[0][0] = force;
        const auto before = s.poses[0];
        auto allocator = NS::TransferPtr(context.Device->newCommandAllocator());
        auto commands = NS::TransferPtr(context.Device->newCommandBuffer());
        commands->beginCommandBuffer(allocator.get());
        auto *encoder = commands->computeCommandEncoder();
        for (size_t i = 0; i < passes.size(); ++i) {
            encoder->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
            encoder->setArgumentTable(i == 2 ? integrate_table.get() : cache_table.get());
            encoder->setComputePipelineState(pipelines[i].get());
            encoder->dispatchThreadgroups({1, 1, 1}, {32, 1, 1});
        }
        encoder->endEncoding(); commands->endCommandBuffer();
        std::binary_semaphore done{0};
        auto options = mtl::Make<MTL4::CommitOptions>();
        options->addFeedbackHandler([&](MTL4::CommitFeedback *) { done.release(); });
        const MTL4::CommandBuffer *list[]{commands.get()};
        context.Queue->commit(list, 1, options.get());
        done.acquire();
        CHECK(s.cache_count == uint32_t(valid));
        CHECK(s.status.iterations == uint32_t(valid));
        CHECK((s.counts[2] == 0) == valid);
        CHECK(s.poses[0].Position.x == (valid ? 2.f : 0.f));
        CHECK(s.velocities[0].Linear.x == (valid ? 1.f : 0.f));
        CHECK(length(s.poses[0].Orientation - before.Orientation) == 0);
        CHECK(length(s.velocities[0].Angular) == 0);
        if (valid) CHECK(s.cache.impulse_on_a[1] == 2);
        else CHECK(length(s.poses[0].Position - before.Position) == 0);
    }
    context.Queue->removeResidencySet(residency.get());
}
