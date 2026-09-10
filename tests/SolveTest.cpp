#include "Shapes.h"
#include "Solver.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numbers>
#include <set>
#include <tuple>
#include <vector>

#include <doctest/doctest.h>

using namespace rbp;

namespace {

constexpr float MaxPenetration = 4 * StepSettings{}.ContactMargin;

const float Gravity = std::abs(StepSettings{}.Gravity.y);

void CheckResting(float height) {
    CHECK(height < Half);
    CHECK(height > Half - MaxPenetration);
}

Index AddGround(World &world, BodyDesc desc = {}) {
    desc.Shape = world.AddShape(GroundPlane);
    return world.AddBody(desc);
}

Index DropBox(World &world, float height, bool ground, float friction = 0.5f) {
    const auto box = world.AddShape(UnitBox);
    if (ground) AddGround(world, {.Friction = friction});
    return world.AddBody({.Pose = At(float3{0, height, 0}), .Shape = box, .Friction = friction});
}

std::vector<Index> AddStack(World &world, Index shape, uint32_t count) {
    std::vector<Index> stack;
    for (uint32_t i = 0; i < count; ++i)
        stack.push_back(world.AddBody({.Pose = At(float3{0, Half + 1.02f * float(i), 0}), .Shape = shape}));
    return stack;
}

StepSettings Tilted(float slope) {
    return {.Gravity = {Gravity * std::sin(slope), -Gravity * std::cos(slope), 0}};
}

std::span<const Contact> Slots(const World &world, Index body) {
    return world.Contacts.All().subspan(body * ContactsPerBody, ContactsPerBody);
}

void CheckIncoming(const World &world) {
    std::vector<std::vector<uint32_t>> expected(world.BodyCount());
    for (uint32_t slot = 0; slot < world.BodyCount() * ContactsPerBody; ++slot) {
        const Contact &contact = world.Contacts[slot];
        if (!contact.Active) continue;
        REQUIRE(contact.BodyB < world.BodyCount());
        expected[contact.BodyB].push_back(slot);
    }
    uint32_t prefix = 0;
    for (Index body = 0; body < world.BodyCount(); ++body) {
        CAPTURE(body);
        const Adjacency list = world.Incoming[body];
        REQUIRE(list.Start == prefix);
        REQUIRE(list.Count == expected[body].size());
        REQUIRE(list.Cursor == prefix + list.Count);
        const auto span = world.IncomingSlots.All().subspan(list.Start, list.Count);
        std::vector<uint32_t> actual(span.begin(), span.end());
        if (Moves(world.Masses[body])) CHECK(std::ranges::is_sorted(actual));
        std::ranges::sort(actual);
        CHECK(actual == expected[body]);
        prefix += list.Count;
    }
}

float3 ContactPoint(const World &world, const Contact &contact, bool side) {
    return WorldPoint(world.Poses[side ? contact.BodyA : contact.BodyB], side ? contact.AnchorA : contact.AnchorB);
}

std::set<uint32_t> LeavesTouching(const World &world, Index body) {
    std::set<uint32_t> leaves;
    for (const Contact &contact : Slots(world, body))
        if (contact.Active) leaves.insert(OwnChild(contact.Children));
    return leaves;
}

using ContactKeySet = std::set<std::tuple<Index, Index, Index, Index, uint32_t, uint64_t>>;

ContactKeySet ContactKeys(const World &world) {
    ContactKeySet keys;
    for (const Contact &contact : world.Contacts.All())
        if (contact.Active) keys.emplace(contact.BodyA, contact.BodyB, contact.SubShapeA, contact.SubShape, contact.Feature, contact.Children);
    return keys;
}

uint32_t Reported(const World &world, Index body, ContactEventKind kind) {
    uint32_t found = 0;
    for (uint32_t i = 0; i < world.ContactEventCounts[body]; ++i)
        found += world.ContactEvents[body * EventsPerBody + i].Kind == kind ? 1 : 0;
    return found;
}

// Compare fields individually because float3 contains padding.
std::vector<Pose> Snapshot(const World &world) {
    return {world.Poses.All().begin(), world.Poses.All().begin() + world.BodyCount()};
}

void CheckIdentical(const std::vector<Pose> &first, const std::vector<Pose> &second) {
    REQUIRE(first.size() == second.size());
    for (size_t body = 0; body < first.size(); ++body) {
        CAPTURE(body);
        CHECK(std::memcmp(&first[body].Position, &second[body].Position, 3 * sizeof(float)) == 0);
        CHECK(std::memcmp(&first[body].Orientation, &second[body].Orientation, sizeof(float4)) == 0);
    }
}

struct OnDevice {
    const mtl::Context context;
    Solver solver{context};
};

void Run(Solver &solver, World &world, uint32_t steps, const StepSettings &settings = {}) {
    solver.Advance(world, settings, steps);
}

void DriveAlongX(Solver &solver, World &world, Index body, float3 &at, float speed, uint32_t steps, const StepSettings &settings, auto each) {
    for (uint32_t step = 0; step < steps; ++step) {
        at.x += speed * settings.DeltaTime;
        Drive(world, body, at, float3{speed, 0, 0});
        solver.Step(world, settings);
        each();
    }
}

float AccelerationX(Solver &solver, World &world, Index body, uint32_t steps, const StepSettings &settings) {
    const float was = world.Velocities[body].Linear.x;
    Run(solver, world, steps, settings);
    return (world.Velocities[body].Linear.x - was) / (float(steps) * settings.DeltaTime);
}

} // namespace

struct Pendulum {
    Index Arm;
    float Inertia;
    float Mass;
};

Pendulum MakePendulum(World &world, float distance, JointDesc joint = {}) {
    const auto shape = world.AddShape(UnitBox);
    const auto pivot = world.AddBody({});
    const auto arm = world.AddBody({.Pose = At(float3{distance, 0, 0}), .Shape = shape});
    joint.BodyA = arm;
    joint.BodyB = pivot;
    world.AddJoint(joint);
    const float mass = 1 / world.Masses[arm].InvMass;

    return {arm, 1 / world.Masses[arm].InvInertiaLocal[2] + mass * distance * distance, mass};
}

namespace {

// The centered pivot excludes the parallel-axis inertia term.
struct Wheel {
    Index Body, Joint;
    float Inertia;
};

Wheel SpinOnAxle(World &world, JointDesc joint, float3 spin) {
    const auto shape = world.AddShape(UnitBox);
    const auto axle = world.AddBody({});
    const auto wheel = world.AddBody({.Pose = At(float3{0, 0, 0}), .Velocity = {.Angular = spin}, .Shape = shape});
    joint.BodyA = wheel;
    joint.BodyB = axle;
    return {wheel, world.AddJoint(joint), 1 / world.Masses[wheel].InvInertiaLocal[2]};
}
} // namespace

namespace {

struct Slider {
    Index Box;
    float Mass;
};

Slider MakeSlider(World &world, JointDesc joint, float3 at = {0, 0, 0}) {
    const auto shape = world.AddShape(UnitBox);
    const auto anchor = world.AddBody({});
    const auto box = world.AddBody({.Pose = At(at), .Shape = shape});
    joint.BodyA = box;
    joint.BodyB = anchor;
    world.AddJoint(joint);
    return {box, 1 / world.Masses[box].InvMass};
}
} // namespace

namespace {

struct Stacked {
    Index Lower, Upper;
};
Stacked TwoOnAPlane(World &world) {
    const auto shape = world.AddShape(UnitBox);
    AddGround(world);
    return {world.AddBody({.Pose = At(float3{0, Half, 0}), .Shape = shape}), world.AddBody({.Pose = At(float3{0, Half + 1, 0}), .Shape = shape})};
}
} // namespace

namespace {

struct Settled {
    float3 Position;
    float3 Normal;
    uint32_t Contacts;
};

Settled Rest(const World &world, Index body) {
    const auto slots = Slots(world, body);
    Settled settled{.Position = world.Poses[body].Position, .Normal = {0, 0, 0}, .Contacts = 0};
    for (const auto &contact : slots) {
        if (!contact.Active) continue;
        settled.Normal += contact.Normal;
        ++settled.Contacts;
    }
    if (settled.Contacts > 0) settled.Normal /= float(settled.Contacts);
    return settled;
}

} // namespace

namespace {

Index FloorMesh(World &world, uint32_t side, float extent, float slope = 0, Pose local = IdentityPose) {
    std::vector<float3> points;
    std::vector<uint32_t> indices;
    for (uint32_t x = 0; x <= side; ++x)
        for (uint32_t z = 0; z <= side; ++z) {
            const float along = extent * (2.f * float(x) / float(side) - 1);
            points.push_back(float3{along * std::cos(slope), along * std::sin(slope), extent * (2.f * float(z) / float(side) - 1)});
        }
    const auto at = [side](uint32_t x, uint32_t z) { return x * (side + 1) + z; };
    for (uint32_t x = 0; x < side; ++x)
        for (uint32_t z = 0; z < side; ++z) {
            indices.insert(indices.end(), {at(x, z), at(x, z + 1), at(x + 1, z + 1)});
            indices.insert(indices.end(), {at(x, z), at(x + 1, z + 1), at(x + 1, z)});
        }
    return world.AddMesh(points, indices, local);
}

Index AddFloor(World &world, uint32_t side, float slope = 0, float friction = BodyDesc{}.Friction) {
    const float3 up{-std::sin(slope), std::cos(slope), 0};
    const Index floor = side == 0 ? world.AddShape({.Normal = up, .Offset = 0, .Kind = ShapePlane}) : FloorMesh(world, side, 5, slope);
    REQUIRE(floor != NoIndex);
    world.AddBody({.Shape = floor, .Friction = friction});
    return floor;
}

Index RidgeMesh(World &world, float extent, float height) {
    std::vector<float3> points;
    for (const float x : {-extent, 0.f, extent})
        for (const float z : {-extent, extent}) points.push_back(float3{x, x == 0 ? height : 0, z});

    const std::vector<uint32_t> indices{0, 1, 3, 0, 3, 2, 2, 3, 5, 2, 5, 4};
    return world.AddMesh(points, indices);
}
} // namespace

namespace {

AuthoredMass CubeMass(float mass, float side) {
    const float inertia = mass * side * side / 6;
    return {.Mass = mass, .Inertia = {inertia, inertia, inertia}};
}

AuthoredMass PinnedCubeMass(float mass, float side) {
    return {.Mass = 0, .Inertia = CubeMass(mass, side).Inertia};
}
} // namespace

TEST_CASE_FIXTURE(OnDevice, "dynamics: friction follows the Coulomb cone and changing load") {
    SUBCASE("frictionless contact conserves motion across the plane and spin about it") {
        World world{context};
        const auto box = DropBox(world, 1.4f, true, 0);
        constexpr float3 Drift{0.7f, 0, -0.3f};
        constexpr float Spin = 1.1f;
        world.Velocities[box] = {.Linear = Drift, .Angular = {0, Spin, 0}};

        Run(solver, world, 200);
        const auto &velocity = world.Velocities[box];

        CHECK(velocity.Linear.x == doctest::Approx(Drift.x).epsilon(1e-3));
        CHECK(velocity.Linear.z == doctest::Approx(Drift.z).epsilon(1e-3));
        CHECK(velocity.Angular.y == doctest::Approx(Spin).epsilon(1e-3));

        CheckResting(world.Poses[box].Position.y);
        CHECK(std::abs(velocity.Linear.y) < 1e-3f);
        CHECK(std::abs(velocity.Angular.x) < 1e-4f);
        CHECK(std::abs(velocity.Angular.z) < 1e-4f);
    }
    SUBCASE("static friction holds a box on a slope inside the cone and lets it go outside") {
        World world{context};
        constexpr float Mu = 0.5f;
        const float cone = std::atan(Mu);

        SUBCASE("inside the cone it does not creep") {
            const auto settings = Tilted(cone * 0.6f);
            const auto box = DropBox(world, Half, true, Mu);

            Run(solver, world, 30, settings);
            const float from = world.Poses[box].Position.x;
            Run(solver, world, 240, settings);
            CHECK(std::abs(world.Poses[box].Position.x - from) < 1e-3f);
            CHECK(simd::length(world.Velocities[box].Linear) < 1e-3f);
            CheckResting(world.Poses[box].Position.y);
        }

        SUBCASE("outside it slides at the closed form's acceleration") {
            constexpr float Slope = 0.6f;
            const auto settings = Tilted(Slope);
            const float expected = Gravity * (std::sin(Slope) - Mu * std::cos(Slope));
            const auto box = DropBox(world, Half, true, Mu);

            Run(solver, world, 30, settings);
            CHECK(AccelerationX(solver, world, box, 60, settings) == doctest::Approx(expected).epsilon(0.05));
            CHECK(simd::length(world.Velocities[box].Angular) < 1e-2f);
        }
    }
    SUBCASE("a sliding box decelerates at mu g and stops where the closed form says") {
        World world{context};
        constexpr float Mu = 0.5f, Speed = 2;
        const auto box = DropBox(world, Half, true, Mu);

        Run(solver, world, 30);
        const float from = world.Poses[box].Position.x;
        world.Velocities[box].Linear = {Speed, 0, 0};

        Run(solver, world, 120);
        CHECK(world.Poses[box].Position.x - from == doctest::Approx(Speed * Speed / (2 * Mu * Gravity)).epsilon(0.05));
        CHECK(std::abs(world.Velocities[box].Linear.x) < 1e-2f);

        CheckResting(world.Poses[box].Position.y);
        CHECK(std::abs(world.Poses[box].Position.z - 0) < 1e-3f);
        CHECK(simd::length(world.Velocities[box].Angular) < 1e-2f);
    }
    SUBCASE("reducing a resting contact load releases its cached friction force") {
        World world{context};
        AddGround(world, {.Friction = 1});
        const auto box = world.AddBody({.Pose = At(float3{0, Half, 0}), .Shape = world.AddShape(UnitBox), .Mass = AuthoredMass{.Mass = 1, .Inertia = {0, 0, 0}}, .Friction = 1});
        StepSettings settings{.Gravity = {400, -1000, 0}, .SleepSteps = ~0u};
        for (uint32_t step = 0; step < 120; ++step) solver.Step(world, settings);
        REQUIRE(simd::length(world.Velocities[box].Linear) < 0.01f);
        settings.Gravity = {0, -100, 0};
        for (uint32_t step = 0; step < 20; ++step) {
            solver.Step(world, settings);
            CHECK(std::abs(world.Velocities[box].Linear.x) < 0.01f);
        }
    }
}

TEST_CASE_FIXTURE(OnDevice, "dynamics: stacked support preserves load and contact identity") {
    SUBCASE("a stack of boxes holds itself up") {
        World world{context};
        constexpr uint32_t Count = 5;
        const StepSettings settings{};
        const auto shape = world.AddShape(UnitBox);
        AddGround(world);
        const std::vector<Index> stack = AddStack(world, shape, Count);

        Run(solver, world, 600);

        float below = -Half;
        for (uint32_t i = 0; i < Count; ++i) {
            CAPTURE(i);
            const float height = world.Poses[stack[i]].Position.y;

            CHECK(std::abs(height - (below + 1 - settings.ContactMargin)) < 1e-3f);
            CHECK(simd::length(world.Velocities[stack[i]].Linear) < 0.02f);
            below = height;
        }
    }
    SUBCASE("a light box carries a box a thousand times its weight") {
        World world{context};
        constexpr float Margin = StepSettings{}.ContactMargin;
        const auto shape = world.AddShape(UnitBox);
        AddGround(world);
        const auto light = world.AddBody({.Pose = At(float3{0, Half, 0}), .Shape = shape, .Density = 10});
        const auto heavy = world.AddBody({.Pose = At(float3{0, 3 * Half + 0.02f, 0}), .Shape = shape, .Density = 10000});

        Run(solver, world, 600);
        CHECK(std::abs(world.Poses[light].Position.y - (Half - Margin)) < 2e-3f);
        CHECK(std::abs(world.Poses[heavy].Position.y - (3 * Half - 2 * Margin)) < 3e-3f);
        CHECK(simd::length(world.Velocities[light].Linear) < 0.1f);
    }
}

TEST_CASE_FIXTURE(OnDevice, "dynamics: impacts conserve momentum and respect unilateral contact") {
    SUBCASE("head-on impacts preserve momentum across mesh and box masses") {
        World world{context};
        float direction = 1, scale = 1, drift = 0, ratio = 1, speed = 1, tolerance_scale = 1;
        float dt = StepSettings{}.DeltaTime;
        uint32_t iterations = StepSettings{}.Iterations;
        bool convex = false;
        SUBCASE("left mesh first") {}
        SUBCASE("right mesh first") { direction = -1; }
        SUBCASE("convex boxes") { convex = true; }
        SUBCASE("one gram meshes") { scale = 0.001f; }
        SUBCASE("one tonne meshes") { scale = 1000; }
        SUBCASE("translating meshes") { drift = 0.5f; }
        SUBCASE("unequal meshes") { ratio = 2; }
        SUBCASE("ten metres per second") {
            speed = 10;
            tolerance_scale = 10;
        }
        SUBCASE("one hundred metres per second at 240 Hz with twenty iterations") {
            speed = 100;
            dt = 1.f / 240;
            iterations = 20;
        }
        const auto shape = convex ? world.AddShape({.HalfExtents = {0.4f, 0.4f, 0.4f}, .Kind = ShapeBox}) : BoxMesh(world, 0.4f);
        world.Shapes[shape].DoubleSided = true;
        const AuthoredMass mass{.Mass = scale, .Inertia = float3{0.1f, 0.1f, 0.1f} * scale};
        const AuthoredMass mass_a{.Mass = scale * ratio, .Inertia = mass.Inertia * ratio};
        const auto a = world.AddBody({.Pose = At(float3{-direction, 0, 0}), .Velocity = {.Linear = {direction * speed + drift, 0, 0}}, .Shape = shape, .Mass = mass_a});
        const auto b = world.AddBody({.Pose = At(float3{direction, 0, 0}), .Velocity = {.Linear = {-direction * speed + drift, 0, 0}}, .Shape = shape, .Mass = mass});
        const float3 velocity{direction * speed * (ratio - 1) / (ratio + 1) + drift, 0, 0};
        const float3 center = (ratio * world.Poses[a].Position + world.Poses[b].Position) / (ratio + 1);
        float worst_momentum = 0, worst_center = 0;
        for (uint32_t step = 0; step < uint32_t(std::lround(2 / dt)); ++step) {
            solver.Step(world, {.Gravity = {0, 0, 0}, .DeltaTime = dt, .Iterations = iterations});
            const auto current_velocity = (ratio * world.Velocities[a].Linear + world.Velocities[b].Linear) / (ratio + 1);
            const auto current_center = (ratio * world.Poses[a].Position + world.Poses[b].Position) / (ratio + 1);
            worst_momentum = std::max(worst_momentum, simd::length(current_velocity - velocity));
            worst_center = std::max(worst_center, simd::length(current_center - center - velocity * (float(step + 1) * dt)));
        }
        CHECK(worst_momentum < 0.002f * tolerance_scale);
        CHECK(worst_center < 0.002f * tolerance_scale);
        CHECK(direction * (world.Poses[b].Position.x - world.Poses[a].Position.x) == doctest::Approx(0.8f).epsilon(0.005));
        CHECK(simd::length((ratio * world.Poses[a].Position + world.Poses[b].Position) / (ratio + 1) - center - velocity * 2) < 0.002f * tolerance_scale);
        CHECK(simd::length((ratio * world.Velocities[a].Linear + world.Velocities[b].Linear) / (ratio + 1) - velocity) < 0.002f * tolerance_scale);
        CHECK(simd::length(world.Velocities[a].Linear - velocity) < 0.01f * tolerance_scale);
        CHECK(simd::length(world.Velocities[b].Linear - velocity) < 0.01f * tolerance_scale);
        CHECK(simd::length(world.Velocities[a].Angular) < 0.01f * tolerance_scale);
        CHECK(simd::length(world.Velocities[b].Angular) < 0.01f * tolerance_scale);
    }
    SUBCASE("glancing sphere impacts conserve isolated momentum and dissipate energy") {
        World world{context};
        float speed = 1;
        SUBCASE("one metre per second") {}
        SUBCASE("ten metres per second") { speed = 10; }
        constexpr float Radius = 0.4f, Inertia = 0.4f * Radius * Radius;
        const auto shape = world.AddShape({.Radius = Radius, .Kind = ShapeSphere});
        const AuthoredMass mass{.Mass = 1, .Inertia = {Inertia, Inertia, Inertia}};
        const auto a = world.AddBody({.Pose = At(float3{-1, 0.15f, 0}), .Velocity = {.Linear = {speed, 0, 0}}, .Shape = shape, .Mass = mass, .Friction = 0});
        const auto b = world.AddBody({.Pose = At(float3{1, -0.15f, 0}), .Velocity = {.Linear = {-speed, 0, 0}}, .Shape = shape, .Mass = mass, .Friction = 0});
        const float3 initial_angular{0, 0, -0.3f * speed};
        float worst_linear = 0, worst_angular = 0, highest_energy = 0, closest = INFINITY;
        for (uint32_t step = 0; step < 120; ++step) {
            solver.Step(world, {.Gravity = {0, 0, 0}});
            const auto va = world.Velocities[a], vb = world.Velocities[b];
            const float3 angular = simd::cross(world.Poses[a].Position, va.Linear) + simd::cross(world.Poses[b].Position, vb.Linear) + Inertia * (va.Angular + vb.Angular);
            worst_linear = std::max(worst_linear, simd::length(va.Linear + vb.Linear));
            worst_angular = std::max(worst_angular, simd::length(angular - initial_angular));
            highest_energy = std::max(highest_energy, 0.5f * (simd::length_squared(va.Linear) + simd::length_squared(vb.Linear) + Inertia * (simd::length_squared(va.Angular) + simd::length_squared(vb.Angular))));
            closest = std::min(closest, simd::distance(world.Poses[a].Position, world.Poses[b].Position));
        }
        CHECK(worst_linear < 0.004f * speed);
        CHECK(worst_angular < 0.01f * simd::length(initial_angular));
        CHECK(highest_energy <= speed * speed * 1.002f);
        CHECK(closest > 2 * Radius - MaxPenetration);
        CHECK(world.Velocities[a].Linear.y > 0.1f * speed);
        CHECK(world.Velocities[b].Linear.y < -0.1f * speed);
        CHECK(simd::length(world.Velocities[a].Angular) < 0.001f);
        CHECK(simd::length(world.Velocities[b].Angular) < 0.001f);
    }
    SUBCASE("a fast box does not tunnel through the plane") {
        World world{context};
        constexpr float Speed = 40;
        const auto box = DropBox(world, 3, true);
        world.Velocities[box].Linear = {0, -Speed, 0};

        float lowest = 1e9f;
        for (uint32_t step = 0; step < 120; ++step) {
            solver.Step(world);
            lowest = std::min(lowest, world.Poses[box].Position.y);
        }

        CHECK(lowest > Half - Speed * StepSettings{}.DeltaTime);
        CHECK(lowest > -Half);
        CheckResting(world.Poses[box].Position.y);
    }
    SUBCASE("a contact across a gap holds its bodies apart and pushes nothing") {
        World world{context};
        const auto box = DropBox(world, Half + 0.05f, true);
        world.Velocities[box].Linear = {10, 0, 0};

        uint32_t apart = 0, touching = 0;
        for (uint32_t step = 0; step < 120; ++step) {
            solver.Step(world);
            const float gap = world.Poses[box].Position.y - Half;
            for (const Contact &contact : world.Contacts.All()) {
                if (!contact.Active) continue;
                if (gap <= StepSettings{}.ContactMargin) {
                    ++touching;
                    continue;
                }
                ++apart;
                CHECK(contact.Lambda[0] == 0);
                CHECK(contact.C0[0] > StepSettings{}.ContactMargin);
            }
        }
        CHECK(apart > 0);
        CHECK(touching > 0);
        CheckResting(world.Poses[box].Position.y);
    }
}

TEST_CASE_FIXTURE(OnDevice, "joints: free motion converges on energy and conserves momentum") {
    SUBCASE("a pendulum converges on the speed energy says it reaches") {
        constexpr float Distance = 1;

        const auto fastest = [&](float rate) {
            World world{context};
            const auto pendulum = MakePendulum(world, Distance);
            const StepSettings settings{.DeltaTime = 1 / rate};
            float peak = 0;
            for (uint32_t step = 0; step < uint32_t(rate); ++step) {
                solver.Step(world, settings);
                peak = std::max(peak, simd::length(world.Velocities[pendulum.Arm].Angular));
            }
            return std::pair{peak, std::sqrt(2 * pendulum.Mass * Gravity * Distance / pendulum.Inertia)};
        };

        const auto [coarse, expected] = fastest(60);
        const auto [fine, same] = fastest(960);
        CHECK(same == doctest::Approx(expected));
        CHECK(coarse < expected);
        CHECK(coarse > 0.9f * expected);
        CHECK(fine > coarse);
        CHECK(fine == doctest::Approx(expected).epsilon(0.02));
    }
    SUBCASE("a joint between two moving bodies gives each what it takes from the other") {
        const StepSettings settings{.Gravity = {0, 0, 0}};

        struct Momentum {
            float3 Linear, Angular, Centre;
        };
        const auto momentum = [](const World &world, Index a, Index b) {
            Momentum total{.Linear = {0, 0, 0}, .Angular = {0, 0, 0}, .Centre = {0, 0, 0}};
            float mass = 0;
            for (const Index body : {a, b}) {
                const float m = 1 / world.Masses[body].InvMass;
                mass += m;
                total.Centre += m * world.Poses[body].Position;
                total.Linear += m * world.Velocities[body].Linear;
            }
            total.Centre /= mass;
            for (const Index body : {a, b}) {
                const float m = 1 / world.Masses[body].InvMass;
                const float3 arm = world.Poses[body].Position - total.Centre;
                total.Angular += m * simd::cross(arm, world.Velocities[body].Linear);

                const float4 q = world.Poses[body].Orientation;
                const float3 spin = world.Velocities[body].Angular;
                const float3 local = Rotate(QuatConjugate(q), spin) / world.Masses[body].InvInertiaLocal;
                total.Angular += Rotate(q, local);
            }
            return total;
        };

        const auto pair = [&](float density_b) {
            World world{context};
            const auto shape = world.AddShape(UnitBox);
            const auto a = world.AddBody({.Pose = At(float3{-1, 0, 0}), .Velocity = {.Linear = {0, 0.5f, 0}, .Angular = {0, 0, 1.5f}}, .Shape = shape});
            const auto b = world.AddBody({.Pose = At(float3{1, 0, 0}), .Velocity = {.Linear = {0.25f, 0, -0.3f}}, .Shape = shape, .Density = density_b});
            REQUIRE(world.AddJoint({.BodyA = a, .BodyB = b, .At = {0, 0, 0}}) != NoIndex);
            return std::tuple{std::move(world), a, b};
        };

        const auto swing = [&](float density_b) {
            auto [world, a, b] = pair(density_b);
            const Momentum began = momentum(world, a, b);
            const float3 started = began.Centre;

            const float3 together = began.Linear * (world.Masses[a].InvMass * world.Masses[b].InvMass) /
                (world.Masses[a].InvMass + world.Masses[b].InvMass);
            float worst_hold = 0, worst_drift = 0;
            for (uint32_t step = 0; step < 600; ++step) {
                solver.Step(world, settings);
                const Pose &pa = world.Poses[a], &pb = world.Poses[b];

                worst_hold = std::max(worst_hold, simd::distance(WorldPoint(pa, world.Joints[0].AnchorA), WorldPoint(pb, world.Joints[0].AnchorB)));

                const Momentum now = momentum(world, a, b);
                const float3 carried = started + together * (float(step + 1) * settings.DeltaTime);
                worst_drift = std::max(worst_drift, simd::distance(now.Centre, carried) / (settings.DeltaTime * simd::length(together)));
            }
            return std::tuple{began, momentum(world, a, b), worst_hold, worst_drift};
        };

        SUBCASE("equal masses") {
            const auto [began, ended, hold, drift] = swing(1000);
            CHECK(hold < 5e-3f);

            CHECK(simd::length(ended.Linear - began.Linear) < 1e-3f * simd::length(began.Linear));
            CHECK(simd::length(ended.Angular - began.Angular) < 2e-2f * simd::length(began.Angular));
            CHECK(drift < 1);
        }

        SUBCASE("a thousand to one in mass") {
            const auto [began, ended, hold, drift] = swing(1);
            CHECK(hold < 5e-3f);
            CHECK(simd::length(ended.Linear - began.Linear) < 1e-3f * simd::length(began.Linear));
            CHECK(drift < 1);
        }
    }
    SUBCASE("a slider spring transfers torque to its moving reference frame") {
        World world{context};
        const auto a = world.AddBody({.Pose = At(float3{1, 1, 0}), .Mass = AuthoredMass{}});
        const auto b = world.AddBody({.Mass = AuthoredMass{}});
        REQUIRE(world.AddJoint({.BodyA = a, .BodyB = b, .AtA = float3{1, 1, 0}, .AtB = float3{0, 0, 0}, .Linear = {AxisFree, AxisLocked, AxisFree}, .LinearStiffness = {0, 1, 0}}) != NoIndex);
        const StepSettings settings{.Gravity = {0, 0, 0}, .DeltaTime = 0.01f};
        solver.Step(world, settings);

        CHECK(std::abs(world.Velocities[a].Linear.y + settings.DeltaTime) < 0.01f * settings.DeltaTime);
        CHECK(std::abs(world.Velocities[b].Angular.z - settings.DeltaTime) < 0.01f * settings.DeltaTime);
        const float3 momentum = world.Velocities[a].Linear + world.Velocities[b].Linear;
        const float3 angular = simd::cross(world.Poses[a].Position, world.Velocities[a].Linear) +
            simd::cross(world.Poses[b].Position, world.Velocities[b].Linear) +
            world.Velocities[a].Angular + world.Velocities[b].Angular;
        CHECK(simd::length(momentum) < 1e-5f);
        CHECK(simd::length(angular) < 1e-5f);
    }
}

TEST_CASE_FIXTURE(OnDevice, "joints: stiff constraints preserve free axes and remain stable") {
    World world{context};
    const auto frame = QuatFromRotationVector(float3{0.4f, 0.7f, -0.2f});
    const auto axis = Rotate(frame, float3{0, 0, 1});
    const auto rotor = world.AddBody({.Velocity = {.Angular = axis}, .Mass = AuthoredMass{.Mass = 0.001f, .Inertia = {1e-8f, 1e-8f, 1e-8f}}});
    const auto axle = world.AddBody({});
    REQUIRE(world.AddJoint({.BodyA = rotor, .BodyB = axle, .Frame = frame, .Angular = {AxisLocked, AxisLocked, AxisFree}}) != NoIndex);
    const StepSettings settings{.Gravity = {0, 0, 0}, .DeltaTime = 1.f / 240, .PenaltyMin = 1e7f, .PenaltyMax = 1e7f};
    for (uint32_t step = 0; step < 240; ++step) {
        solver.Step(world, settings);
        REQUIRE(simd::length(world.Velocities[rotor].Angular - axis) < 0.01f);
    }
    const auto expected = QuatFromRotationVector(axis);
    CHECK(simd::length(RotationVector(QuatMul(world.Poses[rotor].Orientation, QuatConjugate(expected)))) < 0.01f);
}

TEST_CASE_FIXTURE(OnDevice, "joints: authored frames and unwrapped angles define the constrained axes") {
    SUBCASE("a driven hinge turns twenty revolutions on a tilted axle and reports every one") {
        constexpr float Turn = 0.5235988f, Turns = 20;
        const float4 frame = QuatMul(QuatFromRotationVector(float3{0, Turn, 0}), QuatFromRotationVector(float3{Turn, 0, 0}));
        constexpr uint32_t Steps = 600;
        const StepSettings settings{.Gravity = {0, 0, 0}};
        const float wanted = Turns * 2 * std::numbers::pi_v<float>;
        const float speed = wanted / (float(Steps) * settings.DeltaTime);

        World world{context};
        const auto wheel = SpinOnAxle(world, {.Frame = frame, .Angular = {AxisLocked, AxisLocked, AxisDriven}, .MotorSpeed = {0, 0, speed}, .MotorMaxTorque = {0, 0, 1e6f}}, float3{0, 0, 0});
        REQUIRE(wheel.Joint != NoIndex);
        const float3 axle = Rotate(frame, float3{0, 0, 1});
        float off_the_axle = 0;
        for (uint32_t step = 0; step < Steps; ++step) {
            solver.Step(world, settings);
            const float3 rate = world.Velocities[wheel.Body].Angular;
            off_the_axle = std::max(off_the_axle, simd::length(rate - simd::dot(rate, axle) * axle));
        }

        const float turned = world.Joints[wheel.Joint].Twist;
        CHECK(turned == doctest::Approx(wanted).epsilon(0.01));
        CHECK(turned < wanted);
        CHECK(simd::dot(world.Velocities[wheel.Body].Angular, axle) == doctest::Approx(speed).epsilon(0.01));
        CHECK(off_the_axle < 1e-3f);
    }
    SUBCASE("a twist limit past the half turn stops the axis where it says") {
        constexpr float Turn = 0.5235988f, Stop = 3.4906585f, Degrees = 0.017453293f;
        const float4 frame = QuatMul(QuatFromRotationVector(float3{0, Turn, 0}), QuatFromRotationVector(float3{Turn, 0, 0}));
        const StepSettings settings{.Gravity = {0, 0, 0}};

        World world{context};
        const auto wheel = SpinOnAxle(world, {.Frame = frame, .Angular = {AxisLocked, AxisLocked, AxisLimited}, .LimitLow = {0, 0, -0.5f}, .LimitHigh = {0, 0, Stop}}, Rotate(frame, float3{0, 0, 4}));
        REQUIRE(wheel.Joint != NoIndex);
        for (uint32_t step = 0; step < 600; ++step) solver.Step(world, settings);

        const float turned = world.Joints[wheel.Joint].Twist;
        CHECK(turned == doctest::Approx(Stop).epsilon(0.002));
        CHECK(turned > std::numbers::pi_v<float>);
        CHECK(turned / Degrees > 190);
        CHECK(simd::length(world.Velocities[wheel.Body].Angular) < 1e-3f);
    }
    SUBCASE("integration: independently authored joint frames align") {
        World world{context};
        const auto shape = world.AddShape(UnitBox);
        const auto anchor = world.AddBody({.Shape = shape, .Density = 0});
        const auto body = world.AddBody({.Shape = shape, .Density = 1});
        const auto turn = QuatFromRotationVector(float3{0, 0, 0.6f});
        REQUIRE(world.AddJoint({.BodyA = body, .BodyB = anchor, .FrameA = turn, .FrameB = float4{0, 0, 0, 1}, .Angular = {AxisLocked, AxisLocked, AxisLocked}}) != NoIndex);
        for (int i = 0; i < 120; ++i) solver.Step(world, {.Gravity = {0, 0, 0}});
        CHECK(length(RotationVector(QuatMul(world.Poses[body].Orientation, turn))) < 0.002f);
    }
}

TEST_CASE_FIXTURE(OnDevice, "joints: drives obey force bounds and independent stops") {
    SUBCASE("a positioned spring held to a torque climbs at exactly that torque over that inertia") {
        World world{context};
        // A saturated soft drive produces angular acceleration tau / I.
        // A hard drive would also apply positional stabilization.
        constexpr float Torque = 300, Target = 3, Stiffness = 1e6f;
        const StepSettings settings{.Gravity = {0, 0, 0}};
        const auto pendulum = MakePendulum(world, 1, {.Angular = {AxisLocked, AxisLocked, AxisPositioned}, .MotorTarget = {0, 0, Target}, .MotorMaxTorque = {0, 0, Torque}, .AngularStiffness = {INFINITY, INFINITY, Stiffness}});

        constexpr uint32_t Steps = 200;
        Run(solver, world, Steps, settings);
        const float elapsed = Steps * settings.DeltaTime;
        const float reached = world.Velocities[pendulum.Arm].Angular.z;

        REQUIRE(0.5f * Torque / pendulum.Inertia * elapsed * elapsed < Target);
        REQUIRE(Stiffness * Target > Torque);
        CHECK(reached == doctest::Approx(Torque / pendulum.Inertia * elapsed).epsilon(0.05));
    }
    SUBCASE("a motor held to a torque spins up at exactly that torque over that inertia") {
        World world{context};
        constexpr float Torque = 300, Speed = 2;
        const StepSettings settings{.Gravity = {0, 0, 0}};
        const auto pendulum = MakePendulum(world, 1, {.Angular = {AxisLocked, AxisLocked, AxisDriven}, .MotorSpeed = {0, 0, Speed}, .MotorMaxTorque = {0, 0, Torque}});

        constexpr uint32_t Steps = 300;
        Run(solver, world, Steps, settings);
        const float elapsed = Steps * settings.DeltaTime;
        const float reached = world.Velocities[pendulum.Arm].Angular.z;
        REQUIRE(reached < Speed);
        CHECK(reached == doctest::Approx(Torque / pendulum.Inertia * elapsed).epsilon(0.05));
    }
    SUBCASE("a linear drive reaches its speed, and climbs to it at exactly force over mass") {
        constexpr float Speed = 2;
        const StepSettings settings{.Gravity = {0, 0, 0}};

        SUBCASE("with force to spare it reaches the speed") {
            World world{context};
            const auto slider = MakeSlider(world, {.Linear = {AxisDriven, AxisLocked, AxisLocked}, .LinearMotorSpeed = {Speed, 0, 0}, .LinearMotorMaxForce = {1e7f, 0, 0}});
            Run(solver, world, 300, settings);
            CHECK(world.Velocities[slider.Box].Linear.x == doctest::Approx(Speed).epsilon(0.01));
            CHECK(std::abs(world.Poses[slider.Box].Position.y) < 1e-3f);
            CHECK(std::abs(world.Poses[slider.Box].Position.z) < 1e-3f);
        }

        SUBCASE("held to a force it climbs at exactly that force over that mass") {
            constexpr float Force = 3000, Fast = 100;
            World world{context};
            const auto slider = MakeSlider(world, {.Linear = {AxisDriven, AxisLocked, AxisLocked}, .LinearMotorSpeed = {Fast, 0, 0}, .LinearMotorMaxForce = {Force, 0, 0}});
            constexpr uint32_t Steps = 200;
            Run(solver, world, Steps, settings);
            const float elapsed = Steps * settings.DeltaTime;
            const float reached = world.Velocities[slider.Box].Linear.x;
            REQUIRE(reached < Fast);
            CHECK(reached == doctest::Approx(Force / slider.Mass * elapsed).epsilon(0.05));
        }
    }

    SUBCASE("integration: a drive pushes against its independent stop") {
        World world{context};
        const auto shape = world.AddShape(UnitBox);
        const auto anchor = world.AddBody({.Shape = shape, .Density = 0});
        const auto body = world.AddBody({.Shape = shape, .Density = 1});
        JointDesc joint{.BodyA = body, .BodyB = anchor, .Linear = {AxisLimited, AxisLocked, AxisLocked}, .LinearLimitLow = {-0.5f, 0, 0}, .LinearLimitHigh = {0.5f, 0, 0}};
        joint.Drives[0] = {.Enabled = 1, .Speed = 1, .MaxForce = 10, .Damping = 10};
        REQUIRE(world.AddJoint(joint) != NoIndex);
        const StepSettings settings{.Gravity = {0, 0, 0}};
        for (int i = 0; i < 180; ++i) solver.Step(world, settings);
        CHECK(world.Poses[body].Position.x == doctest::Approx(0.5f).epsilon(0.01));
        CHECK(std::abs(world.Velocities[body].Linear.x) < 0.01f);
    }
}

TEST_CASE_FIXTURE(OnDevice, "joints: springs and dampers obey their force and inertia laws") {
    SUBCASE("a soft linear row hangs a body on a spring") {
        World world{context};
        constexpr float Stiffness = 2e5f;
        const auto anchor = world.AddBody({});
        const auto box = world.AddBody({.Shape = world.AddShape(UnitBox)});
        world.AddJoint({.BodyA = box, .BodyB = anchor, .At = {0, 0, 0}, .LinearStiffness = {Stiffness, Stiffness, Stiffness}});

        Run(solver, world, 900);
        const float weight = Gravity / world.Masses[box].InvMass;
        CHECK(world.Poses[box].Position.y == doctest::Approx(-weight / Stiffness).epsilon(0.05).scale(0));
        CHECK(std::abs(world.Poses[box].Position.x) < 1e-3f);
        CHECK(simd::length(world.Velocities[box].Linear) < 1e-2f);
    }
    SUBCASE("a soft locked axis is a torsional spring, and holds both ways") {
        const auto settled = [&](float stiffness, float sign) {
            World world{context};
            const auto pendulum = MakePendulum(world, 1, {.Angular = {AxisLocked, AxisLocked, AxisLocked}, .AngularStiffness = {INFINITY, INFINITY, stiffness}});
            StepSettings settings{};
            settings.Gravity.y *= sign;
            Run(solver, world, 900, settings);
            const auto at = world.Poses[pendulum.Arm].Position;
            const float angle = std::atan2(float(at.y), float(at.x));

            const float balance = -sign * pendulum.Mass * std::abs(settings.Gravity.y) * std::cos(angle) / stiffness;
            CHECK(angle == doctest::Approx(balance).epsilon(0.05).scale(0));
            CHECK(simd::length(world.Velocities[pendulum.Arm].Angular) < 1e-2f);

            const float3 turn = RotationVector(world.Poses[pendulum.Arm].Orientation);
            CHECK(std::abs(turn.x) < 1e-3f);
            CHECK(std::abs(turn.y) < 1e-3f);
            return angle;
        };

        const float down = settled(1e6f, 1), up = settled(1e6f, -1);
        CHECK(down < 0);
        CHECK(up == doctest::Approx(-down).epsilon(0.02));
        CHECK(std::abs(down) > 3 * std::abs(settled(4e6f, 1)));
    }
    SUBCASE("a braked wheel decays at the rate its damping and inertia name") {
        World world{context};
        constexpr float Spin = 10, Damping = 166.667f;
        const StepSettings settings{.Gravity = {0, 0, 0}};
        const auto wheel = SpinOnAxle(world, {.Angular = {AxisLocked, AxisLocked, AxisDriven}, .MotorMaxTorque = {0, 0, INFINITY}, .AngularStiffness = {INFINITY, INFINITY, 0}, .AngularDamping = {0, 0, Damping}}, float3{0, 0, Spin});
        const float per_step = 1 + Damping * settings.DeltaTime / wheel.Inertia;

        for (uint32_t step = 0; step < 120; ++step) {
            solver.Step(world, settings);
            const uint32_t taken = step + 1;
            const float measured = world.Velocities[wheel.Body].Angular.z;
            const float geometric = Spin / std::pow(per_step, float(taken));
            CAPTURE(step);
            CHECK(measured == doctest::Approx(geometric).epsilon(0.001));
            CHECK(measured > 0);
        }

        const float elapsed = 120 * settings.DeltaTime;
        const float measured = world.Velocities[wheel.Body].Angular.z;
        const float exponential = Spin * std::exp(-Damping * elapsed / wheel.Inertia);
        CHECK(measured == doctest::Approx(exponential).epsilon(0.02));
        CHECK(measured > exponential);
    }
    SUBCASE("a drive with damping and no stiffness approaches its speed instead of snapping to it") {
        World world{context};
        constexpr float Target = 5, Damping = 166.667f;
        const StepSettings settings{.Gravity = {0, 0, 0}};
        const auto wheel = SpinOnAxle(world, {.Angular = {AxisLocked, AxisLocked, AxisDriven}, .MotorSpeed = {0, 0, Target}, .MotorMaxTorque = {0, 0, INFINITY}, .AngularStiffness = {INFINITY, INFINITY, 0}, .AngularDamping = {0, 0, Damping}}, float3{0, 0, 0});
        const float per_step = 1 + Damping * settings.DeltaTime / wheel.Inertia;
        const auto approached = [&](uint32_t taken) { return Target * (1 - 1 / std::pow(per_step, float(taken))); };

        solver.Step(world, settings);
        CHECK(world.Velocities[wheel.Body].Angular.z == doctest::Approx(approached(1)).epsilon(0.01));
        CHECK(world.Velocities[wheel.Body].Angular.z < 0.05f * Target);

        Run(solver, world, 59, settings);
        CHECK(world.Velocities[wheel.Body].Angular.z == doctest::Approx(approached(60)).epsilon(0.005));
        CHECK(world.Velocities[wheel.Body].Angular.z < 0.7f * Target);

        Run(solver, world, 240, settings);
        CHECK(world.Velocities[wheel.Body].Angular.z == doctest::Approx(Target).epsilon(0.01));
        CHECK(world.Velocities[wheel.Body].Angular.z < Target);
    }
}

TEST_CASE_FIXTURE(OnDevice, "joints: linear radial and cone limits bound the intended coordinates") {
    SUBCASE("a slider dropped down its axis comes to rest on its stop") {
        World world{context};
        constexpr float Low = -0.5f, High = 0.5f;
        const auto slider = MakeSlider(world, {.Angular = {AxisLocked, AxisLocked, AxisLocked}, .Linear = {AxisLocked, AxisLimited, AxisLocked}, .LinearLimitLow = {0, Low, 0}, .LinearLimitHigh = {0, High, 0}});

        float lowest = 1e9f;
        for (uint32_t step = 0; step < 600; ++step) {
            solver.Step(world);
            lowest = std::min(lowest, float(world.Poses[slider.Box].Position.y));
        }
        const float3 at = world.Poses[slider.Box].Position;
        CHECK(lowest > Low - 0.02f);
        CHECK(lowest < Low);
        CHECK(at.y == doctest::Approx(Low).epsilon(0.002));
        CHECK(std::abs(at.x) < 1e-3f);
        CHECK(std::abs(at.z) < 1e-3f);
        CHECK(simd::length(world.Velocities[slider.Box].Linear) < 1e-2f);
    }
    SUBCASE("integration: a radial limit bounds distance rather than each coordinate") {
        World world{context};
        const auto shape = world.AddShape(UnitBox);
        const auto anchor = world.AddBody({.Shape = shape, .Density = 0});
        const auto body = world.AddBody({.Shape = shape, .Density = 1});
        JointDesc joint{.BodyA = body, .BodyB = anchor, .Linear = {AxisLimited, AxisFree, AxisFree}, .LinearLimitLow = {0, 0, 0}, .LinearLimitHigh = {1, 0, 0}};
        joint.LinearLimitAxes[0] = 7;
        REQUIRE(world.AddJoint(joint) != NoIndex);
        for (int i = 0; i < 240; ++i) solver.Step(world, {.Gravity = {3, 4, 0}});
        const auto at = world.Poses[body].Position;
        CHECK(length(at) == doctest::Approx(1).epsilon(0.005));
        CHECK(at.x == doctest::Approx(0.6f).epsilon(0.01));
        CHECK(at.y == doctest::Approx(0.8f).epsilon(0.01));
    }
    SUBCASE("integration: a cone limit bounds combined swing") {
        World world{context};
        const auto shape = world.AddShape(UnitBox);
        const auto anchor = world.AddBody({.Shape = shape, .Density = 0});
        const auto body = world.AddBody({.Shape = shape, .Density = 1});
        JointDesc joint{.BodyA = body, .BodyB = anchor, .Angular = {AxisFree, AxisLimited, AxisFree}, .LimitLow = {0, 0, 0}, .LimitHigh = {0, 0.4f, 0}};
        joint.AngularLimitAxes[1] = 6;
        joint.Drives[4] = {.Enabled = 1, .Speed = 1, .MaxForce = 1, .Damping = 1};
        joint.Drives[5] = joint.Drives[4];
        REQUIRE(world.AddJoint(joint) != NoIndex);
        for (int i = 0; i < 180; ++i) solver.Step(world, {.Gravity = {0, 0, 0}});
        const auto axis = Rotate(world.Poses[body].Orientation, float3{1, 0, 0});
        CHECK(std::acos(std::clamp(axis.x, -1.f, 1.f)) == doctest::Approx(0.4f).epsilon(0.02));
    }
}

TEST_CASE_FIXTURE(OnDevice, "collision: mesh features preserve support without duplicate rows") {
    SUBCASE("overlapping mesh triangles do not duplicate support or consume contact capacity") {
        World world{context};
        const float3 points[]{{-3, 0, -3}, {0, 0, 3}, {3, 0, -3}};
        std::vector<uint32_t> indices;
        for (uint32_t triangle = 0; triangle < 12; ++triangle) indices.insert(indices.end(), {0, 1, 2});
        world.AddBody({.Shape = world.AddMesh(points, indices), .Density = 0});
        const auto box = world.AddBody({.Pose = At(float3{0, Half, 0}), .Shape = world.AddShape(UnitBox)});
        world.TrackContacts = true;
        for (uint32_t step = 0; step < 180; ++step) {
            solver.Step(world);
            REQUIRE(ActiveContacts(world, box) == ManifoldPoints);
            REQUIRE(world.ContactRefusals[box] == 0);
        }
        CheckResting(world.Poses[box].Position.y);
        CHECK(simd::length(world.Velocities[box].Linear) < 1e-3f);
        float load = 0;
        for (const auto &contact : Slots(world, box))
            if (contact.Active) load -= contact.Lambda.x * contact.Normal.y;
        CHECK(load == doctest::Approx(Gravity / world.Masses[box].InvMass).epsilon(0.01));
    }
    SUBCASE("coincident mesh contact points retain distinct surface normals") {
        World world{context};
        const float3 points[]{{-3, 0, -3}, {0, 0, 3}, {3, 0, -3}, {0, -3, -3}, {0, 3, -3}};
        world.AddBody({.Shape = world.AddMesh(points, std::vector<uint32_t>{0, 1, 2, 3, 4, 1}), .Density = 0});
        const auto box = world.AddBody({.Pose = At(float3{Half, Half, 0}), .Shape = world.AddShape(UnitBox)});
        const StepSettings settings{.Gravity = {-Gravity, -Gravity, 0}};
        for (uint32_t step = 0; step < 180; ++step) {
            solver.Step(world, settings);
            REQUIRE(ActiveContacts(world, box) == 2 * ManifoldPoints);
        }
        float3 load{0, 0, 0};
        for (const auto &contact : Slots(world, box))
            if (contact.Active) load -= contact.Lambda.x * contact.Normal;
        CHECK(load.x == doctest::Approx(Gravity / world.Masses[box].InvMass).epsilon(0.01));
        CHECK(load.y == doctest::Approx(Gravity / world.Masses[box].InvMass).epsilon(0.01));
        CHECK(world.Poses[box].Position.x == doctest::Approx(Half).epsilon(0.005));
        CHECK(world.Poses[box].Position.y == doctest::Approx(Half).epsilon(0.005));
    }
    SUBCASE("a ridge is an edge a body rests on and crosses, not one it catches on") {
        constexpr float Extent = 4, Height = 1.5f;

        SUBCASE("a box set down astride it rests on the crease") {
            World world{context};
            REQUIRE(RidgeMesh(world, Extent, Height) != NoIndex);
            world.AddBody({.Shape = 0});

            const auto box = world.AddBody({.Pose = At(float3{0, Height + Half + 0.2f, 0.13f}), .Shape = world.AddShape(UnitBox)});
            Run(solver, world, 300);
            const Settled settled = Rest(world, box);

            CHECK(settled.Position.y == doctest::Approx(Height + Half).epsilon(5e-3));
            CHECK(std::abs(settled.Position.x) < 0.05f);
            CHECK(settled.Normal.y == doctest::Approx(1).epsilon(1e-2));
            CHECK(simd::length(world.Velocities[box].Linear) < 1e-2f);
            CHECK(world.ContactRefusals[box] == 0);
        }

        SUBCASE("a sphere rolled at it goes over rather than into it") {
            constexpr float Radius = 0.3f;
            World world{context};
            REQUIRE(RidgeMesh(world, Extent, Height) != NoIndex);
            world.AddBody({.Shape = 0});

            const auto ball = world.AddBody({.Pose = At(float3{-3, Height * 0.25f + Radius, 0.11f}), .Velocity = {.Linear = {9, 0, 0}}, .Shape = world.AddShape({.Radius = Radius, .Kind = ShapeSphere})});
            float deepest = 0;
            bool crossed = false;
            for (uint32_t step = 0; step < 240; ++step) {
                solver.Step(world);
                const float3 at = world.Poses[ball].Position;
                crossed = crossed || at.x > 0.5f;

                const float surface = Height * std::max(0.f, 1 - std::abs(float(at.x)) / Extent);
                if (std::abs(at.x) <= Extent) deepest = std::max(deepest, surface + Radius - float(at.y));
            }
            CHECK(crossed);
            CHECK(deepest < 4 * StepSettings{}.ContactMargin);
            CHECK(world.ContactRefusals[ball] == 0);
        }
    }

    SUBCASE("nothing is pushed out of the back of a mesh") {
        World world{context};
        REQUIRE(FloorMesh(world, 4, 5) != NoIndex);
        world.AddBody({.Shape = 0});
        const auto box = world.AddBody({.Pose = At(float3{0.13f, -1, 0.07f}), .Shape = world.AddShape(UnitBox)});

        Run(solver, world, 60);
        CHECK(world.Poses[box].Position.y < -1.f);
        CHECK(Rest(world, box).Contacts == 0);
    }
}

TEST_CASE_FIXTURE(OnDevice, "collision: mesh pairs preserve thin and disconnected surfaces") {
    SUBCASE("crossing thin mesh strips preserve support from either side") {
        World world{context};
        bool below = false;
        SUBCASE("from above") {}
        SUBCASE("from below") { below = true; }
        const float3 floor_points[]{{-0.15f, 0, -2}, {-0.15f, 0, 2}, {0.15f, 0, 2}, {0.15f, 0, -2}};
        const float3 body_points[]{{-2, -0.4f, -0.15f}, {2, -0.4f, -0.15f}, {2, -0.4f, 0.15f}, {-2, -0.4f, 0.15f}};
        const std::vector<uint32_t> indices{0, 1, 2, 0, 2, 3};
        const auto floor = world.AddMesh(floor_points, indices);
        const auto shape = world.AddMesh(body_points, indices);
        world.Shapes[floor].DoubleSided = world.Shapes[shape].DoubleSided = true;
        world.AddBody({.Shape = floor, .Density = 0});
        const auto body = world.AddBody({.Pose = At(float3{0, below ? -1.f : 1.f, 0}), .Shape = shape, .Mass = {{.Mass = 10, .Inertia = {1, 1, 1}}}});
        Run(solver, world, 180, {.Gravity = {0, below ? Gravity : -Gravity, 0}});
        CHECK(world.Poses[body].Position.y == doctest::Approx(0.4f).epsilon(0.01));
        CHECK(simd::length(world.Velocities[body].Linear) < 0.01f);
        CHECK(ActiveContacts(world, body) >= 3);
    }
    SUBCASE("mesh pair collision preserves an empty gap between disconnected feet") {
        World world{context};
        std::vector<float3> points;
        std::vector<uint32_t> indices;
        for (const float side : {-1.f, 1.f}) {
            const uint32_t first = points.size();
            for (const auto point : {float3{side * 0.5f, -0.4f, -0.4f}, float3{side * 0.8f, -0.4f, -0.4f}, float3{side * 0.8f, -0.4f, 0.4f}, float3{side * 0.5f, -0.4f, 0.4f}})
                points.push_back(point);
            indices.insert(indices.end(), {first, first + 1, first + 2, first, first + 2, first + 3});
        }
        const auto feet = world.AddMesh(points, indices);
        world.Shapes[feet].DoubleSided = true;
        const auto pedestal = BoxMesh(world, 0.2f);
        world.AddBody({.Shape = pedestal, .Density = 0});
        const auto body = world.AddBody({.Pose = At(float3{0, 1, 0}), .Shape = feet, .Mass = {{.Mass = 10, .Inertia = {1, 1, 1}}}});
        Run(solver, world, 60);
        CHECK(world.Poses[body].Position.y < -3);
        CHECK(ActiveContacts(world, body) == 0);
    }
    SUBCASE("mesh plane contacts follow local poses and preserve disconnected support") {
        World world{context};
        const Pose scene = At(float3{1, 2, 3}, QuatFromRotationVector(float3{0.2f, 0.1f, 0.3f}));
        const Pose local = At(float3{0, 0.2f, 0}, QuatFromRotationVector(float3{0.3f, 0.2f, 0.1f}));

        std::vector<float3> points;
        std::vector<uint32_t> indices;
        for (const float side : {-1.f, 1.f}) {
            const uint32_t first = points.size();
            for (const auto point : {float3{side * 0.5f, -0.4f, -0.4f}, float3{side * 0.8f, -0.4f, -0.4f}, float3{side * 0.8f, -0.4f, 0.4f}, float3{side * 0.5f, -0.4f, 0.4f}})
                points.push_back(Rotate(QuatConjugate(local.Orientation), point));
            indices.insert(indices.end(), {first, first + 1, first + 2, first, first + 2, first + 3});
        }

        points.push_back(float3{0, -2, 0});
        Index mesh_shape = world.AddMesh(points, indices, local);
        Shape plane = GroundPlane;
        plane.Offset = 0.25f;
        plane.Local = At(float3{0, 0.25f, 0});
        Index plane_shape = world.AddShape(plane);
        bool compound = false, mesh_first = true;
        SUBCASE("separate colliders, mesh first") {}
        SUBCASE("separate colliders, plane first") { mesh_first = false; }
        SUBCASE("compound colliders, mesh first") { compound = true; }
        SUBCASE("compound colliders, plane first") {
            compound = true;
            mesh_first = false;
        }
        SUBCASE("a double-sided plane faces the mesh") {
            world.Shapes[plane_shape].Normal = {0, -1, 0};
            world.Shapes[plane_shape].Offset = -plane.Offset;
            world.Shapes[plane_shape].DoubleSided = 1;
        }
        SUBCASE("a double-sided plane uses the offset mesh geometry") {
            const float3 shift{0, 2, 0};
            for (auto &point : points) point += shift;
            Pose shifted = local;
            shifted.Position -= Rotate(local.Orientation, shift);
            mesh_shape = world.AddMesh(points, indices, shifted);
            world.Shapes[plane_shape].DoubleSided = 1;
        }
        SUBCASE("a bounded double-sided plane supports the thin mesh") {
            world.Shapes[plane_shape].Normal = {0, -1, 0};
            world.Shapes[plane_shape].Offset = -plane.Offset;
            world.Shapes[plane_shape].HalfExtents.x = 0.6f;
            world.Shapes[plane_shape].DoubleSided = 1;
        }
        if (compound) {
            mesh_shape = world.AddCompound(std::vector<Index>{mesh_shape});
            plane_shape = world.AddCompound(std::vector<Index>{plane_shape});
        }
        REQUIRE(mesh_shape != NoIndex);
        REQUIRE(plane_shape != NoIndex);
        const auto add_mesh = [&] {
            return world.AddBody({.Pose = ComposePose(scene, At(float3{0, 2, 0})), .Shape = mesh_shape, .Mass = {{.Mass = 1, .Inertia = {1, 1, 1}}}});
        };
        const auto add_plane = [&] { return world.AddBody({.Pose = scene, .Shape = plane_shape, .Density = 0}); };
        Index mesh = NoIndex, floor = NoIndex;
        if (mesh_first) {
            mesh = add_mesh();
            floor = add_plane();
        } else {
            floor = add_plane();
            mesh = add_mesh();
        }
        const float3 normal = Rotate(scene.Orientation, float3{0, 1, 0});
        Run(solver, world, 300, {.Gravity = -Gravity * normal});
        CHECK(LocalPoint(scene, world.Poses[mesh].Position).y == doctest::Approx(0.7f).epsilon(0.003));
        CHECK(length(world.Velocities[mesh].Linear) < 0.005f);
        CHECK(length(world.Velocities[mesh].Angular) < 0.005f);
        CHECK(ActiveContacts(world, floor) == 0);
        bool left = false, right = false;
        for (const Contact &contact : Slots(world, mesh)) {
            if (!contact.Active) continue;
            CHECK(contact.BodyB == floor);
            const float3 on_mesh = LocalPoint(scene, ContactPoint(world, contact, true));
            left |= on_mesh.x < -0.49f;
            right |= on_mesh.x > 0.49f;
            CHECK(std::abs(on_mesh.x) > 0.49f);
            CHECK(LocalPoint(scene, ContactPoint(world, contact, false)).y == doctest::Approx(0.5f).epsilon(1e-5));
        }
        CHECK(left);
        CHECK(right);
        world.TrackContacts = true;
        solver.Step(world, {.Gravity = -Gravity * normal});
        const auto changes = world.TakeContactChanges();
        REQUIRE_FALSE(changes.empty());
        float support = 0;
        for (const auto &change : changes) {
            CHECK(change.A == world.IdOf(mesh));
            CHECK(change.B == world.IdOf(floor));
            CHECK(dot(change.Normal, normal) > 0.999f);
            support -= change.Lambda.x;
        }
        CHECK(support == doctest::Approx(Gravity).epsilon(0.02));
    }
}

TEST_CASE_FIXTURE(OnDevice, "collision: bounded planes respect edges transforms and live edits") {
    SUBCASE("a plane strip supports spanning bodies and remains unbounded along its length") {
        World world{context};
        const auto floor = world.AddBody({.Shape = world.AddShape({.HalfExtents = {0.3f, 0, 0}, .Normal = {0, 1, 0}, .Kind = ShapePlane}), .Density = 0});
        const auto box = world.AddShape(UnitBox);
        const auto spanning = world.AddBody({.Pose = At(float3{0, 2, 0}), .Shape = box});
        const auto far = world.AddBody({.Pose = At(float3{0, 2, 10000}), .Shape = box});
        const auto outside = world.AddBody({.Pose = At(float3{2, 2, 0}), .Shape = box});
        Run(solver, world, 180);
        for (const auto body : {spanning, far}) {
            CHECK(world.Poses[body].Position.y == doctest::Approx(Half).epsilon(0.004));
            CHECK(length(world.Velocities[body].Linear) < 0.005f);
            REQUIRE(ActiveContacts(world, body) >= 2);
            for (const auto &contact : Slots(world, body)) {
                if (!contact.Active) continue;
                CHECK(contact.BodyB == floor);
                CHECK(std::abs(ContactPoint(world, contact, false).x) <= 0.301f);
            }
        }
        CHECK(world.Poses[outside].Position.y < -2);
    }
    SUBCASE("round shapes meet a plane strip at its face and edge") {
        World world{context};
        const auto plane = world.AddShape({.HalfExtents = {0.3f, 0, 0}, .Normal = {0, 1, 0}, .Kind = ShapePlane});
        world.AddBody({.Shape = plane, .Density = 0});
        SUBCASE("a sphere reaches the edge with its centre outside the strip") {
            const auto sphere = world.AddBody({.Pose = At(float3{0.45f, 0.199f, 0}), .Shape = world.AddShape({.Radius = 0.25f, .Kind = ShapeSphere})});
            solver.Step(world, {.Gravity = {0, 0, 0}});
            REQUIRE(ActiveContacts(world, sphere) == 1);
            const auto &contact = Slots(world, sphere)[0];
            CHECK(ContactPoint(world, contact, false).x == doctest::Approx(0.3f).epsilon(0.005));
            CHECK(std::abs(ContactPoint(world, contact, false).y) < 0.001f);
            CHECK(contact.Normal.x == doctest::Approx(0.6f).epsilon(0.01));
            CHECK(contact.Normal.y == doctest::Approx(0.8f).epsilon(0.01));
        }
        SUBCASE("a capsule spans the strip with both ends outside") {
            const auto capsule = world.AddBody({.Pose = At(float3{0, 2, 0}, QuatFromRotationVector(float3{0, 0, std::numbers::pi_v<float> / 2})), .Shape = world.AddShape({.HalfExtents = {0, 0.8f, 0}, .Radius = 0.2f, .Kind = ShapeCapsule})});
            Run(solver, world, 180);
            CHECK(world.Poses[capsule].Position.y == doctest::Approx(0.2f).epsilon(0.01));
            CHECK(length(world.Velocities[capsule].Linear) < 0.005f);
        }
        SUBCASE("a sphere reaches a finite plane corner") {
            world.Shapes[plane].HalfExtents.z = 0.3f;
            const auto sphere = world.AddBody({.Pose = At(float3{0.45f, 0.1495f, 0.45f}), .Shape = world.AddShape({.Radius = 0.26f, .Kind = ShapeSphere})});
            solver.Step(world, {.Gravity = {0, 0, 0}});
            REQUIRE(ActiveContacts(world, sphere) == 1);
            const auto &contact = Slots(world, sphere)[0];
            CHECK(simd::distance(ContactPoint(world, contact, false), float3{0.3f, 0, 0.3f}) < 0.001f);
            CHECK(simd::distance(contact.Normal, simd::normalize(float3{1, 1, 1})) < 0.005f);
        }
    }

    SUBCASE("plane bounds can change between steps without retaining obsolete triangle contacts") {
        World world{context};
        const auto plane = world.AddShape(GroundPlane);
        world.AddBody({.Shape = plane, .Density = 0});
        const auto mesh = world.AddBody({.Pose = At(float3{0, 0.4f, 0}), .Shape = BoxMesh(world, 0.4f), .Mass = {{.Mass = 10, .Inertia = {1, 1, 1}}}});
        world.TrackContacts = true;
        Run(solver, world, 90);
        (void)world.TakeContactChanges();
        for (const bool bounded : {true, false}) {
            world.Shapes[plane].HalfExtents.x = bounded ? 0.3f : 0;
            world.Wake(mesh);
            solver.Step(world);
            CHECK(world.Poses[mesh].Position.y == doctest::Approx(0.4f).epsilon(0.01));
            REQUIRE(ActiveContacts(world, mesh) >= 2);
            for (const auto &contact : Slots(world, mesh))
                if (contact.Active) CHECK((contact.SubShapeA != NoIndex) == bounded);
            const auto changes = world.TakeContactChanges();
            uint32_t removed = 0;
            for (const auto &change : changes) {
                if (change.Kind != ContactRemoved) continue;
                CHECK((change.SubShapeA != NoIndex) != bounded);
                ++removed;
            }
            CHECK(removed >= 2);
        }
    }
}

TEST_CASE_FIXTURE(OnDevice, "dynamics: authored mass frames and pinned bodies preserve angular motion") {
    SUBCASE("a ball whose mass is off its centre settles with the mass underneath") {
        World world{context};
        AddGround(world);
        constexpr float Radius = 0.5f, Offset = 0.25f, Mass = 500, Inertia = 2.f / 5 * Mass * Radius * Radius;
        const Index shape = world.AddShape({.Radius = Radius, .Kind = ShapeSphere, .Local = {{0, Offset, 0}, {0, 0, 0, 1}}});

        const float4 start = QuatFromRotationVector(float3{0, 0, std::numbers::pi_v<float> / 2});
        REQUIRE(float(Rotate(start, float3{0, Offset, 0}).x) == doctest::Approx(-Offset).epsilon(1e-4));

        const Index body = world.AddBody({.Pose = At(float3{0, Radius, 0}, start), .Shape = shape, .Mass = {{.Mass = Mass, .Inertia = {Inertia, Inertia, Inertia}}}, .LinearDamping = 1.2f, .AngularDamping = 1.2f});
        REQUIRE(body != NoIndex);
        Run(solver, world, 900);

        const Pose pose = world.Poses[body];
        const float3 centre = Rotate(pose.Orientation, float3{0, Offset, 0});
        CAPTURE(centre.x);
        CAPTURE(centre.y);
        CHECK(float(centre.y) > 0.9f * Offset);

        CHECK(float(pose.Position.y) == doctest::Approx(Radius - Offset).epsilon(2e-2));
        CHECK(simd::length(world.Velocities[body].Linear) < 1e-2f);
    }
    SUBCASE("a body of infinite mass hangs where it is put and turns where it hangs") {
        World world{context};
        const auto pinned = world.AddBody({.Pose = At(float3{0, 3, 0}), .Shape = world.AddShape(UnitBox), .Mass = {PinnedCubeMass(10, 1)}});
        REQUIRE(pinned != NoIndex);
        REQUIRE(world.Masses[pinned].InvMass == 0);
        REQUIRE(world.Masses[pinned].InvInertiaLocal.y > 0);
        const Pose was = world.Poses[pinned];
        Run(solver, world, 300);
        CHECK(simd::distance(world.Poses[pinned].Position, was.Position) < 1e-6f);

        world.Velocities[pinned] = {.Angular = {0, 1.5f, 0}};
        Run(solver, world, 120);
        CHECK(float(world.Velocities[pinned].Angular.y) == doctest::Approx(1.5f).epsilon(0.01));
        CHECK(simd::distance(world.Poses[pinned].Position, was.Position) < 1e-6f);
    }
    SUBCASE("a ball striking a pinned body off centre spins it and never moves it") {
        constexpr float BallMass = 5, BallRadius = 0.25f, Offset = 0.25f, Speed = 3, Inertia = 20;
        const StepSettings free_space{.Gravity = {0, 0, 0}};

        const auto strike = [&](bool turns, float friction) {
            World world{context};
            const auto pinned = world.AddBody({.Shape = world.AddShape(UnitBox), .Mass = {{.Mass = 0, .Inertia = turns ? float3{Inertia, Inertia, Inertia} : float3{0, 0, 0}}}, .Friction = friction});
            REQUIRE(pinned != NoIndex);
            REQUIRE(world.Masses[pinned].InvMass == 0);

            const auto ball = world.AddBody({.Pose = At(float3{-Half - BallRadius - 0.3f, Offset, 0}), .Velocity = {.Linear = {Speed, 0, 0}}, .Shape = world.AddShape({.Radius = BallRadius, .Kind = ShapeSphere}), .Mass = {{.Mass = BallMass, .Inertia = {1, 1, 1}}}, .Friction = friction});
            REQUIRE(ball != NoIndex);
            const Pose was = world.Poses[pinned];
            Run(solver, world, 24, free_space);

            const float impulse = BallMass * (Speed - float(world.Velocities[ball].Linear.x));
            return std::tuple{impulse, float(world.Velocities[pinned].Angular.z), float(simd::distance(world.Poses[pinned].Position, was.Position))};
        };

        SUBCASE("frictionless, against the closed form") {
            const auto [impulse, spin, moved] = strike(true, 0);
            CAPTURE(impulse);
            REQUIRE(impulse > 1.f);

            CHECK(spin == doctest::Approx(-Offset * impulse / Inertia).epsilon(0.02));
            CHECK(moved < 1e-6f);
        }
        SUBCASE("with friction, which adds two rows and still moves nothing") {
            const auto [impulse, spin, moved] = strike(true, 0.5f);
            CAPTURE(impulse);
            REQUIRE(impulse > 1.f);
            CHECK(spin < -0.05f);
            CHECK(moved < 1e-6f);
        }
        SUBCASE("with no inertia either, which is the control") {
            const auto [impulse, spin, moved] = strike(false, 0);
            CAPTURE(impulse);
            REQUIRE(impulse > 1.f);
            CHECK(spin == 0);
            CHECK(moved == 0);
        }
    }
}

TEST_CASE_FIXTURE(OnDevice, "collision: compound leaves preserve solid support and seams") {
    constexpr float FloorHalf = 2.5f, FloorTop = 0.25f, Speed = 2, Resting = FloorTop + Half;
    const auto slide = [&](World &world) {
        const Index box = world.AddBody({.Pose = At(float3{-Half - 0.05f, Resting, 0}), .Velocity = {.Linear = {Speed, 0, 0}}, .Shape = world.AddShape(UnitBox), .Density = 1000, .Friction = 0.5f});
        REQUIRE(box != NoIndex);
        return box;
    };

    World jointed{context}, separate{context}, seamless{context};
    std::vector<Index> parts;
    for (const float side : {-1.f, 1.f})
        parts.push_back(jointed.AddShape({.HalfExtents = {FloorHalf, FloorTop, FloorHalf}, .Kind = ShapeBox, .Local = At(float3{side * FloorHalf, 0, 0})}));
    Pose frame{};
    const Index floor = jointed.AddCompound(parts, &frame);
    REQUIRE(floor != NoIndex);

    CHECK(InternalFaces(jointed.Shapes[jointed.Child(floor, 0)]) == 1u << BoxFaceIndex(0, true));
    CHECK(InternalFaces(jointed.Shapes[jointed.Child(floor, 1)]) == 1u << BoxFaceIndex(0, false));
    jointed.AddBody({.Pose = At(frame.Position), .Shape = floor, .Density = 0, .Friction = 0.5f});
    for (const float side : {-1.f, 1.f})
        separate.AddBody({.Pose = At(float3{side * FloorHalf, 0, 0}), .Shape = separate.AddShape({.HalfExtents = {FloorHalf, FloorTop, FloorHalf}, .Kind = ShapeBox}), .Density = 0, .Friction = 0.5f});
    seamless.AddBody({.Shape = seamless.AddShape({.HalfExtents = {2 * FloorHalf, FloorTop, FloorHalf}, .Kind = ShapeBox}), .Density = 0, .Friction = 0.5f});

    const Index over_join = slide(jointed), over_two = slide(separate), over_none = slide(seamless);
    float kick = 0, two_kick = 0;
    for (uint32_t step = 0; step < 240; ++step) {
        solver.Step(jointed, {});
        solver.Step(separate, {});
        solver.Step(seamless, {});
        const float flat = float(seamless.Poses[over_none].Position.y);
        kick = std::max(kick, float(jointed.Poses[over_join].Position.y) - flat);
        two_kick = std::max(two_kick, float(separate.Poses[over_two].Position.y) - flat);
    }

    CHECK(two_kick > 5e-2f);
    CHECK(float(separate.Poses[over_two].Position.x) < -Half);

    CHECK(kick < 0.3f * two_kick);
    CHECK(float(jointed.Poses[over_join].Position.x) > -Half);
    CHECK(float(jointed.Poses[over_join].Position.y) == doctest::Approx(float(seamless.Poses[over_none].Position.y)).epsilon(1e-3));
}

TEST_CASE_FIXTURE(OnDevice, "dynamics: kinematic motion carries strikes and wakes bodies") {
    SUBCASE("a kinematic paddle strikes a ball rather than passing through it") {
        World world{context};
        // Zero gravity isolates momentum transferred by the kinematic paddle.
        constexpr float Swing = 6, Radius = 0.25f;
        const StepSettings settings{.Gravity = {0, 0, 0}};
        const auto paddle = world.AddBody({.Pose = At(float3{-2, 0, 0}), .Shape = world.AddShape({.HalfExtents = {0.1f, 1, 1}, .Kind = ShapeBox}), .Density = 0});
        const auto ball = world.AddBody({.Pose = At(float3{0, 0, 0}), .Shape = world.AddShape({.Radius = Radius, .Kind = ShapeSphere})});

        Run(solver, world, 60, settings);
        REQUIRE(world.Quiet[ball] >= settings.SleepSteps);

        float3 at = world.Poses[paddle].Position;

        DriveAlongX(solver, world, paddle, at, Swing, 90, settings, [&] { CHECK(float(world.Poses[ball].Position.x) + Radius > float(world.Poses[paddle].Position.x)); });

        CHECK(world.Velocities[ball].Linear.x == doctest::Approx(Swing).epsilon(0.01));

        Drive(world, paddle, at, float3(0));
        const float carried = float(world.Poses[ball].Position.x);
        Run(solver, world, 60, settings);
        CHECK(world.Velocities[ball].Linear.x == doctest::Approx(Swing).epsilon(0.01));
        CHECK(world.Poses[ball].Position.x > carried + 0.5f * Swing * 60 * settings.DeltaTime);
    }
    SUBCASE("a sleeping stack wakes when the slab under it starts moving") {
        World world{context};
        const StepSettings settings{};

        constexpr float Speed = 0.5f;
        const auto slab = world.AddBody({.Pose = At(float3{0, -0.25f, 0}), .Shape = world.AddShape({.HalfExtents = {20, 0.25f, 4}, .Kind = ShapeBox}), .Density = 0});
        const auto shape = world.AddShape(UnitBox);
        const std::vector<Index> stack = AddStack(world, shape, 3);
        Run(solver, world, 400);
        for (const auto body : stack) REQUIRE(world.Quiet[body] >= settings.SleepSteps);

        float3 at = world.Poses[slab].Position;
        const auto drive = [&](uint32_t steps) { DriveAlongX(solver, world, slab, at, Speed, steps, settings, [] {}); };

        drive(3);
        for (const auto body : stack) CHECK(world.Quiet[body] == 0u);

        drive(200);
        for (const auto body : stack) CHECK(world.Velocities[body].Linear.x == doctest::Approx(Speed).epsilon(0.02));
        for (uint32_t i = 0; i < stack.size(); ++i) {
            CAPTURE(i);
            CHECK(world.Poses[stack[i]].Position.y == doctest::Approx(Half + float(i) * (1 - settings.ContactMargin)).epsilon(0.01));
        }
    }
}

TEST_CASE_FIXTURE(OnDevice, "collision: materials and exact leaf filters govern contact") {
    SUBCASE("integration: material policies and collider overrides reach the contact") {
        World world{context};
        const Material floor{.StaticFriction = 0.8f, .DynamicFriction = 0.2f, .Restitution = 0.2f, .FrictionCombine = CombineMinimum, .RestitutionCombine = CombineMultiply};
        AddGround(world, {.Surface = floor});
        Shape box = UnitBox;
        box.HasMaterial = true;
        box.Surface = {.StaticFriction = 0.4f, .DynamicFriction = 0.1f, .Restitution = 0.6f, .FrictionCombine = CombineAverage, .RestitutionCombine = CombineMaximum};
        const auto body = world.AddBody({.Pose = At(float3{0, Half, 0}), .Shape = world.AddShape(box), .Surface = Material{}});
        solver.Step(world);
        REQUIRE(ActiveContacts(world, body) > 0);
        for (const auto &contact : world.Contacts.All())
            if (contact.Active) {
                CHECK(contact.Friction == doctest::Approx(0.6f));
                CHECK(contact.Restitution == doctest::Approx(0.6f));
            }
        world.Velocities[body].Linear = {1, 0, 0};
        world.Wake(body);
        solver.Step(world);
        for (const auto &contact : world.Contacts.All())
            if (contact.Active)
                CHECK(contact.Friction == doctest::Approx(0.15f));
    }

    SUBCASE("collider filters: aggregate acceptance still requires an exact mutual leaf match") {
        for (bool sensor : {false, true}) {
            World world{context, {.Bodies = 4, .Shapes = 32}};
            const auto compound = [&](bool other) {
                std::vector<Index> children;
                for (uint32_t i = 0; i < 2; ++i) {
                    Shape shape = UnitBox;
                    shape.Local = At(float3{i ? 0.75f : -0.75f, 0, 0});
                    shape.HasFilter = 1;
                    shape.Mask = other ? CollisionMask{4u << i, i ? 1u : 2u} : CollisionMask{1u << i, 4u << i};
                    children.push_back(world.AddShape(shape));
                }
                return world.AddCompound(children);
            };
            const auto a = compound(false), b = compound(true);
            const auto body_a = world.AddBody({.Shape = a, .Density = 1, .Sensor = sensor});
            const auto body_b = world.AddBody({.Shape = b, .Density = 1});
            solver.Step(world, {.Gravity = {0, 0, 0}});
            CHECK(Allows(world.Filters[body_a].Aggregate, world.Filters[body_b].Aggregate));
            CHECK(ActiveContacts(world, body_a) == 0);
            CHECK(world.Overlaps().empty());

            world.Shapes[world.Child(b, 0)].Mask.Collides = 1;
            world.Shapes[world.Child(b, 1)].Mask.Collides = 2;
            world.Wake(body_a);
            world.Wake(body_b);
            solver.Step(world, {.Gravity = {0, 0, 0}});
            CHECK(world.Filters[body_b].Aggregate.Collides == 3);
            if (sensor) CHECK(world.Overlaps().size() == 2);
            else {
                CHECK(ActiveContacts(world, body_a) > 0);
                for (const Contact &contact : world.Contacts.All())
                    if (contact.Active) CHECK(OwnChild(contact.Children) == OtherChild(contact.Children));
            }
        }
    }
    SUBCASE("collider filters: sleeping contacts end when their collider stops accepting them") {
        World world{context};
        const auto floor = AddGround(world);
        const auto box = DropBox(world, Half, false);
        world.TrackContacts = true;
        solver.Advance(world, {}, 180);
        REQUIRE(world.Quiet[box] >= StepSettings{}.SleepSteps);
        world.TakeContactChanges();
        Shape &shape = world.Shapes[world.BodyShapes[floor]];
        shape.HasFilter = 1;
        shape.Mask = {0, 0};
        solver.Step(world);
        CHECK(ActiveContacts(world, box) == 0);
        const auto changes = world.TakeContactChanges();
        REQUIRE_FALSE(changes.empty());
        for (const auto &change : changes) CHECK(change.Kind == ContactRemoved);
        world.Wake(box);
        solver.Advance(world, {}, 60);
        CHECK(world.Poses[box].Position.y < 0);
    }
    SUBCASE("collider filters: an excluded sibling cannot hide a solid face") {
        World world{context};
        Shape left = UnitBox, right = UnitBox;
        left.Local = At(float3{-0.5f, 0, 0});
        right.Local = At(float3{0.5f, 0, 0});
        left.HasFilter = right.HasFilter = 1;
        left.Mask = {1, 2};
        right.Mask = {4, 8};
        const auto compound = world.AddCompound(std::vector<Index>{world.AddShape(left), world.AddShape(right)});
        REQUIRE(compound != NoIndex);
        REQUIRE(InternalFaces(world.Shapes[world.Child(compound, 0)]) != 0);
        const auto wall = world.AddBody({.Shape = compound, .Density = 0});
        const auto ball = world.AddBody({.Pose = At(float3{0.05f, 0, 0}), .Shape = world.AddShape({.Radius = 0.1f, .Kind = ShapeSphere}), .Density = 1, .Layer = 2, .CollidesWith = 1});
        for (int i = 0; i < 30; ++i) solver.Step(world, {.Gravity = {0, 0, 0}});
        CHECK(world.Filters[wall].Mixed == 1);
        CHECK(world.Poses[ball].Position.x > 0.095f);
    }

    SUBCASE("collider filters: shared compounds inherit each body's defaults independently") {
        World world{context};
        const auto compound = world.AddCompound(std::vector<Index>{world.AddShape(UnitBox)});
        REQUIRE(compound != NoIndex);
        const auto a = world.AddBody({.Shape = compound, .Density = 0, .Layer = 1, .CollidesWith = 2});
        const auto b = world.AddBody({.Shape = compound, .Density = 0, .Layer = 4, .CollidesWith = 8});
        solver.Step(world);
        CHECK(SameMask(world.Filters[a].Aggregate, {1, 2}));
        CHECK(SameMask(world.Filters[b].Aggregate, {4, 8}));
        world.Filters[a].Collides = 16;
        solver.Step(world);
        CHECK(SameMask(world.Filters[a].Aggregate, {1, 16}));
        CHECK(SameMask(world.Filters[b].Aggregate, {4, 8}));
    }
}

TEST_CASE_FIXTURE(OnDevice, "reporting: sensors preserve overlap lifetimes without changing solid motion") {
    SUBCASE("integration: sensors report actual overlap without changing motion") {
        World world{context};
        const auto shape = world.AddShape(UnitBox);
        const auto sensor = world.AddBody({.Shape = shape, .Density = 0, .Sensor = true});
        const auto body = world.AddBody({.Pose = At(float3{-2, 0, 0}), .Velocity = {.Linear = {1, 0, 0}}, .Shape = shape, .Density = 1});
        world.TrackSensors = true;
        const StepSettings settings{.Gravity = {0, 0, 0}};
        solver.Step(world, settings);
        CHECK(world.Overlaps().empty());
        for (int i = 0; i < 100; ++i) solver.Step(world, settings);
        REQUIRE(world.Overlaps().size() == 1);
        REQUIRE(world.TakeSensorChanges().size() == 1);
        CHECK(std::ranges::none_of(world.Contacts.All(), [](const Contact &contact) { return contact.Active; }));
        CHECK(world.Velocities[body].Linear.x == doctest::Approx(1).epsilon(0.001));
        REQUIRE(world.RemoveBody(sensor));
        REQUIRE(world.Overlaps().empty());
        const auto changes = world.TakeSensorChanges();
        REQUIRE(changes.size() == 1);
        CHECK_FALSE(changes[0].Entered);
    }
    SUBCASE("integration: sensors include static pairs and respect filters and teleports") {
        World world{context};
        const auto shape = world.AddShape(UnitBox);
        world.AddBody({.Shape = shape, .Density = 0, .Layer = 1, .CollidesWith = 2, .Sensor = true});
        const auto body = world.AddBody({.Shape = shape, .Density = 0, .Layer = 2, .CollidesWith = 1});
        solver.Step(world);
        REQUIRE(world.Overlaps().size() == 1);
        world.Poses[body].Position = {2, 0, 0};
        solver.Step(world);
        CHECK(world.Overlaps().empty());
        world.Poses[body].Position = {0, 0, 0};
        world.Filters[body].Collides = 0;
        solver.Step(world);
        CHECK(world.Overlaps().empty());
    }
    SUBCASE("integration: coincident sensor leaves retain separate overlap lifetimes") {
        World world{context};
        bool sensor_first = true;
        SUBCASE("sensor owns the pair") {}
        SUBCASE("sensor is the partner") { sensor_first = false; }
        const auto sphere = world.AddShape({.Radius = 0.5f, .Kind = ShapeSphere});
        const auto compound = world.AddCompound(std::vector<Index>{sphere, sphere});
        const BodyDesc sensor_desc{.Shape = compound, .Density = 0, .Sensor = true};
        const BodyDesc probe_desc{.Pose = At(float3{0.75f, 0, 0}), .Shape = sphere, .Density = 0};
        const auto first = world.AddBody(sensor_first ? sensor_desc : probe_desc);
        const auto second = world.AddBody(sensor_first ? probe_desc : sensor_desc);
        const auto sensor = sensor_first ? first : second, probe = sensor_first ? second : first;
        const auto pair = [&](uint32_t leaf) {
            return SensorOverlap{world.IdOf(first), world.IdOf(second), sensor_first ? ChildPair(leaf, 0) : ChildPair(0, leaf)};
        };
        world.TrackSensors = true;
        solver.Step(world);
        REQUIRE(world.Overlaps().size() == 2);
        for (uint32_t leaf = 0; leaf < 2; ++leaf) CHECK(std::ranges::find(world.Overlaps(), pair(leaf)) != world.Overlaps().end());
        CHECK(world.TakeSensorChanges().size() == 2);
        solver.Step(world);
        CHECK(world.TakeSensorChanges().empty());
        auto &leaf = world.Shapes[world.Child(world.BodyShapes[sensor], 1)];
        leaf.HasFilter = 1;
        leaf.Mask.Collides = 0;
        solver.Step(world);
        REQUIRE(world.Overlaps().size() == 1);
        CHECK(world.Overlaps()[0] == pair(0));
        const auto removed = world.TakeSensorChanges();
        REQUIRE(removed.size() == 1);
        CHECK(removed[0].Pair == pair(1));
        CHECK_FALSE(removed[0].Entered);
        world.Poses[probe].Position.x = 2;
        solver.Step(world);
        CHECK(world.Overlaps().empty());
        CHECK(world.TakeSensorChanges().size() == 1);
    }

    SUBCASE("integration: sensor detection preserves solid contact events") {
        World world{context};
        const auto floor = AddGround(world);
        const auto box = DropBox(world, Half, false);
        world.AddBody({.Pose = At(float3{0, Half, 0}), .Shape = world.AddShape(UnitBox), .Density = 0, .Sensor = true});
        world.TrackContacts = world.TrackSensors = true;
        solver.Step(world);
        CHECK_FALSE(world.Overlaps().empty());
        const auto changes = world.TakeContactChanges();
        REQUIRE_FALSE(changes.empty());
        for (const auto &change : changes) {
            CHECK(change.Kind == ContactAdded);
            CHECK(change.A == world.IdOf(box));
            CHECK(change.B == world.IdOf(floor));
        }
    }
}

TEST_CASE_FIXTURE(OnDevice, "execution: live geometry and mixed joint rows survive schedule changes") {
    World reference{context, {.Bodies = 40}}, packed{context, {.Bodies = 40}};
    Index box = NoIndex;
    constexpr JointAxisMode modes[]{AxisFree, AxisLocked, AxisDriven, AxisPositioned, AxisLimited};
    for (World *world : {&reference, &packed}) {
        box = world->AddShape(UnitBox);
        for (uint32_t pair = 0; pair < 16; ++pair) {
            for (uint32_t side = 0; side < 2; ++side) {
                const bool fixed = side == 0 && pair % 3 == 0;
                Pose pose = At(float3{6.f * pair + 1.2f * side, 3, 0});
                pose.Orientation = QuatFromRotationVector(float3{0.02f * pair, 0.05f * side, 0.03f});
                REQUIRE(world->AddBody({.Pose = pose, .Velocity = {.Angular = fixed ? float3{0, 0, 0} : float3{0.04f, 0.03f, 0.02f}}, .Shape = box, .Mass = AuthoredMass{.Mass = fixed ? 0.f : 1.f, .Inertia = fixed ? float3{0, 0, 0} : float3{1, 1, 1}}}) == 2 * pair + side);
            }
            JointDesc joint{.BodyA = 2 * pair + pair % 2, .BodyB = 2 * pair + 1 - pair % 2, .At = {6.f * pair + 0.6f, 3, 0}, .Frame = QuatFromRotationVector(float3{0.1f, 0.2f, 0.3f})};
            for (uint32_t axis = 0; axis < 3; ++axis) {
                joint.Linear[axis] = modes[(pair + axis) % 5];
                joint.Angular[axis] = modes[(pair + axis + 2) % 5];
                joint.LinearMotorSpeed[axis] = 0.03f;
                joint.MotorSpeed[axis] = 0.05f;
                joint.LinearMotorTarget[axis] = joint.MotorTarget[axis] = 0.04f;
                joint.LinearMotorMaxForce[axis] = joint.MotorMaxTorque[axis] = 3;
                joint.LinearLimitLow[axis] = joint.LimitLow[axis] = -0.05f;
                joint.LinearLimitHigh[axis] = joint.LimitHigh[axis] = 0.05f;
                if ((pair + axis) % 2) joint.LinearStiffness[axis] = joint.AngularStiffness[axis] = 100;
                joint.LinearDamping[axis] = joint.AngularDamping[axis] = 0.2f;
            }
            for (uint32_t row = 0; row < 6; ++row)
                joint.Drives[row] = {.Enabled = (pair + row) % 3 != 0, .Speed = 0.05f, .Target = 0.04f, .MaxForce = 3, .Stiffness = row % 2 ? 20.f : 0.f, .Damping = 1};
            REQUIRE(world->AddJoint(joint) == pair);
        }
    }
    const auto add_remote = [&] {
        while (packed.BodyCount() < 35)
            REQUIRE(packed.AddBody({.Pose = At(float3{1000 + 3.f * packed.BodyCount(), 1000, 0}), .Shape = box, .Density = 0}) != NoIndex);
    };
    add_remote();
    for (uint32_t step = 0; step < 120; ++step) {
        CAPTURE(step);
        if (step == 40)
            for (World *world : {&reference, &packed}) {
                REQUIRE(world->RemoveJoint(3));
                REQUIRE(world->RemoveBody(5));
            }
        if (step == 60)
            for (Index body = 32; body < 35; ++body) REQUIRE(packed.RemoveBody(body));
        if (step == 90) {
            for (World *world : {&reference, &packed})
                REQUIRE(world->AddBody({.Pose = At(float3{100, 100, 1}), .Shape = box, .Density = 0}) == 5);
            add_remote();
        }
        solver.Step(reference, {.Gravity = {0, 0, 0}, .SleepSteps = ~0u});
        solver.Step(packed, {.Gravity = {0, 0, 0}, .SleepSteps = ~0u});
        auto actual = Snapshot(packed);
        actual.resize(reference.BodyCount());
        CheckIdentical(Snapshot(reference), actual);
        CHECK(ContactKeys(reference) == ContactKeys(packed));
        for (Index body = 0; body < reference.BodyCount(); ++body) {
            CHECK(std::memcmp(&reference.Velocities[body].Linear, &packed.Velocities[body].Linear, 3 * sizeof(float)) == 0);
            CHECK(std::memcmp(&reference.Velocities[body].Angular, &packed.Velocities[body].Angular, 3 * sizeof(float)) == 0);
        }
        for (Index joint = 0; joint < reference.JointCount(); ++joint) {
            const Joint &a = reference.Joints[joint], &b = packed.Joints[joint];
            CHECK(std::memcmp(&a.LambdaLinear, &b.LambdaLinear, 3 * sizeof(float)) == 0);
            CHECK(std::memcmp(&a.PenaltyLinear, &b.PenaltyLinear, 3 * sizeof(float)) == 0);
            CHECK(std::memcmp(&a.LambdaAngular, &b.LambdaAngular, 3 * sizeof(float)) == 0);
            CHECK(std::memcmp(&a.PenaltyAngular, &b.PenaltyAngular, 3 * sizeof(float)) == 0);
            for (uint32_t row = 0; row < 6; ++row) {
                CHECK(a.Drives[row].Lambda == b.Drives[row].Lambda);
                CHECK(a.Drives[row].Penalty == b.Drives[row].Penalty);
            }
        }
    }
}

TEST_CASE_FIXTURE(OnDevice, "world: contact retirement and slot reuse preserve identity and adjacency") {
    SUBCASE("a shape swapped mid-contact ends its contacts on the stream, on both sides") {
        World world{context};
        world.TrackContacts = true;
        const auto [lower, upper] = TwoOnAPlane(world);
        Run(solver, world, 5);
        REQUIRE(ActiveContacts(world, lower) == 8);
        (void)world.TakeContactChanges();

        REQUIRE(world.SetBodyShape(upper, world.AddShape(UnitBox)));
        const auto changes = world.TakeContactChanges();
        REQUIRE(changes.size() == 4);
        for (const auto &change : changes) {
            CHECK(change.Kind == ContactRemoved);
            CHECK(change.A == world.IdOf(lower));
            CHECK(change.B == world.IdOf(upper));
        }
        Run(solver, world, 1);
        CHECK(Reported(world, lower, ContactAdded) == 4);
        CHECK(Reported(world, lower, ContactPersisted) == 4);
        CHECK(Reported(world, lower, ContactRemoved) == 0);
    }
    SUBCASE("a respawned slot is a different body on the stream") {
        World world{context};
        world.TrackContacts = true;
        const auto [lower, upper] = TwoOnAPlane(world);
        Run(solver, world, 5);
        (void)world.TakeContactChanges();

        const BodyId first = world.IdOf(upper);
        REQUIRE(world.RemoveBody(upper));
        Run(solver, world, 1);
        (void)world.TakeContactChanges();
        const auto again = world.AddBody({.Pose = At(float3{0, Half + 1 - 1e-3f, 0}), .Shape = world.BodyShapes[lower]});
        REQUIRE(again == upper);
        CHECK(world.IdOf(again).Spawn == first.Spawn + 1);

        Run(solver, world, 1);
        const auto changes = world.TakeContactChanges();
        uint32_t added = 0;
        for (const auto &change : changes) {
            CHECK(change.B != first);
            added += change.B == world.IdOf(again) && change.Kind == ContactAdded ? 1 : 0;
        }
        CHECK(added == 4);
    }
    SUBCASE("a slot is not handed out again until a step has reported what left it") {
        World world{context};
        const auto shape = world.AddShape(UnitBox);
        AddGround(world);
        const auto first = world.AddBody({.Pose = At(float3{0, Half, 0}), .Shape = shape});
        world.AddBody({.Pose = At(float3{3, Half, 0}), .Shape = shape});
        Run(solver, world, 30);

        REQUIRE(world.RemoveBody(first));
        CHECK(!world.Alive(first));
        const auto immediate = world.AddBody({.Pose = At(float3{6, Half, 0}), .Shape = shape});
        CHECK(immediate != first);
        Run(solver, world, 1);
        const auto later = world.AddBody({.Pose = At(float3{9, Half, 0}), .Shape = shape});
        CHECK(later == first);
        CHECK(world.Alive(later));
    }
    SUBCASE("all joint partners remain excluded when one body has many joints") {
        World world{context};
        const auto hub = world.AddBody({.Shape = world.AddShape({.HalfExtents = {30, 1, 1}, .Kind = ShapeBox})});
        const auto shape = world.AddShape({.HalfExtents = {0.25f, 0.25f, 0.25f}, .Kind = ShapeBox});
        std::vector<Index> joints;
        Index last = NoIndex;
        for (uint32_t i = 0; i < 24; ++i) {
            last = world.AddBody({.Pose = At(float3{float(i) - 12, 0, 0}), .Shape = shape, .Density = 0});
            joints.push_back(world.AddJoint({.BodyA = hub, .BodyB = last, .Linear = {AxisFree, AxisFree, AxisFree}}));
            REQUIRE(joints.back() != NoIndex);
        }
        const auto duplicate = world.AddJoint({.BodyA = hub, .BodyB = last, .Linear = {AxisFree, AxisFree, AxisFree}});
        REQUIRE(duplicate != NoIndex);
        const StepSettings settings{.Gravity = {0, 0, 0}};
        solver.Step(world, settings);
        CHECK(ActiveContacts(world, hub) == 0);
        CHECK(simd::length(world.Poses[hub].Position) == 0);
        REQUIRE(world.RemoveJoint(joints.back()));
        solver.Step(world, settings);
        CHECK(ActiveContacts(world, hub) == 0);
        REQUIRE(world.RemoveJoint(duplicate));
        solver.Step(world, settings);
        REQUIRE(ActiveContacts(world, hub) > 0);
        for (const auto &contact : Slots(world, hub))
            if (contact.Active) CHECK(contact.BodyB == last);
    }
    SUBCASE("a retired joint stops holding, and lets its bodies collide again") {
        World world{context};
        const StepSettings settings{.Gravity = {0, 0, 0}};
        const auto shape = world.AddShape(UnitBox);
        const auto anchor = world.AddBody({.Shape = shape, .Density = 0});
        const auto hanging = world.AddBody({.Pose = At(float3{0.5f, 0, 0}), .Shape = shape});
        const auto joint = world.AddJoint({.BodyA = hanging, .BodyB = anchor, .At = {0.5f, 0, 0}});
        REQUIRE(joint != NoIndex);
        Run(solver, world, 60, settings);
        REQUIRE(world.Poses[hanging].Position.x == doctest::Approx(0.5f).epsilon(0.01));

        SUBCASE("retired directly") {
            REQUIRE(world.RemoveJoint(joint));
            CHECK(!world.RemoveJoint(joint));
            CHECK(world.JointCount() == 0);
            CHECK(world.Jointed[hanging] == world.Jointed[hanging + 1]);
            CHECK(world.Jointed[anchor] == world.Jointed[anchor + 1]);
            Run(solver, world, 200, settings);
            CHECK(world.Poses[hanging].Position.x > 1 - MaxPenetration);
        }

        SUBCASE("or with the body it held") {
            REQUIRE(world.RemoveBody(hanging));
            CHECK(world.JointCount() == 0);
            CHECK(world.Jointed[anchor] == world.Jointed[anchor + 1]);
        }
    }
    SUBCASE("incoming lists match retained solid contacts through sensors and body reuse") {
        for (uint32_t count : {3u, 35u, 300u}) {
            CAPTURE(count);
            World world{context};
            const auto ground = AddGround(world);
            const auto shape = world.AddShape(UnitBox);
            const BodyDesc desc{.Pose = At(float3{0, Half - 1e-3f, 0}), .Shape = shape};
            Index box = world.AddBody(desc);
            world.AddBody({.Pose = desc.Pose, .Shape = shape, .Density = 0, .Sensor = true});
            while (world.BodyCount() < count) REQUIRE(world.AddBody({.Density = 0}) != NoIndex);
            const StepSettings settings{.Gravity = {0, 0, 0}};
            for (uint32_t step = 0; step < 7; ++step) {
                CAPTURE(step);
                if (step == 1) world.Quiet[box] = settings.SleepSteps;
                if (step == 2) {
                    world.Wake(box);
                    world.Poses[box].Position.y = 100;
                }
                if (step == 3) world.Poses[box] = desc.Pose;
                if (step == 4) REQUIRE(world.RemoveBody(box));
                if (step == 5) {
                    const Index reused = world.AddBody(desc);
                    REQUIRE(reused == box);
                }
                if (step == 6) world.Poses[ground].Position.y = -100;
                solver.Step(world, settings);
                CheckIncoming(world);
                const bool touching = step == 0 || step == 1 || step == 3 || step == 5;
                CHECK(world.Incoming[ground].Count == (touching ? ManifoldPoints : 0));
                if (touching) CHECK_FALSE(world.Overlaps().empty());
            }
        }
    }
}

TEST_CASE_FIXTURE(OnDevice, "dynamics: gravity and damping match discrete closed forms") {
    for (float dt : {1.f / 60, 1.f / 240}) {
        for (const auto [scale, damping] : {std::pair{0.f, 0.f}, {0.5f, 0.f}, {1.f, 0.f}, {2.f, 0.f}, {-1.f, 0.f}, {0.f, 0.1f}, {0.f, 1.5f}}) {
            CAPTURE(dt);
            CAPTURE(scale);
            CAPTURE(damping);
            World world{context};
            const auto body = world.AddBody({.Velocity = {.Linear = {4, 0, 0}, .Angular = {0, 5, 0}}, .Shape = world.AddShape(UnitBox), .GravityScale = scale, .LinearDamping = damping, .AngularDamping = damping});
            constexpr uint32_t Steps = 30;
            const StepSettings settings{.DeltaTime = dt};
            Run(solver, world, Steps, settings);
            const float fraction = std::pow(1 - damping * dt, float(Steps));
            CHECK(world.Velocities[body].Linear.x == doctest::Approx(4 * fraction).epsilon(1e-4));
            CHECK(world.Velocities[body].Angular.y == doctest::Approx(5 * fraction).epsilon(1e-4));
            CHECK(world.Velocities[body].Linear.y == doctest::Approx(scale * Steps * dt * settings.Gravity.y).epsilon(1e-4));
            CHECK(world.Poses[body].Position.y == doctest::Approx(scale * dt * dt * settings.Gravity.y * Steps * (Steps + 1) / 2).epsilon(1e-5));
            CHECK(ActiveContacts(world) == 0);
        }
    }
}

TEST_CASE_FIXTURE(OnDevice, "collision: supported shape matrix settles at geometric heights") {
    const float4 side = QuatFromRotationVector(float3{0, 0, std::numbers::pi_v<float> / 2});
    struct Case {
        Shape ShapeDesc;
        float4 Turn;
        float Height;
        uint32_t Contacts;
    };
    const Case cases[]{
        {UnitBox, {0, 0, 0, 1}, Half, 4},
        {{.Radius = Half, .Kind = ShapeSphere}, {0, 0, 0, 1}, Half, 1},
        {{.HalfExtents = {0, Half, 0}, .Radius = 0.25f, .Kind = ShapeCapsule}, {0, 0, 0, 1}, 0.75f, 1},
        {{.HalfExtents = {0, Half, 0}, .Radius = 0.25f, .Kind = ShapeCapsule}, side, 0.25f, 2},
        {{.HalfExtents = {Half, 0.7f, 0}, .Kind = ShapeCylinder}, {0, 0, 0, 1}, 0.7f, 4},
        {{.HalfExtents = {Half, 0.7f, 0}, .Kind = ShapeCylinder}, side, Half, 2},
        {{.Kind = ShapeHull}, {0, 0, 0, 1}, Half, 4},
        {{.Kind = ShapeMesh}, {0, 0, 0, 1}, Half, 4},
        {{.Kind = ShapeCompound}, {0, 0, 0, 1}, Half, 8}
    };
    for (uint32_t floor : {0u, 1u, 16u}) {
        for (const auto &c : cases) {
            CAPTURE(floor);
            CAPTURE(c.ShapeDesc.Kind);
            CAPTURE(c.Height);
            World world{context};
            AddFloor(world, floor);
            auto shape = c.ShapeDesc.Kind == ShapeHull ? world.AddHull(CubeCorners(1)) : c.ShapeDesc.Kind == ShapeMesh ? BoxMesh(world, Half) :
                                                                                                                         world.AddShape(c.ShapeDesc);
            if (c.ShapeDesc.Kind == ShapeCompound) {
                const auto left = world.AddShape({.HalfExtents = {0.25f, Half, Half}, .Kind = ShapeBox, .Local = At(float3{-0.25f, 0, 0})});
                const auto right = world.AddShape({.HalfExtents = {0.25f, Half, Half}, .Kind = ShapeBox, .Local = At(float3{0.25f, 0, 0})});
                shape = world.AddCompound(std::array{left, right});
            }
            BodyDesc desc{.Pose = At(float3{0.13f, c.Height + (floor == 1 ? -0.01f : 0.1f), 0.07f}, c.Turn), .Shape = shape};
            if (c.ShapeDesc.Kind == ShapeMesh) desc.Mass = AuthoredMass{1000, {1000.f / 6, 1000.f / 6, 1000.f / 6}};
            const auto body = world.AddBody(desc);
            REQUIRE(body != NoIndex);
            solver.Advance(world, {}, 300);
            CHECK(std::abs(world.Poses[body].Position.y - (c.Height - StepSettings{}.ContactMargin)) < 0.002f);
            CHECK(simd::length(world.Velocities[body].Linear) < 0.01f);
            if (c.ShapeDesc.Kind == ShapeCompound) CHECK((LeavesTouching(world, body) == std::set<uint32_t>{0, 1}));
            else if (floor == 0 || (c.ShapeDesc.Kind != ShapeMesh && c.ShapeDesc.Kind != ShapeCylinder)) CHECK(ActiveContacts(world) == c.Contacts);
            else CHECK(ActiveContacts(world) >= c.Contacts);
            float load = 0;
            for (const auto &contact : Slots(world, body))
                if (contact.Active) load -= contact.Lambda.x * contact.Normal.y;
            CHECK(load == doctest::Approx(Gravity / world.Masses[body].InvMass).epsilon(0.01));
            CHECK(world.ContactRefusals[body] == 0);
            if (c.ShapeDesc.Kind == ShapeCylinder && c.Height == 0.7f) {
                const float from = world.Poses[body].Position.x;
                world.Wake(body);
                Run(solver, world, 120, {.Gravity = {1, -9.81f, 0}});
                CHECK(std::abs(world.Poses[body].Position.x - from) < 0.001f);
            }
        }
    }
}

TEST_CASE_FIXTURE(OnDevice, "dynamics: round shapes roll according to their inertia") {
    constexpr float Mu = 0.5f, Slope = 0.3f;
    for (bool capsule : {false, true}) {
        World world{context};
        const float radius = capsule ? 0.25f : 0.5f;
        const Shape shape{.HalfExtents = {0, capsule ? 0.5f : 0.f, 0}, .Radius = radius, .Kind = capsule ? ShapeCapsule : ShapeSphere};
        const auto properties = MassProperties(shape, 1000);
        const float ratio = capsule ? properties.InvMass / properties.InvInertiaLocal.y / (radius * radius) : 2.f / 5;
        REQUIRE(Mu >= ratio / (1 + ratio) * std::tan(Slope));
        AddGround(world, {.Friction = Mu});
        const auto turn = capsule ? QuatFromRotationVector(float3{std::numbers::pi_v<float> / 2, 0, 0}) : float4{0, 0, 0, 1};
        const auto body = world.AddBody({.Pose = At(float3{0, radius, 0}, turn), .Shape = world.AddShape(shape), .Friction = Mu});
        const auto settings = Tilted(Slope);
        Run(solver, world, 30, settings);
        CHECK(AccelerationX(solver, world, body, 60, settings) == doctest::Approx(Gravity * std::sin(Slope) / (1 + ratio)).epsilon(0.05));
        CHECK(world.Velocities[body].Angular.z == doctest::Approx(-world.Velocities[body].Linear.x / radius).epsilon(0.05));
        CHECK(std::abs(world.Poses[body].Position.z) < 0.01f);
    }
}
