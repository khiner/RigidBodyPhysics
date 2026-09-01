// The vertical slice: a box, gravity, a plane, and the whole pipeline on the GPU.
// Every check here is independent of the solver: closed forms, geometric resting heights, and bitwise replays.

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
// A resting body ends slightly inside the surface, since contacts engage a margin early and one stabilization pass removes only part of the error.
// Only this bound on the depth is testable.
constexpr float MaxPenetration = 4 * StepSettings{}.ContactMargin;

// Default gravity as a positive magnitude, the form every closed form below uses.
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

// A stack of `count` bodies at a unit box's pitch whatever the shape, each just above the one below.
std::vector<Index> AddStack(World &world, Index shape, uint32_t count) {
    std::vector<Index> stack;
    for (uint32_t i = 0; i < count; ++i)
        stack.push_back(world.AddBody({.Pose = At(float3{0, Half + 1.02f * float(i), 0}), .Shape = shape}));
    return stack;
}

// A box dropped rotated and moving on every axis, so it tips, lands on a corner and rolls.
// The replay and resting tests run over this asymmetry.
Index TumblingBox(World &world) {
    const auto box = DropBox(world, 1.3f, true);
    world.Poses[box].Orientation = QuatFromRotationVector(float3{0.4f, 0.1f, 0.25f});
    world.Velocities[box] = {.Linear = {0.7f, 0, -0.3f}, .Angular = {0.2f, 1.1f, 0}};
    return box;
}

// Gravity tilted by `slope` about z loads a body exactly as a ramp of that angle would.
// The ground plane stays flat, so the contact is the same one the other tests use.
StepSettings Tilted(float slope) {
    return {.Gravity = {Gravity * std::sin(slope), -Gravity * std::cos(slope), 0}};
}

// How far a shape's geometry reaches from its own origin, which sets the scale below which two of its features are indistinguishable.
float ShapeReach(const World &world, Index shape) {
    if (shape == NoIndex) return 0;
    const Shape &it = world.Shapes[shape];
    // A compound reuses the run fields for its children - see Shape.
    if (it.Kind == ShapeCompound) {
        float reach = 0;
        for (uint32_t i = 0; i < ChildrenPerCompound; ++i) {
            const Index child = ChildOf(it, i);
            if (child == NoIndex) break;
            reach = std::max(reach, simd::length(world.Shapes[child].Local.Position) + ShapeReach(world, child));
        }
        return reach;
    }
    float reach = it.Radius + std::max({std::abs(it.HalfExtents.x), std::abs(it.HalfExtents.y), std::abs(it.HalfExtents.z)});
    for (uint32_t i = 0; i < it.VertexCount; ++i) reach = std::max(reach, simd::length(world.ShapeVertices[it.FirstVertex + i]));
    return reach;
}

// The contact run a body owns, holding every contact where it is body A.
std::span<const Contact> Slots(const World &world, Index body) {
    return world.Contacts.All().subspan(body * ContactsPerBody, ContactsPerBody);
}

// The world position of one end of a contact, with `side` true for body A's anchor and false for body B's.
float3 ContactPoint(const World &world, const Contact &contact, bool side) {
    return WorldPoint(world.Poses[side ? contact.BodyA : contact.BodyB], side ? contact.AnchorA : contact.AnchorB);
}

std::set<uint32_t> LeavesTouching(const World &world, Index body) {
    std::set<uint32_t> leaves;
    for (const Contact &contact : Slots(world, body))
        if (contact.Active) leaves.insert(OwnChild(contact.Children));
    return leaves;
}

// No two rows may reference the same piece of geometry.
// Two points closer together than the narrowphase resolves are one contact written twice, which shows up as ringing and as an overflowing budget.
void CheckManifolds(const World &world) {
    for (uint32_t body = 0; body < world.BodyCount(); ++body) {
        const auto slots = Slots(world, body);
        for (uint32_t i = 0; i < ContactsPerBody; ++i) {
            if (!slots[i].Active) continue;
            for (uint32_t j = i + 1; j < ContactsPerBody; ++j) {
                if (!slots[j].Active || slots[j].BodyB != slots[i].BodyB || slots[j].SubShape != slots[i].SubShape) continue;
                // WeldManifold's own scale, so the check is no stricter than the kernel.
                const float apart = 1e-3f * std::max(ShapeReach(world, world.BodyShapes[body]), ShapeReach(world, world.BodyShapes[slots[i].BodyB])) + 1e-6f;
                const float here = simd::distance(ContactPoint(world, slots[i], true), ContactPoint(world, slots[j], true));
                const float there = simd::distance(ContactPoint(world, slots[i], false), ContactPoint(world, slots[j], false));
                if (here >= apart || there >= apart) continue;
                const Index other = slots[i].BodyB;
                const uint32_t feature_i = slots[i].Feature, feature_j = slots[j].Feature;
                const uint32_t stick_i = slots[i].Stick, stick_j = slots[j].Stick;
                CAPTURE(body);
                CAPTURE(other);
                CAPTURE(here);
                CAPTURE(there);
                CAPTURE(apart);
                CAPTURE(feature_i);
                CAPTURE(feature_j);
                CAPTURE(stick_i);
                CAPTURE(stick_j);
                FAIL_CHECK("two contact rows on one piece of geometry");
            }
        }
    }
}

// The total normal load on a body, over every contact on either side.
// A contact only pushes, so row 0 is at most zero and this returns its magnitude.
float NormalForce(const World &world, Index body) {
    float total = 0;
    for (const Contact &contact : world.Contacts.All())
        if (contact.Active && (contact.BodyA == body || contact.BodyB == body)) total += std::abs(contact.Lambda[0]);
    return total;
}

// Every active contact, keyed the way warm starting keys them.
// The leaf pair is part of the key because two children of one compound can share a feature against the same partner.
using ContactKeySet = std::set<std::tuple<Index, Index, uint32_t, uint32_t>>;

ContactKeySet ContactKeys(const World &world) {
    ContactKeySet keys;
    for (const Contact &contact : world.Contacts.All())
        if (contact.Active) keys.emplace(contact.BodyA, contact.BodyB, contact.Feature, contact.Children);
    return keys;
}

uint32_t Reported(const World &world, Index body, ContactEventKind kind) {
    uint32_t found = 0;
    for (uint32_t i = 0; i < world.ContactEventCounts[body]; ++i)
        found += world.ContactEvents[body * EventsPerBody + i].Kind == kind ? 1 : 0;
    return found;
}

// Nothing in a step depends on thread completion order, so a difference between two runs is a defect rather than a tolerance to widen.
// The comparison runs field by field rather than over the Pose bytes, since a float3's fourth float is padding.
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

// One script run twice, against a fresh world each time.
// The two runs are sequenced explicitly rather than passed as two arguments, since argument evaluation order is unspecified.
void CheckReplay(const mtl::Context &context, auto script) {
    const auto once = [&] {
        World world{context};
        script(world);
        return Snapshot(world);
    };
    const auto first = once(), second = once();
    CheckIdentical(first, second);
}

// The fixture every test below opens with.
// doctest builds a fresh one per test case and per subcase run.
struct OnDevice {
    const mtl::Context context;
    Solver solver{context};
};
struct OneWorld : OnDevice {
    World world{context};
};

void Run(Solver &solver, World &world, uint32_t steps, const StepSettings &settings = {}) {
    for (uint32_t step = 0; step < steps; ++step) {
        solver.Step(world, settings);
        CheckManifolds(world);
    }
}

// Drives a kinematic body along x at `speed`, running `each` after every step.
void DriveAlongX(Solver &solver, World &world, Index body, float3 &at, float speed, uint32_t steps, const StepSettings &settings, auto each) {
    for (uint32_t step = 0; step < steps; ++step) {
        at.x += speed * settings.DeltaTime;
        Drive(world, body, at, float3{speed, 0, 0});
        solver.Step(world, settings);
        CheckManifolds(world);
        each();
    }
}

// The mean acceleration along x over `steps`, measured on a window clear of a landing.
float AccelerationX(Solver &solver, World &world, Index body, uint32_t steps, const StepSettings &settings) {
    const float was = world.Velocities[body].Linear.x;
    Run(solver, world, steps, settings);
    return (world.Velocities[body].Linear.x - was) / (float(steps) * settings.DeltaTime);
}

// A world at rest keeps exactly the contact keys it reached, since a renamed point discards its dual.
void CheckKeysHold(Solver &solver, World &world, const ContactKeySet &settled, uint32_t steps = 300) {
    for (uint32_t step = 0; step < steps; ++step) {
        solver.Step(world);
        CAPTURE(step);
        REQUIRE(ContactKeys(world) == settled);
    }
}
} // namespace

TEST_CASE("a context outlives the worlds and solvers built on it") {
    // A queue accepts thirty-two residency sets in total and both World and Solver take one, so leaking them stops further buffers becoming resident.
    // The failure mode is a step that never returns.
    const mtl::Context context;
    Solver outlives{context}; // one solver spanning every round, the ordinary usage
    for (uint32_t round = 0; round < 40; ++round) {
        CAPTURE(round);
        Solver solver{context};
        World world{context};
        const auto box = DropBox(world, 1, true);
        Run(solver, world, 120);
        CheckResting(world.Poses[box].Position.y);
        // The long-lived solver still steps the same world, so the sets released belonged to the destroyed ones.
        Run(outlives, world, 1);
        CheckResting(world.Poses[box].Position.y);
    }
}

TEST_CASE_FIXTURE(OneWorld, "free fall matches the closed form of the discrete step") {
    const auto box = DropBox(world, 0, false);

    // Implicit Euler over n steps of h puts a body released from rest at h^2 g n (n + 1) / 2, the triangular sum of the velocity gained.
    // This is the discrete result rather than the continuous limit.
    constexpr StepSettings Settings{.DeltaTime = 1.f / 60};
    constexpr uint32_t Steps = 30;
    const float h = Settings.DeltaTime;
    const float expected = h * h * Settings.Gravity.y * float(Steps) * float(Steps + 1) / 2;

    Run(solver, world, Steps, Settings);
    CHECK(world.Poses[box].Position.y == doctest::Approx(expected).epsilon(1e-5));
    CHECK(world.Velocities[box].Linear.y == doctest::Approx(float(Steps) * h * Settings.Gravity.y).epsilon(1e-4));
    CHECK(world.Velocities[box].Linear.x == 0);
    CHECK(world.Poses[box].Position.x == 0);
}

TEST_CASE_FIXTURE(OneWorld, "a body falls at the fraction of gravity its scale names") {
    // The five factors from the KHR MotionProperties sample.
    // A scale multiplies only the gravity applied to the body, so it multiplies the whole closed form above.
    constexpr StepSettings Settings{};
    constexpr uint32_t Steps = 30;
    constexpr float Scales[]{0, 0.5f, 1, 2, -1};
    const float h = Settings.DeltaTime;
    const float fell = h * h * Settings.Gravity.y * float(Steps) * float(Steps + 1) / 2;
    const float sped = float(Steps) * h * Settings.Gravity.y;

    const auto shape = world.AddShape(UnitBox);
    std::vector<Index> boxes;
    for (const float scale : Scales) // in free space and far enough apart never to touch
        boxes.push_back(world.AddBody({.Pose = At(float3{4 * float(boxes.size()), 0, 0}), .Shape = shape, .GravityScale = scale}));

    Run(solver, world, Steps, Settings);
    CHECK(ActiveContacts(world) == 0);
    for (uint32_t i = 0; i < std::size(Scales); ++i) {
        CAPTURE(Scales[i]);
        CHECK(world.Poses[boxes[i]].Position.y == doctest::Approx(Scales[i] * fell).epsilon(1e-5));
        CHECK(world.Velocities[boxes[i]].Linear.y == doctest::Approx(Scales[i] * sped).epsilon(1e-4));
    }
}

TEST_CASE_FIXTURE(OneWorld, "what a resting body weighs its contact down with is its own gravity") {
    // The normal force is the gravity scale times m g, and the boxes start in contact so this is never a landing.
    constexpr float Scales[]{0, 1, 2};

    const auto shape = world.AddShape(UnitBox);
    AddGround(world);
    std::vector<Index> boxes;
    for (const float scale : Scales)
        boxes.push_back(world.AddBody({.Pose = At(float3{4 * float(boxes.size()), Half, 0}), .Shape = shape, .GravityScale = scale}));

    Run(solver, world, 180);
    const float mass = 1 / world.Masses[boxes[1]].InvMass;
    CHECK(world.Poses[boxes[0]].Position.y == doctest::Approx(Half).epsilon(1e-6));
    CHECK(simd::length(world.Velocities[boxes[0]].Linear) < 1e-6f);
    CHECK(ActiveContacts(world, boxes[0]) + ActiveContacts(world, 0) > 0);
    CHECK(NormalForce(world, boxes[0]) == 0);
    for (uint32_t i = 1; i < std::size(Scales); ++i) {
        CAPTURE(Scales[i]);
        CheckResting(world.Poses[boxes[i]].Position.y);
        CHECK(NormalForce(world, boxes[i]) == doctest::Approx(Scales[i] * mass * Gravity).epsilon(0.01));
    }
}

TEST_CASE_FIXTURE(OneWorld, "a damped body keeps the fraction of its speed its coefficient leaves it") {
    // Damping is Jolt's per-step form, v <- v (1 - c h), so a body alone in free space decays as the geometric series (1 - c h)^n exactly.
    // Gravity is off, so the coefficient is the only thing acting.
    constexpr StepSettings Settings{.Gravity = {0, 0, 0}};
    constexpr float Damping = 1.5f, Speed = 4, Spin = 5;
    constexpr uint32_t Steps = 40;
    // Well under the cap, which guards runaway spin and must not affect a damping rate.
    REQUIRE(Spin < Settings.MaxAngularSpeed);

    const auto shape = world.AddShape(UnitBox);
    const auto thrown = world.AddBody({.Velocity = {.Linear = {Speed, 0, 0}}, .Shape = shape, .LinearDamping = Damping});
    const auto spun = world.AddBody({.Pose = At(float3{0, 10, 0}), .Velocity = {.Angular = {0, Spin, 0}}, .Shape = shape, .AngularDamping = Damping});

    Run(solver, world, Steps, Settings);
    const float left = std::pow(1 - Damping * Settings.DeltaTime, float(Steps));
    CHECK(world.Velocities[thrown].Linear.x == doctest::Approx(Speed * left).epsilon(1e-4));
    CHECK(world.Velocities[spun].Angular.y == doctest::Approx(Spin * left).epsilon(1e-4));
    CHECK(simd::length(world.Velocities[thrown].Angular) < 1e-6f);
    CHECK(simd::length(world.Velocities[spun].Linear) < 1e-6f);
    CHECK(std::abs(world.Velocities[spun].Angular.x) < 1e-6f);
    CHECK(std::abs(world.Velocities[spun].Angular.z) < 1e-6f);
}

TEST_CASE_FIXTURE(OneWorld, "a damped bouncing ball keeps its restitution and runs its flights down") {
    // In flight the step is exactly v <- v (1 - c h) + g h, damping first, so gravity arrives undamped.
    constexpr float Damping = 0.8f, e = 0.5f;
    const StepSettings settings{};
    const float h = settings.DeltaTime, gravity = settings.Gravity.y;

    const auto shape = world.AddShape(UnitBox);
    AddGround(world, {.Restitution = e});
    const auto box = world.AddBody({.Pose = At(float3{0, Half + 1, 0}), .Shape = shape, .Restitution = e, .LinearDamping = Damping});

    uint32_t bounces = 0, flights = 0;
    for (uint32_t step = 0; step < 600; ++step) {
        const float before = world.Velocities[box].Linear.y;
        solver.Step(world, settings);
        const float after = world.Velocities[box].Linear.y;
        CAPTURE(step);
        if (ActiveContacts(world) == 0) { // a step spent entirely in flight
            ++flights;
            CHECK(after == doctest::Approx(before * (1 - Damping * h) + gravity * h).epsilon(1e-4));
        } else if (before < 0 && after > 0) {
            ++bounces;
            CAPTURE(bounces);
            CHECK(after == doctest::Approx(e * -before).epsilon(0.02));
        }
    }
    CHECK(bounces >= 3); // the damping leaves it bouncing repeatedly
    CHECK(flights > 40); // the series above was checked over a long flight
    CheckResting(world.Poses[box].Position.y);
}

TEST_CASE_FIXTURE(OneWorld, "a box dropped flat comes to rest on the plane") {
    const auto box = DropBox(world, 2, true);

    Run(solver, world, 180); // three seconds, well past the time it comes to rest
    const auto &pose = world.Poses[box];
    const auto &velocity = world.Velocities[box];

    CheckResting(pose.Position.y);
    CHECK(std::abs(velocity.Linear.y) < 1e-3f);
    CHECK(simd::length(velocity.Angular) < 1e-3f);

    // Dropped level onto a level plane, so no asymmetric force acts on it.
    CHECK(std::abs(pose.Position.x) < 1e-4f);
    CHECK(std::abs(pose.Position.z) < 1e-4f);
    CHECK(simd::length(RotationVector(pose.Orientation)) < 1e-3f);
}

TEST_CASE_FIXTURE(OneWorld, "the resting box holds its four bottom corners in contact") {
    const auto box = DropBox(world, 1, true);
    Run(solver, world, 180);

    const auto slots = Slots(world, box);
    CHECK(ActiveContacts(world, box) == 4); // a level box touches on its bottom face alone

    // A contact's normal force is negative by the reference's convention.
    float total = 0;
    for (const auto &contact : slots) {
        if (!contact.Active) continue;
        CHECK(contact.AnchorA.y == doctest::Approx(-Half));
        CHECK(contact.Lambda[0] < 0);
        total -= contact.Lambda[0];
    }
    // At rest the duals have converged, so the four contacts together support the box's weight, m g.
    CHECK(total == doctest::Approx(1000 * 9.81f).epsilon(0.01));
}

TEST_CASE_FIXTURE(OneWorld, "a stack of boxes holds itself up") {
    // A stack has no closed form, so each box is pinned one box above the one below, with the whole stack at rest.
    constexpr uint32_t Count = 5;
    const StepSettings settings{};
    const auto shape = world.AddShape(UnitBox);
    AddGround(world);
    const std::vector<Index> stack = AddStack(world, shape, Count);

    Run(solver, world, 600); // long enough to expose a stack that is only briefly at rest
    // The plane acts as the box below the first one, with its centre a half-extent under the surface.
    float below = -Half;
    for (uint32_t i = 0; i < Count; ++i) {
        CAPTURE(i);
        const float height = world.Poses[stack[i]].Position.y;
        // One box up, less the margin contacts engage at.
        CHECK(std::abs(height - (below + 1 - settings.ContactMargin)) < 1e-3f);
        CHECK(simd::length(world.Velocities[stack[i]].Linear) < 0.02f);
        below = height;
    }
}

TEST_CASE_FIXTURE(OneWorld, "a box lands on another box's edge and is held by an edge contact") {
    // Crossed edges, so the contact normal is the cross product of the two edge directions, one of the separating axis test's nine cross-product axes.
    // Each cube reaches half its face diagonal.
    constexpr float Diagonal = 0.70710678f; // Half * sqrt(2)
    const auto shape = world.AddShape(UnitBox);
    const float quarter = std::atan(1.f); // pi / 4
    world.AddBody({.Pose = At(float3{0, 0, 0}, QuatFromRotationVector(float3{0, 0, quarter})), .Shape = shape, .Density = 0}); // static, so only the upper box falls
    const auto box = world.AddBody({.Pose = At(float3{0, 2 * Diagonal + 0.25f, 0}, QuatFromRotationVector(float3{quarter, 0, 0})), .Shape = shape});

    // Balancing on crossing edges is unstable, so the check is on the contact at the moment it forms.
    uint32_t contacts = 0;
    for (uint32_t step = 0; step < 60 && contacts == 0; ++step) {
        solver.Step(world);
        for (const Contact &contact : world.Contacts.All()) {
            if (!contact.Active) continue;
            ++contacts;
            CHECK(contact.Normal.y == doctest::Approx(1).epsilon(0.01)); // straight up out of the lower box
        }
    }
    CHECK(contacts == 1); // two crossing edges meet at a single point
    // Contacts are found at the pose the step began from, so a falling box first registers a touch up to one step of its own motion past the meeting point.
    // This is a bound rather than a fitted tolerance.
    const float step = 2.2f * StepSettings{}.DeltaTime;
    CHECK(world.Poses[box].Position.y < 2 * Diagonal + StepSettings{}.ContactMargin);
    CHECK(world.Poses[box].Position.y > 2 * Diagonal - step);

    Run(solver, world, 30);
    CHECK(world.Poses[box].Position.y > 2 * Diagonal - 0.05f);
}

// A step proud of the resting depth acts as a wall and stops the slider, and a floor of one piece is the control.
namespace {
constexpr float FloorHalf = 2.5f, FloorTop = 0.25f, SlideSpeed = 2, SlideMu = 0.5f;
// Just short of the join, since mu g stops the box in 0.41 m and it has to cross the join over that distance.
constexpr float SlideFrom = -Half - 0.05f;

// The slider, resting on a floor whose top is at FloorTop and already moving toward the join.
Index AddSlider(World &world) {
    return world.AddBody({.Pose = At(float3{SlideFrom, FloorTop + Half, 0}), .Velocity = {.Linear = {SlideSpeed, 0, 0}},
                          .Shape = world.AddShape(UnitBox), .Friction = SlideMu});
}
} // namespace

TEST_CASE_FIXTURE(OneWorld, "a box slid at a step proud of the resting depth is stopped by it") {
    // A contact rests ContactMargin inside the face supporting it.
    // For this to be a step rather than a join, the far box has to stand proud of that resting depth.
    // Here it stands proud by twice the margin.
    const float step = 2 * StepSettings{}.ContactMargin;
    for (const float side : {-1.f, 1.f})
        world.AddBody({.Pose = At(float3{side * FloorHalf, side > 0 ? step : 0, 0}),
                       .Shape = world.AddShape({.HalfExtents = {FloorHalf, FloorTop, FloorHalf}, .Kind = ShapeBox}), .Density = 0, .Friction = SlideMu});
    const Index slider = AddSlider(world);

    Run(solver, world, 240);
    CHECK(world.Velocities[slider].Linear.x < 0.05f);
    CHECK(world.Poses[slider].Position.y < FloorTop + Half + step);
    CHECK(world.Poses[slider].Position.x < 0);
}

TEST_CASE_FIXTURE(OneWorld, "a box slid across a floor of one piece stops where Coulomb says") {
    // The same slide over a floor of one piece.
    // Coulomb puts the stopping distance at v^2 / 2 mu g, which the discrete step falls slightly short of.
    world.AddBody({.Shape = world.AddShape({.HalfExtents = {2 * FloorHalf, FloorTop, FloorHalf}, .Kind = ShapeBox}), .Density = 0, .Friction = SlideMu});
    const Index slider = AddSlider(world);

    Run(solver, world, 240);
    const float coulomb = SlideSpeed * SlideSpeed / (2 * SlideMu * Gravity);
    CHECK(world.Velocities[slider].Linear.x < 0.01f);
    CHECK(world.Poses[slider].Position.x - SlideFrom == doctest::Approx(coulomb).epsilon(0.05));
}

TEST_CASE_FIXTURE(OneWorld, "a welded join between two static boxes is not a wall") {
    // The same two boxes with their tops in one plane, welded.
    // Both are static, so the face they share is interior to the combined solid and the narrowphase drops any manifold whose normal lies on it.
    for (const float side : {-1.f, 1.f})
        world.AddBody({.Pose = At(float3{side * FloorHalf, 0, 0}),
                       .Shape = world.AddShape({.HalfExtents = {FloorHalf, FloorTop, FloorHalf}, .Kind = ShapeBox}), .Density = 0, .Friction = SlideMu});
    CHECK(world.WeldStatic() == 2); // one face of each box, the face the other covers
    const Index slider = AddSlider(world);

    const float resting = FloorTop + Half;
    float kick = 0;
    for (uint32_t step = 0; step < 240; ++step) {
        solver.Step(world);
        CheckManifolds(world);
        kick = std::max(kick, float(world.Poses[slider].Position.y) - resting);
    }
    // Well past where an unwelded join stops it.
    // The remaining kick is the engine's own at a box join, since two separate pairs cannot collapse the seam rows the way a compound does.
    CHECK(world.Poses[slider].Position.x - SlideFrom > 0.2f);
    CHECK(kick < 1.6e-2f);
    CHECK(float(world.Poses[slider].Position.y) == doctest::Approx(resting).epsilon(1e-3));
}

TEST_CASE_FIXTURE(OneWorld, "a box resting on a welded join sits level on it and sleeps") {
    // Straddling the join rather than crossing it.
    // The weld drops only the manifolds on the faces the boxes bury in each other, so both tops still support the box over the crack between them.
    for (const float side : {-1.f, 1.f})
        world.AddBody({.Pose = At(float3{side * FloorHalf, 0, 0}),
                       .Shape = world.AddShape({.HalfExtents = {FloorHalf, FloorTop, FloorHalf}, .Kind = ShapeBox}), .Density = 0, .Friction = SlideMu});
    REQUIRE(world.WeldStatic() == 2);
    const Index box = world.AddBody({.Pose = At(float3{0, FloorTop + Half + 1e-3f, 0}), .Shape = world.AddShape(UnitBox), .Friction = SlideMu});
    REQUIRE(box != NoIndex);

    Run(solver, world, 200);
    CheckResting(float(world.Poses[box].Position.y) - FloorTop);
    CHECK(simd::length(RotationVector(world.Poses[box].Orientation)) < 1e-4f);
    CHECK(world.Quiet[box] >= StepSettings{}.SleepSteps);
    CHECK(ActiveContacts(world, box) == 2 * ManifoldPoints); // four points on each box's top, and none on the join
}

TEST_CASE_FIXTURE(OneWorld, "a face the weld buried is a face again once what buried it goes") {
    // The wholly-covered rule: the leg's top is inside the slab, and most of the slab's bottom is open air.
    constexpr float LegHalf = 0.75f, LegTop = 1;
    const Index leg = world.AddBody({.Pose = At(float3{0, LegTop - LegHalf, 0}),
                                     .Shape = world.AddShape({.HalfExtents = {LegHalf, LegHalf, LegHalf}, .Kind = ShapeBox}), .Density = 0, .Friction = SlideMu});
    const Index slab = world.AddBody({.Pose = At(float3{0, LegTop + 0.25f, 0}),
                                      .Shape = world.AddShape({.HalfExtents = {2, 0.25f, 2}, .Kind = ShapeBox}), .Density = 0, .Friction = SlideMu});
    REQUIRE(slab != NoIndex);
    CHECK(world.WeldStatic() == 1); // the leg's top alone
    CHECK(InternalFaces(world.Shapes[world.BodyShapes[leg]]) == 1u << BoxFaceIndex(1, true));

    REQUIRE(world.RemoveBody(slab));
    CHECK(world.WeldStatic() == 0);
    CHECK(InternalFaces(world.Shapes[world.BodyShapes[leg]]) == 0);
    // With the face still buried, the manifold would be dropped every step and the box would fall through the leg.
    const Index box = world.AddBody({.Pose = At(float3{0, LegTop + Half + 0.05f, 0}), .Shape = world.AddShape(UnitBox), .Friction = SlideMu});
    REQUIRE(box != NoIndex);
    Run(solver, world, 200);
    CheckResting(float(world.Poses[box].Position.y) - LegTop);
}

TEST_CASE_FIXTURE(OnDevice, "a bounce leaves at the fraction of its arrival speed restitution names") {
    // Restitution is defined on speeds rather than heights: a body arriving at v leaves at e v.

    struct Bounce {
        float Arrived{}, Left{}, Rose{};
    };
    const auto drop = [&](float e) {
        World world{context};
        const auto shape = world.AddShape(UnitBox);
        AddGround(world, {.Restitution = e});
        const auto box = world.AddBody({.Pose = At(float3{0, Half + 1, 0}), .Shape = shape, .Restitution = e});

        Bounce out;
        float from = 0;
        for (uint32_t step = 0; step < 240; ++step) {
            const float before = world.Velocities[box].Linear.y;
            solver.Step(world);
            const float now = world.Velocities[box].Linear.y;
            const float height = world.Poses[box].Position.y - Half;
            if (out.Left == 0 && before < 0 && now > 0) { // the step the bounce happened on
                out.Arrived = -before;
                out.Left = now;
                from = height;
            }
            if (out.Left != 0) out.Rose = std::max(out.Rose, height - from);
        }
        return out;
    };

    for (const float e : {0.5f, 0.8f}) {
        CAPTURE(e);
        const auto bounce = drop(e);
        // Fast enough to bounce at all, the threshold the step and gravity set.
        const StepSettings settings{};
        CHECK(bounce.Arrived > settings.BounceSpeedFactor * simd::length(settings.Gravity) * settings.DeltaTime);
        CHECK(bounce.Left == doctest::Approx(e * bounce.Arrived).epsilon(0.02));
        // The rise follows from the departure speed, less the part the first step already spent.
        CHECK(bounce.Rose == doctest::Approx(bounce.Left * bounce.Left / (2 * Gravity)).epsilon(0.1));
    }
    // A box coming to rest rebounds off its own penetration, three orders below a bounce, so the result is small rather than exactly zero.
    CHECK(drop(0).Left < 0.05f);
}

TEST_CASE_FIXTURE(OnDevice, "a body fired at a wall never ends up behind it") {
    // A reach covering a step of motion catches a body that would otherwise pass through the wall between steps.
    // A mesh is the important case, having no back face, and the shot is level so gravity plays no part.
    constexpr float Size = 0.05f, Thickness = 0.02f, From = -1;
    const StepSettings settings{.Gravity = {0, 0, 0}};

    // A wall of two triangles in the y-z plane, wound so its face points back down the line of fire.
    const std::vector<float3> points{float3{0, -2, -2}, float3{0, -2, 2}, float3{0, 2, 2}, float3{0, 2, -2}};
    const std::vector<uint32_t> indices{0, 1, 2, 0, 2, 3};

    const auto fire = [&](float speed, bool meshed) {
        World world{context};
        const Index wall = meshed ? world.AddMesh(points, indices) : world.AddShape({.HalfExtents = {Thickness / 2, 2, 2}, .Kind = ShapeBox});
        REQUIRE(wall != NoIndex);
        world.AddBody({.Shape = wall, .Density = 0});
        const auto shot = world.AddBody({.Pose = At(float3{From, 0, 0}), .Velocity = {.Linear = {speed, 0, 0}}, .Shape = world.AddShape({.HalfExtents = {Size, Size, Size}, .Kind = ShapeBox})});
        float deepest = -INFINITY;
        // Long enough to carry it well past the wall at this speed with no contact.
        for (uint32_t step = 0; step < uint32_t(3 / (speed * settings.DeltaTime)) + 60; ++step) {
            solver.Step(world, settings);
            deepest = std::max(deepest, float(world.Poses[shot].Position.x) + Size - (meshed ? 0 : -Thickness / 2));
        }
        return deepest;
    };

    for (const float speed : {5.f, 20.f, 60.f, 120.f, 250.f, 500.f}) {
        CAPTURE(speed);
        // Every speed here travels further in a step than the wall is thick, which is the case under test.
        REQUIRE(speed * settings.DeltaTime > Thickness);
        for (const bool meshed : {false, true}) {
            CAPTURE(meshed);
            // It stops at the margin every other contact rests at, rather than a step's travel beyond it.
            CHECK(fire(speed, meshed) < MaxPenetration);
        }
    }
}

TEST_CASE_FIXTURE(OneWorld, "a bounce series keeps its ratio with speculation active") {
    // Restitution is a velocity pass after the solve rather than a row, since one displacement per step cannot express both an approach and a rebound.
    // The ratio still has to survive bounce after bounce and come to rest.
    constexpr float e = 0.5f;
    const auto shape = world.AddShape(UnitBox);
    AddGround(world, {.Restitution = e});
    const auto box = world.AddBody({.Pose = At(float3{0, Half + 1, 0}), .Shape = shape, .Restitution = e});

    uint32_t bounces = 0;
    for (uint32_t step = 0; step < 900; ++step) {
        const float before = world.Velocities[box].Linear.y;
        solver.Step(world);
        const float after = world.Velocities[box].Linear.y;
        if (before >= 0 || after <= 0) continue; // not a step a bounce happened on
        ++bounces;
        CAPTURE(bounces);
        CAPTURE(-before);
        CHECK(after == doctest::Approx(e * -before).epsilon(0.02));
    }
    CHECK(bounces >= 3); // it bounces repeatedly rather than stopping on the first
    CheckResting(world.Poses[box].Position.y);
    CHECK(simd::length(world.Velocities[box].Linear) < 1e-2f);
}

TEST_CASE_FIXTURE(OneWorld, "a contact across a gap holds its bodies apart and pushes nothing") {
    // A speculative contact exists, so it warm starts and is reported, and it applies no force until the step's motion would consume its gap.
    // The box is thrown across the plane, since a fall closes the gap in one step.
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
            CHECK(contact.Lambda[0] == 0); // apart, so the contact applies no force
            CHECK(contact.C0[0] > StepSettings{}.ContactMargin); // C0 holds the gap it measured
        }
    }
    CHECK(apart > 0); // some steps held a contact across a gap, so the branch above ran
    CHECK(touching > 0); // it goes on to land rather than being held off the plane by a ghost contact
    CheckResting(world.Poses[box].Position.y);
}

TEST_CASE_FIXTURE(OneWorld, "a bouncy box still comes to rest") {
    // The bounce speed threshold makes this hold: a body coming to rest on its own jitter falls below it.
    const auto shape = world.AddShape(UnitBox);
    AddGround(world, {.Restitution = 0.6f});
    const auto box = world.AddBody({.Pose = At(float3{0, Half + 1, 0}), .Shape = shape, .Restitution = 0.6f});

    Run(solver, world, 900);
    CheckResting(world.Poses[box].Position.y);
    CHECK(simd::length(world.Velocities[box].Linear) < 1e-2f);
}

TEST_CASE_FIXTURE(OneWorld, "spheres rest where their radii put them") {
    // A sphere touches at one point, so every rest height is a sum of radii less the contact margin.
    constexpr float Radius = 0.5f, Margin = StepSettings{}.ContactMargin;
    constexpr Shape Ball{.Radius = Radius, .Kind = ShapeSphere};

    SUBCASE("on a plane, at its radius") {
        const auto ball = world.AddShape(Ball);
        AddGround(world);
        const auto sphere = world.AddBody({.Pose = At(float3{0, 1.5f, 0}), .Shape = ball});

        Run(solver, world, 300);
        CHECK(std::abs(world.Poses[sphere].Position.y - (Radius - Margin)) < 2e-3f);
        CHECK(simd::length(world.Velocities[sphere].Linear) < 1e-2f);
    }

    SUBCASE("on another sphere, at the sum of theirs") {
        const auto ball = world.AddShape(Ball);
        AddGround(world);
        // The lower sphere is static, since a sphere balanced on a sphere is an unstable equilibrium.
        world.AddBody({.Pose = At(float3{0, Radius, 0}), .Shape = ball, .Density = 0});
        const auto top = world.AddBody({.Pose = At(float3{0, 3 * Radius + 0.2f, 0}), .Shape = ball});

        Run(solver, world, 300);
        CHECK(std::abs(world.Poses[top].Position.y - (3 * Radius - Margin)) < 2e-3f);
    }

    SUBCASE("on a box, at its radius above the face") {
        const auto ball = world.AddShape(Ball);
        const auto box = world.AddShape(UnitBox);
        AddGround(world);
        world.AddBody({.Pose = At(float3{0, Half, 0}), .Shape = box, .Density = 0});
        const auto sphere = world.AddBody({.Pose = At(float3{0, 2 * Half + Radius + 0.2f, 0}), .Shape = ball});

        Run(solver, world, 300);
        CHECK(std::abs(world.Poses[sphere].Position.y - (2 * Half + Radius - Margin)) < 2e-3f);
        CHECK(std::abs(world.Poses[sphere].Position.x) < 1e-2f);
    }
}

TEST_CASE_FIXTURE(OneWorld, "a sphere rolls down a slope at the rate rolling without slipping implies") {
    // Rolling puts some energy into spin, so it accelerates at g sin(slope) / (1 + 2/5), with 2/5 the sphere's inertia coefficient.
    // Friction is the only off-centre torque, so rolling needs mu >= (2/7) tan(slope).
    constexpr float Radius = 0.5f, Mu = 0.5f, Slope = 0.3f;
    const float expected = 5.f / 7 * Gravity * std::sin(Slope);
    REQUIRE(Mu >= 2.f / 7 * std::tan(Slope)); // below this it slips and a different closed form applies

    const auto settings = Tilted(Slope);
    const auto ball = world.AddShape(Shape{.Radius = Radius, .Kind = ShapeSphere});
    AddGround(world, {.Friction = Mu});
    const auto sphere = world.AddBody({.Pose = At(float3{0, Radius, 0}), .Shape = ball, .Friction = Mu});

    Run(solver, world, 30, settings); // reach the plane before timing anything
    CHECK(AccelerationX(solver, world, sphere, 60, settings) == doctest::Approx(expected).epsilon(0.05));

    // Rolling without slipping: the surface speed at the contact matches the centre's, about -z here.
    const float speed = world.Velocities[sphere].Linear.x;
    CHECK(world.Velocities[sphere].Angular.z == doctest::Approx(-speed / Radius).epsilon(0.05));
}

TEST_CASE_FIXTURE(OneWorld, "capsules rest where their radius puts them, on whatever they land on") {
    // A capsule is every point within Radius of a segment, so it rests that radius from whatever the segment is nearest.
    // The contact count is the point here: two lying down, one on end.
    constexpr float Radius = 0.25f, HalfLength = 0.5f, Margin = StepSettings{}.ContactMargin;
    constexpr Shape Pill{.HalfExtents = {0, HalfLength, 0}, .Radius = Radius, .Kind = ShapeCapsule};
    const float quarter = 2 * std::atan(1.f); // the capsule's length runs along its own y, so this lays it down

    SUBCASE("lying on a plane, on two contacts") {
        const auto pill = world.AddShape(Pill);
        AddGround(world);
        const auto capsule = world.AddBody({.Pose = At(float3{0, 0.6f, 0}, QuatFromRotationVector(float3{0, 0, quarter})), .Shape = pill});

        Run(solver, world, 300);
        CHECK(std::abs(world.Poses[capsule].Position.y - (Radius - Margin)) < 2e-3f);
        CHECK(ActiveContacts(world) == 2); // one under each end of its length
        CHECK(simd::length(world.Velocities[capsule].Linear) < 1e-2f);
    }

    SUBCASE("standing on a plane, on one") {
        const auto pill = world.AddShape(Pill);
        AddGround(world);
        const auto capsule = world.AddBody({.Pose = At(float3{0, 1, 0}), .Shape = pill});

        Run(solver, world, 300);
        CHECK(std::abs(world.Poses[capsule].Position.y - (HalfLength + Radius - Margin)) < 2e-3f);
        CHECK(ActiveContacts(world) == 1); // one cap, and a cap is a sphere
    }

    SUBCASE("across a box, with both ends out over nothing") {
        // The capsule meets the box between its ends, so sampling only the ends would let it pass through.
        constexpr Shape Long{.HalfExtents = {0, 2, 0}, .Radius = 0.1f, .Kind = ShapeCapsule};
        const auto pill = world.AddShape(Long);
        const auto box = world.AddShape(UnitBox);
        AddGround(world);
        world.AddBody({.Pose = At(float3{0, Half, 0}), .Shape = box, .Density = 0});
        const auto capsule = world.AddBody({.Pose = At(float3{0, 1.3f, 0}, QuatFromRotationVector(float3{0, 0, quarter})), .Shape = pill});

        Run(solver, world, 300);
        CHECK(std::abs(world.Poses[capsule].Position.y - (2 * Half + 0.1f - Margin)) < 2e-3f);
        CHECK(ActiveContacts(world) == 2); // where its length crosses the two edges of the box's top face

        // The pair keeps its identity, keyed by the box faces that bounded it.
        CheckKeysHold(solver, world, ContactKeys(world));
    }
}

TEST_CASE_FIXTURE(OneWorld, "a capsule on a slope along its own length holds still") {
    // Lying along the slope a capsule cannot roll, so friction alone keeps it in place.
    // A round contact inherits no static-friction anchors, and a capsule that is not rolling stays put regardless.
    constexpr float Radius = 0.25f, Mu = 0.5f, Slope = 0.3f;
    REQUIRE(std::tan(Slope) < Mu); // inside the friction cone, so it stays where it is
    constexpr Shape Pill{.HalfExtents = {0, 0.5f, 0}, .Radius = Radius, .Kind = ShapeCapsule};

    const auto settings = Tilted(Slope);
    const auto pill = world.AddShape(Pill);
    AddGround(world, {.Friction = Mu});
    const auto capsule = world.AddBody({.Pose = At(float3{0, Radius, 0}, QuatFromRotationVector(float3{0, 0, 2 * std::atan(1.f)})), .Shape = pill, .Friction = Mu});

    Run(solver, world, 30, settings);
    const float from = world.Poses[capsule].Position.x;
    Run(solver, world, 900, settings); // fifteen seconds is long enough for a creep to show
    CHECK(std::abs(world.Poses[capsule].Position.x - from) < 1e-3f);
    CHECK(simd::length(world.Velocities[capsule].Linear) < 1e-3f);
}

TEST_CASE_FIXTURE(OneWorld, "a capsule rolls down a slope at the rate its own inertia implies") {
    // The sphere's law with the capsule's own inertia in place of 2/5 m r^2, so it accelerates at g sin(slope) / (1 + I / m r^2).
    // A capsule rolls across its length, so it lies along z.
    constexpr float Radius = 0.25f, HalfLength = 0.5f, Mu = 0.5f, Slope = 0.3f;
    constexpr Shape Pill{.HalfExtents = {0, HalfLength, 0}, .Radius = Radius, .Kind = ShapeCapsule};
    const auto properties = MassProperties(Pill, 1000);
    const float ratio = (1 / properties.InvInertiaLocal[1]) * properties.InvMass / (Radius * Radius);
    const float expected = Gravity * std::sin(Slope) / (1 + ratio);
    REQUIRE(Mu >= ratio / (1 + ratio) * std::tan(Slope)); // enough friction that it rolls rather than slips

    const auto settings = Tilted(Slope);
    const auto pill = world.AddShape(Pill);
    AddGround(world, {.Friction = Mu});
    const auto capsule = world.AddBody({.Pose = At(float3{0, Radius, 0}, QuatFromRotationVector(float3{2 * std::atan(1.f), 0, 0})), .Shape = pill, .Friction = Mu});

    Run(solver, world, 30, settings);
    CHECK(AccelerationX(solver, world, capsule, 60, settings) == doctest::Approx(expected).epsilon(0.05));
    CHECK(world.Velocities[capsule].Angular.z == doctest::Approx(-world.Velocities[capsule].Linear.x / Radius).epsilon(0.05));
    CHECK(std::abs(world.Poses[capsule].Position.z) < 1e-2f);
}

// A body on a pivot: a shapeless zero-mass body, and an arm jointed to it `distance` away along x.
struct Pendulum {
    World &Bodies;
    Index Arm;
    float Inertia; // about the pivot, the I the closed forms below use
    float Mass;
};

Pendulum MakePendulum(World &world, float distance, JointDesc joint = {}) {
    const auto shape = world.AddShape(UnitBox);
    const auto pivot = world.AddBody({}); // no shape, so no mass and no contacts: a bare pivot
    const auto arm = world.AddBody({.Pose = At(float3{distance, 0, 0}), .Shape = shape});
    joint.BodyA = arm;
    joint.BodyB = pivot;
    world.AddJoint(joint);
    const float mass = 1 / world.Masses[arm].InvMass;
    // Parallel axis: about the pivot rather than about its own center.
    return {world, arm, 1 / world.Masses[arm].InvInertiaLocal[2] + mass * distance * distance, mass};
}

TEST_CASE_FIXTURE(OneWorld, "a joint holds its two anchor points together") {
    // A ball joint keeps the two anchors coincident wherever the bodies move to.
    constexpr float Distance = 1;
    const auto pendulum = MakePendulum(world, Distance);

    float worst = 0;
    for (uint32_t step = 0; step < 600; ++step) {
        solver.Step(world);
        const auto &pose = world.Poses[pendulum.Arm];
        const float3 held = WorldPoint(pose, world.Joints[0].AnchorA);
        worst = std::max(worst, simd::length(held)); // the other body is static at the origin
    }
    CHECK(worst < 5e-3f); // a fifth of a percent of the arm's length
}

TEST_CASE_FIXTURE(OnDevice, "a pendulum converges on the speed energy says it reaches") {
    // Released horizontal, the height the centre of mass loses becomes rotation: 1/2 I w^2 = m g d.
    // Backwards Euler dissipates, so this is a convergence check and only a shorter step closes the gap.
    constexpr float Distance = 1;

    const auto fastest = [&](float rate) {
        World world{context};
        const auto pendulum = MakePendulum(world, Distance);
        const StepSettings settings{.DeltaTime = 1 / rate};
        float peak = 0;
        for (uint32_t step = 0; step < uint32_t(rate); ++step) { // one second, past the bottom of the swing
            solver.Step(world, settings);
            peak = std::max(peak, simd::length(world.Velocities[pendulum.Arm].Angular));
        }
        return std::pair{peak, std::sqrt(2 * pendulum.Mass * Gravity * Distance / pendulum.Inertia)};
    };

    const auto [coarse, expected] = fastest(60);
    const auto [fine, same] = fastest(960);
    CHECK(same == doctest::Approx(expected)); // the closed form does not depend on the step
    CHECK(coarse < expected); // backwards Euler only ever loses energy
    CHECK(coarse > 0.9f * expected);
    CHECK(fine > coarse); // a shorter step loses less of it
    CHECK(fine == doctest::Approx(expected).epsilon(0.02));
}

TEST_CASE_FIXTURE(OneWorld, "a joint that holds rotation makes two boxes into one body") {
    // Fixed side by side and dropped, they land flat and stay square, the behaviour the angular rows produce.
    const auto shape = world.AddShape(UnitBox);
    AddGround(world);
    const auto left = world.AddBody({.Pose = At(float3{-Half, 2, 0}), .Shape = shape});
    const auto right = world.AddBody({.Pose = At(float3{Half, 2, 0}), .Shape = shape});
    world.AddJoint({.BodyA = left, .BodyB = right, .At = {0, 2, 0}, .Angular = {AxisLocked, AxisLocked, AxisLocked}});

    Run(solver, world, 400);
    CheckResting(world.Poses[left].Position.y);
    CheckResting(world.Poses[right].Position.y);
    CHECK(world.Poses[right].Position.x - world.Poses[left].Position.x == doctest::Approx(1).epsilon(0.01));
    const float4 a = world.Poses[left].Orientation, b = world.Poses[right].Orientation;
    CHECK(std::abs(simd::dot(a, b)) == doctest::Approx(1).epsilon(0.001));
    CHECK(simd::length(world.Velocities[left].Linear) < 1e-2f);
}

TEST_CASE_FIXTURE(OneWorld, "a hinge leaves one axis free and holds the other two") {
    // Free about z and locked about x and y, so a push straight out of the xy plane leaves it in the plane.
    const auto pendulum = MakePendulum(world, 1, {.Angular = {AxisLocked, AxisLocked, AxisFree}});
    world.Velocities[pendulum.Arm].Linear = {0, 0, 3};

    // The first steps deflect while the penalty ramps off its floor, so the measurement window starts after them.
    Run(solver, world, 60);
    float worst = 0;
    for (uint32_t step = 0; step < 240; ++step) {
        solver.Step(world);
        worst = std::max(worst, std::abs(world.Poses[pendulum.Arm].Position.z));
    }
    CHECK(worst < 1e-3f);
    CHECK(simd::length(world.Velocities[pendulum.Arm].Angular) > 1); // it swings rather than seizing
}

TEST_CASE_FIXTURE(OneWorld, "a motor turns its axis at the speed it is given") {
    // The row targets a turn of speed * dt every step, so with torque to spare gravity has no effect.
    constexpr float Speed = 2;
    const auto driven = JointDesc{.Angular = {AxisLocked, AxisLocked, AxisDriven}, .MotorSpeed = {0, 0, Speed}, .MotorMaxTorque = {0, 0, 1e6f}};

    const auto pendulum = MakePendulum(world, 1, driven);
    Run(solver, world, 300);
    CHECK(world.Velocities[pendulum.Arm].Angular.z == doctest::Approx(Speed).epsilon(0.01));
    // Its weight is 9810 N m about the pivot at the horizontal, and the motor turns it through that.
    CHECK(simd::length(world.Poses[pendulum.Arm].Position) == doctest::Approx(1).epsilon(0.01));
}

TEST_CASE_FIXTURE(OneWorld, "a positioned motor turns its axis to the angle it is given") {
    // The same row as the velocity motor with a target angle in place of a rate.
    // The row targets the whole remaining error, so the torque bound sets how fast the axis reaches it.
    constexpr float Target = 0.8f;
    const auto pendulum = MakePendulum(world, 1, {.Angular = {AxisLocked, AxisLocked, AxisPositioned}, .MotorTarget = {0, 0, Target}, .MotorMaxTorque = {0, 0, 1e6f}});
    const auto angle = [&] {
        const auto at = world.Poses[pendulum.Arm].Position;
        return std::atan2(float(at.y), float(at.x));
    };

    Run(solver, world, 300);
    // The arm started along x, so its bearing about the pivot is the angle itself.
    CHECK(angle() == doctest::Approx(Target).epsilon(0.01));
    CHECK(simd::length(world.Poses[pendulum.Arm].Position) == doctest::Approx(1).epsilon(0.01));
    CHECK(simd::length(world.Velocities[pendulum.Arm].Angular) < 1e-2f);

    SUBCASE("and brings it back after being moved off it") {
        // A position motor targets the angle every step rather than moving the body once.
        // The arm is repositioned rather than pushed, and woken explicitly since a pose write alone leaves it asleep.
        world.Poses[pendulum.Arm].Position = {0, -1, 0};
        world.Poses[pendulum.Arm].Orientation = QuatFromRotationVector(float3{0, 0, -1.5707963f});
        world.Wake(pendulum.Arm);
        REQUIRE(std::abs(angle() - Target) > 0.02f);
        Run(solver, world, 300);
        CHECK(angle() == doctest::Approx(Target).epsilon(0.01));
        CHECK(simd::length(world.Velocities[pendulum.Arm].Angular) < 1e-2f);
    }
}

TEST_CASE_FIXTURE(OneWorld, "a positioned spring held to a torque climbs at exactly that torque over that inertia") {
    // Torque limited, in free space, from rest, with a target angle far enough away that the row stays saturated, so the rate it gains is tau / I.
    // Eq. 14's secant makes that hold, since a raw penalty against a distant angle reports a stiffness far above the force it delivers.
    // The row is a spring rather than a hard lock, whose bound would size a stabilization correction rather than a torque.
    constexpr float Torque = 300, Target = 3, Stiffness = 1e6f;
    const StepSettings settings{.Gravity = {0, 0, 0}};
    const auto pendulum = MakePendulum(world, 1, {.Angular = {AxisLocked, AxisLocked, AxisPositioned}, .MotorTarget = {0, 0, Target}, .MotorMaxTorque = {0, 0, Torque}, .AngularStiffness = {INFINITY, INFINITY, Stiffness}});

    constexpr uint32_t Steps = 200;
    Run(solver, world, Steps, settings);
    const float elapsed = Steps * settings.DeltaTime;
    const float reached = world.Velocities[pendulum.Arm].Angular.z;
    // It travels tau/I t^2 / 2, short of the target, so the row stays at its bound throughout.
    REQUIRE(0.5f * Torque / pendulum.Inertia * elapsed * elapsed < Target);
    REQUIRE(Stiffness * Target > Torque); // the stiffness alone would demand more than the bound
    CHECK(reached == doctest::Approx(Torque / pendulum.Inertia * elapsed).epsilon(0.05));
}

TEST_CASE_FIXTURE(OneWorld, "a motor held to a torque spins up at exactly that torque over that inertia") {
    // Torque limited, in free space, from rest: a constant torque on a known inertia is tau / I.
    constexpr float Torque = 300, Speed = 2;
    const StepSettings settings{.Gravity = {0, 0, 0}};
    const auto pendulum = MakePendulum(world, 1, {.Angular = {AxisLocked, AxisLocked, AxisDriven}, .MotorSpeed = {0, 0, Speed}, .MotorMaxTorque = {0, 0, Torque}});

    constexpr uint32_t Steps = 300;
    Run(solver, world, Steps, settings);
    const float elapsed = Steps * settings.DeltaTime;
    const float reached = world.Velocities[pendulum.Arm].Angular.z;
    REQUIRE(reached < Speed); // still climbing, so the torque limit set the rate rather than the target
    CHECK(reached == doctest::Approx(Torque / pendulum.Inertia * elapsed).epsilon(0.05));
}

TEST_CASE_FIXTURE(OneWorld, "a limited axis turns freely between its stops and comes to rest on one") {
    // A stop is the same one-sided row a contact uses: outside the range it applies force in one direction only.
    constexpr float Low = -0.5f, High = 0.5f;
    const auto pendulum = MakePendulum(world, 1, {.Angular = {AxisLocked, AxisLocked, AxisLimited}, .LimitLow = {0, 0, Low}, .LimitHigh = {0, 0, High}});

    float lowest = 1e9f, highest = -1e9f;
    for (uint32_t step = 0; step < 600; ++step) {
        solver.Step(world);
        const auto &at = world.Poses[pendulum.Arm].Position;
        const float angle = std::atan2(at.y, at.x);
        lowest = std::min(lowest, angle);
        highest = std::max(highest, angle);
    }

    // The active stop is chosen at the step's start, so the axis travels one step past it.
    CHECK(lowest > Low - 0.05f);
    CHECK(lowest < Low); // it reaches the stop rather than stopping short
    CHECK(highest < High);
    CHECK(std::atan2(world.Poses[pendulum.Arm].Position.y, world.Poses[pendulum.Arm].Position.x) == doctest::Approx(Low).epsilon(0.002));
    CHECK(simd::length(world.Velocities[pendulum.Arm].Angular) < 1e-3f);
}

TEST_CASE_FIXTURE(OnDevice, "a spring on a limit sags until it balances the load") {
    // A limited axis with finite stiffness is soft: no dual, and a penalty ramping no further than that stiffness, Eq. 7 on the current angle.
    // It sags until k theta balances the arm's weight.
    const auto settled = [&](float stiffness) {
        World world{context};
        // Free to swing up, and softly stopped at the horizontal on the way down.
        const auto pendulum = MakePendulum(world, 1, {.Angular = {AxisLocked, AxisLocked, AxisLimited}, .LimitLow = {0, 0, 0}, .LimitHigh = {0, 0, 3.14159f}, .AngularStiffness = {INFINITY, INFINITY, stiffness}});
        Run(solver, world, 900);
        const auto at = world.Poses[pendulum.Arm].Position;
        const float angle = std::atan2(float(at.y), float(at.x));
        // k theta = m g cos(theta) on a one metre arm.
        // It hangs below the stop, so the angle is negative.
        const float balance = -pendulum.Mass * Gravity * std::cos(angle) / stiffness;
        CHECK(angle == doctest::Approx(balance).epsilon(0.1).scale(0));
        CHECK(angle < 0); // it sags, the difference from a hard stop
        CHECK(simd::length(world.Velocities[pendulum.Arm].Angular) < 1e-2f);
        return angle;
    };

    const float soft = settled(1e6f), stiff = settled(4e6f);
    CHECK(std::abs(soft) > 3 * std::abs(stiff)); // four times the stiffness, near enough a quarter the sag
}

TEST_CASE_FIXTURE(OneWorld, "a soft linear row hangs a body on a spring") {
    // The soft branch on the anchor rows, so a body hung on one rests mg/k below the anchor.
    constexpr float Stiffness = 2e5f;
    const auto anchor = world.AddBody({}); // no shape, so no mass: a fixed point to hang from
    const auto box = world.AddBody({.Shape = world.AddShape(UnitBox)});
    world.AddJoint({.BodyA = box, .BodyB = anchor, .At = {0, 0, 0}, .LinearStiffness = {Stiffness, Stiffness, Stiffness}});

    Run(solver, world, 900);
    const float weight = Gravity / world.Masses[box].InvMass;
    CHECK(world.Poses[box].Position.y == doctest::Approx(-weight / Stiffness).epsilon(0.05).scale(0));
    CHECK(std::abs(world.Poses[box].Position.x) < 1e-3f);
    CHECK(simd::length(world.Velocities[box].Linear) < 1e-2f);
}

TEST_CASE_FIXTURE(OnDevice, "a soft locked axis is a torsional spring, and holds both ways") {
    // The soft branch on a row holding an angle rather than a stop.
    // It sags to the same k theta balance on both sides, so reversing gravity mirrors the angle where a stop would let the arm swing away.
    const auto settled = [&](float stiffness, float sign) {
        World world{context};
        const auto pendulum = MakePendulum(world, 1, {.Angular = {AxisLocked, AxisLocked, AxisLocked}, .AngularStiffness = {INFINITY, INFINITY, stiffness}});
        StepSettings settings{};
        settings.Gravity.y *= sign;
        Run(solver, world, 900, settings);
        const auto at = world.Poses[pendulum.Arm].Position;
        const float angle = std::atan2(float(at.y), float(at.x));
        // k theta = m g cos(theta) on a one metre arm, loaded on whichever side gravity acts.
        // scale(0) because these are hundredths of a radian and Approx's default scale of one would absorb the difference.
        const float balance = -sign * pendulum.Mass * std::abs(settings.Gravity.y) * std::cos(angle) / stiffness;
        CHECK(angle == doctest::Approx(balance).epsilon(0.05).scale(0));
        CHECK(simd::length(world.Velocities[pendulum.Arm].Angular) < 1e-2f);
        // The hard rows either side still hold, so the arm turns about z alone.
        const float3 turn = RotationVector(world.Poses[pendulum.Arm].Orientation);
        CHECK(std::abs(turn.x) < 1e-3f);
        CHECK(std::abs(turn.y) < 1e-3f);
        return angle;
    };

    const float down = settled(1e6f, 1), up = settled(1e6f, -1);
    CHECK(down < 0); // it sags, the difference from a hard lock
    CHECK(up == doctest::Approx(-down).epsilon(0.02));
    CHECK(std::abs(down) > 3 * std::abs(settled(4e6f, 1))); // four times the stiffness, near enough a quarter the sag
}

TEST_CASE_FIXTURE(OnDevice, "a soft positioned axis is a spring towards where it is told to be") {
    // The soft branch on the mode measuring error from the target rather than from rest, which moves the spring's zero.
    // The torque bound is clear of the load, so the secant rescaling stays inactive here.
    constexpr float Stiffness = 1e6f;
    const auto settled = [&](float target) {
        World world{context};
        const auto pendulum = MakePendulum(world, 1, {.Angular = {AxisLocked, AxisLocked, AxisPositioned}, .MotorTarget = {0, 0, target}, .MotorMaxTorque = {0, 0, 1e6f}, .AngularStiffness = {INFINITY, INFINITY, Stiffness}});
        Run(solver, world, 900);
        const auto at = world.Poses[pendulum.Arm].Position;
        const float angle = std::atan2(float(at.y), float(at.x));
        // k (theta - target) = m g cos(theta), the spring above with its zero moved to the target.
        const float balance = target - pendulum.Mass * Gravity * std::cos(angle) / Stiffness;
        CHECK(angle == doctest::Approx(balance).epsilon(0.02).scale(0));
        CHECK(simd::length(world.Velocities[pendulum.Arm].Angular) < 1e-2f);
        return angle;
    };

    // A target of zero leaves it hanging one sag below, and a lift target raises it by the whole target angle.
    const float held = settled(0), lifted = settled(0.6f);
    CHECK(held < 0);
    CHECK(lifted - held == doctest::Approx(0.6f).epsilon(0.02));
}

namespace {
// A hinge whose free axis is the joint frame's z, an axis of neither body.
// All the height the arm loses becomes rotation, so the closed form is the flat pendulum's with the in-plane part of gravity.
// Any release angle works, since the locked rows read a swing-twist swing rather than a rotation vector.
struct FramedPendulum {
    Index Arm;
    float3 Axis; // the world axis it is free about, the swing plane's normal
    float Peak; // the speed at the bottom of the swing, from energy alone
};

FramedPendulum HangOnFrame(World &world, float4 frame, float4 pivot_turn, float3 gravity, float release) {
    constexpr float Distance = 1;
    const float3 axis = Rotate(frame, float3{0, 0, 1});
    const float3 down = simd::normalize(gravity - simd::dot(gravity, axis) * axis);
    const float3 along = simd::normalize(simd::cross(axis, gravity)); // level, and square to the swing
    const float3 start = Distance * (std::cos(release) * down + std::sin(release) * along);
    const auto shape = world.AddShape(UnitBox);
    const auto pivot = world.AddBody({.Pose = At(float3{0, 0, 0}, pivot_turn)}); // no shape, so no mass
    const auto arm = world.AddBody({.Pose = At(start), .Shape = shape});
    world.AddJoint({.BodyA = arm, .BodyB = pivot, .At = {0, 0, 0}, .Frame = frame, .Angular = {AxisLocked, AxisLocked, AxisFree}});
    const float mass = 1 / world.Masses[arm].InvMass;
    const float inertia = 1 / world.Masses[arm].InvInertiaLocal[2] + mass * Distance * Distance;
    return {arm, axis, std::sqrt(2 * mass * simd::dot(gravity, Distance * down - start) / inertia)};
}

// A box turning about a fixed point at its own centre, so the closed forms use the box's own inertia with no arm term.
// The joint index comes back as well, since a hinge's turn is read from the joint rather than from a pose.
struct Wheel {
    Index Body, Joint;
    float Inertia;
};

Wheel SpinOnAxle(World &world, JointDesc joint, float3 spin) {
    const auto shape = world.AddShape(UnitBox);
    const auto axle = world.AddBody({}); // no shape, so no mass: a fixed point to turn about
    const auto wheel = world.AddBody({.Pose = At(float3{0, 0, 0}), .Velocity = {.Angular = spin}, .Shape = shape});
    joint.BodyA = wheel;
    joint.BodyB = axle;
    return {wheel, world.AddJoint(joint), 1 / world.Masses[wheel].InvInertiaLocal[2]};
}
} // namespace

TEST_CASE_FIXTURE(OnDevice, "a hinge swings about the joint frame's axis, which is neither body's") {
    // The axis is the frame's z alone, so a joint using body B's axis instead would swing elsewhere.
    constexpr float Turn = 0.5235988f, Release = 1.0471976f; // thirty degrees, and sixty of release
    const float4 frame = QuatMul(QuatFromRotationVector(float3{0, Turn, 0}), QuatFromRotationVector(float3{Turn, 0, 0}));
    const StepSettings settings{};

    const auto swing = [&](float4 pivot_turn) {
        World world{context};
        const auto hinge = HangOnFrame(world, frame, pivot_turn, settings.Gravity, Release);
        float peak = 0, out_of_plane = 0;
        for (uint32_t step = 0; step < 600; ++step) {
            solver.Step(world, settings);
            peak = std::max(peak, simd::length(world.Velocities[hinge.Arm].Angular));
            // The swing plane's normal is the hinge axis, measured only after the penalty has ramped.
            if (step >= 60)
                out_of_plane = std::max(out_of_plane, std::abs(simd::dot(world.Poses[hinge.Arm].Position, hinge.Axis)));
        }
        return std::tuple{hinge, peak, out_of_plane, world.Poses[hinge.Arm].Position};
    };

    const auto [hinge, peak, out_of_plane, at] = swing(float4{0, 0, 0, 1});
    // The same convergence as the flat pendulum, since backwards Euler only loses energy.
    CHECK(peak < hinge.Peak);
    CHECK(peak > 0.9f * hinge.Peak);
    CHECK(out_of_plane < 1e-3f);
    CHECK(std::abs(simd::length(at) - 1) < 5e-3f); // still on its arm

    SUBCASE("and the pivot's own orientation has nothing to do with it") {
        // Every number above is the joint frame's, so turning the pivot moves none of them.
        const auto [turned_hinge, turned_peak, turned_out, turned_at] = swing(QuatFromRotationVector(float3{0.7f, -1.3f, 0.4f}));
        CHECK(turned_peak == doctest::Approx(peak).epsilon(0.01));
        CHECK(turned_out < 1e-3f);
        CHECK(simd::distance(turned_at, at) < 5e-3f);
    }
}

TEST_CASE_FIXTURE(OnDevice, "a hinge released past the half turn swings through the bottom and holds") {
    // Half a turn from the pose a joint was made in is where a rotation vector runs out.
    // The shortest arc flips every sign at once and the log map's perpendicular gain falls to zero.
    constexpr float Turn = 0.5235988f; // thirty degrees about x and again about y, so no body's axis
    const float4 frame = QuatMul(QuatFromRotationVector(float3{0, Turn, 0}), QuatFromRotationVector(float3{Turn, 0, 0}));
    const StepSettings settings{};
    float release = 0;
    SUBCASE("released 150 degrees off the bottom") { release = 2.6179939f; }
    SUBCASE("and at 175, which is as near upside down as it can be let go from") { release = 3.0543261f; }

    World world{context};
    const auto hinge = HangOnFrame(world, frame, QuatFromRotationVector(float3{0.7f, -1.3f, 0.4f}), settings.Gravity, release);
    float peak = 0, out_of_plane = 0, off_the_arm = 0;
    for (uint32_t step = 0; step < 600; ++step) {
        solver.Step(world, settings);
        peak = std::max(peak, simd::length(world.Velocities[hinge.Arm].Angular));
        if (step < 60) continue;
        const float3 at = world.Poses[hinge.Arm].Position;
        out_of_plane = std::max(out_of_plane, std::abs(simd::dot(at, hinge.Axis)));
        off_the_arm = std::max(off_the_arm, std::abs(simd::length(at) - 1));
    }
    // The speed from the height it lost, less what backwards Euler takes over a long arc.
    CHECK(peak < hinge.Peak);
    CHECK(peak > 0.88f * hinge.Peak);
    CHECK(out_of_plane < 1e-3f);
    CHECK(off_the_arm < 5e-3f);
}

TEST_CASE_FIXTURE(OnDevice, "a driven hinge turns twenty revolutions on a tilted axle and reports every one") {
    // Twenty turns is twenty times past the half turn a quaternion can represent.
    // The joint accumulates the twist a step at a time, so the value climbs rather than folding back.
    constexpr float Turn = 0.5235988f, Turns = 20;
    const float4 frame = QuatMul(QuatFromRotationVector(float3{0, Turn, 0}), QuatFromRotationVector(float3{Turn, 0, 0}));
    constexpr uint32_t Steps = 600;
    const StepSettings settings{.Gravity = {0, 0, 0}}; // a box jointed at its own centre has no weight about the axle
    const float wanted = Turns * 2 * std::numbers::pi_v<float>;
    const float speed = wanted / (float(Steps) * settings.DeltaTime);

    World world{context};
    const auto wheel = SpinOnAxle(world, {.Frame = frame,
                                          .Angular = {AxisLocked, AxisLocked, AxisDriven},
                                          .MotorSpeed = {0, 0, speed},
                                          .MotorMaxTorque = {0, 0, 1e6f}},
                                  float3{0, 0, 0});
    REQUIRE(wheel.Joint != NoIndex);
    const float3 axle = Rotate(frame, float3{0, 0, 1});
    float off_the_axle = 0;
    for (uint32_t step = 0; step < Steps; ++step) {
        solver.Step(world, settings);
        const float3 rate = world.Velocities[wheel.Body].Angular;
        off_the_axle = std::max(off_the_axle, simd::length(rate - simd::dot(rate, axle) * axle));
    }
    // Short by the one step the drive spends reaching the speed.
    const float turned = world.Joints[wheel.Joint].Twist;
    CHECK(turned == doctest::Approx(wanted).epsilon(0.01));
    CHECK(turned < wanted); // a row targeting speed * h per step never overshoots
    CHECK(simd::dot(world.Velocities[wheel.Body].Angular, axle) == doctest::Approx(speed).epsilon(0.01));
    CHECK(off_the_axle < 1e-3f);
}

TEST_CASE_FIXTURE(OnDevice, "a twist limit past the half turn stops the axis where it says") {
    // A stop at 200 degrees is past where a rotation vector folds back: modulo a half turn it reads -160, outside the low stop and pushing the other way.
    // The accumulated twist stays at 200.
    constexpr float Turn = 0.5235988f, Stop = 3.4906585f, Degrees = 0.017453293f;
    const float4 frame = QuatMul(QuatFromRotationVector(float3{0, Turn, 0}), QuatFromRotationVector(float3{Turn, 0, 0}));
    const StepSettings settings{.Gravity = {0, 0, 0}};

    World world{context};
    const auto wheel = SpinOnAxle(world, {.Frame = frame,
                                          .Angular = {AxisLocked, AxisLocked, AxisLimited},
                                          .LimitLow = {0, 0, -0.5f},
                                          .LimitHigh = {0, 0, Stop}},
                                  Rotate(frame, float3{0, 0, 4}));
    REQUIRE(wheel.Joint != NoIndex);
    for (uint32_t step = 0; step < 600; ++step) solver.Step(world, settings);

    const float turned = world.Joints[wheel.Joint].Twist;
    CHECK(turned == doctest::Approx(Stop).epsilon(0.002));
    CHECK(turned > std::numbers::pi_v<float>); // past the half turn, the case under test
    CHECK(turned / Degrees > 190); // nowhere near the 160 a wrapped angle would read
    CHECK(simd::length(world.Velocities[wheel.Body].Angular) < 1e-3f); // resting against the stop rather than chattering
}

TEST_CASE_FIXTURE(OnDevice, "a joint made with its two ends apart closes the gap without a kick") {
    // The error exists before any load, so no penalty ramps against it and the stabilization pass alone resolves it.
    // That is why a hard row's penalty is floored at the pair's own inertial stiffness.
    // The correction cannot arrive as velocity, since the pass runs after velocity is read.
    constexpr float Offset = 0.01f;
    const StepSettings settings{};

    const auto hung = [&](float offset) {
        World world{context};
        const auto shape = world.AddShape(UnitBox);
        const auto anchor = world.AddBody({}); // no shape, so no mass: a fixed point to hang from
        const auto left = world.AddBody({.Pose = At(float3{-1, 0, 0}), .Shape = shape});
        const auto right = world.AddBody({.Pose = At(float3{1, 0, 0}), .Shape = shape});
        world.AddJoint({.BodyA = left, .BodyB = anchor, .At = {-1, 1, 0}});
        REQUIRE(world.AddJoint({.BodyA = left, .BodyB = right, .At = {0, 0, 0}, .AtB = float3{offset, 0, 0}}) != NoIndex);

        float worst_closing = 0, gap = 0;
        uint32_t closed = ~0u;
        for (uint32_t step = 0; step < 600; ++step) {
            const float was = gap;
            solver.Step(world, settings);
            const Pose &a = world.Poses[left], &b = world.Poses[right];
            const Joint &joint = world.Joints[1];
            gap = simd::distance(WorldPoint(a, joint.AnchorA), WorldPoint(b, joint.AnchorB));
            // The closing rate of the two ends, where a kick would show.
            if (step > 0) worst_closing = std::max(worst_closing, std::abs(gap - was) / settings.DeltaTime);
            if (closed == ~0u && gap < 1e-4f) closed = step;
        }
        return std::tuple{gap, worst_closing, closed};
    };

    const auto [together_gap, together_closing, together_closed] = hung(0);
    const auto [gap, closing, closed] = hung(Offset);
    CHECK(closed < 60);
    CHECK(gap < 1e-4f); // still closed at the end of the run
    CHECK(closing < Offset / settings.DeltaTime);
    CHECK(closing > together_closing);
    // The control closes as well, since the pair sags while the penalty ramps whether or not there is a gap.
    CHECK(together_gap < 1e-4f);
    CHECK(together_closed < 60);
}

TEST_CASE_FIXTURE(OneWorld, "an axis slammed between both its stops stays between them") {
    // A row carries its dual from step to step and the two stops are different constraints.
    // An axis crossing between them arrives with a dual of the wrong sign.
    // No state records which stop it was on, and the one-sided clamp zeroes the wrong-signed force on the first iteration.
    constexpr float Low = -0.2f, High = 0.2f, Slam = 6;
    const StepSettings settings{.Gravity = {0, 0, 0}}; // no gravity, so the stops do all the work
    const auto pendulum = MakePendulum(world, 1, {.Angular = {AxisLocked, AxisLocked, AxisLimited}, .LimitLow = {0, 0, Low}, .LimitHigh = {0, 0, High}});

    float worst = 0;
    for (uint32_t step = 0; step < 900; ++step) {
        if (step % 120 == 0) world.Velocities[pendulum.Arm].Angular = {0, 0, (step / 120) % 2 == 0 ? Slam : -Slam};
        solver.Step(world, settings);
        const auto &at = world.Poses[pendulum.Arm].Position;
        const float angle = std::atan2(at.y, at.x);
        worst = std::max(worst, std::max(angle - High, Low - angle));
    }
    // Six radians a second is a tenth of a radian past a stop in the step before one engages.
    CHECK(worst < Slam * settings.DeltaTime);
}

namespace {
// A box jointed to a fixed point, both ends at the box's own centre.
// The joint frame's axes are the world's, so a linear row's value is the box's displacement along that axis.
struct Slider {
    Index Box;
    float Mass;
};

Slider MakeSlider(World &world, JointDesc joint, float3 at = {0, 0, 0}) {
    const auto shape = world.AddShape(UnitBox);
    const auto anchor = world.AddBody({}); // no shape, so no mass and no contacts: a fixed point
    const auto box = world.AddBody({.Pose = At(at), .Shape = shape});
    joint.BodyA = box;
    joint.BodyB = anchor;
    world.AddJoint(joint);
    return {box, 1 / world.Masses[box].InvMass};
}
} // namespace

TEST_CASE_FIXTURE(OneWorld, "a slider dropped down its axis comes to rest on its stop") {
    // The linear form of the angular limit: outside the stops, a row that pushes only back inside.
    constexpr float Low = -0.5f, High = 0.5f;
    const auto slider = MakeSlider(world, {.Angular = {AxisLocked, AxisLocked, AxisLocked},
                                           .Linear = {AxisLocked, AxisLimited, AxisLocked},
                                           .LinearLimitLow = {0, Low, 0},
                                           .LinearLimitHigh = {0, High, 0}});

    float lowest = 1e9f;
    for (uint32_t step = 0; step < 600; ++step) {
        solver.Step(world);
        lowest = std::min(lowest, float(world.Poses[slider.Box].Position.y));
    }
    const float3 at = world.Poses[slider.Box].Position;
    CHECK(lowest > Low - 0.02f);
    CHECK(lowest < Low); // it reaches the stop rather than stopping short
    CHECK(at.y == doctest::Approx(Low).epsilon(0.002));
    CHECK(std::abs(at.x) < 1e-3f);
    CHECK(std::abs(at.z) < 1e-3f);
    CHECK(simd::length(world.Velocities[slider.Box].Linear) < 1e-2f);
}

TEST_CASE_FIXTURE(OneWorld, "a body free in a box settles in the corner gravity points at") {
    // JointTypes' free-in-a-box.
    // Gravity pulls along every axis, so resting in the corner means every axis reached a stop.
    constexpr float Reach = 0.25f;
    const StepSettings settings{.Gravity = {-5, -8, -3}};
    const auto slider = MakeSlider(world, {.Angular = {AxisLocked, AxisLocked, AxisLocked},
                                           .Linear = {AxisLimited, AxisLimited, AxisLimited},
                                           .LinearLimitLow = {-Reach, -Reach, -Reach},
                                           .LinearLimitHigh = {Reach, Reach, Reach}});

    Run(solver, world, 600, settings);
    const float3 at = world.Poses[slider.Box].Position;
    for (uint32_t axis = 0; axis < 3; ++axis) {
        CAPTURE(axis);
        CHECK(at[axis] == doctest::Approx(-Reach).epsilon(0.01)); // every component of gravity is negative
    }
    CHECK(simd::length(world.Velocities[slider.Box].Linear) < 1e-2f);
}

TEST_CASE_FIXTURE(OnDevice, "a linear drive reaches its speed, and climbs to it at exactly force over mass") {
    // The angular motor's row with a length in place of an angle: held to a force from rest, the rate is F/m.
    constexpr float Speed = 2;
    const StepSettings settings{.Gravity = {0, 0, 0}};

    SUBCASE("with force to spare it reaches the speed") {
        World world{context};
        const auto slider = MakeSlider(world, {.Linear = {AxisDriven, AxisLocked, AxisLocked},
                                               .LinearMotorSpeed = {Speed, 0, 0},
                                               .LinearMotorMaxForce = {1e7f, 0, 0}});
        Run(solver, world, 300, settings);
        CHECK(world.Velocities[slider.Box].Linear.x == doctest::Approx(Speed).epsilon(0.01));
        CHECK(std::abs(world.Poses[slider.Box].Position.y) < 1e-3f);
        CHECK(std::abs(world.Poses[slider.Box].Position.z) < 1e-3f);
    }

    SUBCASE("held to a force it climbs at exactly that force over that mass") {
        constexpr float Force = 3000, Fast = 100; // a speed it cannot reach, so the bound sets the rate
        World world{context};
        const auto slider = MakeSlider(world, {.Linear = {AxisDriven, AxisLocked, AxisLocked},
                                               .LinearMotorSpeed = {Fast, 0, 0},
                                               .LinearMotorMaxForce = {Force, 0, 0}});
        constexpr uint32_t Steps = 200;
        Run(solver, world, Steps, settings);
        const float elapsed = Steps * settings.DeltaTime;
        const float reached = world.Velocities[slider.Box].Linear.x;
        REQUIRE(reached < Fast); // still climbing, so the force bound set the rate rather than the target
        CHECK(reached == doctest::Approx(Force / slider.Mass * elapsed).epsilon(0.05));
    }
}

TEST_CASE_FIXTURE(OneWorld, "a linear position drive arrives at the offset it is given and holds it") {
    // The same row again with a position in place of a rate.
    // It holds its offset against gravity and arrives without ever gaining velocity, since the stabilization pass moves the error after velocity is read.
    // The bound sizes that correction, so a far target closes at Force h^2 / m per step.
    constexpr float Target = 0.8f, Force = 3e4f; // three times the box's own weight
    const auto slider = MakeSlider(world, {.Angular = {AxisLocked, AxisLocked, AxisLocked},
                                           .Linear = {AxisLocked, AxisPositioned, AxisLocked},
                                           .LinearMotorTarget = {0, Target, 0},
                                           .LinearMotorMaxForce = {0, Force, 0}});

    float highest = -1e9f;
    const auto approach = [&](uint32_t steps) {
        for (uint32_t step = 0; step < steps; ++step) {
            solver.Step(world);
            highest = std::max(highest, float(world.Poses[slider.Box].Position.y));
        }
    };

    approach(400);
    CHECK(world.Poses[slider.Box].Position.y == doctest::Approx(Target).epsilon(0.01));
    CHECK(simd::length(world.Velocities[slider.Box].Linear) < 1e-2f);
    CHECK(highest < Target + 1e-3f); // it arrives rather than overshooting

    SUBCASE("and brings it back after being moved off it") {
        // A drive targets the offset every step rather than moving the body once.
        world.Poses[slider.Box].Position = {0, -1.5f, 0};
        world.Wake(slider.Box);
        highest = -1e9f;
        approach(1100);
        CHECK(world.Poses[slider.Box].Position.y == doctest::Approx(Target).epsilon(0.01));
        CHECK(highest < Target + 1e-3f); // still no overshoot over two metres of travel
    }
}

TEST_CASE_FIXTURE(OnDevice, "a hard drive to a far target arrives without reporting a speed it never had") {
    // A hard positioned row is a lock at an offset, so the stabilization pass removes the error a step begins with.
    // That happens after velocity is read, as for a lock at zero or a stop.
    // A row taking that error whole would cross the gap inside the main iterations, and the differenced positions would report it as velocity.
    // Sec. 3.6: those iterations cover (1 - alpha) of the error and post-stabilization runs at alpha = 1, so the bar below is the whole error over one step.
    const StepSettings settings{.Gravity = {0, 0, 0}}; // nothing but the drive acting

    SUBCASE("an angular drive a quarter turn away") {
        constexpr float Target = 1.5707963f;
        World world{context};
        const auto pendulum = MakePendulum(world, 1, {.Angular = {AxisLocked, AxisLocked, AxisPositioned}, .MotorTarget = {0, 0, Target}, .MotorMaxTorque = {0, 0, INFINITY}});
        float peak = 0;
        for (uint32_t step = 0; step < 300; ++step) {
            solver.Step(world, settings);
            peak = std::max(peak, simd::length(world.Velocities[pendulum.Arm].Angular));
        }
        const auto at = world.Poses[pendulum.Arm].Position;
        CHECK(std::atan2(float(at.y), float(at.x)) == doctest::Approx(Target).epsilon(0.01));
        CHECK(peak < 0.01f * Target / settings.DeltaTime); // taking the error whole reports 94.2 rad/s
    }

    SUBCASE("and a linear one most of a metre away") {
        constexpr float Target = 0.9f;
        World world{context};
        const auto slider = MakeSlider(world, {.Angular = {AxisLocked, AxisLocked, AxisLocked},
                                               .Linear = {AxisLocked, AxisPositioned, AxisLocked},
                                               .LinearMotorTarget = {0, Target, 0},
                                               .LinearMotorMaxForce = {0, INFINITY, 0}});
        float peak = 0;
        for (uint32_t step = 0; step < 300; ++step) {
            solver.Step(world, settings);
            peak = std::max(peak, simd::length(world.Velocities[slider.Box].Linear));
        }
        CHECK(world.Poses[slider.Box].Position.y == doctest::Approx(Target).epsilon(0.01));
        CHECK(peak < 0.01f * Target / settings.DeltaTime); // taking the error whole reports 54 m/s here
    }
}

TEST_CASE_FIXTURE(OnDevice, "an axis locked at an offset holds the two ends that far apart") {
    // KHR's min == max at something other than zero, as in Joint_08's x fixed at 1.0.
    // Made a metre out it stays a metre out, where a row positioned at zero pulls it in.
    constexpr float Offset = 1;
    const auto held = [&](float target) {
        World world{context};
        const auto slider = MakeSlider(world,
                                       {.At = {Offset, 0, 0}, // the box's end, at the box's centre
                                        .AtB = float3{0, 0, 0}, // the fixed point's end, a metre away
                                        .Angular = {AxisLocked, AxisLocked, AxisLocked},
                                        .Linear = {AxisPositioned, AxisLocked, AxisLocked},
                                        .LinearMotorTarget = {target, 0, 0},
                                        .LinearMotorMaxForce = {INFINITY, 0, 0}},
                                       float3{Offset, 0, 0});
        Run(solver, world, 600);
        return world.Poses[slider.Box].Position.x;
    };

    CHECK(held(Offset) == doctest::Approx(Offset).epsilon(0.002)); // still a metre out, having not moved
    CHECK(held(0) == doctest::Approx(0).epsilon(0.002).scale(1)); // pulled in when the target is zero
}

TEST_CASE_FIXTURE(OneWorld, "a braked wheel decays at the rate its damping and inertia name") {
    // A damping coefficient and no stiffness, which is WaterWheel's drive: I dw/dt = -c w, so the wheel decays as w0 exp(-c t / I).
    // Backwards Euler integrates the geometric w0 (1 + c h / I)^-n, checked tightly since any slack there belongs to the solve, and the exponential loosely.
    constexpr float Spin = 10, Damping = 166.667f; // one second of time constant on a unit box
    const StepSettings settings{.Gravity = {0, 0, 0}}; // nothing but the brake acting
    const auto wheel = SpinOnAxle(world, {.Angular = {AxisLocked, AxisLocked, AxisDriven},
                                          .MotorMaxTorque = {0, 0, INFINITY},
                                          .AngularStiffness = {INFINITY, INFINITY, 0},
                                          .AngularDamping = {0, 0, Damping}},
                                  float3{0, 0, Spin});
    const float per_step = 1 + Damping * settings.DeltaTime / wheel.Inertia;

    for (uint32_t step = 0; step < 120; ++step) {
        solver.Step(world, settings);
        const uint32_t taken = step + 1;
        const float measured = world.Velocities[wheel.Body].Angular.z;
        const float geometric = Spin / std::pow(per_step, float(taken));
        CAPTURE(step);
        CHECK(measured == doctest::Approx(geometric).epsilon(0.001));
        CHECK(measured > 0); // a brake slows it rather than reversing it
    }

    const float elapsed = 120 * settings.DeltaTime; // two time constants
    const float measured = world.Velocities[wheel.Body].Angular.z;
    const float exponential = Spin * std::exp(-Damping * elapsed / wheel.Inertia);
    CHECK(measured == doctest::Approx(exponential).epsilon(0.02)); // the integrator's own gap
    CHECK(measured > exponential); // a geometric decay of the same rate stays above the exponential
}

TEST_CASE_FIXTURE(OneWorld, "a drive with damping and no stiffness approaches its speed instead of snapping to it") {
    // The other half of the row: I dw/dt = -c (w - wT), so it arrives as wT (1 - exp(-c t / I)).
    constexpr float Target = 5, Damping = 166.667f;
    const StepSettings settings{.Gravity = {0, 0, 0}};
    const auto wheel = SpinOnAxle(world, {.Angular = {AxisLocked, AxisLocked, AxisDriven},
                                          .MotorSpeed = {0, 0, Target},
                                          .MotorMaxTorque = {0, 0, INFINITY},
                                          .AngularStiffness = {INFINITY, INFINITY, 0},
                                          .AngularDamping = {0, 0, Damping}},
                                  float3{0, 0, 0});
    const float per_step = 1 + Damping * settings.DeltaTime / wheel.Inertia;
    const auto approached = [&](uint32_t taken) { return Target * (1 - 1 / std::pow(per_step, float(taken))); };

    // A single step covers only a fraction, where a rigid motor with torque to spare arrives at once.
    solver.Step(world, settings);
    CHECK(world.Velocities[wheel.Body].Angular.z == doctest::Approx(approached(1)).epsilon(0.01));
    CHECK(world.Velocities[wheel.Body].Angular.z < 0.05f * Target);

    Run(solver, world, 59, settings); // one time constant, which is 63% of the way there
    CHECK(world.Velocities[wheel.Body].Angular.z == doctest::Approx(approached(60)).epsilon(0.005));
    CHECK(world.Velocities[wheel.Body].Angular.z < 0.7f * Target);

    Run(solver, world, 240, settings); // five time constants, within a percent of the target
    CHECK(world.Velocities[wheel.Body].Angular.z == doctest::Approx(Target).epsilon(0.01));
    CHECK(world.Velocities[wheel.Body].Angular.z < Target);
}

TEST_CASE_FIXTURE(OnDevice, "a spring with a damper across it settles without overshooting") {
    // Stiffness and damping on one linear row is a mass on a spring with a dashpot.
    // It overshoots below a damping ratio of one, with critical at c = 2 sqrt(k m).
    // Both come to rest at mg/k.
    constexpr float Stiffness = 2e5f;
    const StepSettings settings{};

    const auto hung = [&](float damping) {
        World world{context};
        const auto shape = world.AddShape(UnitBox);
        const auto anchor = world.AddBody({}); // no shape, so no mass: a fixed point to hang from
        const auto box = world.AddBody({.Shape = shape});
        world.AddJoint({.BodyA = box, .BodyB = anchor, .At = {0, 0, 0},
                        .LinearStiffness = {INFINITY, Stiffness, INFINITY},
                        .LinearDamping = {0, damping, 0}});
        const float mass = 1 / world.Masses[box].InvMass;
        float lowest = 0;
        for (uint32_t step = 0; step < 300; ++step) {
            solver.Step(world, settings);
            lowest = std::min(lowest, float(world.Poses[box].Position.y));
        }
        const float rest = -std::abs(settings.Gravity.y) * mass / Stiffness;
        return std::tuple{float(world.Poses[box].Position.y), lowest, rest, 2 * std::sqrt(Stiffness * mass)};
    };

    const auto [loose_at, loose_lowest, rest, critical] = hung(0);
    CHECK(rest - loose_lowest > 0.5f * std::abs(rest)); // undamped it swings half again past its rest

    const auto [at, lowest, same_rest, same_critical] = hung(critical);
    CHECK(same_rest == rest);
    CHECK(at == doctest::Approx(rest).epsilon(0.01));
    CHECK(rest - lowest < 0.01f * std::abs(rest)); // no overshoot, the definition of critical damping
    CHECK(lowest < 0.5f * rest); // it travels the full distance rather than being held short
}

TEST_CASE_FIXTURE(OneWorld, "a settled stack keeps the same contacts from step to step") {
    // A renamed feature discards a dual, so the key set has to be identical rather than merely the same size.
    constexpr uint32_t Count = 4;
    const auto shape = world.AddShape(UnitBox);
    AddGround(world);
    AddStack(world, shape, Count);

    Run(solver, world, 240); // reach rest first, since contacts legitimately come and go while landing
    const auto settled = ContactKeys(world);
    CHECK(settled.size() == 4 * Count); // four corners against the plane, four against each box below
    CheckKeysHold(solver, world, settled);
}

TEST_CASE_FIXTURE(OneWorld, "a raft with a layer on top settles, and takes the colours it needs to do it") {
    // Every other stack here is a chain, which two colours cover.
    // Each box on the upper layer touches four below, so the colouring needs more, and it grows by one colour per pass.
    const auto shape = world.AddShape(UnitBox);
    AddGround(world);
    for (uint32_t x = 0; x < 4; ++x)
        for (uint32_t z = 0; z < 4; ++z)
            world.AddBody({.Pose = At(float3{float(x) - 1.5f, Half, float(z) - 1.5f}), .Shape = shape});
    for (uint32_t x = 0; x < 3; ++x)
        for (uint32_t z = 0; z < 3; ++z)
            world.AddBody({.Pose = At(float3{float(x) - 1.f, 3 * Half + 0.05f, float(z) - 1.f}), .Shape = shape});

    Run(solver, world, 300);
    uint32_t most = 0;
    float fastest = 0;
    for (uint32_t body = 1; body < world.BodyCount(); ++body) {
        most = std::max(most, ColorOf(world.Colors[body]));
        fastest = std::max(fastest, simd::length(world.Velocities[body].Linear));
    }
    CHECK(most + 1 > 2); // more than the two a chain needs
    CHECK(most < StepSettings{}.MaxColors); // and inside the cap
    CHECK(fastest < 1e-2f);
    CheckResting(world.Poses[1].Position.y); // the lower layer is on the plane
    CHECK(world.Poses[world.BodyCount() - 1].Position.y == doctest::Approx(3 * Half).epsilon(0.005));
}

TEST_CASE_FIXTURE(OneWorld, "a body that has got nowhere for half a second stops being solved") {
    // A sleeping body keeps its contacts, and its velocity is exactly zero.
    const StepSettings settings{};
    const auto asleep = [&](const World &world, Index body) { return world.Quiet[body] >= settings.SleepSteps; };

    const auto shape = world.AddShape(UnitBox);
    AddGround(world);
    const auto box = world.AddBody({.Pose = At(float3{0, Half + 1, 0}), .Shape = shape});

    Run(solver, world, 200);
    REQUIRE(asleep(world, box));
    CHECK(world.Velocities[box].Linear.y == 0);
    CheckResting(world.Poses[box].Position.y);
    const float settled = world.Poses[box].Position.y;
    Run(solver, world, 200);
    CHECK(world.Poses[box].Position.y == settled);

    SUBCASE("and wakes when the host gives it a shove") {
        // A sleeping body's velocity is zero, so anything written there came from outside.
        world.Velocities[box].Linear = {2, 0, 0};
        Run(solver, world, 2);
        CHECK(!asleep(world, box));
        Run(solver, world, 300);
        CHECK(world.Poses[box].Position.x > 0.1f);
        CHECK(asleep(world, box));
    }

    SUBCASE("and wakes when something lands on it") {
        const auto dropped = world.AddBody({.Pose = At(float3{0, Half + 3, 0}), .Shape = shape});
        Run(solver, world, 60);
        CHECK(!asleep(world, box)); // the sleeping box wakes on the impact
        Run(solver, world, 400);
        CHECK(asleep(world, box));
        CHECK(asleep(world, dropped));
        CHECK(std::abs(world.Poses[dropped].Position.y - (settled + 1 - settings.ContactMargin)) < 3e-3f);
    }
}

TEST_CASE_FIXTURE(OneWorld, "nothing sleeps while what it is touching is still moving") {
    // Sleeping is a property of a group rather than a body: a box that sleeps mid-descent leaves the one under it pressing on a body that is no longer solved.
    // No body is counted quieter than its neighbours.
    const StepSettings settings{};
    const auto shape = world.AddShape(UnitBox);
    AddGround(world);
    const std::vector<Index> stack = AddStack(world, shape, 3);

    // Drop a fourth from a height, so it is still falling long after the three below have gone quiet.
    const auto falling = world.AddBody({.Pose = At(float3{0, Half + 12, 0}), .Shape = shape});
    for (uint32_t step = 0; step < 90; ++step) {
        solver.Step(world);
        // In the air it touches nothing, so the stack may sleep.
        // Once it lands, no body may sleep.
        const bool landed = world.Poses[falling].Position.y < Half + 4;
        if (!landed) continue;
        // Measured by motion rather than the quiet counter, since a just-woken body has a count of zero while still at rest.
        uint32_t moving = 0, sleeping = 0;
        for (const auto body : stack) {
            moving += simd::length(world.Velocities[body].Linear) > settings.SleepSpeed ? 1 : 0;
            sleeping += world.Quiet[body] >= settings.SleepSteps ? 1 : 0;
        }
        CAPTURE(step);
        CHECK((moving == 0 || sleeping == 0));
    }
}

TEST_CASE_FIXTURE(OneWorld, "a body only collides with what its mask says it does") {
    // Each is in the ground's mask so both land, and neither is in the other's, so they end up in the same place.
    constexpr uint32_t Ground = 1, First = 2, Second = 4;
    const auto shape = world.AddShape(UnitBox);
    AddGround(world, {.Layer = Ground, .CollidesWith = First | Second});
    const auto low = world.AddBody({.Pose = At(float3{0, Half + 1, 0}), .Shape = shape, .Layer = First, .CollidesWith = Ground});
    const auto high = world.AddBody({.Pose = At(float3{0, Half + 3, 0}), .Shape = shape, .Layer = Second, .CollidesWith = Ground});

    Run(solver, world, 400);
    CheckResting(world.Poses[low].Position.y);
    CheckResting(world.Poses[high].Position.y); // it passes through the other one and rests on the floor
    CHECK(std::abs(world.Poses[high].Position.y - world.Poses[low].Position.y) < 1e-3f);
}

TEST_CASE_FIXTURE(OnDevice, "a joint between two moving bodies gives each what it takes from the other") {
    // Every other joint test here hangs off a static pivot, which exercises half a joint.
    // A ball joint applies its force at one point on both bodies, so the pair's linear and angular momentum are unchanged.
    const StepSettings settings{.Gravity = {0, 0, 0}}; // nothing outside the pair to account for

    // Total momentum, and angular momentum about the pair's own centre of mass.
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
            // The body's inertia in world axes, its local inertia rotated by the body's orientation.
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
        // The velocity of the pair's centre of mass, which nothing in the scene changes.
        const float3 together = began.Linear * (world.Masses[a].InvMass * world.Masses[b].InvMass) /
            (world.Masses[a].InvMass + world.Masses[b].InvMass);
        float worst_hold = 0, worst_drift = 0;
        for (uint32_t step = 0; step < 600; ++step) {
            solver.Step(world, settings);
            const Pose &pa = world.Poses[a], &pb = world.Poses[b];
            // The joint's single point, computed from each body's current pose.
            worst_hold = std::max(worst_hold, simd::distance(WorldPoint(pa, world.Joints[0].AnchorA), WorldPoint(pb, world.Joints[0].AnchorB)));
            // Drift measured in steps of the centre's own travel, since a pose and a velocity are half a step out of phase.
            // A constant offset of that size is bookkeeping, and a growing one is a force.
            const Momentum now = momentum(world, a, b);
            const float3 carried = started + together * (float(step + 1) * settings.DeltaTime);
            worst_drift = std::max(worst_drift, simd::distance(now.Centre, carried) / (settings.DeltaTime * simd::length(together)));
        }
        return std::tuple{began, momentum(world, a, b), worst_hold, worst_drift};
    };

    SUBCASE("equal masses") {
        const auto [began, ended, hold, drift] = swing(1000);
        CHECK(hold < 5e-3f); // the anchor stays one point throughout, as it does off a static pivot
        // Nothing acts from outside the pair, so its momentum is conserved.
        CHECK(simd::length(ended.Linear - began.Linear) < 1e-3f * simd::length(began.Linear));
        CHECK(simd::length(ended.Angular - began.Angular) < 2e-2f * simd::length(began.Angular));
        CHECK(drift < 1); // the centre of mass stays within the half step it is read behind
    }

    SUBCASE("a thousand to one in mass") {
        // Sec. 3.4's stiffness ratio on a joint rather than a contact: neither body drags the other.
        const auto [began, ended, hold, drift] = swing(1);
        CHECK(hold < 5e-3f);
        CHECK(simd::length(ended.Linear - began.Linear) < 1e-3f * simd::length(began.Linear));
        CHECK(drift < 1);
    }
}

TEST_CASE_FIXTURE(OnDevice, "a joint stops its two bodies colliding, unless it is asked not to") {
    // The contacts an overlap generates oppose the joint, so a joint disables them by default.
    const auto overlapping = [&](bool collide) {
        World world{context};
        const auto shape = world.AddShape(UnitBox);
        const auto left = world.AddBody({.Pose = At(float3{0, 4, 0}), .Shape = shape});
        const auto right = world.AddBody({.Pose = At(float3{Half, 4, 0}), .Shape = shape});
        world.AddJoint({.BodyA = left, .BodyB = right, .At = {0.25f, 4, 0}, .Collide = collide});
        Run(solver, world, 60, {.Gravity = {0, 0, 0}}); // no gravity, so contacts are the only thing acting
        return ActiveContacts(world);
    };
    CHECK(overlapping(false) == 0);
    CHECK(overlapping(true) > 0);
}

// The behavioural scenes have no closed form, so each asserts an invariant that holds for any outcome.
namespace {
// The extremes over a set of dynamic bodies, the values the plausibility invariants are built from.
struct Worst {
    float Speed{}, Lowest{1e9f}, Highest{}, Reach{};
};

Worst WorstOf(const World &world, uint32_t from) {
    Worst worst;
    for (uint32_t body = from; body < world.BodyCount(); ++body) {
        if (world.Masses[body].InvMass == 0) continue;
        const auto &at = world.Poses[body].Position;
        worst.Speed = std::max(worst.Speed, simd::length(world.Velocities[body].Linear));
        worst.Lowest = std::min(worst.Lowest, at.y);
        worst.Highest = std::max(worst.Highest, at.y);
        worst.Reach = std::max(worst.Reach, std::max(std::abs(at.x), std::abs(at.z)));
    }
    return worst;
}
} // namespace

TEST_CASE_FIXTURE(OneWorld, "a pyramid holds itself up") {
    // Unlike a stack, every box but the bottom row rests on two below it and is held only by friction.
    const auto shape = world.AddShape(UnitBox);
    AddGround(world);
    for (uint32_t row = 0; row < 5; ++row)
        for (uint32_t i = 0; i < 5 - row; ++i)
            world.AddBody({.Pose = At(float3{1.02f * (float(i) + 0.5f * float(row) - 2), Half + 1.02f * float(row), 0}), .Shape = shape});
    const Index top = world.BodyCount() - 1;
    const float was = world.Poses[top].Position.x;

    Run(solver, world, 600);
    const auto worst = WorstOf(world, 1);
    CHECK(worst.Speed < 1e-2f);
    CHECK(worst.Lowest > Half - 4 * StepSettings{}.ContactMargin);
    CHECK(world.Poses[top].Position.y == doctest::Approx(Half + 4).epsilon(0.01)); // still five rows tall
    CHECK(std::abs(world.Poses[top].Position.x - was) < 0.05f); // the apex stays over the base
}

TEST_CASE_FIXTURE(OneWorld, "a light box carries a box a thousand times its weight") {
    // Sec. 3.4's stiffness ratio: the augmented Lagrangian keeps the heavy body from squeezing the light one flat.
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

TEST_CASE_FIXTURE(OneWorld, "a chain of ten hangs from its anchor without coming apart") {
    // Released straight out sideways, the worst load a chain can take, with no damping.
    // It never comes to rest, so the only requirement is that it stays joined.
    constexpr uint32_t Links = 10;
    const auto shape = world.AddShape(UnitBox);
    const auto anchor = world.AddBody({});
    Index above = anchor;
    for (uint32_t i = 0; i < Links; ++i) {
        const auto link = world.AddBody({.Pose = At(float3{1.f + float(i), 0, 0}), .Shape = shape});
        world.AddJoint({.BodyA = link, .BodyB = above, .At = {Half + float(i), 0, 0}});
        above = link;
    }

    float stretched = 0, furthest = 0;
    for (uint32_t step = 0; step < 900; ++step) {
        solver.Step(world);
        for (uint32_t index = 0; index < world.JointCount(); ++index) {
            const auto &joint = world.Joints[index];
            const auto &a = world.Poses[joint.BodyA];
            const auto &b = world.Poses[joint.BodyB];
            const float3 apart = WorldPoint(a, joint.AnchorA) - WorldPoint(b, joint.AnchorB);
            stretched = std::max(stretched, simd::length(apart));
        }
        furthest = std::max(furthest, WorstOf(world, 1).Reach);
    }
    CHECK(stretched < 0.03f); // a joint holds one point to one point
    CHECK(furthest < Links + Half + 0.1f); // no link goes further out than the chain can reach
}

TEST_CASE_FIXTURE(OneWorld, "a heap of boxes dropped into a bin settles inside it") {
    // Many bodies at once against static geometry other than a plane: they all stop, none is squeezed through a wall, and no tower is left unsupported.
    const auto shape = world.AddShape(UnitBox);
    const auto wall = world.AddShape(Shape{.HalfExtents = {2.5f, 2, 0.25f}, .Kind = ShapeBox});
    AddGround(world);
    for (uint32_t side = 0; side < 4; ++side) {
        const float turn = 2 * std::atan(1.f) * float(side);
        world.AddBody({.Pose = At(float3{std::sin(turn) * 2.25f, 2, std::cos(turn) * 2.25f}, QuatFromRotationVector(float3{0, turn, 0})), .Shape = wall, .Density = 0});
    }
    for (uint32_t i = 0; i < 18; ++i)
        world.AddBody({.Pose = At(float3{1.2f * float(i % 3) - 1.2f, 5.f + 1.3f * float(i / 3), 1.2f * float(i / 3 % 3) - 1.2f}), .Shape = shape});

    float widest = 0;
    for (uint32_t step = 0; step < 900; ++step) {
        solver.Step(world);
        widest = std::max(widest, WorstOf(world, 5).Reach);
    }
    const auto worst = WorstOf(world, 5);
    CHECK(worst.Speed < 1e-2f);
    CHECK(widest < 2.25f - 0.25f); // never reaches the inside face of a wall
    CHECK(worst.Lowest > Half - 4 * StepSettings{}.ContactMargin);
    CHECK(worst.Highest < Half + 3); // no unsupported tower is left standing
}

TEST_CASE_FIXTURE(OneWorld, "contact events report a manifold arriving, holding and going away") {
    // The distinction CollectContacts already draws for warm starting.
    // A feature present last step is carried forward, an unseen one is new, and an unmatched one has ended.
    const auto box = DropBox(world, Half, true); // already touching, so the first step forms the manifold

    Run(solver, world, 1);
    // The box owns every event, since a plane generates no manifold and a contact is reported by its body A.
    CHECK(Reported(world, box, ContactAdded) == 4);
    CHECK(Reported(world, box, ContactPersisted) == 0);
    CHECK(Reported(world, box, ContactRemoved) == 0);
    CHECK(world.ContactEventCounts[0] == 0); // the plane reports nothing for a contact it does not own
    for (uint32_t i = 0; i < world.ContactEventCounts[box]; ++i) {
        const auto &event = world.ContactEvents[box * EventsPerBody + i];
        CHECK(event.BodyA == box);
        CHECK(event.BodyB == 0);
    }

    Run(solver, world, 1);
    CHECK(Reported(world, box, ContactAdded) == 0);
    CHECK(Reported(world, box, ContactPersisted) == 4); // the same four, by the same four features
    CHECK(Reported(world, box, ContactRemoved) == 0);

    // Moved out of reach rather than removed, which keeps this about the events rather than the pools.
    world.Poses[box].Position = {0, Half + 10, 0};
    world.Velocities[box] = {};
    Run(solver, world, 1);
    CHECK(Reported(world, box, ContactRemoved) == 4);
    CHECK(Reported(world, box, ContactAdded) == 0);
    CHECK(Reported(world, box, ContactPersisted) == 0);

    Run(solver, world, 1);
    CHECK(world.ContactEventCounts[box] == 0);
}

TEST_CASE_FIXTURE(OnDevice, "two runs report the same contact events in the same order") {
    // Nothing appends atomically and nothing sorts, since each body writes its own fixed run in slot order.

    const auto run = [&] {
        World world{context};
        // Tips onto a corner and rolls, so features come and go rather than a manifold forming once.
        TumblingBox(world);
        std::vector<ContactEvent> seen;
        for (uint32_t step = 0; step < 90; ++step) {
            solver.Step(world);
            for (uint32_t body = 0; body < world.BodyCount(); ++body)
                for (uint32_t i = 0; i < world.ContactEventCounts[body]; ++i)
                    seen.push_back(world.ContactEvents[body * EventsPerBody + i]);
        }
        return seen;
    };

    const auto first = run(), second = run();
    REQUIRE(first.size() == second.size());
    REQUIRE(first.size() > 8); // enough events to make the comparison meaningful
    for (size_t at = 0; at < first.size(); ++at) {
        CAPTURE(at);
        CHECK(std::memcmp(&first[at], &second[at], sizeof(ContactEvent)) == 0);
    }
}

TEST_CASE_FIXTURE(OneWorld, "a body with more contacts than slots says so instead of dropping them quietly") {
    // A dense pile can need more than ContactsPerBody, and a dropped contact is reported rather than silent.
    const StepSettings settings{.Gravity = {0, 0, 0}}; // nothing moves, so the count is of the geometry alone
    const auto shape = world.AddShape(UnitBox);
    // The middle box is the only dynamic body, so it owns every manifold.
    // Ten face partners fill the run exactly and a full run is not an overflow, so twelve are used.
    // A sphere touching at one point places the refusal on a single point rather than a whole manifold.
    const auto middle = world.AddBody({.Shape = shape});
    for (const auto at : {float3{0.999f, 0, 0.5f}, float3{0.999f, 0, -0.5f}, float3{-0.999f, 0, 0.5f}, float3{-0.999f, 0, -0.5f}, float3{0.5f, 0, 0.999f}, float3{-0.5f, 0, 0.999f}, float3{0.5f, 0, -0.999f}, float3{-0.5f, 0, -0.999f}, float3{0.5f, 0.999f, 0}, float3{-0.5f, 0.999f, 0}, float3{0.5f, -0.999f, 0}, float3{-0.5f, -0.999f, 0}})
        world.AddBody({.Pose = At(at), .Shape = shape, .Density = 0});
    world.AddBody({.Pose = At(float3{0, -1.24f, 0}), .Shape = world.AddShape({.Radius = 0.75f, .Kind = ShapeSphere}), .Density = 0});

    Run(solver, world, 1, settings);
    REQUIRE(ActiveContacts(world, middle) == ContactsPerBody);
    // Forty-nine points in total, an exact count rather than a lower bound.
    // Every partner is collided whether or not the run is full, so which contacts a body keeps is independent of the order they are found in.
    CHECK(world.ContactRefusals[middle] == 49 - ContactsPerBody);
}

TEST_CASE_FIXTURE(OneWorld, "a contact in the run's high slots is claimed rather than reported gone") {
    // The claimed-slot mask is one bit per slot, so a word narrower than the run folds the high slots onto the low ones.
    // Everything past the last bit is then removed and re-added every step, and nothing else here fills a run past bit 31.
    const StepSettings settings{.Gravity = {0, 0, 0}}; // nothing moves, so the count is of the geometry alone
    const auto shape = world.AddShape(UnitBox);
    // Ten partners of four points, filling the run exactly, symmetric, and with no refusals.
    const auto middle = world.AddBody({.Shape = shape});
    std::vector<Index> partners;
    for (const auto at : {float3{0.999f, 0, 0.5f}, float3{0.999f, 0, -0.5f}, float3{-0.999f, 0, 0.5f}, float3{-0.999f, 0, -0.5f}, float3{0.5f, 0, 0.999f}, float3{-0.5f, 0, 0.999f}, float3{0.5f, 0, -0.999f}, float3{-0.5f, 0, -0.999f}, float3{0, 0.999f, 0}, float3{0, -0.999f, 0}})
        partners.push_back(world.AddBody({.Pose = At(at), .Shape = shape, .Density = 0}));

    Run(solver, world, 1, settings);
    REQUIRE(ActiveContacts(world, middle) == ContactsPerBody);
    REQUIRE(world.ContactRefusals[middle] == 0);
    REQUIRE(Reported(world, middle, ContactAdded) == ContactsPerBody); // every slot is new on the first step

    Run(solver, world, 1, settings);
    CHECK(ActiveContacts(world, middle) == ContactsPerBody);
    CHECK(Reported(world, middle, ContactPersisted) == ContactsPerBody);
    CHECK(Reported(world, middle, ContactRemoved) == 0);
    CHECK(Reported(world, middle, ContactAdded) == 0);

    // A full run reads the same under a folded mask either way, so only the step where the low four end and the high four persist distinguishes them.
    // The partner is moved rather than removed, since a removal ends its contacts at the call and this exercises the kernel's own unclaimed-slot path.
    world.Poses[partners.front()].Position.y = 100;
    world.Wake(partners.front());
    Run(solver, world, 1, settings);
    CHECK(ActiveContacts(world, middle) == ContactsPerBody - ManifoldPoints);
    CHECK(Reported(world, middle, ContactRemoved) == ManifoldPoints);
    CHECK(Reported(world, middle, ContactPersisted) == ContactsPerBody - ManifoldPoints);
}

namespace {
// A stack of two boxes on the plane.
// Every manifold belongs to the lower box, since a pair is owned by its lower-indexed body and a plane owns none.
struct Stacked {
    Index Lower, Upper;
};
Stacked TwoOnAPlane(World &world) {
    const auto shape = world.AddShape(UnitBox);
    AddGround(world);
    return {world.AddBody({.Pose = At(float3{0, Half, 0}), .Shape = shape}), world.AddBody({.Pose = At(float3{0, Half + 1, 0}), .Shape = shape})};
}
} // namespace

TEST_CASE_FIXTURE(OneWorld, "a removed body's contacts end at the removal, and every one is on the stream") {
    // Removal ends its contacts in the same call, so a consumer receives the end events immediately.
    world.TrackContacts = true;
    const auto [lower, upper] = TwoOnAPlane(world);
    Run(solver, world, 200);
    REQUIRE(ActiveContacts(world, lower) == 8); // four against the plane and four against the box above
    REQUIRE(ActiveContacts(world, upper) == 0);
    (void)world.TakeContactChanges(); // the descent is not under test

    SUBCASE("the body underneath hears the pairs it was holding end") {
        REQUIRE(world.RemoveBody(upper));
        CHECK(ActiveContacts(world, lower) == 4); // cleared from the run at the call rather than on the next step
        const auto changes = world.TakeContactChanges();
        REQUIRE(changes.size() == 4);
        for (const auto &change : changes) {
            CHECK(change.Kind == ContactRemoved);
            CHECK(change.A == world.IdOf(lower));
            CHECK(change.B.Slot == upper);
            CHECK(change.Lambda[0] == 0); // a removal carries no excitation
        }
        // The next step adds nothing: the plane's four persist and nothing is removed twice.
        Run(solver, world, 1);
        CHECK(Reported(world, lower, ContactRemoved) == 0);
        CHECK(Reported(world, lower, ContactPersisted) == 4);
        CHECK(Reported(world, lower, ContactAdded) == 0);
        CHECK(world.ContactEventCounts[upper] == 0); // it owned none of them, so it reports none
    }

    SUBCASE("and a body that owned them ends all of them itself") {
        REQUIRE(world.RemoveBody(lower));
        const auto changes = world.TakeContactChanges();
        REQUIRE(changes.size() == 8);
        CHECK(std::ranges::all_of(changes, [](const ContactChange &change) { return change.Kind == ContactRemoved; }));
        CHECK(std::ranges::none_of(world.Contacts.All(), [lower = lower](const Contact &contact) {
            return contact.Active && (contact.BodyA == lower || contact.BodyB == lower);
        }));
        Run(solver, world, 1);
        CHECK(Reported(world, lower, ContactRemoved) == 0); // reported once, at the removal
    }
}

TEST_CASE_FIXTURE(OneWorld, "the stream carries the excitation record the contact is applying") {
    // The modal layer reads the force each row applies in the contact's basis, the closing speed, and the restitution contribution.
    // A normal row only pushes, so Lambda[0] < 0 means engaged.
    world.TrackContacts = true;
    const auto box = DropBox(world, Half, true);
    Run(solver, world, 1);
    auto changes = world.TakeContactChanges();
    REQUIRE(changes.size() == 4);
    CHECK(std::ranges::all_of(changes, [](const ContactChange &change) { return change.Kind == ContactAdded; }));

    Run(solver, world, 120); // at rest and asleep, with its contacts carried frozen including the duals
    (void)world.TakeContactChanges();
    Run(solver, world, 1);
    changes = world.TakeContactChanges();
    REQUIRE(changes.size() == 4);
    float load = 0;
    for (const auto &change : changes) {
        CHECK(change.Kind == ContactPersisted);
        CHECK(change.A == world.IdOf(box));
        CHECK(change.Lambda[0] < 0); // engaged, bearing load rather than merely touching
        load += -change.Lambda[0];
    }
    CHECK(load == doctest::Approx(1000 * Gravity).epsilon(0.01)); // the box's weight, in newtons
}

TEST_CASE_FIXTURE(OneWorld, "a shape swapped mid-contact ends its contacts on the stream, on both sides") {
    // The swap that matters is to a shape whose features recur, since a recurring feature would give the new geometry the old shape's dual and stick anchors.
    // The contacts end at the swap.
    world.TrackContacts = true;
    const auto [lower, upper] = TwoOnAPlane(world);
    Run(solver, world, 5);
    REQUIRE(ActiveContacts(world, lower) == 8);
    (void)world.TakeContactChanges();

    REQUIRE(world.SetBodyShape(upper, world.AddShape(UnitBox)));
    const auto changes = world.TakeContactChanges();
    REQUIRE(changes.size() == 4); // the four the lower box owned against it, ended at the call
    for (const auto &change : changes) {
        CHECK(change.Kind == ContactRemoved);
        CHECK(change.A == world.IdOf(lower));
        CHECK(change.B == world.IdOf(upper));
    }
    Run(solver, world, 1);
    CHECK(Reported(world, lower, ContactAdded) == 4); // the same geometry, reported as new
    CHECK(Reported(world, lower, ContactPersisted) == 4); // the plane's four, unchanged
    CHECK(Reported(world, lower, ContactRemoved) == 0); // nothing is removed twice
}

TEST_CASE_FIXTURE(OneWorld, "a respawned slot is a different body on the stream") {
    // A consumer holding last step's identities has to tell a new occupant of a slot from the previous body, which the spawn counter provides.
    world.TrackContacts = true;
    const auto [lower, upper] = TwoOnAPlane(world);
    Run(solver, world, 5);
    (void)world.TakeContactChanges();

    const BodyId first = world.IdOf(upper);
    REQUIRE(world.RemoveBody(upper));
    Run(solver, world, 1); // the step that retires the slot, after which it can be handed out again
    (void)world.TakeContactChanges();
    const auto again = world.AddBody({.Pose = At(float3{0, Half + 1 - 1e-3f, 0}), .Shape = world.BodyShapes[lower]});
    REQUIRE(again == upper); // the same slot
    CHECK(world.IdOf(again).Spawn == first.Spawn + 1); // a different body

    Run(solver, world, 1);
    const auto changes = world.TakeContactChanges();
    uint32_t added = 0;
    for (const auto &change : changes) {
        CHECK(change.B != first); // the old identity matches nothing after its removal
        added += change.B == world.IdOf(again) && change.Kind == ContactAdded ? 1 : 0;
    }
    CHECK(added == 4);
}

TEST_CASE_FIXTURE(OneWorld, "a sleeping island goes on reporting through a mutation elsewhere") {
    // A sleeping island re-emits its contacts as persisted every step, carried frozen with their duals.
    // A mutation elsewhere neither wakes it nor alters its reported contacts.
    const StepSettings settings{};
    world.TrackContacts = true;
    const auto [lower, upper] = TwoOnAPlane(world);
    const auto bystander = world.AddBody({.Pose = At(float3{5, Half, 0}), .Shape = world.BodyShapes[lower]});
    Run(solver, world, 200);
    REQUIRE(world.Quiet[lower] >= settings.SleepSteps);
    (void)world.TakeContactChanges();

    REQUIRE(world.RemoveBody(bystander));
    Run(solver, world, 1);
    uint32_t removed = 0, persisted = 0;
    for (const auto &change : world.TakeContactChanges()) {
        if (change.Kind == ContactRemoved) {
            ++removed;
            CHECK(change.A == world.IdOf(bystander)); // the synthesized four, against the plane
        } else {
            ++persisted;
            CHECK(change.Kind == ContactPersisted);
            CHECK(change.A == world.IdOf(lower)); // the island's contacts, under unchanged identities
            CHECK(change.Lambda[0] < 0); // still engaged, the dual carried frozen through sleep
        }
    }
    CHECK(removed == 4);
    CHECK(persisted == 8);
    CHECK(world.Quiet[lower] >= settings.SleepSteps); // it stays asleep throughout
}

TEST_CASE_FIXTURE(OneWorld, "what a removed body was holding up wakes and falls") {
    // A removed body has no motion, so without the removal waking them a stack sleeps on in mid air.
    const StepSettings settings{};
    const auto [lower, upper] = TwoOnAPlane(world);
    Run(solver, world, 400);
    REQUIRE(world.Quiet[upper] >= settings.SleepSteps);

    SUBCASE("the box under it") {
        REQUIRE(world.RemoveBody(lower));
        CHECK(world.Quiet[upper] == 0); // woken at once, by the removal itself
        Run(solver, world, 300);
        CheckResting(world.Poses[upper].Position.y);
    }

    SUBCASE("or the ground itself") {
        // A plane owns no manifold, so everything resting on it references it as body B, through the incoming list.
        REQUIRE(world.RemoveBody(0));
        CHECK(world.Quiet[lower] == 0); // the box that was on it, woken by the removal
        CHECK(world.Quiet[upper] != 0); // the box above is one link further away
        Run(solver, world, 1);
        CHECK(world.Quiet[upper] == 0); // one step later, since nothing sleeps on a moving body
        Run(solver, world, 60);
        CHECK(world.Poses[lower].Position.y < 0);
        CHECK(world.Poses[upper].Position.y < 1);
    }
}

TEST_CASE_FIXTURE(OneWorld, "a host pose write holds on a sleeping body, and wakes what it is told to") {
    // World::Wake is how the host marks a pose write as a change rather than a cache restore, a distinction nothing else can make.
    // The write itself holds either way.
    const StepSettings settings{};
    const auto asleep = [&](Index body) { return world.Quiet[body] >= settings.SleepSteps; };

    const auto shape = world.AddShape(UnitBox);
    AddGround(world);
    // In the air, because a slab on the ground would leave its box resting where it already was.
    const auto slab = world.AddBody({.Pose = At(float3{0, 1.75f, 0}), .Shape = world.AddShape({.HalfExtents = {2, 0.25f, 2}, .Kind = ShapeBox}), .Density = 0});
    const auto carried = world.AddBody({.Pose = At(float3{0, 2 + Half, 0}), .Shape = shape});
    const auto standing = world.AddBody({.Pose = At(float3{4, Half, 0}), .Shape = shape});
    Run(solver, world, 400);
    REQUIRE(asleep(carried));
    REQUIRE(asleep(standing));
    const float carried_y = world.Poses[carried].Position.y, standing_x = world.Poses[standing].Position.x;

    SUBCASE("a static body moved out from under a sleeper") {
        world.Poses[slab].Position.x += 6;

        SUBCASE("wakes it when the host says so") {
            world.Wake(slab); // through the incoming list: a box owns its pair against a static slab
            CHECK(!asleep(carried));
            Run(solver, world, 300);
            CheckResting(world.Poses[carried].Position.y);
        }

        SUBCASE("and leaves it asleep in mid air when it does not") {
            Run(solver, world, 300);
            CHECK(asleep(carried));
            CHECK(world.Poses[carried].Position.y == carried_y); // the pose is unchanged, which is the contract
        }
    }

    SUBCASE("a sleeper teleported into another") {
        // A deep overlap rather than a touch, so nothing here turns on the contact margin.
        const float landed = standing_x - 1 + 0.8f;
        world.Poses[carried].Position = {landed, Half, 0};

        SUBCASE("pushes it when the host says so") {
            world.Wake(carried);
            Run(solver, world, 300);
            CHECK(world.Poses[standing].Position.x > standing_x + 0.2f);
            CHECK(world.Poses[carried].Position.x < landed); // and it is pushed back by the same contact
            CheckResting(world.Poses[standing].Position.y);
        }

        SUBCASE("and pushes nothing when it does not") {
            // A sleeping pair is carried forward, so the authored overlap is the state of the world.
            Run(solver, world, 300);
            CHECK(asleep(standing));
            CHECK(world.Poses[standing].Position.x == standing_x);
            CHECK(world.Poses[carried].Position.x == landed); // the write holds, with the body unsolved
        }
    }
}

TEST_CASE_FIXTURE(OneWorld, "a box rides a slab the host drives under it") {
    // Kinematics: a body with no inverse mass whose pose and velocity the host writes every step, so its whole effect comes from those two fields.
    const StepSettings settings{};
    constexpr float Speed = 2, Gravity = 9.81f;
    const auto slab = world.AddBody({.Pose = At(float3{0, -0.25f, 0}), .Shape = world.AddShape({.HalfExtents = {20, 0.25f, 4}, .Kind = ShapeBox}), .Density = 0});
    const auto box = world.AddBody({.Pose = At(float3{0, Half, 0}), .Shape = world.AddShape(UnitBox)});
    Run(solver, world, 200);
    REQUIRE(world.Quiet[box] >= settings.SleepSteps); // at rest and asleep, the hard case

    float3 at = world.Poses[slab].Position;
    const auto drive = [&](uint32_t steps) { DriveAlongX(solver, world, slab, at, Speed, steps, settings, [] {}); };

    // Long enough for static friction to take it up to speed at mu g, mu being the geometric mean.
    drive(120);
    CHECK(world.Velocities[box].Linear.x == doctest::Approx(Speed).epsilon(0.01));
    CHECK(world.Quiet[box] == 0u); // it stays awake while being carried

    // The slip is the transient alone, and once up to speed the box holds its place to the micron.
    const float slipped = float(world.Poses[box].Position.x - world.Poses[slab].Position.x);
    const float predicted = 0.5f * Speed * Speed / (0.5f * Gravity); // half v^2 / a, the distance lost
    CHECK(std::abs(-slipped - predicted) < 0.15f);
    drive(120);
    CHECK(float(world.Poses[box].Position.x - world.Poses[slab].Position.x) == doctest::Approx(slipped).epsilon(1e-4));
    CheckResting(world.Poses[box].Position.y - 0);
}

TEST_CASE_FIXTURE(OneWorld, "a kinematic paddle strikes a ball rather than passing through it") {
    // In free space, so the ball's motion afterwards comes from the paddle alone.
    // The reach only creates the contact, and the strike is the paddle's displacement in Eq. 15's J dq, placed there by the warm start.
    constexpr float Swing = 6, Radius = 0.25f;
    const StepSettings settings{.Gravity = {0, 0, 0}};
    const auto paddle = world.AddBody({.Pose = At(float3{-2, 0, 0}), .Shape = world.AddShape({.HalfExtents = {0.1f, 1, 1}, .Kind = ShapeBox}), .Density = 0});
    const auto ball = world.AddBody({.Pose = At(float3{0, 0, 0}), .Shape = world.AddShape({.Radius = Radius, .Kind = ShapeSphere})});

    // Asleep first, the hard case, since a sleeper skips its pairs with massless bodies and a paddle owns no manifold of its own.
    Run(solver, world, 60, settings);
    REQUIRE(world.Quiet[ball] >= settings.SleepSteps);

    float3 at = world.Poses[paddle].Position;
    // The ball stays in front of the paddle, since ending up behind it is the one outcome nothing recovers from.
    DriveAlongX(solver, world, paddle, at, Swing, 90, settings, [&] { CHECK(float(world.Poses[ball].Position.x) + Radius > float(world.Poses[paddle].Position.x)); });
    // An immovable face at e = 0 leaves the ball at exactly the face's speed.
    CHECK(world.Velocities[ball].Linear.x == doctest::Approx(Swing).epsilon(0.01));

    // Keeping that speed once the paddle stops distinguishes a strike from a shove, since a body moved only by post-stabilization carries no velocity.
    Drive(world, paddle, at, float3(0));
    const float carried = float(world.Poses[ball].Position.x);
    Run(solver, world, 60, settings);
    CHECK(world.Velocities[ball].Linear.x == doctest::Approx(Swing).epsilon(0.01));
    CHECK(world.Poses[ball].Position.x > carried + 0.5f * Swing * 60 * settings.DeltaTime);
}

TEST_CASE_FIXTURE(OneWorld, "a sleeping stack wakes when the slab under it starts moving") {
    // A body with no inverse mass has no quiet count, so the wake spread reaches it by a separate path.
    const StepSettings settings{};
    // Gently, because a stack is not a box: static friction accelerates each layer at no more than mu g,
    // so a slab jumping to v slides every level v^2 / 2 mu g behind the one under it and topples.
    constexpr float Speed = 0.5f;
    const auto slab = world.AddBody({.Pose = At(float3{0, -0.25f, 0}), .Shape = world.AddShape({.HalfExtents = {20, 0.25f, 4}, .Kind = ShapeBox}), .Density = 0});
    const auto shape = world.AddShape(UnitBox);
    const std::vector<Index> stack = AddStack(world, shape, 3);
    Run(solver, world, 400);
    for (const auto body : stack) REQUIRE(world.Quiet[body] >= settings.SleepSteps);

    float3 at = world.Poses[slab].Position;
    const auto drive = [&](uint32_t steps) { DriveAlongX(solver, world, slab, at, Speed, steps, settings, [] {}); };

    // Waking travels one contact a step, so three steps covers a stack of three.
    drive(3);
    for (const auto body : stack) CHECK(world.Quiet[body] == 0u);

    drive(200);
    for (const auto body : stack) CHECK(world.Velocities[body].Linear.x == doctest::Approx(Speed).epsilon(0.02));
    for (uint32_t i = 0; i < stack.size(); ++i) {
        CAPTURE(i);
        CHECK(world.Poses[stack[i]].Position.y == doctest::Approx(Half + float(i) * (1 - settings.ContactMargin)).epsilon(0.01));
    }
}

TEST_CASE_FIXTURE(OnDevice, "a slot handed out again behaves as a fresh one") {
    // The same scene twice, the second time into slots two other boxes came to rest in first.
    // A slot carries a colour, a rest pose, a contact run and a sleep counter, and any residue shows up as divergence.
    const auto run = [&](bool recycled) {
        World world{context};
        const auto shape = world.AddShape(UnitBox);
        AddGround(world);
        if (recycled) {
            const auto first = world.AddBody({.Pose = At(float3{0, Half, 0}), .Shape = shape});
            const auto second = world.AddBody({.Pose = At(float3{0, Half + 1.02f, 0}), .Shape = shape});
            Run(solver, world, 90); // long enough to reach rest, take colours and sleep
            // A colour is the one field a step reads before it writes, and the highest value on any body sets how many colours the next step dispatches.
            // Set by hand here rather than built with a raft.
            world.Colors[first] = 3;
            world.Colors[second] = 2;
            REQUIRE(world.RemoveBody(first));
            REQUIRE(world.RemoveBody(second));
            Run(solver, world, 1); // the step that reports them gone and hands the slots back
            REQUIRE(world.BodyCount() == 1);
        }
        // Off-axis and tumbling, so any difference in stepping shows up here.
        const auto box = world.AddBody({.Pose = At(float3{0.2f, 1.4f, -0.1f}, QuatFromRotationVector(float3{0.3f, 0.1f, 0.2f})), .Velocity = {.Linear = {0.4f, 0, -0.2f}, .Angular = {0.1f, 0.7f, 0}}, .Shape = shape});
        const auto other = world.AddBody({.Pose = At(float3{0, 3, 0}), .Shape = shape});
        REQUIRE(box == 1);
        REQUIRE(other == 2);
        Run(solver, world, 150);
        return Snapshot(world);
    };

    const auto fresh = run(false), recycled = run(true);
    CheckIdentical(fresh, recycled);
}

TEST_CASE_FIXTURE(OneWorld, "a slot is not handed out again until a step has reported what left it") {
    // One step of a slot standing idle keeps the events unambiguous: the body that had the contacts reports them gone, rather than the next occupant.
    const auto shape = world.AddShape(UnitBox);
    AddGround(world);
    const auto first = world.AddBody({.Pose = At(float3{0, Half, 0}), .Shape = shape});
    world.AddBody({.Pose = At(float3{3, Half, 0}), .Shape = shape});
    Run(solver, world, 30);

    REQUIRE(world.RemoveBody(first));
    CHECK(!world.Alive(first));
    const auto immediate = world.AddBody({.Pose = At(float3{6, Half, 0}), .Shape = shape});
    CHECK(immediate != first); // still occupied by a body with removals to report
    Run(solver, world, 1);
    const auto later = world.AddBody({.Pose = At(float3{9, Half, 0}), .Shape = shape});
    CHECK(later == first);
    CHECK(world.Alive(later));
}

TEST_CASE_FIXTURE(OneWorld, "a retired joint stops holding, and lets its bodies collide again") {
    // A joint suppresses contacts between its bodies, so retiring it clears that entry from both runs and the two collide again.
    const StepSettings settings{.Gravity = {0, 0, 0}}; // so the only force in the scene is the contact
    const auto shape = world.AddShape(UnitBox);
    const auto anchor = world.AddBody({.Shape = shape, .Density = 0});
    const auto hanging = world.AddBody({.Pose = At(float3{0.5f, 0, 0}), .Shape = shape});
    const auto joint = world.AddJoint({.BodyA = hanging, .BodyB = anchor, .At = {0.5f, 0, 0}});
    REQUIRE(joint != NoIndex);
    Run(solver, world, 60, settings);
    REQUIRE(world.Poses[hanging].Position.x == doctest::Approx(0.5f).epsilon(0.01)); // held where it was

    SUBCASE("retired directly") {
        REQUIRE(world.RemoveJoint(joint));
        CHECK(!world.RemoveJoint(joint)); // a second removal reports failure
        CHECK(world.JointCount() == 0); // the slot is freed, and it was the last joint
        CHECK(world.Jointed[hanging * JointsPerBody] == NoIndex);
        CHECK(world.Jointed[anchor * JointsPerBody] == NoIndex);
        Run(solver, world, 200, settings);
        CHECK(world.Poses[hanging].Position.x > 1 - MaxPenetration); // pushed out to where it touches
    }

    SUBCASE("or with the body it held") {
        REQUIRE(world.RemoveBody(hanging));
        CHECK(world.JointCount() == 0); // a joint to a body that is gone would hold the other end to nothing
        CHECK(world.Jointed[anchor * JointsPerBody] == NoIndex);
    }
}

TEST_CASE_FIXTURE(OnDevice, "the same mutation script steps to bit-identical state twice") {
    // Which slot an add lands in is a function of what was removed, so a mutated world is reproducible only if the free list is deterministic.
    CheckReplay(context, [&](World &world) {
        const auto shape = world.AddShape(UnitBox);
        AddGround(world);
        AddStack(world, shape, 3);
        Run(solver, world, 60);
        REQUIRE(world.RemoveBody(2)); // out of the middle of the stack, so the one above it falls
        Run(solver, world, 20);
        world.AddBody({.Pose = At(float3{0.3f, 4, 0.2f}), .Shape = shape});
        Run(solver, world, 120);
    });
}

TEST_CASE_FIXTURE(OnDevice, "the same scene steps to bit-identical state twice") {
    CheckReplay(context, [&](World &world) {
        TumblingBox(world);
        Run(solver, world, 90);
    });
}

TEST_CASE_FIXTURE(OneWorld, "frictionless contact conserves motion across the plane and spin about it") {
    // A frictionless plane contact pushes only along its normal, so its Jacobian has no component across the surface and none about the normal axis.
    // Dropped level, so it never tips.
    const auto box = DropBox(world, 1.4f, true, 0); // frictionless, the property under test
    constexpr float3 Drift{0.7f, 0, -0.3f};
    constexpr float Spin = 1.1f;
    world.Velocities[box] = {.Linear = Drift, .Angular = {0, Spin, 0}};

    Run(solver, world, 200); // land, then slide and spin for another two seconds
    const auto &velocity = world.Velocities[box];
    // A part in ten thousand rather than exact, since velocity is differenced from two nearby positions.
    // A float32 position three metres out resolves to 2e-7 against a step of 3e-3.
    CHECK(velocity.Linear.x == doctest::Approx(Drift.x).epsilon(1e-3));
    CHECK(velocity.Linear.z == doctest::Approx(Drift.z).epsilon(1e-3));
    CHECK(velocity.Angular.y == doctest::Approx(Spin).epsilon(1e-3));

    CheckResting(world.Poses[box].Position.y);
    CHECK(std::abs(velocity.Linear.y) < 1e-3f);
    CHECK(std::abs(velocity.Angular.x) < 1e-4f);
    CHECK(std::abs(velocity.Angular.z) < 1e-4f);
}

TEST_CASE_FIXTURE(OneWorld, "a tumbling box settles onto the plane instead of running away") {
    // There is no closed form, so the check is the invariant: with friction it comes to rest on some face.
    const auto box = TumblingBox(world);

    Run(solver, world, 300);
    CheckResting(world.Poses[box].Position.y);
    CHECK(simd::length(world.Velocities[box].Linear) < 1e-2f);
    CHECK(simd::length(world.Velocities[box].Angular) < 1e-2f);
}

TEST_CASE_FIXTURE(OneWorld, "static friction holds a box on a slope inside the cone and lets it go outside") {
    // Coulomb: a body stays put while tan(slope) < mu whatever the mass, and past that it accelerates at g (sin - mu cos).
    // Both sides are checked, since a solver that sticks everything passes the first alone.
    constexpr float Mu = 0.5f;
    const float cone = std::atan(Mu);

    SUBCASE("inside the cone it does not creep") {
        const auto settings = Tilted(cone * 0.6f);
        const auto box = DropBox(world, Half, true, Mu);

        Run(solver, world, 30, settings);
        const float from = world.Poses[box].Position.x;
        Run(solver, world, 240, settings); // four seconds is long enough for a creep to show
        CHECK(std::abs(world.Poses[box].Position.x - from) < 1e-3f);
        CHECK(simd::length(world.Velocities[box].Linear) < 1e-3f);
        CheckResting(world.Poses[box].Position.y);
    }

    SUBCASE("outside it slides at the closed form's acceleration") {
        // Past the 26.6 degrees the cone allows and short of the 45 at which a cube tips instead.
        constexpr float Slope = 0.6f;
        const auto settings = Tilted(Slope);
        const float expected = Gravity * (std::sin(Slope) - Mu * std::cos(Slope));
        const auto box = DropBox(world, Half, true, Mu);

        Run(solver, world, 30, settings); // it is already sliding by the end of this window
        CHECK(AccelerationX(solver, world, box, 60, settings) == doctest::Approx(expected).epsilon(0.05));
        CHECK(simd::length(world.Velocities[box].Angular) < 1e-2f); // sliding flat rather than tumbling
    }
}

TEST_CASE_FIXTURE(OneWorld, "a sliding box decelerates at mu g and stops where the closed form says") {
    // Coulomb friction decelerates a sliding box at mu g whatever its mass, over v0^2 / (2 mu g).
    constexpr float Mu = 0.5f, Speed = 2;
    const auto box = DropBox(world, Half, true, Mu);

    Run(solver, world, 30); // reach the plane first, so nothing is bouncing
    const float from = world.Poses[box].Position.x;
    world.Velocities[box].Linear = {Speed, 0, 0};

    Run(solver, world, 120); // twice v0 / (mu g), so it has stopped and stays stopped
    CHECK(world.Poses[box].Position.x - from == doctest::Approx(Speed * Speed / (2 * Mu * Gravity)).epsilon(0.05));
    CHECK(std::abs(world.Velocities[box].Linear.x) < 1e-2f);

    CheckResting(world.Poses[box].Position.y);
    CHECK(std::abs(world.Poses[box].Position.z - 0) < 1e-3f);
    CHECK(simd::length(world.Velocities[box].Angular) < 1e-2f);
}

TEST_CASE_FIXTURE(OneWorld, "a fast box does not tunnel through the plane") {
    constexpr float Speed = 40; // far enough per step to cross the plane outright
    const auto box = DropBox(world, 3, true);
    world.Velocities[box].Linear = {0, -Speed, 0};

    float lowest = 1e9f;
    for (uint32_t step = 0; step < 120; ++step) {
        solver.Step(world);
        lowest = std::min(lowest, world.Poses[box].Position.y);
    }
    // Contacts are found at the pose the step began from, so a step's motion bounds how deep it gets.
    CHECK(lowest > Half - Speed * StepSettings{}.DeltaTime);
    CHECK(lowest > -Half); // nowhere near through to the other side
    CheckResting(world.Poses[box].Position.y);
}

TEST_CASE_FIXTURE(OneWorld, "a full run gives its shallowest place to a deeper contact and no other") {
    // The eviction rule keeps load-bearing geometry from depending on which partner was numbered first.
    // The scene fills a run exactly, then adds a support deeper than anything in it, which takes a place.
    // An eleventh hoverer ties with the ten and is refused.
    constexpr Shape Slab{.HalfExtents = {3, Half, 3}, .Kind = ShapeBox};
    constexpr float Gap = 1e-3f; // inside a step's reach, so the contact exists, and clear of the floor
    REQUIRE(ContactsPerBody == 40); // the hoverers below are laid out to fill exactly this
    REQUIRE(ManifoldPoints == 4);

    const auto slab = world.AddShape(Slab), block = world.AddShape(UnitBox);
    // The slab is dynamic and lowest, so it owns every pair, collided in the order they were added.
    const auto subject = world.AddBody({.Pose = At(float3{0, 0, 0}), .Shape = slab});
    std::vector<Index> hovering;
    for (const float x : {-2.4f, -1.2f, 0.f, 1.2f, 2.4f})
        for (const float z : {-1.5f, 1.5f})
            hovering.push_back(world.AddBody({.Pose = At(float3{x, 2 * Half + Gap, z}), .Shape = block, .Density = 0}));
    REQUIRE(hovering.size() * ManifoldPoints == ContactsPerBody);
    const auto support = world.AddBody({.Pose = At(float3{0, -2 * Half, 0}), .Shape = slab, .Density = 0});
    const auto late = world.AddBody({.Pose = At(float3{0, 2 * Half + Gap, 0}), .Shape = block, .Density = 0});

    const auto against = [&](Index other) {
        uint32_t held = 0;
        for (const Contact &contact : Slots(world, subject)) held += contact.Active && contact.BodyB == other ? 1 : 0;
        return held;
    };

    Run(solver, world, 1);
    CHECK(world.ContactRefusals[subject] == 2 * ManifoldPoints);
    CHECK(against(support) == ManifoldPoints);
    CHECK(against(late) == 0); // the tie goes to the contact already held
    CHECK(against(hovering.front()) == 0); // the four displaced were in the lowest slots
    for (uint32_t i = 1; i < hovering.size(); ++i) {
        CAPTURE(i);
        CHECK(against(hovering[i]) == ManifoldPoints);
    }

    // It stays that way rather than creeping through the support.
    Run(solver, world, 180);
    CHECK(against(support) == ManifoldPoints);
    CheckResting(world.Poses[subject].Position.y + Half); // its underside on the support's top face
    CHECK(simd::length(world.Velocities[subject].Linear) < 1e-3f);
}

// Hulls, each checked against a shape the engine already has.
// A cube given as eight points behaves like the Box of the same size and lands where the box pairs land.
namespace {
// A body's final position, its mean contact normal and its contact count, the values these cases compare.
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

// Where a body came to rest, checking that every point it generated took a slot.
Settled RestAfter(Solver &solver, World &world, Index body, uint32_t steps) {
    Run(solver, world, steps);
    CHECK(world.ContactRefusals[body] == 0);
    return Rest(world, body);
}
} // namespace

TEST_CASE_FIXTURE(OnDevice, "a cube given as a hull rests where the same cube given as a box does") {
    const auto drop = [&](bool as_hull) {
        World world{context};
        AddGround(world);
        const auto shape = as_hull ? world.AddHull(CubeCorners(2 * Half)) : world.AddShape(UnitBox);
        REQUIRE(shape != NoIndex);
        const auto body = world.AddBody({.Pose = At(float3{0, 2, 0}), .Shape = shape});
        return RestAfter(solver, world, body, 180);
    };

    const Settled box = drop(false), hull = drop(true);
    CheckResting(hull.Position.y);
    CHECK(hull.Position.y == doctest::Approx(box.Position.y).epsilon(1e-5));
    CHECK(hull.Contacts == box.Contacts);
    CHECK(hull.Contacts == 4); // its bottom face, one contact a corner
    CHECK(hull.Normal.y == doctest::Approx(1).epsilon(1e-5));
}

TEST_CASE_FIXTURE(OneWorld, "a hull's frame finds the geometry as given after the body has moved") {
    // A body's pose is the frame the cook chose rather than the points supplied.
    // The frame maps a caller's own points back, the transform a renderer applies.
    AddGround(world);
    // A brick rather than a cube, so its principal axes are its own, left off the origin and turned.
    const float4 turn = QuatFromRotationVector(float3{0.3f, -0.7f, 0.2f});
    const float3 half{0.25f, 0.375f, 0.5f};
    std::vector<float3> points;
    for (uint32_t corner = 0; corner < 8; ++corner)
        points.push_back(float3{7, -3, 11} + Rotate(turn, half * CornerSign(corner)));

    Pose frame{};
    const auto shape = world.AddHull(points, &frame);
    REQUIRE(shape != NoIndex);
    const auto body = world.AddBody({.Pose = At(float3{0, 2, 0}), .Shape = shape});
    Run(solver, world, 240);

    const Pose pose = world.Poses[body];
    std::vector<float3> collided;
    for (const float3 vertex : world.ShapeVertices.All().subspan(world.Shapes[shape].FirstVertex, world.Shapes[shape].VertexCount))
        collided.push_back(WorldPoint(pose, vertex));
    float lowest = INFINITY;
    for (const float3 point : points) {
        const float3 in_shape = LocalPoint(frame, point);
        const float3 in_world = WorldPoint(pose, in_shape);
        lowest = std::min(lowest, float(in_world.y));
        CHECK(NearestTo(in_world, collided) < 1e-4f); // the frame is exactly the transform the cook applied
    }
    CheckResting(lowest + Half); // its lowest corner rests on the plane
}

TEST_CASE_FIXTURE(OnDevice, "where a hull rests is a property of the solid, not of the frame it arrived in") {
    // A hull's faces are recovered in the cooked frame and keyed by cooked vertex indices, so a caller's frame leaking into either shows here first.
    // The oracle is an identity: one solid gives one result.
    constexpr float Table = 0.5f;
    const std::vector<float3> shape = WedgePoints();

    const auto drop = [&](float4 turn, float3 move, bool on_a_box = true) {
        World world{context};
        AddGround(world);
        // A box to land on, so the manifold goes through the face recovery rather than the plane path.
        constexpr Shape Top{.HalfExtents = {2, Table, 2}, .Kind = ShapeBox};
        if (on_a_box)
            world.AddBody({.Pose = At(float3{0, Table, 0}), .Shape = world.AddShape(Top), .Density = 0});
        std::vector<float3> points;
        for (const float3 point : shape) points.push_back(move + Rotate(turn, point));
        Pose frame{};
        const auto hull = world.AddHull(points, &frame);
        REQUIRE(hull != NoIndex);
        // Placed through the frame the cook chose, so the caller's own points land in the same world positions in both runs.
        // A principal axis is defined only up to its sign.
        const float3 lift{0, 2 * Table + 0.15f + 0.02f, 0}; // its base is 0.15 below the origin of `shape`
        const auto body = world.AddBody({.Pose = At(lift - Rotate(QuatConjugate(turn), move - frame.Position), QuatMul(QuatConjugate(turn), frame.Orientation)), .Shape = hull});
        {
            const Pose start = world.Poses[body];
            for (uint32_t i = 0; i < points.size(); ++i) {
                const float3 in_shape = LocalPoint(frame, points[i]);
                const float3 at = WorldPoint(start, in_shape);
                CAPTURE(i);
                CHECK(simd::distance(at, shape[i] + lift) < 1e-4f);
            }
        }
        Run(solver, world, 300);
        CHECK(world.ContactRefusals[body] == 0);

        // Where each of the caller's own points ended up, taken back through the frame the cook chose.
        const Pose pose = world.Poses[body];
        std::vector<float> heights;
        for (const float3 point : points) {
            const float3 in_shape = LocalPoint(frame, point);
            heights.push_back(float(WorldPoint(pose, in_shape).y));
        }
        std::ranges::sort(heights); // the cook keeps the corners in its own order, and the solid has none
        return std::pair{heights, Rest(world, body)};
    };

    const float4 spin = QuatFromRotationVector(float3{0.8f, -1.3f, 0.55f});
    const auto [square, square_rest] = drop(float4{0, 0, 0, 1}, float3{0, 0, 0});
    // Thirty units out, where the cook's tolerances are no longer finer than the precision of the points.
    const auto [turned, turned_rest] = drop(spin, float3{31, -17, 8});

    // Every corner at the same height in both runs, since it is one solid however it was supplied.
    REQUIRE(square.size() == turned.size());
    for (uint32_t i = 0; i < square.size(); ++i) {
        CAPTURE(i);
        CHECK(square[i] == doctest::Approx(turned[i]).epsilon(2e-3).scale(0));
    }
    CHECK(square_rest.Contacts == turned_rest.Contacts);
    CHECK(square_rest.Contacts >= 3); // it rests on a face rather than teetering
    CHECK(square[0] == doctest::Approx(2 * Table).epsilon(1e-3).scale(0)); // its lowest corner on the box's top
    CHECK(square_rest.Normal.y == doctest::Approx(1).epsilon(1e-3));
}

TEST_CASE_FIXTURE(OnDevice, "a hull and a box rest on each other the same way round either way") {
    // The geometry determines which shape presents the reference face, rather than which is body A.
    const auto stack = [&](bool hull_on_top) {
        World world{context};
        AddGround(world);
        const auto hull = world.AddHull(CubeCorners(2 * Half));
        const auto box = world.AddShape(UnitBox);
        REQUIRE(hull != NoIndex);
        world.AddBody({.Pose = At(float3{0, Half, 0}), .Shape = hull_on_top ? box : hull, .Density = 0});
        const auto top = world.AddBody({.Pose = At(float3{0, 1.4f, 0}), .Shape = hull_on_top ? hull : box});
        return RestAfter(solver, world, top, 240);
    };

    const Settled hull_up = stack(true), box_up = stack(false);
    for (const Settled &settled : {hull_up, box_up}) {
        // One box up from the static one below, which is where a box on a box already lands.
        CHECK(std::abs(settled.Position.y - (2 * Half + Half - StepSettings{}.ContactMargin)) < 2e-3f);
        CHECK(std::abs(settled.Position.x) < 1e-3f);
        CHECK(settled.Contacts == 4);
        CHECK(settled.Normal.y == doctest::Approx(1).epsilon(1e-4));
    }
    CHECK(hull_up.Position.y == doctest::Approx(box_up.Position.y).epsilon(1e-4));
}

TEST_CASE_FIXTURE(OneWorld, "a tetrahedron settles on the face it was dropped on") {
    // A hull that is not a box, whose resting height is a property of its own geometry.
    AddGround(world);
    // Three-fold symmetry about y leaves the cook no rotation to find, so it comes back on a face.
    const float side = std::sqrt(3.f) / 2;
    const std::vector<float3> points{float3{1, 0, 0}, float3{-0.5f, 0, side}, float3{-0.5f, 0, -side}, float3{0, 1.6f, 0}};
    const auto shape = world.AddHull(points);
    REQUIRE(shape != NoIndex);
    const auto body = world.AddBody({.Pose = At(float3{0, 1.5f, 0}), .Shape = shape});

    // A tetrahedron's centre of mass is the mean of its corners, a quarter of the way up from the base.
    // The cook leaves the base a quarter of the height below the origin, which is the rest height.
    const Shape &cooked = world.Shapes[shape];
    REQUIRE(cooked.VertexCount == 4);
    float lowest = 0;
    for (uint32_t i = 0; i < cooked.VertexCount; ++i) lowest = std::min(lowest, world.ShapeVertices[cooked.FirstVertex + i].y);
    CHECK(lowest == doctest::Approx(-1.6f / 4).epsilon(1e-4));

    Run(solver, world, 300);
    const Settled settled = Rest(world, body);
    CHECK(settled.Position.y == doctest::Approx(1.6f / 4).epsilon(2e-3));
    CHECK(settled.Contacts == 3); // its base, one contact a corner
    CHECK(settled.Normal.y == doctest::Approx(1).epsilon(1e-4));
    CHECK(simd::length(world.Velocities[body].Linear) < 1e-2f);
    CHECK(world.ContactRefusals[body] == 0);
}

TEST_CASE_FIXTURE(OnDevice, "a sphere and a capsule rest on a hull as they rest on a box") {
    // A round shape against a hull goes through the convex path, with its core a one or two point polytope and the radius subtracted from the result.
    const auto drop = [&](bool on_hull, const Shape &shape, float height, float4 turn) {
        World world{context};
        AddGround(world);
        const auto floor = on_hull ? world.AddHull(CubeCorners(2)) : world.AddShape({.HalfExtents = {1, 1, 1}, .Kind = ShapeBox});
        REQUIRE(floor != NoIndex);
        world.AddBody({.Pose = At(float3{0, 1, 0}), .Shape = floor, .Density = 0});
        const auto body = world.AddBody({.Pose = At(float3{0, height, 0}, turn), .Shape = world.AddShape(shape)});
        return RestAfter(solver, world, body, 240);
    };

    SUBCASE("a sphere") {
        constexpr Shape Ball{.Radius = 0.25f, .Kind = ShapeSphere};
        const Settled on_hull = drop(true, Ball, 2.6f, float4{0, 0, 0, 1}), on_box = drop(false, Ball, 2.6f, float4{0, 0, 0, 1});
        CHECK(on_hull.Contacts == 1);
        CHECK(on_hull.Position.y == doctest::Approx(on_box.Position.y).epsilon(1e-4));
        CHECK(on_hull.Position.y == doctest::Approx(2.25f).epsilon(1e-3));
        CHECK(on_hull.Normal.y == doctest::Approx(1).epsilon(1e-4));
    }
    SUBCASE("a capsule across it") {
        constexpr Shape Pill{.HalfExtents = {0, 0.4f, 0}, .Radius = 0.15f, .Kind = ShapeCapsule};
        const float quarter = 3.14159265f / 2; // onto its side, so its length lies across the face
        const float4 turn = QuatFromRotationVector(float3{0, 0, quarter});
        const Settled on_hull = drop(true, Pill, 2.4f, turn), on_box = drop(false, Pill, 2.4f, turn);
        CHECK(on_hull.Contacts == 2); // the two ends of the stretch of its core lying over the face
        CHECK(on_box.Contacts == 2);
        CHECK(on_hull.Position.y == doctest::Approx(on_box.Position.y).epsilon(1e-3));
        CHECK(on_hull.Position.y == doctest::Approx(2.15f).epsilon(2e-3));
    }
}

TEST_CASE_FIXTURE(OneWorld, "two boxes apart and crossed take their edge pair, not a face") {
    // The separating axis test biases towards the reference face, so the edge pair has to beat the best face by a clear margin.
    // That bias compares negative separations while the boxes overlap and positive ones across a gap, which is different arithmetic.
    // This is the across-a-gap case.
    constexpr float Diagonal = 0.70710678f; // half a unit square's diagonal, the tilted box's ridge
    const float tilt = std::numbers::pi_v<float> / 4; // onto an edge, short of the quarter turn back onto a face
    const auto shape = world.AddShape(UnitBox);
    world.AddBody({.Pose = At(float3{0, 0, 0}, QuatFromRotationVector(float3{0, 0, tilt})), .Shape = shape, .Density = 0});
    // Set down a clear gap above and moving fast enough that the reach covers it while still apart.
    const auto box = world.AddBody({.Pose = At(float3{0, 2 * Diagonal + 0.08f, 0}, QuatFromRotationVector(float3{tilt, 0, 0})), .Velocity = {.Linear = {0, -3, 0}}, .Shape = shape});

    // Read from C0, since the axis is chosen at the pose the step began from, so a C0 above the margin is exactly a contact formed while the two were apart.
    bool apart = false;
    for (uint32_t step = 0; step < 4; ++step) {
        solver.Step(world);
        CheckManifolds(world);
        uint32_t live = 0;
        for (const Contact &contact : Slots(world, box)) {
            if (!contact.Active) continue;
            ++live;
            if (contact.C0[0] <= StepSettings{}.ContactMargin) continue;
            apart = true;
            CAPTURE(step);
            CHECK(contact.Normal.y == doctest::Approx(1).epsilon(1e-3));
            CHECK(std::abs(contact.Normal.x) < 1e-3f); // not the forty-five degrees a face would give
            CHECK(std::abs(contact.Normal.z) < 1e-3f);
        }
        if (apart) CHECK(live == 1); // two crossed edges meet in one place
    }
    CHECK(apart); // at least one step held a contact across a gap
}

TEST_CASE_FIXTURE(OnDevice, "two hulls meeting on their edges hold each other up") {
    // ConvexManifold's closest-pair branch, which runs when neither shape presents a face along the contact normal.
    // A defect here costs a wrong contact direction rather than a small position error.
    // A single crossing point is the whole contact, so the checks are the position, the normal, and staying outside the cube below.
    constexpr float Side = 1, Diagonal = Side * 0.70710678f; // half a square's diagonal, the ridge height

    // The tilt is carried by the pose, since a cube's principal axes are degenerate and the cook applies no rotation.
    const auto turn = [](float3 axis, float angle) { return QuatFromRotationVector(simd::normalize(axis) * angle); };
    const float quarter = std::numbers::pi_v<float> / 4;

    // Read while it is still on the ridge, since balanced on one point it topples and a long run would measure the aftermath rather than the contact.
    const auto meet = [&](float4 upper, float reach, uint32_t steps) {
        World world{context};
        const auto shape = world.AddHull(CubeCorners(Side));
        REQUIRE(shape != NoIndex);
        // A floor for whatever slides off, exactly where the tilted cube below stands on it.
        world.AddBody({.Shape = world.AddShape({.Normal = {0, 1, 0}, .Offset = -Diagonal, .Kind = ShapePlane})});
        // Tilted a quarter turn about z, so it presents a ridge upward rather than a face.
        world.AddBody({.Pose = At(float3{0, 0, 0}, turn(float3{0, 0, 1}, quarter)), .Shape = shape, .Density = 0});
        const auto body = world.AddBody({.Pose = At(float3{0, Diagonal + reach, 0}, upper), .Shape = shape});
        Run(solver, world, steps);
        return std::pair{Rest(world, body), simd::length(world.Velocities[body].Linear)};
    };

    SUBCASE("edge across edge") {
        // The upper cube's ridge runs along x, crossing the one below at a right angle.
        const auto [settled, speed] = meet(turn(float3{1, 0, 0}, quarter), Diagonal, 30);
        CHECK(settled.Contacts == 1); // two crossed edges meet in exactly one place
        CHECK(settled.Normal.y == doctest::Approx(1).epsilon(1e-3));
        // Each cube reaches half a diagonal to its ridge, so two of those apart when the ridges touch.
        CHECK(settled.Position.y == doctest::Approx(2 * Diagonal).epsilon(2e-3));
        CHECK(std::abs(settled.Position.x) < 5e-3f);
        CHECK(std::abs(settled.Position.z) < 5e-3f);
    }

    SUBCASE("corner over edge, across the gap") {
        // A vertex against a ridge is the same branch.
        // Measured while still apart, where the result is well defined, since a vertex on a ridge sits between two faces and the normal is ambiguous.
        const float3 diagonal = simd::normalize(float3{1, 1, 1});
        const float3 down{0, -1, 0};
        const float4 corner_down = turn(simd::cross(diagonal, down), std::acos(simd::dot(diagonal, down)));
        const auto [settled, speed] = meet(corner_down, Side * 0.8660254f + 2e-3f, 1);
        CHECK(settled.Contacts == 1);
        CHECK(settled.Normal.y == doctest::Approx(1).epsilon(1e-4));
        CHECK(std::abs(settled.Normal.x) < 1e-4f);
        CHECK(std::abs(settled.Normal.z) < 1e-4f);
    }

    SUBCASE("and a corner that lands on a ridge slides off one side of it") {
        // Once inside, the vertex is behind one of the two faces meeting at the ridge, so the contact takes that face's 45 degree normal and it slides off.
        // The defect would be a normal pointing into the solid.
        const float3 diagonal = simd::normalize(float3{1, 1, 1});
        const float3 down{0, -1, 0};
        const float4 corner_down = turn(simd::cross(diagonal, down), std::acos(simd::dot(diagonal, down)));
        const auto [settled, speed] = meet(corner_down, Side * 0.8660254f, 200);
        CHECK(std::abs(settled.Position.x) > 0.1f); // it leaves the ridge on one side
        // The floor is a diagonal below the origin, so a cube flat on it sits half a side up and one still on an edge half a diagonal.
        // It ends between the two.
        CHECK(settled.Position.y > -Diagonal + Half - MaxPenetration);
        CHECK(settled.Position.y < -Diagonal + Diagonal + MaxPenetration);
        CHECK(speed < 0.1f);
    }

    SUBCASE("and it never ends up inside what it was resting on") {
        // Toppling is the expected physics, and the defect would be a normal pulling it through the cube it falls off.
        const auto [settled, speed] = meet(turn(float3{1, 0, 0}, quarter), Diagonal + 0.05f, 400);
        CHECK(settled.Position.y > -Diagonal); // below this it is inside the static cube, or past it
        CHECK(std::isfinite(settled.Position.x));
        CHECK(simd::length(settled.Position) < 20); // it falls off rather than being thrown
    }
}

namespace {
// The lowest point of a body's geometry, which for a hull on a plane is the point resting on the plane.
float LowestVertex(const World &world, Index body) {
    const Shape &shape = world.Shapes[world.BodyShapes[body]];
    const Pose pose = world.Poses[body];
    float lowest = INFINITY;
    for (uint32_t i = 0; i < shape.VertexCount; ++i)
        lowest = std::min(lowest, float(WorldPoint(pose, world.ShapeVertices[shape.FirstVertex + i]).y));
    return lowest;
}
} // namespace

TEST_CASE_FIXTURE(OnDevice, "a sphere-like hull rests on the face it lands on") {
    // The opposite extreme from the boxy hulls above: eighty faces whose normals are a few degrees apart, the finest the 1e-3 face tolerance handles.
    // Wherever it rests, its lowest vertex is on the plane.
    constexpr float Radius = 0.5f;
    // A sphere's principal axes are degenerate, so the cook chooses the orientation, and as cooked it balances on a single vertex.
    float3 tilt{0, 0, 0};
    uint32_t touching = 1;
    SUBCASE("as cooked, it comes down on one vertex and balances on it") {
        tilt = float3{0, 0, 0};
        touching = 1;
    }
    SUBCASE("turned, it settles onto a whole face") {
        tilt = float3{0.4f, 0.2f, 0.3f};
        touching = 3;
    }
    World world{context};
    AddGround(world);
    const auto shape = world.AddHull(SpherePoints(Radius));
    REQUIRE(shape != NoIndex);
    REQUIRE(world.Shapes[shape].VertexCount == 42); // every point is a corner, with none inside the hull
    const auto body = world.AddBody({.Pose = At(float3{0, 1.5f, 0}, QuatFromRotationVector(tilt)), .Shape = shape});

    Run(solver, world, 300);
    const Settled settled = Rest(world, body);
    CHECK(world.ContactRefusals[body] == 0);
    // Recovering too wide a face would put rows under vertices that are off the plane.
    uint32_t on_plane = 0;
    const Shape &geometry = world.Shapes[world.BodyShapes[body]];
    const Pose pose = world.Poses[body];
    const float lowest = LowestVertex(world, body);
    for (uint32_t i = 0; i < geometry.VertexCount; ++i)
        if (float(WorldPoint(pose, world.ShapeVertices[geometry.FirstVertex + i]).y) < lowest + 1e-3f)
            ++on_plane;
    CHECK(on_plane == touching);
    CHECK(settled.Contacts == touching);
    CHECK(settled.Normal.y == doctest::Approx(1).epsilon(1e-3));
    CHECK(lowest < 0);
    CHECK(lowest > -MaxPenetration);
    CHECK(simd::length(world.Velocities[body].Linear) < 1e-3f);
    CHECK(simd::length(world.Velocities[body].Angular) < 1e-3f);

    // It stays there rather than creeping off the face it found.
    const float3 was = world.Poses[body].Position;
    Run(solver, world, 300);
    CHECK(simd::distance(world.Poses[body].Position, was) < 1e-3f);
}

TEST_CASE_FIXTURE(OneWorld, "a hull the cook had to simplify still holds a body up") {
    // Five hundred points against a limit of sixty-four corners, so the body lands on the cook's simplified hull.
    // The basic requirement, and the reason for simplifying rather than refusing.
    constexpr float Radius = 0.5f;
    AddGround(world);
    const auto shape = world.AddHull(DenseSpherePoints(500, Radius));
    REQUIRE(shape != NoIndex);
    REQUIRE(world.Shapes[shape].VertexCount <= MaxHullVertices);
    REQUIRE(world.Shapes[shape].FaceCount >= 1);
    // Turned, since a sphere's principal axes are degenerate and as cooked it can land on a corner.
    const auto body = world.AddBody({.Pose = At(float3{0, 1.5f, 0}, QuatFromRotationVector(float3{0.4f, 0.2f, 0.3f})), .Shape = shape});

    Run(solver, world, 300);
    const Settled settled = Rest(world, body);
    CHECK(world.ContactRefusals[body] == 0);
    CHECK(settled.Contacts >= 1);
    CHECK(settled.Normal.y == doctest::Approx(1).epsilon(1e-3));
    const float lowest = LowestVertex(world, body);
    CHECK(lowest < 0);
    CHECK(lowest > -MaxPenetration);
    CHECK(simd::length(world.Velocities[body].Linear) < 1e-3f);

    // It stays where it stopped rather than rolling on.
    const float3 was = world.Poses[body].Position;
    Run(solver, world, 300);
    CHECK(simd::distance(world.Poses[body].Position, was) < 1e-3f);
}

TEST_CASE_FIXTURE(OneWorld, "a face wider than eight points still holds a coin level") {
    // MaxFacePoints is eight and both routes into a manifold cap there, so a sixteen-sided coin lying flat exercises both caps.
    // The requirement is that whatever four points survive still support the coin's centre of mass.
    constexpr float Radius = 0.5f, HalfHeight = 0.1f;
    bool on_a_box = false;
    SUBCASE("on a plane, where the cap is the plane path's") { on_a_box = false; }
    SUBCASE("on a box, where it is the recovered face's") { on_a_box = true; }

    AddGround(world);
    float table = 0;
    if (on_a_box) {
        constexpr Shape Table{.HalfExtents = {2, Half, 2}, .Kind = ShapeBox};
        world.AddBody({.Pose = At(float3{0, Half, 0}), .Shape = world.AddShape(Table), .Density = 0});
        table = 2 * Half;
    }
    const auto shape = world.AddHull(PrismPoints(16, Radius, HalfHeight));
    REQUIRE(shape != NoIndex);
    REQUIRE(world.Shapes[shape].VertexCount == 32);
    const auto coin = world.AddBody({.Pose = At(float3{0, table + 1, 0}), .Shape = shape, .Friction = 0.5f});

    Run(solver, world, 300);
    const auto &pose = world.Poses[coin];
    CHECK(pose.Position.y == doctest::Approx(table + HalfHeight).epsilon(0.02).scale(0));
    CHECK(simd::length(RotationVector(pose.Orientation)) < 1e-2f); // flat rather than tipped onto its rim
    CHECK(simd::length(world.Velocities[coin].Linear) < 1e-3f);
    CHECK(simd::length(world.Velocities[coin].Angular) < 1e-3f);

    // Still flat three hundred steps later, where a coin resting on half its face would lean further every step.
    const Pose was = pose;
    Run(solver, world, 300);
    CHECK(simd::distance(world.Poses[coin].Position, was.Position) < 1e-3f);
    CHECK(simd::length(RotationVector(world.Poses[coin].Orientation)) < 1e-2f);
}

TEST_CASE_FIXTURE(OneWorld, "a manifold reduced to four keeps the same four as the body turns") {
    // Reduction picks the four with the most area between them, by geometry rather than by generation order.
    // A resting contact keeps the same four and their duals.
    // A spinning octagon leaves all eight at one depth, so the tie-break alone selects them.
    constexpr float Radius = 0.5f, HalfHeight = 0.25f;
    AddGround(world);
    const auto shape = world.AddHull(PrismPoints(8, Radius, HalfHeight));
    REQUIRE(shape != NoIndex);
    const auto prism = world.AddBody({.Pose = At(float3{0, HalfHeight + 0.2f, 0}), .Shape = shape, .Friction = 0});

    // Sleeping would end the experiment, since a body at rest stops being collided.
    const StepSettings spinning{.SleepSteps = ~0u};
    Run(solver, world, 120, spinning);
    world.Velocities[prism].Angular = {0, 0.6f, 0}; // a third of a turn a second, which is slow

    const auto names = [&] {
        std::set<uint32_t> live;
        for (const Contact &contact : Slots(world, prism)) {
            if (contact.Active) live.insert(contact.Feature);
        }
        return live;
    };
    solver.Step(world, spinning);
    const auto four = names();
    REQUIRE(four.size() == 4); // eight vertices on the plane, and the four with the most area are kept

    float turned = 0;
    for (uint32_t step = 0; step < 300; ++step) {
        solver.Step(world, spinning);
        CheckManifolds(world);
        turned += float(world.Velocities[prism].Angular.y) * spinning.DeltaTime;
        CAPTURE(step);
        REQUIRE(names() == four);
    }
    CHECK(turned > 2); // at least a third of a turn, so the four have travelled
    CHECK(world.Poses[prism].Position.y == doctest::Approx(HalfHeight).epsilon(0.02).scale(0));
    CHECK(simd::length(RotationVector(QuatMul(world.Poses[prism].Orientation, QuatConjugate(QuatFromRotationVector(float3{0, turned, 0}))))) < 5e-2f);
}

TEST_CASE_FIXTURE(OneWorld, "two coins stacked hold each other however they are twisted") {
    // Clipping one face into another gains a vertex per cut, so an eight point face against an eight edge face is a sixteen-gon.
    // Bounding that clip at MaxFacePoints drops a contiguous run of the perimeter, which then fails to support the body's centre.
    // Eight sides and a twist are needed to reach that case.
    constexpr float Radius = 0.5f, HalfHeight = 0.15f, Twist = 0.3927f; // half of an octagon's step
    uint32_t sides = 8;
    SUBCASE("four sides, where the clip has room to spare") { sides = 4; }
    SUBCASE("six sides, which fits by two") { sides = 6; }
    SUBCASE("eight sides, which is where it overflowed") { sides = 8; }

    AddGround(world);
    const auto shape = world.AddHull(PrismPoints(sides, Radius, HalfHeight));
    REQUIRE(shape != NoIndex);
    const float4 turn = QuatFromRotationVector(float3{0, Twist, 0});
    const auto lower = world.AddBody({.Pose = At(float3{0, HalfHeight, 0}), .Shape = shape});
    const auto upper = world.AddBody({.Pose = At(float3{0, 3 * HalfHeight + 0.01f, 0}, turn), .Shape = shape});

    Run(solver, world, 240);
    // Neither has turned out of the pose it was set down in, and a coin sliding off shows first as a tilt.
    CHECK(world.Poses[lower].Position.y == doctest::Approx(HalfHeight).epsilon(0.01).scale(0));
    CHECK(world.Poses[upper].Position.y == doctest::Approx(3 * HalfHeight).epsilon(0.01).scale(0));
    CHECK(simd::length(RotationVector(world.Poses[lower].Orientation)) < 1e-3f);
    CHECK(simd::length(RotationVector(QuatMul(world.Poses[upper].Orientation, QuatConjugate(turn)))) < 1e-3f);
    CHECK(simd::length(world.Velocities[upper].Linear) < 1e-3f);

    // Both pairs belong to the lower coin, the lower-indexed of the two dynamic bodies.
    uint32_t against_plane = 0, against_coin = 0;
    for (const Contact &contact : Slots(world, lower)) {
        if (!contact.Active) continue;
        (world.Shapes[world.BodyShapes[contact.BodyB]].Kind == ShapePlane ? against_plane : against_coin) += 1;
    }
    CHECK(against_plane == ManifoldPoints);
    CHECK(against_coin == ManifoldPoints);
    CHECK(world.ContactRefusals[lower] == 0);
    CHECK(world.ContactRefusals[upper] == 0);
}

TEST_CASE_FIXTURE(OnDevice, "a crowned plate rests on its crown and a chamfered one on its face") {
    // Both plates have a face and, beside it, a facet meeting it at an angle no height tolerance distinguishes from flat.
    // A height-gathered face spans both and fits a plane through neither.
    constexpr float Thick = 0.1f;
    // On a table rather than a plane, since a plane needs no face recovered.
    // `lift` is how far above the table's top the plate starts.
    const auto on_a_table = [&](std::span<const float3> points, float lift) {
        World world{context};
        AddGround(world);
        constexpr Shape Table{.HalfExtents = {2, 0.5f, 2}, .Kind = ShapeBox};
        world.AddBody({.Pose = At(float3{0, 0.5f, 0}), .Shape = world.AddShape(Table), .Density = 0});
        const auto shape = world.AddHull(points);
        REQUIRE(shape != NoIndex);
        const auto plate = world.AddBody({.Pose = At(float3{0, 1 + lift, 0}), .Shape = shape});
        return std::tuple{std::move(world), plate};
    };
    const auto rest = [&](std::span<const float3> points) {
        auto [world, plate] = on_a_table(points, Thick + 0.01f);
        Run(solver, world, 300);
        CHECK(simd::length(world.Velocities[plate].Linear) < 1e-3f);
        // How far its lowest geometry reaches past the table's top, which is the contact margin.
        return std::pair{1 - LowestVertex(world, plate), Rest(world, plate)};
    };

    SUBCASE("a facet a third of a degree off the face it sits beside") {
        // A chamfer rising inside any tolerance a face could be recovered with, and the plate still rests on the face rather than the facet.
        const auto [depth, settled] = rest(ChamferedPlate(0.1f * std::tan(0.3f * std::numbers::pi_v<float> / 180)));
        CHECK(depth == doctest::Approx(StepSettings{}.ContactMargin).epsilon(0.05).scale(0));
        CHECK(settled.Contacts == ManifoldPoints); // the four corners of the face, and none of the facet
    }

    SUBCASE("a bottom crowned by a millimetre and a quarter over a metre") {
        // Eight facets across the bottom, several inside a face tolerance at once.
        // Set down level and touching, so the only question is which geometry the manifold names, and it rocks after that.
        std::vector<float3> points;
        for (const float z : {-Half, Half}) {
            for (int k = -4; k <= 4; ++k) {
                const float x = Half * float(k) / 4;
                points.push_back(float3{x, -Thick + 0.00125f * (x / Half) * (x / Half), z});
            }
            points.push_back(float3{-Half, Thick, z});
            points.push_back(float3{Half, Thick, z});
        }
        auto [world, plate] = on_a_table(points, Thick);
        Run(solver, world, 1);
        // The rim stands a millimetre and a quarter above the crown, where a height-gathered face would place the rows.
        uint32_t held = 0;
        for (const Contact &contact : Slots(world, plate)) {
            if (!contact.Active) continue;
            ++held;
            CAPTURE(float(contact.AnchorA.x));
            CHECK(std::abs(float(contact.AnchorA.x)) <= 0.125f + 1e-3f); // on the crown rather than out at the rim
        }
        CHECK(held > 0);
    }
}

TEST_CASE_FIXTURE(OneWorld, "a settled hull keeps the names of its contacts") {
    // Feature stability where there is most room to break it, since a hull's faces are recovered rather than stored.
    AddGround(world);
    const auto shape = world.AddHull(CubeCorners(1));
    REQUIRE(shape != NoIndex);
    const std::vector<Index> stack = AddStack(world, shape, 3);

    Run(solver, world, 300);
    const auto settled = ContactKeys(world);
    CHECK(settled.size() == 12); // three boxes, four corners each against the one below
    CheckKeysHold(solver, world, settled);
    for (uint32_t i = 0; i < stack.size(); ++i) {
        CAPTURE(i);
        CHECK(std::abs(world.Poses[stack[i]].Position.y - (Half + float(i) * (1 - StepSettings{}.ContactMargin))) < 2e-3f);
    }
}

// Meshes, held to the same bar as the hulls: a floor of triangles supports a body exactly where a plane supports it.
// The seams are new, artefacts of the tessellation that must generate no contact.
namespace {
// A floor of `side` by `side` quads across `2 * extent` metres, wound so it faces up.
// `slope` tilts it by moving the points and `local` by moving the shape within its body, the pair these cases compare.
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

// A plane at `side` == 0, otherwise the same surface cut into `side` quads.
// One call builds both, so the two stay in step.
Index AddFloor(World &world, uint32_t side, float slope = 0, float friction = BodyDesc{}.Friction) {
    const float3 up{-std::sin(slope), std::cos(slope), 0};
    const Index floor = side == 0 ? world.AddShape({.Normal = up, .Offset = 0, .Kind = ShapePlane}) : FloorMesh(world, side, 5, slope);
    REQUIRE(floor != NoIndex);
    world.AddBody({.Shape = floor, .Friction = friction});
    return floor;
}

// A ridge running along z: two slopes meeting at a crease the cook marks active, where the surface genuinely folds.
// Unlike a seam, a body rests on this edge and crosses it.
Index RidgeMesh(World &world, float extent, float height) {
    std::vector<float3> points;
    for (const float x : {-extent, 0.f, extent})
        for (const float z : {-extent, extent}) points.push_back(float3{x, x == 0 ? height : 0, z});
    // Wound so (B - A) x (C - A) points up out of the surface on both sides of the crease.
    const std::vector<uint32_t> indices{0, 1, 3, 0, 3, 2, 2, 3, 5, 2, 5, 4};
    return world.AddMesh(points, indices);
}
} // namespace

TEST_CASE_FIXTURE(OnDevice, "a floor of triangles holds a box exactly where a plane does") {
    // Off the grid lines on purpose, since a corner exactly on a seam is a knife edge with a test of its own.
    const auto drop = [&](uint32_t side) {
        World world{context};
        AddFloor(world, side);
        const auto box = world.AddBody({.Pose = At(float3{0.13f, 2, 0.07f}), .Shape = world.AddShape(UnitBox)});
        return RestAfter(solver, world, box, 180);
    };

    const Settled plane = drop(0), pair = drop(1), tessellated = drop(16);
    for (const Settled &settled : {pair, tessellated}) {
        CheckResting(settled.Position.y);
        CHECK(settled.Position.y == doctest::Approx(plane.Position.y).epsilon(1e-4));
        // Four rather than four per triangle: a box rests on its own corners however the floor was cut up.
        CHECK(settled.Contacts == 4);
        CHECK(settled.Normal.y == doctest::Approx(1).epsilon(1e-4));
    }
}

TEST_CASE_FIXTURE(OnDevice, "a ridge is an edge a body rests on and crosses, not one it catches on") {
    // The mesh path passes ConvexManifold the triangle's own normal rather than searching.
    // A body over a crease then gets a face contact against a face it is only half over.
    constexpr float Extent = 4, Height = 1.5f; // a crease at about 20 degrees either side

    SUBCASE("a box set down astride it rests on the crease") {
        World world{context};
        REQUIRE(RidgeMesh(world, Extent, Height) != NoIndex);
        world.AddBody({.Shape = 0});
        // Off the crease in z so it is not also sitting on the seam that splits each slope.
        const auto box = world.AddBody({.Pose = At(float3{0, Height + Half + 0.2f, 0.13f}), .Shape = world.AddShape(UnitBox)});
        Run(solver, world, 300);
        const Settled settled = Rest(world, box);
        // Its flat underside touches only the crease, so it sits one half extent above the peak.
        // It is supported only if the path searches when the triangle's own normal yields nothing.
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
        // Fast enough to carry over the crease, and started on the up slope where the surface is.
        const auto ball = world.AddBody({.Pose = At(float3{-3, Height * 0.25f + Radius, 0.11f}), .Velocity = {.Linear = {9, 0, 0}}, .Shape = world.AddShape({.Radius = Radius, .Kind = ShapeSphere})});
        float deepest = 0;
        bool crossed = false;
        for (uint32_t step = 0; step < 240; ++step) {
            solver.Step(world);
            CheckManifolds(world);
            const float3 at = world.Poses[ball].Position;
            crossed = crossed || at.x > 0.5f;
            // The slopes are planes through the crease, so the surface height is linear either side.
            const float surface = Height * std::max(0.f, 1 - std::abs(float(at.x)) / Extent);
            if (std::abs(at.x) <= Extent) deepest = std::max(deepest, surface + Radius - float(at.y));
        }
        CHECK(crossed); // it crosses the crease rather than stopping against it
        CHECK(deepest < 4 * StepSettings{}.ContactMargin); // it never enters the surface
        CHECK(world.ContactRefusals[ball] == 0);
    }
}

TEST_CASE_FIXTURE(OnDevice, "a capsule and a hull rest on a floor of triangles where they rest on a plane") {
    // A capsule and a hull reach a mesh by different paths: a core against the triangle, and ConvexManifold with the triangle supplying the direction.
    // Neither may generate a contact on a seam.
    constexpr float Radius = 0.25f, HalfLength = 0.5f;
    constexpr Shape Pill{.HalfExtents = {0, HalfLength, 0}, .Radius = Radius, .Kind = ShapeCapsule};
    const float quarter = 2 * std::atan(1.f); // the capsule's length runs along its own y, so this lays it down

    // Off the grid lines on purpose, since a body exactly on a seam is a knife edge with a test of its own.
    const auto drop = [&](bool tessellated, bool hulled, float4 orientation, float height) {
        World world{context};
        AddFloor(world, tessellated ? 8 : 0);
        const Index shape = hulled ? world.AddHull(CubeCorners(2 * Half)) : world.AddShape(Pill);
        REQUIRE(shape != NoIndex);
        const auto body = world.AddBody({.Pose = At(float3{0.13f, height, 0.07f}, orientation), .Shape = shape});
        Run(solver, world, 240);
        CHECK(world.ContactRefusals[body] == 0);
        return std::pair{Rest(world, body), simd::length(world.Velocities[body].Linear)};
    };

    SUBCASE("a capsule lying across the triangles") {
        const float4 lying = QuatFromRotationVector(float3{0, 0, quarter});
        const auto [mesh, mesh_speed] = drop(true, false, lying, 1.5f);
        const auto [plane, plane_speed] = drop(false, false, lying, 1.5f);
        // Lying down it rests one radius up and touches along the stretch its two ends bound.
        CHECK(mesh.Position.y == doctest::Approx(Radius - StepSettings{}.ContactMargin).epsilon(5e-3));
        CHECK(mesh.Position.y == doctest::Approx(plane.Position.y).epsilon(2e-3));
        CHECK(mesh.Normal.y == doctest::Approx(1).epsilon(1e-3));
        CHECK(mesh.Contacts == plane.Contacts);
        CHECK(mesh_speed < 1e-2f);
    }

    SUBCASE("a capsule stood on one end") {
        const auto [mesh, mesh_speed] = drop(true, false, float4{0, 0, 0, 1}, 1.5f);
        const auto [plane, plane_speed] = drop(false, false, float4{0, 0, 0, 1}, 1.5f);
        // On end it is a sphere to the floor, reaching half length plus radius down from the centre.
        CHECK(mesh.Position.y == doctest::Approx(HalfLength + Radius - StepSettings{}.ContactMargin).epsilon(5e-3));
        CHECK(mesh.Position.y == doctest::Approx(plane.Position.y).epsilon(2e-3));
        CHECK(mesh.Contacts == 1); // a cap touches at one point whatever it is standing on
        CHECK(mesh_speed < 1e-2f);
    }

    SUBCASE("a hull lying flat on the triangles") {
        const auto [mesh, mesh_speed] = drop(true, true, float4{0, 0, 0, 1}, 2);
        const auto [plane, plane_speed] = drop(false, true, float4{0, 0, 0, 1}, 2);
        CHECK(mesh.Position.y == doctest::Approx(plane.Position.y).epsilon(2e-3));
        CHECK(mesh.Normal.y == doctest::Approx(1).epsilon(1e-3));
        // Its own four corners, where clipping the other way round would give the tessellation's corners.
        CHECK(mesh.Contacts == 4);
        CHECK(mesh.Contacts == plane.Contacts);
        CHECK(mesh_speed < 1e-2f);
    }
}

TEST_CASE_FIXTURE(OnDevice, "a coin rests on a floor of triangles where it rests on a plane") {
    // A many-faced hull on a mesh, with every row on a vertex of the coin's own rim, at the height a plane supports it at.
    // The clip bound is never approached, since the coin lies across five triangles.
    constexpr float Radius = 0.5f, HalfHeight = 0.15f;
    const auto drop = [&](bool tessellated) {
        World world{context};
        AddFloor(world, tessellated ? 8 : 0);
        const auto shape = world.AddHull(PrismPoints(8, Radius, HalfHeight));
        REQUIRE(shape != NoIndex);
        // Off the grid lines, as in the tests around this one, since a corner on a seam is its own case.
        const auto body = world.AddBody({.Pose = At(float3{0.13f, 1.f, 0.07f}), .Shape = shape});
        Run(solver, world, 240);
        CHECK(world.ContactRefusals[body] == 0);
        CHECK(simd::length(world.Velocities[body].Linear) < 1e-3f);
        CHECK(simd::length(RotationVector(world.Poses[body].Orientation)) < 1e-3f); // flat rather than tipped onto its rim
        // Every row on a vertex of the rim, which a clip short of the face would remove.
        for (const Contact &contact : Slots(world, body)) {
            if (!contact.Active) continue;
            CHECK(std::hypot(float(contact.AnchorA.x), float(contact.AnchorA.z)) == doctest::Approx(Radius).epsilon(1e-3).scale(0));
        }
        return Rest(world, body);
    };

    const Settled mesh = drop(true), plane = drop(false);
    CHECK(mesh.Position.y == doctest::Approx(HalfHeight).epsilon(0.01).scale(0));
    CHECK(mesh.Position.y == doctest::Approx(plane.Position.y).epsilon(2e-3));
    CHECK(mesh.Normal.y == doctest::Approx(1).epsilon(1e-3));
    // The counts differ by design: a mesh gives one manifold per triangle and five of them hold all eight rim vertices.
    // A plane is one manifold reduced to four.
    CHECK(mesh.Contacts == 8);
    CHECK(plane.Contacts == ManifoldPoints);
    // It stays in place, where a manifold covering half its face would show up as drift.
    CHECK(std::abs(float(mesh.Position.x) - 0.13f) < 2e-3f);
    CHECK(std::abs(float(mesh.Position.z) - 0.07f) < 2e-3f);
}

TEST_CASE_FIXTURE(OnDevice, "a body over more triangles than one batch holds still rests on them") {
    // MaxMeshTriangles caps how many triangles a body takes from a mesh at once, and a wide body on a finely cut floor exceeds it easily.
    // The walk resumes, so the cap bounds registers rather than coverage.
    constexpr float Wide = 2;
    constexpr Shape Slab{.HalfExtents = {Wide, Half, Wide}, .Kind = ShapeBox};
    const auto drop = [&](uint32_t side) {
        World world{context};
        AddFloor(world, side);
        const auto slab = world.AddBody({.Pose = At(float3{0.13f, 1.f, 0.07f}), .Shape = world.AddShape(Slab)});
        uint32_t refused = 0;
        for (uint32_t step = 0; step < 240; ++step) { // measured every step, since a refusal is per step
            solver.Step(world);
            CheckManifolds(world);
            refused = std::max(refused, world.ContactRefusals[slab]);
        }
        return std::pair{Rest(world, slab), refused};
    };

    const auto [plane, plane_refused] = drop(0);
    const auto [coarse, coarse_refused] = drop(4); // eight quads over ten metres, well inside one batch
    const auto [fine, fine_refused] = drop(32); // a third of a metre per quad, which is ten batches

    CHECK(plane_refused == 0);
    CHECK(coarse_refused == 0);
    CHECK(fine_refused == 0); // a batch boundary is not a shortfall and is not counted as one

    for (const Settled &settled : {plane, coarse, fine}) {
        CheckResting(settled.Position.y);
        CHECK(settled.Normal.y == doctest::Approx(1).epsilon(1e-3));
        CHECK(settled.Contacts >= 3); // enough to hold it flat rather than balance it
    }
    CHECK(fine.Position.y == doctest::Approx(plane.Position.y).epsilon(2e-3));
    // It stays over the part of the floor it gathered first.
    CHECK(std::abs(float(fine.Position.x) - 0.13f) < 5e-3f);
    CHECK(std::abs(float(fine.Position.z) - 0.07f) < 5e-3f);
}

TEST_CASE_FIXTURE(OnDevice, "a box whose corners land on the seams is held by four contacts, not six") {
    // The knife edge the test above stays off, which exercises two mechanisms a flat landing never reaches.
    // A point on a plane counts as inside it for a clip, so both triangles along the seam would hold that corner.
    // The reference face goes to whichever shape is flatter, which here is a tie.
    const auto drop = [&](uint32_t side) {
        World world{context};
        AddFloor(world, side);
        // The quads are square, so a box centred on x = z has two corners on a seam at any cut.
        const auto box = world.AddBody({.Pose = At(float3{0.3f, 2, 0.3f}), .Shape = world.AddShape(UnitBox)});
        return RestAfter(solver, world, box, 180);
    };

    const Settled plane = drop(0);
    for (const uint32_t side : {1u, 2u, 16u}) {
        CAPTURE(side);
        const Settled settled = drop(side);
        CHECK(settled.Contacts == 4); // its own four corners, each held by exactly one triangle
        CheckResting(settled.Position.y);
        CHECK(settled.Position.y == doctest::Approx(plane.Position.y).epsilon(1e-4));
        CHECK(settled.Normal.y == doctest::Approx(1).epsilon(1e-4));
        // Flat rather than tipped, since a duplicated corner is stiffer than the other three and rolls the box off.
        CHECK(std::abs(settled.Position.x - plane.Position.x) < 1e-3f);
        CHECK(std::abs(settled.Position.z - plane.Position.z) < 1e-3f);
    }
}

TEST_CASE_FIXTURE(OnDevice, "a slope of triangles holds and lets go exactly as a sloped plane does") {
    // A flat mesh hides most of a mesh's difficulty, since a triangle with a +y normal lies under whatever is above it however the direction was found.
    // The plane of the same slope is the oracle.
    constexpr float Slope = 0.2f; // tan 0.203, so a friction of 0.5 holds it and one of 0.1 does not
    const float3 up{-std::sin(Slope), std::cos(Slope), 0}, down{-std::cos(Slope), -std::sin(Slope), 0};

    const auto drop = [&](uint32_t side, float friction) {
        World world{context};
        AddFloor(world, side, Slope, friction);
        // Resting square on the slope, a little above it and off the grid lines both ways.
        const float3 start = up * (Half + 0.05f) - down * 0.13f + float3{0, 0, 0.07f};
        const auto box = world.AddBody({.Pose = At(start, QuatFromRotationVector(float3{0, 0, Slope})), .Shape = world.AddShape(UnitBox), .Friction = friction});
        return RestAfter(solver, world, box, 180);
    };

    SUBCASE("inside the cone it stays where the plane keeps it") {
        const Settled plane = drop(0, 0.5f), tessellated = drop(16, 0.5f);
        for (const Settled &settled : {plane, tessellated}) {
            CheckResting(simd::dot(settled.Position, up)); // resting on the surface rather than inside it
            CHECK(settled.Contacts == 4); // its own four corners, whatever the floor was cut into
            CHECK(simd::dot(settled.Normal, up) == doctest::Approx(1).epsilon(1e-4));
        }
        // In the same place, since a box held by friction slips by whatever the solver allows and the two descriptions have to allow the same.
        CHECK(simd::length(tessellated.Position - plane.Position) < 1e-3f);
    }

    SUBCASE("outside it they slide the same distance") {
        // A sliding box passes its manifold on every few tenths of a metre, and any point a seam adds is friction the plane does not have.
        const Settled plane = drop(0, 0.1f), tessellated = drop(16, 0.1f);
        const float went = simd::dot(plane.Position, down), also = simd::dot(tessellated.Position, down);
        // mu is well under tan(slope), so it is still moving at g (sin - mu cos) over three seconds.
        const float expected = 0.5f * Gravity * (std::sin(Slope) - 0.1f * std::cos(Slope)) * 3 * 3;
        CHECK(went == doctest::Approx(expected).epsilon(0.05));
        CHECK(also == doctest::Approx(went).epsilon(2e-3));
        CheckResting(simd::dot(tessellated.Position, up)); // still on the surface rather than through it
        CHECK(tessellated.Contacts == 4);
    }
}

TEST_CASE_FIXTURE(OnDevice, "a sphere rolls across a tessellated floor without finding the seams") {
    // The case active edges exist for: every seam of a flat floor is an edge of the tessellation alone, so a ball rolling over one is unaffected.
    World plane_world{context}, mesh_world{context};
    for (World *world : {&plane_world, &mesh_world}) {
        const bool tessellated = world == &mesh_world;
        const Index floor = tessellated ? FloorMesh(*world, 16, 5) : world->AddShape(GroundPlane);
        REQUIRE(floor != NoIndex);
        world->AddBody({.Shape = floor});
        world->AddBody({.Pose = At(float3{-2, 0.25f, 0.07f}), .Velocity = {.Linear = {2, 0, 0}}, .Shape = world->AddShape({.Radius = 0.25f, .Kind = ShapeSphere})});
    }
    float highest = 0, lowest = 1;
    for (uint32_t step = 0; step < 120; ++step) {
        solver.Step(plane_world);
        solver.Step(mesh_world);
        highest = std::max(highest, mesh_world.Poses[1].Position.y);
        lowest = std::min(lowest, mesh_world.Poses[1].Position.y);
    }
    // It never leaves the floor and never sinks into it, however many seams it has crossed.
    CHECK(highest < 0.25f + 1e-3f);
    CHECK(lowest > 0.25f - 4 * StepSettings{}.ContactMargin);
    // It travels as far as the same ball on a plane, rather than being tripped or slowed.
    CHECK(mesh_world.Poses[1].Position.x == doctest::Approx(plane_world.Poses[1].Position.x).epsilon(1e-3));
    CHECK(std::abs(mesh_world.Poses[1].Position.z - plane_world.Poses[1].Position.z) < 1e-3f);
    CHECK(mesh_world.ContactRefusals[1] == 0);
}

TEST_CASE_FIXTURE(OneWorld, "nothing is pushed out of the back of a mesh") {
    // A mesh is a surface, so a body behind it has passed it.
    // A plane is a half space, which is where the two floors differ.
    REQUIRE(FloorMesh(world, 4, 5) != NoIndex);
    world.AddBody({.Shape = 0});
    const auto box = world.AddBody({.Pose = At(float3{0.13f, -1, 0.07f}), .Shape = world.AddShape(UnitBox)});

    Run(solver, world, 60);
    CHECK(world.Poses[box].Position.y < -1.f); // still falling, never caught from beneath
    CHECK(Rest(world, box).Contacts == 0);
}

TEST_CASE_FIXTURE(OnDevice, "a mesh scene steps to bit-identical state twice") {
    CheckReplay(context, [&](World &world) {
        FloorMesh(world, 8, 4);
        world.AddBody({.Shape = 0});
        const auto shape = world.AddShape(UnitBox);
        for (uint32_t i = 0; i < 4; ++i)
            world.AddBody({.Pose = At(float3{0.13f + 0.02f * float(i), Half + 1.02f * float(i), 0.07f}), .Shape = shape});
        for (uint32_t step = 0; step < 120; ++step) solver.Step(world);
    });
}

TEST_CASE_FIXTURE(OneWorld, "a mesh body moves only where the host says what it weighs") {
    // A surface has no volume to integrate, so only the host can supply a mass for a body using one, and without that it stays static geometry.
    const Index shape = BoxMesh(world, 0.4f);
    REQUIRE(shape != NoIndex);
    const auto scenery = world.AddBody({.Pose = At(float3{0, 4, 0}), .Shape = shape});
    const auto moving = world.AddBody({.Pose = At(float3{4, 4, 0}), .Shape = shape, .Mass = {{.Mass = 10, .Inertia = {2, 3, 4}}}});
    CHECK(world.Masses[scenery].InvMass == 0);
    CHECK(world.Masses[moving].InvMass == doctest::Approx(0.1f));
    CHECK(world.Masses[moving].InvInertiaLocal.x == doctest::Approx(0.5f));
    CHECK(world.Masses[moving].InvInertiaLocal.y == doctest::Approx(1 / 3.f));
    CHECK(world.Masses[moving].InvInertiaLocal.z == doctest::Approx(0.25f));

    // In free space, so this measures the integrator alone rather than any landing.
    const StepSettings settings{};
    Run(solver, world, 30, settings);
    CHECK(world.Poses[scenery].Position.y == 4); // static geometry, which does not move
    CHECK(world.Poses[moving].Position.y < 4 - 0.5f);
}

TEST_CASE_FIXTURE(OneWorld, "a moving mesh rests on a box, and a plane holds it nothing") {
    // A pair is owned by a body that presents a manifold, and neither a plane nor a mesh presents one.
    // A mesh against a plane has no owner and no contact, so a moving mesh needs a box to rest on.
    constexpr float MeshHalf = 0.4f;
    const Index shape = BoxMesh(world, MeshHalf);
    REQUIRE(shape != NoIndex);
    AddGround(world);
    world.AddBody({.Pose = At(float3{4, -0.25f, 0}), .Shape = world.AddShape({.HalfExtents = {2, 0.25f, 2}, .Kind = ShapeBox}), .Density = 0});
    const AuthoredMass mass{.Mass = 10, .Inertia = {1, 1, 1}};
    const auto over_plane = world.AddBody({.Pose = At(float3{0, 2, 0}), .Shape = shape, .Mass = mass});
    const auto over_box = world.AddBody({.Pose = At(float3{4, 2, 0}), .Shape = shape, .Mass = mass});

    Run(solver, world, 300);
    // Where its own half extent puts it, a contact margin in, the same result a box shape gets.
    CHECK(world.Poses[over_box].Position.y < MeshHalf);
    CHECK(world.Poses[over_box].Position.y > MeshHalf - MaxPenetration);
    CHECK(std::abs(float(world.Velocities[over_box].Linear.y)) < 1e-3f);
    CHECK(world.Poses[over_plane].Position.y < -5); // straight through the plane, which is the contract
}

TEST_CASE_FIXTURE(OneWorld, "the convex side owns a pair against a mesh whatever the indices say") {
    // A pair belongs to the lower-indexed of two dynamic bodies, and a mesh body's own thread returns before colliding, so ownership passes to the convex side.
    // Built mesh-first on purpose.
    const Index shape = BoxMesh(world, 0.4f);
    REQUIRE(shape != NoIndex);
    world.AddBody({.Pose = At(float3{0, -0.25f, 0}), .Shape = world.AddShape({.HalfExtents = {2, 0.25f, 2}, .Kind = ShapeBox}), .Density = 0});
    // Weighing the same as a cube of water, so a body standing on it does not squash it aside.
    // The box on top is deliberately narrower than the face.
    // A straddling body would also be supported by the side faces, which is geometry rather than ownership.
    constexpr float MeshMass = 512, MeshInertia = MeshMass * 2 * 0.64f / 12, TopHalf = 0.2f;
    bool moving = true;
    SUBCASE("a mesh the host gave a mass") { moving = true; }
    SUBCASE("against the same mesh left as scenery") { moving = false; }
    const auto mesh = moving
        ? world.AddBody({.Pose = At(float3{0, 0.4f, 0}), .Shape = shape, .Mass = {{.Mass = MeshMass, .Inertia = {MeshInertia, MeshInertia, MeshInertia}}}})
        : world.AddBody({.Pose = At(float3{0, 0.4f, 0}), .Shape = shape});
    const auto box = world.AddBody({.Pose = At(float3{0, 1.05f, 0}), .Shape = world.AddShape({.HalfExtents = {TopHalf, TopHalf, TopHalf}, .Kind = ShapeBox})});
    REQUIRE(mesh < box);

    Run(solver, world, 600);
    // The mesh owns nothing, and the box above it owns the pair despite its higher index.
    CHECK(ActiveContacts(world, mesh) == 0);
    CHECK(ActiveContacts(world, box) > 0);
    for (const Contact &contact : Slots(world, box))
        if (contact.Active) CHECK(contact.BodyB == mesh);
    // Held its own half extent above the mesh's top face, and still there three hundred steps later.
    CHECK(world.Poses[box].Position.y == doctest::Approx(0.8f + TopHalf).epsilon(0.02));
    const float held = float(world.Poses[box].Position.y);
    Run(solver, world, 300);
    CHECK(world.Poses[box].Position.y == doctest::Approx(held).epsilon(0.005));

    // It comes to rest on a mesh given a mass exactly as on the same mesh left as scenery, the subcase above being the control.
    // Correct positions reached through flickering impulses is a failure under the impulse-quality rule - see CollectContacts.
    CHECK(simd::length(world.Velocities[box].Linear) < 1e-3f);
    CHECK(simd::length(world.Velocities[mesh].Linear) < 1e-3f);
    CHECK(world.Quiet[box] >= StepSettings{}.SleepSteps); // a pile that comes to rest also sleeps
    // Standing the way it was set down, which a height alone cannot establish for a cube.
    CHECK(simd::length(RotationVector(world.Poses[mesh].Orientation)) < 1e-3f);
    // The mesh is held by the slab under it, which presents a manifold and so owns that pair.
    if (moving) CHECK(ActiveContacts(world, 0) > 0);
    CHECK(world.Poses[mesh].Position.y == doctest::Approx(0.4f).epsilon(0.02));
}

TEST_CASE_FIXTURE(OneWorld, "a mesh cube's bottom face is held on all four of its corners") {
    // Two triangles make that face and their seam is its diagonal, so the diagonal's ends are corners of both.
    // Dropping every point a seam cut is correct while the seam is the only cut, and wrong at a corner, where an active edge cut it too.
    // The cook names an owner for every seam, which selects the triangle that keeps the point.
    constexpr float MeshHalf = 0.4f, SlabHalf = 0.25f;
    const Index shape = BoxMesh(world, MeshHalf);
    REQUIRE(shape != NoIndex);
    // The slab presents a manifold and the mesh does not, so the pair lives in the slab's run whatever the indices are.
    // The mass is that of a cube of water, so the rows carry a real load.
    const auto slab = world.AddBody({.Pose = At(float3{0, -SlabHalf, 0}), .Shape = world.AddShape({.HalfExtents = {2, SlabHalf, 2}, .Kind = ShapeBox}), .Density = 0});
    constexpr float MeshMass = 512, MeshInertia = MeshMass * (2 * MeshHalf) * (2 * MeshHalf) / 6;
    // The yaw keeps the corners on the seam, since a slab that wide clipped into one triangle is that triangle.
    // It changes the alignment with the world axes, which nothing here depends on.
    float yaw = 0;
    bool square_on = true;
    SUBCASE("set down square on") { }
    SUBCASE("and turned ten degrees about the upright") {
        yaw = 10 * std::numbers::pi_v<float> / 180;
        square_on = false;
    }
    const auto mesh = world.AddBody({.Pose = At(float3{0, MeshHalf, 0}, QuatFromRotationVector(float3{0, yaw, 0})), .Shape = shape,
                                     .Mass = {{.Mass = MeshMass, .Inertia = {MeshInertia, MeshInertia, MeshInertia}}}});

    Run(solver, world, 600);
    CheckManifolds(world);
    CHECK(world.Poses[mesh].Position.y == doctest::Approx(MeshHalf).epsilon(0.02));
    // Standing the way it was set down, which a height alone cannot establish for a cube.
    CHECK(simd::length(RotationVector(world.Poses[mesh].Orientation)) == doctest::Approx(yaw).epsilon(0.01));

    std::vector<float> load;
    for (const Contact &contact : Slots(world, slab))
        if (contact.Active && contact.BodyB == mesh) load.push_back(std::abs(contact.Lambda[0]));
    // Every corner of the face it stands on has a row, the seam's two ends included.
    const Pose &at = world.Poses[mesh];
    for (const float x : {-MeshHalf, MeshHalf})
        for (const float z : {-MeshHalf, MeshHalf}) {
            const float3 corner = WorldPoint(at, float3{x, -MeshHalf, z});
            uint32_t rows = 0;
            for (const Contact &contact : Slots(world, slab))
                if (contact.Active && contact.BodyB == mesh)
                    rows += simd::distance(WorldPoint(at, contact.AnchorB), corner) < 1e-2f ? 1 : 0;
            CHECK(rows >= 1);
        }
    if (square_on) {
        // Square on, those four corners are the whole manifold and a level cube puts a quarter of its weight on each.
        // Turned, the side triangles reach them too, so the load split is left unchecked there.
        REQUIRE(load.size() == 4);
        const auto [least, most] = std::minmax_element(load.begin(), load.end());
        CHECK(*least > 0);
        CHECK(*most < 2 * *least);
    }
}

TEST_CASE_FIXTURE(OneWorld, "a kinematic paddle mesh strikes a ball") {
    // The thinnest possible mesh: one open quad, with no volume in the striker.
    // This works only if a massless body carries its motion into the constraint and a sleeping ball is collided again.
    constexpr float Swing = 5, Radius = 0.2f;
    const StepSettings settings{.Gravity = {0, 0, 0}};
    // Wound so (B - A) x (C - A) faces the direction of travel, since a mesh pushes from its front only.
    const std::vector<float3> quad{float3{0, -1, -1}, float3{0, -1, 1}, float3{0, 1, 1}, float3{0, 1, -1}};
    const std::vector<uint32_t> wound{0, 2, 1, 0, 3, 2};
    const Index shape = world.AddMesh(quad, wound);
    REQUIRE(shape != NoIndex);
    const auto paddle = world.AddBody({.Pose = At(float3{-2, 0, 0}), .Shape = shape});
    const auto ball = world.AddBody({.Pose = At(float3{0, 0, 0}), .Shape = world.AddShape({.Radius = Radius, .Kind = ShapeSphere})});
    REQUIRE(world.Masses[paddle].InvMass == 0); // a mesh with no authored mass, which is a kinematic body

    Run(solver, world, 60, settings);
    REQUIRE(world.Quiet[ball] >= settings.SleepSteps);

    float3 at = world.Poses[paddle].Position;
    DriveAlongX(solver, world, paddle, at, Swing, 90, settings, [&] { CHECK(float(world.Poses[ball].Position.x) + Radius > float(world.Poses[paddle].Position.x)); });
    // It leaves at the speed of the face that struck it and keeps that speed.
    CHECK(world.Velocities[ball].Linear.x == doctest::Approx(Swing).epsilon(0.01));
    Drive(world, paddle, at, float3(0));
    Run(solver, world, 60, settings);
    CHECK(world.Velocities[ball].Linear.x == doctest::Approx(Swing).epsilon(0.01));
}

TEST_CASE_FIXTURE(OnDevice, "a paddle wheel of one-sided quads turns, and its bare hub does not") {
    // The WaterWheel of the glTF physics samples: an open mesh on a hinge with host-supplied mass, turned by balls dropped onto the side its paddles face.
    // The same wheel with no paddles is the control.
    const auto run = [&](uint32_t paddles) {
        World world{context};
        const Index shape = WheelMesh(world, 16, paddles);
        REQUIRE(shape != NoIndex);
        AddGround(world); // so a ball that misses stops, rather than falling for ever at ghost speeds
        const auto wheel = world.AddBody({.Pose = At(float3{0, 0, 0}), .Shape = shape, .Mass = WheelMass(), .AngularDamping = 0.6f});
        // A hinge about z, with the axes body B's, and B is upright so the free axis is the world's z.
        const auto axle = world.AddBody({});
        world.AddJoint({.BodyA = wheel, .BodyB = axle, .At = float3{0, 0, 0}, .Angular = {AxisLocked, AxisLocked, AxisFree}});

        const Index ball = world.AddShape({.Radius = 0.16f, .Kind = ShapeSphere});
        float turned = 0;
        float4 was = world.Poses[wheel].Orientation;
        for (uint32_t step = 0; step < 600; ++step) {
            if (step % 20 == 0 && step < 240)
                world.AddBody({.Pose = At(float3{0.62f, 2.2f, 0}), .Shape = ball});
            solver.Step(world);
            // Accumulated, since a quaternion difference only gives the short way round.
            const float4 now = world.Poses[wheel].Orientation;
            turned += RotationVector(QuatMul(now, QuatConjugate(was))).z;
            was = now;
        }
        // The hinge holds it, with two locked axes and an anchor it never leaves.
        CHECK(simd::length(world.Poses[wheel].Position) < 1e-2f);
        return turned;
    };

    const float paddled = run(8), bare = run(0);
    CAPTURE(paddled);
    CAPTURE(bare);
    // It turns one way, since the far side's paddles present their backs to a falling ball.
    // A deliberately loose bound, since twelve balls on a paddle wheel is chaotic and an ulp of fast-math contraction is worth radians.
    CHECK(std::abs(paddled) > 1.f);
    CHECK(std::abs(bare) < 0.25f * std::abs(paddled));
}

TEST_CASE_FIXTURE(OnDevice, "a scene with a dynamic mesh in it steps to bit-identical state twice") {
    // The case with the most moving parts: a batched tree walk and per-triangle manifolds.
    // A body is integrated while something else is clipped into its own triangles.
    CheckReplay(context, [&](World &world) {
        const Index shape = BoxMesh(world, 0.4f);
        REQUIRE(shape != NoIndex);
        world.AddBody({.Pose = At(float3{0, -0.25f, 0}), .Shape = world.AddShape({.HalfExtents = {3, 0.25f, 3}, .Kind = ShapeBox}), .Density = 0});
        world.AddBody({.Pose = At(float3{0, 1.2f, 0}), .Shape = shape, .Mass = {{.Mass = 10, .Inertia = {1, 1, 1}}}});
        world.AddBody({.Pose = At(float3{0.13f, 2.4f, 0.07f}), .Shape = world.AddShape(UnitBox)});
        world.AddBody({.Pose = At(float3{-0.2f, 3.6f, 0.1f}), .Shape = world.AddShape({.Radius = 0.3f, .Kind = ShapeSphere})});
        Run(solver, world, 240);
    });
}

TEST_CASE_FIXTURE(OneWorld, "a manifold wider than four points keeps the four that hold it") {
    // Eight corners of a face touching at once, more than the solver uses and more than the contact budget allows.
    // The four that survive still have to resist rocking, so the reduction picks for area.
    AddGround(world);
    std::vector<float3> points;
    for (uint32_t corner = 0; corner < 8; ++corner) {
        const float angle = 2 * 3.14159265f * float(corner) / 8;
        for (const float y : {-Half, Half}) points.push_back(float3{std::cos(angle), y, std::sin(angle)});
    }
    const auto shape = world.AddHull(points);
    REQUIRE(shape != NoIndex);
    REQUIRE(world.Shapes[shape].VertexCount == 16);
    const auto prism = world.AddBody({.Pose = At(float3{0, 1.5f, 0}), .Shape = shape});

    Run(solver, world, 240);
    const Settled settled = Rest(world, prism);
    CheckResting(settled.Position.y);
    CHECK(settled.Contacts == 4); // its base has eight corners on the plane, reduced to four rows
    CHECK(world.ContactRefusals[prism] == 0);

    // Four points along one edge would let it rock, which is why the choice maximizes area.
    float3 low{1e9f, 0, 1e9f}, high{-1e9f, 0, -1e9f};
    const auto slots = Slots(world, prism);
    for (const auto &contact : slots) {
        if (!contact.Active) continue;
        CHECK(contact.AnchorA.y == doctest::Approx(-Half).epsilon(1e-3)); // all on the base
        low = simd::min(low, contact.AnchorA);
        high = simd::max(high, contact.AnchorA);
    }
    // The base is a unit-radius octagon, so four points spanning it reach across most of it both ways.
    CHECK(high.x - low.x > 1.4f);
    CHECK(high.z - low.z > 1.4f);
    CHECK(simd::length(world.Velocities[prism].Linear) < 1e-2f);
}

// A shape's pose within the body frame is the one mechanism behind a collider offset from its node.
// It is also the mechanism behind a centre of mass off the geometric one and an authored inertia orientation.
// These cases are mostly identities: one body described two ways gives one result.
namespace {
// Compares two descriptions of one body, a contact at a time.
// The comparison is in world space and matched by position, since the body frames differ and a feature names an index from a different cook.
void CheckSameContacts(const World &one, Index a, const World &other, Index b, float tolerance) {
    uint32_t found = 0;
    for (const Contact &mine : Slots(one, a)) {
        if (!mine.Active) continue;
        ++found;
        const Contact *nearest = nullptr;
        float closest = INFINITY;
        for (const Contact &theirs : Slots(other, b)) {
            if (!theirs.Active) continue;
            const float apart = simd::distance(ContactPoint(one, mine, true), ContactPoint(other, theirs, true));
            if (apart >= closest) continue;
            closest = apart;
            nearest = &theirs;
        }
        REQUIRE(nearest != nullptr);
        CHECK(closest < tolerance); // where the point sits on the body
        CHECK(simd::distance(ContactPoint(one, mine, false), ContactPoint(other, *nearest, false)) < tolerance); // and where it sits on the supporting body
        CHECK(simd::distance(mine.Normal, nearest->Normal) < tolerance);
        CHECK(simd::distance(mine.C0, nearest->C0) < tolerance);
    }
    CHECK(found > 0);
    CHECK(found == ActiveContacts(other, b));
}

// How far apart two runs have left one body: the larger of the position distance and the angle between the orientations.
// An angle rather than a quaternion distance, since q and -q are the same rotation.
float PosesApart(Pose one, Pose other) {
    const float turned = simd::length(RotationVector(QuatMul(one.Orientation, QuatConjugate(other.Orientation))));
    return std::max(float(simd::distance(one.Position, other.Position)), turned);
}

// Every corner of a shape in the body frame, taken through its local pose.
std::vector<float3> ShapeCorners(const World &world, Index shape) {
    const Shape &it = world.Shapes[shape];
    std::vector<float3> corners;
    if (it.Kind == ShapeBox)
        for (uint32_t corner = 0; corner < 8; ++corner)
            corners.push_back(it.HalfExtents * CornerSign(corner));
    else
        for (uint32_t i = 0; i < it.VertexCount; ++i) corners.push_back(world.ShapeVertices[it.FirstVertex + i]);
    for (float3 &corner : corners) corner = WorldPoint(it.Local, corner);
    return corners;
}

// Where to put a body's origin so that, turned this way, its lowest corner starts `gap` above the plane.
// Two runs are comparable only once contacts exist, and a body dropped from a height arrives tumbling.
float DropHeight(const World &world, Index shape, float4 turn, float gap) {
    float lowest = INFINITY;
    for (const float3 corner : ShapeCorners(world, shape)) lowest = std::min(lowest, float(Rotate(turn, corner).y));
    return gap - lowest;
}
} // namespace

TEST_CASE_FIXTURE(OnDevice, "a hull at a pose within its body is those same points cooked already there") {
    // One caller supplies the shape's pose in the body frame and lets the cook move the points, the other supplies the points already moved.
    // Both need an authored mass, since neither frame is one the cook chose.
    constexpr AuthoredMass Weight{.Mass = 30, .Inertia = {4, 5, 6}};
    const Pose local = At(float3{0.35f, -0.2f, 0.15f}, QuatFromRotationVector(float3{0.4f, 0.9f, -0.25f}));
    const std::vector<float3> points = WedgePoints();
    std::vector<float3> moved;
    for (const float3 point : points) moved.push_back(WorldPoint(local, point));

    World placed{context}, baked{context};
    AddGround(placed);
    AddGround(baked);
    const Index placed_shape = placed.AddHull(points, nullptr, local);
    // An explicit identity states that the points are already in the body frame, which differs from omitting the pose.
    const Index baked_shape = baked.AddHull(moved, nullptr, IdentityPose);
    REQUIRE(placed_shape != NoIndex);
    REQUIRE(baked_shape != NoIndex);
    // The body is turned to undo the shape's own turn, so the wedge lands squarely on its base rather than toppling.
    // It starts close enough that the first step already has the contacts the runs are compared on.
    const float4 turn = QuatConjugate(placed.Shapes[placed_shape].Local.Orientation);
    const Pose start = At(float3{0.2f, DropHeight(placed, placed_shape, turn, 5e-4f), -0.1f}, turn);
    const Index one = placed.AddBody({.Pose = start, .Shape = placed_shape, .Mass = Weight});
    const Index other = baked.AddBody({.Pose = start, .Shape = baked_shape, .Mass = Weight});
    REQUIRE(one != NoIndex);
    REQUIRE(other != NoIndex);

    solver.Step(placed);
    solver.Step(baked);
    CheckManifolds(placed);
    CheckManifolds(baked);
    CheckSameContacts(placed, one, baked, other, 1e-5f);

    float worst = 0;
    for (uint32_t step = 1; step < 200; ++step) {
        solver.Step(placed);
        solver.Step(baked);
        worst = std::max(worst, PosesApart(placed.Poses[one], baked.Poses[other]));
    }
    CAPTURE(worst);
    CHECK(worst < 1e-5f);
    CHECK(placed.ContactRefusals[one] == 0);
}

TEST_CASE_FIXTURE(OnDevice, "a box turned inside its body is a body turned under its box") {
    // Turning the shape within the body and turning the body under the shape put the same geometry in the same place.
    // They are one body only while the mass reads the same in both frames.
    // R I R^T equals I only for isotropic I, so both are authored with an isotropic inertia.
    const float4 turn = QuatFromRotationVector(float3{0.3f, -0.5f, 0.8f});
    constexpr AuthoredMass Weight{.Mass = 1000, .Inertia = {160, 160, 160}};

    World inside{context}, under{context};
    AddGround(inside);
    AddGround(under);
    const Index turned = inside.AddShape({.HalfExtents = {Half, Half, Half}, .Kind = ShapeBox, .Local = At(float3{0, 0, 0}, turn)});
    // Turned to undo the shape's own turn and set just clear of the plane, as the hull above is.
    const float4 tilt = QuatConjugate(turn);
    const Pose start = At(float3{0, DropHeight(inside, turned, tilt, 5e-4f), 0}, tilt);
    const Index one = inside.AddBody({.Pose = start, .Shape = turned, .Mass = Weight});
    const Index other = under.AddBody({.Pose = ComposePose(start, At(float3{0, 0, 0}, turn)), .Shape = under.AddShape(UnitBox), .Mass = Weight});
    REQUIRE(one != NoIndex);
    REQUIRE(other != NoIndex);

    solver.Step(inside);
    solver.Step(under);
    CheckSameContacts(inside, one, under, other, 1e-5f);

    // The two frames stay exactly that rotation apart.
    float worst = 0;
    for (uint32_t step = 1; step < 200; ++step) {
        solver.Step(inside);
        solver.Step(under);
        worst = std::max(worst, PosesApart(ComposePose(inside.Poses[one], At(float3{0, 0, 0}, turn)), under.Poses[other]));
    }
    CAPTURE(worst);
    CHECK(worst < 1e-5f);
}

TEST_CASE_FIXTURE(OneWorld, "an offset box rests on the face its own geometry puts on the plane") {
    // The collider sits three tenths of a metre along x from the body's origin, which is its centre of mass.
    // The box rests at the height its own geometry gives, with every anchor three tenths off the origin.
    AddGround(world);
    constexpr float Offset = 0.3f;
    const Index shape = world.AddShape({.HalfExtents = {Half, Half, Half}, .Kind = ShapeBox, .Local = {{Offset, 0, 0}, {0, 0, 0, 1}}});
    // Weighed about the body's origin.
    // The parallel axis carry is the host's arithmetic, and the offset is along x, so only the two axes across it gain a term.
    constexpr float Mass = 1000, Centred = Mass / 12 * (1 + 1), Carry = Mass * Offset * Offset;
    const Index body = world.AddBody({.Pose = At(float3{0, 1.2f, 0}), .Shape = shape, .Mass = {{.Mass = Mass, .Inertia = {Centred, Centred + Carry, Centred + Carry}}}});
    REQUIRE(body != NoIndex);
    Run(solver, world, 240);

    CheckResting(world.Poses[body].Position.y); // the box's own bottom on the plane rather than the origin
    CHECK(std::abs(float(world.Poses[body].Position.x)) < 1e-3f); // it does not drift along x
    const Settled settled = Rest(world, body);
    CHECK(settled.Contacts == 4);
    CHECK(settled.Normal.y == doctest::Approx(1).epsilon(1e-3));
    float3 low{1e9f, 1e9f, 1e9f}, high{-1e9f, -1e9f, -1e9f}, total{0, 0, 0};
    for (const Contact &contact : Slots(world, body)) {
        if (!contact.Active) continue;
        low = simd::min(low, contact.AnchorA);
        high = simd::max(high, contact.AnchorA);
        total += contact.AnchorA;
    }
    // The four corners of the box's own bottom face, a metre across and centred on the offset.
    CHECK(float(total.x / 4) == doctest::Approx(Offset).epsilon(1e-3));
    CHECK(float(low.x) == doctest::Approx(Offset - Half).epsilon(1e-3));
    CHECK(float(high.x) == doctest::Approx(Offset + Half).epsilon(1e-3));
    CHECK(float(low.y) == doctest::Approx(-Half).epsilon(1e-3));
}

TEST_CASE_FIXTURE(OnDevice, "a body whose geometry is off its centre of mass rests differently from a centred one") {
    // The qualitative half: both boxes present the same face on a ledge narrower than they are, and differ only in where the mass sits.
    // One stands and the other topples.
    constexpr float Offset = 0.3f, LedgeHalf = 0.2f, Top = 0.5f;
    constexpr float Mass = 1000, Centred = Mass / 12 * (1 + 1);

    const auto stand = [&](bool offset) {
        World world{context};
        AddGround(world);
        // A ledge standing on the plane, narrower than the box that balances on it.
        world.AddBody({.Pose = At(float3{0, Top / 2, 0}), .Shape = world.AddShape({.HalfExtents = {LedgeHalf, Top / 2, 1}, .Kind = ShapeBox}), .Density = 0});
        const float local = offset ? Offset : 0;
        const Index shape = world.AddShape({.HalfExtents = {Half, Half, Half}, .Kind = ShapeBox, .Local = {{local, 0, 0}, {0, 0, 0, 1}}});
        const float carry = Mass * local * local;
        // Placed so the box is centred over the ledge in both, with only the mass position differing.
        const Index body = world.AddBody({.Pose = At(float3{-local, Top + Half, 0}), .Shape = shape, .Mass = {{.Mass = Mass, .Inertia = {Centred, Centred + carry, Centred + carry}}}});
        REQUIRE(body != NoIndex);
        Run(solver, world, 400);
        return world.Poses[body];
    };

    const Pose centred_rest = stand(false), offset_rest = stand(true);
    // The centred one is still on the ledge and level.
    CHECK(float(centred_rest.Position.y) == doctest::Approx(Top + Half).epsilon(2e-3));
    CHECK(simd::length(RotationVector(centred_rest.Orientation)) < 0.05f);
    // The offset one has toppled and lies on the plane below.
    CHECK(float(offset_rest.Position.y) < Top);
    CHECK(simd::length(RotationVector(offset_rest.Orientation)) > 0.5f);
}

TEST_CASE_FIXTURE(OneWorld, "a ball whose mass is off its centre settles with the mass underneath") {
    // A weeble: the sphere's centre sits along the body's y from the centre of mass, so there is one rest orientation and it has to roll to reach it.
    // This is the case where a swept contact must not inherit anchors.
    AddGround(world);
    constexpr float Radius = 0.5f, Offset = 0.25f, Mass = 500, Inertia = 2.f / 5 * Mass * Radius * Radius;
    const Index shape = world.AddShape({.Radius = Radius, .Kind = ShapeSphere, .Local = {{0, Offset, 0}, {0, 0, 0, 1}}});
    // Lying on its side to start with, so the mass is out beside the sphere and it has to roll.
    const float4 start = QuatFromRotationVector(float3{0, 0, std::numbers::pi_v<float> / 2});
    REQUIRE(float(Rotate(start, float3{0, Offset, 0}).x) == doctest::Approx(-Offset).epsilon(1e-4));
    // Damped, since nothing models rolling resistance and an undamped ball never arrives at rest.
    const Index body = world.AddBody({.Pose = At(float3{0, Radius, 0}, start), .Shape = shape, .Mass = {{.Mass = Mass, .Inertia = {Inertia, Inertia, Inertia}}}, .LinearDamping = 1.2f, .AngularDamping = 1.2f});
    REQUIRE(body != NoIndex);
    Run(solver, world, 900);

    // Where the sphere's own centre ended up relative to the centre of mass: straight above it.
    const Pose pose = world.Poses[body];
    const float3 centre = Rotate(pose.Orientation, float3{0, Offset, 0});
    CAPTURE(centre.x);
    CAPTURE(centre.y);
    CHECK(float(centre.y) > 0.9f * Offset);
    // It rests with that centre one radius up, so the body's origin is a radius less the offset.
    CHECK(float(pose.Position.y) == doctest::Approx(Radius - Offset).epsilon(2e-2));
    CHECK(simd::length(world.Velocities[body].Linear) < 1e-2f);
}

TEST_CASE_FIXTURE(OneWorld, "a capsule laid down by its local pose rests on its side where its radius puts it") {
    // Both halves of a local pose at once, on a round shape.
    // The core is turned onto world x and moved along it, so its rest position cannot come from the body's own axes.
    AddGround(world);
    // The offset is shorter than the core's half length, so the centre of mass stays between the ends
    // the capsule rests on rather than out past one, where it would tip.
    constexpr float Radius = 0.3f, HalfLength = 0.4f, Offset = 0.2f;
    const float4 lay = QuatFromRotationVector(float3{0, 0, -std::numbers::pi_v<float> / 2});
    const Index shape = world.AddShape({.HalfExtents = {0, HalfLength, 0}, .Radius = Radius, .Kind = ShapeCapsule, .Local = At(float3{Offset, 0, 0}, lay)});
    const Index body = world.AddBody({.Pose = At(float3{0, 1, 0}), .Shape = shape, .Mass = {{.Mass = 100, .Inertia = {10, 10, 10}}}});
    REQUIRE(body != NoIndex);
    Run(solver, world, 300);

    // On its side, so its own axis is one radius above the plane and the body's origin is with it.
    CHECK(float(world.Poses[body].Position.y) < Radius);
    CHECK(float(world.Poses[body].Position.y) > Radius - MaxPenetration);
    const Settled settled = Rest(world, body);
    CHECK(settled.Contacts == 2); // the two ends of the core, which is the whole of a capsule's manifold
    float3 total{0, 0, 0};
    for (const Contact &contact : Slots(world, body)) {
        if (!contact.Active) continue;
        total += contact.AnchorA;
        CHECK(float(contact.AnchorA.y) == doctest::Approx(-Radius).epsilon(2e-2));
    }
    // Centred on where the local pose put the capsule rather than on the body's origin.
    CHECK(float(total.x / 2) == doctest::Approx(Offset).epsilon(2e-2));
}

TEST_CASE_FIXTURE(OnDevice, "a mesh at a pose within its body is the floor those triangles cooked already there") {
    // The same identity on the mesh path, where a local pose has to reach the tree's frame as well as the triangles.
    // A Local missed in either place puts the ball on a level floor, metres away.
    constexpr float Slope = 0.2f, BallRadius = 0.4f;
    const Pose tilt = At(float3{0, 0, 0}, QuatFromRotationVector(float3{0, 0, Slope}));
    // Three ways to express one slope: turn the shape, turn the body, or supply the triangles already turned.
    constexpr uint32_t InTheShape = 0, InTheBody = 1, InThePoints = 2;
    const auto roll = [&](uint32_t how) {
        World world{context};
        const Index floor = how == InThePoints ? FloorMesh(world, 8, 5, Slope) : FloorMesh(world, 8, 5, 0, how == InTheShape ? tilt : IdentityPose);
        REQUIRE(floor != NoIndex);
        // A surface with no authored mass is static geometry, wherever it sits.
        world.AddBody({.Pose = how == InTheBody ? tilt : IdentityPose, .Shape = floor});
        // Set down on the slope rather than dropped onto it, so the comparison is of the rolling.
        const float3 up{-std::sin(Slope), std::cos(Slope), 0}; // the slope's own normal, one radius out
        const Index ball = world.AddBody({.Pose = At(BallRadius * up + float3{0, 0, 0.35f}), .Shape = world.AddShape({.Radius = BallRadius, .Kind = ShapeSphere})});
        REQUIRE(ball != NoIndex);
        std::vector<Pose> track;
        for (uint32_t step = 0; step < 180; ++step) {
            solver.Step(world);
            track.push_back(world.Poses[ball]);
        }
        return track;
    };

    const auto shaped = roll(InTheShape), bodied = roll(InTheBody), pointed = roll(InThePoints);
    const auto worst = [](const std::vector<Pose> &one, const std::vector<Pose> &other) {
        REQUIRE(one.size() == other.size());
        float apart = 0;
        for (uint32_t step = 0; step < one.size(); ++step) apart = std::max(apart, PosesApart(one[step], other[step]));
        return apart;
    };
    CHECK(float(shaped.back().Position.x) < -1.f); // it rolls down the slope rather than staying still

    // Three metres of rolling, the distance the tolerances below are a fraction of.
    const float travelled = simd::distance(shaped.back().Position, shaped.front().Position);
    CAPTURE(travelled);
    CHECK(travelled > 2.5f);

    // One cooked mesh, one tree, one tessellation, with only the pose the narrowphase reads it through differing.
    // The remaining difference is float32 rounding: the mesh-side anchor lives in the mesh body's frame, which the two runs keep a rotation apart.
    // The bound is a part in a hundred thousand of the distance travelled.
    const float mechanism = worst(shaped, bodied);
    CAPTURE(mechanism);
    CHECK(mechanism < 1e-4f);
    // Against the same surface authored already tilted, which is a separate cook.
    // Different coordinates split the tree differently, so the triangles arrive in a different order.
    const float cooked = worst(shaped, pointed);
    CAPTURE(cooked);
    CHECK(cooked < 2e-4f);
}

TEST_CASE_FIXTURE(OnDevice, "a scene of offset shapes steps to bit-identical state twice") {
    CheckReplay(context, [&](World &world) {
        AddGround(world);
        constexpr float Mass = 1000, Centred = Mass / 12 * (1 + 1), Carry = Mass * 0.1f;
        const Index box = world.AddShape({.HalfExtents = {Half, Half, Half}, .Kind = ShapeBox, .Local = {{0.3f, 0, 0.1f}, {0, 0, 0, 1}}});
        const Index ball = world.AddShape({.Radius = 0.4f, .Kind = ShapeSphere, .Local = {{0, 0.2f, 0}, {0, 0, 0, 1}}});
        for (uint32_t i = 0; i < 3; ++i)
            world.AddBody({.Pose = At(float3{0.11f * float(i), Half + 1.05f * float(i), 0.07f}, QuatFromRotationVector(float3{0.1f, 0.3f * float(i), 0})), .Shape = box, .Mass = {{.Mass = Mass, .Inertia = {Centred + Carry, Centred + Carry, Centred + Carry}}}});
        world.AddBody({.Pose = At(float3{1.4f, 1, 0}), .Shape = ball, .Mass = {{.Mass = 300, .Inertia = {20, 20, 20}}}});
        Run(solver, world, 120);
    });
}

// An explicit mass of zero, which KHR physics rigid bodies Sec. 128 defines as infinite rather than absent.
// Nothing translates the body, and its inertia stays finite so it turns freely in place.
// The two halves of the 6x6 block are gated separately, on Moves rather than on inverse mass.
namespace {
// A solid cube of `mass` and `side` about its own centre: m (e^2 + e^2) / 12, the same about each axis.
AuthoredMass CubeMass(float mass, float side) {
    const float inertia = mass * side * side / 6;
    return {.Mass = mass, .Inertia = {inertia, inertia, inertia}};
}
// The same cube pinned: its inertia, with a mass of zero.
AuthoredMass PinnedCubeMass(float mass, float side) {
    return {.Mass = 0, .Inertia = CubeMass(mass, side).Inertia};
}
} // namespace

TEST_CASE_FIXTURE(OneWorld, "a body of infinite mass hangs where it is put and turns where it hangs") {
    // Gravity acts through the inverse mass, which is zero, so no acceleration reaches the body.
    // It stays exactly where it started rather than approximately, since nothing here converges.
    const auto pinned = world.AddBody({.Pose = At(float3{0, 3, 0}), .Shape = world.AddShape(UnitBox), .Mass = {PinnedCubeMass(10, 1)}});
    REQUIRE(pinned != NoIndex);
    REQUIRE(world.Masses[pinned].InvMass == 0);
    REQUIRE(world.Masses[pinned].InvInertiaLocal.y > 0);
    const Pose was = world.Poses[pinned];
    Run(solver, world, 300);
    CHECK(simd::distance(world.Poses[pinned].Position, was.Position) < 1e-6f);

    // A body with an inertia has a rotational block, and free flight about its own axis keeps its rate.
    world.Velocities[pinned] = {.Angular = {0, 1.5f, 0}};
    Run(solver, world, 120);
    CHECK(float(world.Velocities[pinned].Angular.y) == doctest::Approx(1.5f).epsilon(0.01));
    CHECK(simd::distance(world.Poses[pinned].Position, was.Position) < 1e-6f);
}

TEST_CASE_FIXTURE(OnDevice, "a ball striking a pinned body off centre spins it and never moves it") {
    // In free space with restitution and friction both zero the only force is along the contact normal.
    // The ball's change of momentum measures that force whole, as J = m dv.
    // The pinned body then turns at r x J / I, with the arm exactly the ball's centre height, since the normal is the box face's own +-x.
    constexpr float BallMass = 5, BallRadius = 0.25f, Offset = 0.25f, Speed = 3, Inertia = 20;
    const StepSettings free_space{.Gravity = {0, 0, 0}};

    // Struck as a body of infinite mass, and again as scenery for the control, since a body with no inertia has no rotational block.
    const auto strike = [&](bool turns, float friction) {
        World world{context};
        const auto pinned = world.AddBody({.Shape = world.AddShape(UnitBox),
                                           .Mass = {{.Mass = 0, .Inertia = turns ? float3{Inertia, Inertia, Inertia} : float3{0, 0, 0}}},
                                           .Friction = friction});
        REQUIRE(pinned != NoIndex);
        REQUIRE(world.Masses[pinned].InvMass == 0);
        // Started close so it arrives before anything sleeps, and read soon after, so the measurement is of the collision rather than the motion that follows.
        const auto ball = world.AddBody({.Pose = At(float3{-Half - BallRadius - 0.3f, Offset, 0}),
                                         .Velocity = {.Linear = {Speed, 0, 0}},
                                         .Shape = world.AddShape({.Radius = BallRadius, .Kind = ShapeSphere}),
                                         .Mass = {{.Mass = BallMass, .Inertia = {1, 1, 1}}}, .Friction = friction});
        REQUIRE(ball != NoIndex);
        const Pose was = world.Poses[pinned];
        Run(solver, world, 24, free_space);
        // The momentum the ball loses along x is the impulse on the pinned body, nothing else acting.
        const float impulse = BallMass * (Speed - float(world.Velocities[ball].Linear.x));
        return std::tuple{impulse, float(world.Velocities[pinned].Angular.z),
                          float(simd::distance(world.Poses[pinned].Position, was.Position))};
    };

    SUBCASE("frictionless, against the closed form") {
        const auto [impulse, spin, moved] = strike(true, 0);
        CAPTURE(impulse);
        REQUIRE(impulse > 1.f); // the strike landed
        // Pushed along +x above the centre turns it towards -z: r x J is (0, 0, -Offset J).
        CHECK(spin == doctest::Approx(-Offset * impulse / Inertia).epsilon(0.02));
        CHECK(moved < 1e-6f);
    }
    SUBCASE("with friction, which adds two rows and still moves nothing") {
        // Friction gives the force a second direction, so the spin is no longer the normal impulse alone.
        // The checks that remain are the sense of the spin and the body holding its place.
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

TEST_CASE_FIXTURE(OneWorld, "a wheel pinned by an infinite mass turns on no joint at all") {
    // KHR's Wheel_InfiniteMassFiniteInertia, a wheel rather than a hinge: nothing joins it to anything and its axle is held by its infinite mass alone.
    // Its inertia covers one axis, so the block drops a single direction rather than a whole half.
    const Index shape = WheelMesh(world, 16, 8);
    REQUIRE(shape != NoIndex);
    AddGround(world); // so a ball that misses stops, rather than falling for ever at ghost speeds
    const auto wheel = world.AddBody({.Shape = shape, .Mass = {{.Mass = 0, .Inertia = {0, 0, 20 * WheelRadius * WheelRadius / 2}}},
                                      .Friction = 0.5f, .AngularDamping = 0.6f});
    REQUIRE(wheel != NoIndex);
    REQUIRE(world.Masses[wheel].InvMass == 0);
    REQUIRE(world.Masses[wheel].InvInertiaLocal.x == 0);
    REQUIRE(world.Masses[wheel].InvInertiaLocal.z > 0);

    const Index ball = world.AddShape({.Radius = 0.16f, .Kind = ShapeSphere});
    float turned = 0, wandered = 0;
    float4 was = world.Poses[wheel].Orientation;
    for (uint32_t step = 0; step < 600; ++step) {
        if (step % 20 == 0 && step < 240) REQUIRE(world.AddBody({.Pose = At(float3{0.62f, 2.2f, 0}), .Shape = ball}) != NoIndex);
        solver.Step(world);
        CheckManifolds(world);
        // Accumulated, since a quaternion difference only gives the short way round.
        const float4 now = world.Poses[wheel].Orientation;
        turned += RotationVector(QuatMul(now, QuatConjugate(was))).z;
        was = now;
        wandered = std::max(wandered, float(simd::length(world.Poses[wheel].Position)));
    }
    CAPTURE(turned);
    // It turns one way, as the hinged wheel does, and is coarsely bounded for the same reason.
    CHECK(std::abs(turned) > 1.f);
    // It never moves with no joint holding it, since the mass alone pins it.
    CHECK(wandered < 1e-6f);
    // About its own axis alone, since the two rigid directions are dropped from the block.
    const float3 tilt = RotationVector(world.Poses[wheel].Orientation);
    CAPTURE(float(tilt.x));
    CAPTURE(float(tilt.y));
    CHECK(std::abs(float(tilt.x)) < 1e-4f);
    CHECK(std::abs(float(tilt.y)) < 1e-4f);
}

TEST_CASE_FIXTURE(OneWorld, "a box rests on a pinned body, held up by a mass that is not finite") {
    // A slab nothing can translate supports a box exactly as static geometry would, and the pair's reduced mass is the box's alone.
    // The penalty floor is that box's own m/h^2, since the sum of inverses drops an infinite mass rather than dividing by zero.
    constexpr float BoxMass = 1000;
    const auto slab = world.AddBody({.Pose = At(float3{0, -0.25f, 0}),
                                     .Shape = world.AddShape({.HalfExtents = {3, 0.25f, 3}, .Kind = ShapeBox}),
                                     .Mass = {{.Mass = 0, .Inertia = {200, 200, 200}}}, .Friction = 0.5f});
    const auto box = world.AddBody({.Pose = At(float3{0, Half + 0.2f, 0}), .Shape = world.AddShape(UnitBox),
                                    .Mass = {CubeMass(BoxMass, 1)}, .Friction = 0.5f});
    REQUIRE(slab != NoIndex);
    REQUIRE(box != NoIndex);
    const Pose was = world.Poses[slab];
    Run(solver, world, 300);

    CheckResting(float(world.Poses[box].Position.y));
    CHECK(simd::length(world.Velocities[box].Linear) < 0.01f);
    // Infinite rather than merely large: the slab takes the whole weight without sinking.
    CHECK(simd::distance(world.Poses[slab].Position, was.Position) < 1e-6f);
    // The load sits squarely over its middle, so it is not turned either.
    CHECK(simd::length(world.Velocities[slab].Angular) < 1e-3f);

    const float inertial = BoxMass / (StepSettings{}.DeltaTime * StepSettings{}.DeltaTime);
    uint32_t rows = 0;
    for (uint32_t slot = 0; slot < world.Contacts.Capacity; ++slot) {
        const Contact &contact = world.Contacts[slot];
        if (!contact.Active) continue;
        ++rows;
        CAPTURE(slot);
        // Finite, and the box's own inertial stiffness, where an infinite mass would give 1/0 in the sum.
        CHECK(std::isfinite(contact.Penalty[0]));
        CHECK(contact.Penalty[0] > 0.3f * inertial);
        CHECK(contact.Penalty[0] < 3 * inertial);
    }
    CHECK(rows == 4); // a box face on a box face, which clips to exactly four points
}

TEST_CASE_FIXTURE(OneWorld, "a pinned body sleeps when it stops turning, and a strike wakes it") {
    // It has a quiet count like any other body the solve moves, where testing on inverse mass alone would leave it unable to come to rest or to be woken.
    const StepSettings free_space{.Gravity = {0, 0, 0}};
    const auto pinned = world.AddBody({.Velocity = {.Angular = {0, 0, 2}}, .Shape = world.AddShape(UnitBox),
                                       .Mass = {PinnedCubeMass(40, 1)}, .Friction = 0, .AngularDamping = 4});
    REQUIRE(pinned != NoIndex);
    Run(solver, world, 300, free_space);
    CHECK(world.Quiet[pinned] >= free_space.SleepSteps);
    const float4 settled = world.Poses[pinned].Orientation;

    // A ball into its face off centre wakes it and turns it.
    const auto ball = world.AddBody({.Pose = At(float3{-1.2f, 0.25f, 0}), .Velocity = {.Linear = {4, 0, 0}},
                                     .Shape = world.AddShape({.Radius = 0.25f, .Kind = ShapeSphere}),
                                     .Mass = {{.Mass = 30, .Inertia = {1, 1, 1}}}, .Friction = 0});
    REQUIRE(ball != NoIndex);
    Run(solver, world, 30, free_space);
    CHECK(world.Quiet[pinned] < free_space.SleepSteps);
    CHECK(simd::length(RotationVector(QuatMul(world.Poses[pinned].Orientation, QuatConjugate(settled)))) > 1e-3f);
}

TEST_CASE_FIXTURE(OnDevice, "a scene holding a pinned body steps to bit-identical state twice") {
    CheckReplay(context, [&](World &world) {
        AddGround(world);
        const Index box = world.AddShape(UnitBox);
        REQUIRE(world.AddBody({.Pose = At(float3{0, 1.5f, 0}),
                               .Shape = world.AddShape({.HalfExtents = {1.5f, 0.2f, 1.5f}, .Kind = ShapeBox}),
                               .Mass = {{.Mass = 0, .Inertia = {0, 40, 0}}}, .Friction = 0.5f}) != NoIndex);
        for (uint32_t i = 0; i < 3; ++i)
            REQUIRE(world.AddBody({.Pose = At(float3{0.2f * float(i) - 0.2f, 2.4f + 1.2f * float(i), 0.13f}), .Shape = box,
                                   .Mass = {CubeMass(200, 1)}, .Friction = 0.5f}) != NoIndex);
        Run(solver, world, 200);
    });
}

// A body made of several pieces, which KHR authors as a node with a motion and several collider descendants.
// A solid described in pieces behaves as the solid, and each piece is identified apart from its siblings.
namespace {
// A slab and four legs, authored where a modeller would put them, returning the body frame the compound resolves to.
constexpr float TableTop = 0.75f, TableHalf = 0.05f, LegHalf = 0.05f;

Index AddTable(World &world, Pose &frame) {
    const float leg_high = (TableTop - TableHalf) / 2;
    std::vector<Index> parts{world.AddShape({.HalfExtents = {1, TableHalf, 0.6f}, .Kind = ShapeBox, .Local = At(float3{0, TableTop, 0})})};
    for (const float x : {-0.9f, 0.9f})
        for (const float z : {-0.5f, 0.5f})
            parts.push_back(world.AddShape({.HalfExtents = {LegHalf, leg_high, LegHalf}, .Kind = ShapeBox, .Local = At(float3{x, leg_high, z})}));
    return world.AddCompound(parts, &frame);
}
} // namespace

TEST_CASE_FIXTURE(OnDevice, "a compound of one child is that child") {
    // The narrowphase composes the body pose with the leaf's own Local, the same meaning a lone shape's Local already has.
    // A compound of one is the shape it holds, compared here against that shape with no local pose.
    // A compound moves the body frame onto the centre of mass and a lone shape does not, so only the geometry is compared.
    const Pose local = At(float3{0.2f, -0.15f, 0.1f}, QuatFromRotationVector(float3{0.3f, 0.5f, -0.2f}));
    constexpr Shape Piece{.HalfExtents = {0.3f, 0.4f, 0.5f}, .Kind = ShapeBox};

    World alone{context}, wrapped{context};
    AddGround(alone, {.Friction = 0.5f});
    AddGround(wrapped, {.Friction = 0.5f});
    const Index bare = alone.AddShape(Piece);
    const Index inner = wrapped.AddShape({.HalfExtents = Piece.HalfExtents, .Kind = ShapeBox, .Local = local});
    Pose frame{};
    const Index compound = wrapped.AddCompound(std::vector<Index>{inner}, &frame);
    REQUIRE(bare != NoIndex);
    REQUIRE(compound != NoIndex);
    // A single child's centre of mass is its own, so the frame lands on that child's Local exactly.
    CHECK(simd::distance(frame.Position, local.Position) < 1e-6f);
    // The compound and the child weigh the same, the two frames being principal frames of one solid.
    // The three moments match in whatever order the diagonalization produced them.
    const BodyMass weighed = MassProperties(Piece, 1000);
    const BodyMass built = MassOf(wrapped, compound, 1000);
    CHECK(built.InvMass == doctest::Approx(weighed.InvMass).epsilon(1e-5));
    std::vector<float> mine{built.InvInertiaLocal.x, built.InvInertiaLocal.y, built.InvInertiaLocal.z};
    std::vector<float> theirs{weighed.InvInertiaLocal.x, weighed.InvInertiaLocal.y, weighed.InvInertiaLocal.z};
    std::ranges::sort(mine);
    std::ranges::sort(theirs);
    for (uint32_t axis = 0; axis < 3; ++axis) CHECK(mine[axis] == doctest::Approx(theirs[axis]).epsilon(1e-5));

    // Neither needs an authored mass: one is centred by construction and the other by the compound.
    const Pose start = At(float3{0, 2, 0}, QuatFromRotationVector(float3{0.1f, 0.2f, 0.35f}));
    const Index one = alone.AddBody({.Pose = ComposePose(start, local), .Shape = bare, .Friction = 0.5f});
    const Index other = wrapped.AddBody({.Pose = ComposePose(start, frame), .Shape = compound, .Friction = 0.5f});
    REQUIRE(one != NoIndex);
    REQUIRE(other != NoIndex);
    CHECK(wrapped.OffsetsWithoutMass == 0);

    float worst = 0;
    for (uint32_t step = 0; step < 200; ++step) {
        solver.Step(alone, {});
        solver.Step(wrapped, {});
        CheckManifolds(alone);
        CheckManifolds(wrapped);
        // Compared through the geometry rather than the origin, since the compound moved its own.
        const Pose mine = ComposePose(alone.Poses[one], alone.Shapes[bare].Local);
        const Pose theirs = ComposePose(wrapped.Poses[other], wrapped.Shapes[ChildOf(wrapped.Shapes[compound], 0)].Local);
        worst = std::max(worst, float(simd::distance(mine.Position, theirs.Position)));
        worst = std::max(worst, simd::length(RotationVector(QuatMul(mine.Orientation, QuatConjugate(theirs.Orientation)))));
    }
    CHECK(worst < 1e-5f);
    CHECK(ActiveContacts(alone, one) == ActiveContacts(wrapped, other)); // one leaf, so one manifold each
}

TEST_CASE_FIXTURE(OnDevice, "a solid described in two halves rests and slides where the solid does") {
    World whole{context}, split{context};
    AddGround(whole, {.Friction = 0.5f});
    AddGround(split, {.Friction = 0.5f});
    const Index one = whole.AddShape(UnitBox);
    std::vector<Index> halves;
    for (const float side : {-1.f, 1.f})
        halves.push_back(split.AddShape({.HalfExtents = {Half / 2, Half, Half}, .Kind = ShapeBox, .Local = At(float3{side * Half / 2, 0, 0})}));
    const Index pieces = split.AddCompound(halves);
    REQUIRE(pieces != NoIndex);

    // Same mass and same principal inertia, which makes the rest of this a fair comparison.
    const auto solid = MassOf(whole, one, 1000);
    const auto built = MassOf(split, pieces, 1000);
    CHECK(built.InvMass == doctest::Approx(solid.InvMass).epsilon(1e-5));
    for (uint32_t axis = 0; axis < 3; ++axis)
        CHECK(built.InvInertiaLocal[axis] == doctest::Approx(solid.InvInertiaLocal[axis]).epsilon(1e-5));

    constexpr float Speed = 2;
    const Index box = whole.AddBody({.Pose = At(float3{0, Half, 0}), .Velocity = {.Linear = {Speed, 0, 0}}, .Shape = one, .Friction = 0.5f});
    const Index built_box = split.AddBody({.Pose = At(float3{0, Half, 0}), .Velocity = {.Linear = {Speed, 0, 0}}, .Shape = pieces, .Friction = 0.5f});
    REQUIRE(box != NoIndex);
    REQUIRE(built_box != NoIndex);

    solver.Step(whole, {});
    solver.Step(split, {});
    // Step one, where the two descriptions give the same geometry.
    // The same normal at the same separation, with each half's rows keyed by the leaf that made them.
    std::set<uint32_t> leaves;
    for (const Contact &contact : Slots(split, built_box)) {
        if (!contact.Active) continue;
        leaves.insert(OwnChild(contact.Children));
        CHECK(contact.Normal.y == doctest::Approx(1));
        CHECK(contact.C0.x == doctest::Approx(Slots(whole, box)[0].C0.x));
    }
    CHECK(leaves == std::set<uint32_t>{0, 1}); // both halves support it, and they are distinguished
    CHECK(ActiveContacts(whole, box) == 4);
    CHECK(ActiveContacts(split, built_box) == 6); // four per leaf, less the two corners the shared edge would duplicate

    // They slide to a stop at mu g together, since the load and the friction are the same however many pieces carry them.
    Run(solver, whole, 60);
    Run(solver, split, 60);
    CheckResting(float(whole.Poses[box].Position.y));
    CheckResting(float(split.Poses[built_box].Position.y));
    CHECK(simd::length(whole.Velocities[box].Linear) < 1e-3f);
    CHECK(simd::length(split.Velocities[built_box].Linear) < 1e-3f);
    // Coulomb puts the stopping distance at v^2 / (2 mu g), and both cover the same.
    const float expected = Speed * Speed / (2 * 0.5f * Gravity);
    CHECK(float(whole.Poses[box].Position.x) == doctest::Approx(expected).epsilon(0.05));
    CHECK(float(split.Poses[built_box].Position.x) == doctest::Approx(float(whole.Poses[box].Position.x)).epsilon(0.02));
}

TEST_CASE_FIXTURE(OneWorld, "a table stands on its four legs at the height its legs say") {
    AddGround(world, {.Friction = 0.5f});
    Pose frame{};
    const Index table = AddTable(world, frame);
    REQUIRE(table != NoIndex);
    // Placed at the frame the compound came back with, which puts every piece where it was authored.
    const Index body = world.AddBody({.Pose = At(frame.Position + float3{0, 1e-3f, 0}, frame.Orientation), .Shape = table, .Friction = 0.5f});
    REQUIRE(body != NoIndex);
    CHECK(world.OffsetsWithoutMass == 0); // a compound is centred on the body's origin by construction

    Run(solver, world, 200);
    const Pose pose = world.Poses[body];
    // It stands where the legs put it, the frame's own height above the plane.
    CHECK(float(pose.Position.y) < float(frame.Position.y));
    CHECK(float(pose.Position.y) > float(frame.Position.y) - MaxPenetration);
    CHECK(simd::length(RotationVector(QuatMul(pose.Orientation, QuatConjugate(frame.Orientation)))) < 1e-4f);
    // On four leg manifolds alone, since the slab is nowhere near the ground.
    CHECK(ActiveContacts(world, body) == 4 * ManifoldPoints);
    const std::set<uint32_t> leaves = LeavesTouching(world, body);
    CHECK(leaves == std::set<uint32_t>{1, 2, 3, 4}); // the four legs, without the top
    CHECK(world.ContactRefusals[body] == 0);

    // `frame` maps the top back to where the host authored it.
    const Index top = ChildOf(world.Shapes[table], 0);
    REQUIRE(top != NoIndex);
    const Pose local = world.Shapes[top].Local;
    const float3 at = WorldPoint(pose, local.Position);
    CHECK(simd::distance(at, float3{0, TableTop, 0}) < MaxPenetration);
}

TEST_CASE_FIXTURE(OneWorld, "a dumbbell rolls on the axis its geometry gives it") {
    // Two balls on a rod along x, which rolls only about x, since the balls touch the plane and lie on that axis.
    // A push along z turns it about x alone.
    constexpr float Radius = 0.25f, Reach = 0.5f;
    AddGround(world, {.Friction = 0.5f});
    std::vector<Index> parts{world.AddShape({.HalfExtents = {Reach, 0, 0}, .Radius = 0.06f, .Kind = ShapeCapsule,
                                             .Local = At(float3{0, 0, 0}, QuatFromRotationVector(float3{0, 0, std::numbers::pi_v<float> / 2}))})};
    for (const float side : {-1.f, 1.f})
        parts.push_back(world.AddShape({.Radius = Radius, .Kind = ShapeSphere, .Local = At(float3{side * Reach, 0, 0})}));
    Pose frame{};
    const Index dumbbell = world.AddCompound(parts, &frame);
    REQUIRE(dumbbell != NoIndex);

    const Index body = world.AddBody({.Pose = At(float3{0, Radius, 0}), .Velocity = {.Linear = {0, 0, 1}}, .Shape = dumbbell, .Friction = 0.5f});
    REQUIRE(body != NoIndex);
    Run(solver, world, 120);

    // Rolling rather than sliding: the balls stay on the plane, it keeps moving, and it turns about x.
    const Pose pose = world.Poses[body];
    CHECK(float(pose.Position.y) == doctest::Approx(Radius).epsilon(2e-3));
    CHECK(float(pose.Position.z) > 0.5f);
    CHECK(simd::length(world.Velocities[body].Linear) > 0.5f);
    const float3 spin = world.Velocities[body].Angular;
    CHECK(std::abs(spin.x) > 1);
    CHECK(std::abs(spin.y) < 0.05f * std::abs(spin.x));
    CHECK(std::abs(spin.z) < 0.05f * std::abs(spin.x));
    // A rolling contact is one ball on the plane each, and the rod never touches it.
    const std::set<uint32_t> leaves = LeavesTouching(world, body);
    CHECK(leaves == std::set<uint32_t>{1, 2});
}

TEST_CASE_FIXTURE(OneWorld, "eight leaves on a plane each get a manifold of their own") {
    // One manifold per leaf is eight of the ten slots a run holds, the most a single partner can take from a compound.
    // Nothing is refused, and it rests level on all eight.
    constexpr float Cube = 0.2f;
    AddGround(world, {.Friction = 0.5f});
    std::vector<Index> cubes;
    for (uint32_t i = 0; i < ChildrenPerCompound; ++i)
        cubes.push_back(world.AddShape({.HalfExtents = {Cube, Cube, Cube}, .Kind = ShapeBox, .Local = At(float3{2.5f * Cube * (float(i) - 3.5f), 0, 0})}));
    Pose frame{};
    const Index row = world.AddCompound(cubes, &frame);
    REQUIRE(row != NoIndex);
    const Index body = world.AddBody({.Pose = At(frame.Position + float3{0, Cube + 1e-3f, 0}), .Shape = row, .Friction = 0.5f});
    REQUIRE(body != NoIndex);

    Run(solver, world, 200);
    CHECK(ActiveContacts(world, body) == ChildrenPerCompound * ManifoldPoints); // four points per leaf, eight leaves
    CHECK(world.ContactRefusals[body] == 0);
    const Pose pose = world.Poses[body];
    CheckResting(float(pose.Position.y) + Half - Cube);
    CHECK(simd::length(RotationVector(pose.Orientation)) < 1e-4f); // level rather than standing on a few of them
    const std::set<uint32_t> leaves = LeavesTouching(world, body);
    CHECK(leaves.size() == ChildrenPerCompound);
}

TEST_CASE_FIXTURE(OnDevice, "the join two coplanar siblings share is not a wall") {
    // Two boxes edge to edge make a floor with an internal face down the middle.
    // AddCompound marks that face buried and the narrowphase drops any manifold on it, with two separate bodies as the control.
    constexpr float FloorHalf = 2.5f, FloorTop = 0.25f, Speed = 2, Resting = FloorTop + Half;
    const auto slide = [&](World &world) {
        const Index box = world.AddBody({.Pose = At(float3{-Half - 0.05f, Resting, 0}), .Velocity = {.Linear = {Speed, 0, 0}},
                                         .Shape = world.AddShape(UnitBox), .Density = 1000, .Friction = 0.5f});
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
    // Each half has exactly the one face of itself the other covers.
    CHECK(InternalFaces(jointed.Shapes[ChildOf(jointed.Shapes[floor], 0)]) == 1u << BoxFaceIndex(0, true));
    CHECK(InternalFaces(jointed.Shapes[ChildOf(jointed.Shapes[floor], 1)]) == 1u << BoxFaceIndex(0, false));
    jointed.AddBody({.Pose = At(frame.Position), .Shape = floor, .Density = 0, .Friction = 0.5f});
    for (const float side : {-1.f, 1.f})
        separate.AddBody({.Pose = At(float3{side * FloorHalf, 0, 0}),
                          .Shape = separate.AddShape({.HalfExtents = {FloorHalf, FloorTop, FloorHalf}, .Kind = ShapeBox}),
                          .Density = 0, .Friction = 0.5f});
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

    // The two halves as separate bodies climb the join and stop against it.
    // That measures the effect the rule removes, and the effect World::WeldStatic removes for immovable bodies.
    CHECK(two_kick > 5e-2f);
    CHECK(float(separate.Poses[over_two].Position.x) < -Half); // never crossed
    // As one compound it crosses, with a worst excursion a quarter of theirs.
    // The remaining kick is the engine's own at a box join, so the bound is where the measurement puts it rather than at zero.
    CHECK(kick < 0.3f * two_kick);
    CHECK(float(jointed.Poses[over_join].Position.x) > -Half);
    CHECK(float(jointed.Poses[over_join].Position.y) == doctest::Approx(float(seamless.Poses[over_none].Position.y)).epsilon(1e-3));
}

TEST_CASE_FIXTURE(OnDevice, "a scene holding a compound steps to bit-identical state twice") {
    CheckReplay(context, [&](World &world) {
        AddGround(world, {.Friction = 0.5f});
        Pose frame{};
        const Index table = AddTable(world, frame);
        REQUIRE(table != NoIndex);
        REQUIRE(world.AddBody({.Pose = At(frame.Position + float3{0, 1e-3f, 0}, frame.Orientation), .Shape = table, .Friction = 0.5f}) != NoIndex);
        const Index box = world.AddShape(UnitBox);
        for (uint32_t i = 0; i < 3; ++i)
            REQUIRE(world.AddBody({.Pose = At(float3{0.13f * float(i), TableTop + TableHalf + Half + 1.1f * float(i), 0.07f},
                                              QuatFromRotationVector(float3{0.05f, 0.3f * float(i), 0})),
                                   .Shape = box, .Friction = 0.5f}) != NoIndex);
        Run(solver, world, 200);
    });
}
