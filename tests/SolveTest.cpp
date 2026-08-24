// The vertical slice: a box, gravity, a plane, and the whole pipeline on the GPU. These are the
// answers that can be checked without trusting the solver - a closed form for free flight, a resting
// height that is a property of the geometry, and two identical runs agreeing to the bit.

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

namespace {
constexpr float Half = 0.5f;
// A box at rest settles slightly inside the surface rather than exactly on it: contacts engage a
// margin early and one stabilization pass per step removes only part of the accumulated error, so the
// depth it lands at depends on how hard the landing was. What has to hold is that it is bounded.
constexpr float MaxPenetration = 4 * StepSettings{}.ContactMargin;

void CheckResting(float height) {
    CHECK(height < Half);
    CHECK(height > Half - MaxPenetration);
}
constexpr Shape UnitBox{.HalfExtents = {Half, Half, Half}, .Kind = ShapeBox};
constexpr Shape GroundPlane{.Normal = {0, 1, 0}, .Offset = 0, .Kind = ShapePlane};

// A box at `height` over a ground plane, or over nothing when `ground` is false.
Index DropBox(World &world, float height, bool ground, float friction = 0.5f) {
    const auto box = world.AddShape(UnitBox);
    if (ground) world.AddBody({.Shape = world.AddShape(GroundPlane), .Friction = friction});
    return world.AddBody({.Pose = {.Position = {0, height, 0}, .Orientation = {0, 0, 0, 1}}, .Shape = box, .Friction = friction});
}

// Gravity tilted by `slope` about z, which loads a body exactly as standing it on a ramp of that angle
// would, and leaves the ground plane flat so the contact is the same one every other test here uses.
StepSettings Tilted(float slope) {
    const float gravity = std::abs(StepSettings{}.Gravity.y);
    return {.Gravity = {gravity * std::sin(slope), -gravity * std::cos(slope), 0}};
}

// How far a shape's geometry reaches from its own origin - the scale the narrowphase resolves
// geometry at, and so the scale at which two pieces of it stop being distinguishable.
float ShapeReach(const World &world, Index shape) {
    if (shape == NoIndex) return 0;
    const Shape &it = world.Shapes[shape];
    float reach = it.Radius + std::max({std::abs(it.HalfExtents.x), std::abs(it.HalfExtents.y), std::abs(it.HalfExtents.z)});
    for (uint32_t i = 0; i < it.VertexCount; ++i) reach = std::max(reach, simd::length(world.ShapeVertices[it.FirstVertex + i]));
    return reach;
}

// No two rows may name the same piece of geometry. Two contact points closer together than the scale
// the narrowphase resolves geometry at are one contact written twice - two duals and two penalties at
// one spot, which shows up as ringing and as a slot budget that will not fit rather than as a wrong
// position. Checked over every body's own run, between rows naming the same partner and sub-shape.
void CheckManifolds(const World &world) {
    for (uint32_t body = 0; body < world.BodyCount(); ++body) {
        const auto slots = world.Contacts.All().subspan(body * ContactsPerBody, ContactsPerBody);
        const auto at = [&world](const Contact &c, bool side) {
            const Pose &pose = world.Poses[side ? c.BodyA : c.BodyB];
            return pose.Position + Rotate(pose.Orientation, side ? c.AnchorA : c.AnchorB);
        };
        for (uint32_t i = 0; i < ContactsPerBody; ++i) {
            if (!slots[i].Active) continue;
            for (uint32_t j = i + 1; j < ContactsPerBody; ++j) {
                if (!slots[j].Active || slots[j].BodyB != slots[i].BodyB || slots[j].SubShape != slots[i].SubShape) continue;
                // The scale WeldManifold welds at, taken the same way - the coarser of the pair's two
                // shapes, so the check is never stricter than the kernel.
                const float apart = 1e-3f * std::max(ShapeReach(world, world.BodyShapes[body]),
                                                     ShapeReach(world, world.BodyShapes[slots[i].BodyB])) + 1e-6f;
                const float here = simd::distance(at(slots[i], true), at(slots[j], true));
                const float there = simd::distance(at(slots[i], false), at(slots[j], false));
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

uint32_t ActiveContacts(const World &world) {
    uint32_t live = 0;
    for (uint32_t slot = 0; slot < world.Contacts.Capacity; ++slot) live += world.Contacts[slot].Active ? 1 : 0;
    return live;
}

uint32_t ActiveContacts(const World &world, Index body) {
    uint32_t live = 0;
    for (uint32_t slot = 0; slot < ContactsPerBody; ++slot) live += world.Contacts[body * ContactsPerBody + slot].Active ? 1 : 0;
    return live;
}

// Every contact the world is holding, named the way warm starting names them. A set that changes under
// something settled is a dual thrown away, and a disturbance the solver did not ask for.
std::set<std::tuple<Index, Index, uint32_t>> ContactKeys(const World &world) {
    std::set<std::tuple<Index, Index, uint32_t>> keys;
    for (uint32_t slot = 0; slot < world.Contacts.Capacity; ++slot) {
        const auto &contact = world.Contacts[slot];
        if (contact.Active) keys.emplace(contact.BodyA, contact.BodyB, contact.Feature);
    }
    return keys;
}

// How many of each kind a body reported over the step just taken.
uint32_t Reported(const World &world, Index body, ContactEventKind kind) {
    uint32_t found = 0;
    for (uint32_t i = 0; i < world.ContactEventCounts[body]; ++i)
        found += world.ContactEvents[body * EventsPerBody + i].Kind == kind ? 1 : 0;
    return found;
}

void Run(Solver &solver, World &world, uint32_t steps, const StepSettings &settings = {}) {
    for (uint32_t step = 0; step < steps; ++step) {
        solver.Step(world, settings);
        CheckManifolds(world);
    }
}
} // namespace

TEST_CASE("a context outlives the worlds and solvers built on it") {
    // Both hand the queue a residency set and a queue holds thirty-two of those ever, so without
    // giving them back a context that has built that many stops making further buffers resident and
    // the next step never returns. Forty rounds is past the bound whichever of the two is counted.
    //
    // It fails by hanging rather than by asserting, since a step whose buffers are not resident never
    // signals its event. If this one ever stops returning, that is the defect and not a flake.
    const mtl::Context context;
    Solver outlives{context}; // and one that spans every round, since that is the ordinary shape
    for (uint32_t round = 0; round < 40; ++round) {
        CAPTURE(round);
        Solver solver{context};
        World world{context};
        const auto box = DropBox(world, 1, true);
        Run(solver, world, 120);
        CheckResting(world.Poses[box].Position.y);
        // The long-lived one still reaches the same world, which says the sets that went back were its.
        Run(outlives, world, 1);
        CheckResting(world.Poses[box].Position.y);
    }
}

TEST_CASE("free fall matches the closed form of the discrete step") {
    const mtl::Context context;
    Solver solver{context};
    World world{context};
    const auto box = DropBox(world, 0, false);

    // Implicit Euler over n steps of h puts a body released from rest at h^2 g n (n + 1) / 2, the
    // triangular sum of the velocity it picks up. That tends to the continuous g t^2 / 2 as h shrinks,
    // and this checks the discrete answer exactly rather than the limit it is heading for.
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

TEST_CASE("a box dropped flat comes to rest on the plane") {
    const mtl::Context context;
    Solver solver{context};
    World world{context};
    const auto box = DropBox(world, 2, true);

    Run(solver, world, 180); // three seconds, long after it should have settled
    const auto &pose = world.Poses[box];
    const auto &velocity = world.Velocities[box];

    // Resting on the plane means the lower face sits on it, so the center is one half-extent up.
    CheckResting(pose.Position.y);
    CHECK(std::abs(velocity.Linear.y) < 1e-3f);
    CHECK(simd::length(velocity.Angular) < 1e-3f);

    // Dropped level onto a level plane, nothing asymmetric ever acts on it.
    CHECK(std::abs(pose.Position.x) < 1e-4f);
    CHECK(std::abs(pose.Position.z) < 1e-4f);
    CHECK(simd::length(RotationVector(pose.Orientation)) < 1e-3f);
}

TEST_CASE("the resting box holds its four bottom corners in contact") {
    const mtl::Context context;
    Solver solver{context};
    World world{context};
    const auto box = DropBox(world, 1, true);
    Run(solver, world, 180);

    const auto slots = world.Contacts.All().subspan(box * ContactsPerBody, ContactsPerBody);
    const auto active = std::ranges::count_if(slots, [](const Contact &c) { return c.Active; });
    CHECK(active == 4); // a level box touches with exactly its bottom face

    // The corners carry the weight between them, and each is pushing rather than pulling: a contact's
    // normal force is negative by the reference's convention, and never positive.
    float total = 0;
    for (const auto &contact : slots) {
        if (!contact.Active) continue;
        CHECK(contact.AnchorA.y == doctest::Approx(-Half));
        CHECK(contact.Lambda[0] < 0);
        total -= contact.Lambda[0];
    }
    // At rest the duals have converged to the contact force, so the four of them carry the box's weight.
    CHECK(total == doctest::Approx(1000 * 9.81f).epsilon(0.01));
}

TEST_CASE("a stack of boxes holds itself up") {
    // No closed form for a stack, so this asserts what has to be true: the boxes stay in order, each
    // one sits exactly one box above the one below, and the whole thing is still.
    constexpr uint32_t Count = 5;
    const StepSettings settings{};
    const mtl::Context context;
    Solver solver{context};
    World world{context};
    const auto shape = world.AddShape(UnitBox);
    world.AddBody({.Shape = world.AddShape(GroundPlane)});
    std::vector<Index> stack;
    for (uint32_t i = 0; i < Count; ++i)
        stack.push_back(world.AddBody({.Pose = {.Position = {0, Half + 1.02f * float(i), 0}, .Orientation = {0, 0, 0, 1}}, .Shape = shape}));

    Run(solver, world, 600); // long enough that a stack which only looks settled has stopped looking it
    // The plane behaves as the box below the first one, its centre a half-extent under the surface.
    float below = -Half;
    for (uint32_t i = 0; i < Count; ++i) {
        CAPTURE(i);
        const float height = world.Poses[stack[i]].Position.y;
        // Resting on the one below means one box up, less the margin contacts engage at.
        CHECK(std::abs(height - (below + 1 - settings.ContactMargin)) < 1e-3f);
        CHECK(simd::length(world.Velocities[stack[i]].Linear) < 0.02f);
        below = height;
    }
}

TEST_CASE("a box lands on another box's edge and is held by an edge contact") {
    // Two cubes turned 45 degrees about perpendicular axes present crossed edges, one along z and one
    // along x. No face of either faces the way the contact pushes, so only the cross product of the two
    // edge directions describes it - the nine cross-product axes of the separating axis test. Half a
    // unit square's diagonal is what a cube turned 45 degrees reaches, so two of those stacked is where
    // the upper box has to come to rest.
    constexpr float Diagonal = 0.70710678f; // Half * sqrt(2)
    const mtl::Context context;
    Solver solver{context};
    World world{context};
    const auto shape = world.AddShape(UnitBox);
    const float quarter = std::atan(1.f); // pi / 4
    world.AddBody({.Pose = {.Position = {0, 0, 0}, .Orientation = QuatFromRotationVector(float3{0, 0, quarter})},
                   .Shape = shape,
                   .Density = 0}); // static, so it holds its pose and the upper box does the falling
    const auto box = world.AddBody(
        {.Pose = {.Position = {0, 2 * Diagonal + 0.25f, 0}, .Orientation = QuatFromRotationVector(float3{quarter, 0, 0})},
         .Shape = shape}
    );

    // Balanced on a crossing edge is unstable and it tips off soon enough, so what is checked is the
    // contact at the moment it forms and that it is what stops the fall.
    uint32_t contacts = 0;
    for (uint32_t step = 0; step < 60 && contacts == 0; ++step) {
        solver.Step(world);
        for (uint32_t slot = 0; slot < world.Contacts.Capacity; ++slot) {
            const auto &contact = world.Contacts[slot];
            if (!contact.Active) continue;
            ++contacts;
            CHECK(contact.Normal.y == doctest::Approx(1).epsilon(0.01)); // straight up out of the lower box
        }
    }
    CHECK(contacts == 1); // where two edges cross there is one point, not a face's worth
    // Contacts are found at the pose the step began from, so a falling box is first seen to touch up
    // to one step of its own motion past the height the edges actually meet at. That is the bound
    // here rather than a tolerance picked to fit: at this drop it arrives at about 2.2 m/s.
    const float step = 2.2f * StepSettings{}.DeltaTime;
    CHECK(world.Poses[box].Position.y < 2 * Diagonal + StepSettings{}.ContactMargin);
    CHECK(world.Poses[box].Position.y > 2 * Diagonal - step);

    Run(solver, world, 30);
    CHECK(world.Poses[box].Position.y > 2 * Diagonal - 0.05f); // held up, not fallen through
}

TEST_CASE("a bounce leaves at the fraction of its arrival speed restitution names") {
    // Restitution is defined on speeds, not heights: a body arriving at v leaves at e v. A discrete
    // step lands the contact wherever it falls, so the speeds either side of the bounce are the thing
    // to check and the rise afterwards is the sanity check on it.
    const float gravity = std::abs(StepSettings{}.Gravity.y);

    struct Bounce {
        float Arrived{}, Left{}, Rose{};
    };
    const auto drop = [&](float e) {
        const mtl::Context context;
        Solver solver{context};
        World world{context};
        const auto shape = world.AddShape(UnitBox);
        world.AddBody({.Shape = world.AddShape(GroundPlane), .Restitution = e});
        const auto box = world.AddBody(
            {.Pose = {.Position = {0, Half + 1, 0}, .Orientation = {0, 0, 0, 1}}, .Shape = shape, .Restitution = e}
        );

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
        // Fast enough that it should bounce at all, which is the threshold the step and gravity set.
        const StepSettings settings{};
        CHECK(bounce.Arrived > settings.BounceSpeedFactor * simd::length(settings.Gravity) * settings.DeltaTime);
        CHECK(bounce.Left == doctest::Approx(e * bounce.Arrived).epsilon(0.02));
        // And the rise is what leaving at that speed buys, less what the first step of it already used.
        CHECK(bounce.Rose == doctest::Approx(bounce.Left * bounce.Left / (2 * gravity)).epsilon(0.1));
    }
    // No restitution asked for, none given. Not exactly zero: a box settling onto the plane rebounds
    // off its own penetration by a few millimetres a second, which is three orders under a bounce.
    CHECK(drop(0).Left < 0.05f);
}

TEST_CASE("a body fired at a wall never ends up behind it") {
    // The whole of what a speculative contact is for: a reach covering a step of motion catches a body
    // that would otherwise be through the wall before anything looked. A mesh is where it matters most,
    // there being no back face to be pushed out of, so a body past one is past it for good.
    //
    // Fired level in free space, since the question is which side of the wall it ends on and gravity
    // would only slide it off the bottom while the answer was taken. RbpScenes bullet is the same shot
    // with the reach clamped alongside, which is what prices the trade.
    constexpr float Size = 0.05f, Thickness = 0.02f, From = -1;
    const StepSettings settings{.Gravity = {0, 0, 0}};

    // A wall of two triangles in the y-z plane, wound so its face looks back down the line of fire.
    const std::vector<float3> points{float3{0, -2, -2}, float3{0, -2, 2}, float3{0, 2, 2}, float3{0, 2, -2}};
    const std::vector<uint32_t> indices{0, 1, 2, 0, 2, 3};

    const auto fire = [&](float speed, bool meshed) {
        const mtl::Context context;
        Solver solver{context};
        World world{context};
        const Index wall = meshed ? world.AddMesh(points, indices)
                                  : world.AddShape({.HalfExtents = {Thickness / 2, 2, 2}, .Kind = ShapeBox});
        REQUIRE(wall != NoIndex);
        world.AddBody({.Shape = wall, .Density = 0});
        const auto shot = world.AddBody({.Pose = {.Position = {From, 0, 0}, .Orientation = {0, 0, 0, 1}},
                                         .Velocity = {.Linear = {speed, 0, 0}},
                                         .Shape = world.AddShape({.HalfExtents = {Size, Size, Size}, .Kind = ShapeBox})});
        float deepest = -INFINITY;
        // Long enough to carry it well past the wall at this speed if nothing stops it.
        for (uint32_t step = 0; step < uint32_t(3 / (speed * settings.DeltaTime)) + 60; ++step) {
            solver.Step(world, settings);
            // How far its leading face got past the wall's near one, which is the whole question.
            deepest = std::max(deepest, float(world.Poses[shot].Position.x) + Size - (meshed ? 0 : -Thickness / 2));
        }
        return deepest;
    };

    for (const float speed : {5.f, 20.f, 60.f, 120.f, 250.f, 500.f}) {
        CAPTURE(speed);
        // Half a metre a step at the slowest and eight at the fastest, against a two centimetre wall.
        REQUIRE(speed * settings.DeltaTime > Thickness);
        for (const bool meshed : {false, true}) {
            CAPTURE(meshed);
            // It stops at the margin every other contact rests at, and never a step's travel beyond it.
            CHECK(fire(speed, meshed) < MaxPenetration);
        }
    }
}

TEST_CASE("a bounce series keeps its ratio with speculation active") {
    // Restitution is a velocity pass after the solve rather than a row, since one displacement per step
    // cannot carry an approach and a rebound both. What that has to give is the definition: each
    // arrival leaves at e times the speed it came in at, bounce after bounce and not just the first,
    // and the series has to end rather than buzzing on the floor for ever.
    constexpr float e = 0.5f;
    const mtl::Context context;
    Solver solver{context};
    World world{context};
    const auto shape = world.AddShape(UnitBox);
    world.AddBody({.Shape = world.AddShape(GroundPlane), .Restitution = e});
    const auto box = world.AddBody(
        {.Pose = {.Position = {0, Half + 1, 0}, .Orientation = {0, 0, 0, 1}}, .Shape = shape, .Restitution = e}
    );

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
    CHECK(bounces >= 3); // it really did bounce repeatedly rather than stopping dead on the first
    CheckResting(world.Poses[box].Position.y);
    CHECK(simd::length(world.Velocities[box].Linear) < 1e-2f);
}

TEST_CASE("a contact across a gap holds its bodies apart and pushes nothing") {
    // A speculative contact exists - so it warm starts, and a listener hears about it before anything
    // touches - and applies no force until the step's motion would more than consume the gap it
    // measured.
    //
    // Thrown across the plane rather than dropped at it: the reach is one step of motion, so a body
    // falling straight at something spends the whole of its reach getting there and the gap is closed
    // the same step. Travelling sideways the reach comes from a speed that is closing nothing, so the
    // contact stands open for as long as gravity alone takes to bring the box down.
    const mtl::Context context;
    Solver solver{context};
    World world{context};
    const auto box = DropBox(world, Half + 0.05f, true);
    world.Velocities[box].Linear = {10, 0, 0};

    uint32_t apart = 0, touching = 0;
    for (uint32_t step = 0; step < 120; ++step) {
        solver.Step(world);
        const float gap = world.Poses[box].Position.y - Half;
        for (uint32_t slot = 0; slot < world.Contacts.Capacity; ++slot) {
            const auto &contact = world.Contacts[slot];
            if (!contact.Active) continue;
            if (gap <= StepSettings{}.ContactMargin) {
                ++touching;
                continue;
            }
            ++apart;
            CHECK(contact.Lambda[0] == 0); // apart, so there is nothing for it to be pushing
            CHECK(contact.C0[0] > StepSettings{}.ContactMargin); // and it is carrying the gap it measured
        }
    }
    CHECK(apart > 0); // there really were steps holding a contact across a gap, or nothing above ran
    CHECK(touching > 0); // and it did go on to land, rather than being held off the plane by a ghost
    CheckResting(world.Poses[box].Position.y);
}

TEST_CASE("a bouncy box still comes to rest") {
    // The threshold is what makes this true: below it an impact does not bounce, so a body settling
    // on its own jitter is not handed a new bounce every step and the series terminates.
    const mtl::Context context;
    Solver solver{context};
    World world{context};
    const auto shape = world.AddShape(UnitBox);
    world.AddBody({.Shape = world.AddShape(GroundPlane), .Restitution = 0.6f});
    const auto box = world.AddBody(
        {.Pose = {.Position = {0, Half + 1, 0}, .Orientation = {0, 0, 0, 1}}, .Shape = shape, .Restitution = 0.6f}
    );

    Run(solver, world, 900);
    CheckResting(world.Poses[box].Position.y);
    CHECK(simd::length(world.Velocities[box].Linear) < 1e-2f);
}

TEST_CASE("spheres rest where their radii put them") {
    // A sphere touches at a single point whatever it is touching, so every one of these has an exact
    // answer that is a sum of radii, less the margin contacts engage at.
    constexpr float Radius = 0.5f, Margin = StepSettings{}.ContactMargin;
    constexpr Shape Ball{.Radius = Radius, .Kind = ShapeSphere};

    SUBCASE("on a plane, at its radius") {
        const mtl::Context context;
        Solver solver{context};
        World world{context};
        const auto ball = world.AddShape(Ball);
        world.AddBody({.Shape = world.AddShape(GroundPlane)});
        const auto sphere = world.AddBody({.Pose = {.Position = {0, 1.5f, 0}, .Orientation = {0, 0, 0, 1}}, .Shape = ball});

        Run(solver, world, 300);
        CHECK(std::abs(world.Poses[sphere].Position.y - (Radius - Margin)) < 2e-3f);
        CHECK(simd::length(world.Velocities[sphere].Linear) < 1e-2f);
    }

    SUBCASE("on another sphere, at the sum of theirs") {
        const mtl::Context context;
        Solver solver{context};
        World world{context};
        const auto ball = world.AddShape(Ball);
        world.AddBody({.Shape = world.AddShape(GroundPlane)});
        // The lower one static: a sphere balanced on a sphere is not an equilibrium worth asking for.
        world.AddBody({.Pose = {.Position = {0, Radius, 0}, .Orientation = {0, 0, 0, 1}}, .Shape = ball, .Density = 0});
        const auto top = world.AddBody({.Pose = {.Position = {0, 3 * Radius + 0.2f, 0}, .Orientation = {0, 0, 0, 1}}, .Shape = ball});

        Run(solver, world, 300);
        CHECK(std::abs(world.Poses[top].Position.y - (3 * Radius - Margin)) < 2e-3f);
    }

    SUBCASE("on a box, at its radius above the face") {
        const mtl::Context context;
        Solver solver{context};
        World world{context};
        const auto ball = world.AddShape(Ball);
        const auto box = world.AddShape(UnitBox);
        world.AddBody({.Shape = world.AddShape(GroundPlane)});
        world.AddBody({.Pose = {.Position = {0, Half, 0}, .Orientation = {0, 0, 0, 1}}, .Shape = box, .Density = 0});
        const auto sphere = world.AddBody({.Pose = {.Position = {0, 2 * Half + Radius + 0.2f, 0}, .Orientation = {0, 0, 0, 1}}, .Shape = ball});

        Run(solver, world, 300);
        CHECK(std::abs(world.Poses[sphere].Position.y - (2 * Half + Radius - Margin)) < 2e-3f);
        CHECK(std::abs(world.Poses[sphere].Position.x) < 1e-2f); // it stayed on top rather than rolling off
    }
}

TEST_CASE("a sphere rolls down a slope at the rate rolling without slipping implies") {
    // A ball that rolls rather than slides carries some of its energy as spin, so it accelerates at
    // g sin(slope) / (1 + 2/5) rather than g sin(slope) - the 2/5 being the sphere's own inertia
    // coefficient. Friction has to be able to supply that, which needs mu >= (2/7) tan(slope), and it
    // is what makes the sphere turn at all: the contact torque is the only thing acting off centre.
    constexpr float Radius = 0.5f, Mu = 0.5f, Slope = 0.3f;
    const float gravity = std::abs(StepSettings{}.Gravity.y);
    const float expected = 5.f / 7 * gravity * std::sin(Slope);
    REQUIRE(Mu >= 2.f / 7 * std::tan(Slope)); // otherwise it slips and the closed form is a different one

    const auto settings = Tilted(Slope);
    const mtl::Context context;
    Solver solver{context};
    World world{context};
    const auto ball = world.AddShape(Shape{.Radius = Radius, .Kind = ShapeSphere});
    world.AddBody({.Shape = world.AddShape(GroundPlane), .Friction = Mu});
    const auto sphere = world.AddBody(
        {.Pose = {.Position = {0, Radius, 0}, .Orientation = {0, 0, 0, 1}}, .Shape = ball, .Friction = Mu}
    );

    Run(solver, world, 30, settings); // settle onto the plane before timing anything
    const float was = world.Velocities[sphere].Linear.x;
    constexpr uint32_t Steps = 60;
    Run(solver, world, Steps, settings);
    const float elapsed = Steps * settings.DeltaTime;
    CHECK((world.Velocities[sphere].Linear.x - was) / elapsed == doctest::Approx(expected).epsilon(0.05));

    // Rolling without slipping: the surface speed at the contact matches the centre's. Gravity tilts
    // towards +x and the contact is underneath, so it turns about -z.
    const float speed = world.Velocities[sphere].Linear.x;
    CHECK(world.Velocities[sphere].Angular.z == doctest::Approx(-speed / Radius).epsilon(0.05));
}

TEST_CASE("capsules rest where their radius puts them, on whatever they land on") {
    // A capsule is every point within Radius of a segment, so it rests that radius off whatever the
    // segment ends up nearest - a sum of radii again. What is new is the count: a capsule lying along
    // what it touches is held by two contacts, one meeting it at a point by one.
    constexpr float Radius = 0.25f, HalfLength = 0.5f, Margin = StepSettings{}.ContactMargin;
    constexpr Shape Pill{.HalfExtents = {0, HalfLength, 0}, .Radius = Radius, .Kind = ShapeCapsule};
    const float quarter = 2 * std::atan(1.f); // the capsule's length runs along its own y, so this lays it down

    SUBCASE("lying on a plane, on two contacts") {
        const mtl::Context context;
        Solver solver{context};
        World world{context};
        const auto pill = world.AddShape(Pill);
        world.AddBody({.Shape = world.AddShape(GroundPlane)});
        const auto capsule = world.AddBody(
            {.Pose = {.Position = {0, 0.6f, 0}, .Orientation = QuatFromRotationVector(float3{0, 0, quarter})}, .Shape = pill}
        );

        Run(solver, world, 300);
        CHECK(std::abs(world.Poses[capsule].Position.y - (Radius - Margin)) < 2e-3f);
        CHECK(ActiveContacts(world) == 2); // one under each end of its length
        CHECK(simd::length(world.Velocities[capsule].Linear) < 1e-2f);
    }

    SUBCASE("standing on a plane, on one") {
        const mtl::Context context;
        Solver solver{context};
        World world{context};
        const auto pill = world.AddShape(Pill);
        world.AddBody({.Shape = world.AddShape(GroundPlane)});
        const auto capsule = world.AddBody({.Pose = {.Position = {0, 1, 0}, .Orientation = {0, 0, 0, 1}}, .Shape = pill});

        Run(solver, world, 300);
        CHECK(std::abs(world.Poses[capsule].Position.y - (HalfLength + Radius - Margin)) < 2e-3f);
        CHECK(ActiveContacts(world) == 1); // one cap, and a cap is a sphere
    }

    SUBCASE("across a box, with both ends out over nothing") {
        // The case its own ends cannot find: the capsule meets the box between them, and sampling
        // only the ends would walk it straight through.
        constexpr Shape Long{.HalfExtents = {0, 2, 0}, .Radius = 0.1f, .Kind = ShapeCapsule};
        const mtl::Context context;
        Solver solver{context};
        World world{context};
        const auto pill = world.AddShape(Long);
        const auto box = world.AddShape(UnitBox);
        world.AddBody({.Shape = world.AddShape(GroundPlane)});
        world.AddBody({.Pose = {.Position = {0, Half, 0}, .Orientation = {0, 0, 0, 1}}, .Shape = box, .Density = 0});
        const auto capsule = world.AddBody(
            {.Pose = {.Position = {0, 1.3f, 0}, .Orientation = QuatFromRotationVector(float3{0, 0, quarter})}, .Shape = pill}
        );

        Run(solver, world, 300);
        CHECK(std::abs(world.Poses[capsule].Position.y - (2 * Half + 0.1f - Margin)) < 2e-3f);
        CHECK(ActiveContacts(world) == 2); // where its length crosses the two edges of the box's top face

        // And the pair keeps its identity, which is why the two are named by the box faces that bounded
        // them rather than by where along the capsule they landed.
        const auto settled = ContactKeys(world);
        for (uint32_t step = 0; step < 300; ++step) {
            solver.Step(world);
            CAPTURE(step);
            REQUIRE(ContactKeys(world) == settled);
        }
    }
}

TEST_CASE("a capsule on a slope along its own length holds still") {
    // Lying along the slope a capsule cannot roll down it, so friction alone holds it - the case for
    // the other half of the anchor rule. A round contact does not inherit static friction's anchors,
    // and this says that costs a capsule which is *not* rolling nothing.
    constexpr float Radius = 0.25f, Mu = 0.5f, Slope = 0.3f;
    REQUIRE(std::tan(Slope) < Mu); // inside the cone, so it must not move at all
    constexpr Shape Pill{.HalfExtents = {0, 0.5f, 0}, .Radius = Radius, .Kind = ShapeCapsule};

    const auto settings = Tilted(Slope);
    const mtl::Context context;
    Solver solver{context};
    World world{context};
    const auto pill = world.AddShape(Pill);
    world.AddBody({.Shape = world.AddShape(GroundPlane), .Friction = Mu});
    const auto capsule = world.AddBody({.Pose = {.Position = {0, Radius, 0},
                                                 .Orientation = QuatFromRotationVector(float3{0, 0, 2 * std::atan(1.f)})},
                                        .Shape = pill,
                                        .Friction = Mu});

    Run(solver, world, 30, settings);
    const float from = world.Poses[capsule].Position.x;
    Run(solver, world, 900, settings); // fifteen seconds is long enough for a creep to show
    CHECK(std::abs(world.Poses[capsule].Position.x - from) < 1e-3f);
    CHECK(simd::length(world.Velocities[capsule].Linear) < 1e-3f);
}

TEST_CASE("a capsule rolls down a slope at the rate its own inertia implies") {
    // Same law as the sphere, with the capsule's own inertia in place of 2/5 m r^2: it accelerates at
    // g sin(slope) / (1 + I / m r^2), where I is what MassProperties works out about its length. A
    // capsule rolls across its length, so it lies along z and travels in x.
    constexpr float Radius = 0.25f, HalfLength = 0.5f, Mu = 0.5f, Slope = 0.3f;
    constexpr Shape Pill{.HalfExtents = {0, HalfLength, 0}, .Radius = Radius, .Kind = ShapeCapsule};
    const float gravity = std::abs(StepSettings{}.Gravity.y);
    const auto properties = MassProperties(Pill, 1000);
    const float ratio = (1 / properties.InvInertiaLocal[1]) * properties.InvMass / (Radius * Radius);
    const float expected = gravity * std::sin(Slope) / (1 + ratio);
    REQUIRE(Mu >= ratio / (1 + ratio) * std::tan(Slope)); // enough friction that it rolls rather than slips

    const auto settings = Tilted(Slope);
    const mtl::Context context;
    Solver solver{context};
    World world{context};
    const auto pill = world.AddShape(Pill);
    world.AddBody({.Shape = world.AddShape(GroundPlane), .Friction = Mu});
    const auto capsule = world.AddBody({.Pose = {.Position = {0, Radius, 0},
                                                 .Orientation = QuatFromRotationVector(float3{2 * std::atan(1.f), 0, 0})},
                                        .Shape = pill,
                                        .Friction = Mu});

    Run(solver, world, 30, settings);
    const float was = world.Velocities[capsule].Linear.x;
    constexpr uint32_t Steps = 60;
    Run(solver, world, Steps, settings);
    const float elapsed = Steps * settings.DeltaTime;
    CHECK((world.Velocities[capsule].Linear.x - was) / elapsed == doctest::Approx(expected).epsilon(0.05));
    CHECK(world.Velocities[capsule].Angular.z == doctest::Approx(-world.Velocities[capsule].Linear.x / Radius).epsilon(0.05));
    CHECK(std::abs(world.Poses[capsule].Position.z) < 1e-2f); // it rolled straight rather than veering
}

// A body on a pivot: a shapeless zero-mass body to hang it from, and the arm jointed to it at the
// origin with its center of mass `distance` away along x.
struct Pendulum {
    World &Bodies;
    Index Arm;
    float Inertia; // about the pivot, which is the closed form's I and not the body's own
    float Mass;
};

Pendulum MakePendulum(World &world, float distance, JointDesc joint = {}) {
    const auto shape = world.AddShape(UnitBox);
    const auto pivot = world.AddBody({}); // no shape, so no mass and no contacts: a bare pivot
    const auto arm = world.AddBody({.Pose = {.Position = {distance, 0, 0}, .Orientation = {0, 0, 0, 1}}, .Shape = shape});
    joint.BodyA = arm;
    joint.BodyB = pivot;
    world.AddJoint(joint);
    const float mass = 1 / world.Masses[arm].InvMass;
    // Parallel axis: about the pivot rather than about its own center.
    return {world, arm, 1 / world.Masses[arm].InvInertiaLocal[2] + mass * distance * distance, mass};
}

TEST_CASE("a joint holds its two anchor points together") {
    // The whole of what a ball joint claims: wherever the bodies get to, the point they were pinned at
    // is one point still. Measured as the distance between where each body says that point now is.
    constexpr float Distance = 1;
    const mtl::Context context;
    Solver solver{context};
    World world{context};
    const auto pendulum = MakePendulum(world, Distance);

    float worst = 0;
    for (uint32_t step = 0; step < 600; ++step) {
        solver.Step(world);
        const auto &pose = world.Poses[pendulum.Arm];
        const float3 held = pose.Position + Rotate(pose.Orientation, world.Joints[0].AnchorA);
        worst = std::max(worst, simd::length(held)); // the other body is static at the origin
    }
    CHECK(worst < 5e-3f); // a fifth of a percent of the arm it is holding up
}

TEST_CASE("a pendulum converges on the speed energy says it reaches") {
    // Released horizontal, its center of mass falls by the arm and all of that becomes rotation:
    // 1/2 I w^2 = m g d. Backwards Euler dissipates, so this is a convergence check rather than an
    // equality: at 1/60 it reaches about 93% of the closed form, and the shortfall is the integrator's
    // rather than the solve's - more iterations do not touch it and a shorter step does.
    constexpr float Distance = 1;
    const float gravity = std::abs(StepSettings{}.Gravity.y);

    const auto fastest = [&](float rate) {
        const mtl::Context context;
        Solver solver{context};
        World world{context};
        const auto pendulum = MakePendulum(world, Distance);
        const StepSettings settings{.DeltaTime = 1 / rate};
        float peak = 0;
        for (uint32_t step = 0; step < uint32_t(rate); ++step) { // one second, past the bottom of the swing
            solver.Step(world, settings);
            peak = std::max(peak, simd::length(world.Velocities[pendulum.Arm].Angular));
        }
        return std::pair{peak, std::sqrt(2 * pendulum.Mass * gravity * Distance / pendulum.Inertia)};
    };

    const auto [coarse, expected] = fastest(60);
    const auto [fine, same] = fastest(960);
    CHECK(same == doctest::Approx(expected)); // the closed form does not depend on the step
    CHECK(coarse < expected); // backwards Euler only ever loses energy
    CHECK(coarse > 0.9f * expected);
    CHECK(fine > coarse); // and a shorter step loses less of it
    CHECK(fine == doctest::Approx(expected).epsilon(0.02));
}

TEST_CASE("a joint that holds rotation makes two boxes into one body") {
    // Fixed together side by side and dropped, they have to land flat and stay square to each other,
    // which is the angular rows' whole claim.
    const mtl::Context context;
    Solver solver{context};
    World world{context};
    const auto shape = world.AddShape(UnitBox);
    world.AddBody({.Shape = world.AddShape(GroundPlane)});
    const auto left = world.AddBody({.Pose = {.Position = {-Half, 2, 0}, .Orientation = {0, 0, 0, 1}}, .Shape = shape});
    const auto right = world.AddBody({.Pose = {.Position = {Half, 2, 0}, .Orientation = {0, 0, 0, 1}}, .Shape = shape});
    world.AddJoint({.BodyA = left, .BodyB = right, .At = {0, 2, 0}, .Angular = {AxisLocked, AxisLocked, AxisLocked}});

    Run(solver, world, 400);
    CheckResting(world.Poses[left].Position.y);
    CheckResting(world.Poses[right].Position.y);
    // Still one box apart, and still pointing the same way.
    CHECK(world.Poses[right].Position.x - world.Poses[left].Position.x == doctest::Approx(1).epsilon(0.01));
    const float4 a = world.Poses[left].Orientation, b = world.Poses[right].Orientation;
    CHECK(std::abs(simd::dot(a, b)) == doctest::Approx(1).epsilon(0.001)); // the same orientation
    CHECK(simd::length(world.Velocities[left].Linear) < 1e-2f);
}

TEST_CASE("a hinge leaves one axis free and holds the other two") {
    // Free about z and held about x and y, so whatever it is hit with it swings in the xy plane and
    // stays there. Pushed straight out of that plane at three metres a second, it must not leave it.
    const mtl::Context context;
    Solver solver{context};
    World world{context};
    const auto pendulum = MakePendulum(world, 1, {.Angular = {AxisLocked, AxisLocked, AxisFree}});
    world.Velocities[pendulum.Arm].Linear = {0, 0, 3};

    // The first few steps give way by about a centimetre while the penalty ramps off its floor. What
    // the axis claims is that it takes that back and holds, so the window measured starts after it.
    Run(solver, world, 60);
    float worst = 0;
    for (uint32_t step = 0; step < 240; ++step) {
        solver.Step(world);
        worst = std::max(worst, std::abs(world.Poses[pendulum.Arm].Position.z));
    }
    CHECK(worst < 1e-3f);
    CHECK(simd::length(world.Velocities[pendulum.Arm].Angular) > 1); // it did swing, rather than seizing
}

TEST_CASE("a motor turns its axis at the speed it is given") {
    // With torque to spare the driven axis reaches its target relative speed and holds it, gravity or
    // no gravity, since what the row asks for is a turn of speed * dt every step whatever else happens.
    constexpr float Speed = 2;
    const auto driven = JointDesc{.Angular = {AxisLocked, AxisLocked, AxisDriven},
                                  .MotorSpeed = {0, 0, Speed},
                                  .MotorMaxTorque = {0, 0, 1e6f}};

    const mtl::Context context;
    Solver solver{context};
    World world{context};
    const auto pendulum = MakePendulum(world, 1, driven);
    Run(solver, world, 300);
    CHECK(world.Velocities[pendulum.Arm].Angular.z == doctest::Approx(Speed).epsilon(0.01));
    // Its own weight is 9810 N m about the pivot at the horizontal and the motor carries it round anyway.
    CHECK(simd::length(world.Poses[pendulum.Arm].Position) == doctest::Approx(1).epsilon(0.01));
}

TEST_CASE("a positioned motor turns its axis to the angle it is given") {
    // The same row as the velocity motor with a target angle in place of a target rate. What it asks
    // for is the whole remaining error rather than a step of turn, so it is the torque bound and not
    // the row that decides how fast the axis gets there.
    constexpr float Target = 0.8f;
    const mtl::Context context;
    Solver solver{context};
    World world{context};
    const auto pendulum = MakePendulum(world, 1,
                                       {.Angular = {AxisLocked, AxisLocked, AxisPositioned},
                                        .MotorTarget = {0, 0, Target},
                                        .MotorMaxTorque = {0, 0, 1e6f}});
    const auto angle = [&world, &pendulum] {
        const auto at = world.Poses[pendulum.Arm].Position;
        return std::atan2(float(at.y), float(at.x));
    };

    Run(solver, world, 300);
    // The arm started along x, so where it ends up around the pivot is the angle itself. Its own
    // weight is 9810 N m about the pivot at the horizontal and the motor holds it against that.
    CHECK(angle() == doctest::Approx(Target).epsilon(0.01));
    CHECK(simd::length(world.Poses[pendulum.Arm].Position) == doctest::Approx(1).epsilon(0.01));
    CHECK(simd::length(world.Velocities[pendulum.Arm].Angular) < 1e-2f); // and stopped there

    SUBCASE("and brings it back after being moved off it") {
        // Which is what makes it a position motor rather than a move: the row asks for the angle every
        // step, so wherever the axis is when it is asked, the answer is the same. Moved rather than
        // shoved, since a motor with a megatonne of torque absorbs a shove inside one step.
        world.Poses[pendulum.Arm].Position = {0, -1, 0};
        world.Poses[pendulum.Arm].Orientation = QuatFromRotationVector(float3{0, 0, -1.5707963f});
        REQUIRE(std::abs(angle() - Target) > 0.02f);
        Run(solver, world, 300);
        CHECK(angle() == doctest::Approx(Target).epsilon(0.01));
        CHECK(simd::length(world.Velocities[pendulum.Arm].Angular) < 1e-2f);
    }
}

TEST_CASE("a positioned motor held to a torque climbs at exactly that torque over that inertia") {
    // Torque limited, in free space, from rest, and asked for an angle far enough away that the row
    // stays saturated the whole time. Then the torque is constant and the arm is a body of known
    // inertia about its pivot, so the rate it picks up is tau / I and nothing else is in it. The same
    // check the velocity motor gets, which is what says the two really are one row with two targets.
    constexpr float Torque = 300, Target = 3;
    const StepSettings settings{.Gravity = {0, 0, 0}};
    const mtl::Context context;
    Solver solver{context};
    World world{context};
    const auto pendulum = MakePendulum(world, 1,
                                       {.Angular = {AxisLocked, AxisLocked, AxisPositioned},
                                        .MotorTarget = {0, 0, Target},
                                        .MotorMaxTorque = {0, 0, Torque}});

    constexpr uint32_t Steps = 200;
    Run(solver, world, Steps, settings);
    const float elapsed = Steps * settings.DeltaTime;
    const float reached = world.Velocities[pendulum.Arm].Angular.z;
    // Half of tau/I t^2 is how far it has come, and it is short of the target, so the row never came
    // off its bound and the rate below is the bound's and not the angle's.
    REQUIRE(0.5f * Torque / pendulum.Inertia * elapsed * elapsed < Target);
    CHECK(reached == doctest::Approx(Torque / pendulum.Inertia * elapsed).epsilon(0.05));
}

TEST_CASE("a motor held to a torque spins up at exactly that torque over that inertia") {
    // Torque limited, in free space, from rest: the arm is a body of known inertia about its pivot and
    // a constant torque on it is a constant angular acceleration of tau / I. Nothing else in that.
    constexpr float Torque = 300, Speed = 2;
    const StepSettings settings{.Gravity = {0, 0, 0}};
    const mtl::Context context;
    Solver solver{context};
    World world{context};
    const auto pendulum = MakePendulum(world, 1,
                                       {.Angular = {AxisLocked, AxisLocked, AxisDriven},
                                        .MotorSpeed = {0, 0, Speed},
                                        .MotorMaxTorque = {0, 0, Torque}});

    constexpr uint32_t Steps = 300;
    Run(solver, world, Steps, settings);
    const float elapsed = Steps * settings.DeltaTime;
    const float reached = world.Velocities[pendulum.Arm].Angular.z;
    REQUIRE(reached < Speed); // still climbing, so the limit and not the target is what set the rate
    CHECK(reached == doctest::Approx(Torque / pendulum.Inertia * elapsed).epsilon(0.05));
}

TEST_CASE("a limited axis turns freely between its stops and comes to rest on one") {
    // A stop is a contact's one-sided row wearing a different hat: nothing at all while the axis is
    // inside its range, and outside it a force that may only push the one way. Released horizontal
    // with stops half a radian either side, the arm swings down and stays on the low one.
    constexpr float Low = -0.5f, High = 0.5f;
    const mtl::Context context;
    Solver solver{context};
    World world{context};
    const auto pendulum = MakePendulum(world, 1,
                                       {.Angular = {AxisLocked, AxisLocked, AxisLimited},
                                        .LimitLow = {0, 0, Low},
                                        .LimitHigh = {0, 0, High}});

    float lowest = 1e9f, highest = -1e9f;
    for (uint32_t step = 0; step < 600; ++step) {
        solver.Step(world);
        const auto &at = world.Poses[pendulum.Arm].Position;
        const float angle = std::atan2(at.y, at.x);
        lowest = std::min(lowest, angle);
        highest = std::max(highest, angle);
    }

    // Which stop a row argues for is settled from the angle the step began at, so the axis carries a
    // step of its own rotation past the stop before anything answers, and stabilization takes it back.
    CHECK(lowest > Low - 0.05f);
    CHECK(lowest < Low); // it did reach the stop rather than being held short of it
    CHECK(highest < High);
    CHECK(std::atan2(world.Poses[pendulum.Arm].Position.y, world.Poses[pendulum.Arm].Position.x) ==
          doctest::Approx(Low).epsilon(0.002));
    CHECK(simd::length(world.Velocities[pendulum.Arm].Angular) < 1e-3f); // and it is still there
}

TEST_CASE("a spring on a limit sags until it balances the load") {
    // A limited axis given a finite stiffness is soft rather than hard: no dual, a penalty that ramps
    // to that stiffness and no further, and a force that is Eq. 7 on the angle it actually has. So it
    // does not hold its stop - it sags until k theta balances the arm's own weight about the pivot.
    const mtl::Context context;
    Solver solver{context};
    const auto settled = [&context, &solver](float stiffness) {
        World world{context};
        // Free to swing up, stopped at the horizontal on the way down, and softly.
        const auto pendulum = MakePendulum(world, 1,
                                           {.Angular = {AxisLocked, AxisLocked, AxisLimited},
                                            .LimitLow = {0, 0, 0},
                                            .LimitHigh = {0, 0, 3.14159f},
                                            .AngularStiffness = {INFINITY, INFINITY, stiffness}});
        Run(solver, world, 900);
        const auto at = world.Poses[pendulum.Arm].Position;
        const float angle = std::atan2(float(at.y), float(at.x));
        // k theta = m g cos(theta) on a one metre arm. It hangs below the stop, so the angle is negative.
        const float balance = -pendulum.Mass * std::abs(StepSettings{}.Gravity.y) * std::cos(angle) / stiffness;
        CHECK(angle == doctest::Approx(balance).epsilon(0.1).scale(0));
        CHECK(angle < 0); // it did sag, which is the whole difference from a hard stop
        CHECK(simd::length(world.Velocities[pendulum.Arm].Angular) < 1e-2f); // and stopped sagging
        return angle;
    };

    const float soft = settled(1e6f), stiff = settled(4e6f);
    CHECK(std::abs(soft) > 3 * std::abs(stiff)); // four times the stiffness, near enough a quarter the sag
}

TEST_CASE("a soft linear row hangs a body on a spring") {
    // The same soft branch on the rows holding the anchors together, which is what makes a joint soft
    // rather than only its stops. A body hung on one settles at mg/k below where it is held.
    constexpr float Stiffness = 2e5f;
    const mtl::Context context;
    Solver solver{context};
    World world{context};
    const auto anchor = world.AddBody({}); // no shape, so no mass: a fixed point to hang from
    const auto box = world.AddBody({.Shape = world.AddShape(UnitBox)});
    world.AddJoint({.BodyA = box,
                    .BodyB = anchor,
                    .At = {0, 0, 0},
                    .LinearStiffness = {Stiffness, Stiffness, Stiffness}});

    Run(solver, world, 900);
    const float weight = std::abs(StepSettings{}.Gravity.y) / world.Masses[box].InvMass;
    CHECK(world.Poses[box].Position.y == doctest::Approx(-weight / Stiffness).epsilon(0.05).scale(0));
    CHECK(std::abs(world.Poses[box].Position.x) < 1e-3f); // straight down, and only down
    CHECK(simd::length(world.Velocities[box].Linear) < 1e-2f);
}

TEST_CASE("a soft locked axis is a torsional spring, and holds both ways") {
    // The soft branch on a row that holds an angle rather than a stop. It sags until k theta balances
    // the arm's weight about the pivot, the same number the soft limit settles at. The difference from
    // a limit is that it is two-sided, so turning gravity over has to give the mirror image - a stop at
    // the same angle would let the arm swing away from it freely.
    const mtl::Context context;
    Solver solver{context};
    const auto settled = [&context, &solver](float stiffness, float sign) {
        World world{context};
        const auto pendulum = MakePendulum(world, 1,
                                           {.Angular = {AxisLocked, AxisLocked, AxisLocked},
                                            .AngularStiffness = {INFINITY, INFINITY, stiffness}});
        StepSettings settings{};
        settings.Gravity.y *= sign;
        Run(solver, world, 900, settings);
        const auto at = world.Poses[pendulum.Arm].Position;
        const float angle = std::atan2(float(at.y), float(at.x));
        // k theta = m g cos(theta) on a one metre arm, with the load on whichever side gravity is.
        // scale(0) because these are hundredths of a radian and Approx's default absolute slack is a
        // whole unit, which at this size is the answer.
        const float balance = -sign * pendulum.Mass * std::abs(settings.Gravity.y) * std::cos(angle) / stiffness;
        CHECK(angle == doctest::Approx(balance).epsilon(0.05).scale(0));
        CHECK(simd::length(world.Velocities[pendulum.Arm].Angular) < 1e-2f); // and stopped sagging
        // The hard rows either side of it hold: the arm turns about z and about nothing else.
        const float3 turn = RotationVector(world.Poses[pendulum.Arm].Orientation);
        CHECK(std::abs(turn.x) < 1e-3f);
        CHECK(std::abs(turn.y) < 1e-3f);
        return angle;
    };

    const float down = settled(1e6f, 1), up = settled(1e6f, -1);
    CHECK(down < 0); // it did sag, which is the whole difference from a hard lock
    CHECK(up == doctest::Approx(-down).epsilon(0.02)); // and the same amount the other way
    CHECK(std::abs(down) > 3 * std::abs(settled(4e6f, 1))); // four times the stiffness, near enough a quarter the sag
}

TEST_CASE("a soft positioned axis is a spring towards where it is told to be") {
    // The same soft branch on the mode that measures its error from the target rather than from rest.
    // That moves the spring's zero and nothing else, so where it settles says which of the two rows is
    // taken: a locked row given the same arm holds it a whole target angle away.
    //
    // The torque bound is well clear of the load, so the secant rescaling never fires here - a bounded
    // motor is asked in the position-motor tests.
    constexpr float Stiffness = 1e6f;
    const mtl::Context context;
    Solver solver{context};
    const auto settled = [&context, &solver](float target) {
        World world{context};
        const auto pendulum = MakePendulum(world, 1,
                                           {.Angular = {AxisLocked, AxisLocked, AxisPositioned},
                                            .MotorTarget = {0, 0, target},
                                            .MotorMaxTorque = {0, 0, 1e6f},
                                            .AngularStiffness = {INFINITY, INFINITY, Stiffness}});
        Run(solver, world, 900);
        const auto at = world.Poses[pendulum.Arm].Position;
        const float angle = std::atan2(float(at.y), float(at.x));
        // k (theta - target) = m g cos(theta), the spring above with its zero moved to the target.
        const float balance = target - pendulum.Mass * std::abs(StepSettings{}.Gravity.y) * std::cos(angle) / Stiffness;
        CHECK(angle == doctest::Approx(balance).epsilon(0.02).scale(0));
        CHECK(simd::length(world.Velocities[pendulum.Arm].Angular) < 1e-2f);
        return angle;
    };

    // Told to hold where it already is, it hangs the sag below that. Told to lift, it arrives a whole
    // target angle higher, having climbed against gravity to get there.
    const float held = settled(0), lifted = settled(0.6f);
    CHECK(held < 0);
    CHECK(lifted - held == doctest::Approx(0.6f).epsilon(0.02));
}

TEST_CASE("an axis slammed between both its stops stays between them") {
    // A row carries its dual from step to step and the two stops are different constraints, so an axis
    // crossing from one to the other carries a dual that argues the wrong way. Nothing tracks which
    // stop it was on - the one-sided clamp takes the wrong-signed force to zero on the first iteration
    // and the dual update writes the right one, and this says that is enough.
    constexpr float Low = -0.2f, High = 0.2f, Slam = 6;
    const StepSettings settings{.Gravity = {0, 0, 0}}; // no gravity, so it is the stops doing all the work
    const mtl::Context context;
    Solver solver{context};
    World world{context};
    const auto pendulum = MakePendulum(world, 1,
                                       {.Angular = {AxisLocked, AxisLocked, AxisLimited},
                                        .LimitLow = {0, 0, Low},
                                        .LimitHigh = {0, 0, High}});

    float worst = 0;
    for (uint32_t step = 0; step < 900; ++step) {
        if (step % 120 == 0) world.Velocities[pendulum.Arm].Angular = {0, 0, (step / 120) % 2 == 0 ? Slam : -Slam};
        solver.Step(world, settings);
        const auto &at = world.Poses[pendulum.Arm].Position;
        const float angle = std::atan2(at.y, at.x);
        worst = std::max(worst, std::max(angle - High, Low - angle));
    }
    // Arriving at six radians a second it would carry a tenth of a radian past a stop in the step
    // before one engages. It does not get near that, and it does not accumulate over eight slams.
    CHECK(worst < Slam * settings.DeltaTime);
}

TEST_CASE("a settled stack keeps the same contacts from step to step") {
    // A feature that changes under a stack which is not moving throws that contact's dual away and
    // kicks the stack for no physical reason, so a settled stack must produce exactly the same set of
    // contacts every step - not merely the same number of them.
    constexpr uint32_t Count = 4;
    const mtl::Context context;
    Solver solver{context};
    World world{context};
    const auto shape = world.AddShape(UnitBox);
    world.AddBody({.Shape = world.AddShape(GroundPlane)});
    for (uint32_t i = 0; i < Count; ++i)
        world.AddBody({.Pose = {.Position = {0, Half + 1.02f * float(i), 0}, .Orientation = {0, 0, 0, 1}}, .Shape = shape});

    Run(solver, world, 240); // settle first: contacts legitimately come and go while it is landing
    const auto settled = ContactKeys(world);
    CHECK(settled.size() == 4 * Count); // four corners against the plane, four against each box below
    for (uint32_t step = 0; step < 300; ++step) {
        solver.Step(world);
        CAPTURE(step);
        REQUIRE(ContactKeys(world) == settled);
    }
}

TEST_CASE("a raft with a layer on top settles, and takes the colours it needs to do it") {
    // Every other stack here is a chain, which two colours hold. Each box on the upper layer touches
    // four below it, so the colouring has to find more - and it grows by one colour a step, so a scene
    // closing up needs a few steps to reach its count. What has to be true is that it does, and rests.
    const mtl::Context context;
    Solver solver{context};
    World world{context};
    const auto shape = world.AddShape(UnitBox);
    world.AddBody({.Shape = world.AddShape(GroundPlane)});
    for (uint32_t x = 0; x < 4; ++x)
        for (uint32_t z = 0; z < 4; ++z)
            world.AddBody({.Pose = {.Position = {float(x) - 1.5f, Half, float(z) - 1.5f}, .Orientation = {0, 0, 0, 1}},
                           .Shape = shape});
    for (uint32_t x = 0; x < 3; ++x)
        for (uint32_t z = 0; z < 3; ++z)
            world.AddBody({.Pose = {.Position = {float(x) - 1.f, 3 * Half + 0.05f, float(z) - 1.f},
                                    .Orientation = {0, 0, 0, 1}},
                           .Shape = shape});

    Run(solver, world, 300);
    uint32_t most = 0;
    float fastest = 0;
    for (uint32_t body = 1; body < world.BodyCount(); ++body) {
        most = std::max(most, ColorOf(world.Colors[body]));
        fastest = std::max(fastest, simd::length(world.Velocities[body].Linear));
    }
    CHECK(most + 1 > 2); // a chain's two would not have held it
    CHECK(most < StepSettings{}.MaxColors); // and it found what it needed inside the cap
    CHECK(fastest < 1e-2f);
    CheckResting(world.Poses[1].Position.y); // the lower layer is on the plane
    CHECK(world.Poses[world.BodyCount() - 1].Position.y == doctest::Approx(3 * Half).epsilon(0.005)); // upper on lower
}

TEST_CASE("a body that has got nowhere for half a second stops being solved") {
    // A sleeping body keeps its contacts, so whatever rests on it stays held up, and its velocity is
    // exactly zero rather than nearly - a stack that has stopped rather than one still fidgeting.
    const StepSettings settings{};
    const auto asleep = [&](const World &world, Index body) { return world.Quiet[body] >= settings.SleepSteps; };

    const mtl::Context context;
    Solver solver{context};
    World world{context};
    const auto shape = world.AddShape(UnitBox);
    world.AddBody({.Shape = world.AddShape(GroundPlane)});
    const auto box = world.AddBody({.Pose = {.Position = {0, Half + 1, 0}, .Orientation = {0, 0, 0, 1}}, .Shape = shape});

    Run(solver, world, 200);
    REQUIRE(asleep(world, box));
    CHECK(world.Velocities[box].Linear.y == 0);
    CheckResting(world.Poses[box].Position.y);
    const float settled = world.Poses[box].Position.y;
    Run(solver, world, 200);
    CHECK(world.Poses[box].Position.y == settled); // and it has not moved a bit since

    SUBCASE("and wakes when the host gives it a shove") {
        // Nothing hooks a write to the velocity buffer, so waking on one is what makes an external
        // push work at all - a sleeping body's velocity is zero, so anything in there came from outside.
        world.Velocities[box].Linear = {2, 0, 0};
        Run(solver, world, 2);
        CHECK(!asleep(world, box));
        Run(solver, world, 300);
        CHECK(world.Poses[box].Position.x > 0.1f); // it went somewhere
        CHECK(asleep(world, box)); // and then stopped again
    }

    SUBCASE("and wakes when something lands on it") {
        const auto dropped = world.AddBody({.Pose = {.Position = {0, Half + 3, 0}, .Orientation = {0, 0, 0, 1}}, .Shape = shape});
        Run(solver, world, 60);
        CHECK(!asleep(world, box)); // the sleeper felt it arrive
        Run(solver, world, 400);
        CHECK(asleep(world, box));
        CHECK(asleep(world, dropped));
        CHECK(std::abs(world.Poses[dropped].Position.y - (settled + 1 - settings.ContactMargin)) < 3e-3f);
    }
}

TEST_CASE("nothing sleeps while what it is touching is still moving") {
    // Sleeping is a property of a group, not of a body: a box that sleeps mid-settle leaves the one
    // under it pressing against something that has stopped answering. So a body is never counted
    // quieter than its neighbours, and one link of a stack moving holds all of it awake.
    const StepSettings settings{};
    const mtl::Context context;
    Solver solver{context};
    World world{context};
    const auto shape = world.AddShape(UnitBox);
    world.AddBody({.Shape = world.AddShape(GroundPlane)});
    std::vector<Index> stack;
    for (uint32_t i = 0; i < 3; ++i)
        stack.push_back(world.AddBody({.Pose = {.Position = {0, Half + 1.02f * float(i), 0}, .Orientation = {0, 0, 0, 1}},
                                       .Shape = shape}));

    // Drop a fourth from a height, so it is still falling long after the three below have gone quiet.
    const auto falling = world.AddBody({.Pose = {.Position = {0, Half + 12, 0}, .Orientation = {0, 0, 0, 1}}, .Shape = shape});
    for (uint32_t step = 0; step < 90; ++step) {
        solver.Step(world);
        // While it is in the air it touches nothing, so the stack may sleep - but once it lands, nothing
        // in the pile it joined may be asleep while any of it is moving.
        const bool landed = world.Poses[falling].Position.y < Half + 4;
        if (!landed) continue;
        // Motion rather than the quiet counter: a body the spread has just woken carries a count of
        // zero and has not moved at all, waking travelling one contact a step. The two steps after an
        // impact arrives are exactly that - awake and still.
        uint32_t moving = 0, sleeping = 0;
        for (const auto body : stack) {
            moving += simd::length(world.Velocities[body].Linear) > settings.SleepSpeed ? 1 : 0;
            sleeping += world.Quiet[body] >= settings.SleepSteps ? 1 : 0;
        }
        CAPTURE(step);
        CHECK((moving == 0 || sleeping == 0));
    }
}

TEST_CASE("a body only collides with what its mask says it does") {
    // Two boxes dropped onto the same spot on the ground. Each is in the ground's mask so both land,
    // but neither is in the other's, so they pass through each other on the way down and end up
    // occupying the same place - which is exactly what asking for it should give.
    constexpr uint32_t Ground = 1, First = 2, Second = 4;
    const mtl::Context context;
    Solver solver{context};
    World world{context};
    const auto shape = world.AddShape(UnitBox);
    world.AddBody({.Shape = world.AddShape(GroundPlane), .Layer = Ground, .CollidesWith = First | Second});
    const auto low = world.AddBody({.Pose = {.Position = {0, Half + 1, 0}, .Orientation = {0, 0, 0, 1}},
                                    .Shape = shape, .Layer = First, .CollidesWith = Ground});
    const auto high = world.AddBody({.Pose = {.Position = {0, Half + 3, 0}, .Orientation = {0, 0, 0, 1}},
                                     .Shape = shape, .Layer = Second, .CollidesWith = Ground});

    Run(solver, world, 400);
    CheckResting(world.Poses[low].Position.y);
    CheckResting(world.Poses[high].Position.y); // through the other one and onto the floor beside it
    CHECK(std::abs(world.Poses[high].Position.y - world.Poses[low].Position.y) < 1e-3f);
}

TEST_CASE("a joint between two moving bodies gives each what it takes from the other") {
    // Every other joint test here hangs off a static pivot, which is half a joint: one side cannot
    // move, so nothing checks that what the joint does to one body it does equally and oppositely to
    // the other. A ball joint applies its force at one point on both, so it can move the pair's centre
    // of mass nowhere and change the angular momentum about that centre not at all.
    const StepSettings settings{.Gravity = {0, 0, 0}}; // nothing outside the pair to account for
    const mtl::Context context;
    Solver solver{context};

    // Total momentum, and angular momentum about the pair's own centre of mass - which is what a joint
    // may not touch. Both bodies, both terms.
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
            // The inertia the body has in the world, which is its own turned by where it has turned to.
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
        const auto a = world.AddBody({.Pose = {.Position = {-1, 0, 0}, .Orientation = {0, 0, 0, 1}},
                                      .Velocity = {.Linear = {0, 0.5f, 0}, .Angular = {0, 0, 1.5f}},
                                      .Shape = shape});
        const auto b = world.AddBody({.Pose = {.Position = {1, 0, 0}, .Orientation = {0, 0, 0, 1}},
                                      .Velocity = {.Linear = {0.25f, 0, -0.3f}},
                                      .Shape = shape,
                                      .Density = density_b});
        REQUIRE(world.AddJoint({.BodyA = a, .BodyB = b, .At = {0, 0, 0}}) != NoIndex);
        return std::tuple{std::move(world), a, b};
    };

    const auto swing = [&](float density_b) {
        auto [world, a, b] = pair(density_b);
        const Momentum began = momentum(world, a, b);
        const float3 started = began.Centre;
        // What the pair's centre of mass is doing, which nothing in the scene can change.
        const float3 together = began.Linear * (world.Masses[a].InvMass * world.Masses[b].InvMass) /
            (world.Masses[a].InvMass + world.Masses[b].InvMass);
        float worst_hold = 0, worst_drift = 0;
        for (uint32_t step = 0; step < 600; ++step) {
            solver.Step(world, settings);
            const Pose &pa = world.Poses[a], &pb = world.Poses[b];
            // The one point the joint claims, as each body says where it now is.
            worst_hold = std::max(worst_hold, simd::distance(pa.Position + Rotate(pa.Orientation, world.Joints[0].AnchorA),
                                                             pb.Position + Rotate(pb.Orientation, world.Joints[0].AnchorB)));
            // And where the centre of mass has got to, in steps of its own travel. A pose is where the
            // body is now and a velocity is the last step's displacement over the step, so the two are
            // half a step out of phase: a constant offset of that size is bookkeeping, a growing one
            // is a force.
            const Momentum now = momentum(world, a, b);
            const float3 carried = started + together * (float(step + 1) * settings.DeltaTime);
            worst_drift = std::max(worst_drift, simd::distance(now.Centre, carried) / (settings.DeltaTime * simd::length(together)));
        }
        return std::tuple{began, momentum(world, a, b), worst_hold, worst_drift};
    };

    SUBCASE("equal masses") {
        const auto [began, ended, hold, drift] = swing(1000);
        CHECK(hold < 5e-3f); // the anchor is one point throughout, as it is off a static pivot
        // Nothing outside the pair, so what it had it keeps. Ten seconds of a joint working hard.
        CHECK(simd::length(ended.Linear - began.Linear) < 1e-3f * simd::length(began.Linear));
        CHECK(simd::length(ended.Angular - began.Angular) < 2e-2f * simd::length(began.Angular));
        CHECK(drift < 1); // and the centre of mass never left the step it is read half of behind
    }

    SUBCASE("a thousand to one in mass") {
        // Sec. 3.4's stiffness ratio, on a joint rather than a contact: the light body must not be
        // able to drag the heavy one, and the heavy one must not fling the light one.
        const auto [began, ended, hold, drift] = swing(1);
        CHECK(hold < 5e-3f);
        CHECK(simd::length(ended.Linear - began.Linear) < 1e-3f * simd::length(began.Linear));
        CHECK(drift < 1);
    }
}

TEST_CASE("a joint stops its two bodies colliding, unless it is asked not to") {
    // Two boxes overlapping by half and pinned together. The contacts that overlap would otherwise
    // generate fight the joint for no physical reason, which is why a joint disables them by default.
    const auto overlapping = [&](bool collide) {
        const mtl::Context context;
        Solver solver{context};
        World world{context};
        const auto shape = world.AddShape(UnitBox);
        const auto left = world.AddBody({.Pose = {.Position = {0, 4, 0}, .Orientation = {0, 0, 0, 1}}, .Shape = shape});
        const auto right = world.AddBody({.Pose = {.Position = {Half, 4, 0}, .Orientation = {0, 0, 0, 1}}, .Shape = shape});
        world.AddJoint({.BodyA = left, .BodyB = right, .At = {0.25f, 4, 0}, .Collide = collide});
        Run(solver, world, 60, {.Gravity = {0, 0, 0}}); // no gravity, so contacts are the only thing acting
        return ActiveContacts(world);
    };
    CHECK(overlapping(false) == 0);
    CHECK(overlapping(true) > 0);
}

// The behavioural scenes: no closed form to check any against, so each asserts what has to be true of
// it whatever the numbers come out as. They are the shapes a solver fails on - a pyramid loses its
// footing, a light body under a heavy one is crushed, a chain flies apart, a heap squeezes something
// out through a wall - and none of the analytic tests above would say a word about any of it.
namespace {
// The worst of what a set of dynamic bodies is doing, which is what a plausibility invariant is made of.
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

TEST_CASE("a pyramid holds itself up") {
    // Five rows down to one. Unlike a stack, every box but the bottom row rests on two below it and is
    // held in place by friction alone - nothing stops it sliding out sideways except the contacts.
    const mtl::Context context;
    Solver solver{context};
    World world{context};
    const auto shape = world.AddShape(UnitBox);
    world.AddBody({.Shape = world.AddShape(GroundPlane)});
    for (uint32_t row = 0; row < 5; ++row)
        for (uint32_t i = 0; i < 5 - row; ++i)
            world.AddBody({.Pose = {.Position = {1.02f * (float(i) + 0.5f * float(row) - 2), Half + 1.02f * float(row), 0},
                                    .Orientation = {0, 0, 0, 1}},
                           .Shape = shape});
    const Index top = world.BodyCount() - 1;
    const float was = world.Poses[top].Position.x;

    Run(solver, world, 600);
    const auto worst = WorstOf(world, 1);
    CHECK(worst.Speed < 1e-2f); // it came to rest
    CHECK(worst.Lowest > Half - 4 * StepSettings{}.ContactMargin); // and did not sink into the floor
    CHECK(world.Poses[top].Position.y == doctest::Approx(Half + 4).epsilon(0.01)); // still five rows tall
    CHECK(std::abs(world.Poses[top].Position.x - was) < 0.05f); // and the apex did not walk off it
}

TEST_CASE("a light box carries a box a thousand times its weight") {
    // Sec. 3.4's stiffness ratio: the failure the augmented Lagrangian prevents is the light body being
    // squeezed out of existence by the heavy one's constraint. Both end up where geometry says.
    constexpr float Margin = StepSettings{}.ContactMargin;
    const mtl::Context context;
    Solver solver{context};
    World world{context};
    const auto shape = world.AddShape(UnitBox);
    world.AddBody({.Shape = world.AddShape(GroundPlane)});
    const auto light = world.AddBody({.Pose = {.Position = {0, Half, 0}, .Orientation = {0, 0, 0, 1}}, .Shape = shape, .Density = 10});
    const auto heavy = world.AddBody({.Pose = {.Position = {0, 3 * Half + 0.02f, 0}, .Orientation = {0, 0, 0, 1}},
                                      .Shape = shape,
                                      .Density = 10000});

    Run(solver, world, 600);
    CHECK(std::abs(world.Poses[light].Position.y - (Half - Margin)) < 2e-3f);
    CHECK(std::abs(world.Poses[heavy].Position.y - (3 * Half - 2 * Margin)) < 3e-3f);
    CHECK(simd::length(world.Velocities[light].Linear) < 0.1f);
}

TEST_CASE("a chain of ten hangs from its anchor without coming apart") {
    // Released straight out sideways, which is the worst load a chain can be given, and there is nothing
    // to damp it afterwards - so it does not come to rest and is not asked to. What it must do is stay
    // joined: every link within reach of the anchor, and no joint pulled open.
    constexpr uint32_t Links = 10;
    const mtl::Context context;
    Solver solver{context};
    World world{context};
    const auto shape = world.AddShape(UnitBox);
    const auto anchor = world.AddBody({});
    Index above = anchor;
    for (uint32_t i = 0; i < Links; ++i) {
        const auto link = world.AddBody({.Pose = {.Position = {1.f + float(i), 0, 0}, .Orientation = {0, 0, 0, 1}}, .Shape = shape});
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
            const float3 apart = (a.Position + Rotate(a.Orientation, joint.AnchorA)) -
                (b.Position + Rotate(b.Orientation, joint.AnchorB));
            stretched = std::max(stretched, simd::length(apart));
        }
        furthest = std::max(furthest, WorstOf(world, 1).Reach);
    }
    CHECK(stretched < 0.03f); // a joint is a point held to a point, and it stayed one
    CHECK(furthest < Links + Half + 0.1f); // no link got further out than the chain can reach
}

TEST_CASE("a heap of boxes dropped into a bin settles inside it") {
    // Eighteen boxes falling into a four by four box with walls, which is many bodies deciding where to
    // be at once against static geometry that is not a plane. What has to hold is that they all stop,
    // none is squeezed out through a wall, and nothing is left standing on nothing.
    const mtl::Context context;
    Solver solver{context};
    World world{context};
    const auto shape = world.AddShape(UnitBox);
    const auto wall = world.AddShape(Shape{.HalfExtents = {2.5f, 2, 0.25f}, .Kind = ShapeBox});
    world.AddBody({.Shape = world.AddShape(GroundPlane)});
    for (uint32_t side = 0; side < 4; ++side) {
        const float turn = 2 * std::atan(1.f) * float(side);
        world.AddBody({.Pose = {.Position = {std::sin(turn) * 2.25f, 2, std::cos(turn) * 2.25f},
                                .Orientation = QuatFromRotationVector(float3{0, turn, 0})},
                       .Shape = wall,
                       .Density = 0});
    }
    for (uint32_t i = 0; i < 18; ++i)
        world.AddBody({.Pose = {.Position = {1.2f * float(i % 3) - 1.2f, 5.f + 1.3f * float(i / 3), 1.2f * float(i / 3 % 3) - 1.2f},
                                .Orientation = {0, 0, 0, 1}},
                       .Shape = shape});

    float widest = 0;
    for (uint32_t step = 0; step < 900; ++step) {
        solver.Step(world);
        widest = std::max(widest, WorstOf(world, 5).Reach);
    }
    const auto worst = WorstOf(world, 5);
    CHECK(worst.Speed < 1e-2f);
    CHECK(widest < 2.25f - 0.25f); // never reached the inside face of a wall, let alone through it
    CHECK(worst.Lowest > Half - 4 * StepSettings{}.ContactMargin);
    CHECK(worst.Highest < Half + 3); // no tower left standing on nothing
}

TEST_CASE("contact events report a manifold arriving, holding and going away") {
    // The distinction CollectContacts already draws to warm start: a point whose feature was held last
    // step is the same contact carried forward, one whose feature is new is a contact that did not
    // exist, and a feature nothing claims names one that has ended.
    const mtl::Context context;
    Solver solver{context};
    World world{context};
    const auto box = DropBox(world, Half, true); // already touching, so the first step forms the manifold

    Run(solver, world, 1);
    // Four corners of a face against a plane, and the box owns every one of them: a plane never
    // generates a manifold, and the body a contact names as A is the one that reports it.
    CHECK(Reported(world, box, ContactAdded) == 4);
    CHECK(Reported(world, box, ContactPersisted) == 0);
    CHECK(Reported(world, box, ContactRemoved) == 0);
    CHECK(world.ContactEventCounts[0] == 0); // the plane says nothing about a contact it does not own
    for (uint32_t i = 0; i < world.ContactEventCounts[box]; ++i) {
        const auto &event = world.ContactEvents[box * EventsPerBody + i];
        CHECK(event.BodyA == box);
        CHECK(event.BodyB == 0);
    }

    Run(solver, world, 1);
    CHECK(Reported(world, box, ContactAdded) == 0);
    CHECK(Reported(world, box, ContactPersisted) == 4); // the same four, by the same four features
    CHECK(Reported(world, box, ContactRemoved) == 0);

    // Taken away by moving it out of reach rather than by removing it, which keeps this test about the
    // events rather than the world's pools. What a removal reports is checked where removal is.
    world.Poses[box].Position = {0, Half + 10, 0};
    world.Velocities[box] = {};
    Run(solver, world, 1);
    CHECK(Reported(world, box, ContactRemoved) == 4);
    CHECK(Reported(world, box, ContactAdded) == 0);
    CHECK(Reported(world, box, ContactPersisted) == 0);

    Run(solver, world, 1);
    CHECK(world.ContactEventCounts[box] == 0); // and having reported them gone it says no more
}

TEST_CASE("two runs report the same contact events in the same order") {
    // Events are what a host hangs behaviour off, so a replay that reports them in a different order
    // is not a replay. Nothing appends atomically and nothing sorts: each body writes its own fixed
    // run in slot order, so the layout is settled before any thread starts.
    const mtl::Context context;
    Solver solver{context};

    const auto run = [&context, &solver] {
        World world{context};
        // A box that tips onto a corner and rolls, so features come and go all through the run rather
        // than a manifold forming once and holding.
        const auto box = DropBox(world, 1.3f, true);
        world.Poses[box].Orientation = QuatFromRotationVector(float3{0.4f, 0.1f, 0.25f});
        world.Velocities[box] = {.Linear = {0.7f, 0, -0.3f}, .Angular = {0.2f, 1.1f, 0}};
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
    REQUIRE(first.size() > 8); // it did report something worth comparing
    for (size_t at = 0; at < first.size(); ++at) {
        CAPTURE(at);
        CHECK(std::memcmp(&first[at], &second[at], sizeof(ContactEvent)) == 0);
    }
}

TEST_CASE("a body with more contacts than slots says so instead of dropping them quietly") {
    // A body owns a fixed run of ContactsPerBody and a dense pile can want more. What it must not do is
    // hold fewer contacts than it is touching without saying so, which from anywhere but here looks
    // like the solver having lost them.
    const StepSettings settings{.Gravity = {0, 0, 0}}; // nothing moves, so it is the geometry being counted
    const mtl::Context context;
    Solver solver{context};
    World world{context};
    const auto shape = world.AddShape(UnitBox);
    // The middle one is the only dynamic body, so it owns every manifold. Eight face partners fill the
    // run exactly and a run filled to the brim is not an overflow, so it takes nine - two boxes to a
    // side, each overlapping half of it. The sphere lands the arithmetic off the boundary: it touches
    // at one point, so the run fills part way through a manifold and the remainder is the refusal.
    const auto middle = world.AddBody({.Shape = shape});
    for (const auto at : {float3{0.999f, 0, 0.5f}, float3{0.999f, 0, -0.5f}, float3{-0.999f, 0, 0.5f}, float3{-0.999f, 0, -0.5f},
                          float3{0.5f, 0, 0.999f}, float3{-0.5f, 0, 0.999f}, float3{0.5f, 0, -0.999f}, float3{-0.5f, 0, -0.999f},
                          float3{0, 0.999f, 0}})
        world.AddBody({.Pose = {.Position = at, .Orientation = {0, 0, 0, 1}}, .Shape = shape, .Density = 0});
    world.AddBody({.Pose = {.Position = {0, -1.24f, 0}, .Orientation = {0, 0, 0, 1}},
                   .Shape = world.AddShape({.Radius = 0.75f, .Kind = ShapeSphere}), .Density = 0});

    Run(solver, world, 1, settings);
    REQUIRE(ActiveContacts(world, middle) == ContactsPerBody); // it filled every slot it had
    // Four points from each of nine faces and one from the sphere is thirty-seven wanted. Exact rather
    // than a lower bound: every partner is collided whether or not the run is full, because which
    // contacts a body keeps must not depend on which it happened to see first.
    CHECK(world.ContactRefusals[middle] == 37 - ContactsPerBody);
}

namespace {
// A stack of two boxes on the plane, settled. Every manifold here belongs to the lower box: a pair is
// owned by its lower-indexed body, and a plane never owns one.
struct Stacked {
    Index Lower, Upper;
};
Stacked TwoOnAPlane(World &world) {
    const auto shape = world.AddShape(UnitBox);
    world.AddBody({.Shape = world.AddShape(GroundPlane)});
    return {world.AddBody({.Pose = {.Position = {0, Half, 0}, .Orientation = {0, 0, 0, 1}}, .Shape = shape}),
            world.AddBody({.Pose = {.Position = {0, Half + 1, 0}, .Orientation = {0, 0, 0, 1}}, .Shape = shape})};
}
} // namespace

TEST_CASE("a removed body's contacts end, and every one of them is reported gone") {
    // Removal needs no kernel to know about it: a body with no shape and no mass is a state every
    // per-body kernel already early-outs on, and EndUnclaimed already reports what a body that has
    // stopped colliding was holding. What this checks is the routing from both sides of a manifold.
    const mtl::Context context;
    Solver solver{context};
    World world{context};
    const auto [lower, upper] = TwoOnAPlane(world);
    Run(solver, world, 200);
    REQUIRE(ActiveContacts(world, lower) == 8); // four against the plane and four against the box above
    REQUIRE(ActiveContacts(world, upper) == 0);

    SUBCASE("the body underneath reports the pairs it was holding") {
        REQUIRE(world.RemoveBody(upper));
        Run(solver, world, 1);
        CHECK(Reported(world, lower, ContactRemoved) == 4); // the four against what has gone
        CHECK(Reported(world, lower, ContactPersisted) == 4); // and the plane, which has not
        CHECK(Reported(world, lower, ContactAdded) == 0);
        CHECK(world.ContactEventCounts[upper] == 0); // it owned none of them, so it reports none
        CHECK(ActiveContacts(world, lower) == 4);
    }

    SUBCASE("and a body that owned them reports all of them itself") {
        REQUIRE(world.RemoveBody(lower));
        Run(solver, world, 1);
        CHECK(Reported(world, lower, ContactRemoved) == 8);
        CHECK(Reported(world, lower, ContactAdded) == 0);
        CHECK(Reported(world, lower, ContactPersisted) == 0);
        // And nothing anywhere is still holding one against it, since its run is what held them all.
        CHECK(std::ranges::none_of(world.Contacts.All(), [lower = lower](const Contact &contact) {
            return contact.Active && (contact.BodyA == lower || contact.BodyB == lower);
        }));
    }
}

TEST_CASE("what a removed body was holding up wakes and falls") {
    // Waking spreads from a body that is moving to the ones it touches, and a body that has been
    // removed is not moving - so nothing would ever tell a sleeping stack that what it was standing on
    // has gone, and it would sleep on in mid air.
    const StepSettings settings{};
    const mtl::Context context;
    Solver solver{context};
    World world{context};
    const auto [lower, upper] = TwoOnAPlane(world);
    Run(solver, world, 400);
    REQUIRE(world.Quiet[upper] >= settings.SleepSteps);

    SUBCASE("the box under it") {
        REQUIRE(world.RemoveBody(lower));
        CHECK(world.Quiet[upper] == 0); // told at once, by the removal itself
        Run(solver, world, 300);
        CheckResting(world.Poses[upper].Position.y); // and it fell the metre to the plane
    }

    SUBCASE("or the ground itself") {
        // Static geometry taken away, which is the same routing from the other end: a plane owns no
        // manifold, so everything it is holding names it as B and is found through the incoming list.
        REQUIRE(world.RemoveBody(0));
        CHECK(world.Quiet[lower] == 0); // the box that was on it, told by the removal
        CHECK(world.Quiet[upper] != 0); // and the box on that one is a link further away
        Run(solver, world, 1);
        CHECK(world.Quiet[upper] == 0); // which is one step, since nothing sleeps on something moving
        Run(solver, world, 60);
        CHECK(world.Poses[lower].Position.y < 0); // both of them falling, with nothing left to catch them
        CHECK(world.Poses[upper].Position.y < 1);
    }
}

TEST_CASE("a slot handed out again behaves as a fresh one") {
    // The same scene twice, once into slots two other boxes lived and settled in first. A slot carries
    // a colour, a rest pose, a contact run and a sleep counter, and anything of that left behind shows
    // up as a scene that steps differently from the same scene built fresh.
    const mtl::Context context;
    Solver solver{context};
    const auto run = [&context, &solver](bool recycled) {
        World world{context};
        const auto shape = world.AddShape(UnitBox);
        world.AddBody({.Shape = world.AddShape(GroundPlane)});
        if (recycled) {
            const auto first = world.AddBody({.Pose = {.Position = {0, Half, 0}, .Orientation = {0, 0, 0, 1}}, .Shape = shape});
            const auto second = world.AddBody({.Pose = {.Position = {0, Half + 1.02f, 0}, .Orientation = {0, 0, 0, 1}}, .Shape = shape});
            Run(solver, world, 90); // long enough to settle, take colours and sleep
            // A colour is the one lane a step reads before it writes - incremental colouring starts
            // from it, and the highest any body holds is how many colours the next step dispatches. A
            // stack of two takes 0 and 1 on its own, so these are set to what a denser scene would
            // have left behind rather than building a raft here to produce it.
            world.Colors[first] = 3;
            world.Colors[second] = 2;
            REQUIRE(world.RemoveBody(first));
            REQUIRE(world.RemoveBody(second));
            Run(solver, world, 1); // the step that reports them gone and hands the slots back
            REQUIRE(world.BodyCount() == 1);
        }
        // Off-axis and tumbling, so anything that steps differently at all steps differently here.
        const auto box = world.AddBody({.Pose = {.Position = {0.2f, 1.4f, -0.1f}, .Orientation = QuatFromRotationVector(float3{0.3f, 0.1f, 0.2f})},
                                        .Velocity = {.Linear = {0.4f, 0, -0.2f}, .Angular = {0.1f, 0.7f, 0}},
                                        .Shape = shape});
        const auto other = world.AddBody({.Pose = {.Position = {0, 3, 0}, .Orientation = {0, 0, 0, 1}}, .Shape = shape});
        REQUIRE(box == 1);
        REQUIRE(other == 2);
        Run(solver, world, 150);
        return std::vector<Pose>{world.Poses.All().begin(), world.Poses.All().begin() + world.BodyCount()};
    };

    const auto fresh = run(false), recycled = run(true);
    REQUIRE(fresh.size() == recycled.size());
    for (size_t body = 0; body < fresh.size(); ++body) {
        CAPTURE(body);
        CHECK(std::memcmp(&fresh[body], &recycled[body], sizeof(Pose)) == 0);
    }
}

TEST_CASE("a slot is not handed out again until a step has reported what left it") {
    // One step of a slot standing idle is what makes the events unambiguous: the body that was holding
    // the contacts is the one that reports them gone, rather than whoever moved in on top of them.
    const mtl::Context context;
    Solver solver{context};
    World world{context};
    const auto shape = world.AddShape(UnitBox);
    world.AddBody({.Shape = world.AddShape(GroundPlane)});
    const auto first = world.AddBody({.Pose = {.Position = {0, Half, 0}, .Orientation = {0, 0, 0, 1}}, .Shape = shape});
    world.AddBody({.Pose = {.Position = {3, Half, 0}, .Orientation = {0, 0, 0, 1}}, .Shape = shape});
    Run(solver, world, 30);

    REQUIRE(world.RemoveBody(first));
    CHECK(!world.Alive(first));
    const auto immediate = world.AddBody({.Pose = {.Position = {6, Half, 0}, .Orientation = {0, 0, 0, 1}}, .Shape = shape});
    CHECK(immediate != first); // still occupied by a body with removals to report
    Run(solver, world, 1);
    const auto later = world.AddBody({.Pose = {.Position = {9, Half, 0}, .Orientation = {0, 0, 0, 1}}, .Shape = shape});
    CHECK(later == first); // and now it is back
    CHECK(world.Alive(later));
}

TEST_CASE("a retired joint stops holding, and lets its bodies collide again") {
    // Two boxes overlapping by half their width, which a joint holds them in - and which is why a joint
    // suppresses contacts between its two bodies. Retiring it has to undo that entry in both bodies'
    // runs: leave it behind and the two go on not colliding for ever, with nothing left to say why.
    const StepSettings settings{.Gravity = {0, 0, 0}}; // so the only force in the scene is the contact
    const mtl::Context context;
    Solver solver{context};
    World world{context};
    const auto shape = world.AddShape(UnitBox);
    const auto anchor = world.AddBody({.Shape = shape, .Density = 0});
    const auto hanging = world.AddBody({.Pose = {.Position = {0.5f, 0, 0}, .Orientation = {0, 0, 0, 1}}, .Shape = shape});
    const auto joint = world.AddJoint({.BodyA = hanging, .BodyB = anchor, .At = {0.5f, 0, 0}});
    REQUIRE(joint != NoIndex);
    Run(solver, world, 60, settings);
    REQUIRE(world.Poses[hanging].Position.x == doctest::Approx(0.5f).epsilon(0.01)); // held where it was

    SUBCASE("retired directly") {
        REQUIRE(world.RemoveJoint(joint));
        CHECK(!world.RemoveJoint(joint)); // and only once
        CHECK(world.JointCount() == 0); // the slot came back, and it was the last of them
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

TEST_CASE("the same mutation script steps to bit-identical state twice") {
    // Which slot an add lands in is a function of what was removed before it, so a world that has been
    // mutated is only reproducible if the free list is. Same script, same bits.
    const mtl::Context context;
    Solver solver{context};
    const auto run = [&context, &solver] {
        World world{context};
        const auto shape = world.AddShape(UnitBox);
        world.AddBody({.Shape = world.AddShape(GroundPlane)});
        for (uint32_t i = 0; i < 3; ++i)
            world.AddBody({.Pose = {.Position = {0, Half + 1.02f * float(i), 0}, .Orientation = {0, 0, 0, 1}}, .Shape = shape});
        Run(solver, world, 60);
        REQUIRE(world.RemoveBody(2)); // out of the middle of the stack, so the one above it falls
        Run(solver, world, 20);
        world.AddBody({.Pose = {.Position = {0.3f, 4, 0.2f}, .Orientation = {0, 0, 0, 1}}, .Shape = shape});
        Run(solver, world, 120);
        return std::vector<Pose>{world.Poses.All().begin(), world.Poses.All().begin() + world.BodyCount()};
    };

    const auto first = run(), second = run();
    REQUIRE(first.size() == second.size());
    for (size_t body = 0; body < first.size(); ++body) {
        CAPTURE(body);
        CHECK(std::memcmp(&first[body], &second[body], sizeof(Pose)) == 0);
    }
}

TEST_CASE("the same scene steps to bit-identical state twice") {
    const mtl::Context context;
    Solver solver{context};

    const auto run = [&context, &solver] {
        World world{context};
        const auto box = DropBox(world, 1.3f, true);
        // Off-axis so the box tips, lands on a corner and rolls: every asymmetry the slice can make.
        world.Poses[box].Orientation = QuatFromRotationVector(float3{0.4f, 0.1f, 0.25f});
        world.Velocities[box] = {.Linear = {0.7f, 0, -0.3f}, .Angular = {0.2f, 1.1f, 0}};
        Run(solver, world, 90);
        return std::vector<Pose>{world.Poses.All().begin(), world.Poses.All().begin() + world.BodyCount()};
    };

    const auto first = run(), second = run();
    REQUIRE(first.size() == second.size());
    for (size_t body = 0; body < first.size(); ++body) {
        CAPTURE(body);
        // Bit-identical, not merely close: nothing in the step depends on which thread got there first.
        CHECK(std::memcmp(&first[body], &second[body], sizeof(Pose)) == 0);
    }
}

TEST_CASE("frictionless contact conserves motion across the plane and spin about it") {
    // A frictionless plane contact can only push along its normal, so its Jacobian has no component
    // across the surface and none about the normal axis: dropping a spinning, drifting box onto it
    // must leave both untouched however hard the landing is. Dropped level so it never tips, since a
    // large rotation resolved in one implicit step does not preserve angular velocity exactly.
    const mtl::Context context;
    Solver solver{context};
    World world{context};
    const auto box = DropBox(world, 1.4f, true, 0); // frictionless, which is what is being measured
    constexpr float3 Drift{0.7f, 0, -0.3f};
    constexpr float Spin = 1.1f;
    world.Velocities[box] = {.Linear = Drift, .Angular = {0, Spin, 0}};

    Run(solver, world, 200); // land, then slide and spin for another two seconds
    const auto &velocity = world.Velocities[box];
    // A part in ten thousand rather than exactly: velocity is recovered by differencing two nearby
    // positions, so by three metres out a float32 position resolves to about 2e-7 against a step of
    // 3e-3, which is most of this tolerance. The conservation is exact, the arithmetic reporting it not.
    CHECK(velocity.Linear.x == doctest::Approx(Drift.x).epsilon(1e-3));
    CHECK(velocity.Linear.z == doctest::Approx(Drift.z).epsilon(1e-3));
    CHECK(velocity.Angular.y == doctest::Approx(Spin).epsilon(1e-3));

    // Resting on the plane the whole time it does so, and never picking up a tip.
    CheckResting(world.Poses[box].Position.y);
    CHECK(std::abs(velocity.Linear.y) < 1e-3f);
    CHECK(std::abs(velocity.Angular.x) < 1e-4f);
    CHECK(std::abs(velocity.Angular.z) < 1e-4f);
}

TEST_CASE("a tumbling box settles onto the plane instead of running away") {
    // Nothing here has a closed form, so this asserts only what has to be true: with friction it ends
    // up resting on the surface, upright on some face, and no longer moving at all.
    const mtl::Context context;
    Solver solver{context};
    World world{context};
    const auto box = DropBox(world, 1.3f, true);
    world.Poses[box].Orientation = QuatFromRotationVector(float3{0.4f, 0.1f, 0.25f});
    world.Velocities[box] = {.Linear = {0.7f, 0, -0.3f}, .Angular = {0.2f, 1.1f, 0}};

    Run(solver, world, 300);
    CheckResting(world.Poses[box].Position.y); // flat on a face
    CHECK(simd::length(world.Velocities[box].Linear) < 1e-2f);
    CHECK(simd::length(world.Velocities[box].Angular) < 1e-2f);
}

TEST_CASE("static friction holds a box on a slope inside the cone and lets it go outside") {
    // Coulomb's law in its oldest form: a body stays put while tan(slope) < mu, whatever the mass,
    // and once past it accelerates at g (sin - mu cos). Both sides of the angle, since a solver that
    // sticks everything passes the first half on its own.
    constexpr float Mu = 0.5f;
    const float cone = std::atan(Mu), gravity = std::abs(StepSettings{}.Gravity.y);

    SUBCASE("inside the cone it does not creep") {
        const auto settings = Tilted(cone * 0.6f);
        const mtl::Context context;
        Solver solver{context};
        World world{context};
        const auto box = DropBox(world, Half, true, Mu);

        Run(solver, world, 30, settings);
        const float from = world.Poses[box].Position.x;
        Run(solver, world, 240, settings); // four seconds is long enough for a creep to show
        CHECK(std::abs(world.Poses[box].Position.x - from) < 1e-3f);
        CHECK(simd::length(world.Velocities[box].Linear) < 1e-3f);
        CheckResting(world.Poses[box].Position.y);
    }

    SUBCASE("outside it slides at the closed form's acceleration") {
        // 34 degrees: past the 26.6 the cone allows, short of the 45 at which a cube tips instead of
        // sliding, so the box is in steady sliding and the closed form is the whole story.
        constexpr float Slope = 0.6f;
        const auto settings = Tilted(Slope);
        const float expected = gravity * (std::sin(Slope) - Mu * std::cos(Slope));
        const mtl::Context context;
        Solver solver{context};
        World world{context};
        const auto box = DropBox(world, Half, true, Mu);

        Run(solver, world, 30, settings); // it is already sliding by the end of this, not at rest
        const float was = world.Velocities[box].Linear.x;
        constexpr uint32_t Steps = 60;
        Run(solver, world, Steps, settings);
        const float elapsed = Steps * settings.DeltaTime;
        CHECK((world.Velocities[box].Linear.x - was) / elapsed == doctest::Approx(expected).epsilon(0.05));
        CHECK(simd::length(world.Velocities[box].Angular) < 1e-2f); // sliding flat, not tumbling
    }
}

TEST_CASE("a sliding box decelerates at mu g and stops where the closed form says") {
    // Coulomb friction takes mu m g off a sliding box, so it decelerates at mu g regardless of mass
    // and travels v0^2 / (2 mu g) before stopping. Neither number involves the solver.
    constexpr float Mu = 0.5f, Speed = 2;
    const mtl::Context context;
    Solver solver{context};
    World world{context};
    const auto box = DropBox(world, Half, true, Mu);

    Run(solver, world, 30); // let it settle onto the plane first, so nothing is bouncing
    const float from = world.Poses[box].Position.x;
    world.Velocities[box].Linear = {Speed, 0, 0};

    const float gravity = std::abs(StepSettings{}.Gravity.y);
    Run(solver, world, 120); // twice as long as v0 / (mu g), so it is stopped and staying stopped
    CHECK(world.Poses[box].Position.x - from == doctest::Approx(Speed * Speed / (2 * Mu * gravity)).epsilon(0.05));
    CHECK(std::abs(world.Velocities[box].Linear.x) < 1e-2f);

    // It slid rather than tipped or crept sideways.
    CheckResting(world.Poses[box].Position.y);
    CHECK(std::abs(world.Poses[box].Position.z - 0) < 1e-3f);
    CHECK(simd::length(world.Velocities[box].Angular) < 1e-2f);
}

TEST_CASE("a fast box does not tunnel through the plane") {
    const mtl::Context context;
    Solver solver{context};
    World world{context};
    constexpr float Speed = 40; // far enough per step to cross the plane outright
    const auto box = DropBox(world, 3, true);
    world.Velocities[box].Linear = {0, -Speed, 0};

    float lowest = 1e9f;
    for (uint32_t step = 0; step < 120; ++step) {
        solver.Step(world);
        lowest = std::min(lowest, world.Poses[box].Position.y);
    }
    // It is stopped by the plane rather than passing through it, and caught within the step that
    // carried it in: contacts are found at the pose the step began from, so a step's worth of motion
    // is the whole of what a body can bury itself by.
    CHECK(lowest > Half - Speed * StepSettings{}.DeltaTime);
    CHECK(lowest > -Half); // and nowhere near through to the other side
    CheckResting(world.Poses[box].Position.y);
}

TEST_CASE("a full run gives its shallowest place to a deeper contact and no other") {
    // The eviction rule on its own. What a full run does with the next point is the whole of what keeps
    // load-bearing geometry from being decided by which partner was numbered first.
    //
    // The scene fills a run exactly - a slab with eight blocks hovering a millimetre over it, eight
    // manifolds of four - and then adds the two cases that matter: a support the slab is resting on,
    // whose points are deeper than anything held and have to take a place, and a ninth hoverer at the
    // same millimetre as the eight, which is a tie and has to be refused. Which four give way is a tie
    // too, broken by slot, so a tie is settled by something fixed.
    constexpr Shape Slab{.HalfExtents = {3, Half, 3}, .Kind = ShapeBox};
    constexpr float Gap = 1e-3f; // inside a step's reach, so the contact exists, and clear of the floor
    REQUIRE(ContactsPerBody == 32);
    REQUIRE(ManifoldPoints == 4);

    const mtl::Context context;
    Solver solver{context};
    World world{context};
    const auto slab = world.AddShape(Slab), block = world.AddShape(UnitBox);
    // The slab is dynamic and lowest, so it owns every pair in the scene and they are collided in the
    // order the partners were added.
    const auto subject = world.AddBody({.Pose = {.Position = {0, 0, 0}, .Orientation = {0, 0, 0, 1}}, .Shape = slab});
    std::vector<Index> hovering;
    for (const float x : {-2.f, -0.7f, 0.7f, 2.f})
        for (const float z : {-1.5f, 1.5f})
            hovering.push_back(world.AddBody(
                {.Pose = {.Position = {x, 2 * Half + Gap, z}, .Orientation = {0, 0, 0, 1}}, .Shape = block, .Density = 0}));
    REQUIRE(hovering.size() * ManifoldPoints == ContactsPerBody);
    const auto support = world.AddBody({.Pose = {.Position = {0, -2 * Half, 0}, .Orientation = {0, 0, 0, 1}}, .Shape = slab, .Density = 0});
    const auto late = world.AddBody(
        {.Pose = {.Position = {0, 2 * Half + Gap, 0}, .Orientation = {0, 0, 0, 1}}, .Shape = block, .Density = 0});

    const auto against = [&world, subject](Index other) {
        uint32_t held = 0;
        for (uint32_t slot = 0; slot < ContactsPerBody; ++slot) {
            const Contact &contact = world.Contacts[subject * ContactsPerBody + slot];
            held += contact.Active && contact.BodyB == other ? 1 : 0;
        }
        return held;
    };

    Run(solver, world, 1);
    // Four points asked for a full run and were counted, and the four that took a place are the ones
    // the slab is standing on.
    CHECK(world.ContactRefusals[subject] == 2 * ManifoldPoints);
    CHECK(against(support) == ManifoldPoints);
    CHECK(against(late) == 0); // the tie went to what was already there
    CHECK(against(hovering.front()) == 0); // and the four it displaced were the lowest slots
    for (uint32_t i = 1; i < hovering.size(); ++i) {
        CAPTURE(i);
        CHECK(against(hovering[i]) == ManifoldPoints);
    }

    // And it stays that way: the slab rests on what it is standing on rather than creeping through it
    // while its run is spent on eight contacts that carry nothing.
    Run(solver, world, 180);
    CHECK(against(support) == ManifoldPoints);
    CheckResting(world.Poses[subject].Position.y + Half); // its underside on the support's top face
    CHECK(simd::length(world.Velocities[subject].Linear) < 1e-3f);
}

// Hulls. The oracle for every one of these is a shape the engine already had: a cube given as eight
// points has to behave like the Box of the same size, and where it meets a box, a sphere or a capsule
// it has to land where those pairs already land. Nothing here trusts the new path to grade itself.
namespace {
// How a body ended up, and on how many points, which is the whole of what these compare.
struct Settled {
    float3 Position;
    float3 Normal;
    uint32_t Contacts;
};

Settled Rest(const World &world, Index body) {
    const auto slots = world.Contacts.All().subspan(body * ContactsPerBody, ContactsPerBody);
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

TEST_CASE("a cube given as a hull rests where the same cube given as a box does") {
    const mtl::Context context;
    Solver solver{context};
    const auto drop = [&](bool as_hull) {
        World world{context};
        world.AddBody({.Shape = world.AddShape(GroundPlane)});
        const auto shape = as_hull ? world.AddHull(CubeCorners(2 * Half)) : world.AddShape(UnitBox);
        REQUIRE(shape != NoIndex);
        const auto body = world.AddBody({.Pose = {.Position = {0, 2, 0}, .Orientation = {0, 0, 0, 1}}, .Shape = shape});
        Run(solver, world, 180);
        CHECK(world.ContactRefusals[body] == 0);
        return Rest(world, body);
    };

    const Settled box = drop(false), hull = drop(true);
    // The same solid by another description, so the same answer - not merely a plausible one.
    CheckResting(hull.Position.y);
    CHECK(hull.Position.y == doctest::Approx(box.Position.y).epsilon(1e-5));
    CHECK(hull.Contacts == box.Contacts);
    CHECK(hull.Contacts == 4); // its bottom face, one contact a corner
    CHECK(hull.Normal.y == doctest::Approx(1).epsilon(1e-5));
}

TEST_CASE("a hull's frame finds the geometry as given after the body has moved") {
    // A body's pose is the pose of the frame the cook chose, not of the points the caller handed in,
    // and a caller holding those points has to be able to find them again - which is what the frame is
    // for. The check a renderer makes: the points as given, taken through the frame and then the body's
    // pose, are the collision geometry the solver settled, resting on the plane.
    const mtl::Context context;
    Solver solver{context};
    World world{context};
    world.AddBody({.Shape = world.AddShape(GroundPlane)});
    // A brick rather than a cube, so its principal axes are its own rather than any three orthogonal
    // ones - and left off the origin and turned, the way a modeller leaves a mesh.
    const float4 turn = QuatFromRotationVector(float3{0.3f, -0.7f, 0.2f});
    const float3 half{0.25f, 0.375f, 0.5f};
    std::vector<float3> points;
    for (uint32_t corner = 0; corner < 8; ++corner)
        points.push_back(float3{7, -3, 11} +
                         Rotate(turn, half * float3{(corner & 1) ? 1.f : -1.f, (corner & 2) ? 1.f : -1.f, (corner & 4) ? 1.f : -1.f}));

    Pose frame{};
    const auto shape = world.AddHull(points, &frame);
    REQUIRE(shape != NoIndex);
    const auto body = world.AddBody({.Pose = {.Position = {0, 2, 0}, .Orientation = {0, 0, 0, 1}}, .Shape = shape});
    Run(solver, world, 240);

    const Pose pose = world.Poses[body];
    const auto vertices = world.ShapeVertices.All().subspan(world.Shapes[shape].FirstVertex, world.Shapes[shape].VertexCount);
    float lowest = INFINITY;
    for (const float3 point : points) {
        const float3 in_shape = Rotate(QuatConjugate(frame.Orientation), point - frame.Position);
        const float3 in_world = pose.Position + Rotate(pose.Orientation, in_shape);
        lowest = std::min(lowest, float(in_world.y));
        // The same corner the engine is colliding, since the frame is exactly the transform it applied.
        float nearest = INFINITY;
        for (const float3 vertex : vertices)
            nearest = std::min(nearest, float(simd::distance(in_world, pose.Position + Rotate(pose.Orientation, vertex))));
        CHECK(nearest < 1e-4f);
    }
    CheckResting(lowest + Half); // its lowest corner is on the plane, and not through it
}

TEST_CASE("where a hull rests is a property of the solid, not of the frame it arrived in") {
    // A hull's faces are recovered in the cooked frame and named by cooked vertex indices, so a
    // caller's frame leaking into either shows here first - a plane needs no face recovered, so the
    // resting-height tests do not ask it.
    //
    // The oracle is an identity: the same solid handed in from two frames is one solid. Not a
    // plausibility bound - the two runs must agree on where it rests and on what is touching.
    constexpr float Table = 0.5f;
    const mtl::Context context;
    Solver solver{context};
    // A wedge: no two faces the same and no symmetry to hide a wrong axis behind.
    const std::vector<float3> shape{float3{-0.4f, -0.15f, -0.3f}, float3{0.5f, -0.15f, -0.3f}, float3{0.5f, -0.15f, 0.35f},
                                    float3{-0.4f, -0.15f, 0.35f}, float3{-0.2f, 0.25f, -0.1f}, float3{0.3f, 0.25f, 0.2f}};

    const auto drop = [&](float4 turn, float3 move, bool on_a_box = true) {
        World world{context};
        world.AddBody({.Shape = world.AddShape(GroundPlane)});
        // A box to land on, so the manifold goes through the face recovery rather than the plane path.
        constexpr Shape Top{.HalfExtents = {2, Table, 2}, .Kind = ShapeBox};
        if (on_a_box)
            world.AddBody({.Pose = {.Position = {0, Table, 0}, .Orientation = {0, 0, 0, 1}}, .Shape = world.AddShape(Top), .Density = 0});
        std::vector<float3> points;
        for (const float3 point : shape) points.push_back(move + Rotate(turn, point));
        Pose frame{};
        const auto hull = world.AddHull(points, &frame);
        REQUIRE(hull != NoIndex);
        // Placed through the frame the cook chose, so the *caller's own* points land in the same world
        // positions in both runs - `shape` itself, lifted. A principal axis is only defined up to its
        // sign, so a body set down with an identity orientation is not the same solid either way up.
        const float3 lift{0, 2 * Table + 0.15f + 0.02f, 0}; // its base is 0.15 below the origin of `shape`
        const auto body = world.AddBody({.Pose = {.Position = lift - Rotate(QuatConjugate(turn), move - frame.Position),
                                                  .Orientation = QuatMul(QuatConjugate(turn), frame.Orientation)},
                                         .Shape = hull});
        // The placement above has to put the caller's own points at `shape` lifted, in both runs.
        {
            const Pose start = world.Poses[body];
            for (uint32_t i = 0; i < points.size(); ++i) {
                const float3 in_shape = Rotate(QuatConjugate(frame.Orientation), points[i] - frame.Position);
                const float3 at = start.Position + Rotate(start.Orientation, in_shape);
                CAPTURE(i);
                CHECK(simd::distance(at, shape[i] + lift) < 1e-4f);
            }
        }
        Run(solver, world, 300);
        CHECK(world.ContactRefusals[body] == 0);

        // Where each of the caller's own points ended up, taken back through the frame the cook chose.
        // Sorted, since the cook keeps the corners in its own order and the question is about the solid.
        const Pose pose = world.Poses[body];
        std::vector<float> heights;
        for (const float3 point : points) {
            const float3 in_shape = Rotate(QuatConjugate(frame.Orientation), point - frame.Position);
            heights.push_back(float((pose.Position + Rotate(pose.Orientation, in_shape)).y));
        }
        std::ranges::sort(heights); // the cook keeps the corners in its own order; the solid has no order
        return std::pair{heights, Rest(world, body)};
    };

    const float4 spin = QuatFromRotationVector(float3{0.8f, -1.3f, 0.55f});
    const auto [square, square_rest] = drop(float4{0, 0, 0, 1}, float3{0, 0, 0});
    // Turned and moved thirty units out, which is where a modeller leaves a mesh and where the cook's
    // own tolerances stop being finer than the points it was handed.
    const auto [turned, turned_rest] = drop(spin, float3{31, -17, 8});

    // Every corner of the solid at the same height in both, which is the whole claim: one solid, one
    // answer, however it was handed in.
    REQUIRE(square.size() == turned.size());
    for (uint32_t i = 0; i < square.size(); ++i) {
        CAPTURE(i);
        CHECK(square[i] == doctest::Approx(turned[i]).epsilon(2e-3).scale(0));
    }
    CHECK(square_rest.Contacts == turned_rest.Contacts);
    CHECK(square_rest.Contacts >= 3); // it settled on a face rather than teetering
    CHECK(square[0] == doctest::Approx(2 * Table).epsilon(1e-3).scale(0)); // its lowest corner on the box's top
    CHECK(square_rest.Normal.y == doctest::Approx(1).epsilon(1e-3));
}

TEST_CASE("a hull and a box rest on each other the same way round either way") {
    // Which of the two presents the reference face is decided by the geometry rather than by which is
    // body A, so a hull on a box and a box on a hull are the same contact seen from either side.
    const mtl::Context context;
    Solver solver{context};
    const auto stack = [&](bool hull_on_top) {
        World world{context};
        world.AddBody({.Shape = world.AddShape(GroundPlane)});
        const auto hull = world.AddHull(CubeCorners(2 * Half));
        const auto box = world.AddShape(UnitBox);
        REQUIRE(hull != NoIndex);
        world.AddBody({.Pose = {.Position = {0, Half, 0}, .Orientation = {0, 0, 0, 1}}, .Shape = hull_on_top ? box : hull, .Density = 0});
        const auto top = world.AddBody({.Pose = {.Position = {0, 1.4f, 0}, .Orientation = {0, 0, 0, 1}}, .Shape = hull_on_top ? hull : box});
        Run(solver, world, 240);
        CHECK(world.ContactRefusals[top] == 0);
        return Rest(world, top);
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

TEST_CASE("a tetrahedron settles on the face it was dropped on") {
    // A hull that is not a box in disguise, and whose resting height is a property of its own
    // geometry: whatever the cook put its lowest vertex at is how far its centre of mass sits up.
    const mtl::Context context;
    Solver solver{context};
    World world{context};
    world.AddBody({.Shape = world.AddShape(GroundPlane)});
    // An equilateral base in the xz plane with an apex over its middle. Three-fold symmetry about y
    // makes that axis principal and leaves the other two interchangeable, so the cook has no rotation
    // to find and hands the shape back in the frame it arrived in - it is dropped already on a face.
    const float side = std::sqrt(3.f) / 2;
    const std::vector<float3> points{float3{1, 0, 0}, float3{-0.5f, 0, side}, float3{-0.5f, 0, -side}, float3{0, 1.6f, 0}};
    const auto shape = world.AddHull(points);
    REQUIRE(shape != NoIndex);
    const auto body = world.AddBody({.Pose = {.Position = {0, 1.5f, 0}, .Orientation = {0, 0, 0, 1}}, .Shape = shape});

    // The centre of mass of a tetrahedron is the mean of its corners, a quarter of the way up from the
    // base, so the cook leaves the base a quarter of the height below the origin - and that, not the
    // distance to the lowest corner, is how high a body resting on that face sits.
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

TEST_CASE("a sphere and a capsule rest on a hull as they rest on a box") {
    // The round path does not know about hulls, so a round shape against one goes through the convex
    // path with its core as a one or two point polytope and its radius taken off the answer. What that
    // has to produce is what a box already produces: one contact under a sphere, two under a capsule
    // lying across a face.
    const mtl::Context context;
    Solver solver{context};
    const auto drop = [&](bool on_hull, const Shape &shape, float height, float4 turn) {
        World world{context};
        world.AddBody({.Shape = world.AddShape(GroundPlane)});
        const auto floor = on_hull ? world.AddHull(CubeCorners(2)) : world.AddShape({.HalfExtents = {1, 1, 1}, .Kind = ShapeBox});
        REQUIRE(floor != NoIndex);
        world.AddBody({.Pose = {.Position = {0, 1, 0}, .Orientation = {0, 0, 0, 1}}, .Shape = floor, .Density = 0});
        const auto body = world.AddBody({.Pose = {.Position = {0, height, 0}, .Orientation = turn}, .Shape = world.AddShape(shape)});
        Run(solver, world, 240);
        CHECK(world.ContactRefusals[body] == 0);
        return Rest(world, body);
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

TEST_CASE("two boxes apart and crossed take their edge pair, not a face") {
    // The separating axis test biases towards the reference face it already had, relative and absolute
    // both, and the edge pair has to beat the best face by a clear margin. That bias compares negative
    // separations while the boxes overlap and positive ones once speculative reach builds a contact
    // across a gap, which is not the same arithmetic - so this is the case that says it still picks
    // right.
    //
    // Two boxes tilted a quarter turn about different axes present nothing but a pair of crossed edges.
    // The contact direction is straight up and every face either box has points forty-five degrees away
    // from it, so a tolerance that picked a face would be unmissable.
    constexpr float Diagonal = 0.70710678f; // half a unit square's diagonal, the tilted box's ridge
    const float tilt = std::numbers::pi_v<float> / 4; // onto an edge, not a quarter turn back onto a face
    const mtl::Context context;
    Solver solver{context};
    World world{context};
    const auto shape = world.AddShape(UnitBox);
    world.AddBody({.Pose = {.Position = {0, 0, 0}, .Orientation = QuatFromRotationVector(float3{0, 0, tilt})},
                   .Shape = shape,
                   .Density = 0});
    // Set down a clear gap above and moving fast enough that the reach covers it while still apart.
    const auto box = world.AddBody({.Pose = {.Position = {0, 2 * Diagonal + 0.08f, 0}, .Orientation = QuatFromRotationVector(float3{tilt, 0, 0})},
                                    .Velocity = {.Linear = {0, -3, 0}},
                                    .Shape = shape});

    // Read off C0 rather than where the box ends the step: the axis was chosen at the pose the step
    // began from and C0's normal row is that pose's separation plus the margin, so above the margin is
    // exactly a contact the two were still apart for.
    bool apart = false;
    for (uint32_t step = 0; step < 4; ++step) {
        solver.Step(world);
        CheckManifolds(world);
        uint32_t live = 0;
        for (uint32_t slot = box * ContactsPerBody; slot < (box + 1) * ContactsPerBody; ++slot) {
            const Contact &contact = world.Contacts[slot];
            if (!contact.Active) continue;
            ++live;
            if (contact.C0[0] <= StepSettings{}.ContactMargin) continue;
            apart = true;
            CAPTURE(step);
            CHECK(contact.Normal.y == doctest::Approx(1).epsilon(1e-3));
            CHECK(std::abs(contact.Normal.x) < 1e-3f); // and not the forty-five degrees a face would give
            CHECK(std::abs(contact.Normal.z) < 1e-3f);
        }
        if (apart) CHECK(live == 1); // two crossed edges meet in one place
    }
    CHECK(apart); // there really was a step holding a contact across a gap
}

TEST_CASE("two hulls meeting on their edges hold each other up") {
    // ConvexManifold's closest-pair branch, which runs when neither shape presents a face along the
    // contact normal - an edge across an edge, or a corner against one. Every other hull test rests
    // something flat and takes the face branch, and what a defect here costs is not a small position
    // error but a wrong contact direction.
    //
    // A single point where two edges cross is all the contact there is, so nothing about this is stable
    // and the test does not ask it to be. What has to hold is that the point is in the right place with
    // the right normal while the two meet, and that the body is never inside what it rested on.
    constexpr float Side = 1, Diagonal = Side * 0.70710678f; // half a square's diagonal, the ridge height
    const mtl::Context context;
    Solver solver{context};

    // A turn of `angle` about `axis`, as the pose carries it - the cook centres a hull and turns it
    // onto its principal axes, and a cube's are degenerate, so the tilt has to be the body's own.
    const auto turn = [](float3 axis, float angle) { return QuatFromRotationVector(simd::normalize(axis) * angle); };
    const float quarter = std::numbers::pi_v<float> / 4;

    // Set down at the height its own lowest feature touches the ridge at, and read while it is still
    // there: balanced on one point it topples, so a long run measures the aftermath not the contact.
    const auto meet = [&](float4 upper, float reach, uint32_t steps) {
        World world{context};
        const auto shape = world.AddHull(CubeCorners(Side));
        REQUIRE(shape != NoIndex);
        // A floor for whatever slides off, exactly where the tilted cube below stands on it.
        world.AddBody({.Shape = world.AddShape({.Normal = {0, 1, 0}, .Offset = -Diagonal, .Kind = ShapePlane})});
        // Tilted a quarter turn about z, so what it presents upward is a ridge running along z rather
        // than a face. Static, so the pair is the dynamic body's whatever the indices are.
        world.AddBody({.Pose = {.Position = {0, 0, 0}, .Orientation = turn(float3{0, 0, 1}, quarter)}, .Shape = shape, .Density = 0});
        const auto body = world.AddBody(
            {.Pose = {.Position = {0, Diagonal + reach, 0}, .Orientation = upper}, .Shape = shape}
        );
        Run(solver, world, steps);
        return std::pair{Rest(world, body), simd::length(world.Velocities[body].Linear)};
    };

    SUBCASE("edge across edge") {
        // The upper cube tilted the same quarter turn about x, so its lowest feature is a ridge running
        // along x - crossed with the one below at a right angle, and meeting it at one point.
        const auto [settled, speed] = meet(turn(float3{1, 0, 0}, quarter), Diagonal, 30);
        CHECK(settled.Contacts == 1); // two crossed edges meet in exactly one place
        CHECK(settled.Normal.y == doctest::Approx(1).epsilon(1e-3));
        // Each cube reaches half a diagonal from its centre to its ridge, so the two centres sit two
        // of those apart when the ridges touch.
        CHECK(settled.Position.y == doctest::Approx(2 * Diagonal).epsilon(2e-3));
        CHECK(std::abs(settled.Position.x) < 5e-3f);
        CHECK(std::abs(settled.Position.z) < 5e-3f);
    }

    SUBCASE("corner over edge, across the gap") {
        // The upper cube stood on a corner, so the lowest feature it presents is a single vertex and it
        // reaches half its body diagonal down. A vertex against a ridge is the same branch.
        //
        // Measured while the two are still apart, which is where the answer is well defined: a vertex
        // exactly on a ridge is the boundary between the two faces meeting there, so once they overlap
        // the normal is genuinely ambiguous - see the subcase below. Across a gap the nearest points
        // are unique and the direction between them is straight up.
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
        // Once the vertex is inside it is inside one of the two faces meeting at the ridge, so the
        // contact is that face's 45 degree normal - the geometry rather than a defect, since a cube
        // dropped on a knife edge does end up on one side of it. Both faces are steeper than the
        // friction cone, so it slides. What must not happen is a normal that leaves it inside.
        const float3 diagonal = simd::normalize(float3{1, 1, 1});
        const float3 down{0, -1, 0};
        const float4 corner_down = turn(simd::cross(diagonal, down), std::acos(simd::dot(diagonal, down)));
        const auto [settled, speed] = meet(corner_down, Side * 0.8660254f, 200);
        CHECK(std::abs(settled.Position.x) > 0.1f); // it left the ridge, one way or the other
        // And it is on the floor beside the static cube rather than inside either. The floor is a
        // diagonal below the origin, so a cube resting flat on it sits half a side above that and one
        // still up on an edge half a diagonal - it has to be between the two.
        CHECK(settled.Position.y > -Diagonal + Half - MaxPenetration);
        CHECK(settled.Position.y < -Diagonal + Diagonal + MaxPenetration);
        CHECK(speed < 0.1f); // and it stopped
    }

    SUBCASE("and it never ends up inside what it was resting on") {
        // Balanced on a point it topples, which is the physics. What would be a defect is a normal
        // pointing the wrong way while it does, pulling the body through what it is falling off.
        const auto [settled, speed] = meet(turn(float3{1, 0, 0}, quarter), Diagonal + 0.05f, 400);
        CHECK(settled.Position.y > -Diagonal); // below this it is inside the static cube, or past it
        CHECK(std::isfinite(settled.Position.x));
        CHECK(simd::length(settled.Position) < 20); // and it fell off rather than being thrown
    }
}

namespace {
// The lowest point of a body's collision geometry in the world, which for a hull resting on a plane is
// what has to be on the plane whichever of its faces it landed on.
float LowestVertex(const World &world, Index body) {
    const Shape &shape = world.Shapes[world.BodyShapes[body]];
    const Pose pose = world.Poses[body];
    float lowest = INFINITY;
    for (uint32_t i = 0; i < shape.VertexCount; ++i)
        lowest = std::min(lowest, float((pose.Position + Rotate(pose.Orientation, world.ShapeVertices[shape.FirstVertex + i])).y));
    return lowest;
}
} // namespace

TEST_CASE("a sphere-like hull rests on the face it lands on") {
    // The other end from the boxy hulls above: eighty faces whose normals are a few degrees apart,
    // which is the most the 1e-3 face tolerance can be asked within the vertex pool. Too coarse a face
    // and the hull rests on a plane that is not one of its own.
    //
    // The oracle needs no face at all: wherever it comes to rest its lowest vertex is on the plane, and
    // a body resting on a face rather than teetering on a vertex stops moving and stays put.
    constexpr float Radius = 0.5f;
    const mtl::Context context;
    Solver solver{context};
    // A sphere's principal axes are degenerate, so which way up the cook leaves it is not the caller's
    // to choose. Dropped as cooked it balances on a single vertex, nothing being asymmetric enough to
    // tip it. Turned, it finds a face.
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
    world.AddBody({.Shape = world.AddShape(GroundPlane)});
    const auto shape = world.AddHull(SpherePoints(Radius));
    REQUIRE(shape != NoIndex);
    REQUIRE(world.Shapes[shape].VertexCount == 42); // every point a corner, none swallowed by the hull
    const auto body = world.AddBody(
        {.Pose = {.Position = {0, 1.5f, 0}, .Orientation = QuatFromRotationVector(tilt)}, .Shape = shape});

    Run(solver, world, 300);
    const Settled settled = Rest(world, body);
    CHECK(world.ContactRefusals[body] == 0);
    // The manifold names exactly the geometry touching, which is what the face tolerance is under:
    // recover too wide a face and there are rows under vertices that are off the plane.
    uint32_t on_plane = 0;
    const Shape &geometry = world.Shapes[world.BodyShapes[body]];
    const Pose pose = world.Poses[body];
    const float lowest = LowestVertex(world, body);
    for (uint32_t i = 0; i < geometry.VertexCount; ++i)
        if (float((pose.Position + Rotate(pose.Orientation, world.ShapeVertices[geometry.FirstVertex + i])).y) < lowest + 1e-3f)
            ++on_plane;
    CHECK(on_plane == touching);
    CHECK(settled.Contacts == touching);
    CHECK(settled.Normal.y == doctest::Approx(1).epsilon(1e-3));
    CHECK(lowest < 0); // it is on the plane
    CHECK(lowest > -MaxPenetration); // and not through it
    CHECK(simd::length(world.Velocities[body].Linear) < 1e-3f);
    CHECK(simd::length(world.Velocities[body].Angular) < 1e-3f);

    // And it stays there rather than creeping off the face it found.
    const float3 was = world.Poses[body].Position;
    Run(solver, world, 300);
    CHECK(simd::distance(world.Poses[body].Position, was) < 1e-3f);
}

TEST_CASE("a face wider than eight points still holds a coin level") {
    // MaxFacePoints is eight and both routes into a manifold cap there, so a sixteen-sided coin lying
    // flat has both caps biting. What they cost is a support polygon narrower than the geometry's: if
    // the eight kept were a contiguous run around the rim it would be half a disc, and the coin would
    // tip off the side its face does not cover.
    //
    // So the claim is not that the manifold is complete. It is that whatever eight it keeps, and the
    // four reduction leaves of them, still carry the coin's own centre of mass.
    constexpr float Radius = 0.5f, HalfHeight = 0.1f;
    const mtl::Context context;
    Solver solver{context};
    World world{context};
    bool on_a_box = false;
    SUBCASE("on a plane, where the cap is the plane path's") { on_a_box = false; }
    SUBCASE("on a box, where it is the recovered face's") { on_a_box = true; }

    world.AddBody({.Shape = world.AddShape(GroundPlane)});
    float table = 0;
    if (on_a_box) {
        constexpr Shape Table{.HalfExtents = {2, Half, 2}, .Kind = ShapeBox};
        world.AddBody({.Pose = {.Position = {0, Half, 0}, .Orientation = {0, 0, 0, 1}}, .Shape = world.AddShape(Table), .Density = 0});
        table = 2 * Half;
    }
    const auto shape = world.AddHull(PrismPoints(16, Radius, HalfHeight));
    REQUIRE(shape != NoIndex);
    REQUIRE(world.Shapes[shape].VertexCount == 32);
    const auto coin = world.AddBody(
        {.Pose = {.Position = {0, table + 1, 0}, .Orientation = {0, 0, 0, 1}}, .Shape = shape, .Friction = 0.5f});

    Run(solver, world, 300);
    const auto &pose = world.Poses[coin];
    CHECK(pose.Position.y == doctest::Approx(table + HalfHeight).epsilon(0.02).scale(0));
    CHECK(simd::length(RotationVector(pose.Orientation)) < 1e-2f); // flat, rather than tipped onto its rim
    CHECK(simd::length(world.Velocities[coin].Linear) < 1e-3f);
    CHECK(simd::length(world.Velocities[coin].Angular) < 1e-3f);

    // Still flat three hundred steps later: a coin resting on half its face leans further every step.
    const Pose was = pose;
    Run(solver, world, 300);
    CHECK(simd::distance(world.Poses[coin].Position, was.Position) < 1e-3f);
    CHECK(simd::length(RotationVector(world.Poses[coin].Orientation)) < 1e-2f);
}

TEST_CASE("a manifold reduced to four keeps the same four as the body turns") {
    // Reduction picks the four with the most area between them, by geometry rather than by the order
    // the points came out in, so that a settled contact keeps the same four every step and their duals
    // with them - asked here while the body is moving.
    //
    // An octagon lying flat has eight points on the plane and keeps four, and spinning it about its own
    // axis leaves all eight in the same place relative to each other, so the four are a property of the
    // body and must not move however far round it has turned. All eight are also at exactly one depth,
    // which leaves nothing but the tie-break to decide it.
    constexpr float Radius = 0.5f, HalfHeight = 0.25f;
    const mtl::Context context;
    Solver solver{context};
    World world{context};
    world.AddBody({.Shape = world.AddShape(GroundPlane)});
    const auto shape = world.AddHull(PrismPoints(8, Radius, HalfHeight));
    REQUIRE(shape != NoIndex);
    const auto prism = world.AddBody(
        {.Pose = {.Position = {0, HalfHeight + 0.2f, 0}, .Orientation = {0, 0, 0, 1}}, .Shape = shape, .Friction = 0});

    // Sleeping would end the experiment: a body at rest stops being collided at all.
    const StepSettings spinning{.SleepSteps = ~0u};
    Run(solver, world, 120, spinning);
    world.Velocities[prism].Angular = {0, 0.6f, 0}; // a third of a turn a second, which is slow

    const auto names = [&] {
        std::set<uint32_t> live;
        for (uint32_t slot = 0; slot < ContactsPerBody; ++slot) {
            const Contact &contact = world.Contacts[prism * ContactsPerBody + slot];
            if (contact.Active) live.insert(contact.Feature);
        }
        return live;
    };
    solver.Step(world, spinning);
    const auto four = names();
    REQUIRE(four.size() == 4); // eight vertices on the plane, and it kept the four with the area

    float turned = 0;
    for (uint32_t step = 0; step < 300; ++step) {
        solver.Step(world, spinning);
        CheckManifolds(world);
        turned += float(world.Velocities[prism].Angular.y) * spinning.DeltaTime;
        CAPTURE(step);
        REQUIRE(names() == four);
    }
    CHECK(turned > 2); // a third of a turn at least, so the four really did travel
    CHECK(world.Poses[prism].Position.y == doctest::Approx(HalfHeight).epsilon(0.02).scale(0));
    CHECK(simd::length(RotationVector(QuatMul(world.Poses[prism].Orientation,
                                              QuatConjugate(QuatFromRotationVector(float3{0, turned, 0}))))) < 5e-2f);
}

TEST_CASE("two coins stacked hold each other however they are twisted") {
    // Clipping one face into another produces more points than either holds - a convex polygon cut by
    // a half-plane gains a vertex - so an eight point face against an eight edge face is a sixteen-gon.
    // Bounding that clip at MaxFacePoints drops the overflow as a contiguous run of the perimeter,
    // which is half the shape and does not carry the body's centre, and the pair falls over.
    //
    // The failure is graded by how much of the true intersection survives, which is why it takes eight
    // sides to see: a hexagon on a hexagon keeps eight of twelve and holds, an octagon on an octagon
    // keeps eight of sixteen and does not. Twisted, since two faces meeting square clip to four points
    // whatever their size.
    constexpr float Radius = 0.5f, HalfHeight = 0.15f, Twist = 0.3927f; // half of an octagon's step
    uint32_t sides = 8;
    SUBCASE("four sides, where the clip has room to spare") { sides = 4; }
    SUBCASE("six sides, which fits by two") { sides = 6; }
    SUBCASE("eight sides, which is where it overflowed") { sides = 8; }

    const mtl::Context context;
    Solver solver{context};
    World world{context};
    world.AddBody({.Shape = world.AddShape(GroundPlane)});
    const auto shape = world.AddHull(PrismPoints(sides, Radius, HalfHeight));
    REQUIRE(shape != NoIndex);
    const float4 turn = QuatFromRotationVector(float3{0, Twist, 0});
    const auto lower = world.AddBody({.Pose = {.Position = {0, HalfHeight, 0}, .Orientation = {0, 0, 0, 1}}, .Shape = shape});
    const auto upper = world.AddBody({.Pose = {.Position = {0, 3 * HalfHeight + 0.01f, 0}, .Orientation = turn}, .Shape = shape});

    Run(solver, world, 240);
    // Each rests a contact margin into what is under it, and neither has turned out of the pose it was
    // set down in - a coin sliding off the one below shows first as a tilt.
    CHECK(world.Poses[lower].Position.y == doctest::Approx(HalfHeight).epsilon(0.01).scale(0));
    CHECK(world.Poses[upper].Position.y == doctest::Approx(3 * HalfHeight).epsilon(0.01).scale(0));
    CHECK(simd::length(RotationVector(world.Poses[lower].Orientation)) < 1e-3f);
    CHECK(simd::length(RotationVector(QuatMul(world.Poses[upper].Orientation, QuatConjugate(turn)))) < 1e-3f);
    CHECK(simd::length(world.Velocities[upper].Linear) < 1e-3f);

    // And on four points each way. Both pairs belong to the lower coin, it being the lower-indexed of
    // the two dynamic bodies, so its run holds four against the plane and four against the coin above.
    uint32_t against_plane = 0, against_coin = 0;
    for (uint32_t slot = 0; slot < ContactsPerBody; ++slot) {
        const Contact &contact = world.Contacts[lower * ContactsPerBody + slot];
        if (!contact.Active) continue;
        (world.Shapes[world.BodyShapes[contact.BodyB]].Kind == ShapePlane ? against_plane : against_coin) += 1;
    }
    CHECK(against_plane == ManifoldPoints);
    CHECK(against_coin == ManifoldPoints);
    CHECK(world.ContactRefusals[lower] == 0);
    CHECK(world.ContactRefusals[upper] == 0);
}

TEST_CASE("a crowned plate rests on its crown and a chamfered one on its face") {
    // What a face recovered by a height tolerance could not do, and what a cook that keeps the faces
    // does by construction. Both plates here have a face and, next to it, a facet meeting it at an
    // angle far shallower than any tolerance can distinguish from flat.
    //
    // Measured with a height-gathered face: the chamfered plate sank by the whole of the facet's rise
    // on top of the margin, and the crowned one rested on the *rim* of its bottom - the highest points
    // it has - at twice the margin, on a manifold not symmetric about a body that is. Both were resting
    // on a plane that is none of their faces, which is what a Newell fit spanning two of them comes to.
    constexpr float Half = 0.5f, Thick = 0.1f;
    const mtl::Context context;
    Solver solver{context};
    // A table rather than a plane, since a plane needs no face recovered and would not ask the question.
    const auto rest = [&](std::span<const float3> points) {
        World world{context};
        world.AddBody({.Shape = world.AddShape(GroundPlane)});
        constexpr Shape Table{.HalfExtents = {2, 0.5f, 2}, .Kind = ShapeBox};
        world.AddBody({.Pose = {.Position = {0, 0.5f, 0}, .Orientation = {0, 0, 0, 1}}, .Shape = world.AddShape(Table), .Density = 0});
        const auto shape = world.AddHull(points);
        REQUIRE(shape != NoIndex);
        const auto plate = world.AddBody({.Pose = {.Position = {0, 1 + Thick + 0.01f, 0}, .Orientation = {0, 0, 0, 1}}, .Shape = shape});
        Run(solver, world, 300);
        CHECK(simd::length(world.Velocities[plate].Linear) < 1e-3f);
        // How far the lowest of its own geometry got past the table's top, which whatever face it found
        // must be the contact margin.
        return std::pair{1 - LowestVertex(world, plate), Rest(world, plate)};
    };

    SUBCASE("a facet a third of a degree off the face it sits beside") {
        // A tenth of a metre of chamfer rising half a millimetre, which is inside any tolerance a face
        // could be recovered with, and the plate must still rest on the face rather than on the facet.
        const float rise = 0.1f * std::tan(0.3f * std::numbers::pi_v<float> / 180);
        std::vector<float3> points;
        for (const float z : {-Half, Half}) {
            points.push_back(float3{-Half, -Thick, z});
            points.push_back(float3{Half - 0.1f, -Thick, z});
            points.push_back(float3{Half, -Thick + rise, z});
            points.push_back(float3{-Half, Thick, z});
            points.push_back(float3{Half, Thick, z});
        }
        const auto [depth, settled] = rest(points);
        CHECK(depth == doctest::Approx(StepSettings{}.ContactMargin).epsilon(0.05).scale(0));
        CHECK(settled.Contacts == ManifoldPoints); // the four corners of the face, and none of the facet
    }

    SUBCASE("a bottom crowned by a millimetre and a quarter over a metre") {
        // Eight facets across the bottom, several inside a face tolerance at once - a whole cap
        // swallowed rather than one facet. Set down level and touching, the question is only which of
        // its own geometry the manifold names, and it has to be the crown. A plate on a convex bottom
        // is a rocker, so where it goes from there is not asserted.
        std::vector<float3> points;
        for (const float z : {-Half, Half}) {
            for (int k = -4; k <= 4; ++k) {
                const float x = Half * float(k) / 4;
                points.push_back(float3{x, -Thick + 0.00125f * (x / Half) * (x / Half), z});
            }
            points.push_back(float3{-Half, Thick, z});
            points.push_back(float3{Half, Thick, z});
        }
        World world{context};
        world.AddBody({.Shape = world.AddShape(GroundPlane)});
        constexpr Shape Table{.HalfExtents = {2, 0.5f, 2}, .Kind = ShapeBox};
        world.AddBody({.Pose = {.Position = {0, 0.5f, 0}, .Orientation = {0, 0, 0, 1}}, .Shape = world.AddShape(Table), .Density = 0});
        const auto shape = world.AddHull(points);
        REQUIRE(shape != NoIndex);
        const auto plate = world.AddBody({.Pose = {.Position = {0, 1 + Thick, 0}, .Orientation = {0, 0, 0, 1}}, .Shape = shape});
        Run(solver, world, 1);
        // Its crown is the middle of the bottom and the rim is a millimetre and a quarter higher. A
        // face gathered by height swallows the two together and fits a plane through both, which puts
        // the rows on the rim - the highest geometry the plate has - and pushes it down onto them.
        uint32_t held = 0;
        for (uint32_t slot = 0; slot < ContactsPerBody; ++slot) {
            const Contact &contact = world.Contacts[plate * ContactsPerBody + slot];
            if (!contact.Active) continue;
            ++held;
            CAPTURE(float(contact.AnchorA.x));
            CHECK(std::abs(float(contact.AnchorA.x)) <= 0.125f + 1e-3f); // on the crown, not out at the rim
        }
        CHECK(held > 0);
    }
}

TEST_CASE("a settled hull keeps the names of its contacts") {
    // Feature stability on the path with the most room to break it: a hull's faces are recovered rather
    // than stored, so a point is named by which vertices of which two faces made it.
    const mtl::Context context;
    Solver solver{context};
    World world{context};
    world.AddBody({.Shape = world.AddShape(GroundPlane)});
    const auto shape = world.AddHull(CubeCorners(1));
    REQUIRE(shape != NoIndex);
    std::vector<Index> stack;
    for (uint32_t i = 0; i < 3; ++i)
        stack.push_back(world.AddBody({.Pose = {.Position = {0, Half + 1.02f * float(i), 0}, .Orientation = {0, 0, 0, 1}}, .Shape = shape}));

    Run(solver, world, 300);
    const auto settled = ContactKeys(world);
    CHECK(settled.size() == 12); // three boxes, four corners each against the one below
    for (uint32_t step = 0; step < 300; ++step) {
        solver.Step(world);
        CAPTURE(step);
        REQUIRE(ContactKeys(world) == settled);
    }
    for (uint32_t i = 0; i < stack.size(); ++i) {
        CAPTURE(i);
        CHECK(std::abs(world.Poses[stack[i]].Position.y - (Half + float(i) * (1 - StepSettings{}.ContactMargin))) < 2e-3f);
    }
}

// Meshes. Same bar as the hulls: a floor made of triangles is a plane described a harder way, and it
// has to hold what a plane holds, in the same place, on the same number of points. What is new is the
// seams - a flat floor cut into triangles has edges running all over it that are not edges of anything
// - and the whole of the mesh path's difficulty is not letting a body find them.
namespace {
// A floor of `side` by `side` quads across `2 * extent` metres, each quad cut into two triangles,
// centred on the origin and wound so it faces up - tilted by `slope` about z, which turns its normal
// the same way turning a plane's would.
Index FloorMesh(World &world, uint32_t side, float extent, float slope = 0) {
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
    return world.AddMesh(points, indices);
}

// A ridge running along z: two slopes meeting at a crease over x = 0, an edge the cook marks active
// because the surface genuinely folds away there. The opposite case from the flat floors above, where
// every edge is a seam - here a body has to be able to rest on the edge and cross it.
Index RidgeMesh(World &world, float extent, float height) {
    std::vector<float3> points;
    for (const float x : {-extent, 0.f, extent})
        for (const float z : {-extent, extent}) points.push_back(float3{x, x == 0 ? height : 0, z});
    // Wound so (B - A) x (C - A) points up out of the surface on both sides of the crease.
    const std::vector<uint32_t> indices{0, 1, 3, 0, 3, 2, 2, 3, 5, 2, 5, 4};
    return world.AddMesh(points, indices);
}
} // namespace

TEST_CASE("a floor of triangles holds a box exactly where a plane does") {
    const mtl::Context context;
    Solver solver{context};
    // Off the grid lines on purpose: a corner landing exactly on a seam is a real case and a knife
    // edge, and this test is about the ordinary one.
    const auto drop = [&](uint32_t side) {
        World world{context};
        const Index floor = side == 0 ? world.AddShape(GroundPlane) : FloorMesh(world, side, 5);
        REQUIRE(floor != NoIndex);
        world.AddBody({.Shape = floor});
        const auto box = world.AddBody({.Pose = {.Position = {0.13f, 2, 0.07f}, .Orientation = {0, 0, 0, 1}}, .Shape = world.AddShape(UnitBox)});
        Run(solver, world, 180);
        CHECK(world.ContactRefusals[box] == 0);
        return Rest(world, box);
    };

    const Settled plane = drop(0), pair = drop(1), tessellated = drop(16);
    for (const Settled &settled : {pair, tessellated}) {
        CheckResting(settled.Position.y);
        CHECK(settled.Position.y == doctest::Approx(plane.Position.y).epsilon(1e-4));
        // Four, not four per triangle it happens to be standing over: a box on a flat floor rests on
        // its own four corners however the floor was cut up, and every point a seam made is a point
        // the triangle across that seam would have made too.
        CHECK(settled.Contacts == 4);
        CHECK(settled.Normal.y == doctest::Approx(1).epsilon(1e-4));
    }
}

TEST_CASE("a ridge is an edge a body rests on and crosses, not one it catches on") {
    // A crease is the opposite of a seam - the surface really does fold away and a body meeting the
    // edge is meeting geometry. The mesh path hands ConvexManifold the triangle's own normal rather
    // than searching for one, so a body over a crease gets a face contact against a face it is only
    // half over, and whether the clip keeps the part that matters is the question.
    constexpr float Extent = 4, Height = 1.5f; // a crease at about 20 degrees either side
    const mtl::Context context;
    Solver solver{context};

    SUBCASE("a box set down astride it rests on the crease") {
        World world{context};
        REQUIRE(RidgeMesh(world, Extent, Height) != NoIndex);
        world.AddBody({.Shape = 0});
        // Off the crease in z so it is not also sitting on the seam that splits each slope.
        const auto box = world.AddBody(
            {.Pose = {.Position = {0, Height + Half + 0.2f, 0.13f}, .Orientation = {0, 0, 0, 1}}, .Shape = world.AddShape(UnitBox)}
        );
        Run(solver, world, 300);
        const Settled settled = Rest(world, box);
        // Its flat underside can only touch the crease, so it sits one half extent above the peak and
        // stays over it. Nothing holds it unless the mesh path searches for a direction where the
        // triangle's own finds no contact.
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
        const auto ball = world.AddBody({.Pose = {.Position = {-3, Height * 0.25f + Radius, 0.11f}, .Orientation = {0, 0, 0, 1}},
                                         .Velocity = {.Linear = {9, 0, 0}},
                                         .Shape = world.AddShape({.Radius = Radius, .Kind = ShapeSphere})});
        float deepest = 0;
        bool crossed = false;
        for (uint32_t step = 0; step < 240; ++step) {
            solver.Step(world);
            CheckManifolds(world);
            const float3 at = world.Poses[ball].Position;
            crossed = crossed || at.x > 0.5f;
            // How far under the surface it ever got. The slopes are planes through the crease, so the
            // height of the surface under it is a straight line either side.
            const float surface = Height * std::max(0.f, 1 - std::abs(float(at.x)) / Extent);
            if (std::abs(at.x) <= Extent) deepest = std::max(deepest, surface + Radius - float(at.y));
        }
        CHECK(crossed); // it went over the crease rather than stopping dead against it
        CHECK(deepest < 4 * StepSettings{}.ContactMargin); // and was never inside the surface
        CHECK(world.ContactRefusals[ball] == 0);
    }
}

TEST_CASE("a capsule and a hull rest on a floor of triangles where they rest on a plane") {
    // A capsule and a hull each reach a mesh by a different path - the round path takes its core
    // against the triangle, and a hull goes through ConvexManifold with the triangle handing in the
    // direction. Against a plane in the same scene, because where a body rests is a property of its own
    // geometry: what must not happen is one finding a seam of the tessellation.
    constexpr float Radius = 0.25f, HalfLength = 0.5f;
    constexpr Shape Pill{.HalfExtents = {0, HalfLength, 0}, .Radius = Radius, .Kind = ShapeCapsule};
    const float quarter = 2 * std::atan(1.f); // the capsule's length runs along its own y, so this lays it down
    const mtl::Context context;
    Solver solver{context};

    // Set down off the grid lines on purpose - a body landing exactly on a seam is a knife edge with a
    // test of its own, and this one is about the ordinary case.
    const auto drop = [&](bool tessellated, bool hulled, float4 orientation, float height) {
        World world{context};
        const Index floor = tessellated ? FloorMesh(world, 8, 5) : world.AddShape(GroundPlane);
        REQUIRE(floor != NoIndex);
        world.AddBody({.Shape = floor});
        const Index shape = hulled ? world.AddHull(CubeCorners(2 * Half)) : world.AddShape(Pill);
        REQUIRE(shape != NoIndex);
        const auto body = world.AddBody({.Pose = {.Position = {0.13f, height, 0.07f}, .Orientation = orientation}, .Shape = shape});
        Run(solver, world, 240);
        CHECK(world.ContactRefusals[body] == 0);
        return std::pair{Rest(world, body), simd::length(world.Velocities[body].Linear)};
    };

    SUBCASE("a capsule lying across the triangles") {
        const float4 lying = QuatFromRotationVector(float3{0, 0, quarter});
        const auto [mesh, mesh_speed] = drop(true, false, lying, 1.5f);
        const auto [plane, plane_speed] = drop(false, false, lying, 1.5f);
        // Lying down it rests on its side, one radius up, and touches along the stretch its two ends
        // bound - which is two points on a plane and must be two on the triangles as well.
        CHECK(mesh.Position.y == doctest::Approx(Radius - StepSettings{}.ContactMargin).epsilon(5e-3));
        CHECK(mesh.Position.y == doctest::Approx(plane.Position.y).epsilon(2e-3));
        CHECK(mesh.Normal.y == doctest::Approx(1).epsilon(1e-3));
        CHECK(mesh.Contacts == plane.Contacts);
        CHECK(mesh_speed < 1e-2f);
    }

    SUBCASE("a capsule stood on one end") {
        const auto [mesh, mesh_speed] = drop(true, false, float4{0, 0, 0, 1}, 1.5f);
        const auto [plane, plane_speed] = drop(false, false, float4{0, 0, 0, 1}, 1.5f);
        // On end it is a sphere as far as the floor is concerned, and reaches its half length plus its
        // radius down from the centre.
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
        // Its four corners and nothing else. Clipped the other way round it would be the corners of the
        // tessellation holding it up.
        CHECK(mesh.Contacts == 4);
        CHECK(mesh.Contacts == plane.Contacts);
        CHECK(mesh_speed < 1e-2f);
    }
}

TEST_CASE("a coin rests on a floor of triangles where it rests on a plane") {
    // A many-faced hull on a mesh, which is coverage rather than a guard on the clip bound: a coin's
    // eight point face cut by a triangle's three side planes is eleven, but measured, the coin lies
    // across five triangles and each clips only the part above it, so no single manifold gets near the
    // bound. What it asserts is what a coin on triangles has to be - every row on a vertex of its own
    // rim, at the same height a plane holds it at.
    constexpr float Radius = 0.5f, HalfHeight = 0.15f;
    const mtl::Context context;
    Solver solver{context};
    const auto drop = [&](bool tessellated) {
        World world{context};
        const Index floor = tessellated ? FloorMesh(world, 8, 5) : world.AddShape(GroundPlane);
        REQUIRE(floor != NoIndex);
        world.AddBody({.Shape = floor});
        const auto shape = world.AddHull(PrismPoints(8, Radius, HalfHeight));
        REQUIRE(shape != NoIndex);
        // Off the grid lines, as the tests around this one are: a corner on a seam is its own case.
        const auto body = world.AddBody({.Pose = {.Position = {0.13f, 1.f, 0.07f}, .Orientation = {0, 0, 0, 1}}, .Shape = shape});
        Run(solver, world, 240);
        CHECK(world.ContactRefusals[body] == 0);
        CHECK(simd::length(world.Velocities[body].Linear) < 1e-3f);
        CHECK(simd::length(RotationVector(world.Poses[body].Orientation)) < 1e-3f); // flat, not tipped onto its rim
        // Every row is on a vertex of the rim, which is what a clip short of the face takes away.
        for (uint32_t slot = 0; slot < ContactsPerBody; ++slot) {
            const Contact &contact = world.Contacts[body * ContactsPerBody + slot];
            if (!contact.Active) continue;
            CHECK(std::hypot(float(contact.AnchorA.x), float(contact.AnchorA.z)) == doctest::Approx(Radius).epsilon(1e-3).scale(0));
        }
        return Rest(world, body);
    };

    const Settled mesh = drop(true), plane = drop(false);
    CHECK(mesh.Position.y == doctest::Approx(HalfHeight).epsilon(0.01).scale(0));
    CHECK(mesh.Position.y == doctest::Approx(plane.Position.y).epsilon(2e-3));
    CHECK(mesh.Normal.y == doctest::Approx(1).epsilon(1e-3));
    // Not the same count, and that is right rather than a shortfall: a mesh gives one manifold per
    // triangle and this coin lies across five, so between them they hold all eight rim vertices, while
    // a plane is one manifold and reduction takes it to four. Each vertex is held once either way.
    CHECK(mesh.Contacts == 8);
    CHECK(plane.Contacts == ManifoldPoints);
    // It did not drift off the spot it was set down on, which is what a manifold covering half its face
    // shows as once the coin starts leaning into the half it has.
    CHECK(std::abs(float(mesh.Position.x) - 0.13f) < 2e-3f);
    CHECK(std::abs(float(mesh.Position.z) - 0.07f) < 2e-3f);
}

TEST_CASE("a body over more triangles than one batch holds still rests on them") {
    // MaxMeshTriangles caps what a body holds from a mesh at once, and a wide body on a finely cut
    // floor passes it easily: four metres across third-of-a-metre quads is some three hundred triangles
    // against a batch of thirty-two. Stopping the walk at the cap does not leave the body resting on
    // what it gathered - a one metre box fell through 0.31 m quads and a four metre slab through
    // anything finer than 1.7 m. The walk resumes, so the cap bounds registers rather than coverage and
    // there is nothing to refuse. Against a plane in the same scene, since where a slab rests is the
    // slab's own property.
    constexpr float Wide = 2;
    constexpr Shape Slab{.HalfExtents = {Wide, Half, Wide}, .Kind = ShapeBox};
    const mtl::Context context;
    Solver solver{context};
    const auto drop = [&](uint32_t side) {
        World world{context};
        const Index floor = side == 0 ? world.AddShape(GroundPlane) : FloorMesh(world, side, 5);
        REQUIRE(floor != NoIndex);
        world.AddBody({.Shape = floor});
        const auto slab = world.AddBody({.Pose = {.Position = {0.13f, 1.f, 0.07f}, .Orientation = {0, 0, 0, 1}}, .Shape = world.AddShape(Slab)});
        uint32_t refused = 0;
        for (uint32_t step = 0; step < 240; ++step) { // watched every step, since a refusal is per step
            solver.Step(world);
            CheckManifolds(world);
            refused = std::max(refused, world.ContactRefusals[slab]);
        }
        return std::pair{Rest(world, slab), refused};
    };

    const auto [plane, plane_refused] = drop(0);
    const auto [coarse, coarse_refused] = drop(4); // eight quads over ten metres, well inside one batch
    const auto [fine, fine_refused] = drop(32); // and a third of a metre a quad, which is ten batches

    CHECK(plane_refused == 0);
    CHECK(coarse_refused == 0);
    CHECK(fine_refused == 0); // the batch boundary is not a shortfall and must not be counted as one

    for (const Settled &settled : {plane, coarse, fine}) {
        CheckResting(settled.Position.y);
        CHECK(settled.Normal.y == doctest::Approx(1).epsilon(1e-3));
        CHECK(settled.Contacts >= 3); // enough to hold it flat rather than balance it
    }
    CHECK(fine.Position.y == doctest::Approx(plane.Position.y).epsilon(2e-3));
    // And it did not walk off the part of the floor it happened to gather first.
    CHECK(std::abs(float(fine.Position.x) - 0.13f) < 5e-3f);
    CHECK(std::abs(float(fine.Position.z) - 0.07f) < 5e-3f);
}

TEST_CASE("a box whose corners land on the seams is held by four contacts, not six") {
    // The knife edge the test above stays off: two of the box's corners exactly on the diagonals of the
    // quads they are over, which fires two mechanisms a flat landing never shows. A point on a plane is
    // inside it as far as a clip is concerned, so both triangles meeting along the seam hold that
    // corner - one piece of geometry with two rows and two duals. And the reference face goes to
    // whichever of the box and the triangle is flatter against the normal, which for a box lying flat
    // is a tie between two numbers both one to within a normalize's rounding: it came out differently
    // for different triangles under the same box, and where it came out on the box the manifold was the
    // triangle clipped into it - points at the corners of the tessellation, named after the body rather
    // than the geometry under them. Eight contacts under a box with four corners, settled off flat.
    const mtl::Context context;
    Solver solver{context};
    const auto drop = [&](uint32_t side) {
        World world{context};
        const Index floor = side == 0 ? world.AddShape(GroundPlane) : FloorMesh(world, side, 5);
        REQUIRE(floor != NoIndex);
        world.AddBody({.Shape = floor});
        // The quads are square, so their diagonals are the lines x = z: a box centred on one has two
        // of its corners on one, whatever the mesh was cut into.
        const auto box = world.AddBody({.Pose = {.Position = {0.3f, 2, 0.3f}, .Orientation = {0, 0, 0, 1}}, .Shape = world.AddShape(UnitBox)});
        Run(solver, world, 180);
        CHECK(world.ContactRefusals[box] == 0);
        return Rest(world, box);
    };

    const Settled plane = drop(0);
    for (const uint32_t side : {1u, 2u, 16u}) {
        CAPTURE(side);
        const Settled settled = drop(side);
        CHECK(settled.Contacts == 4); // its own four corners, each held by one triangle and not two
        CheckResting(settled.Position.y);
        CHECK(settled.Position.y == doctest::Approx(plane.Position.y).epsilon(1e-4));
        CHECK(settled.Normal.y == doctest::Approx(1).epsilon(1e-4));
        // Flat, not tipped: a duplicated corner is stiffer than the three that are not, and what it
        // does is roll the box off it.
        CHECK(std::abs(settled.Position.x - plane.Position.x) < 1e-3f);
        CHECK(std::abs(settled.Position.z - plane.Position.z) < 1e-3f);
    }
}

TEST_CASE("a slope of triangles holds and lets go exactly as a sloped plane does") {
    // A flat mesh hides most of what a mesh has to get right: a triangle whose normal is +y is over
    // whatever is above it and its face is the one being stood on however the direction was found, and
    // the other slope tests tilt gravity rather than the ground. Tilt the ground and none of that is
    // free - and a box sliding down it crosses a seam every few tenths of a metre.
    //
    // The plane of the same slope is the oracle, the way a box is for a cube given as a hull, so what
    // has to hold is not a bound but an agreement.
    constexpr float Slope = 0.2f; // tan 0.203, so a friction of 0.5 holds it and one of 0.1 does not
    const mtl::Context context;
    Solver solver{context};
    const float3 up{-std::sin(Slope), std::cos(Slope), 0}, down{-std::cos(Slope), -std::sin(Slope), 0};

    const auto drop = [&](uint32_t side, float friction) {
        World world{context};
        const Index floor = side == 0 ? world.AddShape({.Normal = up, .Offset = 0, .Kind = ShapePlane})
                                      : FloorMesh(world, side, 5, Slope);
        REQUIRE(floor != NoIndex);
        world.AddBody({.Shape = floor, .Friction = friction});
        // Resting square on the slope, a little above it and off the grid lines both ways.
        const float3 start = up * (Half + 0.05f) - down * 0.13f + float3{0, 0, 0.07f};
        const auto box = world.AddBody({.Pose = {.Position = start, .Orientation = QuatFromRotationVector(float3{0, 0, Slope})},
                                        .Shape = world.AddShape(UnitBox),
                                        .Friction = friction});
        Run(solver, world, 180);
        CHECK(world.ContactRefusals[box] == 0);
        return Rest(world, box);
    };

    SUBCASE("inside the cone it stays where the plane keeps it") {
        const Settled plane = drop(0, 0.5f), tessellated = drop(16, 0.5f);
        for (const Settled &settled : {plane, tessellated}) {
            CheckResting(simd::dot(settled.Position, up)); // resting on the surface, not in it
            CHECK(settled.Contacts == 4); // its own four corners, whatever the floor was cut into
            CHECK(simd::dot(settled.Normal, up) == doctest::Approx(1).epsilon(1e-4));
        }
        // And in the same place: a box held by friction on a slope has slipped by however much the
        // solver let it, and the two descriptions of the slope have to have let it slip the same.
        CHECK(simd::length(tessellated.Position - plane.Position) < 1e-3f);
    }

    SUBCASE("outside it they slide the same distance") {
        // The seams are what this is for. A box sliding down a tessellated slope hands its manifold
        // from one pair of triangles to the next every few tenths of a metre, and any point a seam
        // adds is friction the box on the plane never feels.
        const Settled plane = drop(0, 0.1f), tessellated = drop(16, 0.1f);
        const float went = simd::dot(plane.Position, down), also = simd::dot(tessellated.Position, down);
        // mu is well under tan(slope), so it is still going: g (sin - mu cos) over three seconds.
        const float gravity = std::abs(StepSettings{}.Gravity.y);
        const float expected = 0.5f * gravity * (std::sin(Slope) - 0.1f * std::cos(Slope)) * 3 * 3;
        CHECK(went == doctest::Approx(expected).epsilon(0.05));
        CHECK(also == doctest::Approx(went).epsilon(2e-3));
        CheckResting(simd::dot(tessellated.Position, up)); // and it is still on the surface, not through it
        CHECK(tessellated.Contacts == 4);
    }
}

TEST_CASE("a sphere rolls across a tessellated floor without finding the seams") {
    // The case active edges exist for. Every seam of a flat floor is an edge in the triangle soup and
    // none of them is an edge in the surface, so a ball rolling over one must not feel it - not as a
    // bump, not as a normal pointing sideways, and not as a step in its speed.
    const mtl::Context context;
    Solver solver{context};
    World plane_world{context}, mesh_world{context};
    for (World *world : {&plane_world, &mesh_world}) {
        const bool tessellated = world == &mesh_world;
        const Index floor = tessellated ? FloorMesh(*world, 16, 5) : world->AddShape(GroundPlane);
        REQUIRE(floor != NoIndex);
        world->AddBody({.Shape = floor});
        world->AddBody({.Pose = {.Position = {-2, 0.25f, 0.07f}, .Orientation = {0, 0, 0, 1}},
                        .Velocity = {.Linear = {2, 0, 0}},
                        .Shape = world->AddShape({.Radius = 0.25f, .Kind = ShapeSphere})});
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
    // And it has got as far as the same ball rolling on a plane, rather than being tripped or slowed.
    CHECK(mesh_world.Poses[1].Position.x == doctest::Approx(plane_world.Poses[1].Position.x).epsilon(1e-3));
    CHECK(std::abs(mesh_world.Poses[1].Position.z - plane_world.Poses[1].Position.z) < 1e-3f);
    CHECK(mesh_world.ContactRefusals[1] == 0);
}

TEST_CASE("nothing is pushed out of the back of a mesh") {
    // A mesh is a surface and not a solid: it has a side, and a body that is behind it is past it
    // rather than inside it. A plane is the opposite - it is a half space and holds everything above
    // it - so this is the one place the two floors are meant to disagree.
    const mtl::Context context;
    Solver solver{context};
    World world{context};
    REQUIRE(FloorMesh(world, 4, 5) != NoIndex);
    world.AddBody({.Shape = 0});
    const auto box = world.AddBody({.Pose = {.Position = {0.13f, -1, 0.07f}, .Orientation = {0, 0, 0, 1}}, .Shape = world.AddShape(UnitBox)});

    Run(solver, world, 60);
    CHECK(world.Poses[box].Position.y < -1.f); // still falling, and never caught from beneath
    CHECK(Rest(world, box).Contacts == 0);
}

TEST_CASE("a mesh scene steps to bit-identical state twice") {
    const mtl::Context context;
    Solver solver{context};
    const auto run = [&context, &solver] {
        World world{context};
        FloorMesh(world, 8, 4);
        world.AddBody({.Shape = 0});
        const auto shape = world.AddShape(UnitBox);
        for (uint32_t i = 0; i < 4; ++i)
            world.AddBody({.Pose = {.Position = {0.13f + 0.02f * float(i), Half + 1.02f * float(i), 0.07f}, .Orientation = {0, 0, 0, 1}},
                           .Shape = shape});
        for (uint32_t step = 0; step < 120; ++step) solver.Step(world);
        return std::vector<Pose>{world.Poses.All().begin(), world.Poses.All().begin() + world.BodyCount()};
    };
    const auto first = run(), second = run();
    for (uint32_t body = 0; body < first.size(); ++body) {
        CAPTURE(body);
        CHECK(std::memcmp(&first[body], &second[body], sizeof(Pose)) == 0);
    }
}

TEST_CASE("a manifold wider than four points keeps the four that hold it") {
    // Eight corners of a face all touching at once, which is more than a solver can use and more than
    // a body can afford: every slot one pair takes is a slot the rest of its contacts do not get. What
    // has to survive is not any four but four that still resist the body rocking - the reduction picks
    // for area, so they come out spread around the face rather than bunched along one edge of it.
    const mtl::Context context;
    Solver solver{context};
    World world{context};
    world.AddBody({.Shape = world.AddShape(GroundPlane)});
    std::vector<float3> points;
    for (uint32_t corner = 0; corner < 8; ++corner) {
        const float angle = 2 * 3.14159265f * float(corner) / 8;
        for (const float y : {-Half, Half}) points.push_back(float3{std::cos(angle), y, std::sin(angle)});
    }
    const auto shape = world.AddHull(points);
    REQUIRE(shape != NoIndex);
    REQUIRE(world.Shapes[shape].VertexCount == 16);
    const auto prism = world.AddBody({.Pose = {.Position = {0, 1.5f, 0}, .Orientation = {0, 0, 0, 1}}, .Shape = shape});

    Run(solver, world, 240);
    const Settled settled = Rest(world, prism);
    CheckResting(settled.Position.y);
    CHECK(settled.Contacts == 4); // its base has eight corners on the plane and holds them with four
    CHECK(world.ContactRefusals[prism] == 0);

    // Spread rather than bunched: four points along one edge of the base would let it rock about that
    // edge however deep they were, which is why the choice maximizes area and not distance.
    float3 low{1e9f, 0, 1e9f}, high{-1e9f, 0, -1e9f};
    const auto slots = world.Contacts.All().subspan(prism * ContactsPerBody, ContactsPerBody);
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
