#include "Shapes.h"
#include "Solver.h"

#include <doctest/doctest.h>

using namespace rbp;

namespace {
struct Reporting {
    const mtl::Context context;
    Solver solver{context};
    World world{context};

    Index Floor() { return world.AddBody({.Shape = world.AddShape(GroundPlane), .Friction = 0.3f}); }
    Index Box(float3 at = {0, Half, 0}) {
        return world.AddBody({.Pose = At(at), .Shape = world.AddShape(UnitBox), .Density = 2, .Friction = 0.3f});
    }
};
} // namespace

TEST_CASE_FIXTURE(Reporting, "reporting: sleeping support includes patch geometry when reporting starts late") {
    Floor();
    const auto body = Box();
    for (int i = 0; i < 180; ++i) solver.Step(world);
    REQUIRE(world.Quiet[body] >= StepSettings{}.SleepSteps);
    std::vector<Contact> cached;
    for (uint32_t i = 0; i < 4; ++i) cached.push_back(world.Contacts[body * ContactsPerBody + i]);
    world.TrackContacts = true;
    solver.Step(world);
    for (uint32_t i = 0; i < 4; ++i) {
        const Contact &now = world.Contacts[body * ContactsPerBody + i];
        CHECK(now.Feature == cached[i].Feature);
        CHECK(length(now.AnchorA - cached[i].AnchorA) == 0);
        CHECK(length(now.AnchorB - cached[i].AnchorB) == 0);
        CHECK(length(now.C0 - cached[i].C0) == 0);
        CHECK(length(now.Lambda - cached[i].Lambda) == 0);
        CHECK(length(now.Penalty - cached[i].Penalty) == 0);
    }
    const auto changes = world.TakeContactChanges();
    REQUIRE(changes.size() == 4);
    float3 force{}, impulse{};
    for (const auto &c : changes) {
        CHECK(c.Kind == ContactPersisted);
        CHECK(c.Step == 181);
        CHECK(c.NominalArea == doctest::Approx(1).epsilon(0.001));
        CHECK(c.NominalExtent == doctest::Approx(std::sqrt(2.f)).epsilon(0.001));
        CHECK(c.BounceImpulse == 0);
        CHECK(c.Approach == 0);
        CHECK(c.Manifold() == changes.front().Manifold());
        force += c.ForceOnA();
        impulse += c.ImpulseOnA();
    }
    CHECK(force.y == doctest::Approx(2 * 9.81f).epsilon(0.01));
    CHECK(impulse.y == doctest::Approx(2 * 9.81f / 60).epsilon(0.01));
}

TEST_CASE_FIXTURE(Reporting, "reporting: impact impulse matches momentum with and without restitution") {
    for (float dt : {1.f / 60, 1.f / 600}) {
        for (float restitution : {0.f, 0.7f}) {
            World w{context};
            w.TrackContacts = true;
            w.AddBody({.Shape = w.AddShape(GroundPlane)});
            const auto ball = w.AddBody({.Pose = At(float3{0, Half, 0}), .Velocity = {.Linear = {0, -2, 0}}, .Shape = w.AddShape({.Radius = Half, .Kind = ShapeSphere}), .Mass = AuthoredMass{2, {0.2f, 0.2f, 0.2f}}, .Restitution = restitution});
            solver.Step(w, {.Gravity = {0, 0, 0}, .DeltaTime = dt, .Iterations = 60});
            const auto events = w.TakeContactChanges();
            REQUIRE(events.size() == 1);
            const auto &c = events[0];
            CHECK(c.Kind == ContactAdded);
            CHECK(c.Step == 1);
            CHECK(c.DeltaTime == dt);
            CHECK(c.Approach == doctest::Approx(2));
            CHECK(c.NominalArea == 0);
            CHECK(c.NominalExtent == 0);
            CHECK(c.Restitution == restitution);
            CHECK(c.SideA.InvMass == 0.5f);
            CHECK(c.SideB.InvMass == 0);
            CHECK(c.ImpulseOnA().y == doctest::Approx(2 * (w.Velocities[ball].Linear.y + 2)).epsilon(0.01));
            CHECK(c.ImpulseOnA().y == doctest::Approx(4 * (1 + restitution)).epsilon(0.01));
            CHECK(c.BounceImpulse == doctest::Approx(4 * restitution).epsilon(0.01));
        }
    }
}

TEST_CASE_FIXTURE(Reporting, "reporting: sliding impulse opposes slip and accounts for momentum") {
    Floor();
    const auto body = Box();
    for (int i = 0; i < 120; ++i) solver.Step(world);
    world.Velocities[body].Linear = {1, 0, 0};
    world.Wake(body);
    world.TrackContacts = true;
    const StepSettings settings{.Iterations = 60};
    solver.Step(world, settings);
    const auto events = world.TakeContactChanges();
    REQUIRE(events.size() == 4);
    float3 impulse{};
    for (const auto &c : events) {
        const auto force = c.ForceOnA();
        CHECK(force.x < 0);
        CHECK(std::abs(force.x) <= c.Friction * force.y + 1e-4f);
        CHECK(c.NominalExtent == doctest::Approx(1).epsilon(0.001));
        CHECK(c.NominalArea == doctest::Approx(1).epsilon(0.001));
        CHECK(c.SideA.Velocity.Linear.x == float(world.Velocities[body].Linear.x));
        impulse += c.ImpulseOnA();
    }
    CHECK(impulse.x == doctest::Approx(2 * (world.Velocities[body].Linear.x - 1)).epsilon(0.02));
    CHECK(impulse.y == doctest::Approx(2 * (world.Velocities[body].Linear.y + 9.81f * settings.DeltaTime)).epsilon(0.02));
}

TEST_CASE_FIXTURE(Reporting, "reporting: octagonal patch retains area before reduction to four points") {
    Floor();
    const auto hull = world.AddHull(PrismPoints(8, 1, Half));
    REQUIRE(hull != NoIndex);
    world.AddBody({.Pose = At(float3{0, Half, 0}), .Shape = hull});
    world.TrackContacts = true;
    solver.Step(world);
    const auto events = world.TakeContactChanges();
    REQUIRE(events.size() == 4);
    for (const auto &c : events) {
        CHECK(c.NominalArea == doctest::Approx(2 * std::sqrt(2.f)).epsilon(0.001));
        CHECK(c.NominalExtent == doctest::Approx(2).epsilon(0.001));
    }
}

TEST_CASE_FIXTURE(Reporting, "reporting: queued samples survive separation and host-side removal") {
    Floor();
    const auto body = Box();
    const auto id = world.IdOf(body);
    world.Shapes[world.BodyShapes[body]].UserData = 404;
    world.TrackContacts = true;
    solver.Step(world, {.DeltaTime = 1.f / 120});
    const auto pose = world.Poses[body];
    uint64_t removal_step = 1;
    float removal_dt = 0;
    SUBCASE("separation during a later step") {
        world.Shapes[world.BodyShapes[body]].UserData = 999;
        world.Poses[body].Position = {0, 3, 0};
        world.Wake(body);
        removal_step = 2;
        removal_dt = 1.f / 240;
        solver.Step(world, {.DeltaTime = removal_dt});
        REQUIRE(world.RemoveBody(body));
        solver.Step(world);
        CHECK(world.IdOf(Box({0, 4, 0})) != id);
    }
    SUBCASE("host-side removal") { REQUIRE(world.RemoveBody(body)); }
    const auto events = world.TakeContactChanges();
    REQUIRE(events.size() == 8);
    for (uint32_t i = 0; i < 4; ++i) {
        const auto &c = events[i];
        CHECK(c.A == id);
        CHECK(c.SideA.UserData == 404);
        CHECK(c.Kind == ContactAdded);
        CHECK(c.Step == 1);
        CHECK(c.DeltaTime == 1.f / 120);
        CHECK(c.SideA.Pose.Position.y == pose.Position.y);
        CHECK(c.SideA.Point.y == doctest::Approx(-Half));
        CHECK(c.SideB.Point.y == doctest::Approx(0));
        CHECK(c.Normal.y == doctest::Approx(1));
        CHECK(c.ImpulseOnA().y > 0);
        CHECK(events[i + 4].Kind == ContactRemoved);
        CHECK(events[i + 4].Step == removal_step);
        CHECK(events[i + 4].DeltaTime == removal_dt);
        CHECK(events[i + 4].Manifold() == c.Manifold());
        CHECK(length(events[i + 4].ImpulseOnA()) == 0);
    }
}

TEST_CASE_FIXTURE(Reporting, "reporting: compound leaf manifolds retain collider attribution") {
    const auto floor = Floor();
    Shape left = UnitBox, right = UnitBox;
    left.UserData = 101;
    right.UserData = 102;
    left.Local = At(float3{-1, 0, 0});
    right.Local = At(float3{1, 0, 0});
    const auto compound = world.AddCompound(std::vector<Index>{world.AddShape(left), world.AddShape(right)});
    REQUIRE(compound != NoIndex);
    const auto body = world.AddBody({.Pose = At(float3{0, Half, 0}), .Shape = compound});
    world.TrackContacts = true;
    solver.Step(world);
    const auto events = world.TakeContactChanges();
    REQUIRE(events.size() == 8);
    uint32_t points[2]{};
    for (const auto &c : events) {
        const auto key = c.Manifold();
        CHECK(key.A == world.IdOf(floor));
        CHECK(key.B == world.IdOf(body));
        CHECK(key.ChildA == 0);
        REQUIRE(key.ChildB < 2);
        ++points[key.ChildB];
        CHECK(key.SubShapeA == NoIndex);
        CHECK(key.SubShapeB == NoIndex);
        CHECK((c.SideA.Point.x < 0) == (key.ChildB == 0));
        CHECK(c.SideA.UserData == 101 + key.ChildB);
    }
    CHECK(points[0] == 4);
    CHECK(points[1] == 4);
}

TEST_CASE_FIXTURE(Reporting, "reporting: a dynamic pair reports equal and opposite momentum transfer") {
    const auto sphere = world.AddShape({.Radius = Half, .Kind = ShapeSphere});
    const auto a = world.AddBody({.Pose = At(float3{-Half, 0, 0}), .Velocity = {.Linear = {2, 0, 0}}, .Shape = sphere, .Mass = AuthoredMass{2, {0.2f, 0.2f, 0.2f}}, .Restitution = 0.5f});
    const auto b = world.AddBody({.Pose = At(float3{Half, 0, 0}), .Shape = sphere, .Mass = AuthoredMass{3, {0.3f, 0.3f, 0.3f}}});
    world.TrackContacts = true;
    solver.Step(world, {.Gravity = {0, 0, 0}, .DeltaTime = 1.f / 600, .Iterations = 60});
    const auto events = world.TakeContactChanges();
    REQUIRE(events.size() == 1);
    const auto impulse = events[0].ImpulseOnA();
    CHECK(impulse.x == doctest::Approx(-3.6f).epsilon(0.01));
    CHECK(impulse.x == doctest::Approx(2 * (world.Velocities[a].Linear.x - 2)).epsilon(0.01));
    CHECK(-impulse.x == doctest::Approx(3 * world.Velocities[b].Linear.x).epsilon(0.01));
}

TEST_CASE_FIXTURE(Reporting, "reporting: an edge has zero area and a nonzero extent") {
    Floor();
    const float angle = 0.4f;
    world.AddBody({.Pose = At(float3{0, Half * (std::cos(angle) + std::sin(angle)), 0}, QuatFromRotationVector(float3{0, 0, angle})), .Shape = world.AddShape(UnitBox)});
    world.TrackContacts = true;
    solver.Step(world);
    const auto events = world.TakeContactChanges();
    REQUIRE(events.size() == 2);
    for (const auto &c : events) {
        CHECK(c.NominalArea == doctest::Approx(0).epsilon(1e-6));
        CHECK(c.NominalExtent == doctest::Approx(1).epsilon(0.001));
    }
}

TEST_CASE_FIXTURE(Reporting, "reporting: measurement does not change simulated motion") {
    World measured{context}, unmeasured{context};
    measured.TrackContacts = true;
    for (World *w : {&measured, &unmeasured}) {
        w->AddBody({.Shape = w->AddShape(GroundPlane)});
        w->AddBody({.Pose = At(float3{0, 0.9f, 0}, QuatFromRotationVector(float3{0.3f, 0.1f, 0.2f})), .Velocity = {.Linear = {0.5f, -1, 0}}, .Shape = w->AddShape(UnitBox), .Restitution = 0.3f});
    }
    for (int i = 0; i < 90; ++i) {
        solver.Step(measured);
        solver.Step(unmeasured);
        const Pose a = measured.Poses[1], b = unmeasured.Poses[1];
        CHECK(length(a.Position - b.Position) == 0);
        CHECK(length(a.Orientation - b.Orientation) == 0);
        CHECK(length(measured.Velocities[1].Linear - unmeasured.Velocities[1].Linear) == 0);
    }
    CHECK_FALSE(measured.TakeContactChanges().empty());
    CHECK(unmeasured.TakeContactChanges().empty());
}

TEST_CASE_FIXTURE(Reporting, "reporting: mesh triangle keys remain on the correct side after canonicalization") {
    const std::vector<float3> vertices{float3{-2, 0, -2}, float3{2, 0, -2}, float3{2, 0, 2}, float3{-2, 0, 2}};
    const auto mesh = world.AddMesh(vertices, std::vector<uint32_t>{0, 2, 1, 0, 3, 2});
    REQUIRE(mesh != NoIndex);
    world.Shapes[mesh].UserData = 303;
    const auto compound = world.AddCompound(std::vector<Index>{mesh});
    REQUIRE(compound != NoIndex);
    const auto floor = world.AddBody({.Shape = compound, .Density = 0});
    const auto box = Box({0, Half - 1e-4f, 0});
    const auto first = world.Shapes[world.Child(compound, 0)].FirstTriangle;
    world.TrackContacts = true;
    solver.Step(world);
    const auto events = world.TakeContactChanges();
    REQUIRE_FALSE(events.empty());
    bool triangles[2]{};
    for (const auto &c : events) {
        const auto key = c.Manifold();
        CHECK(key.A == world.IdOf(floor));
        CHECK(key.B == world.IdOf(box));
        CHECK(key.SubShapeB == NoIndex);
        CHECK(c.SideB.UserData == 303);
        REQUIRE(key.SubShapeA >= first);
        REQUIRE(key.SubShapeA < first + 2);
        triangles[key.SubShapeA - first] = true;
        CHECK(c.NominalArea == doctest::Approx(0.5f).epsilon(0.002));
    }
    CHECK(triangles[0]);
    CHECK(triangles[1]);
}

TEST_CASE_FIXTURE(Reporting, "reporting: force application points account for angular momentum") {
    Floor();
    const float angle = 0.4f;
    const Pose initial = At(float3{0, Half * (std::cos(angle) + std::sin(angle)), 0}, QuatFromRotationVector(float3{0, 0, angle}));
    const auto box = world.AddBody({.Pose = initial, .Velocity = {.Linear = {0, -2, 0}}, .Shape = world.AddShape(UnitBox), .Density = 2, .Friction = 0});
    world.TrackContacts = true;
    solver.Step(world, {.Gravity = {0, 0, 0}, .DeltaTime = 1.f / 600, .Iterations = 60});
    const auto events = world.TakeContactChanges();
    REQUIRE_FALSE(events.empty());
    float3 angular_impulse{};
    for (const auto &c : events) {
        CHECK(length(c.SideA.InitialPose.Position - initial.Position) == 0);
        CHECK(length(c.SideA.InitialPose.Orientation - initial.Orientation) == 0);
        angular_impulse += cross(Rotate(c.SideA.InitialPose.Orientation, c.SideA.Anchor), c.ImpulseOnA());
    }
    // The cube's inertia is isotropic, I = m side^2 / 6 = 1/3 kg m^2.
    const float3 momentum = world.Velocities[box].Angular / 3;
    CHECK(length(angular_impulse - momentum) < 0.01f * length(momentum));
    CHECK(length(momentum) > 0.1f);
}

TEST_CASE_FIXTURE(Reporting, "planes: body and collider frames carry the surface") {
    const auto turn = QuatFromRotationVector(float3{0, 0, -std::numbers::pi_v<float> / 2});
    const Pose surface = ComposePose(At(float3{2, 3, 4}, turn), At(float3{0, 1, 0}));
    const float3 normal = Rotate(surface.Orientation, float3{0, 1, 0});
    for (bool compound : {false, true}) {
        for (bool round : {false, true}) {
            World w{context};
            Shape plane = GroundPlane;
            plane.Local = At(float3{0, 1, 0});
            Index shape = w.AddShape(plane);
            if (compound) shape = w.AddCompound(std::vector<Index>{shape});
            w.AddBody({.Pose = At(float3{2, 3, 4}, turn), .Shape = shape, .Density = 0});
            const auto body = w.AddBody({.Pose = At(surface.Position + 0.499f * normal), .Shape = w.AddShape(round ? Shape{.Radius = Half, .Kind = ShapeSphere} : UnitBox)});
            w.TrackContacts = true;
            solver.Step(w, {.Gravity = {0, 0, 0}});
            const auto contacts = w.TakeContactChanges();
            REQUIRE(!contacts.empty());
            for (const auto &c : contacts) {
                CHECK(c.A.Slot == body);
                CHECK(simd::distance(c.Normal, normal) < 1e-4f);
                const float3 on_plane = c.SideB.InitialPose.Position + Rotate(c.SideB.InitialPose.Orientation, c.SideB.Point);
                CHECK(std::abs(dot(on_plane - surface.Position, normal)) < 1e-4f);
            }
        }
    }
}

TEST_CASE_FIXTURE(Reporting, "planes: a double-sided surface supports either side") {
    for (float side : {-1.f, 1.f}) {
        World w{context};
        Shape plane = GroundPlane;
        plane.DoubleSided = true;
        w.AddBody({.Pose = At(float3{0, 2, 0}), .Shape = w.AddShape(plane)});
        const auto body = w.AddBody({.Pose = At(float3{0, 2 + side * Half, 0}), .Shape = w.AddShape({.Radius = Half, .Kind = ShapeSphere}), .Density = 1});
        for (int step = 0; step < 60; ++step) solver.Step(w, {.Gravity = {0, -side * 9.81f, 0}});
        CHECK(w.Poses[body].Position.y == doctest::Approx(2 + side * Half).epsilon(0.001));
        CHECK(length(w.Velocities[body].Linear) < 0.01f);
    }
}

TEST_CASE_FIXTURE(Reporting, "compounds: the root collider frame reaches both sides of collision") {
    for (bool moving : {false, true}) {
        World w{context};
        w.TrackContacts = true;
        const auto box = w.AddShape({.HalfExtents = {0.5f, 1, 1.5f}, .Kind = ShapeBox});
        Pose cooked;
        const auto compound = w.AddCompound(std::vector<Index>{box}, &cooked);
        const auto rotation = QuatFromRotationVector(float3{0, 0, std::numbers::pi_v<float> / 2});
        w.Shapes[compound].Local = ComposePose(At(float3{2, 3, 0}, rotation), cooked);
        const auto body = w.AddBody({.Shape = compound, .Mass = moving ? AuthoredMass{1, {1, 1, 1}} : AuthoredMass{0, {0, 0, 0}}});
        Index other;
        if (moving) {
            other = w.AddBody({.Pose = At(float3{0, 2.5f, 0}), .Shape = w.AddShape(GroundPlane)});
        } else {
            other = w.AddBody({.Pose = At(float3{2, 3.999f, 0}), .Shape = w.AddShape({.Radius = Half, .Kind = ShapeSphere})});
        }
        solver.Step(w, {.Gravity = {0, 0, 0}});
        const auto events = w.TakeContactChanges();
        REQUIRE(!events.empty());
        for (const auto &event : events) {
            CHECK(event.A.Slot == (moving ? body : other));
            CHECK(simd::distance(event.Normal, float3{0, 1, 0}) < 1e-4f);
            const auto &on = moving ? event.SideA : event.SideB;
            const float3 point = on.InitialPose.Position + Rotate(on.InitialPose.Orientation, on.Point);
            CHECK(point.y == doctest::Approx(moving ? 2.5f : 3.5f).epsilon(1e-4));
        }
    }
}

TEST_CASE_FIXTURE(Reporting, "meshes: double-sided surfaces keep one winding and preserve seams") {
    const float3 points[]{{-4, 0, -4}, {-4, 0, 4}, {4, 0, 4}, {4, 0, -4}};
    const std::vector<uint32_t> indices{0, 1, 2, 0, 2, 3};
    for (float side : {-1.f, 1.f}) {
        World w{context};
        const auto mesh = w.AddMesh(points, indices);
        w.Shapes[mesh].DoubleSided = true;
        const auto target = side < 0 ? w.AddCompound(std::vector<Index>{mesh}) : mesh;
        w.AddBody({.Pose = At(float3{0, 2, 0}), .Shape = target, .Friction = 0});
        const auto box = w.AddBody({.Pose = At(float3{-0.5f, 2 + side * Half, 0}), .Velocity = {.Linear = {0.5f, 0, 0}}, .Shape = w.AddShape(UnitBox), .Density = 1, .Friction = 0});
        w.TrackContacts = true;
        for (int step = 0; step < 120; ++step) {
            solver.Step(w, {.Gravity = {0, -side * 9.81f, 0}});
            for (const auto &c : w.TakeContactChanges()) {
                if (c.Kind == ContactRemoved) continue;
                CHECK(dot(c.Normal, float3{0, side, 0}) > 0.99f);
            }
        }
        CHECK(w.Poses[box].Position.y == doctest::Approx(2 + side * Half).epsilon(0.001));
        CHECK(w.Poses[box].Position.x == doctest::Approx(0.5f).epsilon(0.02));
        CHECK(w.Velocities[box].Linear.x == doctest::Approx(0.5f).epsilon(0.02));
        CHECK(w.Shapes[mesh].TriangleCount == 2);
    }
}
