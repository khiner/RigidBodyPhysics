// The world's pools: stable indices, mass properties against their closed forms, and a full pool refusing an add rather than growing.

#include "World.h"
#include "Shapes.h"

#include <algorithm>
#include <numbers>
#include <vector>

#include <doctest/doctest.h>

using namespace rbp;

namespace {
// A world of the default capacities, rebuilt per test case.
// A case needing other capacities takes `Device` and builds its own.
struct Device {
    const mtl::Context context;
};
struct OneWorld : Device {
    World world{context};
};

// The smallest surface: a flat quad as two triangles.
Index AddQuadMesh(World &world, Pose local = IdentityPose) {
    const std::vector<float3> points{float3{-1, 0, -1}, float3{1, 0, -1}, float3{1, 0, 1}, float3{-1, 0, 1}};
    const std::vector<uint32_t> indices{0, 1, 2, 0, 2, 3};
    return world.AddMesh(points, indices, local);
}
} // namespace

TEST_CASE("a solid box gets the mass and inertia its density implies") {
    // A 1 m cube of water masses 1000 kg, and its inertia is m * (e^2 + e^2) / 12 on every axis.
    const auto mass = MassProperties(UnitBox, 1000);
    CHECK(1 / mass.InvMass == doctest::Approx(1000));
    for (uint32_t axis = 0; axis < 3; ++axis)
        CHECK(1 / mass.InvInertiaLocal[axis] == doctest::Approx(1000 * 2.f / 12));

    // Inertia scales as m*e^2, so doubling every extent is 8x the mass and 32x the inertia.
    constexpr Shape DoubleBox{.HalfExtents = {1, 1, 1}, .Kind = ShapeBox};
    const auto bigger = MassProperties(DoubleBox, 1000);
    CHECK(mass.InvMass / bigger.InvMass == doctest::Approx(8));
    CHECK(mass.InvInertiaLocal[0] / bigger.InvInertiaLocal[0] == doctest::Approx(32));
}

TEST_CASE("a solid sphere gets the mass and inertia its density implies") {
    // A sphere of radius r masses rho 4/3 pi r^3, with inertia 2/5 m r^2 about every axis.
    constexpr float Radius = 0.5f;
    constexpr Shape UnitSphere{.Radius = Radius, .Kind = ShapeSphere};
    const auto mass = MassProperties(UnitSphere, 1000);
    const float expected = 1000 * 4.f / 3 * std::numbers::pi_v<float> * Radius * Radius * Radius;
    CHECK(1 / mass.InvMass == doctest::Approx(expected));
    for (uint32_t axis = 0; axis < 3; ++axis)
        CHECK(1 / mass.InvInertiaLocal[axis] == doctest::Approx(2.f / 5 * expected * Radius * Radius));

    // Mass goes as r^3 and inertia as m r^2, so doubling the radius is 8x and 32x.
    constexpr Shape BigSphere{.Radius = 2 * Radius, .Kind = ShapeSphere};
    const auto bigger = MassProperties(BigSphere, 1000);
    CHECK(mass.InvMass / bigger.InvMass == doctest::Approx(8));
    CHECK(mass.InvInertiaLocal[0] / bigger.InvInertiaLocal[0] == doctest::Approx(32));
}

TEST_CASE("a capsule's mass properties bracket the shapes it is made of") {
    constexpr float Radius = 0.5f;
    constexpr Shape Sphere{.Radius = Radius, .Kind = ShapeSphere};

    // A capsule of zero half-extent is a sphere.
    constexpr Shape Degenerate{.HalfExtents = {0, 0, 0}, .Radius = Radius, .Kind = ShapeCapsule};
    const auto collapsed = MassProperties(Degenerate, 1000), ball = MassProperties(Sphere, 1000);
    CHECK(1 / collapsed.InvMass == doctest::Approx(1 / ball.InvMass));
    for (uint32_t axis = 0; axis < 3; ++axis)
        CHECK(collapsed.InvInertiaLocal[axis] == doctest::Approx(ball.InvInertiaLocal[axis]));

    constexpr Shape Pill{.HalfExtents = {0, 1, 0}, .Radius = Radius, .Kind = ShapeCapsule};
    const auto mass = MassProperties(Pill, 1000);
    // Mass is the cylinder plus the sphere formed by its two caps.
    constexpr float Pi = std::numbers::pi_v<float>;
    const float expected = 1000 * Pi * Radius * Radius * 2 + 1000 * 4.f / 3 * Pi * Radius * Radius * Radius;
    CHECK(1 / mass.InvMass == doctest::Approx(expected));
    // Long and thin, so it resists turning end over end far more than turning about its own length.
    CHECK(1 / mass.InvInertiaLocal[0] > 4 / mass.InvInertiaLocal[1]);
    CHECK(mass.InvInertiaLocal[0] == doctest::Approx(mass.InvInertiaLocal[2])); // symmetric about the long axis
}

TEST_CASE("a plane and a zero-density body are static") {
    CHECK(MassProperties(GroundPlane, 1000).InvMass == 0);
    CHECK(MassProperties(UnitBox, 0).InvMass == 0);
}

TEST_CASE_FIXTURE(OneWorld, "bodies and shapes take stable indices") {
    const auto box = world.AddShape(UnitBox);
    const auto plane = world.AddShape(GroundPlane);
    CHECK(box == 0);
    CHECK(plane == 1);
    CHECK(world.ShapeCount() == 2);

    const auto falling = world.AddBody({.Pose = At(float3{0, 5, 0}), .Shape = box});
    const auto ground = world.AddBody({.Shape = plane});
    CHECK(falling == 0);
    CHECK(ground == 1);
    CHECK(world.BodyCount() == 2);

    CHECK(world.Poses[falling].Position.y == 5);
    CHECK(world.Masses[falling].InvMass == doctest::Approx(1.f / 1000));
    CHECK(world.Masses[ground].InvMass == 0);
    CHECK(world.BodyShapes[ground] == plane);
}

TEST_CASE_FIXTURE(OneWorld, "new geometry recomputes what the shape decides and keeps what the body decides") {
    // Mass and inertia come from the shape and density, and the motion properties are the body's.
    const auto box = world.AddShape(UnitBox);
    constexpr Shape BigBox{.HalfExtents = {1, 1, 1}, .Kind = ShapeBox};
    const auto body = world.AddBody({.Shape = box, .GravityScale = 0.5f, .LinearDamping = 0.04f, .AngularDamping = 0.1f});
    REQUIRE(world.SetBodyShape(body, world.AddShape(BigBox)));

    CHECK(world.Masses[body].InvMass == doctest::Approx(1.f / 8000));
    CHECK(world.Masses[body].GravityScale == 0.5f);
    CHECK(world.Masses[body].LinearDamping == 0.04f);
    CHECK(world.Masses[body].AngularDamping == 0.1f);
}

TEST_CASE_FIXTURE(Device, "a full pool refuses the add and counts it") {
    World world{context, {.Bodies = 2, .Shapes = 1}};

    const auto box = world.AddShape(UnitBox);
    CHECK(box == 0);
    CHECK(world.AddShape(GroundPlane) == NoIndex);
    CHECK(world.Overflow.Shapes == 1);
    CHECK(world.ShapeCount() == 1);

    CHECK(world.AddBody({.Shape = box}) == 0);
    CHECK(world.AddBody({.Shape = box}) == 1);
    CHECK(world.AddBody({.Shape = box}) == NoIndex);
    CHECK(world.Overflow.Bodies == 1);
    CHECK(world.BodyCount() == 2);
}

TEST_CASE_FIXTURE(Device, "every body owns a fixed run of contact slots") {
    World world{context, {.Bodies = 3}};
    CHECK(world.Contacts.Capacity == 3 * ContactsPerBody);
    // The pool is never appended to, so an unfilled slot reads as inactive rather than as stale.
    CHECK(std::ranges::none_of(world.Contacts.All(), [](const Contact &c) { return c.Active; }));
}

TEST_CASE_FIXTURE(OneWorld, "a shape a live body still has is not removed") {
    const auto box = world.AddShape(UnitBox);
    const auto body = world.AddBody({.Shape = box});

    CHECK(!world.RemoveShape(box)); // a live body still uses it
    REQUIRE(world.RemoveBody(body));
    CHECK(!world.Alive(body));
    CHECK(!world.RemoveBody(body)); // a second remove reports failure
    // The body's slot stays reserved until a step has run, and the shape it used is free immediately.
    CHECK(world.RemoveShape(box));
    CHECK(!world.RemoveShape(box));
}

TEST_CASE_FIXTURE(Device, "a removed mesh gives its runs back") {
    // Every pool sized to one quad, so a second quad fits only if the first released its storage.
    World world{context, {.Shapes = 4, .ShapeVertices = 4, .Triangles = 2, .BvhNodes = 1}};

    const auto mesh = AddQuadMesh(world);
    REQUIRE(mesh != NoIndex);
    CHECK(AddQuadMesh(world) == NoIndex); // all three pools are full
    REQUIRE(world.RemoveShape(mesh));

    const auto again = AddQuadMesh(world);
    REQUIRE(again != NoIndex);
    CHECK(again == mesh);
    CHECK(world.Shapes[again].FirstVertex == 0);
    CHECK(world.Shapes[again].FirstTriangle == 0);
    CHECK(world.Shapes[again].RootNode == 0);
    // The triangles are rebased onto the run actually allocated.
    CHECK(world.Triangles[0].A < world.Shapes[again].VertexCount);
}

TEST_CASE_FIXTURE(Device, "runs given back next to each other are one run again") {
    // A twelve vertex hull fits in a pool of sixteen only if the two eight-vertex runs merged.
    World world{context, {.ShapeVertices = 16}};
    const std::vector<float3> cube = CubeCorners(2);
    const std::vector<float3> prism = PrismPoints(6, 1, 1); // twelve corners and no interior points

    const auto first = world.AddHull(cube), second = world.AddHull(cube);
    REQUIRE(first != NoIndex);
    REQUIRE(second != NoIndex);
    CHECK(world.AddHull(prism) == NoIndex);
    REQUIRE(world.RemoveShape(first)); // a hole in the middle
    REQUIRE(world.RemoveShape(second)); // the tail, which absorbs the hole

    const auto big = world.AddHull(prism);
    REQUIRE(big != NoIndex);
    CHECK(world.Shapes[big].VertexCount == 12);
    CHECK(world.Shapes[big].FirstVertex == 0);
}

TEST_CASE_FIXTURE(OneWorld, "a host-supplied mass is what a shape with no volume cannot say for itself") {
    const auto mesh = AddQuadMesh(world);
    REQUIRE(mesh != NoIndex);

    // A surface has no interior to integrate, so a mesh body with no authored mass is static.
    const auto scenery = world.AddBody({.Shape = mesh});
    CHECK(world.Masses[scenery].InvMass == 0);

    const auto moving = world.AddBody({.Shape = mesh, .Mass = {{.Mass = 4, .Inertia = {2, 8, 16}}}});
    CHECK(world.Masses[moving].InvMass == doctest::Approx(0.25f));
    CHECK(world.Masses[moving].InvInertiaLocal[0] == doctest::Approx(0.5f));
    CHECK(world.Masses[moving].InvInertiaLocal[1] == doctest::Approx(0.125f));
    CHECK(world.Masses[moving].InvInertiaLocal[2] == doctest::Approx(0.0625f));

    // An authored mass takes precedence, and the density is unread when one is given.
    const auto box = world.AddShape(UnitBox);
    const auto authored = world.AddBody({.Shape = box, .Density = 1000, .Mass = {{.Mass = 2, .Inertia = {1, 1, 1}}}});
    CHECK(world.Masses[authored].InvMass == doctest::Approx(0.5f));

    // Motion properties live in their own fields, so no value is stored in two places.
    const auto floaty = world.AddBody({.Shape = mesh, .Mass = {{.Mass = 4, .Inertia = {1, 1, 1}}}, .GravityScale = 0.5f, .LinearDamping = 0.25f});
    CHECK(world.Masses[floaty].GravityScale == doctest::Approx(0.5f));
    CHECK(world.Masses[floaty].LinearDamping == doctest::Approx(0.25f));

    // SetBodyShape follows the same rule and preserves the motion properties.
    REQUIRE(world.SetBodyShape(scenery, mesh, 1000, AuthoredMass{.Mass = 5, .Inertia = {1, 2, 4}}));
    CHECK(world.Masses[scenery].InvMass == doctest::Approx(0.2f));
    CHECK(world.Masses[scenery].InvInertiaLocal[2] == doctest::Approx(0.25f));
    REQUIRE(world.SetBodyShape(floaty, mesh)); // omitting the mass makes it static again
    CHECK(world.Masses[floaty].InvMass == 0);
    CHECK(world.Masses[floaty].GravityScale == doctest::Approx(0.5f));
}

TEST_CASE_FIXTURE(OneWorld, "a hull's local pose is the caller's frame with the cook's underneath it") {
    // Shape::Local is the caller's frame composed onto the cook's, in that order.
    // Neither is the identity here, so a swapped order places the shape somewhere else.
    const std::vector<float3> points = CubeCorners(1, float3{3, -2, 5});
    const Pose local = At(float3{0.4f, 0.25f, -0.1f}, QuatFromRotationVector(float3{0.3f, -0.7f, 0.2f}));

    // With no local pose given, the cooked frame is the body frame.
    Pose cooked{};
    const auto plain = world.AddHull(points, &cooked);
    REQUIRE(plain != NoIndex);
    CHECK(simd::length(world.Shapes[plain].Local.Position) == 0);
    CHECK(world.Shapes[plain].Local.Orientation.w == 1);

    const auto placed = world.AddHull(points, nullptr, local);
    REQUIRE(placed != NoIndex);
    const Pose want = ComposePose(local, cooked);
    CHECK(simd::distance(world.Shapes[placed].Local.Position, want.Position) < 1e-6f);
    CHECK(simd::distance(world.Shapes[placed].Local.Orientation, want.Orientation) < 1e-6f);

    // The same solid supplied already moved places its geometry identically, and SolveTest steps that equivalence.
    // A principal axis is defined only up to sign, so the comparison goes through the placed geometry rather than the quaternions.
    std::vector<float3> moved;
    for (const float3 point : points) moved.push_back(WorldPoint(local, point));
    const auto baked = world.AddHull(moved, nullptr, IdentityPose);
    REQUIRE(baked != NoIndex);
    CHECK(simd::distance(world.Shapes[baked].Local.Position, want.Position) < 1e-5f);
    const auto corners = [this](Index shape, Pose local_pose) {
        std::vector<float3> at;
        for (uint32_t i = 0; i < world.Shapes[shape].VertexCount; ++i)
            at.push_back(WorldPoint(local_pose, world.ShapeVertices[world.Shapes[shape].FirstVertex + i]));
        return at;
    };
    const std::vector<float3> theirs = corners(placed, want);
    for (const float3 mine : corners(baked, world.Shapes[baked].Local)) CHECK(NearestTo(mine, theirs) < 1e-4f);
}

TEST_CASE_FIXTURE(OneWorld, "a body wearing an offset shape is refused unless the host says what it weighs") {
    // A shape's inertia is about its own centre, and the body frame is the centre of mass with the inertia diagonal.
    // The two coincide only while the shape sits on the origin unrotated.
    constexpr Shape OffsetBox{.HalfExtents = {Half, Half, Half}, .Kind = ShapeBox, .Local = {{0.3f, 0, 0}, {0, 0, 0, 1}}};
    const auto offset = world.AddShape(OffsetBox);
    const auto centred = world.AddShape(UnitBox);

    CHECK(world.AddBody({.Shape = offset}) == NoIndex);
    CHECK(world.OffsetsWithoutMass == 1);
    CHECK(world.BodyCount() == 0);
    CHECK(world.Overflow.Bodies == 0); // the refusal is the offset rule, not an exhausted pool

    const auto authored = world.AddBody({.Shape = offset, .Mass = {{.Mass = 1000, .Inertia = {200, 250, 250}}}});
    REQUIRE(authored != NoIndex);
    CHECK(world.Masses[authored].InvMass == doctest::Approx(1.f / 1000));

    // A static body has no inertia about a point to get wrong, so an offset shape is allowed.
    const auto fixed = world.AddBody({.Shape = offset, .Density = 0});
    REQUIRE(fixed != NoIndex);
    CHECK(world.Masses[fixed].InvMass == 0);
    const auto mesh = AddQuadMesh(world, At(float3{0, 0, 0}, QuatFromRotationVector(float3{0, 0.7f, 0})));
    REQUIRE(mesh != NoIndex);
    CHECK(world.AddBody({.Shape = mesh}) != NoIndex);
    CHECK(world.OffsetsWithoutMass == 1); // none of those three was refused

    // New geometry follows the same rule, and a refusal leaves the body unchanged.
    const auto body = world.AddBody({.Shape = centred});
    REQUIRE(body != NoIndex);
    CHECK(!world.SetBodyShape(body, offset));
    CHECK(world.OffsetsWithoutMass == 2);
    CHECK(world.BodyShapes[body] == centred);
    CHECK(world.Masses[body].InvMass == doctest::Approx(1.f / 1000));
    CHECK(world.SetBodyShape(body, offset, 1000, AuthoredMass{.Mass = 2, .Inertia = {1, 1, 1}}));
    CHECK(world.BodyShapes[body] == offset);
    CHECK(world.Masses[body].InvMass == doctest::Approx(0.5f));
}

// A compound is the hull cook's arithmetic over children, so the same closed forms apply.
TEST_CASE_FIXTURE(OneWorld, "a solid described in two halves weighs what the solid weighs") {
    std::vector<Index> halves;
    for (const float side : {-1.f, 1.f})
        halves.push_back(world.AddShape({.HalfExtents = {Half / 2, Half, Half}, .Kind = ShapeBox, .Local = At(float3{side * Half / 2, 0, 0})}));
    REQUIRE(halves[0] != NoIndex);
    REQUIRE(halves[1] != NoIndex);

    Pose frame{};
    const auto compound = world.AddCompound(halves, &frame);
    REQUIRE(compound != NoIndex);
    // The two halves are already centred on the origin, so the body frame is unmoved.
    CHECK(simd::length(frame.Position) < 1e-6f);
    CHECK(std::abs(frame.Orientation.w) == doctest::Approx(1));

    const auto solid = MassProperties(UnitBox, 1000);
    const auto built = MassOf(world, compound, 1000);
    CHECK(1 / built.InvMass == doctest::Approx(1 / solid.InvMass).epsilon(1e-5));
    for (uint32_t axis = 0; axis < 3; ++axis)
        CHECK(1 / built.InvInertiaLocal[axis] == doctest::Approx(1 / solid.InvInertiaLocal[axis]).epsilon(1e-5));

    // A compound is one material, so its mass scales with density as a single piece does.
    const auto lighter = MassOf(world, compound, 500);
    CHECK(lighter.InvMass / built.InvMass == doctest::Approx(2));

    // The children are copies re-expressed in that frame, and the shapes passed in are left unchanged.
    CHECK(world.ShapeCount() == 5); // two given, two copied, and the compound itself
    for (uint32_t i = 0; i < 2; ++i) {
        const Index child = ChildOf(world.Shapes[compound], i);
        REQUIRE(child != NoIndex);
        CHECK(child != halves[i]);
        CHECK(simd::distance(world.Shapes[child].Local.Position, world.Shapes[halves[i]].Local.Position) < 1e-6f);
    }
    CHECK(ChildOf(world.Shapes[compound], 2) == NoIndex); // the run ends at the terminator
    // The compound sits on the body's origin, so the body frame is the compound's own frame.
    CHECK(simd::length(world.Shapes[compound].Local.Position) == 0);
}

TEST_CASE_FIXTURE(OneWorld, "a dumbbell's inertia is its pieces carried by the parallel axis theorem") {
    // Two spheres on a rod along y: about y each contributes only its own inertia, and about x and z it contributes m d^2 as well.
    constexpr float Radius = 0.2f, Rod = 0.05f, Reach = 0.6f, Density = 1000;
    constexpr Shape RodShape{.HalfExtents = {0, Reach, 0}, .Radius = Rod, .Kind = ShapeCapsule};
    constexpr Shape BallShape{.Radius = Radius, .Kind = ShapeSphere};
    std::vector<Index> parts{world.AddShape(RodShape)};
    for (const float side : {-1.f, 1.f})
        parts.push_back(world.AddShape({.Radius = Radius, .Kind = ShapeSphere, .Local = At(float3{0, side * Reach, 0})}));

    Pose frame{};
    const auto dumbbell = world.AddCompound(parts, &frame);
    REQUIRE(dumbbell != NoIndex);
    CHECK(simd::length(frame.Position) < 1e-6f); // symmetric about the origin

    const auto bar = MassProperties(RodShape, Density);
    const auto ball = MassProperties(BallShape, Density);
    const float ball_mass = 1 / ball.InvMass;
    const float along = 1 / bar.InvInertiaLocal.y + 2 / ball.InvInertiaLocal.y;
    const float across = 1 / bar.InvInertiaLocal.x + 2 * (1 / ball.InvInertiaLocal.x + ball_mass * Reach * Reach);

    const auto built = MassOf(world, dumbbell, Density);
    CHECK(1 / built.InvMass == doctest::Approx(1 / bar.InvMass + 2 * ball_mass).epsilon(1e-5));
    CHECK(1 / built.InvInertiaLocal.x == doctest::Approx(across).epsilon(1e-5));
    CHECK(1 / built.InvInertiaLocal.y == doctest::Approx(along).epsilon(1e-5));
    CHECK(1 / built.InvInertiaLocal.z == doctest::Approx(across).epsilon(1e-5));
}

TEST_CASE_FIXTURE(OneWorld, "a compound off the origin says where it put the body frame") {
    // The frame maps a body's pose to the pose of the pieces as authored, like CookedHull::Frame.
    constexpr float TopY = 0.75f, LegHigh = 0.35f;
    std::vector<Index> parts{world.AddShape({.HalfExtents = {1, 0.05f, 0.6f}, .Kind = ShapeBox, .Local = At(float3{0, TopY, 0})})};
    for (const float x : {-0.9f, 0.9f})
        for (const float z : {-0.5f, 0.5f})
            parts.push_back(world.AddShape({.HalfExtents = {0.05f, LegHigh, 0.05f}, .Kind = ShapeBox, .Local = At(float3{x, LegHigh, z})}));

    Pose frame{};
    const auto table = world.AddCompound(parts, &frame);
    REQUIRE(table != NoIndex);
    CHECK(std::abs(frame.Position.x) < 1e-6f);
    CHECK(std::abs(frame.Position.z) < 1e-6f);
    CHECK(frame.Position.y > 0); // the mass is above the floor the legs stand on
    CHECK(frame.Position.y < TopY); // below the top, most of the mass being in the slab

    // The top's centre is furthest from the origin, so it is the piece checked.
    const Index top = ChildOf(world.Shapes[table], 0);
    REQUIRE(top != NoIndex);
    const Pose local = world.Shapes[top].Local;
    const float3 placed = WorldPoint(frame, local.Position);
    CHECK(simd::distance(placed, float3{0, TopY, 0}) < 1e-5f);
}

TEST_CASE_FIXTURE(OneWorld, "a compound the engine will not make is refused and counted") {
    const auto box = world.AddShape(UnitBox);
    const auto plane = world.AddShape(GroundPlane);
    const auto mesh = AddQuadMesh(world);
    REQUIRE(mesh != NoIndex);
    const auto pair = world.AddCompound(std::vector<Index>{box, box});
    REQUIRE(pair != NoIndex);

    uint32_t refused = 0;
    // One more child than the run has room for.
    CHECK(world.AddCompound(std::vector<Index>(ChildrenPerCompound + 1, box)) == NoIndex);
    CHECK(world.RefusedCompounds == ++refused);
    // A child that is itself a compound, making the tree two deep.
    CHECK(world.AddCompound(std::vector<Index>{box, pair}) == NoIndex);
    CHECK(world.RefusedCompounds == ++refused);
    // A surface has no interior and a plane is unbounded, so neither has a volume.
    CHECK(world.AddCompound(std::vector<Index>{box, mesh}) == NoIndex);
    CHECK(world.RefusedCompounds == ++refused);
    CHECK(world.AddCompound(std::vector<Index>{box, plane}) == NoIndex);
    CHECK(world.RefusedCompounds == ++refused);
    // An empty list, or a child that is not a live shape.
    CHECK(world.AddCompound({}) == NoIndex);
    CHECK(world.RefusedCompounds == ++refused);
    CHECK(world.AddCompound(std::vector<Index>{box, world.ShapeCount() + 7}) == NoIndex);
    CHECK(world.RefusedCompounds == ++refused);
    // The world had room for every one, so none of them is an overflow.
    CHECK(world.Overflow.Shapes == 0);

    // Eight fits, so the refusal above is the run's size rather than an off-by-one.
    CHECK(world.AddCompound(std::vector<Index>(ChildrenPerCompound, box)) != NoIndex);
    CHECK(world.RefusedCompounds == refused);
}

TEST_CASE_FIXTURE(OneWorld, "a compound gives its children's slots back with it, and not before") {
    const std::vector<float3> cube = CubeCorners(1);
    const auto box = world.AddShape(UnitBox);
    const auto hull = world.AddHull(cube);
    REQUIRE(hull != NoIndex);
    const uint32_t vertices = world.Shapes[hull].VertexCount, faces = world.Shapes[hull].FaceCount;
    const uint32_t before = world.ShapeCount();

    const auto compound = world.AddCompound(std::vector<Index>{box, hull});
    REQUIRE(compound != NoIndex);
    CHECK(world.ShapeCount() == before + 3); // two copies and the compound
    const Index child = ChildOf(world.Shapes[compound], 1);
    REQUIRE(child != NoIndex);
    // A hull child takes its own run rather than sharing the original's, so either can be freed.
    CHECK(world.Shapes[child].FirstVertex != world.Shapes[hull].FirstVertex);
    CHECK(world.Shapes[child].VertexCount == vertices);
    CHECK(world.Shapes[child].FaceCount == faces);

    // A child belongs to its compound, and freeing one would strand the compound's reference.
    CHECK(!world.RemoveShape(child));
    // A compound a live body uses is refused like any other shape.
    const auto body = world.AddBody({.Shape = compound});
    REQUIRE(body != NoIndex);
    CHECK(!world.RemoveShape(compound));
    REQUIRE(world.RemoveBody(body));

    REQUIRE(world.RemoveShape(compound));
    CHECK(world.ShapeCount() == before); // the children's slots were freed with it
    // The shapes it was built from are untouched, since the children were copies.
    CHECK(world.RemoveShape(hull));
    CHECK(world.RemoveShape(box));
}

TEST_CASE_FIXTURE(OneWorld, "a compound weighs a body the way any other shape does") {
    const auto box = world.AddShape(UnitBox);
    const auto compound = world.AddCompound(std::vector<Index>{box});
    REQUIRE(compound != NoIndex);
    const auto expected = MassOf(world, compound, 1000);

    // A compound sits on the body's origin by construction, so no authored mass is needed.
    const auto body = world.AddBody({.Shape = compound});
    REQUIRE(body != NoIndex);
    CHECK(world.OffsetsWithoutMass == 0);
    CHECK(world.Masses[body].InvMass == doctest::Approx(expected.InvMass));

    // New geometry recomputes from it like any shape, and an authored mass still takes precedence.
    CHECK(world.SetBodyShape(body, compound, 500));
    CHECK(world.Masses[body].InvMass == doctest::Approx(2 * expected.InvMass));
    CHECK(world.SetBodyShape(body, compound, 1000, AuthoredMass{.Mass = 4, .Inertia = {1, 1, 1}}));
    CHECK(world.Masses[body].InvMass == doctest::Approx(0.25f));
}

// Two static boxes laid edge to edge, forming the join World::WeldStatic handles.
// They share one shape, since a shape is shared and a buried face belongs to the body rather than to the shape.
namespace {
constexpr float TileHalf = 2.5f, TileTop = 0.25f;

Index AddTile(World &world, Index shape, float side) {
    return world.AddBody({.Pose = At(float3{side * TileHalf, 0, 0}), .Shape = shape, .Density = 0});
}
} // namespace

TEST_CASE_FIXTURE(OneWorld, "static faces that cover each other are buried, on copies the weld owns") {
    const auto tile = world.AddShape({.HalfExtents = {TileHalf, TileTop, TileHalf}, .Kind = ShapeBox});
    const auto near = AddTile(world, tile, -1), far = AddTile(world, tile, 1);
    REQUIRE(far != NoIndex);
    const uint32_t before = world.ShapeCount();

    CHECK(world.WeldStatic() == 2); // one face of each, the face the other covers
    // Each body takes a private copy with its own mask, and the host's shape is untouched.
    CHECK(world.BodyShapes[near] != tile);
    CHECK(world.BodyShapes[far] != tile);
    CHECK(world.ShapeCount() == before + 2);
    CHECK(InternalFaces(world.Shapes[tile]) == 0);
    CHECK(InternalFaces(world.Shapes[world.BodyShapes[near]]) == 1u << BoxFaceIndex(0, true));
    CHECK(InternalFaces(world.Shapes[world.BodyShapes[far]]) == 1u << BoxFaceIndex(0, false));

    // Each call recomputes the whole result rather than accumulating, reusing the copies already made.
    const Index copy = world.BodyShapes[near];
    CHECK(world.WeldStatic() == 2);
    CHECK(world.ShapeCount() == before + 2);
    CHECK(world.BodyShapes[near] == copy);

    // A host passing back the shape it read from BodyShapes passes in the weld's copy.
    REQUIRE(world.SetBodyShape(near, world.BodyShapes[near], 0));
    CHECK(world.BodyShapes[near] == copy);
    CHECK(InternalFaces(world.Shapes[copy]) == 1u << BoxFaceIndex(0, true));

    // A weld copy belongs to the weld rather than the host, so the removed tile's copy is released with it.
    REQUIRE(world.RemoveBody(far));
    CHECK(world.WeldStatic() == 0);
    CHECK(InternalFaces(world.Shapes[world.BodyShapes[near]]) == 0);
    CHECK(world.ShapeCount() == before + 1);
}

TEST_CASE_FIXTURE(OneWorld, "the weld speaks only for what nothing can move") {
    const auto tile = world.AddShape({.HalfExtents = {TileHalf, TileTop, TileHalf}, .Kind = ShapeBox});
    const auto near = AddTile(world, tile, -1);
    const auto far = AddTile(world, tile, 1);
    REQUIRE(far != NoIndex);

    // A kinematic body is static by its mass alone, but it can move away, so it buries no faces.
    world.Velocities[far].Linear = {1, 0, 0};
    CHECK(world.WeldStatic() == 0);
    CHECK(InternalFaces(world.Shapes[world.BodyShapes[near]]) == 0);
    world.Velocities[far] = {};
    CHECK(world.WeldStatic() == 2);

    // A body the solve can move buries no faces either.
    // The weld is recomputed rather than unwound, so the call after the mass write restores the neighbour's face.
    world.Masses[far].InvMass = 1e-3f;
    CHECK(world.WeldStatic() == 0);
    CHECK(InternalFaces(world.Shapes[world.BodyShapes[near]]) == 0);
    CHECK(InternalFaces(world.Shapes[world.BodyShapes[far]]) == 0);
}

TEST_CASE_FIXTURE(OneWorld, "a partly covered face is not buried") {
    // A leg under a slab: the leg's top is fully covered, and most of the slab's bottom is open air.
    const auto leg = world.AddBody({.Pose = At(float3{0, 0.5f, 0}), .Shape = world.AddShape({.HalfExtents = {0.5f, 0.5f, 0.5f}, .Kind = ShapeBox}), .Density = 0});
    const auto slab = world.AddBody({.Pose = At(float3{0, 1.25f, 0}), .Shape = world.AddShape({.HalfExtents = {2, 0.25f, 2}, .Kind = ShapeBox}), .Density = 0});
    REQUIRE(slab != NoIndex);
    const uint32_t before = world.ShapeCount();

    CHECK(world.WeldStatic() == 1);
    CHECK(InternalFaces(world.Shapes[world.BodyShapes[leg]]) == 1u << BoxFaceIndex(1, true));
    CHECK(world.ShapeCount() == before + 1); // only the leg needed a copy, the slab having no buried face

    // Removing the slab removes the burial it caused, and the slab never needed a copy of its own.
    REQUIRE(world.RemoveBody(slab));
    CHECK(world.WeldStatic() == 0);
    CHECK(InternalFaces(world.Shapes[world.BodyShapes[leg]]) == 0);
    CHECK(world.ShapeCount() == before + 1);
}
