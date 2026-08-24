// The world's pools: stable indices, mass properties against their closed forms, and a full pool
// refusing rather than growing.

#include "World.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include <doctest/doctest.h>

namespace {
constexpr Shape UnitBox{.HalfExtents = {0.5f, 0.5f, 0.5f}, .Kind = ShapeBox};
constexpr Shape GroundPlane{.Normal = {0, 1, 0}, .Offset = 0, .Kind = ShapePlane};
} // namespace

TEST_CASE("a solid box gets the mass and inertia its density implies") {
    // A 1 m cube of water masses 1000 kg, and a cube's inertia is m * (e^2 + e^2) / 12 on every axis.
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
    // A sphere of radius r masses rho 4/3 pi r^3 and carries 2/5 m r^2 about every axis.
    constexpr float Radius = 0.5f;
    constexpr Shape UnitSphere{.Radius = Radius, .Kind = ShapeSphere};
    const auto mass = MassProperties(UnitSphere, 1000);
    const float expected = 1000 * 4.f / 3 * 3.14159265358979f * Radius * Radius * Radius;
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
    // No single closed form worth restating, so this checks the two limits it has to sit between and
    // the one case where it collapses to something already checked.
    constexpr float Radius = 0.5f;
    constexpr Shape Sphere{.Radius = Radius, .Kind = ShapeSphere};

    // A capsule with no cylinder in it is a sphere.
    constexpr Shape Degenerate{.HalfExtents = {0, 0, 0}, .Radius = Radius, .Kind = ShapeCapsule};
    CHECK(1 / MassProperties(Degenerate, 1000).InvMass == doctest::Approx(1 / MassProperties(Sphere, 1000).InvMass));
    for (uint32_t axis = 0; axis < 3; ++axis)
        CHECK(MassProperties(Degenerate, 1000).InvInertiaLocal[axis] ==
              doctest::Approx(MassProperties(Sphere, 1000).InvInertiaLocal[axis]));

    constexpr Shape Pill{.HalfExtents = {0, 1, 0}, .Radius = Radius, .Kind = ShapeCapsule};
    const auto mass = MassProperties(Pill, 1000);
    // Mass is the cylinder plus the sphere its two caps make.
    const float pi = 3.14159265358979f;
    const float expected = 1000 * pi * Radius * Radius * 2 + 1000 * 4.f / 3 * pi * Radius * Radius * Radius;
    CHECK(1 / mass.InvMass == doctest::Approx(expected));
    // Long and thin: it resists turning end over end far more than turning about its own length.
    CHECK(1 / mass.InvInertiaLocal[0] > 4 / mass.InvInertiaLocal[1]);
    CHECK(mass.InvInertiaLocal[0] == doctest::Approx(mass.InvInertiaLocal[2])); // and is symmetric across it
}

TEST_CASE("a plane and a zero-density body are static") {
    CHECK(MassProperties(GroundPlane, 1000).InvMass == 0);
    CHECK(MassProperties(UnitBox, 0).InvMass == 0);
}

TEST_CASE("bodies and shapes take stable indices") {
    const mtl::Context context;
    World world{context};

    const auto box = world.AddShape(UnitBox);
    const auto plane = world.AddShape(GroundPlane);
    CHECK(box == 0);
    CHECK(plane == 1);
    CHECK(world.ShapeCount() == 2);

    const auto falling = world.AddBody({.Pose = {.Position = {0, 5, 0}, .Orientation = {0, 0, 0, 1}}, .Shape = box});
    const auto ground = world.AddBody({.Shape = plane});
    CHECK(falling == 0);
    CHECK(ground == 1);
    CHECK(world.BodyCount() == 2);

    CHECK(world.Poses[falling].Position.y == 5);
    CHECK(world.Masses[falling].InvMass == doctest::Approx(1.f / 1000));
    CHECK(world.Masses[ground].InvMass == 0);
    CHECK(world.BodyShapes[ground] == plane);
}

TEST_CASE("a full pool refuses the add and counts it") {
    const mtl::Context context;
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

TEST_CASE("every body owns a fixed run of contact slots") {
    const mtl::Context context;
    World world{context, {.Bodies = 3}};
    CHECK(world.Contacts.Capacity == 3 * ContactsPerBody);
    // Nothing is appended, so a slot that was never filled reads as inactive rather than as stale.
    CHECK(std::ranges::none_of(world.Contacts.All(), [](const Contact &c) { return c.Active; }));
}

TEST_CASE("a shape a live body still has is not removed") {
    const mtl::Context context;
    World world{context};
    const auto box = world.AddShape(UnitBox);
    const auto body = world.AddBody({.Shape = box});

    CHECK(!world.RemoveShape(box)); // something is standing on it
    REQUIRE(world.RemoveBody(body));
    CHECK(!world.Alive(body));
    CHECK(!world.RemoveBody(body)); // and saying so twice is the caller's mistake, not a second removal
    // The body's slot is not free until a step has run, but the body itself is already gone, and the
    // shape is what it was holding rather than the slot.
    CHECK(world.RemoveShape(box));
    CHECK(!world.RemoveShape(box));
}

TEST_CASE("a removed mesh gives its runs back") {
    // Every pool sized to exactly one quad, so a second one fits only if the first really gave its
    // storage back rather than leaving it stranded behind the bump pointer.
    const mtl::Context context;
    World world{context, {.Shapes = 4, .ShapeVertices = 4, .Triangles = 2, .BvhNodes = 1}};
    const std::vector<float3> points{float3{-1, 0, -1}, float3{1, 0, -1}, float3{1, 0, 1}, float3{-1, 0, 1}};
    const std::vector<uint32_t> indices{0, 1, 2, 0, 2, 3};

    const auto mesh = world.AddMesh(points, indices);
    REQUIRE(mesh != NoIndex);
    CHECK(world.AddMesh(points, indices) == NoIndex); // nothing left of any of the three pools
    REQUIRE(world.RemoveShape(mesh));

    const auto again = world.AddMesh(points, indices);
    REQUIRE(again != NoIndex);
    CHECK(again == mesh); // the same slot, since it was the one given back
    CHECK(world.Shapes[again].FirstVertex == 0);
    CHECK(world.Shapes[again].FirstTriangle == 0);
    CHECK(world.Shapes[again].RootNode == 0);
    // And the triangles were rebased onto the run it actually got, rather than onto the one it asked for.
    CHECK(world.Triangles[0].A < world.Shapes[again].VertexCount);
}

TEST_CASE("runs given back next to each other are one run again") {
    // Two hulls of eight vertices in a pool of sixteen, then a twelve vertex hull, which fits only if
    // the two runs merged: neither of them alone is long enough for it.
    const mtl::Context context;
    World world{context, {.ShapeVertices = 16}};
    std::vector<float3> cube;
    for (uint32_t corner = 0; corner < 8; ++corner)
        cube.push_back(float3{(corner & 1) ? 1.f : -1.f, (corner & 2) ? 1.f : -1.f, (corner & 4) ? 1.f : -1.f});
    std::vector<float3> prism; // a hexagonal prism, which is twelve corners and no interior points
    for (uint32_t i = 0; i < 6; ++i) {
        const float angle = float(i) * 2 * 3.14159265358979f / 6;
        for (const float y : {-1.f, 1.f}) prism.push_back(float3{std::cos(angle), y, std::sin(angle)});
    }

    const auto first = world.AddHull(cube), second = world.AddHull(cube);
    REQUIRE(first != NoIndex);
    REQUIRE(second != NoIndex);
    CHECK(world.AddHull(prism) == NoIndex);
    REQUIRE(world.RemoveShape(first)); // a hole in the middle
    REQUIRE(world.RemoveShape(second)); // and the tail, which absorbs the hole

    const auto big = world.AddHull(prism);
    REQUIRE(big != NoIndex);
    CHECK(world.Shapes[big].VertexCount == 12);
    CHECK(world.Shapes[big].FirstVertex == 0);
}
