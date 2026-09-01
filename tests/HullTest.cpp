// Cooking a convex hull, checked against closed forms.
// The input is raw geometry in an arbitrary frame, so the cook derives both the solid and its body frame from the points alone.

#include "Hull.h"
#include "Shapes.h"
#include "World.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <span>
#include <vector>

#include <doctest/doctest.h>

using namespace rbp;

namespace {
// A cube's corners plus a few interior points, so the cook has points to drop as well as points to keep.
std::vector<float3> CubePoints(float side, float3 at = {0, 0, 0}) {
    std::vector<float3> points = CubeCorners(side, at);
    points.push_back(at);
    points.push_back(at + float3{side / 4, 0, -side / 6});
    return points;
}

// A cube of side two moved off the origin and rotated, in the kind of frame a modeller leaves a mesh in.
std::vector<float3> TurnedCube() {
    std::vector<float3> points = CubePoints(2, float3{7, -3, 11});
    const float4 turn = QuatFromRotationVector(float3{0.3f, -0.7f, 0.2f});
    for (float3 &point : points) point = Rotate(turn, point);
    return points;
}

// Checks the faces against the hull: every vertex on or behind each face plane, every vertex on at least one face, and every face convex and wound outward.
void CheckFaces(const CookedHull &cooked) {
    REQUIRE(!cooked.Faces.empty());
    float reach = 0;
    for (const float3 vertex : cooked.Vertices) reach = std::max(reach, simd::length(vertex));
    const float slack = 1e-4f * reach;

    std::vector<uint32_t> uses(cooked.Vertices.size(), 0);
    for (const HullFace &face : cooked.Faces) {
        CHECK(face.Count >= 3);
        CHECK(simd::length(face.Normal) == doctest::Approx(1).epsilon(1e-4));
        // A supporting plane, with the whole solid on or behind it.
        for (const float3 vertex : cooked.Vertices) CHECK(dot(face.Normal, vertex) <= face.Offset + slack);
        CHECK(face.Count <= MaxFacePoints);
        // Winding and convexity are one test: every turn goes the same way about the face normal.
        for (uint32_t i = 0; i < face.Count; ++i) {
            const uint32_t index = face.Corner[i];
            REQUIRE(index < cooked.Vertices.size());
            ++uses[index];
            CHECK(dot(face.Normal, cooked.Vertices[index]) == doctest::Approx(face.Offset).epsilon(1e-3).scale(0));
            const float3 a = cooked.Vertices[face.Corner[i]], b = cooked.Vertices[face.Corner[(i + 1) % face.Count]],
                         c = cooked.Vertices[face.Corner[(i + 2) % face.Count]];
            CHECK(dot(cross(b - a, c - b), face.Normal) >= -slack * reach);
        }
        // A face starts at its lowest corner index, so its identity is independent of which triangle it began as.
        for (uint32_t i = 1; i < face.Count; ++i) CHECK(face.Corner[0] < face.Corner[i]);
    }
    for (const uint32_t used : uses) CHECK(used >= 1); // a corner of the solid is a corner of some face
}

// The greatest distance any source point lies outside a face plane, which the cook's tolerance bounds.
float OutsideBy(const CookedHull &cooked, std::span<const float3> points) {
    float deepest = 0;
    for (const float3 point : points) {
        const float3 there = LocalPoint(cooked.Frame, point);
        for (const HullFace &face : cooked.Faces) deepest = std::max(deepest, float(dot(face.Normal, there)) - face.Offset);
    }
    return deepest;
}
} // namespace

TEST_CASE("the cook gives back the faces the builder cut up") {
    // The builder triangulates, and the merge restores the original faces for the support query.
    // Every shape here has a face count known by construction.
    SUBCASE("a cube is six squares") {
        const CookedHull cooked = CookHull(CubePoints(2));
        CheckFaces(cooked);
        CHECK(cooked.Faces.size() == 6);
        for (const HullFace &face : cooked.Faces) CHECK(face.Count == 4);
    }
    SUBCASE("a tetrahedron is four triangles, and merges nothing") {
        const CookedHull cooked = CookHull({{float3{0, 0, 0}, float3{1, 0, 0}, float3{0, 1, 0}, float3{0, 0, 1}}});
        CheckFaces(cooked);
        CHECK(cooked.Faces.size() == 4);
        for (const HullFace &face : cooked.Faces) CHECK(face.Count == 3);
    }
    SUBCASE("a sixteen sided prism is two ends and sixteen walls, its ends sampled to eight") {
        // An end has sixteen corners against a limit of MaxFacePoints, so the cook samples the rim evenly into an inscribed octagon.
        const CookedHull cooked = CookHull(PrismPoints(16, 0.5f, 0.2f));
        CheckFaces(cooked);
        CHECK(cooked.Faces.size() == 18);
        uint32_t ends = 0, walls = 0;
        for (const HullFace &face : cooked.Faces) {
            ends += face.Count == MaxFacePoints ? 1 : 0;
            walls += face.Count == 4 ? 1 : 0;
        }
        CHECK(ends == 2);
        CHECK(walls == 16);
        // The octagon takes every other corner of the rim, so its centre stays on the axis and it spans the whole end.
        for (const HullFace &face : cooked.Faces) {
            if (face.Count != MaxFacePoints) continue;
            float3 centre{0, 0, 0};
            for (uint32_t i = 0; i < face.Count; ++i) centre += cooked.Vertices[face.Corner[i]];
            centre /= float(face.Count);
            CHECK(std::abs(float(centre.x)) < 1e-3f);
            CHECK(std::abs(float(centre.z)) < 1e-3f);
        }
    }
    SUBCASE("a sphere-like hull merges nothing at all") {
        // No two facets are coplanar, so nothing merges and the face count is Euler's 2 * 42 - 4.
        const std::vector<float3> points = SpherePoints(1);
        REQUIRE(points.size() == 42);
        const CookedHull cooked = CookHull(points);
        CheckFaces(cooked);
        CHECK(cooked.Faces.size() == 80);
        for (const HullFace &face : cooked.Faces) CHECK(face.Count == 3);
    }
    SUBCASE("the same solid from two frames comes back the same solid") {
        // Tolerances scale with the hull, and point precision degrades with distance from the origin.
        // Where the precision is the coarser of the two, the tolerance yields to it.
        const std::vector<float3> wedge = WedgePoints();
        std::vector<uint32_t> counts;
        for (const bool turn : {false, true}) {
            CAPTURE(turn);
            const float4 spin = turn ? QuatFromRotationVector(float3{0.8f, -1.3f, 0.55f}) : float4{0, 0, 0, 1};
            const float3 move = turn ? float3{31, -17, 8} : float3{0, 0, 0};
            std::vector<float3> points;
            for (const float3 point : wedge) points.push_back(move + Rotate(spin, point));
            const CookedHull cooked = CookHull(points);
            CheckFaces(cooked);
            CHECK(cooked.Vertices.size() == wedge.size());
            counts.push_back(cooked.Faces.size());
        }
        // Six corners is eight triangles by Euler, and the base's two merge into one, giving seven.
        CHECK(counts[0] == 7);
        CHECK(counts[1] == counts[0]);
    }

    SUBCASE("a facet a hundredth of a degree off a face is still its own face") {
        // A guard on the merge: the bottom's two facets stay two however shallow the angle between them.
        for (const float degrees : {3.f, 0.3f, 0.03f, 0.01f}) {
            CAPTURE(degrees);
            const CookedHull cooked = CookHull(ChamferedPlate(0.1f * std::tan(degrees * std::numbers::pi_v<float> / 180)));
            CheckFaces(cooked);
            uint32_t downward = 0;
            for (const HullFace &face : cooked.Faces) downward += face.Normal.y < -0.5f ? 1 : 0;
            CHECK(downward == 2); // the face and the facet stay separate
        }
    }
}

TEST_CASE("a cube's hull is its corners, and its mass properties are the box's") {
    const CookedHull cooked = CookHull(CubePoints(2));

    CHECK(cooked.Vertices.size() == 8); // the two interior points are dropped
    CHECK(cooked.Volume == doctest::Approx(8).epsilon(1e-5));
    // A cube of side s: I = m (s^2 + s^2) / 12, the same about all three axes.
    for (uint32_t axis = 0; axis < 3; ++axis) CHECK(cooked.Inertia[axis] == doctest::Approx(8 * 8.f / 12).epsilon(1e-5));

    // Cooked into its own frame, so the corners land where a Box of the same size has them.
    for (const float3 vertex : cooked.Vertices)
        for (uint32_t axis = 0; axis < 3; ++axis) CHECK(std::abs(vertex[axis]) == doctest::Approx(1).epsilon(1e-5));
}

TEST_CASE("a hull is cooked into its own frame, whatever frame it arrives in") {
    // The engine's body frame is centred on the centre of mass with the inertia diagonal, so the cook establishes both from any input frame.
    const std::vector<float3> points = TurnedCube();
    const CookedHull cooked = CookHull(points);
    CHECK(cooked.Vertices.size() == 8);
    CHECK(cooked.Volume == doctest::Approx(8).epsilon(1e-4));
    for (uint32_t axis = 0; axis < 3; ++axis) CHECK(cooked.Inertia[axis] == doctest::Approx(8 * 8.f / 12).epsilon(1e-4));

    float3 center{0, 0, 0};
    for (const float3 vertex : cooked.Vertices) {
        center += vertex / 8;
        CHECK(length(vertex) == doctest::Approx(std::sqrt(3.f)).epsilon(1e-4)); // still a cube of side 2
    }
    CHECK(length(center) < 1e-4f);
}

TEST_CASE("a tetrahedron's hull matches the closed form") {
    // Four alternating corners of a cube of side two: a regular tetrahedron of a third its volume.
    // Its second moment about the centroid is V/20 times the sum of the corners' outer products, four times the identity here, giving 8/15 on each axis.
    const std::vector<float3> points{float3{1, 1, 1}, float3{1, -1, -1}, float3{-1, 1, -1}, float3{-1, -1, 1}};
    const CookedHull cooked = CookHull(points);

    CHECK(cooked.Vertices.size() == 4);
    CHECK(cooked.Volume == doctest::Approx(8 / 3.f).epsilon(1e-5));
    for (uint32_t axis = 0; axis < 3; ++axis) CHECK(cooked.Inertia[axis] == doctest::Approx(16 / 15.f).epsilon(1e-4));
}

TEST_CASE("points that make no solid make no hull") {
    CHECK(CookHull(std::vector<float3>{float3{0, 0, 0}, float3{1, 0, 0}, float3{0, 1, 0}}).Vertices.empty()); // too few
    CHECK(CookHull(std::vector<float3>{float3{0, 0, 0}, float3{1, 0, 0}, float3{0, 1, 0}, float3{1, 1, 0}, float3{2, 3, 0}}).Vertices.empty()); // flat
    CHECK(CookHull(std::vector<float3>{float3{0, 0, 0}, float3{1, 1, 1}, float3{2, 2, 2}, float3{3, 3, 3}}).Vertices.empty()); // a line
}

TEST_CASE("a cube added as a hull weighs what the same cube added as a box weighs") {
    const mtl::Context context;
    World world{context};
    const auto hull = world.AddHull(CubePoints(1));
    const auto box = world.AddShape({.HalfExtents = {0.5f, 0.5f, 0.5f}, .Kind = ShapeBox});
    REQUIRE(hull != NoIndex);

    const auto from_hull = MassProperties(world.Shapes[hull], 1000, world.ShapeVertices.All());
    const auto from_box = MassProperties(world.Shapes[box], 1000);
    CHECK(from_hull.InvMass == doctest::Approx(from_box.InvMass).epsilon(1e-5));
    for (uint32_t axis = 0; axis < 3; ++axis)
        CHECK(from_hull.InvInertiaLocal[axis] == doctest::Approx(from_box.InvInertiaLocal[axis]).epsilon(1e-5));
}

TEST_CASE("a hull of more points than a shape may name is simplified rather than refused") {
    // A caller's points are often a mesh's vertices, hundreds of them and all corners, and refusing those leaves the body without a shape.
    // The cook drops detail finer than a tolerance and returns that tolerance as the caller's error bound.
    constexpr float Radius = 1;
    const std::vector<float3> points = DenseSpherePoints(500, Radius);
    const CookedHull cooked = CookHull(points);
    CheckFaces(cooked);
    CHECK(cooked.Vertices.size() <= MaxHullVertices);
    CHECK(cooked.Tolerance > 0);

    // Every corner kept is one of the input points, so the simplified solid is contained in the exact one.
    for (const float3 vertex : cooked.Vertices)
        CHECK(NearestTo(WorldPoint(cooked.Frame, vertex), points) < 1e-4f);
    // The tolerance is measured from the result, so the worst point lies exactly that far outside.
    CHECK(OutsideBy(cooked, points) == doctest::Approx(cooked.Tolerance).epsilon(1e-3));

    // Bounds rather than an exact closed form.
    // Every corner lies on the sphere and every face plane is the tolerance inside it.
    // The solid contains the ball of radius R - t and lies inside the ball of radius R.
    // Volume and inertia are monotone in the radius.
    const auto ball = [](float radius) { return 4.f / 3 * std::numbers::pi_v<float> * radius * radius * radius; };
    CHECK(cooked.Volume <= ball(Radius));
    CHECK(cooked.Volume >= ball(Radius - cooked.Tolerance));
    for (uint32_t axis = 0; axis < 3; ++axis) {
        CHECK(cooked.Inertia[axis] <= 2.f / 5 * ball(Radius) * Radius * Radius);
        CHECK(cooked.Inertia[axis] >= 2.f / 5 * ball(Radius - cooked.Tolerance) * (Radius - cooked.Tolerance) * (Radius - cooked.Tolerance));
    }
    // Those bounds assume the two balls are concentric, so the centre is checked too.
    // A ball has no distinguished principal axis, so the three moments are equal.
    CHECK(simd::length(cooked.Frame.Position) < cooked.Tolerance);
    for (uint32_t axis = 0; axis < 3; ++axis) CHECK(cooked.Inertia[axis] == doctest::Approx(cooked.Inertia[0]).epsilon(0.05));
}

TEST_CASE("a hull that already fits is cooked exactly, and says so") {
    // Simplification runs only above MaxHullVertices, so a smaller shape is cooked exactly.
    SUBCASE("a cube, which is eight corners and two points inside it") {
        const CookedHull cooked = CookHull(CubePoints(2));
        CHECK(cooked.Vertices.size() == 8);
        CHECK(cooked.Tolerance == 0);
    }
    SUBCASE("a thirty-two sided prism, which is exactly MaxHullVertices") {
        // The size a cylinder typically arrives at, one corner below the simplification threshold.
        const CookedHull cooked = CookHull(PrismPoints(32, 0.5f, 0.2f));
        CheckFaces(cooked);
        CHECK(cooked.Vertices.size() == MaxHullVertices);
        CHECK(cooked.Tolerance == 0);
        CHECK(cooked.Faces.size() == 34);
    }
}

TEST_CASE("a hull with no room left in the pool is refused rather than truncated") {
    const mtl::Context context;
    // The cook fits every hull to MaxHullVertices, so an exhausted pool is the one remaining refusal.
    World world{context, {.ShapeVertices = 4}};
    CHECK(world.AddHull(CubePoints(2)) == NoIndex);
    CHECK(world.Overflow.ShapeVertices == 1);
    CHECK(world.ShapeCount() == 0);
}

TEST_CASE("a mesh's worth of points added as a hull makes a shape a body can wear") {
    const mtl::Context context;
    World world{context};
    const auto shape = world.AddHull(DenseSpherePoints(500, 0.5f));
    REQUIRE(shape != NoIndex);
    CHECK(world.Shapes[shape].VertexCount <= MaxHullVertices);
    CHECK(world.Shapes[shape].FaceCount >= 1);
    CHECK(world.Overflow.ShapeVertices == 0);
    CHECK(world.Overflow.HullFaces == 0);
}

TEST_CASE("a cooked hull says where its frame sits in the one its points arrived in") {
    // Only the cook computes the transform between the two frames.
    // Every cooked vertex taken back through the frame lands on its source point.
    const std::vector<float3> points = TurnedCube();
    const CookedHull cooked = CookHull(points);
    REQUIRE(cooked.Vertices.size() == 8);
    for (const float3 vertex : cooked.Vertices) {
        const float3 back = WorldPoint(cooked.Frame, vertex);
        CHECK(NearestTo(back, points) < 1e-4f);
    }
    // The reverse direction, the one a caller uses.
    // The corner-to-corner mapping is left unasserted, since a cube's principal axes are any three orthogonal ones.
    for (uint32_t corner = 0; corner < 8; ++corner) {
        const float3 there = LocalPoint(cooked.Frame, points[corner]);
        CHECK(NearestTo(there, cooked.Vertices) < 1e-4f);
    }
}
