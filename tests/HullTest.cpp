// Cooking a convex hull, against closed forms. A hull is the first shape given as raw geometry rather
// than as a definition, so what has to be checked is that the cook finds the same solid the definition
// would have: the same volume, the same principal moments, and the same answers from whatever frame
// the points arrive in.

#include "Hull.h"
#include "Shapes.h"
#include "World.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <vector>

#include <doctest/doctest.h>

namespace {
// A cube's corners with a few points inside it that the hull has no use for, so the cook has something
// to drop as well as something to keep.
std::vector<float3> CubePoints(float side, float3 at = {0, 0, 0}) {
    std::vector<float3> points = CubeCorners(side, at);
    points.push_back(at);
    points.push_back(at + float3{side / 4, 0, -side / 6});
    return points;
}

// A hull's faces have to be faces of the hull: every vertex on or behind each face's plane, every
// vertex on at least one of them, every face wound so its normal points out, and every face convex.
// The whole of what the merge has to preserve, and it needs no closed form to say it.
void CheckFaces(const CookedHull &cooked) {
    REQUIRE(!cooked.Faces.empty());
    float reach = 0;
    for (const float3 vertex : cooked.Vertices) reach = std::max(reach, simd::length(vertex));
    const float slack = 1e-4f * reach;

    std::vector<uint32_t> uses(cooked.Vertices.size(), 0);
    for (const HullFace &face : cooked.Faces) {
        CHECK(face.Count >= 3);
        CHECK(simd::length(face.Normal) == doctest::Approx(1).epsilon(1e-4));
        // A supporting plane: nothing of the solid is outside it.
        for (const float3 vertex : cooked.Vertices) CHECK(dot(face.Normal, vertex) <= face.Offset + slack);
        CHECK(face.Count <= MaxFacePoints);
        // Wound about its own normal, and convex, which is one test - every turn the same way round.
        for (uint32_t i = 0; i < face.Count; ++i) {
            const uint32_t index = face.Corner[i];
            REQUIRE(index < cooked.Vertices.size());
            ++uses[index];
            CHECK(dot(face.Normal, cooked.Vertices[index]) == doctest::Approx(face.Offset).epsilon(1e-3).scale(0));
            const float3 a = cooked.Vertices[face.Corner[i]], b = cooked.Vertices[face.Corner[(i + 1) % face.Count]],
                        c = cooked.Vertices[face.Corner[(i + 2) % face.Count]];
            CHECK(dot(cross(b - a, c - b), face.Normal) >= -slack * reach);
        }
        // Named by the lowest index it holds, so which triangle the builder started from cannot be read
        // off the face and a face keeps its name.
        for (uint32_t i = 1; i < face.Count; ++i) CHECK(face.Corner[0] < face.Corner[i]);
    }
    for (const uint32_t used : uses) CHECK(used >= 1); // a corner of the solid is a corner of some face
}
} // namespace

TEST_CASE("the cook gives back the faces the builder cut up") {
    // The builder triangulates, so a square face arrives as two triangles and a sixteen-gon as
    // fourteen. What a support query needs is the face, and the count is known by construction for
    // every shape here.
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
        // The ends have sixteen corners each and a face may name MaxFacePoints of them, so the cook
        // samples the rim - spread rather than truncated, which is the whole difference between an
        // octagon inscribed in the end and a contiguous wedge of it.
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
        // Every other corner of the rim, so what it keeps spans the whole end rather than half of it.
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
        // Forty-two vertices on a sphere, every face a triangle a few degrees from its neighbours.
        // Nothing here is coplanar and the count is Euler's: 2 * 42 - 4.
        const std::vector<float3> points = SpherePoints(1);
        REQUIRE(points.size() == 42);
        const CookedHull cooked = CookHull(points);
        CheckFaces(cooked);
        CHECK(cooked.Faces.size() == 80);
        for (const HullFace &face : cooked.Faces) CHECK(face.Count == 3);
    }
    SUBCASE("the same solid from two frames comes back the same solid") {
        // A cook's tolerances are relative to how big a hull is, and the precision of the points it is
        // handed to how far from the origin they were authored. Where the second is coarser the first
        // has to give way, or a face flat in one frame is two faces in another - a frame leaking into
        // the geometry, and what a modeller's export looks like.
        const std::vector<float3> wedge{float3{-0.4f, -0.15f, -0.3f}, float3{0.5f, -0.15f, -0.3f}, float3{0.5f, -0.15f, 0.35f},
                                        float3{-0.4f, -0.15f, 0.35f}, float3{-0.2f, 0.25f, -0.1f}, float3{0.3f, 0.25f, 0.2f}};
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
        // Six corners is eight triangles by Euler, of which the base's two are one rectangle: seven.
        CHECK(counts[0] == 7);
        CHECK(counts[1] == counts[0]);
    }

    SUBCASE("a facet a hundredth of a degree off a face is still its own face") {
        // What the merge must not do, and what a height tolerance in the support query cannot avoid
        // doing. The bottom is two facets meeting at a shallow angle and they have to stay two.
        for (const float degrees : {3.f, 0.3f, 0.03f, 0.01f}) {
            CAPTURE(degrees);
            const float rise = 0.1f * std::tan(degrees * std::numbers::pi_v<float> / 180);
            std::vector<float3> points;
            for (const float z : {-0.5f, 0.5f}) {
                points.push_back(float3{-0.5f, -0.1f, z});
                points.push_back(float3{0.4f, -0.1f, z});
                points.push_back(float3{0.5f, -0.1f + rise, z});
                points.push_back(float3{-0.5f, 0.1f, z});
                points.push_back(float3{0.5f, 0.1f, z});
            }
            const CookedHull cooked = CookHull(points);
            CheckFaces(cooked);
            uint32_t downward = 0;
            for (const HullFace &face : cooked.Faces) downward += face.Normal.y < -0.5f ? 1 : 0;
            CHECK(downward == 2); // the face and the facet, never one
        }
    }
}

TEST_CASE("a cube's hull is its corners, and its mass properties are the box's") {
    const CookedHull cooked = CookHull(CubePoints(2));

    CHECK(cooked.Vertices.size() == 8); // the two interior points are not corners of anything
    CHECK(cooked.Volume == doctest::Approx(8).epsilon(1e-5));
    // A cube of side s: I = m (s^2 + s^2) / 12, the same about all three axes.
    for (uint32_t axis = 0; axis < 3; ++axis) CHECK(cooked.Inertia[axis] == doctest::Approx(8 * 8.f / 12).epsilon(1e-5));

    // Its own frame, so the corners come back where a Box of the same size keeps them.
    for (const float3 vertex : cooked.Vertices)
        for (uint32_t axis = 0; axis < 3; ++axis) CHECK(std::abs(vertex[axis]) == doctest::Approx(1).epsilon(1e-5));
}

TEST_CASE("a hull is cooked into its own frame, whatever frame it arrives in") {
    // Off the origin and turned, which is where a modeller leaves a mesh and is not where a body's
    // pose can hold it: the engine's transforms are centred on the centre of mass and its inertia is
    // the diagonal in the body frame, so the cook has to establish both.
    std::vector<float3> points = CubePoints(2, float3{7, -3, 11});
    const float4 turn = QuatFromRotationVector(float3{0.3f, -0.7f, 0.2f});
    for (float3 &point : points) point = Rotate(turn, point);

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
    // Four alternating corners of a cube of side two, which is a regular tetrahedron of a third its
    // volume. Its second moment about the centroid is V/20 times the sum of the corners' outer
    // products, which here is four times the identity, so the covariance is 8/15 on each axis and the
    // moment about one is what the other two carry.
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

TEST_CASE("a hull too big for the pool is refused rather than truncated") {
    const mtl::Context context;
    World world{context};
    // A sphere's worth of points, all of them corners, which is more than one hull may hold.
    std::vector<float3> points;
    for (uint32_t i = 0; i < 200; ++i) {
        const float angle = 2.39996f * float(i), height = 1 - 2 * (float(i) + 0.5f) / 200;
        const float ring = std::sqrt(1 - height * height);
        points.push_back(float3{ring * std::cos(angle), height, ring * std::sin(angle)});
    }
    CHECK(world.AddHull(points) == NoIndex);
    CHECK(world.Overflow.ShapeVertices == 1);
    CHECK(world.ShapeCount() == 0);
}

TEST_CASE("a cooked hull says where its frame sits in the one its points arrived in") {
    // Cooking moves the points, and nothing but the cook can work out the transform between the two -
    // it falls out of an integration over the solid and a diagonalization of what that found. The
    // check is an identity: every cooked vertex, taken back through the frame, lands where it came from.
    std::vector<float3> points = CubePoints(2, float3{7, -3, 11});
    const float4 turn = QuatFromRotationVector(float3{0.3f, -0.7f, 0.2f});
    for (float3 &point : points) point = Rotate(turn, point);

    const CookedHull cooked = CookHull(points);
    REQUIRE(cooked.Vertices.size() == 8);
    for (const float3 vertex : cooked.Vertices) {
        const float3 back = cooked.Frame.Position + Rotate(cooked.Frame.Orientation, vertex);
        float nearest = INFINITY;
        for (const float3 point : points) nearest = std::min(nearest, float(simd::distance(back, point)));
        CHECK(nearest < 1e-4f);
    }
    // And the other way, which is the direction a caller uses it in: a corner of the geometry as given,
    // taken into the frame the body's pose is the pose of, is a corner of the cooked hull. Which corner
    // is not something to assert - the principal axes of a cube are any three orthogonal ones, so the
    // frame Jacobi lands on is its own business.
    for (uint32_t corner = 0; corner < 8; ++corner) {
        const float3 there = Rotate(QuatConjugate(cooked.Frame.Orientation), points[corner] - cooked.Frame.Position);
        float nearest = INFINITY;
        for (const float3 vertex : cooked.Vertices) nearest = std::min(nearest, float(simd::distance(there, vertex)));
        CHECK(nearest < 1e-4f);
    }
}
