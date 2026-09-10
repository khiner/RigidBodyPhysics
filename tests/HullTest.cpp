
#include "Hull.h"
#include "Shapes.h"
#include "World.h"
#include "fixtures/WaterWheelSign.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <span>
#include <vector>

#include <doctest/doctest.h>

using namespace rbp;

namespace {

std::vector<float3> CubePoints(float side, float3 at = {0, 0, 0}) {
    std::vector<float3> points = CubeCorners(side, at);
    points.push_back(at);
    points.push_back(at + float3{side / 4, 0, -side / 6});
    return points;
}

std::vector<float3> TurnedCube() {
    std::vector<float3> points = CubePoints(2, float3{7, -3, 11});
    const float4 turn = QuatFromRotationVector(float3{0.3f, -0.7f, 0.2f});
    for (float3 &point : points) point = Rotate(turn, point);
    return points;
}

void CheckFaces(const CookedHull &cooked) {
    REQUIRE(!cooked.Faces.empty());
    float reach = 0;
    for (const float3 vertex : cooked.Vertices) reach = std::max(reach, simd::length(vertex));
    const float slack = 1e-4f * reach;

    std::vector<uint32_t> uses(cooked.Vertices.size(), 0);
    for (const HullFace &face : cooked.Faces) {
        CHECK(face.Count >= 3);
        CHECK(simd::length(face.Normal) == doctest::Approx(1).epsilon(1e-4));

        for (const float3 vertex : cooked.Vertices) CHECK(dot(face.Normal, vertex) <= face.Offset + slack);
        CHECK(face.Count <= MaxFacePoints);

        for (uint32_t i = 0; i < face.Count; ++i) {
            const uint32_t index = face.Corner[i];
            REQUIRE(index < cooked.Vertices.size());
            ++uses[index];
            CHECK(dot(face.Normal, cooked.Vertices[index]) == doctest::Approx(face.Offset).epsilon(1e-3).scale(0));
            const float3 a = cooked.Vertices[face.Corner[i]], b = cooked.Vertices[face.Corner[(i + 1) % face.Count]],
                         c = cooked.Vertices[face.Corner[(i + 2) % face.Count]];
            CHECK(dot(cross(b - a, c - b), face.Normal) >= -slack * reach);
        }
        for (uint32_t i = 1; i < face.Count; ++i) CHECK(face.Corner[0] < face.Corner[i]);
    }
    for (const uint32_t used : uses) CHECK(used >= 1);
}

// Return the maximum distance of a source point outside the hull.
float OutsideBy(const CookedHull &cooked, std::span<const float3> points) {
    float deepest = 0;
    for (const float3 point : points) {
        const float3 there = LocalPoint(cooked.Frame, point);
        for (const HullFace &face : cooked.Faces) deepest = std::max(deepest, float(dot(face.Normal, there)) - face.Offset);
    }
    return deepest;
}
} // namespace

TEST_CASE("cooking: hull topology mass and frames match analytic solids") {
    SUBCASE("a cube's hull is its corners, and its mass properties are the box's") {
        const CookedHull cooked = CookHull(CubePoints(2));
        CheckFaces(cooked);

        CHECK(cooked.Vertices.size() == 8);
        CHECK(cooked.Volume == doctest::Approx(8).epsilon(1e-5));

        for (uint32_t axis = 0; axis < 3; ++axis) CHECK(cooked.Inertia[axis] == doctest::Approx(8 * 8.f / 12).epsilon(1e-5));

        for (const float3 vertex : cooked.Vertices)
            for (uint32_t axis = 0; axis < 3; ++axis) CHECK(std::abs(vertex[axis]) == doctest::Approx(1).epsilon(1e-5));
    }
    SUBCASE("a hull is cooked into its own frame, whatever frame it arrives in") {
        const std::vector<float3> points = TurnedCube();
        const CookedHull cooked = CookHull(points);
        CHECK(cooked.Vertices.size() == 8);
        CHECK(cooked.Volume == doctest::Approx(8).epsilon(1e-4));
        for (uint32_t axis = 0; axis < 3; ++axis) CHECK(cooked.Inertia[axis] == doctest::Approx(8 * 8.f / 12).epsilon(1e-4));

        float3 center{0, 0, 0};
        for (const float3 vertex : cooked.Vertices) {
            center += vertex / 8;
            CHECK(length(vertex) == doctest::Approx(std::sqrt(3.f)).epsilon(1e-4));
        }
        CHECK(length(center) < 1e-4f);
    }
    SUBCASE("a tetrahedron's hull matches the closed form") {
        const std::vector<float3> points{float3{1, 1, 1}, float3{1, -1, -1}, float3{-1, 1, -1}, float3{-1, -1, 1}};
        const CookedHull cooked = CookHull(points);

        CHECK(cooked.Vertices.size() == 4);
        CHECK(cooked.Volume == doctest::Approx(8 / 3.f).epsilon(1e-5));
        for (uint32_t axis = 0; axis < 3; ++axis) CHECK(cooked.Inertia[axis] == doctest::Approx(16 / 15.f).epsilon(1e-4));
    }
}

TEST_CASE("cooking: hull simplification bounds error and respects capacity") {
    SUBCASE("points that make no solid make no hull") {
        CHECK(CookHull(std::vector<float3>{float3{0, 0, 0}, float3{1, 0, 0}, float3{0, 1, 0}}).Vertices.empty());
        CHECK(CookHull(std::vector<float3>{float3{0, 0, 0}, float3{1, 0, 0}, float3{0, 1, 0}, float3{1, 1, 0}, float3{2, 3, 0}}).Vertices.empty());
        CHECK(CookHull(std::vector<float3>{float3{0, 0, 0}, float3{1, 1, 1}, float3{2, 2, 2}, float3{3, 3, 3}}).Vertices.empty());
    }
    SUBCASE("a hull of more points than a shape may name is simplified rather than refused") {
        // The reported tolerance bounds simplification error.
        constexpr float Radius = 1;
        const std::vector<float3> points = DenseSpherePoints(500, Radius);
        const CookedHull cooked = CookHull(points);
        CheckFaces(cooked);
        CHECK(cooked.Vertices.size() <= MaxHullVertices);
        CHECK(cooked.Tolerance > 0);

        for (const float3 vertex : cooked.Vertices)
            CHECK(NearestTo(WorldPoint(cooked.Frame, vertex), points) < 1e-4f);
        CHECK(OutsideBy(cooked, points) == doctest::Approx(cooked.Tolerance).epsilon(1e-3));

        // The hull contains a sphere of radius R - t and fits inside a sphere of radius R.
        // These radii bound its volume and inertia.
        const auto ball = [](float radius) { return 4.f / 3 * std::numbers::pi_v<float> * radius * radius * radius; };
        CHECK(cooked.Volume <= ball(Radius));
        CHECK(cooked.Volume >= ball(Radius - cooked.Tolerance));
        for (uint32_t axis = 0; axis < 3; ++axis) {
            CHECK(cooked.Inertia[axis] <= 2.f / 5 * ball(Radius) * Radius * Radius);
            CHECK(cooked.Inertia[axis] >= 2.f / 5 * ball(Radius - cooked.Tolerance) * (Radius - cooked.Tolerance) * (Radius - cooked.Tolerance));
        }

        CHECK(simd::length(cooked.Frame.Position) < cooked.Tolerance);
        for (uint32_t axis = 0; axis < 3; ++axis) CHECK(cooked.Inertia[axis] == doctest::Approx(cooked.Inertia[0]).epsilon(0.05));
    }
}

TEST_CASE("cooking: thin hulls retain unique supporting corners") {
    SUBCASE("simplification inserts each nearly collinear input corner once") {
        const auto cooked = CookHull(WaterWheelSignPoints);
        REQUIRE(!cooked.Vertices.empty());
        CHECK(cooked.Vertices.size() <= MaxHullVertices);
        CHECK(std::isfinite(cooked.Volume));
        CHECK(cooked.Volume > 0);
        for (int axis = 0; axis < 3; ++axis) {
            CHECK(std::isfinite(cooked.Inertia[axis]));
            CHECK(cooked.Inertia[axis] > 0);
        }
        CheckFaces(cooked);
        for (auto point : WaterWheelSignPoints) {
            const auto local = LocalPoint(cooked.Frame, point);
            for (const auto &face : cooked.Faces) CHECK(dot(face.Normal, local) <= face.Offset + cooked.Tolerance + 2e-6f);
        }
    }
    SUBCASE("hull insertion removes shallow visible faces around a thin rim") {
        const float3 points[]{
            {-0.025003517f, 0.0235066973f, 0.000853277103f},
            {-0.0229702462f, -0.0371100865f, 0.000977357733f},
            {-0.0248761028f, -0.0276519116f, 0.000958686462f},
            {0.0227206163f, -0.0369963348f, 0.00095721829f},
            {0.0245793276f, -0.0275287852f, 0.000936870987f},
            {-0.00139440177f, 0.0291323662f, 0.00169858534f},
        };

        constexpr double volume = 4.958245770658764e-7;
        auto order = std::to_array<uint32_t>({0, 1, 2, 3, 4, 5});
        do {
            std::array<float3, std::size(points)> input;
            for (uint32_t i = 0; i < input.size(); ++i) input[i] = points[order[i]];
            const auto cooked = CookHull(input);
            REQUIRE(!cooked.Vertices.empty());
            CHECK(cooked.Faces.size() <= 2 * cooked.Vertices.size() - 4);
            CHECK(cooked.Volume == doctest::Approx(volume).epsilon(1e-5).scale(0));
            CheckFaces(cooked);
            CHECK(OutsideBy(cooked, points) < 1e-6f);
        } while (std::next_permutation(order.begin(), order.end()));
    }
}
