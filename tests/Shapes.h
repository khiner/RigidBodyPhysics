#pragma once

// The shapes the tests share. They have to be the same shapes: HullTest says what the cook makes of a
// point cloud - how many faces, how wide - and SolveTest says what a body wearing it does, and the two
// are only about one shape while there is one definition of it.

#include "gpu/Shared.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <span>
#include <vector>

constexpr float Half = 0.5f;
constexpr Shape UnitBox{.HalfExtents = {Half, Half, Half}, .Kind = ShapeBox};
constexpr Shape GroundPlane{.Normal = {0, 1, 0}, .Offset = 0, .Kind = ShapePlane};

// The eight corners of a cube of side `side` centred on `at`, in no particular order, which is what
// the cook is for.
inline std::vector<float3> CubeCorners(float side, float3 at = {0, 0, 0}) {
    const float half = side / 2;
    std::vector<float3> points;
    for (uint32_t corner = 0; corner < 8; ++corner)
        points.push_back(at + half * float3{(corner & 1) ? 1.f : -1.f, (corner & 2) ? 1.f : -1.f, (corner & 4) ? 1.f : -1.f});
    return points;
}

// A coin: `sides` around, flat top and bottom. Each flat face carries `sides` vertices, which past
// eight is more than a recovered face may hold, and its walls are quads - so its face list is known by
// construction.
inline std::vector<float3> PrismPoints(uint32_t sides, float radius, float half_height) {
    std::vector<float3> points;
    for (uint32_t i = 0; i < sides; ++i) {
        const float angle = 2 * std::numbers::pi_v<float> * float(i) / float(sides);
        for (const float y : {-half_height, half_height})
            points.push_back(float3{radius * std::cos(angle), y, radius * std::sin(angle)});
    }
    return points;
}

// The unit icosahedron's twelve vertices with every edge's midpoint pushed out onto the sphere, which
// is forty-two points and eighty faces. Two subdivisions would be a hundred and sixty-two, so this is
// the finest sphere MaxHullVertices holds - and every original face has become four whose normals are
// a few degrees apart, which is what makes it the case to ask the face tolerance about.
inline std::vector<float3> SpherePoints(float radius) {
    const float phi = (1 + std::sqrt(5.f)) / 2;
    const std::vector<float3> corners{float3{0, 1, phi}, float3{0, -1, phi}, float3{0, 1, -phi}, float3{0, -1, -phi}, float3{1, phi, 0}, float3{-1, phi, 0}, float3{1, -phi, 0}, float3{-1, -phi, 0}, float3{phi, 0, 1}, float3{-phi, 0, 1}, float3{phi, 0, -1}, float3{-phi, 0, -1}};
    std::vector<float3> points;
    const auto add = [&points, radius](float3 point) {
        const float3 on_sphere = radius * simd::normalize(point);
        for (const float3 already : points)
            if (simd::distance(already, on_sphere) < 1e-4f) return;
        points.push_back(on_sphere);
    };
    for (const float3 corner : corners) add(corner);
    // Every pair two units apart is an edge of the icosahedron, and its midpoint is a new vertex.
    for (uint32_t i = 0; i < corners.size(); ++i)
        for (uint32_t j = i + 1; j < corners.size(); ++j)
            if (simd::distance(corners[i], corners[j]) < 2.01f) add(corners[i] + corners[j]);
    return points;
}

// A wedge: no two faces the same and no symmetry to hide a wrong axis behind, which is what makes it
// the shape to hand the cook from two frames and the shape to drop on a box.
inline std::vector<float3> WedgePoints() {
    return {float3{-0.4f, -0.15f, -0.3f}, float3{0.5f, -0.15f, -0.3f}, float3{0.5f, -0.15f, 0.35f}, float3{-0.4f, -0.15f, 0.35f}, float3{-0.2f, 0.25f, -0.1f}, float3{0.3f, 0.25f, 0.2f}};
}

// A plate a metre across and two tenths thick whose bottom is a face and, beside it, a chamfer a tenth
// of a metre wide rising by `rise`. At the angles these tests ask about, the facet is inside any
// tolerance a face could be recovered by height with - which is the whole question, in the cook and
// again in what the plate rests on.
inline std::vector<float3> ChamferedPlate(float rise) {
    std::vector<float3> points;
    for (const float z : {-Half, Half}) {
        points.push_back(float3{-Half, -0.1f, z});
        points.push_back(float3{Half - 0.1f, -0.1f, z});
        points.push_back(float3{Half, -0.1f + rise, z});
        points.push_back(float3{-Half, 0.1f, z});
        points.push_back(float3{Half, 0.1f, z});
    }
    return points;
}

// How far `point` is from the nearest of `among` - which is how a set of points is checked to be the
// same set when nothing says which of them is which.
inline float NearestTo(float3 point, std::span<const float3> among) {
    float nearest = INFINITY;
    for (const float3 other : among) nearest = std::min(nearest, float(simd::distance(point, other)));
    return nearest;
}
