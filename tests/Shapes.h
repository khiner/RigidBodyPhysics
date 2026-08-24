#pragma once

// The point clouds the hull tests and the solver tests both cook. They have to be the same clouds:
// HullTest says what the cook makes of one - how many faces, how wide - and SolveTest says what a body
// wearing it does, and the two are only about one shape while there is one definition of it.

#include "gpu/Shared.h"

#include <cmath>
#include <numbers>
#include <vector>

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
    const std::vector<float3> corners{float3{0, 1, phi}, float3{0, -1, phi}, float3{0, 1, -phi}, float3{0, -1, -phi},
                                      float3{1, phi, 0}, float3{-1, phi, 0}, float3{1, -phi, 0}, float3{-1, -phi, 0},
                                      float3{phi, 0, 1}, float3{-phi, 0, 1}, float3{phi, 0, -1}, float3{-phi, 0, -1}};
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
