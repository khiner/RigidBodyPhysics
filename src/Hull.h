#pragma once

#include "gpu/Shared.h"

#include <span>
#include <vector>

namespace rbp {

// Vertices use the center-of-mass frame with diagonal inertia.
struct CookedHull {
    std::vector<float3> Vertices;
    // Face corners index Vertices.
    std::vector<HullFace> Faces;
    float Volume{}; // Mass equals Volume times density.
    float3 Inertia{}; // Principal moments are measured at unit density.
    // Tolerance bounds the distance of any original hull corner outside the simplified hull.
    float Tolerance{};
    // Frame transforms cooked coordinates into input coordinates.
    Pose Frame{.Position = {0, 0, 0}, .Orientation = {0, 0, 0, 1}};
};

// The eigenvectors form a right-handed orthonormal frame.
struct Diagonalized {
    simd::double3 Values;
    simd::double3 Axis[3]; // Axis contains the rotation columns.
    float4 Orientation;
};
// Return eigenvalues and a principal-axis rotation for a symmetric matrix.
Diagonalized DiagonalizeSymmetric(const double (&symmetric)[3][3]);

// Return a hull with at most MaxHullVertices, or an empty hull for coplanar, collinear or insufficient input.
CookedHull CookHull(std::span<const float3> points);

} // namespace rbp
