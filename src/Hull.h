#pragma once

#include "gpu/Shared.h"

#include <span>
#include <vector>

namespace rbp {

// A convex hull in its body frame: centred on the centre of mass, with the inertia tensor diagonal.
// The input points may be in any frame, so cooking integrates the mass properties and moves the vertices onto the centre of mass and principal axes.
// A body's pose is the pose of this frame rather than of the input points, and `Frame` carries the transform between the two.
struct CookedHull {
    std::vector<float3> Vertices; // empty for degenerate input: flat, collinear, or fewer than four points
    // Polygonal faces, merged back from the builder's triangles, with corners as indices into `Vertices`.
    std::vector<HullFace> Faces;
    float Volume{}; // mass is Volume times density
    float3 Inertia{}; // principal moments at unit density, linear in density
    // Simplification error from fitting `MaxHullVertices`: no corner of the exact hull lies more than this far outside a face of this one.
    // Zero means the two are the same solid.
    float Tolerance{};
    // The cooked frame in the input frame, so a cooked vertex maps back as `Frame.Position + Rotate(Frame.Orientation, vertex)`.
    Pose Frame{.Position = {0, 0, 0}, .Orientation = {0, 0, 0, 1}};
};

// The eigendecomposition of a symmetric 3x3: the diagonal values, and the rotation whose columns are the axes.
// The axes are made right handed, so the result is a rotation rather than a reflection.
struct Diagonalized {
    simd::double3 Values;
    simd::double3 Axis[3]; // the columns of the rotation
    float4 Orientation; // the same rotation as a quaternion
};
// Exposed for World::AddCompound, which runs over a compound's children the arithmetic the hull cook runs over tetrahedra.
// The body frame it produces means the same as CookedHull::Frame.
Diagonalized DiagonalizeSymmetric(const double (&symmetric)[3][3]);

// Returns an empty hull for degenerate input: fewer than four points, or points all on one plane or line.
// Input with more corners than `MaxHullVertices` is simplified rather than refused, and every field of the result then describes the simplified hull.
CookedHull CookHull(std::span<const float3> points);

} // namespace rbp
