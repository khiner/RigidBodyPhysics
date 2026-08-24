#pragma once

#include "gpu/Shared.h"

#include <span>
#include <vector>

// A convex hull, in the frame the engine's conventions require rather than the one it was handed.
//
// Every other shape is an analytic solid already centred and axis aligned by its own definition. A
// hull arrives in whatever frame the modeller left it in, so section 1.1's conventions - transforms
// centred on the centre of mass, inertia diagonal in the body frame - have to be established: the
// mass properties are integrated and the vertices moved onto the centre of mass and principal axes.
//
// So a body's pose is the pose of *this* frame, not of the points as given. The transform between the
// two comes back with the hull, since nothing else can work it out - it falls out of an integration
// over the solid and a diagonalization of what that integral found.
struct CookedHull {
    std::vector<float3> Vertices; // empty when the points were degenerate - flat, collinear, or fewer than four
    // The real faces, recovered from the triangles the builder cut them into, each naming its corners
    // by their index in `Vertices` - see HullFace in Shared.h for why the cook is where that happens.
    std::vector<HullFace> Faces;
    float Volume{}; // so mass is Volume times density
    float3 Inertia{}; // principal moments at unit density, which scale with it the same way
    // Where the cooked frame sits in the frame the points arrived in: its origin is the centre of mass
    // and its axes are the principal ones. So a point of the cooked hull is `Frame.Position +
    // Rotate(Frame.Orientation, vertex)` back in the caller's frame, and a body holding this shape puts
    // the geometry as given at its own pose composed with this.
    Pose Frame{.Position = {0, 0, 0}, .Orientation = {0, 0, 0, 1}};
};

// Fewer than four points, or points that all lie on one plane or line, make no solid and come back
// as an empty hull. A hull is a body's shape and a body needs a volume.
CookedHull CookHull(std::span<const float3> points);
