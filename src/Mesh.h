#pragma once

#include "gpu/Shared.h"

#include <span>
#include <vector>

namespace rbp {

// A triangle mesh, prepared for collision.
//
// The cook does not move the vertices, unlike the hull cook: a mesh has no interior, so no volume, centre of mass or principal axes to move onto.
// The input frame is the cooked frame, and a host that gives a moving mesh a mass gives it in that frame.
// The cook precomputes the two things the narrowphase cannot derive per triangle at runtime: which triangles are near a body, and which edges are seams.
struct CookedMesh {
    std::vector<float3> Vertices; // welded, so triangles that share an edge share its two indices
    std::vector<Triangle> Triangles; // reordered so each leaf of the tree below covers a contiguous run
    std::vector<BvhNode> Nodes; // the root first, and every interior node's left child right after it
};

// Takes points and triples of indices into them.
// Degenerate triangles are dropped and coincident points welded, so two triangles across one edge share its two indices.
// A mesh authored for rendering has one vertex per corner per face, and without welding it has no shared edges to classify.
// The result is empty when the input holds no surface.
CookedMesh CookMesh(std::span<const float3> points, std::span<const uint32_t> indices);

} // namespace rbp
