#pragma once

#include "gpu/Shared.h"

#include <span>
#include <vector>

// A triangle mesh, prepared for collision.
//
// Unlike a hull, nothing here is moved: a mesh is static geometry with no inside, so it has no volume,
// no centre of mass and no principal axes to be turned onto, and the frame it arrives in is the frame
// it keeps. What cooking does instead is answer the two questions the narrowphase cannot answer per
// triangle at runtime - which triangles are anywhere near a body, and which of their edges are real
// features rather than seams of the tessellation.
struct CookedMesh {
    std::vector<float3> Vertices; // welded, so triangles that share an edge share its two indices
    std::vector<Triangle> Triangles; // reordered so a leaf of the tree below owns a run of them
    std::vector<BvhNode> Nodes; // the root first, and every interior node's left child right after it
};

// Points and triples of indices into them. Degenerate triangles are dropped and coincident points are
// welded, which is what lets two triangles be seen to share an edge at all - a mesh authored for
// rendering usually has a vertex per corner per face, and untouched it would have no shared edges and
// so no seams to recognise. An empty result means there was no surface in it.
CookedMesh CookMesh(std::span<const float3> points, std::span<const uint32_t> indices);
