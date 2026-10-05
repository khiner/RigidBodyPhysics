#pragma once

#include "gpu/Shared.h"

#include <span>
#include <vector>

namespace rbp {

// Mesh coordinates retain the input frame.

struct CookedMesh {
    std::vector<float3> Vertices; // Adjacent triangles share vertex indices.
    std::vector<Triangle> Triangles; // Each leaf covers a contiguous triangle range.
    std::vector<BvhNode> Nodes; // The root is first, and each left child immediately follows its parent.
};

// Weld coincident vertices and remove degenerate triangles and unused vertices.
// Return an empty mesh when the input contains no valid triangles.
CookedMesh CookMesh(std::span<const float3> points, std::span<const uint32_t> indices);

// Refit cooked-local geometry without welding, reordering or changing identity.
// Preserves triangle indices/OwnedEdges and node First/Count; refreshes active
// front/back edge flags and bounds. Rejects empty, nonfinite, degenerate or
// malformed geometry/topology without changing either output span.
bool RefitMesh(std::span<const float3> points, std::span<Triangle> triangles, std::span<BvhNode> nodes);

} // namespace rbp
