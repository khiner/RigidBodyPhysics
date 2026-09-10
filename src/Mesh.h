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

} // namespace rbp
