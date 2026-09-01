#include "Mesh.h"

#include <algorithm>
#include <cmath>
#include <map>

namespace rbp {

namespace {
// Triangles per leaf.
// Small enough that a leaf is a few narrowphase calls rather than a scan, large enough that the tree is not mostly nodes.
constexpr uint32_t LeafSize = 4;

// How far from flat two faces must fold before the edge between them is a feature.
// Below this they are one surface to anything sliding over them, and marking the seam a feature would make a body catch on it.
// One degree, above the noise in a normal computed from three points and below any real crease.
constexpr float ActiveEdgeSine = 0.0175f;

float3 Normal(const std::vector<float3> &points, const Triangle &triangle) {
    const float3 turn = cross(points[triangle.B] - points[triangle.A], points[triangle.C] - points[triangle.A]);
    const float area = length(turn);
    return area > 0 ? turn / area : float3{0, 0, 0};
}

// Reorders the triangles into tree order and builds the tree.
// Split on the longest axis of the centroids at their median, which needs no cost model and is deterministic for a given input.
uint32_t Build(std::vector<Triangle> &triangles, const std::vector<float3> &points, std::vector<BvhNode> &nodes, uint32_t first, uint32_t count) {
    const uint32_t self = nodes.size();
    nodes.push_back({});
    float3 low{INFINITY, INFINITY, INFINITY}, high{-INFINITY, -INFINITY, -INFINITY};
    for (uint32_t i = first; i < first + count; ++i)
        for (const Index corner : {triangles[i].A, triangles[i].B, triangles[i].C}) {
            low = simd::min(low, points[corner]);
            high = simd::max(high, points[corner]);
        }
    nodes[self].Low = low;
    nodes[self].High = high;
    if (count <= LeafSize) {
        nodes[self].First = first;
        nodes[self].Count = count;
        return self;
    }

    const auto centre = [&points](const Triangle &triangle) {
        return (points[triangle.A] + points[triangle.B] + points[triangle.C]) / 3;
    };
    float3 spread_low{INFINITY, INFINITY, INFINITY}, spread_high{-INFINITY, -INFINITY, -INFINITY};
    for (uint32_t i = first; i < first + count; ++i) {
        spread_low = simd::min(spread_low, centre(triangles[i]));
        spread_high = simd::max(spread_high, centre(triangles[i]));
    }
    const float3 spread = spread_high - spread_low;
    const uint32_t axis = spread.x >= spread.y && spread.x >= spread.z ? 0 : (spread.y >= spread.z ? 1 : 2);
    const auto begin = triangles.begin() + first;
    const uint32_t half = count / 2;
    std::nth_element(begin, begin + half, begin + count, [&](const Triangle &a, const Triangle &b) {
        return centre(a)[axis] < centre(b)[axis];
    });

    // The left child is built first and lands immediately after this node, so only the right one needs an index of its own.
    Build(triangles, points, nodes, first, half);
    nodes[self].First = Build(triangles, points, nodes, first + half, count - half);
    nodes[self].Count = 0;
    return self;
}
} // namespace

CookedMesh CookMesh(std::span<const float3> points, std::span<const uint32_t> indices) {
    if (indices.size() < 3 || indices.size() % 3 != 0) return {};
    float3 low = points.empty() ? float3{0, 0, 0} : points[0], high = low;
    for (const float3 point : points) {
        low = simd::min(low, point);
        high = simd::max(high, point);
    }
    // Points within a millionth of the mesh's own size are the same corner.
    // Two points either side of a grid line at that scale stay apart, which costs one seam its recognition and nothing else.
    const float grain = 1e-6f * std::max({high.x - low.x, high.y - low.y, high.z - low.z, 1e-6f});

    CookedMesh cooked;
    std::map<std::tuple<int64_t, int64_t, int64_t>, uint32_t> welded;
    std::vector<uint32_t> where(points.size());
    for (uint32_t i = 0; i < points.size(); ++i) {
        const auto key = std::tuple{int64_t(std::llround(points[i].x / grain)), int64_t(std::llround(points[i].y / grain)), int64_t(std::llround(points[i].z / grain))};
        const auto [at, fresh] = welded.try_emplace(key, uint32_t(cooked.Vertices.size()));
        if (fresh) cooked.Vertices.push_back(points[i]);
        where[i] = at->second;
    }

    for (uint32_t i = 0; i + 2 < indices.size(); i += 3) {
        if (indices[i] >= points.size() || indices[i + 1] >= points.size() || indices[i + 2] >= points.size()) return {};
        const Triangle triangle{where[indices[i]], where[indices[i + 1]], where[indices[i + 2]], 0};
        if (triangle.A == triangle.B || triangle.B == triangle.C || triangle.A == triangle.C) continue; // welded flat
        if (length(cross(cooked.Vertices[triangle.B] - cooked.Vertices[triangle.A], cooked.Vertices[triangle.C] - cooked.Vertices[triangle.A])) <= 0) continue;
        cooked.Triangles.push_back(triangle);
    }
    if (cooked.Triangles.empty()) return {};

    // Which triangles meet along each edge, so each triangle can test whether the surface folds there.
    std::map<std::pair<Index, Index>, std::vector<uint32_t>> along;
    for (uint32_t t = 0; t < cooked.Triangles.size(); ++t) {
        const Triangle &triangle = cooked.Triangles[t];
        const Index corner[3]{triangle.A, triangle.B, triangle.C};
        for (uint32_t e = 0; e < 3; ++e) {
            const Index from = corner[e], to = corner[(e + 1) % 3];
            along[{std::min(from, to), std::max(from, to)}].push_back(t);
        }
    }
    for (uint32_t t = 0; t < cooked.Triangles.size(); ++t) {
        Triangle &triangle = cooked.Triangles[t];
        const Index corner[3]{triangle.A, triangle.B, triangle.C};
        const float3 normal = Normal(cooked.Vertices, triangle);
        for (uint32_t e = 0; e < 3; ++e) {
            const Index from = corner[e], to = corner[(e + 1) % 3];
            const auto &shared = along[{std::min(from, to), std::max(from, to)}];
            // An edge belonging to one triangle is the boundary of an open surface, and an edge three or more triangles meet along is not a surface.
            // Both are active by default, being reachable and covered by no other rule.
            bool active = shared.size() != 2;
            for (const uint32_t other : shared) {
                if (other == t) continue;
                const Triangle &neighbour = cooked.Triangles[other];
                // The neighbour's corner off the shared edge gives the fold direction.
                // Behind this triangle's plane is a ridge a body can hit, and level with it or in front is a seam or a valley out of reach.
                for (const Index far : {neighbour.A, neighbour.B, neighbour.C}) {
                    if (far == from || far == to) continue;
                    const float3 offset = cooked.Vertices[far] - cooked.Vertices[from];
                    const float span = length(offset);
                    if (span > 0 && dot(normal, offset) / span < -ActiveEdgeSine) active = true;
                }
            }
            if (active) triangle.ActiveEdges |= 1u << e;
            // And which triangle owns points lying along the edge: the lower-numbered of the two, so exactly one does.
            // A point at a corner several triangles meet at is then still owned.
            if (shared.size() != 2 || t == std::min(shared[0], shared[1])) triangle.OwnedEdges |= 1u << e;
        }
    }

    Build(cooked.Triangles, cooked.Vertices, cooked.Nodes, 0, cooked.Triangles.size());
    return cooked;
}

} // namespace rbp
