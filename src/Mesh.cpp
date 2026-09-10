#include "Mesh.h"

#include <algorithm>
#include <cmath>
#include <map>

namespace rbp {

namespace {
constexpr uint32_t LeafSize = 4;

// Treat folds below one degree as coplanar to suppress contacts on tessellation edges.
constexpr float ActiveEdgeSine = 0.0175f;

float3 Normal(const std::vector<float3> &points, const Triangle &triangle) {
    const float3 turn = cross(points[triangle.B] - points[triangle.A], points[triangle.C] - points[triangle.A]);
    const float area = length(turn);
    return area > 0 ? turn / area : float3{0, 0, 0};
}

uint32_t Build(std::vector<Triangle> &triangles, const std::vector<float3> &points, std::vector<BvhNode> &nodes, uint32_t first, uint32_t count, uint32_t depth = 0) {
    // Minimize surface area times triangle count over binned centroid splits.
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
    const auto begin = triangles.begin() + first;
    uint32_t half = count / 2;
    constexpr uint32_t Bins = 16;
    struct Bin {
        float3 Low{INFINITY, INFINITY, INFINITY}, High{-INFINITY, -INFINITY, -INFINITY};
        uint32_t Count{};
        void Include(const Bin &b) {
            if (!b.Count) return;
            Low = simd::min(Low, b.Low);
            High = simd::max(High, b.High);
            Count += b.Count;
        }
        double Cost() const {
            if (!Count) return 0;
            const float3 d = High - Low;
            return Count * (double(d.x) * d.y + double(d.y) * d.z + double(d.z) * d.x);
        }
    };
    double best = INFINITY;
    uint32_t selected_axis = 0, selected_bin = 0;
    const auto bin_index = [&](const Triangle &t, uint32_t dimension) {
        return std::min(Bins - 1, uint32_t((centre(t)[dimension] - spread_low[dimension]) / spread[dimension] * Bins));
    };
    // Bound tree depth by the GPU traversal stack capacity.
    const uint64_t child_capacity = uint64_t(LeafSize) << (30 - depth);
    for (uint32_t dimension = 0; dimension < 3; ++dimension) {
        if (!(spread[dimension] > 0)) continue;
        Bin bins[Bins], right[Bins];
        for (uint32_t i = first; i < first + count; ++i) {
            Bin &bin = bins[bin_index(triangles[i], dimension)];
            ++bin.Count;
            for (const Index corner : {triangles[i].A, triangles[i].B, triangles[i].C}) {
                bin.Low = simd::min(bin.Low, points[corner]);
                bin.High = simd::max(bin.High, points[corner]);
            }
        }
        Bin accumulated;
        for (uint32_t b = Bins; b-- > 0;) {
            accumulated.Include(bins[b]);
            right[b] = accumulated;
        }
        Bin left;
        for (uint32_t b = 0; b + 1 < Bins; ++b) {
            left.Include(bins[b]);
            if (!left.Count || !right[b + 1].Count || left.Count > child_capacity || right[b + 1].Count > child_capacity) continue;
            const double cost = left.Cost() + right[b + 1].Cost();
            if (cost < best) {
                best = cost;
                selected_axis = dimension;
                selected_bin = b;
            }
        }
    }
    if (std::isfinite(best)) {
        half = uint32_t(std::stable_partition(begin, begin + count, [&](const Triangle &t) {
                            return bin_index(t, selected_axis) <= selected_bin;
                        }) -
                        begin);
    } else {
        const uint32_t axis = spread.x >= spread.y && spread.x >= spread.z ? 0 : (spread.y >= spread.z ? 1 : 2);
        std::nth_element(begin, begin + half, begin + count, [&](const Triangle &a, const Triangle &b) {
            return centre(a)[axis] < centre(b)[axis];
        });
    }

    // The left child immediately follows its parent.
    Build(triangles, points, nodes, first, half, depth + 1);
    nodes[self].First = Build(triangles, points, nodes, first + half, count - half, depth + 1);
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
    // Quantize vertices to a grid with spacing of one millionth of the mesh extent.
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
        if (triangle.A == triangle.B || triangle.B == triangle.C || triangle.A == triangle.C) continue;
        if (length(cross(cooked.Vertices[triangle.B] - cooked.Vertices[triangle.A], cooked.Vertices[triangle.C] - cooked.Vertices[triangle.A])) <= 0) continue;
        cooked.Triangles.push_back(triangle);
    }
    if (cooked.Triangles.empty()) return {};

    // Remove unused vertices before mesh-plane queries can produce contacts from them.
    where.assign(cooked.Vertices.size(), NoIndex);
    for (const Triangle &triangle : cooked.Triangles)
        for (const Index corner : {triangle.A, triangle.B, triangle.C}) where[corner] = 0;
    uint32_t used = 0;
    for (uint32_t i = 0; i < cooked.Vertices.size(); ++i) {
        if (where[i] == NoIndex) continue;
        where[i] = used;
        cooked.Vertices[used++] = cooked.Vertices[i];
    }
    cooked.Vertices.resize(used);
    for (Triangle &triangle : cooked.Triangles) {
        triangle.A = where[triangle.A];
        triangle.B = where[triangle.B];
        triangle.C = where[triangle.C];
    }

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
            // Boundary and nonmanifold edges remain active.
            bool active = shared.size() != 2;
            bool back_active = active;
            for (const uint32_t other : shared) {
                if (other == t) continue;
                const Triangle &neighbour = cooked.Triangles[other];
                // The adjacent triangle lies behind this plane at a convex crease.
                for (const Index far : {neighbour.A, neighbour.B, neighbour.C}) {
                    if (far == from || far == to) continue;
                    const float3 offset = cooked.Vertices[far] - cooked.Vertices[from];
                    const float span = length(offset);
                    if (span > 0 && dot(normal, offset) / span < -ActiveEdgeSine) active = true;
                    if (span > 0 && dot(normal, offset) / span > ActiveEdgeSine) back_active = true;
                }
            }
            if (active) triangle.ActiveEdges |= 1u << e;
            if (back_active) triangle.BackActiveEdges |= 1u << e;
            // The lower triangle index owns contacts on a shared edge.
            if (shared.size() != 2 || t == std::min(shared[0], shared[1])) triangle.OwnedEdges |= 1u << e;
        }
    }

    Build(cooked.Triangles, cooked.Vertices, cooked.Nodes, 0, cooked.Triangles.size());
    return cooked;
}

} // namespace rbp
