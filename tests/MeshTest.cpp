// Cooking a triangle mesh.
// Two properties are computed up front rather than per triangle at runtime: which triangles lie near a body, and which edges are features rather than seams.

#include "Mesh.h"
#include "World.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <doctest/doctest.h>
#include <set>
#include <utility>
#include <vector>

using namespace rbp;

namespace {
// The quad a-b-c-d as two triangles wound so their normals point along `out`.
// A test then specifies a facing direction rather than a corner order.
void Quad(std::vector<uint32_t> &indices, const std::vector<float3> &points, uint32_t a, uint32_t b, uint32_t c, uint32_t d, float3 out) {
    for (const auto triple : {std::array{a, b, c}, std::array{a, c, d}}) {
        const float3 turn = cross(points[triple[1]] - points[triple[0]], points[triple[2]] - points[triple[0]]);
        if (dot(turn, out) >= 0) indices.insert(indices.end(), {triple[0], triple[1], triple[2]});
        else indices.insert(indices.end(), {triple[0], triple[2], triple[1]});
    }
}

uint32_t Find(const CookedMesh &mesh, float3 at) {
    for (uint32_t i = 0; i < mesh.Vertices.size(); ++i)
        if (simd::distance(mesh.Vertices[i], at) < 1e-5f) return i;
    return ~0u;
}

// Whether the edge between these two points is a feature, checking the bit is the same on every triangle along it.
bool EdgeActive(const CookedMesh &mesh, float3 from, float3 to) {
    const uint32_t a = Find(mesh, from), b = Find(mesh, to);
    REQUIRE(a != ~0u);
    REQUIRE(b != ~0u);
    bool found = false, active = false;
    for (const Triangle &triangle : mesh.Triangles) {
        const Index corner[3]{triangle.A, triangle.B, triangle.C};
        for (uint32_t e = 0; e < 3; ++e) {
            const Index from_index = corner[e], to_index = corner[(e + 1) % 3];
            if (std::min(from_index, to_index) != std::min(Index(a), Index(b)) ||
                std::max(from_index, to_index) != std::max(Index(a), Index(b))) continue;
            const bool bit = (triangle.ActiveEdges & (1u << e)) != 0;
            if (found) CHECK(bit == active); // the edge bit is identical on both triangles
            found = true;
            active = bit;
        }
    }
    REQUIRE(found);
    return active;
}
} // namespace

TEST_CASE("a flat quad's diagonal is a seam and its boundary is not") {
    const std::vector<float3> points{float3{0, 0, 0}, float3{1, 0, 0}, float3{1, 0, 1}, float3{0, 0, 1}};
    std::vector<uint32_t> indices;
    Quad(indices, points, 0, 1, 2, 3, float3{0, 1, 0});
    const CookedMesh mesh = CookMesh(points, indices);

    REQUIRE(mesh.Vertices.size() == 4);
    REQUIRE(mesh.Triangles.size() == 2);
    // The diagonal has no fold across it, so it is a seam.
    // Every outside edge bounds the surface, so it is a feature.
    CHECK(!EdgeActive(mesh, points[0], points[2]));
    CHECK(EdgeActive(mesh, points[0], points[1]));
    CHECK(EdgeActive(mesh, points[1], points[2]));
    CHECK(EdgeActive(mesh, points[2], points[3]));
    CHECK(EdgeActive(mesh, points[3], points[0]));
}

TEST_CASE("a ridge is a feature and a valley is not") {
    // The same two quads folded each way round.
    // A ridge is a feature because a body strikes it directly, and a valley is reached only after touching a face.
    const auto fold = [](float height) {
        const std::vector<float3> points{float3{-1, 0, -1}, float3{1, 0, -1}, float3{-1, height, 0}, float3{1, height, 0}, float3{-1, 0, 1}, float3{1, 0, 1}};
        std::vector<uint32_t> indices;
        Quad(indices, points, 0, 1, 3, 2, float3{0, 1, 0});
        Quad(indices, points, 2, 3, 5, 4, float3{0, 1, 0});
        return std::pair{points, CookMesh(points, indices)};
    };

    const auto [up, ridge] = fold(0.5f);
    REQUIRE(ridge.Triangles.size() == 4);
    CHECK(EdgeActive(ridge, up[2], up[3]));

    const auto [down, valley] = fold(-0.5f);
    REQUIRE(valley.Triangles.size() == 4);
    CHECK(!EdgeActive(valley, down[2], down[3]));
}

TEST_CASE("a mesh authored with a corner per face is welded back together") {
    // The form a renderer produces.
    // Unwelded, no two triangles share an edge and every seam would count as a feature.
    std::vector<float3> points;
    std::vector<uint32_t> indices;
    for (const auto corner : {float3{0, 0, 0}, float3{1, 0, 0}, float3{1, 0, 1}, float3{0, 0, 0}, float3{1, 0, 1}, float3{0, 0, 1}}) {
        indices.push_back(uint32_t(points.size()));
        points.push_back(corner);
    }
    const CookedMesh loose = CookMesh(points, indices);
    CHECK(loose.Vertices.size() == 4); // six authored corners weld to four vertices
    CHECK(loose.Triangles.size() == 2);
    CHECK(!EdgeActive(loose, float3{0, 0, 0}, float3{1, 0, 1})); // the shared diagonal is a seam
}

TEST_CASE("the tree covers every triangle exactly once") {
    // A grid large enough that the tree splits rather than fitting in one leaf.
    std::vector<float3> points;
    std::vector<uint32_t> indices;
    constexpr uint32_t Side = 8;
    for (uint32_t x = 0; x <= Side; ++x)
        for (uint32_t z = 0; z <= Side; ++z) points.push_back(float3{float(x), 0, float(z)});
    const auto at = [](uint32_t x, uint32_t z) { return x * (Side + 1) + z; };
    for (uint32_t x = 0; x < Side; ++x)
        for (uint32_t z = 0; z < Side; ++z) Quad(indices, points, at(x, z), at(x + 1, z), at(x + 1, z + 1), at(x, z + 1), float3{0, 1, 0});

    const CookedMesh mesh = CookMesh(points, indices);
    REQUIRE(mesh.Triangles.size() == 2 * Side * Side);
    REQUIRE(mesh.Nodes.size() > 1);

    std::set<uint32_t> covered;
    float3 low{INFINITY, INFINITY, INFINITY}, high{-INFINITY, -INFINITY, -INFINITY};
    for (const BvhNode &node : mesh.Nodes) {
        CHECK(simd::all(node.Low <= node.High));
        if (node.Count == 0) {
            CHECK(node.First < mesh.Nodes.size()); // First is the right child, the left child being the next node
            continue;
        }
        for (uint32_t i = 0; i < node.Count; ++i) {
            REQUIRE(node.First + i < mesh.Triangles.size());
            CHECK(covered.insert(node.First + i).second); // no triangle in two leaves
            const Triangle &triangle = mesh.Triangles[node.First + i];
            for (const Index corner : {triangle.A, triangle.B, triangle.C}) {
                // A leaf's box must contain its triangles, or a body overlapping them is never tested against them.
                CHECK(simd::all(mesh.Vertices[corner] >= node.Low - 1e-5f));
                CHECK(simd::all(mesh.Vertices[corner] <= node.High + 1e-5f));
                low = simd::min(low, mesh.Vertices[corner]);
                high = simd::max(high, mesh.Vertices[corner]);
            }
        }
    }
    CHECK(covered.size() == mesh.Triangles.size());
    CHECK(simd::all(mesh.Nodes[0].Low <= low + 1e-5f)); // the root box contains every vertex
    CHECK(simd::all(mesh.Nodes[0].High >= high - 1e-5f));
}

TEST_CASE("what is not a surface makes no mesh") {
    const std::vector<float3> points{float3{0, 0, 0}, float3{1, 0, 0}, float3{1, 0, 1}};
    CHECK(CookMesh(points, std::vector<uint32_t>{0, 1}).Triangles.empty()); // not whole triangles
    CHECK(CookMesh(points, std::vector<uint32_t>{0, 1, 9}).Triangles.empty()); // out of range
    CHECK(CookMesh(points, std::vector<uint32_t>{0, 1, 1}).Triangles.empty()); // no area
}

TEST_CASE("a body given a mesh is static whatever density it asks for") {
    const mtl::Context context;
    World world{context};
    const std::vector<float3> points{float3{0, 0, 0}, float3{1, 0, 0}, float3{1, 0, 1}, float3{0, 0, 1}};
    std::vector<uint32_t> indices;
    Quad(indices, points, 0, 1, 2, 3, float3{0, 1, 0});
    const auto shape = world.AddMesh(points, indices);
    REQUIRE(shape != NoIndex);

    // A mesh is a surface with no interior, so it has no volume and no centre of mass.
    const auto body = world.AddBody({.Shape = shape, .Density = 1000});
    CHECK(world.Masses[body].InvMass == 0);
    for (uint32_t axis = 0; axis < 3; ++axis) CHECK(world.Masses[body].InvInertiaLocal[axis] == 0);
}
