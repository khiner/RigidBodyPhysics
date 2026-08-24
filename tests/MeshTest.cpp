// Cooking a triangle mesh. Two questions have to be answered before a mesh can be collided at all,
// and neither can be answered per triangle at runtime: which triangles are anywhere near a body, and
// which of their edges are features of the shape rather than seams of the tessellation. These check
// both against surfaces whose answers are obvious by eye.

#include "Mesh.h"
#include "World.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <set>
#include <utility>
#include <vector>
#include <doctest/doctest.h>

namespace {
// Two triangles making the quad a-b-c-d, wound so their normals point along `out`. Written this way
// rather than with the winding spelled out, so a test says what surface it means rather than which
// order its corners happen to be in.
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

// Whether the edge between these two points is a feature, and that every triangle along it agrees.
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
            if (found) CHECK(bit == active); // both sides of an edge have to say the same thing about it
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
    // The diagonal is where the quad was cut in two and there is no fold across it, so nothing can hit
    // it. Every edge on the outside is where the surface stops, which is something anything can hit.
    CHECK(!EdgeActive(mesh, points[0], points[2]));
    CHECK(EdgeActive(mesh, points[0], points[1]));
    CHECK(EdgeActive(mesh, points[1], points[2]));
    CHECK(EdgeActive(mesh, points[2], points[3]));
    CHECK(EdgeActive(mesh, points[3], points[0]));
}

TEST_CASE("a ridge is a feature and a valley is not") {
    // The same two quads either side of the same shared edge, folded the two ways round. Which one is
    // a feature is not about how sharp the fold is but about which way it goes: a ridge stands out of
    // the surface where something can strike it, and a valley is a place nothing can reach without
    // touching one of the two faces that make it first.
    const auto fold = [](float height) {
        const std::vector<float3> points{float3{-1, 0, -1}, float3{1, 0, -1}, float3{-1, height, 0},
                                         float3{1, height, 0}, float3{-1, 0, 1}, float3{1, 0, 1}};
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
    // What a renderer hands over: every triangle with its own copies of its corners, so nothing shares
    // an index with anything. Untouched, no two triangles would be seen to share an edge and every seam
    // in the mesh would be taken for a feature.
    std::vector<float3> points;
    std::vector<uint32_t> indices;
    for (const auto corner : {float3{0, 0, 0}, float3{1, 0, 0}, float3{1, 0, 1}, float3{0, 0, 0}, float3{1, 0, 1}, float3{0, 0, 1}}) {
        indices.push_back(uint32_t(points.size()));
        points.push_back(corner);
    }
    const CookedMesh loose = CookMesh(points, indices);
    CHECK(loose.Vertices.size() == 4); // six corners, four places
    CHECK(loose.Triangles.size() == 2);
    CHECK(!EdgeActive(loose, float3{0, 0, 0}, float3{1, 0, 1})); // and the seam is recognised as one
}

TEST_CASE("the tree covers every triangle exactly once") {
    // A grid, big enough that the tree has to split rather than fitting in one leaf.
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
    REQUIRE(mesh.Nodes.size() > 1); // it split

    std::set<uint32_t> covered;
    float3 low{INFINITY, INFINITY, INFINITY}, high{-INFINITY, -INFINITY, -INFINITY};
    for (const BvhNode &node : mesh.Nodes) {
        CHECK(simd::all(node.Low <= node.High));
        if (node.Count == 0) {
            CHECK(node.First < mesh.Nodes.size()); // the right child, the left being the node after this
            continue;
        }
        for (uint32_t i = 0; i < node.Count; ++i) {
            REQUIRE(node.First + i < mesh.Triangles.size());
            CHECK(covered.insert(node.First + i).second); // no triangle in two leaves
            const Triangle &triangle = mesh.Triangles[node.First + i];
            for (const Index corner : {triangle.A, triangle.B, triangle.C}) {
                // A leaf's box has to hold what is in it, or a body over it is never told about it.
                CHECK(simd::all(mesh.Vertices[corner] >= node.Low - 1e-5f));
                CHECK(simd::all(mesh.Vertices[corner] <= node.High + 1e-5f));
                low = simd::min(low, mesh.Vertices[corner]);
                high = simd::max(high, mesh.Vertices[corner]);
            }
        }
    }
    CHECK(covered.size() == mesh.Triangles.size());
    CHECK(simd::all(mesh.Nodes[0].Low <= low + 1e-5f)); // and the root holds the whole of it
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

    // A mesh is a surface with no inside: there is no volume to weigh and no centre of mass to turn
    // about, so a body wearing one keeps its pose whatever it was asked to weigh.
    const auto body = world.AddBody({.Shape = shape, .Density = 1000});
    CHECK(world.Masses[body].InvMass == 0);
    for (uint32_t axis = 0; axis < 3; ++axis) CHECK(world.Masses[body].InvInertiaLocal[axis] == 0);
}
