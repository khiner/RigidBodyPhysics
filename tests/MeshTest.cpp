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
            if (found) CHECK(bit == active);
            found = true;
            active = bit;
        }
    }
    REQUIRE(found);
    return active;
}
} // namespace

TEST_CASE("cooking: mesh seams winding and welding preserve surface features") {
    SUBCASE("a flat quad's diagonal is a seam and its boundary is not") {
        const std::vector<float3> points{float3{0, 0, 0}, float3{1, 0, 0}, float3{1, 0, 1}, float3{0, 0, 1}};
        std::vector<uint32_t> indices;
        Quad(indices, points, 0, 1, 2, 3, float3{0, 1, 0});
        const CookedMesh mesh = CookMesh(points, indices);

        REQUIRE(mesh.Vertices.size() == 4);
        REQUIRE(mesh.Triangles.size() == 2);

        CHECK(!EdgeActive(mesh, points[0], points[2]));
        CHECK(EdgeActive(mesh, points[0], points[1]));
        CHECK(EdgeActive(mesh, points[1], points[2]));
        CHECK(EdgeActive(mesh, points[2], points[3]));
        CHECK(EdgeActive(mesh, points[3], points[0]));
    }
    SUBCASE("a ridge is a feature and a valley is not") {
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
    SUBCASE("a mesh authored with a corner per face is welded back together") {
        std::vector<float3> points;
        std::vector<uint32_t> indices;
        for (const auto corner : {float3{0, 0, 0}, float3{1, 0, 0}, float3{1, 0, 1}, float3{0, 0, 0}, float3{1, 0, 1}, float3{0, 0, 1}}) {
            indices.push_back(uint32_t(points.size()));
            points.push_back(corner);
        }
        const CookedMesh loose = CookMesh(points, indices);
        CHECK(loose.Vertices.size() == 4);
        CHECK(loose.Triangles.size() == 2);
        CHECK(!EdgeActive(loose, float3{0, 0, 0}, float3{1, 0, 1}));
    }
    SUBCASE("a cooked mesh keeps only vertices used by surviving triangles") {
        const float3 points[]{{0, -2, 0}, {0, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 0, 1}};
        const std::vector<uint32_t> indices{1, 4, 2, 0, 3, 0};
        const CookedMesh mesh = CookMesh(points, indices);
        REQUIRE(mesh.Vertices.size() == 3);
        REQUIRE(mesh.Triangles.size() == 1);
        const Triangle triangle = mesh.Triangles[0];
        CHECK(simd::distance(mesh.Vertices[triangle.A], points[1]) == 0);
        CHECK(simd::distance(mesh.Vertices[triangle.B], points[4]) == 0);
        CHECK(simd::distance(mesh.Vertices[triangle.C], points[2]) == 0);
    }
    SUBCASE("what is not a surface makes no mesh") {
        const std::vector<float3> points{float3{0, 0, 0}, float3{1, 0, 0}, float3{1, 0, 1}};
        CHECK(CookMesh(points, std::vector<uint32_t>{0, 1}).Triangles.empty());
        CHECK(CookMesh(points, std::vector<uint32_t>{0, 1, 9}).Triangles.empty());
        CHECK(CookMesh(points, std::vector<uint32_t>{0, 1, 1}).Triangles.empty());
    }
}

TEST_CASE("cooking: mesh hierarchy bounds cover each surviving triangle once") {
    for (uint32_t layout = 0; layout < 3; ++layout) {
        CAPTURE(layout);
        std::vector<float3> points;
        std::vector<uint32_t> indices;
        for (uint32_t i = 0; i < 1025; ++i) {
            const float x = layout == 0 ? float(i % 31) * 0.01f + (i == 1024 ? 100.f : 0.f) :
                layout == 1             ? 0.f :
                                          float(i % 29);
            const float z = layout == 1 ? 0.f : float(i / 31) * 0.02f;
            const float radius = layout == 1 ? 0.01f + float(i) * 0.001f : 0.004f;
            for (float3 point : {float3{x - radius, 0, z - radius}, float3{x + radius, 0, z - radius}, float3{x, 0, z + 2 * radius}}) {
                indices.push_back(points.size());
                points.push_back(point);
            }
        }
        const CookedMesh mesh = CookMesh(points, indices), repeated = CookMesh(points, indices);
        REQUIRE(mesh.Triangles.size() == 1025);
        REQUIRE(mesh.Nodes.size() == repeated.Nodes.size());
        REQUIRE(mesh.Triangles.size() == repeated.Triangles.size());
        std::vector<uint32_t> node_visits(mesh.Nodes.size()), triangle_visits(mesh.Triangles.size());
        const auto visit = [&](auto &&self, uint32_t at, uint32_t depth) -> void {
            REQUIRE(at < mesh.Nodes.size());
            REQUIRE(depth < 32);
            REQUIRE(node_visits[at]++ == 0);
            const BvhNode &node = mesh.Nodes[at], &again = repeated.Nodes[at];
            CHECK(simd::all(node.Low == again.Low));
            CHECK(simd::all(node.High == again.High));
            CHECK(node.First == again.First);
            CHECK(node.Count == again.Count);
            if (node.Count) {
                REQUIRE(node.Count <= 4);
                REQUIRE(node.First + node.Count <= mesh.Triangles.size());
                for (uint32_t i = node.First; i < node.First + node.Count; ++i) {
                    ++triangle_visits[i];
                    const Triangle &t = mesh.Triangles[i], &other = repeated.Triangles[i];
                    CHECK(t.A == other.A);
                    CHECK(t.B == other.B);
                    CHECK(t.C == other.C);
                    CHECK(t.ActiveEdges == other.ActiveEdges);
                    CHECK(t.BackActiveEdges == other.BackActiveEdges);
                    CHECK(t.OwnedEdges == other.OwnedEdges);
                    for (Index corner : {t.A, t.B, t.C}) {
                        REQUIRE(corner < mesh.Vertices.size());
                        CHECK(simd::all(mesh.Vertices[corner] >= node.Low));
                        CHECK(simd::all(mesh.Vertices[corner] <= node.High));
                    }
                }
            } else {
                for (uint32_t child : {at + 1, node.First}) {
                    REQUIRE(child < mesh.Nodes.size());
                    CHECK(simd::all(mesh.Nodes[child].Low >= node.Low));
                    CHECK(simd::all(mesh.Nodes[child].High <= node.High));
                    self(self, child, depth + 1);
                }
            }
        };
        visit(visit, 0, 0);
        CHECK(std::ranges::all_of(node_visits, [](uint32_t n) { return n == 1; }));
        CHECK(std::ranges::all_of(triangle_visits, [](uint32_t n) { return n == 1; }));
    }
}

TEST_CASE("refit: deformation flips front and back creases without changing features") {
    std::vector<float3> points{float3{-1,0,-1},float3{1,0,-1},float3{-1,.5f,0},float3{1,.5f,0},float3{-1,0,1},float3{1,0,1}};
    std::vector<uint32_t> indices;
    Quad(indices,points,0,1,3,2,float3{0,1,0});Quad(indices,points,2,3,5,4,float3{0,1,0});
    auto mesh=CookMesh(points,indices);
    const auto topology=mesh.Triangles;const auto nodes=mesh.Nodes;
    const auto crease_flags=[&](bool front) {
        int found=0;
        for(const auto& t:mesh.Triangles) {
            const Index corner[]{t.A,t.B,t.C};
            for(int e=0;e<3;++e) if(mesh.Vertices[corner[e]].z==0 && mesh.Vertices[corner[(e+1)%3]].z==0) {
                CHECK(bool(t.ActiveEdges&(1u<<e))==front);
                CHECK(bool(t.BackActiveEdges&(1u<<e))==!front);++found;
            }
        }
        CHECK(found==2);
    };
    crease_flags(true);
    for(auto& p:mesh.Vertices) if(p.z==0) p.y=-.5f;
    REQUIRE(RefitMesh(mesh.Vertices,mesh.Triangles,mesh.Nodes));crease_flags(false);
    for(std::size_t i=0;i<topology.size();++i) {
        CHECK(mesh.Triangles[i].A==topology[i].A);CHECK(mesh.Triangles[i].B==topology[i].B);
        CHECK(mesh.Triangles[i].C==topology[i].C);CHECK(mesh.Triangles[i].OwnedEdges==topology[i].OwnedEdges);
    }
    for(std::size_t i=0;i<nodes.size();++i) {CHECK(mesh.Nodes[i].First==nodes[i].First);CHECK(mesh.Nodes[i].Count==nodes[i].Count);}
    CHECK(simd::all(mesh.Nodes[0].Low==float3{-1,-.5f,-1}));CHECK(simd::all(mesh.Nodes[0].High==float3{1,0,1}));
    for(auto& p:mesh.Vertices) if(p.z==0) p.y=0;
    REQUIRE(RefitMesh(mesh.Vertices,mesh.Triangles,mesh.Nodes));
    CHECK(!EdgeActive(mesh,float3{-1,0,0},float3{1,0,0}));
    for(const auto& t:mesh.Triangles) {
        const Index corner[]{t.A,t.B,t.C};
        for(int e=0;e<3;++e) if(mesh.Vertices[corner[e]].z==0 && mesh.Vertices[corner[(e+1)%3]].z==0)
            CHECK(!(t.BackActiveEdges&(1u<<e)));
    }
}

TEST_CASE("refit: every BVH bound equals its deformed descendant geometry") {
    std::vector<float3> points;std::vector<uint32_t> indices;
    for(int t=0;t<17;++t) for(float3 p:{float3{float(t),0,0},float3{float(t)+.3f,0,0},float3{float(t),0,.4f}}) {
        indices.push_back(points.size());points.push_back(p);
    }
    auto mesh=CookMesh(points,indices);const auto triangles=mesh.Triangles;const auto topology=mesh.Nodes;
    REQUIRE(mesh.Nodes.size()>1);
    for(auto& p:mesh.Vertices) {p.y=.07f*p.x*p.x;p.x=-3*p.x+2;p.z=2*p.z-1;}
    REQUIRE(RefitMesh(mesh.Vertices,mesh.Triangles,mesh.Nodes));
    // Collect each node's complete descendant triangle set, then bound the
    // vertices directly instead of reproducing the refit's child-box reduction.
    const auto descendants=[&](auto&& self,uint32_t at)->std::vector<uint32_t> {
        const auto& node=mesh.Nodes[at];std::vector<uint32_t> found;
        if(node.Count) for(uint32_t i=0;i<node.Count;++i) found.push_back(node.First+i);
        else {found=self(self,at+1);const auto right=self(self,node.First);found.insert(found.end(),right.begin(),right.end());}
        float3 low{INFINITY,INFINITY,INFINITY},high=-low;
        for(auto i:found) for(auto v:{mesh.Triangles[i].A,mesh.Triangles[i].B,mesh.Triangles[i].C}) {
            low=simd::min(low,mesh.Vertices[v]);high=simd::max(high,mesh.Vertices[v]);
        }
        CHECK(simd::all(node.Low==low));CHECK(simd::all(node.High==high));
        CHECK(node.First==topology[at].First);CHECK(node.Count==topology[at].Count);
        return found;
    };
    CHECK(descendants(descendants,0).size()==triangles.size());
    for(std::size_t i=0;i<triangles.size();++i) {
        CHECK(mesh.Triangles[i].A==triangles[i].A);CHECK(mesh.Triangles[i].B==triangles[i].B);
        CHECK(mesh.Triangles[i].C==triangles[i].C);CHECK(mesh.Triangles[i].OwnedEdges==triangles[i].OwnedEdges);
    }
}

TEST_CASE("refit: malformed geometry and topology reject without partial metadata updates") {
    std::vector<float3> points;std::vector<uint32_t> indices;
    for(int t=0;t<9;++t) for(float3 p:{float3{float(t),0,0},float3{float(t)+.3f,0,0},float3{float(t),0,.4f}}) {
        indices.push_back(points.size());points.push_back(p);
    }
    auto mesh=CookMesh(points,indices);REQUIRE(mesh.Nodes.size()>1);
    SUBCASE("nonfinite coordinate") {mesh.Vertices.back().y=INFINITY;}
    SUBCASE("degenerate triangle") {mesh.Vertices[mesh.Triangles.back().C]=mesh.Vertices[mesh.Triangles.back().A];}
    SUBCASE("invalid vertex index") {mesh.Triangles.back().B=uint32_t(mesh.Vertices.size());}
    SUBCASE("cycle") {mesh.Nodes[0].First=0;}
    SUBCASE("missing right child") {mesh.Nodes[0].First=uint32_t(mesh.Nodes.size());}
    SUBCASE("overlapping child subtrees") {mesh.Nodes[0].First=1;}
    SUBCASE("overflowing leaf range") {mesh.Nodes.back().First=~0u;}
    SUBCASE("repeated triangle range") {mesh.Nodes.back().First=mesh.Nodes[1].Count?mesh.Nodes[1].First:mesh.Nodes[2].First;}
    SUBCASE("unreachable extra node") {mesh.Nodes.push_back(mesh.Nodes.back());}
    const auto before_triangles=mesh.Triangles;const auto before_nodes=mesh.Nodes;
    CHECK_FALSE(RefitMesh(mesh.Vertices,mesh.Triangles,mesh.Nodes));
    for(std::size_t i=0;i<mesh.Triangles.size();++i) {
        const auto& a=mesh.Triangles[i];const auto& b=before_triangles[i];
        CHECK(a.A==b.A);CHECK(a.B==b.B);CHECK(a.C==b.C);CHECK(a.ActiveEdges==b.ActiveEdges);
        CHECK(a.BackActiveEdges==b.BackActiveEdges);CHECK(a.OwnedEdges==b.OwnedEdges);
    }
    for(std::size_t i=0;i<mesh.Nodes.size();++i) {
        const auto& a=mesh.Nodes[i];const auto& b=before_nodes[i];
        CHECK(simd::all(a.Low==b.Low));CHECK(simd::all(a.High==b.High));CHECK(a.First==b.First);CHECK(a.Count==b.Count);
    }
}
