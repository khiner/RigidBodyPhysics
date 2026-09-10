#include "Hull.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <utility>

namespace rbp {

namespace {
using double3 = simd::double3;

double3 ToDouble(float3 v) { return {v.x, v.y, v.z}; }

// Integrate mass properties in double precision to retain small off-diagonal inertia terms.
struct Mat3 {
    double M[3][3]{};
};

Mat3 Multiply(const Mat3 &a, const Mat3 &b) {
    Mat3 out;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k) out.M[i][j] += a.M[i][k] * b.M[k][j];
    return out;
}

Mat3 Transpose(const Mat3 &a) {
    Mat3 out;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) out.M[i][j] = a.M[j][i];
    return out;
}

struct Face {
    uint32_t Corner[3];
    float3 Normal;
    float Offset;
    bool Live = true;
};

Face MakeFace(std::span<const float3> points, uint32_t a, uint32_t b, uint32_t c) {
    const float3 turn = cross(points[b] - points[a], points[c] - points[a]);
    const float area = length(turn);
    // Reject slivers whose normal cannot be normalized reliably.
    if (area < 1e-20f) return {.Corner = {a, b, c}, .Normal = {0, 0, 0}, .Offset = 1};
    const float3 normal = turn / area;
    return {.Corner = {a, b, c}, .Normal = normal, .Offset = dot(normal, points[a])};
}

std::vector<Face> SeedTetrahedron(std::span<const float3> points, float epsilon) {
    uint32_t first = 0, second = 0;
    float furthest = 0;
    for (uint32_t axis = 0; axis < 3; ++axis) {
        uint32_t low = 0, high = 0;
        for (uint32_t i = 0; i < points.size(); ++i) {
            if (points[i][axis] < points[low][axis]) low = i;
            if (points[i][axis] > points[high][axis]) high = i;
        }
        const float span = simd::distance(points[low], points[high]);
        if (span > furthest) {
            furthest = span;
            first = low;
            second = high;
        }
    }
    if (furthest <= epsilon) return {};

    const float3 along = points[second] - points[first];
    uint32_t third = 0;
    float widest = 0;
    for (uint32_t i = 0; i < points.size(); ++i) {
        const float off_line = length(cross(along, points[i] - points[first])) / furthest;
        if (off_line > widest) {
            widest = off_line;
            third = i;
        }
    }
    if (widest <= epsilon) return {};

    const Face base = MakeFace(points, first, second, third);
    uint32_t fourth = 0;
    float deepest = 0;
    for (uint32_t i = 0; i < points.size(); ++i) {
        const float off_plane = std::abs(dot(base.Normal, points[i]) - base.Offset);
        if (off_plane > deepest) {
            deepest = off_plane;
            fourth = i;
        }
    }
    if (deepest <= epsilon) return {};

    const bool flip = dot(base.Normal, points[fourth]) - base.Offset > 0;
    const uint32_t a = first, b = flip ? third : second, c = flip ? second : third;
    return {MakeFace(points, a, b, c), MakeFace(points, b, a, fourth), MakeFace(points, c, b, fourth), MakeFace(points, a, c, fourth)};
}

float OutsideBy(std::span<const Face> faces, float3 point) {
    float outside = -INFINITY;
    for (const Face &face : faces)
        if (face.Live) outside = std::max(outside, dot(face.Normal, point) - face.Offset);
    return outside;
}

void AddPoint(std::span<const float3> points, std::vector<Face> &faces, uint32_t point, std::vector<std::pair<uint32_t, uint32_t>> &rim) {
    rim.clear();
    for (Face &face : faces) {
        if (!face.Live) continue;
        const double3 a = ToDouble(points[face.Corner[0]]);
        const double3 normal = cross(ToDouble(points[face.Corner[1]]) - a, ToDouble(points[face.Corner[2]]) - a);
        if (dot(normal, ToDouble(points[point]) - a) <= 0) continue;
        face.Live = false;
        for (uint32_t e = 0; e < 3; ++e) rim.emplace_back(face.Corner[e], face.Corner[(e + 1) % 3]);
    }
    for (const auto edge : rim) {
        const auto reverse = std::pair{edge.second, edge.first};
        if (std::ranges::find(rim, reverse) == rim.end()) faces.push_back(MakeFace(points, edge.first, edge.second, point));
    }
}

std::vector<Face> BuildFaces(std::span<const float3> points, float epsilon) {
    std::vector<Face> faces = SeedTetrahedron(points, epsilon);
    if (faces.empty()) return {};
    std::vector<std::pair<uint32_t, uint32_t>> rim;
    for (uint32_t i = 0; i < points.size(); ++i)
        if (OutsideBy(faces, points[i]) > epsilon) AddPoint(points, faces, i, rim);
    std::erase_if(faces, [](const Face &face) { return !face.Live; });
    return faces;
}

std::vector<uint32_t> Corners(std::span<const Face> faces) {
    std::vector<uint32_t> corners;
    for (const Face &face : faces) {
        if (!face.Live) continue;
        for (const uint32_t corner : face.Corner)
            if (std::ranges::find(corners, corner) == corners.end()) corners.push_back(corner);
    }
    std::ranges::sort(corners);
    return corners;
}

std::vector<Face> BuildSimplified(std::span<const float3> points, float epsilon, uint32_t limit, float &tolerance) {
    std::vector<Face> faces = SeedTetrahedron(points, epsilon);
    if (faces.empty()) return {};
    std::vector<std::pair<uint32_t, uint32_t>> rim;
    std::vector<uint8_t> inserted(points.size());
    for (const auto corner : Corners(faces)) inserted[corner] = true;

    while (Corners(faces).size() < limit) {
        uint32_t furthest = NoIndex;
        float outside = epsilon;
        for (uint32_t i = 0; i < points.size(); ++i)
            if (const float by = inserted[i] ? -INFINITY : OutsideBy(faces, points[i]); by > outside) {
                outside = by;
                furthest = i;
            }
        if (furthest == NoIndex) break;
        AddPoint(points, faces, furthest, rim);

        inserted[furthest] = true;
        std::erase_if(faces, [](const Face &face) { return !face.Live; });
    }
    tolerance = 0;
    for (const float3 point : points) tolerance = std::max(tolerance, OutsideBy(faces, point));
    if (tolerance <= epsilon) tolerance = 0;
    std::erase_if(faces, [](const Face &face) { return !face.Live; });
    return faces;
}

struct Loop {
    std::vector<uint32_t> Corner;
    float3 Normal;
    float Offset;
};

uint64_t EdgeKey(uint32_t from, uint32_t to) { return (uint64_t(from) << 32) | to; }

std::vector<Loop> MergeCoplanar(std::span<const float3> points, std::span<const Face> faces, float epsilon) {
    std::unordered_map<uint64_t, uint32_t> owner;
    owner.reserve(faces.size() * 3);
    for (uint32_t t = 0; t < faces.size(); ++t)
        for (uint32_t e = 0; e < 3; ++e) owner.emplace(EdgeKey(faces[t].Corner[e], faces[t].Corner[(e + 1) % 3]), t);

    const auto has_area = [&faces](uint32_t t) { return simd::length_squared(faces[t].Normal) > 0.5f; };
    std::vector<Loop> loops;
    std::vector<bool> taken(faces.size(), false);
    std::vector<uint32_t> members, pending;
    std::vector<std::pair<uint32_t, uint32_t>> edges;
    for (uint32_t seed = 0; seed < faces.size(); ++seed) {
        if (taken[seed] || !has_area(seed)) continue;
        const float3 normal = faces[seed].Normal;
        // Measure plane distance relative to a face corner to limit cancellation.
        const float3 anchor = points[faces[seed].Corner[0]];
        members.clear();
        pending.assign(1, seed);
        taken[seed] = true;
        while (!pending.empty()) {
            const uint32_t at = pending.back();
            pending.pop_back();
            members.push_back(at);
            for (uint32_t e = 0; e < 3; ++e) {
                const auto across = owner.find(EdgeKey(faces[at].Corner[(e + 1) % 3], faces[at].Corner[e]));
                if (across == owner.end() || taken[across->second]) continue;
                bool flat = true;
                for (const uint32_t corner : faces[across->second].Corner)
                    flat = flat && std::abs(dot(normal, points[corner] - anchor)) <= epsilon;
                if (!flat) continue;
                taken[across->second] = true;
                pending.push_back(across->second);
            }
        }

        edges.clear();
        for (const uint32_t member : members)
            for (uint32_t e = 0; e < 3; ++e) edges.emplace_back(faces[member].Corner[e], faces[member].Corner[(e + 1) % 3]);
        std::vector<bool> cancelled(edges.size(), false);
        for (uint32_t i = 0; i < edges.size(); ++i) {
            if (cancelled[i]) continue;
            for (uint32_t j = i + 1; j < edges.size(); ++j)
                if (!cancelled[j] && edges[j].first == edges[i].second && edges[j].second == edges[i].first) {
                    cancelled[i] = cancelled[j] = true;
                    break;
                }
        }
        std::unordered_map<uint32_t, uint32_t> rim;
        bool simple = true;
        for (uint32_t i = 0; i < edges.size(); ++i)
            if (!cancelled[i]) simple = rim.emplace(edges[i].first, edges[i].second).second && simple;

        Loop loop{.Normal = normal, .Offset = dot(normal, anchor)};
        if (simple && !rim.empty()) {
            uint32_t start = ~0u;
            for (const auto &[from, to] : rim) start = std::min(start, from);
            for (uint32_t at = start; loop.Corner.size() <= rim.size();) {
                loop.Corner.push_back(at);
                at = rim.find(at)->second;
                if (at == start) break;
            }
        }

        if (loop.Corner.size() != rim.size()) {
            for (const uint32_t member : members)
                if (has_area(member))
                    loops.push_back({.Corner = {faces[member].Corner[0], faces[member].Corner[1], faces[member].Corner[2]}, .Normal = faces[member].Normal, .Offset = faces[member].Offset});
            continue;
        }
        loops.push_back(std::move(loop));
    }
    return loops;
}

float4 QuatFromAxes(double3 x, double3 y, double3 z) {
    const double trace = x.x + y.y + z.z;
    if (trace > 0) {
        const double s = std::sqrt(trace + 1) * 2;
        return {float((y.z - z.y) / s), float((z.x - x.z) / s), float((x.y - y.x) / s), float(0.25 * s)};
    }
    if (x.x > y.y && x.x > z.z) {
        const double s = std::sqrt(1 + x.x - y.y - z.z) * 2;
        return {float(0.25 * s), float((y.x + x.y) / s), float((z.x + x.z) / s), float((y.z - z.y) / s)};
    }
    if (y.y > z.z) {
        const double s = std::sqrt(1 + y.y - x.x - z.z) * 2;
        return {float((y.x + x.y) / s), float(0.25 * s), float((z.y + y.z) / s), float((z.x - x.z) / s)};
    }
    const double s = std::sqrt(1 + z.z - x.x - y.y) * 2;
    return {float((z.x + x.z) / s), float((z.y + y.z) / s), float(0.25 * s), float((x.y - y.x) / s)};
}

void Diagonalize(Mat3 a, Mat3 &vectors, double3 &values) {
    vectors = {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
    // Compare off-diagonal inertia against diagonal scale before further Jacobi rotations.
    const double scale = std::abs(a.M[0][0]) + std::abs(a.M[1][1]) + std::abs(a.M[2][2]);
    const double negligible = 1e-12 * scale;
    for (int sweep = 0; sweep < 16; ++sweep) {
        if (std::abs(a.M[0][1]) + std::abs(a.M[0][2]) + std::abs(a.M[1][2]) <= negligible) break;
        for (const auto [p, q] : {std::pair{0, 1}, std::pair{0, 2}, std::pair{1, 2}}) {
            if (std::abs(a.M[p][q]) <= negligible) continue;
            const double theta = (a.M[q][q] - a.M[p][p]) / (2 * a.M[p][q]);
            const double t = (theta >= 0 ? 1. : -1.) / (std::abs(theta) + std::sqrt(theta * theta + 1));
            const double c = 1 / std::sqrt(t * t + 1), s = t * c;
            for (int k = 0; k < 3; ++k) {
                const double ap = a.M[k][p], aq = a.M[k][q];
                a.M[k][p] = c * ap - s * aq;
                a.M[k][q] = s * ap + c * aq;
            }
            for (int k = 0; k < 3; ++k) {
                const double ap = a.M[p][k], aq = a.M[q][k];
                a.M[p][k] = c * ap - s * aq;
                a.M[q][k] = s * ap + c * aq;
            }
            for (int k = 0; k < 3; ++k) {
                const double vp = vectors.M[k][p], vq = vectors.M[k][q];
                vectors.M[k][p] = c * vp - s * vq;
                vectors.M[k][q] = s * vp + c * vq;
            }
        }
    }
    values = {a.M[0][0], a.M[1][1], a.M[2][2]};
}
} // namespace

Diagonalized DiagonalizeSymmetric(const double (&symmetric)[3][3]) {
    Mat3 given;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) given.M[i][j] = symmetric[i][j];
    Mat3 axes;
    double3 values;
    Diagonalize(given, axes, values);
    // Preserve a right-handed principal-axis frame.
    const double3 y{axes.M[0][1], axes.M[1][1], axes.M[2][1]}, z{axes.M[0][2], axes.M[1][2], axes.M[2][2]};
    double3 x{axes.M[0][0], axes.M[1][0], axes.M[2][0]};
    if (dot(cross(x, y), z) < 0) x = -x;
    return {.Values = values, .Axis = {x, y, z}, .Orientation = QuatFromAxes(x, y, z)};
}

CookedHull CookHull(std::span<const float3> points) {
    if (points.size() < 4) return {};
    float3 low = points[0], high = points[0];
    for (const float3 point : points) {
        low = simd::min(low, point);
        high = simd::max(high, point);
    }
    // Scale degeneracy tests by hull extent so validity is independent of units.
    float carried = 0;
    for (const float3 point : points) carried = std::max(carried, std::abs(point.x) + std::abs(point.y) + std::abs(point.z));
    const float epsilon = std::max(1e-6f * std::max({high.x - low.x, high.y - low.y, high.z - low.z, 1e-6f}), 3 * std::numeric_limits<float>::epsilon() * carried);

    const float3 origin = 0.5f * (low + high);
    std::vector<float3> centred;
    centred.reserve(points.size());
    for (const float3 point : points) centred.push_back(point - origin);
    points = centred;

    std::vector<Face> faces = BuildFaces(points, epsilon);
    if (faces.empty()) return {};

    float tolerance = 0;
    std::vector<float3> exact;
    if (const std::vector<uint32_t> corners = Corners(faces); corners.size() > MaxHullVertices) {
        for (const uint32_t corner : corners) exact.push_back(points[corner]);
        faces = BuildSimplified(exact, epsilon, MaxHullVertices, tolerance);
        if (faces.empty()) return {};
        points = exact;
    }

    // Integrate boundary tetrahedra using the divergence theorem.
    constexpr Mat3 Canonical{{{2 / 120., 1 / 120., 1 / 120.}, {1 / 120., 2 / 120., 1 / 120.}, {1 / 120., 1 / 120., 2 / 120.}}};
    double volume = 0;
    double3 moment{0, 0, 0};
    Mat3 covariance;
    for (const Face &face : faces) {
        const double3 a = ToDouble(points[face.Corner[0]]), b = ToDouble(points[face.Corner[1]]), c = ToDouble(points[face.Corner[2]]);
        const double determinant = dot(a, cross(b, c));
        volume += determinant / 6;
        moment += determinant / 6 * (a + b + c) / 4;
        const Mat3 edges{{{a.x, b.x, c.x}, {a.y, b.y, c.y}, {a.z, b.z, c.z}}};
        const Mat3 tetrahedron = Multiply(Multiply(edges, Canonical), Transpose(edges));
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) covariance.M[i][j] += determinant * tetrahedron.M[i][j];
    }
    if (volume <= 0) return {};

    const double3 center = moment / volume;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) covariance.M[i][j] -= volume * center[i] * center[j];

    const Diagonalized principal = DiagonalizeSymmetric(covariance.M);
    const double3 spread = principal.Values;
    const double3 x = principal.Axis[0], y = principal.Axis[1], z = principal.Axis[2];

    CookedHull cooked{.Volume = float(volume), .Inertia = {float(spread.y + spread.z), float(spread.x + spread.z), float(spread.x + spread.y)}, .Tolerance = tolerance, .Frame = {.Position = origin + float3{float(center.x), float(center.y), float(center.z)}, .Orientation = principal.Orientation}};

    const std::vector<uint32_t> corners = Corners(faces);
    for (const uint32_t corner : corners) {
        const double3 offset = ToDouble(points[corner]) - center;
        cooked.Vertices.push_back(float3{float(dot(offset, x)), float(dot(offset, y)), float(dot(offset, z))});
    }

    std::vector<uint32_t> renamed(points.size(), NoIndex);
    for (uint32_t i = 0; i < corners.size(); ++i) renamed[corners[i]] = i;
    for (const Loop &loop : MergeCoplanar(points, faces, epsilon)) {
        const double3 normal = ToDouble(loop.Normal);
        const float3 turned{float(dot(normal, x)), float(dot(normal, y)), float(dot(normal, z))};
        std::vector<uint32_t> rim;
        for (const uint32_t corner : loop.Corner) {
            if (renamed[corner] == NoIndex) {
                rim.clear();
                break;
            }
            rim.push_back(renamed[corner]);
        }
        if (rim.size() < 3) continue;

        // Sample oversized face boundaries cyclically to preserve their extent.
        if (rim.size() > MaxFacePoints) {
            std::vector<uint32_t> keep{0};
            while (keep.size() < MaxFacePoints) {
                uint32_t pick = 0;
                float widest = -1;
                for (uint32_t i = 0; i < rim.size(); ++i) {
                    if (std::ranges::find(keep, i) != keep.end()) continue;
                    float nearest = INFINITY;
                    for (const uint32_t k : keep) nearest = std::min(nearest, simd::distance_squared(cooked.Vertices[rim[i]], cooked.Vertices[rim[k]]));
                    if (nearest <= widest) continue;
                    widest = nearest;
                    pick = i;
                }
                keep.push_back(pick);
            }
            std::ranges::sort(keep);
            std::vector<uint32_t> sampled;
            for (const uint32_t k : keep) sampled.push_back(rim[k]);
            rim = std::move(sampled);
        }

        HullFace face{.Normal = turned, .Offset = dot(turned, cooked.Vertices[rim[0]]), .Count = uint32_t(rim.size())};
        for (uint32_t i = 0; i < rim.size(); ++i) face.Corner[i] = uchar(rim[i]);
        cooked.Faces.push_back(face);
    }
    if (cooked.Faces.empty()) return {};
    return cooked;
}

} // namespace rbp
