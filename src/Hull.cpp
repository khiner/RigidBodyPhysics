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

// Mass properties are integrated in double.
// The integral below sums signed tetrahedra over the whole boundary, so it is a difference of much larger numbers.
// In float a cube's covariance comes out isotropic only to a part in ten million.
// The diagonalization then picks a principal frame out of that noise and turns the cube forty-five degrees.
// Row major, being an integral rather than a transform.
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

// One triangle of the hull, wound so its normal points out of the solid.
struct Face {
    uint32_t Corner[3];
    float3 Normal;
    float Offset; // dot(Normal, corner), so a point is outside when dot(Normal, p) exceeds it
    bool Live = true;
};

Face MakeFace(std::span<const float3> points, uint32_t a, uint32_t b, uint32_t c) {
    const float3 turn = cross(points[b] - points[a], points[c] - points[a]);
    const float area = length(turn);
    // A sliver, from three points a rounding error away from collinear.
    // It is given a normal no point can be outside of and left in place, because dropping it would hole the boundary the integral runs over.
    if (area < 1e-20f) return {.Corner = {a, b, c}, .Normal = {0, 0, 0}, .Offset = 1};
    const float3 normal = turn / area;
    return {.Corner = {a, b, c}, .Normal = normal, .Offset = dot(normal, points[a])};
}

// The tetrahedron an incremental hull starts from: the four points furthest from being one point, one line and one plane.
// Empty when the input holds no solid.
std::vector<Face> SeedTetrahedron(std::span<const float3> points, float epsilon) {
    // Two points far apart, so the tetrahedron is not built out of three that were nearly the same one.
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
    if (furthest <= epsilon) return {}; // every point in one place

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
    if (widest <= epsilon) return {}; // collinear

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
    if (deepest <= epsilon) return {}; // coplanar, so no solid

    // Wound away from the fourth point, so every normal points out.
    const bool flip = dot(base.Normal, points[fourth]) - base.Offset > 0;
    const uint32_t a = first, b = flip ? third : second, c = flip ? second : third;
    return {MakeFace(points, a, b, c), MakeFace(points, b, a, fourth), MakeFace(points, c, b, fourth), MakeFace(points, a, c, fourth)};
}

// How far past the furthest face plane a point stands, negative for a point inside.
// Faces marked not live were replaced by an earlier point.
float OutsideBy(std::span<const Face> faces, float3 point) {
    float outside = -INFINITY;
    for (const Face &face : faces)
        if (face.Live) outside = std::max(outside, dot(face.Normal, point) - face.Offset);
    return outside;
}

// Adds one point to the hull: deletes every face it stands outside of, and fills the resulting hole with triangles from the rim.
// The rim is the directed edges of the deleted faces whose reverse no other deleted face carries.
// Building on those edges in their own direction keeps every normal pointing out.
//
// The deletion always uses the precision epsilon, whatever tolerance the caller keeps points for, this being a question about geometry rather than detail.
// Leaving a face this point stands in front of makes the rim more than one cycle.
// The new triangles then cross the surviving ones and the solid stops being convex.
// Every point after that cuts the whole boundary and the cook never finishes.
void AddPoint(std::span<const float3> points, std::vector<Face> &faces, uint32_t point, float epsilon, std::vector<std::pair<uint32_t, uint32_t>> &rim) {
    rim.clear();
    for (Face &face : faces) {
        if (!face.Live || dot(face.Normal, points[point]) - face.Offset <= epsilon) continue;
        face.Live = false;
        for (uint32_t e = 0; e < 3; ++e) rim.emplace_back(face.Corner[e], face.Corner[(e + 1) % 3]);
    }
    for (const auto edge : rim) {
        const auto reverse = std::pair{edge.second, edge.first};
        if (std::ranges::find(rim, reverse) == rim.end()) faces.push_back(MakeFace(points, edge.first, edge.second, point));
    }
}

// The incremental hull, exact.
// One pass over the points suffices because the hull only grows, so a point inside the current hull is inside every later one.
std::vector<Face> BuildFaces(std::span<const float3> points, float epsilon) {
    std::vector<Face> faces = SeedTetrahedron(points, epsilon);
    if (faces.empty()) return {};
    std::vector<std::pair<uint32_t, uint32_t>> rim;
    for (uint32_t i = 0; i < points.size(); ++i)
        if (OutsideBy(faces, points[i]) > epsilon) AddPoint(points, faces, i, epsilon, rim);
    std::erase_if(faces, [](const Face &face) { return !face.Live; });
    return faces;
}

// The points the hull kept, in the order they were given in.
// A point the boundary never names is inside the solid.
std::vector<uint32_t> Corners(std::span<const Face> faces) {
    std::vector<uint32_t> corners;
    for (const Face &face : faces) {
        if (!face.Live) continue; // called on a hull still being built, where a replaced face is still in the list
        for (const uint32_t corner : face.Corner)
            if (std::ranges::find(corners, corner) == corners.end()) corners.push_back(corner);
    }
    std::ranges::sort(corners); // so the vertex order is the input order, whatever order the faces came out in
    return corners;
}

// The same builder under a corner budget, taking at every step the point standing furthest outside the current hull rather than the next one in the list.
// Furthest-first is Quickhull's own choice of candidate, and it matters here in a way it cannot for an exact hull.
// A point is measured against the hull at its own turn, so taking the points in order spends the whole budget wherever the list starts.
// A latitude-row sphere then comes out as one polar cap with a flat ball behind it.
// That is 0.168 of the radius in error, against 0.061 for the same budget spent furthest-first.
//
// `tolerance` returns how far outside the finished hull the worst input point stands, measured against that hull rather than tracked as it grew.
// A face plane is linear over the solid, so every point of the exact hull is within that tolerance too.
std::vector<Face> BuildSimplified(std::span<const float3> points, float epsilon, uint32_t limit, float &tolerance) {
    std::vector<Face> faces = SeedTetrahedron(points, epsilon);
    if (faces.empty()) return {};
    std::vector<std::pair<uint32_t, uint32_t>> rim;
    // A point adds itself and can only remove corners, so the count rises by at most one per step and stopping at the budget lands exactly on it.
    while (Corners(faces).size() < limit) {
        uint32_t furthest = NoIndex;
        float outside = epsilon; // under which a point is on the hull rather than outside it
        for (uint32_t i = 0; i < points.size(); ++i)
            if (const float by = OutsideBy(faces, points[i]); by > outside) { // ties go to the lower index, so the hull is deterministic
                outside = by;
                furthest = i;
            }
        if (furthest == NoIndex) break; // every point on or inside it, which is the exact hull
        AddPoint(points, faces, furthest, epsilon, rim);
    }
    tolerance = 0;
    for (const float3 point : points) tolerance = std::max(tolerance, OutsideBy(faces, point));
    if (tolerance <= epsilon) tolerance = 0; // the budget held every point, so the hull is exact
    std::erase_if(faces, [](const Face &face) { return !face.Live; });
    return faces;
}

// A group of the builder's triangles lying in one plane, as the loop of vertices bounding them.
struct Loop {
    std::vector<uint32_t> Corner;
    float3 Normal;
    float Offset;
};

// Directed edges, so the triangle across (a, b) is the one carrying (b, a).
// A closed triangulation carries every directed edge exactly once, so one owner per key suffices.
uint64_t EdgeKey(uint32_t from, uint32_t to) { return (uint64_t(from) << 32) | to; }

// The builder's triangles, merged back into the faces they were cut from.
// Two triangles belong to one face when they share an edge and the second lies in the first's plane to within `epsilon`.
// That is the distance the builder used to place a point on a face.
// A facet meeting a face at a hundredth of a degree lifts its corners well past `epsilon` and stays a face of its own.
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
        // A sliver carries no direction to seed a plane with, so it is absorbed by a neighbour it lies in the plane of, and dropped when there is none.
        if (taken[seed] || !has_area(seed)) continue;
        const float3 normal = faces[seed].Normal;
        // Measured from a corner of the seed rather than against its offset.
        // `dot(normal, p) - offset` is a difference of numbers the size of the points, and subtracting the points first cancels the offset exactly.
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

        // The boundary of the group: every directed edge of a member whose reverse no member carries.
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
        std::unordered_map<uint32_t, uint32_t> rim; // from -> to, which is one cycle when the boundary is simple
        bool simple = true;
        for (uint32_t i = 0; i < edges.size(); ++i)
            if (!cancelled[i]) simple = rim.emplace(edges[i].first, edges[i].second).second && simple;

        // Walked from the lowest vertex index, so a face is named by its geometry rather than by the order the builder produced its triangles in.
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
        // A group whose boundary is not one simple cycle reverts to its triangles, a face having to be a polygon.
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

// The rotation whose columns are these three axes, as a quaternion.
float4 QuatFromAxes(double3 x, double3 y, double3 z) {
    // Shepperd's method: branching on the largest diagonal term keeps the divisor away from zero, which the trace alone does not at a half turn.
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

// The eigendecomposition of a symmetric matrix: the eigenvectors are the frame the inertia is diagonal in, and the eigenvalues are that diagonal.
void Diagonalize(Mat3 a, Mat3 &vectors, double3 &values) {
    // Cyclic Jacobi, where a handful of sweeps reaches machine precision at this size.
    vectors = {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
    // Relative to the diagonal, because an off-diagonal term below the precision the diagonal is known to is rounding rather than coupling.
    // Rotating such a term away turns a symmetric solid to an arbitrary angle.
    const double scale = std::abs(a.M[0][0]) + std::abs(a.M[1][1]) + std::abs(a.M[2][2]);
    const double negligible = 1e-12 * scale;
    for (int sweep = 0; sweep < 16; ++sweep) {
        if (std::abs(a.M[0][1]) + std::abs(a.M[0][2]) + std::abs(a.M[1][2]) <= negligible) break;
        for (const auto [p, q] : {std::pair{0, 1}, std::pair{0, 2}, std::pair{1, 2}}) {
            if (std::abs(a.M[p][q]) <= negligible) continue;
            const double theta = (a.M[q][q] - a.M[p][p]) / (2 * a.M[p][q]);
            const double t = (theta >= 0 ? 1. : -1.) / (std::abs(theta) + std::sqrt(theta * theta + 1));
            const double c = 1 / std::sqrt(t * t + 1), s = t * c;
            for (int k = 0; k < 3; ++k) { // a <- J^T a J, one pair of rows then the matching columns
                const double ap = a.M[k][p], aq = a.M[k][q];
                a.M[k][p] = c * ap - s * aq;
                a.M[k][q] = s * ap + c * aq;
            }
            for (int k = 0; k < 3; ++k) {
                const double ap = a.M[p][k], aq = a.M[q][k];
                a.M[p][k] = c * ap - s * aq;
                a.M[q][k] = s * ap + c * aq;
            }
            for (int k = 0; k < 3; ++k) { // and the accumulated rotation with it
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
    // Right handed, so the frame is a rotation rather than a reflection.
    // An eigenvector's sign is arbitrary, and one of the eight sign choices is spent on this.
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
    // Relative, because a hull may be a millimetre across or a kilometre and neither is degenerate.
    // It is also never finer than the points themselves carry.
    // A face authored flat thirty units from the origin arrives flat to about a part in ten million of thirty.
    // A tighter epsilon splits that one face into two.
    // The second term is Jolt's DetermineCoplanarDistance, after Gregorius's Implementing Quickhull.
    // It is measured on the points as given, because their precision was decided there.
    // Without both terms, one solid handed in from two frames came out with two different face counts.
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

    // The exact hull is built first, and its corner count decides whether it is used.
    // Over budget, the same builder runs again under the limit, and everything below - faces, volume, inertia, frame - describes that simplified hull.
    // The rebuild is handed the exact hull's own corners rather than the points as given, so its tolerance is measured against the hull the caller asked for.
    float tolerance = 0;
    std::vector<float3> exact;
    if (const std::vector<uint32_t> corners = Corners(faces); corners.size() > MaxHullVertices) {
        for (const uint32_t corner : corners) exact.push_back(points[corner]);
        faces = BuildSimplified(exact, epsilon, MaxHullVertices, tolerance);
        if (faces.empty()) return {};
        points = exact;
    }

    // The divergence theorem over the boundary: each triangle with the origin makes a tetrahedron.
    // The signed volumes sum to the solid wherever the origin is.
    // The covariance integral of x x^T over one tetrahedron is the canonical matrix carried by its own edges, which is Blow and Binstock's exact form.
    constexpr Mat3 Canonical{{{2 / 120., 1 / 120., 1 / 120.}, {1 / 120., 2 / 120., 1 / 120.}, {1 / 120., 1 / 120., 2 / 120.}}};
    double volume = 0;
    double3 moment{0, 0, 0};
    Mat3 covariance;
    for (const Face &face : faces) {
        const double3 a = ToDouble(points[face.Corner[0]]), b = ToDouble(points[face.Corner[1]]), c = ToDouble(points[face.Corner[2]]);
        const double determinant = dot(a, cross(b, c)); // six times the tetrahedron's signed volume
        volume += determinant / 6;
        moment += determinant / 6 * (a + b + c) / 4;
        const Mat3 edges{{{a.x, b.x, c.x}, {a.y, b.y, c.y}, {a.z, b.z, c.z}}};
        const Mat3 tetrahedron = Multiply(Multiply(edges, Canonical), Transpose(edges));
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) covariance.M[i][j] += determinant * tetrahedron.M[i][j];
    }
    if (volume <= 0) return {};

    // Shifted onto the centre of mass, the parallel axis theorem in covariance form.
    const double3 center = moment / volume;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) covariance.M[i][j] -= volume * center[i] * center[j];

    const Diagonalized principal = DiagonalizeSymmetric(covariance.M);
    const double3 spread = principal.Values;
    const double3 x = principal.Axis[0], y = principal.Axis[1], z = principal.Axis[2];

    // A moment of inertia is the spread away from an axis, so each is the sum of the other two spreads.
    CookedHull cooked{.Volume = float(volume), .Inertia = {float(spread.y + spread.z), float(spread.x + spread.z), float(spread.x + spread.y)}, .Tolerance = tolerance, .Frame = {.Position = origin + float3{float(center.x), float(center.y), float(center.z)}, .Orientation = principal.Orientation}};
    // Only the points the hull kept, in the frame it was moved into.
    // A support search scans corners of the solid, and an interior point is never one.
    const std::vector<uint32_t> corners = Corners(faces);
    for (const uint32_t corner : corners) {
        const double3 offset = ToDouble(points[corner]) - center;
        cooked.Vertices.push_back(float3{float(dot(offset, x)), float(dot(offset, y)), float(dot(offset, z))});
    }

    // And the faces the builder cut up, merged back together and moved into the same frame.
    // A face's offset is re-measured from one of its own vertices rather than carried through the transform, the cook moving the origin as well as the axes.
    std::vector<uint32_t> renamed(points.size(), NoIndex);
    for (uint32_t i = 0; i < corners.size(); ++i) renamed[corners[i]] = i;
    for (const Loop &loop : MergeCoplanar(points, faces, epsilon)) {
        const double3 normal = ToDouble(loop.Normal);
        const float3 turned{float(dot(normal, x)), float(dot(normal, y)), float(dot(normal, z))};
        std::vector<uint32_t> rim;
        for (const uint32_t corner : loop.Corner) {
            if (renamed[corner] == NoIndex) { // a face of points the hull did not keep is not a face
                rim.clear();
                break;
            }
            rim.push_back(renamed[corner]);
        }
        if (rim.size() < 3) continue;

        // A face with more corners than MaxFacePoints is sampled around its rim rather than truncated at the first corners.
        // Truncation would leave a contiguous wedge off the body's centre.
        // The lowest index anchors the set, and each corner after it is the one furthest from the corners already kept.
        // The survivors are put back into the loop's own order, which keeps the result a convex polygon.
        if (rim.size() > MaxFacePoints) {
            std::vector<uint32_t> keep{0};
            while (keep.size() < MaxFacePoints) {
                uint32_t pick = 0;
                float widest = -1;
                for (uint32_t i = 0; i < rim.size(); ++i) {
                    if (std::ranges::find(keep, i) != keep.end()) continue;
                    float nearest = INFINITY;
                    for (const uint32_t k : keep) nearest = std::min(nearest, simd::distance_squared(cooked.Vertices[rim[i]], cooked.Vertices[rim[k]]));
                    if (nearest <= widest) continue; // ties go to the lower index, so the set is deterministic
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
    if (cooked.Faces.empty()) return {}; // no boundary recovered, so no face for a support query to name
    return cooked;
}

} // namespace rbp
