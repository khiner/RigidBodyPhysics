#ifndef MESH_SHAPES
#define MESH_SHAPES 1
#endif
#ifndef RECOMPUTE_QUERIES
#define RECOMPUTE_QUERIES 0
#endif
#ifndef PREPARE_QUERIES
#define PREPARE_QUERIES 0
#endif
#ifndef QUEUED_QUERIES
#define QUEUED_QUERIES 0
#endif
// Rigid-body AVBD; source attribution and algorithm references are in NOTICE.md.
// Normals point from B to A; penetration and the normal dual are negative.

#ifndef COLLECT_LANES
#define COLLECT_LANES 1
#endif
#ifndef BOUNDS_LANES
#define BOUNDS_LANES 1
#endif
#if COLLECT_LANES > 1
#define COLLECT_STORAGE threadgroup
#else
#define COLLECT_STORAGE thread
#endif

#ifndef BROAD_PHASE_MODE
#define BROAD_PHASE_MODE 0
#endif

#ifndef BOUNDED_PLANES
#define BOUNDED_PLANES 1
#endif
#ifndef MESH_PAIRS
#define MESH_PAIRS 1
#endif

#ifndef SOLVE_BODIES_PER_GROUP
#define SOLVE_BODIES_PER_GROUP 1
#endif
#ifndef COMPACT_BODY_WORK
#define COMPACT_BODY_WORK (SOLVE_BODIES_PER_GROUP > 1)
#endif
constant uint Dof = SolveDof;

static bool Asleep(uint quiet, constant StepParams &p) { return quiet >= p.SleepSteps; }

static bool Moving(Velocity v, constant StepParams &p) { return length(v.Linear) > p.SleepSpeed || length(v.Angular) > p.SleepSpeed; }

static bool Driven(BodyMass mass, Velocity v, constant StepParams &p) {
    return !Moves(mass) && Moving(v, p);
}

static bool Frozen(BodyMass mass, Velocity v, uint quiet, constant StepParams &p) {
    return !Moves(mass) ? !Driven(mass, v, p) : Asleep(quiet, p);
}

static bool Solved(BodyMass mass, uint quiet, constant StepParams &p) {
    return Moves(mass) && !Asleep(quiet, p);
}

// Convex leaves own mesh and plane pairs; meshes own plane pairs.
static bool ConvexLeaf(uint kind) { return kind != ShapePlane && kind != ShapeMesh; }
static bool BoundedPlane(Shape shape) { return BOUNDED_PLANES && IsBoundedPlane(shape); }
constant uint PlaneBackFeature = 1u << 31;

// Box2D-lite reference-face bias preserves contact identity.
constant float RelativeTolerance = 0.95f;
constant float AbsoluteTolerance = 0.01f;
// Absolute edge/face bias; an edge pair has no single half-extent scale.
constant float EdgeTolerance = 0.01f;

#ifndef STABILIZE
#define STABILIZE 0
#endif
// Hard rows retain C0 during stabilization and relax it during ordinary iterations.
#if STABILIZE
constant float ConstraintAlpha = 0;
constant float Stabilizing = 1;
#else
constant float ConstraintAlpha = 1;
constant float Stabilizing = 0;
#endif

// Pair inertial stiffness is reduced mass / h^2, including pinned and static endpoints.
static float PairStiffness(float inverse, float dt) { return inverse > 0 ? 1 / (inverse * dt * dt) : 0; }

static float3 UnitAxis(uint i) { return float3(i == 0 ? 1.f : 0.f, i == 1 ? 1.f : 0.f, i == 2 ? 1.f : 0.f); }

static float3x3 Diagonal(float3 d) {
    return float3x3(float3(d.x, 0, 0), float3(0, d.y, 0), float3(0, 0, d.z));
}

static float3x3 QuatToMatrix(float4 q) {
    return float3x3(Rotate(q, float3(1, 0, 0)), Rotate(q, float3(0, 1, 0)), Rotate(q, float3(0, 0, 1)));
}

static float3x3 WorldTensor(float3x3 rotation, float3 diagonal) {
    return rotation * Diagonal(diagonal) * transpose(rotation);
}

static float3x3 WorldInverseInertia(float4 orientation, float3 inverse_local) {
    return WorldTensor(QuatToMatrix(orientation), inverse_local);
}

// Compensated pairs retain inertial terms beside stiff constraint rows.
// Normalize each accumulation before factorization.
static float2 WideSum(float a, float b) {
#pragma clang fp reassociate(off) contract(off)
    const float sum = a + b;
    const float tail = sum - a;
    return float2(sum, (a - (sum - tail)) + (b - tail));
}
static float2 WideAdd(float2 a, float2 b) {
#pragma clang fp reassociate(off) contract(off)
    const float2 sum = WideSum(a.x, b.x);
    // Normalized input pairs permit Fast2Sum for the final normalization.
    const float error = sum.y + a.y + b.y;
    const float high = sum.x + error;
    return float2(high, error - (high - sum.x));
}
static float2 WideMul(float2 a, float2 b) {
#pragma clang fp reassociate(off) contract(off)
    const float product = a.x * b.x;
    const float error = fma(a.x, b.x, -product) + a.x * b.y + a.y * b.x + a.y * b.y;
    // Normalized input pairs keep the product larger than its correction.
    // Fast2Sum retains that correction with three operations instead of six.
    const float sum = product + error;
    return float2(sum, error - (sum - product));
}
static float2 WideDiv(float2 a, float2 b) {
#pragma clang fp reassociate(off) contract(off)
    const float quotient = a.x / b.x;
    const float2 remainder = WideAdd(a, -WideMul(b, float2(quotient, 0)));
    return WideSum(quotient, (remainder.x + remainder.y) / b.x);
}

// Six adjacent lanes own one body's rows; shuffles stay inside that tile.
static float SolveBlock(thread float2 H[Dof], float2 g, uint row, uint base) {
    uint coupled = 0;
    float2 diagonal_value = 0;
#pragma unroll
    for (uint j = 0; j < Dof; ++j) {
        if (j == row) diagonal_value = H[j];
        else if (any(H[j] != float2(0))) coupled = 1;
    }
    const uint coupled_rows = uint(ulong(simd_ballot(coupled != 0)));
    if (((coupled_rows >> base) & ((1u << Dof) - 1)) == 0) {
        const float2 result = WideDiv(-g, diagonal_value);
        return result.x + result.y;
    }
#pragma unroll
    for (uint j = 0; j < Dof; ++j) {
#pragma unroll
        for (uint k = 0; k < j; ++k) {
            const float2 column = simd_shuffle(H[k], base + j), diagonal = simd_shuffle(H[k], base + k);
            if (row >= j) H[j] = WideAdd(H[j], -WideMul(WideMul(H[k], column), diagonal));
        }
        const float2 pivot = simd_shuffle(H[j], base + j);
        if (row > j) H[j] = WideDiv(H[j], pivot);
        if (row == j) diagonal_value = pivot;
    }
    float2 y = -g;
#pragma unroll
    for (uint k = 0; k < Dof; ++k) {
        const float2 previous = simd_shuffle(y, base + k);
        if (row > k) y = WideAdd(y, -WideMul(H[k], previous));
    }
    y = WideDiv(y, diagonal_value);
#pragma unroll
    for (uint i = Dof; i-- > 0;) {
#pragma unroll
        for (uint k = i + 1; k < Dof; ++k) {
            const float2 column = simd_shuffle(H[i], base + k), previous = simd_shuffle(y, base + k);
            if (row == i) y = WideAdd(y, -WideMul(column, previous));
        }
    }
    return y.x + y.y;
}

// Gradient of dot(Log(Exp(delta) Exp(turn)), axis) at delta = 0.
static float3 LogGradient(float3 turn, float3 axis) {
    // Inverse left Jacobian transpose: https://borglab.github.io/gtsam/so3/
    const float squared = dot(turn, turn);
    float coefficient;
    if (squared < 0.01f) coefficient = 1.f / 12 + squared / 720 + squared * squared / 30240;
    else {
        const float angle = sqrt(squared);
        coefficient = (1 - 0.5f * angle * cos(0.5f * angle) / sin(0.5f * angle)) / squared;
    }
    return axis + 0.5f * cross(turn, axis) + coefficient * cross(turn, cross(turn, axis));
}

static void AddRow(
    thread float2 H[Dof], thread float2 &g, uint block_row, float3 axis, float3 angular, float side, float force, float stiffness
) {
    float row[Dof];
#pragma unroll
    for (uint k = 0; k < 3; ++k) {
        row[k] = side * axis[k];
        row[3 + k] = side * angular[k];
    }
    g = WideAdd(g, WideMul(float2(row[block_row], 0), float2(force, 0)));
#pragma unroll
    for (uint j = 0; j < Dof; ++j) H[j] = WideAdd(H[j], WideMul(WideMul(float2(stiffness, 0), float2(row[block_row], 0)), float2(row[j], 0)));
}

static void MergePreparedBlock(thread float2 H[Dof], thread float2 &g, thread const float2 partial[Dof], float2 partial_g, uint source) {
    g = WideAdd(g, simd_shuffle(partial_g, source));
#pragma unroll
    for (uint j = 0; j < Dof; ++j) H[j] = WideAdd(H[j], simd_shuffle(partial[j], source));
}

// Preserve the ordered projection sums in P H P + held a a^T and P g.
static void LockDirection(thread float2 H[Dof], thread float2 &g, uint row, uint base, uint first, float3 axis) {
    float a[Dof], held = 1;
    float2 product = 0, along = 0, gradient = 0;
#pragma unroll
    for (uint i = 0; i < Dof; ++i) a[i] = i >= first && i < first + 3 ? axis[i - first] : 0;
#pragma unroll
    for (uint j = 0; j < Dof; ++j) product = WideAdd(product, WideMul(H[j], float2(a[j], 0)));
#pragma unroll
    for (uint i = 0; i < Dof; ++i) {
        along = WideAdd(along, WideMul(float2(a[i], 0), simd_shuffle(product, base + i)));
        gradient = WideAdd(gradient, WideMul(float2(a[i], 0), simd_shuffle(g, base + i)));
        held = max(held, simd_shuffle(H[i].x, base + i));
    }
    g = WideAdd(g, -WideMul(float2(a[row], 0), gradient));
#pragma unroll
    for (uint j = 0; j < Dof; ++j) {
        const float2 term = WideAdd(WideMul(WideAdd(along, float2(held, 0)), float2(a[j], 0)), -simd_shuffle(product, base + j));
        H[j] = WideAdd(H[j], WideAdd(WideMul(float2(a[row], 0), term), -WideMul(product, float2(a[j], 0))));
    }
}

static Displacement Since(Pose now, Pose start) {
    return {now.Position - start.Position, RotationVector(QuatMul(now.Orientation, QuatConjugate(start.Orientation)))};
}

static Pose FreeFlight(Pose pose, Velocity v, float3 gravity, float share, float dt) {
    return {pose.Position + dt * v.Linear + (share * dt * dt) * gravity, normalize(QuatMul(QuatFromRotationVector(dt * v.Angular), pose.Orientation))};
}

static void IntegrateBody(
    device const Pose *poses, device Pose *initial, device Pose *inertial,
    device Velocity *velocities,
    device const BodyMass *masses, device Adjacency *incoming,
    device uint *quiet, constant StepParams &p,
    uint body
) {
    if (body >= p.BodyCount) return;
    incoming[body].Count = 0;
    const Pose pose = poses[body];
    initial[body] = pose;
    const BodyMass mass = masses[body];
    if (!Moves(mass)) {
        inertial[body] = pose;
        return;
    }

    Velocity v = velocities[body];
    // External velocity changes wake sleeping bodies.
    if (Asleep(quiet[body], p) && Moving(v, p)) quiet[body] = 0;
    if (Asleep(quiet[body], p)) {
        inertial[body] = pose;
        velocities[body] = {float3(0), float3(0)};
        return;
    }

    if (Translates(mass)) v.Linear *= max(0.f, 1 - mass.LinearDamping * p.DeltaTime);
    if (Turns(mass)) v.Angular *= max(0.f, 1 - mass.AngularDamping * p.DeltaTime);

    const float spin = length(v.Angular);
    if (spin > p.MaxAngularSpeed) v.Angular *= p.MaxAngularSpeed / spin;
    velocities[body] = v;

    inertial[body] = FreeFlight(pose, v, (Translates(mass) ? mass.GravityScale : 0) * p.Gravity, 1, p.DeltaTime);
}

// Warm-start guesses differ from inertial targets and use the previous acceleration.
static void WarmStartBody(
    device Pose *poses, device const Pose *initial,
    device const Velocity *velocities, device Velocity *previous,
    device const BodyMass *masses, device Displacement *displacements,
    device BodyIterate *iterates,
    constant StepParams &p,
    uint body
) {
    if (body >= p.BodyCount) return;
    const BodyMass mass = masses[body];
    const Pose pose = initial[body];
    const Velocity v = velocities[body];
    const float dt = p.DeltaTime;
    if (!Moves(mass)) {
        // Advance kinematic poses before solving so contact constraints see their imposed motion.
        if (Driven(mass, v, p)) poses[body] = FreeFlight(pose, v, float3(0), 0, dt);
        displacements[body] = Since(poses[body], pose);
        iterates[body] = {poses[body], displacements[body]};
        return;
    }
    const float3 pull = (Translates(mass) ? mass.GravityScale : 0) * p.Gravity;
    const float gravity = length(pull);
    const float3 fall = gravity > 1e-6f ? pull / gravity : float3(0);
    const float along = dot((v.Linear - previous[body].Linear) / dt, fall);
    const float weight = gravity > 1e-6f ? clamp(along / gravity, 0.f, 1.f) : 0.f;
    previous[body] = v;
    poses[body] = FreeFlight(pose, v, pull, weight, dt);
    displacements[body] = Since(poses[body], pose);
    iterates[body] = {poses[body], displacements[body]};
}

kernel void WarmStart(
    device Pose *poses [[buffer(0)]], device const Pose *initial [[buffer(1)]],
    device const Velocity *velocities [[buffer(3)]], device Velocity *previous [[buffer(9)]],
    device const BodyMass *masses [[buffer(4)]], device Displacement *displacements [[buffer(10)]],
    device BodyIterate *iterates [[buffer(11)]],
    device ColorWork &color_work [[buffer(26)]],
    constant StepParams &p [[buffer(7)]],
    uint body [[thread_position_in_grid]]
) {
    if (p.BodyCount > SolveLanes && body < MaxSupportedColors) color_work.Counts[body] = 0;
    if (p.BodyCount > SolveLanes && body == 0) {
        color_work.ColoringActive = 1;
        color_work.ColoringChanged = 0;
    }
    WarmStartBody(poses, initial, velocities, previous, masses, displacements, iterates, p, body);
}

struct BoxPose {
    float3 Center;
    float3 Axis[3];
    float3 Half;
};

static BoxPose MakeBox(Pose pose, Shape shape) {
    BoxPose box;
    box.Center = pose.Position;
    for (uint i = 0; i < 3; ++i) {
        box.Axis[i] = Rotate(pose.Orientation, UnitAxis(i));
    }
    box.Half = shape.HalfExtents;
    return box;
}

static float Reach(BoxPose box, float3 axis) {
    return abs(dot(box.Axis[0], axis)) * box.Half[0] + abs(dot(box.Axis[1], axis)) * box.Half[1] +
        abs(dot(box.Axis[2], axis)) * box.Half[2];
}

static float Overlap(BoxPose a, BoxPose b, float3 axis) {
    return Reach(a, axis) + Reach(b, axis) - abs(dot(b.Center - a.Center, axis));
}

static void FaceCorners(BoxPose box, uint axis, float side, thread float3 *out) {
    const uint u = (axis + 1) % 3, v = (axis + 2) % 3;
    const float3 centre = box.Center + box.Axis[axis] * (side * box.Half[axis]);
    const float3 du = box.Axis[u] * box.Half[u], dv = box.Axis[v] * box.Half[v];
    out[0] = centre - du - dv;
    out[1] = centre + du - dv;
    out[2] = centre + du + dv;
    out[3] = centre - du + dv;
}

struct Poly {
    float3 Center;
    float4 Orientation;
    float3 Half; // Box half extents or cylinder radius and half height use X/Y; capsule half-segment length uses Y.
    float Radius; // Spheres and capsules have a nonzero radius.
    uint First, Count; // Hull vertices and faces use contiguous ranges; triangle vertices use absolute indices.
    uint FirstFace, FaceCount;
    uint3 Corner;
    uint Kind;
};

static Poly MakePoly(Pose pose, Shape shape) {
    return {pose.Position, pose.Orientation, shape.HalfExtents, shape.Radius, shape.FirstVertex, shape.VertexCount, shape.FirstFace, shape.FaceCount, uint3(0), shape.Kind};
}

static Poly MakeTriangle(Pose pose, Triangle triangle) {
    return {pose.Position, pose.Orientation, float3(0), 0, 0, 0, 0, 0, uint3(triangle.A, triangle.B, triangle.C), ShapeMesh};
}

static Poly MeshBounds(Shape shape, Pose pose, device const BvhNode *nodes) {
    const BvhNode root = nodes[shape.RootNode];
    return {WorldPoint(pose, (root.Low + root.High) * 0.5f), pose.Orientation, (root.High - root.Low) * 0.5f, 0, 0, 0, 0, 0, uint3(0), ShapeBox};
}

static uint PolyCount(Poly poly) {
    if (poly.Kind == ShapeBox) return 8;
    if (BOUNDED_PLANES && poly.Kind == ShapePlane) return 4;
    if (poly.Kind == ShapeHull) return poly.Count;
    if (poly.Kind == ShapeMesh) return 3;
    return poly.Kind == ShapeCapsule ? 2 : 1;
}

static float3 LocalVertex(Poly poly, device const float3 *pool, uint i) {
    float3 local = float3(0);
    if (poly.Kind == ShapeBox) local = poly.Half * float3((i & 1) ? 1.f : -1.f, (i & 2) ? 1.f : -1.f, (i & 4) ? 1.f : -1.f);
    else if (BOUNDED_PLANES && poly.Kind == ShapePlane) local = poly.Half * float3(i < 2 ? -1.f : 1.f, 0, i == 1 || i == 2 ? 1.f : -1.f);
    else if (poly.Kind == ShapeHull) local = pool[poly.First + i];
    else if (poly.Kind == ShapeMesh) local = pool[poly.Corner[i]];
    else if (poly.Kind == ShapeCapsule) local = float3(0, i == 0 ? -poly.Half.y : poly.Half.y, 0);
    return local;
}

static float3 PolyVertex(Poly poly, device const float3 *pool, uint i) {
    return poly.Center + Rotate(poly.Orientation, LocalVertex(poly, pool, i));
}

// Return the support vertex index to keep contact identity tied to geometry.
static uint PolySupport(Poly poly, device const float3 *pool, float3 direction, uint lane = NoIndex) {
    const float3 local = Rotate(QuatConjugate(poly.Orientation), direction);
    if (poly.Kind == ShapeBox) return (local.x > 0 ? 1u : 0u) | (local.y > 0 ? 2u : 0u) | (local.z > 0 ? 4u : 0u);
    if (poly.Kind == ShapeCapsule) return local.y > 0 ? 1u : 0u;
    uint best = 0;
    float furthest = -INFINITY;
    for (uint i = lane == NoIndex ? 0 : lane, count = PolyCount(poly); i < count; i += lane == NoIndex ? 1 : 32) {
        const float reach = dot(LocalVertex(poly, pool, i), local);
        if (reach > furthest) {
            furthest = reach;
            best = i;
        }
    }
    if (lane != NoIndex) {
        const float most = simd_max(furthest);
        best = simd_min(furthest == most ? best : NoIndex);
    }
    return best;
}

static float3 PolySupportPoint(Poly poly, device const float3 *pool, float3 direction, uint lane = NoIndex) {
    if (poly.Kind == ShapeCylinder) {
        const float3 local = Rotate(QuatConjugate(poly.Orientation), direction);
        const float radial = length(local.xz);
        const float3 point = float3(radial > 0 ? poly.Half.x * local.x / radial : 0, local.y >= 0 ? poly.Half.y : -poly.Half.y, radial > 0 ? poly.Half.x * local.z / radial : 0);
        return poly.Center + Rotate(poly.Orientation, point);
    }
    return PolyVertex(poly, pool, PolySupport(poly, pool, direction, poly.Kind == ShapeHull ? lane : NoIndex));
}

// With lane != NoIndex, all 32 SIMD lanes project the same polytope into the same frame.
static void PolyBounds(Poly own, Pose pose, device const float3 *pool, thread float3 &low, thread float3 &high, uint lane = NoIndex) {
    if (own.Kind == ShapeCylinder) {
        const float3 center = LocalPoint(pose, own.Center);
        const float3 axis = Rotate(QuatConjugate(pose.Orientation), Rotate(own.Orientation, float3(0, 1, 0)));
        const float3 radial_x = Rotate(QuatConjugate(pose.Orientation), Rotate(own.Orientation, float3(1, 0, 0)));
        const float3 radial_z = Rotate(QuatConjugate(pose.Orientation), Rotate(own.Orientation, float3(0, 0, 1)));
        const float3 extent = own.Half.y * abs(axis) + own.Half.x * sqrt(radial_x * radial_x + radial_z * radial_z);
        low = center - extent;
        high = center + extent;
        return;
    }
    low = INFINITY;
    high = -INFINITY;
    for (uint i = lane == NoIndex ? 0 : lane; i < PolyCount(own); i += lane == NoIndex ? 1 : 32) {
        const float3 at = LocalPoint(pose, PolyVertex(own, pool, i));
        low = min(low, at);
        high = max(high, at);
    }
    if (lane != NoIndex)
        for (uint axis = 0; axis < 3; ++axis) {
            low[axis] = simd_min(low[axis]);
            high[axis] = simd_max(high[axis]);
        }
}

// Extend unbounded plane edges beyond the partner projection so only authored boundaries clip contacts.
static bool PlanePatch(Poly own, Pose pose, Shape plane, device const float3 *pool, float margin, thread Poly &patch, uint lane = NoIndex) {
    pose.Position += (plane.Offset - dot(plane.Normal, pose.Position)) * plane.Normal;
    float3 low, high;
    PolyBounds(own, pose, pool, low, high, lane);
    const float padding = own.Radius + margin + 1e-5f * max(1.f, max(length(low), length(high)));
    low -= padding;
    high += padding;
    for (uint side = 0; side < 2; ++side) {
        const uint axis = 2 * side;
        if (plane.HalfExtents[axis] <= 0) continue;
        low[axis] = max(low[axis], -plane.HalfExtents[axis]);
        high[axis] = min(high[axis], plane.HalfExtents[axis]);
        if (low[axis] >= high[axis]) return false;
    }
    low.y = high.y = 0;
    pose.Position = WorldPoint(pose, (low + high) * 0.5f);
    patch = MakePoly(pose, plane);
    patch.Half = (high - low) * 0.5f;
    if (dot(Rotate(pose.Orientation, float3(0, 1, 0)), plane.Normal) < 0)
        patch.Orientation = QuatMul(pose.Orientation, float4(1, 0, 0, 0));
    return true;
}

// Face-plane rejection excludes triangles whose bounds overlap the convex body.
static bool OutsideFaces(Poly convex, Poly triangle, device const float3 *pool, device const HullFace *faces, float margin, uint lane = NoIndex) {
    if (convex.Kind != ShapeHull && convex.Kind != ShapeBox && convex.Kind != ShapeCylinder) return false;
    margin += convex.Radius;
    float3 points[3];
    for (uint i = 0; i < 3; ++i)
        points[i] = Rotate(QuatConjugate(convex.Orientation), PolyVertex(triangle, pool, i) - convex.Center);
    if (convex.Kind == ShapeCylinder) {
        const float bottom = min(points[0].y, min(points[1].y, points[2].y));
        const float top = max(points[0].y, max(points[1].y, points[2].y));
        if (bottom > convex.Half.y + margin || top < -convex.Half.y - margin) return true;
        float nearest = INFINITY;
        bool positive = false, negative = false;
        for (uint i = 0; i < 3; ++i) {
            const float2 from = points[i].xz, to = points[(i + 1) % 3].xz;
            const float2 edge = to - from;
            const float square = dot(edge, edge);
            const float t = square > 0 ? clamp(-dot(from, edge) / square, 0.f, 1.f) : 0;
            const float2 point = from + t * edge;
            nearest = min(nearest, dot(point, point));
            const float side = from.x * to.y - from.y * to.x;
            positive |= side > 0;
            negative |= side < 0;
        }
        // Use edges when the projected triangle is degenerate.
        if (positive != negative) return false;
        const float radius = convex.Half.x + margin;
        return nearest > radius * radius;
    }
    if (convex.Kind == ShapeBox) {
        const float3 low = min(points[0], min(points[1], points[2]));
        const float3 high = max(points[0], max(points[1], points[2]));
        return any(low > convex.Half + margin) || any(high < -convex.Half - margin);
    }
    const uint width = lane == NoIndex ? 1u : 32u;
    for (uint base = 0; base < convex.FaceCount; base += width) {
        const uint i = base + (lane == NoIndex ? 0u : lane);
        bool outside = false;
        if (i < convex.FaceCount) {
            const HullFace face = faces[convex.FirstFace + i];
            const float nearest = min(dot(face.Normal, points[0]), min(dot(face.Normal, points[1]), dot(face.Normal, points[2])));
            outside = nearest > face.Offset + margin;
        }
        if (lane != NoIndex ? simd_any(outside) : outside) return true;
    }
    return false;
}

static float PolyReach(Poly poly, device const float3 *pool, uint lane = NoIndex) {
    if (poly.Kind == ShapeCylinder) return length(poly.Half.xy);
    if (poly.Kind == ShapeBox || (BOUNDED_PLANES && poly.Kind == ShapePlane)) return length(poly.Half);
    if (poly.Kind == ShapeCapsule) return poly.Half.y;

    if (poly.Kind == ShapeMesh)
        return max(max(distance(pool[poly.Corner[0]], pool[poly.Corner[1]]), distance(pool[poly.Corner[1]], pool[poly.Corner[2]])), distance(pool[poly.Corner[2]], pool[poly.Corner[0]]));
    if (poly.Kind != ShapeHull) return 0;
    float reach = 0;
    for (uint i = lane == NoIndex ? 0 : lane; i < poly.Count; i += lane == NoIndex ? 1 : 32) reach = max(reach, length(pool[poly.First + i]));
    if (lane != NoIndex) reach = simd_max(reach);
    return reach;
}

static float FaceTolerance(Poly poly, device const float3 *pool, uint lane = NoIndex) { return 1e-3f * PolyReach(poly, pool, lane) + 1e-6f; }

// Sphere and capsule surfaces are at Radius from a point or segment core.
struct Core {
    float3 From, To;
    float Radius;
};

static Core MakeCore(Pose pose, Shape shape) {
    const float3 along = Rotate(pose.Orientation, float3(0, shape.HalfExtents.y, 0));
    return {pose.Position - along, pose.Position + along, shape.Radius};
}

static bool IsRound(uint kind) { return kind == ShapeSphere || kind == ShapeCapsule; }
static bool CurvedSurface(uint kind, float4 orientation, float3 normal) {
    return IsRound(kind) || (kind == ShapeCylinder && abs(dot(Rotate(orientation, float3(0, 1, 0)), normal)) < 0.999999f);
}

static float3 ClosestOnBox(BoxPose box, float3 at, thread float &distance, thread uint &feature) {
    const float3 offset = at - box.Center;
    float3 local = float3(dot(offset, box.Axis[0]), dot(offset, box.Axis[1]), dot(offset, box.Axis[2]));
    const float3 clamped = clamp(local, -box.Half, box.Half);
    const float3 outside = local - clamped;
    if (dot(outside, outside) > 1e-14f) {
        feature = 0;
        for (uint i = 0; i < 3; ++i) feature |= (local[i] != clamped[i] ? (local[i] > 0 ? 1u : 2u) : 0u) << (2 * i);
        distance = length(outside);
        return box.Center + box.Axis[0] * clamped[0] + box.Axis[1] * clamped[1] + box.Axis[2] * clamped[2];
    }
    uint axis = 0;
    float least = INFINITY;
    for (uint i = 0; i < 3; ++i) {
        const float depth = box.Half[i] - abs(local[i]);
        if (depth < least) {
            least = depth;
            axis = i;
        }
    }
    distance = -least;
    feature = (local[axis] >= 0 ? 1u : 2u) << (2 * axis);
    local[axis] = local[axis] >= 0 ? box.Half[axis] : -box.Half[axis];
    return box.Center + box.Axis[0] * local[0] + box.Axis[1] * local[1] + box.Axis[2] * local[2];
}

static float3 OutOfBox(float3 at, float3 nearest, float away) {
    return away < 0 ? normalize(nearest - at) : (away > 1e-9f ? (at - nearest) / away : float3(0, 1, 0));
}

static void SupportEdge(BoxPose box, uint axis, float3 direction, thread float3 *ends, thread uint &which) {
    const uint u = (axis + 1) % 3, v = (axis + 2) % 3;
    const float su = dot(direction, box.Axis[u]) >= 0 ? 1.f : -1.f;
    const float sv = dot(direction, box.Axis[v]) >= 0 ? 1.f : -1.f;
    const float3 centre = box.Center + box.Axis[u] * (box.Half[u] * su) + box.Axis[v] * (box.Half[v] * sv);
    ends[0] = centre - box.Axis[axis] * box.Half[axis];
    ends[1] = centre + box.Axis[axis] * box.Half[axis];
    which = (su > 0 ? 2u : 0u) | (sv > 0 ? 1u : 0u);
}

static float3 ClosestOnSegment(float3 from, float3 to, float3 at) {
    const float3 along = to - from;
    const float length_squared = dot(along, along);
    return length_squared < 1e-12f ? from : from + along * clamp(dot(at - from, along) / length_squared, 0.f, 1.f);
}

static void ClosestOnSegments(float3 p0, float3 p1, float3 q0, float3 q1, thread float3 &on_p, thread float3 &on_q) {
    const float3 d1 = p1 - p0, d2 = q1 - q0, r = p0 - q0;
    const float a = dot(d1, d1), e = dot(d2, d2), f = dot(d2, r);
    float s = 0, t = 0;
    if (a > 1e-12f || e > 1e-12f) {
        if (a <= 1e-12f) {
            t = clamp(f / e, 0.f, 1.f);
        } else {
            const float c = dot(d1, r);
            if (e <= 1e-12f) {
                s = clamp(-c / a, 0.f, 1.f);
            } else {
                const float b = dot(d1, d2), denom = a * e - b * b;
                if (abs(denom) > 1e-12f) s = clamp((b * f - c * e) / denom, 0.f, 1.f);
                t = (b * s + f) / e;
                if (t < 0) {
                    t = 0;
                    s = clamp(-c / a, 0.f, 1.f);
                } else if (t > 1) {
                    t = 1;
                    s = clamp((b - c) / a, 0.f, 1.f);
                }
            }
        }
    }
    on_p = p0 + d1 * s;
    on_q = q0 + d2 * t;
}

static uint ClipAgainst(
    thread float3 *poly, thread uint *names, uint count, float3 normal, float offset, uint plane,
    float tolerance, uint limit, thread float3 *out, thread uint *out_names
) {
    // Sutherland-Hodgman clipping with geometry-derived point identities.
    uint kept = 0;
    for (uint i = 0; i < count && kept < limit; ++i) {
        const uint next = (i + 1) % count;
        const float3 from = poly[i], to = poly[next];
        const float in_from = dot(normal, from) - offset, in_to = dot(normal, to) - offset;
        const bool from_inside = in_from <= tolerance, to_inside = in_to <= tolerance;
        if (from_inside) {
            out[kept] = from;
            out_names[kept] = names[i];
            ++kept;
        }
        if (from_inside == to_inside || kept == limit) continue;
        // Interior cuts receive new identities; endpoint cuts retain the endpoint identity.
        const float at = in_from / (in_from - in_to);
        if (at <= 1e-4f || at >= 1 - 1e-4f) continue;
        out[kept] = from + (to - from) * at;
        out_names[kept] = (names[i] & names[next]) | plane;
        ++kept;
    }
    for (uint i = 0; i < kept; ++i) {
        poly[i] = out[i];
        names[i] = out_names[i];
    }
    return kept;
}

constant uint MaxClipPoints = 2 * MaxFacePoints;

constant uint MaxEpaVertices = 16;

struct Mink {
    float3 At;
    uint IndexA, IndexB;
};

static Mink MinkSupport(Poly a, Poly b, device const float3 *pool, float3 direction, uint lane) {
    if (a.Kind == ShapeCylinder || b.Kind == ShapeCylinder) {
        const float3 pa = a.Kind == ShapeCylinder ? PolySupportPoint(a, pool, direction) : PolyVertex(a, pool, PolySupport(a, pool, direction, lane));
        const float3 pb = b.Kind == ShapeCylinder ? PolySupportPoint(b, pool, -direction) : PolyVertex(b, pool, PolySupport(b, pool, -direction, lane));
        return {pa - pb, 0, 0};
    }
    const uint ia = PolySupport(a, pool, direction, lane), ib = PolySupport(b, pool, -direction, lane);
    return {PolyVertex(a, pool, ia) - PolyVertex(b, pool, ib), ia, ib};
}

static bool SameSign(float a, float b) { return (a > 0 && b > 0) || (a < 0 && b < 0); }

static float Signed4(float3 a, float3 b, float3 c, float3 d) { return dot(b - a, cross(c - a, d - a)); }

static float Signed2(float3 a, float3 b, float3 c, uint u, uint v) {
    return (b[u] - a[u]) * (c[v] - a[v]) - (b[v] - a[v]) * (c[u] - a[u]);
}

static float3 SignedVolume1(float3 s1, float3 s2, thread uint &mask) {
    const float3 along = s2 - s1;
    const float3 projected = s2 - along * (dot(s2, along) / dot(along, along));
    // Use the largest segment coordinate for numerical conditioning.
    uint axis = 0;
    float span = 0;
    for (uint i = 0; i < 3; ++i) {
        if (abs(s1[i] - s2[i]) <= abs(span)) continue;
        span = s1[i] - s2[i];
        axis = i;
    }
    const float first = projected[axis] - s2[axis], second = s1[axis] - projected[axis];
    if (SameSign(span, first) && SameSign(span, second)) {
        mask = 3;
        return (s1 * first + s2 * second) / span;
    }
    mask = 1; // The origin projects beyond the newest vertex.
    return s1;
}

static float3 SignedVolume2(float3 s1, float3 s2, float3 s3, thread uint &mask) {
    const float3 turn = cross(s2 - s1, s3 - s1);
    // Degenerate triangles can produce a nonfinite projection.
    const float3 projected = turn * (dot(s1, turn) / dot(turn, turn));
    // Project onto the largest-area Cartesian plane in cyclic order to preserve orientation.
    uint u = 1, v = 2;
    float area = 0;
    for (uint i = 0; i < 3; ++i) {
        const uint k = (i + 1) % 3, l = (i + 2) % 3;
        const float shaded = Signed2(s1, s2, s3, k, l);
        if (abs(shaded) <= abs(area)) continue;
        area = shaded;
        u = k;
        v = l;
    }
    const float first = Signed2(projected, s2, s3, u, v);
    const float second = Signed2(s1, projected, s3, u, v);
    const float third = Signed2(s1, s2, projected, u, v);
    if (SameSign(area, first) && SameSign(area, second) && SameSign(area, third)) {
        mask = 7;
        return (s1 * first + s2 * second + s3 * third) / area;
    }
    float nearest = INFINITY;
    float3 closest = s1;
    mask = 1;
    for (uint j = 1; j < 3; ++j) {
        if (!SameSign(area, -(j == 1 ? second : third))) continue;
        uint sub;
        const float3 point = SignedVolume1(s1, j == 1 ? s3 : s2, sub);
        if (dot(point, point) >= nearest) continue;
        nearest = dot(point, point);
        closest = point;
        mask = (sub & 1) | (((sub >> 1) & 1) << (j == 1 ? 2 : 1));
    }
    return closest;
}

static float3 SignedVolume3(float3 s1, float3 s2, float3 s3, float3 s4, thread uint &mask, thread bool &inside) {
    const float volume = Signed4(s1, s2, s3, s4);
    const float first = Signed4(float3(0), s2, s3, s4), second = Signed4(s1, float3(0), s3, s4);
    const float third = Signed4(s1, s2, float3(0), s4), fourth = Signed4(s1, s2, s3, float3(0));
    if (SameSign(volume, first) && SameSign(volume, second) && SameSign(volume, third) && SameSign(volume, fourth)) {
        inside = true;
        mask = 15;
        return float3(0);
    }
    inside = false;
    float nearest = INFINITY;
    float3 closest = s1;
    mask = 1;
    for (uint j = 1; j < 4; ++j) {
        if (!SameSign(volume, -(j == 1 ? second : (j == 2 ? third : fourth)))) continue;
        uint sub;
        const float3 point = SignedVolume2(s1, j == 1 ? s3 : s2, j == 3 ? s3 : s4, sub);
        if (dot(point, point) >= nearest) continue;
        nearest = dot(point, point);
        closest = point;
        const uint from[3] = {0, j == 1 ? 2u : 1u, j == 3 ? 2u : 3u};
        mask = 0;
        for (uint b = 0; b < 3; ++b) mask |= ((sub >> b) & 1) << from[b];
    }
    return closest;
}

static float3 ReduceSimplex(thread Mink *simplex, thread uint &count, thread bool &inside) {
    // Signed-volume simplex reduction (Montanari, Petrinic and Barbieri, 2017).
    inside = false;
    uint mask = 1;
    float3 closest = simplex[0].At;
    if (count == 2) closest = SignedVolume1(simplex[0].At, simplex[1].At, mask);
    else if (count == 3) closest = SignedVolume2(simplex[0].At, simplex[1].At, simplex[2].At, mask);
    else if (count == 4) closest = SignedVolume3(simplex[0].At, simplex[1].At, simplex[2].At, simplex[3].At, mask, inside);
    if (inside) return closest;
    Mink kept[4];
    uint n = 0;
    for (uint i = 0; i < count; ++i)
        if (mask & (1u << i)) kept[n++] = simplex[i];
    for (uint i = 0; i < n; ++i) simplex[i] = kept[i];
    count = n;
    return closest;
}

static bool Gjk(
    Poly a, Poly b, device const float3 *pool, thread Mink *simplex, thread uint &count,
    thread float3 &direction, thread float &distance, uint lane
) {
    float3 search = a.Center - b.Center;
    if (dot(search, search) < 1e-12f) search = float3(1, 0, 0);
    direction = normalize(search);
    distance = 0;
    simplex[0] = MinkSupport(a, b, pool, search, lane);
    count = 1;
    float3 closest = simplex[0].At;
    for (uint iteration = 0; iteration < 32; ++iteration) {
        const float squared = dot(closest, closest);
        // Equation 9 tests whether the simplex contains the origin.
        float scale = 0;
        for (uint i = 0; i < count; ++i) scale = max(scale, dot(simplex[i].At, simplex[i].At));
        if (squared <= 1e-10f * scale) return false;
        const Mink next = MinkSupport(a, b, pool, -closest, lane);
        // Equation 10 bounds further progress toward the origin.
        bool repeated = false;
        for (uint i = 0; i < count; ++i)
            repeated = repeated || ((a.Kind == ShapeCylinder || b.Kind == ShapeCylinder) ? all(simplex[i].At == next.At) : (simplex[i].IndexA == next.IndexA && simplex[i].IndexB == next.IndexB));
        if (repeated || squared - dot(next.At, closest) <= 1e-8f * squared) break;
        for (uint i = count; i > 0; --i) simplex[i] = simplex[i - 1];
        simplex[0] = next;
        ++count;
        bool inside = false;
        const float3 reduced = ReduceSimplex(simplex, count, inside);
        if (inside) return true;
        // The exact distance cannot increase when a support point is added.
        // Keep the previous closest point if reduction fails to improve it, including NaN.
        if (!(dot(reduced, reduced) < squared)) break;
        closest = reduced;
    }
    distance = length(closest);
    if (distance > 1e-9f) direction = closest / distance;
    return false;
}

// Compensated orientation preserves EPA topology when curved support points become nearly coplanar.
static float3 EpaTurn(float3 a, float3 b, float3 c) {
    float2 u[3], v[3];
    for (uint i = 0; i < 3; ++i) {
        u[i] = WideSum(b[i], -a[i]);
        v[i] = WideSum(c[i], -a[i]);
    }
    float3 result;
    for (uint i = 0; i < 3; ++i) {
        uint j = (i + 1) % 3, k = (i + 2) % 3;
        float2 n = WideAdd(WideMul(u[j], v[k]), -WideMul(u[k], v[j]));
        result[i] = n.x + n.y;
    }
    return result;
}
static float EpaSide(float3 a, float3 b, float3 c, float3 d) {
    const float3 u0 = b - a, v0 = c - a, w0 = d - a;
    const float det = dot(w0, cross(u0, v0));
    const float bound = dot(abs(w0), abs(u0.yzx * v0.zxy) + abs(u0.zxy * v0.yzx));
    if (abs(det) > 1e-5f * bound) return det;
    float2 u[3], v[3], w[3];
    for (uint i = 0; i < 3; ++i) {
        u[i] = WideSum(b[i], -a[i]);
        v[i] = WideSum(c[i], -a[i]);
        w[i] = WideSum(d[i], -a[i]);
    }
    float2 sum = 0;
    for (uint i = 0; i < 3; ++i) {
        uint j = (i + 1) % 3, k = (i + 2) % 3;
        sum = WideAdd(sum, WideMul(w[i], WideAdd(WideMul(u[j], v[k]), -WideMul(u[k], v[j]))));
    }
    return sum.x + sum.y;
}

static bool PushFace(
    thread const Mink *vertices, thread uint (*faces)[3], thread float3 *planes, thread float *offsets,
    thread uint &face_count, uint i, uint j, uint k, float3 turn
) {
    const float area = length(turn);
    if (area < 1e-18f) return false;
    faces[face_count][0] = i;
    faces[face_count][1] = j;
    faces[face_count][2] = k;
    planes[face_count] = turn / area;
    offsets[face_count] = dot(planes[face_count], vertices[i].At);
    ++face_count;
    return true;
}

template<uint Capacity = MaxEpaVertices, bool Curved = false>
static bool Epa(Poly a, Poly b, device const float3 *pool, thread Mink *simplex, thread float3 &normal, thread float &depth, uint lane) {
    Mink vertices[Capacity];
    uint faces[(2 * Capacity)][3];
    float3 planes[(2 * Capacity)];
    float offsets[(2 * Capacity)];
    uint vertex_count = 4, face_count = 0;
    for (uint i = 0; i < 4; ++i) vertices[i] = simplex[i];

    const uint corners[4][3] = {{0, 1, 2}, {0, 2, 3}, {0, 3, 1}, {1, 3, 2}};
    const uint apexes[4] = {3, 1, 2, 0};
    for (uint f = 0; f < 4; ++f) {
        uint i = corners[f][0], j = corners[f][1], k = corners[f][2];
        float3 turn = Curved ? EpaTurn(vertices[i].At, vertices[j].At, vertices[k].At) : cross(vertices[j].At - vertices[i].At, vertices[k].At - vertices[i].At);
        if ((Curved ? EpaSide(vertices[i].At, vertices[j].At, vertices[k].At, vertices[apexes[f]].At) : dot(turn, vertices[apexes[f]].At - vertices[i].At)) > 0) {
            const uint swap = j;
            j = k;
            k = swap;
            turn = -turn;
        }
        // EPA requires a nonzero-volume simplex.
        if (!PushFace(vertices, faces, planes, offsets, face_count, i, j, k, turn)) return false;
    }

    for (uint iteration = 0; iteration < (Curved ? Capacity : 24); ++iteration) {
        uint best = 0;
        float least = INFINITY;
        for (uint f = 0; f < face_count; ++f) {
            if (offsets[f] >= least) continue;
            least = offsets[f];
            best = f;
        }
        normal = planes[best];
        depth = max(least, 0.f);
        // Return the nearest face when expansion reaches capacity.
        if (vertex_count == Capacity || face_count + 8 > (2 * Capacity)) return true;
        const Mink next = MinkSupport(a, b, pool, normal, lane);
        if (dot(next.At, normal) - least < 1e-6f) return true;

        uint2 rim[(2 * Capacity)];
        uint rim_count = 0, kept = 0;
        for (uint f = 0; f < face_count; ++f) {
            if (Curved ? EpaSide(vertices[faces[f][0]].At, vertices[faces[f][1]].At, vertices[faces[f][2]].At, next.At) > 0 : dot(planes[f], next.At) - offsets[f] > 1e-9f) {
                for (uint e = 0; e < 3; ++e) {
                    const uint2 edge = uint2(faces[f][e], faces[f][(e + 1) % 3]);
                    bool cancelled = false;
                    for (uint r = 0; r < rim_count && !cancelled; ++r) {
                        if (rim[r].x != edge.y || rim[r].y != edge.x) continue;
                        rim[r] = rim[--rim_count];
                        cancelled = true;
                    }
                    if (!cancelled && rim_count < (2 * Capacity)) rim[rim_count++] = edge;
                }
                continue;
            }
            faces[kept][0] = faces[f][0];
            faces[kept][1] = faces[f][1];
            faces[kept][2] = faces[f][2];
            planes[kept] = planes[f];
            offsets[kept] = offsets[f];
            ++kept;
        }
        face_count = kept;
        const uint apex = vertex_count;
        vertices[vertex_count++] = next;
        for (uint r = 0; r < rim_count && face_count < (2 * Capacity); ++r) {
            const uint i = rim[r].x, j = rim[r].y;
            // Discard slivers with an undefined normal.
            PushFace(vertices, faces, planes, offsets, face_count, i, j, apex, Curved ? EpaTurn(vertices[i].At, vertices[j].At, vertices[apex].At) : cross(vertices[j].At - vertices[i].At, vertices[apex].At - vertices[i].At));
        }
        if (face_count == 0) return false;
    }
    return true;
}

// Keep up to limit support vertices at or beyond threshold, named by vertex index.
static uint SpreadSupport(
    Poly poly, device const float3 *pool, float3 axis, float threshold, uint limit, thread float3 *out, thread uint *names
) {
    const uint count = PolyCount(poly);
    float deepest = -INFINITY;
    uint qualify = 0;
    for (uint i = 0; i < count; ++i) {
        const float along = dot(PolyVertex(poly, pool, i), axis);
        deepest = max(deepest, along);
        qualify += along >= threshold ? 1 : 0;
    }

    // Retain extremal support points in stable order.
    const float presented = deepest - FaceTolerance(poly, pool);
    if (qualify > limit && presented > threshold) {
        threshold = presented;
        qualify = 0;
        for (uint i = 0; i < count; ++i) qualify += dot(PolyVertex(poly, pool, i), axis) >= threshold ? 1 : 0;
    }

    uint found = 0;
    if (qualify <= limit) {
        for (uint i = 0; i < count; ++i) {
            const float3 at = PolyVertex(poly, pool, i);
            if (dot(at, axis) < threshold) continue;
            out[found] = at;
            names[found] = i;
            ++found;
        }
        return found;
    }
    for (uint i = 0; i < count && found == 0; ++i) {
        const float3 at = PolyVertex(poly, pool, i);
        if (dot(at, axis) < threshold) continue;
        out[0] = at;
        names[0] = i;
        found = 1;
    }
    while (found < limit) {
        uint pick = 0;
        float widest = -1;
        for (uint i = 0; i < count; ++i) {
            const float3 at = PolyVertex(poly, pool, i);
            if (dot(at, axis) < threshold) continue;
            float nearest = INFINITY;
            for (uint k = 0; k < found; ++k) nearest = min(nearest, distance_squared(at, out[k]));
            if (nearest <= widest) continue;
            widest = nearest;
            pick = i;
        }
        out[found] = PolyVertex(poly, pool, pick);
        names[found] = pick;
        ++found;
    }
    return found;
}

// Return the support face with outward winding and stable vertex identities.
static uint SupportFace(
    Poly poly, device const float3 *pool, device const HullFace *hull_faces, float3 direction,
    thread float3 *out, thread uint *names, thread float3 &plane, uint lane = NoIndex, float3 towards = float3(0)
) {
    const float3 local = normalize(Rotate(QuatConjugate(poly.Orientation), direction));
    if (poly.Kind == ShapeCylinder) {
        const float radial = length(local.xz);
        const float tolerance = FaceTolerance(poly, pool);
        const float2 along = radial > 0 ? local.xz / radial : float2(1, 0);
        const bool side = radial > abs(local.y);
        const bool top = local.y >= 0;
        const float2 rim[8] = {float2(1, 0), float2(0.7071067812f, 0.7071067812f), float2(0, 1), float2(-0.7071067812f, 0.7071067812f), float2(-1, 0), float2(-0.7071067812f, -0.7071067812f), float2(0, -1), float2(0.7071067812f, -0.7071067812f)};
        const float2 target = Rotate(QuatConjugate(poly.Orientation), towards).xz;
        const float2 basis = radial * poly.Half.x > tolerance ? along : length(target) > 1e-9f ? normalize(target) :
                                                                                                 float2(1, 0);
        const float support = radial * poly.Half.x + abs(local.y) * poly.Half.y;
        uint found = 0;
        for (uint i = 0; i < (side ? 2u : 8u); ++i) {
            float3 point;
            if (side) point = float3(poly.Half.x * along.x, i == 0 ? -poly.Half.y : poly.Half.y, poly.Half.x * along.y);
            else {
                const float2 r = rim[top ? (8 - i) % 8 : i];
                const float2 at = poly.Half.x * (basis * r.x + float2(-basis.y, basis.x) * r.y);
                point = float3(at.x, top ? poly.Half.y : -poly.Half.y, at.y);
            }
            if (dot(point, local) < support - tolerance) continue;
            out[found] = poly.Center + Rotate(poly.Orientation, point);
            names[found++] = side ? 16 + i : (top ? 8 : 0) + i;
        }
        plane = Rotate(poly.Orientation, side ? float3(along.x, 0, along.y) : float3(0, top ? 1.f : -1.f, 0));
        return found;
    }
    float3 local_plane = local;
    uint corners[MaxFacePoints], found = 0;

    if (poly.Kind == ShapeHull) {
        uint best = 0;
        float most = -INFINITY;
        for (uint f = lane == NoIndex ? 0 : lane; f < poly.FaceCount; f += lane == NoIndex ? 1 : 32) {
            const float along = dot(hull_faces[poly.FirstFace + f].Normal, local);
            if (along <= most) continue;
            most = along;
            best = f;
        }
        if (lane != NoIndex) {
            const float greatest = simd_max(most);
            best = simd_min(most == greatest ? best : NoIndex);
        }
        if (poly.FaceCount == 0) return 0;
        const HullFace face = hull_faces[poly.FirstFace + best];
        local_plane = face.Normal;
        for (uint i = 0; i < face.Count && i < MaxFacePoints; ++i) corners[found++] = face.Corner[i];
    } else if (poly.Kind == ShapeBox) {
        const float3 magnitude = abs(local);
        const uint axis = magnitude.x >= magnitude.y && magnitude.x >= magnitude.z ? 0u : (magnitude.y >= magnitude.z ? 1u : 2u);
        const bool positive = local[axis] > 0;
        local_plane = float3(0);
        local_plane[axis] = positive ? 1 : -1;
        const uint across = (axis + 1) % 3, along = (axis + 2) % 3;
        const uint2 order[4] = {uint2(0, 0), uint2(1, 0), uint2(1, 1), uint2(0, 1)};
        for (uint i = 0; i < 4; ++i) {
            const uint2 corner = order[positive ? i : 3 - i];
            corners[found++] = (positive ? (1u << axis) : 0u) | (corner.x << across) | (corner.y << along);
        }
    } else if (BOUNDED_PLANES && poly.Kind == ShapePlane) {
        const bool forwards = local.y >= 0;
        local_plane = float3(0, forwards ? 1.f : -1.f, 0);
        for (uint i = 0; i < 4; ++i) corners[found++] = forwards ? i : 3 - i;
    } else if (poly.Kind == ShapeMesh) {
        const float3 a = pool[poly.Corner[0]], b = pool[poly.Corner[1]], c = pool[poly.Corner[2]];
        const float3 turn = cross(b - a, c - a);
        if (length(turn) < 1e-18f) return 0;
        const bool forwards = dot(turn, local) >= 0;
        local_plane = forwards ? normalize(turn) : -normalize(turn);
        for (uint i = 0; i < 3; ++i) corners[found++] = forwards ? i : 2 - i;
    } else {
        if (poly.Kind == ShapeCapsule && 2 * poly.Half.y * abs(local.y) > FaceTolerance(poly, pool)) corners[found++] = local.y > 0 ? 1 : 0;
        else
            for (uint i = 0; i < PolyCount(poly); ++i) corners[found++] = i;
    }

    if (poly.Kind == ShapeHull || poly.Kind == ShapeBox || (BOUNDED_PLANES && poly.Kind == ShapePlane)) {
        float furthest = -INFINITY;
        for (uint i = 0; i < found; ++i) furthest = max(furthest, dot(LocalVertex(poly, pool, corners[i]), local));
        const float tolerance = FaceTolerance(poly, pool, lane);
        uint kept = 0;
        for (uint i = 0; i < found; ++i)
            if (dot(LocalVertex(poly, pool, corners[i]), local) >= furthest - tolerance) corners[kept++] = corners[i];
        found = kept;
    }

    for (uint i = 0; i < found; ++i) {
        out[i] = PolyVertex(poly, pool, corners[i]);
        names[i] = corners[i];
    }
    plane = Rotate(poly.Orientation, local_plane);

    // Start at the lowest feature name to preserve cyclic identity.
    uint first = 0;
    for (uint i = 1; i < found; ++i)
        if (names[i] < names[first]) first = i;
    if (first > 0) {
        float3 turned[MaxFacePoints];
        uint turned_names[MaxFacePoints];
        for (uint i = 0; i < found; ++i) {
            turned[i] = out[(i + first) % found];
            turned_names[i] = names[(i + first) % found];
        }
        for (uint i = 0; i < found; ++i) {
            out[i] = turned[i];
            names[i] = turned_names[i];
        }
    }
    return found;
}

static bool SidePlane(thread const float3 *face, uint count, uint e, float3 plane, thread float3 &unit) {
    const float3 side = cross(face[(e + 1) % count] - face[e], plane);
    const float span = length(side);
    if (span < 1e-12f) return false;
    unit = side / span;
    return true;
}

static void AppendClipPoint(thread float3 *points, thread uint *names, thread uint &count, float3 point, uint name) {
    for (uint i = 0; i < count; ++i)
        if (distance_squared(points[i], point) < 1e-12f) return;
    if (count == MaxClipPoints) return;
    points[count] = point;
    names[count++] = name;
}

constant float BuriedAlignment = 0.999f;

static bool BuriedAlong(Shape shape, Pose pose, device const HullFace *hull_faces, float3 direction) {
    const uint internal = InternalFaces(shape);
    if (internal == 0) return false;
    const float3 local = normalize(Rotate(QuatConjugate(ComposePose(pose, shape.Local).Orientation), direction));
    if (shape.Kind == ShapeCylinder)
        return ((internal & 1u) != 0 && -local.y > BuriedAlignment) || ((internal & 2u) != 0 && local.y > BuriedAlignment);
    if (shape.Kind == ShapeBox) {
        for (uint face = 0; face < 6; ++face) {
            if ((internal & (1u << face)) == 0) continue;
            const uint axis = face >> 1;
            if (((face & 1) != 0 ? local[axis] : -local[axis]) > BuriedAlignment) return true;
        }
        return false;
    }
    for (uint f = 0; f < shape.FaceCount && f < MaxInternalFaces; ++f)
        if ((internal & (1u << f)) != 0 && dot(hull_faces[shape.FirstFace + f].Normal, local) > BuriedAlignment) return true;
    return false;
}

static bool ConvexSeparation(Poly a, Poly b, device const float3 *pool, thread float3 &axis, thread float &distance, uint lane) {
    if (a.Kind == ShapeCylinder || b.Kind == ShapeCylinder) {
        const Poly cylinder = a.Kind == ShapeCylinder ? a : b;
        const Poly other = a.Kind == ShapeCylinder ? b : a;
        const float3 along = Rotate(cylinder.Orientation, float3(0, 1, 0));
        const float3 other_axis = Rotate(other.Orientation, float3(0, 1, 0));
        if (other.Kind == ShapeSphere || ((other.Kind == ShapeCylinder || other.Kind == ShapeCapsule) && length(cross(along, other_axis)) <= 1e-6f)) {
            // Parallel cylinders (including a point or segment core) have a cylindrical Minkowski difference.
            const float3 delta = a.Center - b.Center;
            const float height = dot(delta, along);
            const float3 radial = delta - height * along;
            const float span = length(radial);
            const float3 radial_axis = span > 1e-9f ? radial / span : Rotate(cylinder.Orientation, float3(1, 0, 0));
            const float3 cap_axis = height >= 0 ? along : -along;
            const float side = span - cylinder.Half.x - (other.Kind == ShapeCylinder ? other.Half.x : 0);
            const float cap = abs(height) - cylinder.Half.y - (other.Kind == ShapeSphere ? 0 : other.Half.y);
            if (side > 0 && cap > 0) {
                distance = length(float2(side, cap));
                axis = (side * radial_axis + cap * cap_axis) / distance;
            } else {
                distance = max(side, cap);
                axis = side >= cap ? radial_axis : cap_axis;
            }
            return true;
        }
    }
    Mink simplex[4];
    uint simplex_count = 0;
    const bool inside = Gjk(a, b, pool, simplex, simplex_count, axis, distance, lane);
    const bool cylinder = a.Kind == ShapeCylinder || b.Kind == ShapeCylinder;
    if (inside || (cylinder && distance == 0)) {
        if (cylinder && simplex_count < 4) {
            // A line or triangle through the origin can lie inside a solid Minkowski difference.
            // Start a full-dimensional hull; EPA expands its outward faces until it encloses the origin.
            const float3 seed[4] = {float3(1, 1, 1), float3(-1, -1, 1), float3(-1, 1, -1), float3(1, -1, -1)};
            const float4 frame = a.Kind == ShapeCylinder ? a.Orientation : b.Orientation;
            for (uint i = 0; i < 4; ++i) simplex[i] = MinkSupport(a, b, pool, Rotate(frame, seed[i]), lane);
            simplex_count = 4;
        }
        float depth;
        float3 out_of_a;
        if (simplex_count < 4 || !(cylinder ? Epa<64, true>(a, b, pool, simplex, out_of_a, depth, lane) : Epa<>(a, b, pool, simplex, out_of_a, depth, lane))) return false;

        axis = -out_of_a;
        distance = -depth;
    }
    return true;
}

// Return up to four contacts with normals from B to A and geometry-derived identities.
static uint ConvexManifold(
    Poly a, Poly b, device const float3 *pool, device const HullFace *hull_faces, float margin, float3 known, thread float3 *here,
    thread float3 *there, thread uint *names, thread float3 &normal, uint lane = NoIndex
) {
    float3 axis;
    float distance;
    const bool told = any(known != 0);
    if (told) {
        axis = known;
        const float3 near_a = PolySupportPoint(a, pool, -axis, lane);
        const float3 near_b = PolySupportPoint(b, pool, axis, lane);
        distance = dot(near_a - near_b, axis);
    } else if (!ConvexSeparation(a, b, pool, axis, distance, lane)) return 0;
    const float gap = distance - a.Radius - b.Radius;
    if (gap >= margin) return 0;

    float3 face_a[MaxFacePoints], face_b[MaxFacePoints];
    uint name_a[MaxFacePoints], name_b[MaxFacePoints];
    float3 plane_a, plane_b;
    const uint count_a = SupportFace(a, pool, hull_faces, -axis, face_a, name_a, plane_a, lane, b.Center - a.Center);
    const uint count_b = SupportFace(b, pool, hull_faces, axis, face_b, name_b, plane_b, lane, a.Center - b.Center);

    if (count_a < 3 && count_b < 3) {
        float3 on_a, on_b;
        ClosestOnSegments(face_a[0], face_a[count_a - 1], face_b[0], face_b[count_b - 1], on_a, on_b);
        normal = axis;
        here[0] = on_a - a.Radius * axis;
        there[0] = on_b + b.Radius * axis;
        names[0] = (1u << 29) | name_a[0] | (name_a[count_a - 1] << 6) | (name_b[0] << 12) | (name_b[count_b - 1] << 18);
        return 1;
    }

    const bool reference_is_a = count_a >= 3 && (told || count_b < 3 || abs(dot(plane_a, axis)) >= abs(dot(plane_b, axis)));
    const float3 reference_plane = reference_is_a ? plane_a : plane_b;
    normal = reference_is_a ? -reference_plane : reference_plane;
    thread float3 *reference = reference_is_a ? face_a : face_b;
    thread float3 *incident = reference_is_a ? face_b : face_a;
    thread uint *reference_names = reference_is_a ? name_a : name_b;
    thread uint *incident_names = reference_is_a ? name_b : name_a;
    const uint reference_count = reference_is_a ? count_a : count_b;
    const uint incident_count = reference_is_a ? count_b : count_a;

    float3 poly[MaxClipPoints], clipped[MaxClipPoints];
    uint poly_names[MaxClipPoints], clipped_names[MaxClipPoints];
    uint poly_count = incident_count;
    for (uint i = 0; i < incident_count; ++i) {
        poly[i] = incident[i];

        poly_names[i] = incident_count >= 3 ? ((1u << i) | (1u << ((i + incident_count - 1) % incident_count))) : (1u << i);
    }

    float scale = 1;
    for (uint i = 0; i < reference_count; ++i) scale = max(scale, length(reference[i]));
    const float tolerance = 1e-5f * scale;
    const Poly reference_shape = reference_is_a ? a : b;
    const Poly incident_shape = reference_is_a ? b : a;
    if (reference_shape.Kind == ShapeCylinder) {
        // Clip against the full cylinder disk to retain contacts between rim samples.
        const float4 inverse = QuatConjugate(reference_shape.Orientation);
        const float radius_squared = reference_shape.Half.x * reference_shape.Half.x;
        poly_count = 0;
        for (uint i = 0; i < incident_count; ++i) {
            const float2 from = Rotate(inverse, incident[i] - reference_shape.Center).xz;
            if (dot(from, from) <= radius_squared)
                AppendClipPoint(poly, poly_names, poly_count, incident[i], 1u << i);
            if (incident_count == 1 || (incident_count == 2 && i == 1)) continue;
            const uint next = (i + 1) % incident_count;
            const float3 edge = incident[next] - incident[i];
            const float2 delta = Rotate(inverse, edge).xz;
            const float aa = dot(delta, delta), bb = dot(from, delta), cc = dot(from, from) - radius_squared;
            const float discriminant = bb * bb - aa * cc;
            if (aa <= 1e-20f || discriminant < 0) continue;
            const float root = sqrt(discriminant);
            for (uint end = 0; end < 2; ++end) {
                const float t = (-bb + (end ? root : -root)) / aa;
                if (t < 0 || t > 1) continue;
                AppendClipPoint(poly, poly_names, poly_count, incident[i] + t * edge, (1u << i) | (1u << (8 + end)));
            }
        }
        if (incident_count >= 3) {
            const float3 incident_plane = reference_is_a ? plane_b : plane_a;
            const float projection = dot(incident_plane, reference_plane);
            if (abs(projection) > 1e-6f) {
                for (uint i = 0; i < reference_count; ++i) {
                    const float3 point = reference[i] + reference_plane * (dot(incident_plane, incident[0] - reference[i]) / projection);
                    bool inside = true;
                    if (incident_shape.Kind == ShapeCylinder) {
                        const float2 radial = Rotate(QuatConjugate(incident_shape.Orientation), point - incident_shape.Center).xz;
                        inside = dot(radial, radial) <= incident_shape.Half.x * incident_shape.Half.x;
                    } else {
                        for (uint e = 0; e < incident_count && inside; ++e) {
                            float3 unit;
                            if (SidePlane(incident, incident_count, e, incident_plane, unit) && dot(unit, point - incident[e]) > tolerance) inside = false;
                        }
                    }
                    if (inside) AppendClipPoint(poly, poly_names, poly_count, point, 1u << (8 + i));
                }
            }
        }
    } else if (incident_count >= 3) {
        for (uint e = 0; e < reference_count; ++e) {
            float3 unit;
            if (!SidePlane(reference, reference_count, e, reference_plane, unit)) continue;
            poly_count = ClipAgainst(poly, poly_names, poly_count, unit, dot(unit, reference[e]), 1u << (8 + e), tolerance, MaxClipPoints, clipped, clipped_names);
        }
    } else if (incident_count == 1) {
        // Reject a lone incident point outside a side plane; polygon clipping would retain it unconditionally.
        for (uint e = 0; e < reference_count; ++e) {
            float3 unit;
            if (!SidePlane(reference, reference_count, e, reference_plane, unit)) continue;
            if (dot(unit, poly[0] - reference[e]) > tolerance) poly_count = 0;
        }
    } else if (incident_count == 2) {
        // Interval clipping emits each segment intersection once.
        const float3 from = poly[0], along = poly[1] - poly[0];
        float low = 0, high = 1;
        uint low_name = 1u << 0, high_name = 1u << 1;
        for (uint e = 0; e < reference_count; ++e) {
            float3 unit;
            if (!SidePlane(reference, reference_count, e, reference_plane, unit)) continue;
            const float offset = dot(unit, reference[e]);
            const float at_from = dot(unit, from) - offset, at_to = dot(unit, poly[1]) - offset;
            const float slope = at_to - at_from;
            if (abs(slope) < 1e-12f) {
                if (at_from > tolerance) poly_count = 0;
                continue;
            }
            const float crossing = -at_from / slope;
            if (slope > 0 && crossing < high) {
                high = crossing;
                high_name = (1u << 1) | (1u << (8 + e));
            } else if (slope < 0 && crossing > low) {
                low = crossing;
                low_name = (1u << 0) | (1u << (8 + e));
            }
        }
        if (poly_count > 0 && low <= high) {
            poly[0] = from + along * low;
            poly[1] = from + along * high;
            poly_names[0] = low_name;
            poly_names[1] = high_name;

            poly_count = high - low > 1e-6f ? 2 : 1;
        } else {
            poly_count = 0;
        }
    }

    uint reference_face = 63, incident_face = 63;
    for (uint i = 0; i < reference_count; ++i) reference_face = min(reference_face, reference_names[i]);
    for (uint i = 0; i < incident_count; ++i) incident_face = min(incident_face, incident_names[i]);
    const float reference_offset = dot(reference_plane, reference[0]);
    const float reference_radius = reference_is_a ? a.Radius : b.Radius;
    const float incident_radius = reference_is_a ? b.Radius : a.Radius;

    uint found = 0;
    for (uint i = 0; i < poly_count && found < MaxClipPoints; ++i) {
        const float stand_off = dot(reference_plane, poly[i]) - reference_offset;
        if (stand_off - reference_radius - incident_radius >= margin) continue;
        const float3 on_reference = poly[i] - (stand_off - reference_radius) * reference_plane;
        const float3 on_incident = poly[i] - incident_radius * reference_plane;
        here[found] = reference_is_a ? on_reference : on_incident;
        there[found] = reference_is_a ? on_incident : on_reference;
        names[found] = poly_names[i] | (reference_face << 16) | (incident_face << 22) | ((reference_is_a ? 0u : 1u) << 28);
        ++found;
    }
    return found;
}

// Clip both triangle footprints and choose the face requiring the smaller correction.
static uint MeshManifold(
    Poly a, Poly b, bool double_a, bool double_b, float3 center_a, float3 center_b, uint previous,
    device const float3 *pool, device const HullFace *faces, float margin,
    thread float3 *here, thread float3 *there, thread uint *names, thread float3 &normal
) {
    const float3 origin = a.Center;
    a.Center -= origin;
    b.Center -= origin;
    center_a -= origin;
    center_b -= origin;
    float3 av[3], bv[3], ae[3], be[3];
    for (uint i = 0; i < 3; ++i) {
        av[i] = PolyVertex(a, pool, i);
        bv[i] = PolyVertex(b, pool, i);
    }
    for (uint i = 0; i < 3; ++i) {
        ae[i] = av[(i + 1) % 3] - av[i];
        be[i] = bv[(i + 1) % 3] - bv[i];
    }
    // Reject separated triangles before clipping potentially overlapping half-spaces.
    for (uint axis_index = 0; axis_index < 11; ++axis_index) {
        const float3 axis = axis_index == 0 ? cross(ae[0], ae[1]) :
                                              (axis_index == 1 ? cross(be[0], be[1]) : cross(ae[(axis_index - 2) / 3], be[(axis_index - 2) % 3]));
        const float span = length(axis);
        if (span < 1e-18f) continue;
        float al = INFINITY, ah = -INFINITY, bl = INFINITY, bh = -INFINITY;
        for (uint i = 0; i < 3; ++i) {
            const float pa = dot(axis, av[i]), pb = dot(axis, bv[i]);
            al = min(al, pa);
            ah = max(ah, pa);
            bl = min(bl, pb);
            bh = max(bh, pb);
        }
        if (max(bl - ah, al - bh) > margin * span + 1e-6f * max(max(abs(al), abs(ah)), max(abs(bl), abs(bh)))) return 0;
    }
    uint kept = 0, side_bits = 0;
    float best = -INFINITY;
    for (uint side = 0; side < 2; ++side) {
        const Poly reference = side == 0 ? b : a, incident = side == 0 ? a : b;
        const float3 first = PolyVertex(reference, pool, 0);
        const float3 turn = cross(PolyVertex(reference, pool, 1) - first, PolyVertex(reference, pool, 2) - first);
        if (length(turn) < 1e-18f) return 0;
        float3 outward = normalize(turn);
        const float signed_center = dot(outward, (side == 0 ? center_a : center_b) - first);
        const uint back_bit = side == 0 ? (1u << 31) : (1u << 29);
        bool back = (side == 0 ? double_b : double_a) && signed_center < 0;
        if ((side == 0 ? double_b : double_a) && previous != NoIndex && abs(signed_center) <= margin)
            back = (previous & back_bit) != 0;
        if (back) {
            outward = -outward;
            side_bits |= back_bit;
        }
        float3 on_reference[MaxClipPoints], on_incident[MaxClipPoints], axis;
        uint features[MaxClipPoints];
        const uint found = ConvexManifold(reference, incident, pool, faces, margin, -outward, on_reference, on_incident, features, axis);
        if (found == 0) return 0;
        float deepest = INFINITY;
        for (uint i = 0; i < found; ++i) deepest = min(deepest, dot(outward, on_incident[i] - on_reference[i]));
        const bool prefer_previous = previous != NoIndex && (previous & (1u << 30)) != 0 && abs(deepest - best) <= 1e-6f;
        if (side != 0 && deepest <= best + 1e-6f && !prefer_previous) continue;
        kept = found;
        best = deepest;
        normal = side == 0 ? outward : -outward;
        for (uint i = 0; i < found; ++i) {
            here[i] = (side == 0 ? on_incident[i] : on_reference[i]) + origin;
            there[i] = (side == 0 ? on_reference[i] : on_incident[i]) + origin;
            names[i] = features[i] | (side << 30);
        }
    }
    const float3 face_a = normalize(cross(ae[0], ae[1])), face_b = normalize(cross(be[0], be[1]));
    const float tolerance = 1e-5f * max(PolyReach(a, pool), PolyReach(b, pool)) + 1e-9f;
    // An edge axis can require less separation than either face normal.
    for (uint ea = 0; ea < 3; ++ea) {
        for (uint eb = 0; eb < 3; ++eb) {
            float3 axis = cross(ae[ea], be[eb]);
            const float span = length(axis);
            if (span < 1e-18f) continue;
            axis /= span;
            if (abs(dot(axis, face_a)) > 0.99999f || abs(dot(axis, face_b)) > 0.99999f) continue;
            if (dot(axis, center_a - center_b) < 0) axis = -axis;
            float low = INFINITY, high = -INFINITY;
            for (uint i = 0; i < 3; ++i) {
                low = min(low, dot(axis, av[i]));
                high = max(high, dot(axis, bv[i]));
            }
            const float separation = low - high;
            if (separation <= best + tolerance) continue;
            uint ca[3], cb[3], na = 0, nb = 0;
            for (uint i = 0; i < 3; ++i) {
                if (dot(axis, av[i]) <= low + tolerance) ca[na++] = i;
                if (dot(axis, bv[i]) >= high - tolerance) cb[nb++] = i;
            }
            float3 on_a, on_b;
            ClosestOnSegments(av[ca[0]], av[ca[na - 1]], bv[cb[0]], bv[cb[nb - 1]], on_a, on_b);
            kept = 1;
            best = separation;
            normal = axis;
            here[0] = on_a + origin;
            there[0] = on_b + origin;
            names[0] = (1u << 28) | ca[0] | (ca[na - 1] << 2) | (cb[0] << 4) | (cb[nb - 1] << 6);
        }
    }
    for (uint i = 0; i < kept; ++i) names[i] |= side_bits;
    return kept;
}

// Weld coincident features into one constraint while preserving distinct contact geometry.
static uint WeldManifold(thread float3 *here, thread float3 *there, thread uint *names, uint found, float tolerance) {
    uint kept = 0;
    for (uint i = 0; i < found; ++i) {
        bool twice = false;
        for (uint j = 0; j < kept && !twice; ++j)
            twice = distance(here[i], here[j]) <= tolerance && distance(there[i], there[j]) <= tolerance;
        if (twice) continue;
        here[kept] = here[i];
        there[kept] = there[i];
        names[kept] = names[i];
        ++kept;
    }
    return kept;
}

// Gift wrapping includes boundary points from an unordered projected patch.
// Stationary patches use their diameter as extent.
static float2 ManifoldGeometry(thread const float3 *points, uint count, float3 normal, Pose a, Pose b, Velocity va, Velocity vb) {
    if (count < 2) return float2(0);
    float3 center = float3(0);
    for (uint i = 0; i < count; ++i) center += points[i];
    center /= count;
    const float3 relative = va.Linear + cross(va.Angular, center - a.Position) - (vb.Linear + cross(vb.Angular, center - b.Position));
    const float3 slip = relative - normal * dot(relative, normal);
    const ContactBasis basis = MakeContactBasis(normal);
    float2 projected[MaxClipPoints];
    uint first = 0;
    for (uint i = 0; i < count; ++i) {
        const float3 offset = points[i] - points[0];
        projected[i] = float2(dot(offset, basis.Axis[1]), dot(offset, basis.Axis[2]));
        if (projected[i].x < projected[first].x || (projected[i].x == projected[first].x && projected[i].y < projected[first].y)) first = i;
    }
    const float speed = length(slip);
    const float2 direction = speed > 1e-6f ? float2(dot(slip, basis.Axis[1]), dot(slip, basis.Axis[2])) / speed : float2(0);
    float extent = 0, twice_area = 0;
    uint at = first;
    for (uint edge = 0; edge < count; ++edge) {
        uint next = (at + 1) % count;
        for (uint i = 0; i < count; ++i) {
            const float2 a = projected[next] - projected[at], b = projected[i] - projected[at];

            extent = max(extent, speed > 1e-6f ? abs(dot(b, direction)) : length(b));
            const float turn = a.x * b.y - a.y * b.x;
            if (turn < 0 || (turn == 0 && dot(b, b) > dot(a, a))) next = i;
        }
        const float2 a = projected[at], b = projected[next];
        twice_area += a.x * b.y - a.y * b.x;
        at = next;
        if (all(projected[at] == projected[first])) break;
    }
    return float2(0.5f * abs(twice_area), extent);
}

static uint ReduceManifold(thread float3 *here, thread float3 *there, thread uint *names, uint found, float3 normal) {
    // Gregorius manifold reduction: deepest point, furthest point, largest triangle, then greatest added area.
    if (found <= ManifoldPoints) return found;
    // Break ties with feature identity so rounding of world coordinates does not rename contacts.
    float extent = 0;
    for (uint i = 0; i < found; ++i) extent = max(extent, distance(here[i], here[0]));
    const float slack = 1e-5f * extent + 1e-9f, area_slack = 1e-5f * extent * extent + 1e-9f;

    uint keep[4];

    float deepest = INFINITY;
    keep[0] = 0;
    for (uint i = 0; i < found; ++i) {
        const float separation = dot(normal, here[i] - there[i]);
        if (separation >= deepest - slack) continue;
        deepest = separation;
        keep[0] = i;
    }

    float furthest = -1;
    keep[1] = keep[0];
    for (uint i = 0; i < found; ++i) {
        const float3 span = here[i] - here[keep[0]];
        if (dot(span, span) <= furthest + area_slack) continue;
        furthest = dot(span, span);
        keep[1] = i;
    }
    float widest = 0;
    keep[2] = keep[0];
    for (uint i = 0; i < found; ++i) {
        const float area = dot(cross(here[keep[1]] - here[keep[0]], here[i] - here[keep[0]]), normal);
        if (abs(area) <= abs(widest) + area_slack) continue;
        widest = area;
        keep[2] = i;
    }
    if (widest == 0) return 4;
    if (widest < 0) {
        const uint swap = keep[1];
        keep[1] = keep[2];
        keep[2] = swap;
    }
    float best = 0;
    keep[3] = keep[0];
    for (uint i = 0; i < found; ++i) {
        for (uint e = 0; e < 3; ++e) {
            const float3 from = here[keep[e]], to = here[keep[(e + 1) % 3]];
            const float area = dot(cross(to - from, here[i] - from), normal);
            if (area >= best - area_slack) continue;
            best = area;
            keep[3] = i;
        }
    }

    uint kept = 0;
    for (uint i = 0; i < found; ++i) {
        bool wanted = false;
        for (uint k = 0; k < 4; ++k) wanted = wanted || keep[k] == i;
        if (!wanted) continue;
        here[kept] = here[i];
        there[kept] = there[i];
        names[kept] = names[i];
        ++kept;
    }
    return kept;
}

// Batch capacity bounds temporary storage; traversal resumes until all candidates are visited.
constant uint MaxMeshTriangles = COLLECT_LANES > 32 ? COLLECT_LANES : 32;
constant uint MeshStackDepth = 32;

static bool OutsideNode(Poly convex, BvhNode node, float3 offset, float3x3 rotation, device const HullFace *faces, float margin, uint lane) {
    if (convex.Kind != ShapeHull && convex.Kind != ShapeBox && (!BOUNDED_PLANES || convex.Kind != ShapePlane)) return false;
    const float3 radii = (node.High - node.Low) * 0.5f;
    const float3 node_center = (node.High + node.Low) * 0.5f;
    const float3 center = offset + rotation * node_center;
    // Roundoff can grow when large mesh and body offsets cancel.
    margin += convex.Radius + 1e-5f * max(1.f, length(offset) + length(node_center) + length(radii));
    if (convex.Kind == ShapeBox || (BOUNDED_PLANES && convex.Kind == ShapePlane)) {
        const float3 extent = abs(rotation[0]) * radii.x + abs(rotation[1]) * radii.y + abs(rotation[2]) * radii.z;
        if (convex.Kind == ShapePlane)
            return abs(center.x) > convex.Half.x + extent.x + margin || abs(center.z) > convex.Half.z + extent.z + margin || center.y > extent.y + margin;
        return any(abs(center) > convex.Half + extent + margin);
    }
    // Repeat the rejection vote in each SIMD group to keep threadgroup traversal uniform.
    constexpr uint face_lanes = COLLECT_LANES > 32 ? 32 : COLLECT_LANES;
    for (uint base = 0; base < convex.FaceCount; base += face_lanes) {
        const uint i = base + lane % face_lanes;
        bool outside = false;
        if (i < convex.FaceCount) {
            const HullFace face = faces[convex.FirstFace + i];
            const float3 projected = float3(dot(face.Normal, rotation[0]), dot(face.Normal, rotation[1]), dot(face.Normal, rotation[2]));
            const float nearest = dot(face.Normal, center) - dot(abs(projected), radii);
            outside = nearest > face.Offset + margin;
        }
#if COLLECT_LANES > 1
        if (simd_any(outside)) return true;
#else
        if (outside) return true;
#endif
    }
    return false;
}

// Return the next triangle batch overlapping bounds in the mesh frame.
// The caller retains traversal state between calls.

static uint GatherTriangles(
    Shape mesh, float3 low, float3 high, Poly convex, float3 offset, float3x3 rotation, device const HullFace *faces, float margin,
    device const BvhNode *nodes, thread uint *stack, thread uint &depth, thread uint *out, uint lane
) {
    uint found = 0;
    while (depth > 0) {
        const uint at = stack[--depth];
        const BvhNode node = nodes[mesh.RootNode + at];
        if (any(node.High < low) || any(node.Low > high)) continue;
        if (node.Count == 0 && OutsideNode(convex, node, offset, rotation, faces, margin, lane)) continue;
        if (node.Count > 0) {
            // Defer a complete leaf when the remaining batch capacity is insufficient.
            if (found + node.Count > MaxMeshTriangles) {
                stack[depth++] = at;
                return found;
            }
            for (uint i = 0; i < node.Count; ++i) out[found++] = node.First + i;
            continue;
        }
        if (depth + 2 > MeshStackDepth) continue;
        stack[depth++] = at + 1;
        stack[depth++] = node.First;
    }
    return found;
}

static bool TriangleBoundsOverlap(
    Shape a, Shape b, uint ai, uint bi, float3 offset, float3x3 rotation, float margin,
    device const Triangle *triangles, device const float3 *vertices
) {
    const Triangle ta = triangles[a.FirstTriangle + ai], tb = triangles[b.FirstTriangle + bi];
    const float3 al = min(vertices[ta.A], min(vertices[ta.B], vertices[ta.C]));
    const float3 ah = max(vertices[ta.A], max(vertices[ta.B], vertices[ta.C]));
    const float3 bl = min(vertices[tb.A], min(vertices[tb.B], vertices[tb.C]));
    const float3 bh = max(vertices[tb.A], max(vertices[tb.B], vertices[tb.C]));
    const float3 ca = (al + ah) * 0.5f, cb = (bl + bh) * 0.5f;
    const float3 ra = (ah - al) * 0.5f, rb = (bh - bl) * 0.5f;
    const float3 d = offset + rotation * ca - cb;
    const float guard = margin + 1e-5f * max(1.f, length(offset) + length(ca) + length(cb) + length(ra) + length(rb));
    const float3 ea = abs(rotation[0]) * ra.x + abs(rotation[1]) * ra.y + abs(rotation[2]) * ra.z;
    if (any(abs(d) > ea + rb + guard)) return false;
    const float3x3 inverse = transpose(rotation);
    const float3 eb = abs(inverse[0]) * rb.x + abs(inverse[1]) * rb.y + abs(inverse[2]) * rb.z;
    return !any(abs(inverse * d) > eb + ra + guard);
}

// Walk both trees, then refine overlapping leaves without changing candidate order.
static uint GatherMeshPairs(
    Shape a, Shape b, float3 offset, float3x3 rotation, float margin,
    device const BvhNode *nodes, device const Triangle *triangles, device const float3 *vertices, bool cached, uint lane, thread uint2 *stack, thread uint &depth, thread uint2 *out
) {
    uint found = 0;
    while (depth > 0) {
        const uint2 at = stack[--depth];
        const BvhNode na = nodes[a.RootNode + at.x], nb = nodes[b.RootNode + at.y];
        const float3 ca = (na.Low + na.High) * 0.5f, cb = (nb.Low + nb.High) * 0.5f;
        const float3 ra = (na.High - na.Low) * 0.5f, rb = (nb.High - nb.Low) * 0.5f;
        const float3 distance = offset + rotation * ca - cb;
        const float allowance = margin + 1e-5f * max(1.f, length(offset) + length(ca) + length(cb) + length(ra) + length(rb));
        const float3 extent = abs(rotation[0]) * ra.x + abs(rotation[1]) * ra.y + abs(rotation[2]) * ra.z;
        if (any(abs(distance) > extent + rb + allowance)) continue;
        const float3x3 inverse = transpose(rotation);
        const float3 other_extent = abs(inverse[0]) * rb.x + abs(inverse[1]) * rb.y + abs(inverse[2]) * rb.z;
        if (any(abs(inverse * distance) > other_extent + ra + allowance)) continue;
        if (na.Count > 0 && nb.Count > 0) {
            if (found + na.Count * nb.Count > MaxMeshTriangles) {
                stack[depth++] = at;
                return found;
            }
            // At most four triangles per leaf fit their Cartesian pairs in one SIMD mask.
            const uint pairs = na.Count * nb.Count;
            uint present = (1u << pairs) - 1;
            // Empty visited queries update patch metadata when reporting starts on sleeping contacts.
            if (!cached) {
#if COLLECT_LANES >= 32
                const uint pair = lane % 32;
                bool keep = false;
                if (pair < pairs) keep = TriangleBoundsOverlap(a, b, na.First + pair / nb.Count, nb.First + pair % nb.Count, offset, rotation, margin, triangles, vertices);
                present = uint(ulong(simd_ballot(keep)));
#else
                present = 0;
                for (uint pair = 0; pair < pairs; ++pair)
                    if (TriangleBoundsOverlap(a, b, na.First + pair / nb.Count, nb.First + pair % nb.Count, offset, rotation, margin, triangles, vertices)) present |= 1u << pair;
#endif
            }
            while (present) {
                const uint pair = ctz(present);
                present &= present - 1;
                out[found++] = uint2(na.First + pair / nb.Count, nb.First + pair % nb.Count);
            }
            continue;
        }
        if (depth + 2 > 2 * MeshStackDepth) continue;
        if (na.Count == 0 && (nb.Count > 0 || dot(ra, ra) >= dot(rb, rb))) {
            stack[depth++] = uint2(at.x + 1, at.y);
            stack[depth++] = uint2(na.First, at.y);
        } else {
            stack[depth++] = uint2(at.x, at.y + 1);
            stack[depth++] = uint2(at.x, nb.First);
        }
    }
    return found;
}

// Return whether the point lies on the edge line within clipping tolerance.
static bool OnEdgeLine(Poly face, device const float3 *pool, float3 outward, uint e, float3 at_point, float seam) {
    const float3 at = PolyVertex(face, pool, e);
    const float3 side = cross(PolyVertex(face, pool, (e + 1) % 3) - at, outward);
    const float span = length(side);
    return span > 1e-12f && abs(dot(side / span, at_point - at)) <= seam;
}

// Report prior contacts that were unclaimed during collection.
static void EndUnclaimed(
    device ContactEvent *events, device uint *counts, uint body, ulong claimed, uint reported,
    COLLECT_STORAGE const uint *was_feature, COLLECT_STORAGE const Index *was_other, COLLECT_STORAGE const Index *was_sub, COLLECT_STORAGE const Index *was_sub_a,
    COLLECT_STORAGE const ulong *was_children
) {
#if !SENSOR_PASS
    for (uint j = 0; j < ContactsPerBody; ++j) {
        if (was_feature[j] == NoIndex) break;
        if ((claimed & (1ul << j)) != 0) continue;
        events[reported++] = ContactEvent{body, was_other[j], was_feature[j], was_sub[j], was_children[j], uint(ContactRemoved), was_sub_a[j]};
    }
    counts[body] = reported;
#endif
}

// Resolve the same inheritance for new pairs and cached sleeping contacts.
static CollisionMask LeafFilter(Shape root, uint leaf, Filter body, device const Shape *shapes, device const Index *children) {
    const CollisionMask inherited = ResolveFilter(root, {body.Layer, body.Collides});
    return root.Kind == ShapeCompound ? ResolveFilter(shapes[ChildOf(root, leaf, children)], inherited) : inherited;
}

#ifndef CACHE_BOUNDS_HINT
#define CACHE_BOUNDS_HINT 0
#endif
kernel void BuildBodyBounds(
#if CACHE_BOUNDS_HINT
    device QueryArenaHeader *query_header [[buffer(11)]], device uint *query_snapshot [[buffer(18)]],
#endif
    device const Pose *poses [[buffer(0)]], device const Index *body_shapes [[buffer(6)]],
    device const Shape *shapes [[buffer(8)]], device BodyBounds *bounds [[buffer(14)]],
#if INTEGRATE_BOUNDS
    device Pose *initial [[buffer(1)]], device Pose *inertial [[buffer(2)]],
    device Velocity *velocities [[buffer(3)]], device const BodyMass *masses [[buffer(4)]],
    device Adjacency *incoming [[buffer(17)]], device uint *quiet [[buffer(21)]],
#else
    device const Velocity *velocities [[buffer(3)]],
#endif
    device const Index *children [[buffer(15)]], device const float3 *vertices [[buffer(27)]],
    device const BvhNode *nodes [[buffer(29)]], constant StepParams &p [[buffer(7)]],
#if BOUNDS_LANES > 1
    uint body [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]
#else
    uint body [[thread_position_in_grid]]
#endif
) {
#if BOUNDS_LANES == 1
    const uint lane = 0;
#endif
    if (body >= p.BodyCount) return;
#if CACHE_BOUNDS_HINT
    if (lane == 0) {
        bool changed = false;
        for (uint word = 0; word < sizeof(Pose) / sizeof(uint); ++word) {
            const uint at = body * (sizeof(Pose) / sizeof(uint)) + word;
            const uint value = ((device const uint *)poses)[at];
            changed |= query_snapshot[at] != value;
            query_snapshot[at] = value;
        }
        if (changed && query_header->Valid) {
            atomic_store_explicit((device atomic_uint *)&query_header->Reuse, 0u, memory_order_relaxed);
            atomic_store_explicit((device atomic_uint *)&query_header->DynamicSame, 0u, memory_order_relaxed);
            atomic_store_explicit((device atomic_uint *)&query_header->InputX, 0u, memory_order_relaxed);
            atomic_store_explicit((device atomic_uint *)&query_header->GeometryX, 0u, memory_order_relaxed);
        }
    }
#endif

    BodyBounds result{float3(INFINITY), float3(-INFINITY)};
    const Index root = body_shapes[body];
    if (root != NoIndex) {
        const Shape owner = shapes[root];
        const uint count = owner.Kind == ShapeCompound ? owner.VertexCount : 1;
        for (uint i = 0; i < count; ++i) {
            Shape shape = shapes[owner.Kind == ShapeCompound ? ChildOf(owner, i, children) : root];
            if (shape.Kind == ShapePlane) {
                result = {float3(-INFINITY), float3(INFINITY)};
                break;
            }
            if (owner.Kind == ShapeCompound) shape.Local = ComposePose(owner.Local, shape.Local);
            const Pose pose = ComposePose(poses[body], shape.Local);
            const Poly poly = (MESH_SHAPES && shape.Kind == ShapeMesh) ? MeshBounds(shape, pose, nodes) : MakePoly(pose, shape);
            float3 low = INFINITY, high = -INFINITY;
            if (poly.Kind == ShapeCylinder) {
                PolyBounds(poly, IdentityPose, vertices, low, high);
            } else {
                for (uint corner = lane; corner < PolyCount(poly); corner += BOUNDS_LANES) {
                    const float3 point = PolyVertex(poly, vertices, corner);
                    low = min(low, point);
                    high = max(high, point);
                }
#if BOUNDS_LANES > 1
                for (uint axis = 0; axis < 3; ++axis) {
                    low[axis] = simd_min(low[axis]);
                    high[axis] = simd_max(high[axis]);
                }
#endif
            }
            const float roundoff = 1e-5f * max(1.f, length(poly.Center) + max(length(low - poly.Center), length(high - poly.Center)));
            result.Low = min(result.Low, low - poly.Radius - roundoff);
            result.High = max(result.High, high + poly.Radius + roundoff);
        }
    }
    if (lane == 0) {
#if !SENSOR_PASS
        // The bound |vA-vB| <= |vA|+|vB| permits conservative per-body expansion for speculative contacts.
        const float reach = max(0.f, p.ContactMargin) + min(p.DeltaTime * (length(velocities[body].Linear) + length(p.Gravity) * p.DeltaTime), max(0.f, p.MaxContactReach));
        result.Low -= reach;
        result.High += reach;
#endif
        bounds[body] = result;
#if INTEGRATE_BOUNDS
        IntegrateBody(poses, initial, inertial, velocities, masses, incoming, quiet, p, body);
#endif
    }
}

// Ordered collection performs contact-slot writes, caching and event reporting.
struct GeometryQuery {
    Shape Shape, Target;
    Pose Pose, TargetPose, ShapePose, TargetShapePose;
    Poly OwnPoly, PlanePatch;
    float3 OwnCenter, TargetCenter;
    Velocity OwnVelocity, OtherVelocity;
    float Reach, Weld;
    bool MeshPair, PlaneBack, Report;
};

struct GeometryManifold {
    float3 Here[ManifoldPoints], There[ManifoldPoints], Normal;
    uint Feature[ManifoldPoints];
    float2 Patch;
    Index SubShape, SubShapeA;
    uint Count;
    float Weld;
    bool Visited;
};

static device QueryArenaHeader &QueryHeader(device uint *pool) { return *((device QueryArenaHeader *)pool); }

// Batches retain each owner's traversal order independently of storage allocation order.
static uint ReserveQueries(device atomic_uint *count, uint capacity, uint size) {
    uint first = atomic_load_explicit(count, memory_order_relaxed);
    while (first <= capacity && size <= capacity - first)
        if (atomic_compare_exchange_weak_explicit(count, &first, first + size, memory_order_relaxed, memory_order_relaxed)) return first;
    return NoIndex;
}

struct QueuedQuery {
    uint Batch, Candidate, OwnTriangle, Previous;
    float Weld;
    uint Result;
};
struct QueryContext {
    GeometryQuery Query;
    uint Owner, Other, OwnLeaf, TargetLeaf, Cached;
};
struct QueryBatch {
    uint Context, First, Count, Next, Present[2];
};
static device QueryBatch *QueryBatches(device uint *pool) { return (device QueryBatch *)((device uchar *)pool + QueryHeader(pool).BatchOffset); }
static device QueryContext *QueryContexts(device uint *pool) { return (device QueryContext *)((device uchar *)pool + QueryHeader(pool).ContextOffset); }
static device QueuedQuery *QueryRecords(device uint *pool) { return (device QueuedQuery *)((device uchar *)pool + QueryHeader(pool).TaskOffset); }
static device GeometryManifold *QueryResults(device uint *pool) { return (device GeometryManifold *)((device uchar *)pool + QueryHeader(pool).ResultOffset); }

kernel void ResetQueryReuse(device uint *pool [[buffer(11)]], constant QueryInputSpec &spec [[buffer(9)]]) {
    device QueryArenaHeader &h = QueryHeader(pool);
    h.DynamicSame = h.Valid;
    h.Reuse = h.DynamicSame && h.GeometryValid;
    h.GeometryX = h.DynamicSame && !h.GeometryChecked ? (spec.Offsets[13] - spec.Offsets[8] + 127) / 128 : 0;
    h.GeometryY = h.GeometryZ = 1;
    h.InputX = (spec.Words - (spec.Offsets[13] - spec.Offsets[8]) + 127) / 128;
    h.InputY = h.InputZ = 1;
}
kernel void CheckQueryInputs(
    device const uint *poses [[buffer(0)]], device const uint *velocities [[buffer(3)]],
    device const uint *masses [[buffer(4)]], device const Contact *contacts [[buffer(5)]], device const uint *body_shapes [[buffer(6)]],
    constant StepParams &p [[buffer(7)]], device const uint *shapes [[buffer(8)]], constant QueryInputSpec &spec [[buffer(9)]],
    device const uint *materials [[buffer(10)]], device uint *pool [[buffer(11)]], device const uint *children [[buffer(15)]],
    device uint *snapshot [[buffer(18)]], device const uint *filters [[buffer(19)]], device const uint *jointed [[buffer(20)]],
    device const uint *quiet [[buffer(21)]], device const uint *vertices [[buffer(27)]], device const uint *triangles [[buffer(28)]],
    device const uint *nodes [[buffer(29)]], device const uint *faces [[buffer(30)]],
    uint id [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]
) {
    const uint skip = spec.Offsets[13] - spec.Offsets[8];
    if (id >= spec.Offsets[8]) id += skip;
    bool changed = false;
    if (id < spec.Words) {
        uint region = 0;
        while (region < 15 && id >= spec.Offsets[region + 1]) ++region;
        const uint at = id - spec.Offsets[region];
        uint value = 0;
        switch (region) {
            case 0: value = poses[at]; break;
            case 1: value = velocities[at]; break;
            case 2: value = masses[at]; break;
            case 3: value = body_shapes[at]; break;
            case 4: value = shapes[at]; break;
            case 5: value = materials[at]; break;
            case 6: value = filters[at]; break;
            case 7: value = jointed[at]; break;
            case 8: value = vertices[at]; break;
            case 9: value = faces[at]; break;
            case 10: value = triangles[at]; break;
            case 11: value = nodes[at]; break;
            case 12: value = children[at]; break;
            default: break;
        }
        if (region == 13) value = ((constant uint *)&p)[at];
        else if (region == 14) {
            value = Frozen(((device const BodyMass *)masses)[at], ((device const Velocity *)velocities)[at], quiet[at], p);
            // Partial queues must be rebuilt so overflow owners retain their normal recomputation.
            if (at < p.BodyCount && QueryHeader(pool).Valid && pool[QueryOwnerOffset(p.BodyCount) + at]) changed = true;
        } else if (region == 15) {
            device const Contact &c = contacts[at / 8];
            switch (at % 8) {
                case 0: value = c.Active; break;
                case 1: value = c.BodyB; break;
                case 2: value = c.Feature; break;
                case 3: value = c.SubShape; break;
                case 4: value = c.SubShapeA; break;
                case 5: value = uint(c.Children); break;
                case 6: value = uint(c.Children >> 32); break;
                case 7: value = as_type<uint>(c.NominalArea); break;
            }
        }
        changed |= snapshot[id] != value;
        snapshot[id] = value;
    }
    const bool wave_changed = simd_any(changed);
    if (lane == 0 && wave_changed) {
        atomic_store_explicit((device atomic_uint *)&QueryHeader(pool).Reuse, 0u, memory_order_relaxed);
        atomic_store_explicit((device atomic_uint *)&QueryHeader(pool).DynamicSame, 0u, memory_order_relaxed);
        atomic_store_explicit((device atomic_uint *)&QueryHeader(pool).GeometryX, 0u, memory_order_relaxed);
    }
}
kernel void CheckQueryGeometry(
    constant QueryInputSpec &spec [[buffer(9)]], device uint *pool [[buffer(11)]],
    device const uint *children [[buffer(15)]], device uint *snapshot [[buffer(18)]],
    device const uint *vertices [[buffer(27)]], device const uint *triangles [[buffer(28)]],
    device const uint *nodes [[buffer(29)]], device const uint *faces [[buffer(30)]],
    uint id [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]
) {
    // Motion can suppress this dispatch; only an executed check validates the snapshot.
    if (id == 0) QueryHeader(pool).GeometryChecked = 1;
    id += spec.Offsets[8];
    bool changed = false;
    if (id < spec.Offsets[13]) {
        uint region = 8;
        while (region < 12 && id >= spec.Offsets[region + 1]) ++region;
        const uint at = id - spec.Offsets[region];
        uint value = 0;
        switch (region) {
            case 8: value = vertices[at]; break;
            case 9: value = faces[at]; break;
            case 10: value = triangles[at]; break;
            case 11: value = nodes[at]; break;
            case 12: value = children[at]; break;
        }
        changed = snapshot[id] != value;
        snapshot[id] = value;
    }
    const bool wave_changed = simd_any(changed);
    if (lane == 0 && wave_changed) atomic_store_explicit((device atomic_uint *)&QueryHeader(pool).Reuse, 0u, memory_order_relaxed);
}
kernel void ResetQueries(device uint *pool [[buffer(11)]], constant StepParams &p [[buffer(7)]], uint body [[thread_position_in_grid]]) {
    if (body >= p.BodyCount) return;
    if (QueryHeader(pool).Reuse) {
        if (body == 0) QueryHeader(pool).TaskCount = 0;
        return;
    }
    if (body == 0) {
        QueryHeader(pool).GeometryValid = QueryHeader(pool).DynamicSame;
        QueryHeader(pool).Valid = 1;
        QueryHeader(pool).TaskCount = 0;
        QueryHeader(pool).DispatchY = QueryHeader(pool).DispatchZ = 1;
        QueryHeader(pool).ContextCount = QueryHeader(pool).ResultCount = QueryHeader(pool).BatchCount = 0;
        const uint bytes = QueryHeader(pool).Bytes, begin = ((QueryOwnerOffset(p.BodyCount) + p.BodyCount) * sizeof(uint) + 15) & ~15u;
        const uint part = (bytes - begin) / 4;
        QueryHeader(pool).TaskCapacity = part / sizeof(QueuedQuery);
        QueryHeader(pool).ContextCapacity = part / sizeof(QueryContext);
        QueryHeader(pool).BatchCapacity = part / sizeof(QueryBatch);
        QueryHeader(pool).ContextOffset = begin;
        QueryHeader(pool).BatchOffset = (begin + QueryHeader(pool).ContextCapacity * sizeof(QueryContext) + 15) & ~15u;
        QueryHeader(pool).TaskOffset = (QueryHeader(pool).BatchOffset + QueryHeader(pool).BatchCapacity * sizeof(QueryBatch) + 15) & ~15u;
        QueryHeader(pool).ResultOffset = (QueryHeader(pool).TaskOffset + QueryHeader(pool).TaskCapacity * sizeof(QueuedQuery) + 15) & ~15u;
        QueryHeader(pool).ResultCapacity = (bytes - QueryHeader(pool).ResultOffset) / sizeof(GeometryManifold);
    }
    for (uint part = 0; part < QueryPartitions(p.BodyCount); ++part)
        pool[QueryHeadOffset(p.BodyCount, body, part)] = NoIndex;
    pool[QueryOwnerOffset(p.BodyCount) + body] = 0;
}

static GeometryManifold QueryGeometry(
    GeometryQuery q, uint candidate, Index own_triangle, uint previous,
    device const float3 *hull_vertices, device const HullFace *hull_faces, device const Triangle *mesh_triangles, uint lane
) {
    Index sub_shape = NoIndex, sub_shape_a = own_triangle;
    // Paired manifold points lie on their respective surfaces.
    // Feature identities encode source geometry to preserve warm starting across changes in point order.
    float3 points_here[MaxClipPoints], points_there[MaxClipPoints], normal;
    uint features[MaxClipPoints];
    uint found = 0;
    float2 patch{-1, 0};

    const BoxPose box = MakeBox(q.ShapePose, q.Shape);
    const bool curved = IsRound(q.Shape.Kind) || IsRound(q.Target.Kind);
    const bool hulled = q.Shape.Kind == ShapeHull || q.Target.Kind == ShapeHull || q.Shape.Kind == ShapeCylinder || q.Target.Kind == ShapeCylinder;
    if (q.MeshPair) {
        sub_shape = q.Target.FirstTriangle + candidate;
        const Poly face = MakeTriangle(q.TargetShapePose, mesh_triangles[sub_shape]);
        found = MeshManifold(q.OwnPoly, face, q.Shape.DoubleSided, q.Target.DoubleSided, q.OwnCenter, q.TargetCenter, previous, hull_vertices, hull_faces, q.Reach, points_here, points_there, features, normal);
    } else if (BoundedPlane(q.Target)) {
        Poly incident = q.OwnPoly;
        if ((MESH_SHAPES && q.Shape.Kind == ShapeMesh)) {
            sub_shape_a = q.Shape.FirstTriangle + candidate;
            incident = MakeTriangle(q.ShapePose, mesh_triangles[sub_shape_a]);
        }
        incident.Center -= q.ShapePose.Position;
        found = ConvexManifold(q.PlanePatch, incident, hull_vertices, hull_faces, q.Reach, -q.Target.Normal, points_there, points_here, features, normal);
        if (found == 0)
            found = ConvexManifold(q.PlanePatch, incident, hull_vertices, hull_faces, q.Reach, float3(0), points_there, points_here, features, normal);
        normal = -normal;
        for (uint i = 0; i < found; ++i) {
            points_here[i] += q.ShapePose.Position;
            points_there[i] += q.ShapePose.Position;
            if ((MESH_SHAPES && q.Shape.Kind == ShapeMesh) && q.PlaneBack) features[i] |= PlaneBackFeature;
        }
    } else if ((MESH_SHAPES && q.Target.Kind == ShapeMesh)) {
        const Index index = q.Target.FirstTriangle + candidate;
        Triangle triangle = mesh_triangles[index];
        sub_shape = index;
        Poly face = MakeTriangle(q.TargetShapePose, triangle);
        if (OutsideFaces(q.OwnPoly, face, hull_vertices, hull_faces, q.Reach, lane)) return {};
        const float3 first = PolyVertex(face, hull_vertices, 0);
        const float3 turn = cross(PolyVertex(face, hull_vertices, 1) - first, PolyVertex(face, hull_vertices, 2) - first);
        const float area = length(turn);
        if (area < 1e-18f) return {};
        float3 outward = turn / area;
        // Select the triangle side geometrically because penetration can exceed speculative reach.
        if (dot(q.OwnPoly.Center - first, outward) < 0) {
            if (!q.Target.DoubleSided) return {};
            const Index b = triangle.B;
            triangle.B = triangle.C;
            triangle.C = b;
            // Reversing B/C reverses edges 0 and 2; edge 1 keeps its index.
            const uint active = triangle.BackActiveEdges, owned = triangle.OwnedEdges;
            triangle.ActiveEdges = (active & 2u) | ((active & 1u) << 2) | ((active & 4u) >> 2);
            triangle.OwnedEdges = (owned & 2u) | ((owned & 1u) << 2) | ((owned & 4u) >> 2);
            face = MakeTriangle(q.TargetShapePose, triangle);
            outward = -outward;
        }

        const float3 top = PolySupportPoint(q.OwnPoly, hull_vertices, outward, lane);
        if (dot(top - first, outward) + q.OwnPoly.Radius <= 0) return {};
        const float3 bottom = PolySupportPoint(q.OwnPoly, hull_vertices, -outward, lane);
        if (!(dot(bottom - first, outward) - q.OwnPoly.Radius < q.Reach)) return {};

        // Use the triangle as the reference face and reverse the resulting normal.
        found = ConvexManifold(face, q.OwnPoly, hull_vertices, hull_faces, q.Reach, -outward, points_there, points_here, features, normal, lane);

        // Search active edges when face-normal clipping misses a contact on a convex crease.
        const bool searched = found == 0 && triangle.ActiveEdges != 0;
        if (searched)
            found = ConvexManifold(face, q.OwnPoly, hull_vertices, hull_faces, q.Reach, float3(0), points_there, points_here, features, normal, lane);
        normal = -normal;
#if !SENSOR_PASS
        // Measure the patch before removing duplicate points on shared edges.
        if (q.Report) patch = ManifoldGeometry(points_here, found, normal, q.Pose, q.TargetPose, q.OwnVelocity, q.OtherVelocity);
#endif

        // Match the tolerance used during clipping.
        float scale = 1;
        for (uint v = 0; v < 3; ++v) scale = max(scale, length(PolyVertex(face, hull_vertices, v)));
        const float seam = 1e-5f * scale;

        uint kept = 0;
        for (uint i = 0; i < found; ++i) {
            const bool triangle_led = ((features[i] >> 28) & 1) == 0;
            // Bits 8 through 10 identify triangle clipping edges when bit 28 is clear.
            const uint cut_by = triangle_led ? (features[i] >> 8) & 7 : 0u;
            const uint cut_by_seam = cut_by & ~triangle.ActiveEdges;
            // Retain a surface corner only on its owning triangle.
            const bool corner_is_mine = (cut_by & triangle.ActiveEdges) != 0 && (cut_by_seam & ~triangle.OwnedEdges) == 0;
            if (cut_by_seam != 0 && !corner_is_mine) continue;
            // Restrict fallback contacts to active edges to avoid false contacts on adjacent faces.
            if (searched) {
                bool on_feature = false;
                for (uint e = 0; e < 3 && !on_feature; ++e)
                    on_feature = (triangle.ActiveEdges & (1u << e)) != 0 &&
                        OnEdgeLine(face, hull_vertices, outward, e, points_there[i], seam);
                if (!on_feature) continue;
            }
            // Apply edge ownership to points that coincide with an edge without being clipped by it.
            bool disowned = false;
            for (uint e = 0; e < 3 && !disowned; ++e)
                disowned = ((triangle.ActiveEdges | triangle.OwnedEdges) & (1u << e)) == 0 &&
                    OnEdgeLine(face, hull_vertices, outward, e, points_there[i], seam);
            if (disowned) continue;
            points_here[kept] = points_here[i];
            points_there[kept] = points_there[i];
            features[kept] = features[i];
            ++kept;
        }
        found = kept;
    } else if (curved && !hulled) {
        // A capsule manifold uses the endpoints of the core interval nearest the other shape.
        const bool mine_is_round = IsRound(q.Shape.Kind);
        const Shape round_shape = mine_is_round ? q.Shape : q.Target;
        const Shape against = mine_is_round ? q.Target : q.Shape;
        const Core core = MakeCore(mine_is_round ? q.ShapePose : q.TargetShapePose, round_shape);
        const Pose other_pose = mine_is_round ? q.TargetShapePose : q.ShapePose;
        const bool other_is_round = IsRound(against.Kind);
        const Core other_core = other_is_round ? MakeCore(other_pose, against) : Core{};

        float3 samples[2], others[2];
        uint names[2];
        uint taken = 0;
        if (other_is_round) {
            const float3 mine_along = core.To - core.From, theirs_along = other_core.To - other_core.From;
            const float mine_length = length(mine_along), theirs_length = length(theirs_along);
            const bool parallel = mine_length > 1e-6f && theirs_length > 1e-6f &&
                abs(dot(mine_along / mine_length, theirs_along / theirs_length)) > 0.999f;
            if (parallel) {
                const float3 direction = mine_along / mine_length;
                const float base = dot(core.From, direction);
                const float their_low = dot(other_core.From, direction) - base;
                const float their_high = dot(other_core.To, direction) - base;
                const float low = max(0.f, min(their_low, their_high));
                const float high = min(mine_length, max(their_low, their_high));
                if (high - low > 1e-5f) {
                    for (uint end = 0; end < 2; ++end) {
                        const float at = end == 0 ? low : high;
                        samples[taken] = core.From + direction * at;
                        const bool theirs = end == 0 ? their_low > 0 || their_high > 0 : their_high < mine_length || their_low < mine_length;
                        names[taken] = (end << 1) | (theirs ? 1u : 0u);
                        ++taken;
                    }
                }
            }
            if (taken == 0) {
                ClosestOnSegments(core.From, core.To, other_core.From, other_core.To, samples[0], others[0]);
                names[0] = 0;
                taken = 1;
            }
        } else if (against.Kind == ShapePlane) {
            samples[0] = core.From;
            names[0] = 0;
            taken = 1;
            if (distance(core.From, core.To) > 1e-6f) {
                samples[1] = core.To;
                names[1] = 1;
                taken = 2;
            }
        } else {
            // Alternating projection identifies the nearest box face, including contact along the capsule interior.
            const BoxPose target_box = MakeBox(other_pose, against);
            float3 on_core = ClosestOnSegment(core.From, core.To, target_box.Center), on_box;
            float away;
            uint face = 0;
            for (uint round = 0; round < 3; ++round) {
                on_box = ClosestOnBox(target_box, on_core, away, face);
                on_core = ClosestOnSegment(core.From, core.To, on_box);
            }
            on_box = ClosestOnBox(target_box, on_core, away, face);

            const float3 out_of = OutOfBox(on_core, on_box, away);
            uint axis = 0;
            float most = 0;
            for (uint i = 0; i < 3; ++i) {
                const float aligned = abs(dot(out_of, target_box.Axis[i]));
                if (aligned > most) {
                    most = aligned;
                    axis = i;
                }
            }

            // Clipped core endpoints retain the identities of their bounding features.
            const float3 along = core.To - core.From;
            float low = 0, high = 1;
            uint low_name = 0, high_name = 1;
            for (uint side = 0; side < 2; ++side) {
                const uint slab = (axis + 1 + side) % 3;
                const float direction = dot(along, target_box.Axis[slab]);
                const float from = dot(core.From - target_box.Center, target_box.Axis[slab]);
                for (uint face_side = 0; face_side < 2; ++face_side) {
                    const float edge = face_side == 0 ? target_box.Half[slab] : -target_box.Half[slab];
                    if (abs(direction) < 1e-9f) continue;
                    const float at = (edge - from) / direction;
                    const bool entering = (face_side == 0) == (direction < 0);
                    const uint slab_name = 2 + slab * 2 + face_side;
                    if (entering && at > low) {
                        low = at;
                        low_name = slab_name;
                    } else if (!entering && at < high) {
                        high = at;
                        high_name = slab_name;
                    }
                }
            }

            if (high - low > 1e-5f && length(along) * (high - low) > 1e-5f) {
                samples[0] = core.From + along * low;
                samples[1] = core.From + along * high;
                names[0] = low_name;
                names[1] = high_name;
                taken = 2;
            } else {
                samples[0] = on_core;
                names[0] = 8 + face;
                taken = 1;
            }
        }

        for (uint sample = 0; sample < taken && found < MaxFacePoints; ++sample) {
            const float3 at = samples[sample];
            // `out_of` points away from the other shape and `gap` spans them.
            float3 nearest, out_of;
            float gap;
            if (against.Kind == ShapePlane) {
                const float above = dot(against.Normal, at) - against.Offset;
                out_of = against.Normal;
                nearest = at - above * against.Normal;
                gap = above - core.Radius;
            } else if (other_is_round) {
                const float3 on_theirs = taken == 1 ? others[0] : ClosestOnSegment(other_core.From, other_core.To, at);
                const float3 apart = at - on_theirs;
                const float span = length(apart);
                out_of = span > 1e-9f ? apart / span : float3(0, 1, 0);
                nearest = on_theirs + out_of * other_core.Radius;
                gap = span - other_core.Radius - core.Radius;
            } else {
                float away;
                uint face;
                nearest = ClosestOnBox(MakeBox(other_pose, against), at, away, face);
                out_of = OutOfBox(at, nearest, away);
                gap = away - core.Radius;
            }

            if (gap >= q.Reach) continue;
            normal = mine_is_round ? out_of : -out_of;
            const float3 on_round = at - out_of * core.Radius;
            points_here[found] = mine_is_round ? on_round : nearest;
            points_there[found] = mine_is_round ? nearest : on_round;
            features[found] = names[sample];
            ++found;
        }
    } else if (q.Target.Kind == ShapePlane) {
        normal = q.Target.Normal;
        // Distribute samples across the support region within MaxFacePoints.
        if (q.Shape.Kind == ShapeCylinder) {
            float3 plane;
            const uint count = SupportFace(q.OwnPoly, hull_vertices, hull_faces, -normal, points_here, features, plane);
            for (uint i = 0; i < count; ++i) {
                if (dot(normal, points_here[i]) >= q.Target.Offset + q.Reach) continue;
                points_here[found] = points_here[i];
                features[found++] = features[i];
            }
        } else found = SpreadSupport(q.OwnPoly, hull_vertices, -normal, -(q.Target.Offset + q.Reach), MaxFacePoints, points_here, features);
        for (uint i = 0; i < found; ++i) {
            points_there[i] = points_here[i] - (dot(normal, points_here[i]) - q.Target.Offset) * normal;
            if ((MESH_SHAPES && q.Shape.Kind == ShapeMesh) && q.PlaneBack) features[i] |= PlaneBackFeature;
        }
    } else if (!hulled) {
        const BoxPose other_box = MakeBox(q.TargetShapePose, q.Target);

        // Test edge axes as well as face normals to resolve crossed boxes.
        bool apart = false;
        uint edge_i = 0, edge_j = 0;
        float3 edge_normal = float3(0);
        float least_edge = INFINITY;
        for (uint i = 0; i < 3 && !apart; ++i) {
            for (uint j = 0; j < 3; ++j) {
                const float3 axis = cross(box.Axis[i], other_box.Axis[j]);
                const float len = length(axis);
                if (len < 1e-6f) continue; // Face axes cover parallel edges.
                const float3 unit = axis / len;
                const float overlap = Overlap(box, other_box, unit);
                // A projection gap beyond speculative reach proves separation.
                if (overlap < -q.Reach) {
                    apart = true;
                    break;
                }
                if (overlap < least_edge) {
                    least_edge = overlap;
                    edge_i = i;
                    edge_j = j;
                    edge_normal = dot(unit, box.Center - other_box.Center) < 0 ? -unit : unit;
                }
            }
        }
        // Reference-face bias preserves warm starts when face penetrations are nearly equal.
        uint best_axis = 0, best_owner = 0;
        float least = INFINITY;
        // Keep the separation guard in loop conditions to avoid a fast-math miscompile of early continue.
        for (uint owner = 0; owner < 2 && !apart; ++owner) {
            for (uint i = 0; i < 3; ++i) {
                const float3 axis = owner == 0 ? box.Axis[i] : other_box.Axis[i];
                const float overlap = Overlap(box, other_box, axis);
                if (overlap < -q.Reach) {
                    apart = true;
                    break;
                }
                const float extent = owner == 0 ? box.Half[i] : other_box.Half[i];
                const float incumbent = isinf(least) ? least : least - (1 - RelativeTolerance) * abs(least);
                if (overlap < incumbent - AbsoluteTolerance * extent) {
                    least = overlap;
                    best_axis = i;
                    best_owner = owner;
                }
            }
        }
        if (apart || least > 1e18f) return {};

        // Prefer face axes when edge and face separation are within tolerance.
        const bool on_edge = least_edge < 1e18f && RelativeTolerance * -least_edge > -least + EdgeTolerance;
        if (on_edge) {
            float3 mine[2], theirs[2];
            uint which_mine, which_theirs;
            SupportEdge(box, edge_i, -edge_normal, mine, which_mine);
            SupportEdge(other_box, edge_j, edge_normal, theirs, which_theirs);
            normal = edge_normal;
            ClosestOnSegments(mine[0], mine[1], theirs[0], theirs[1], points_here[0], points_there[0]);
            // Edge-pair identity prevents parallel edges from sharing a cached dual.
            features[0] = (1u << 15) | (edge_i << 13) | (edge_j << 11) | (which_mine << 9) | (which_theirs << 7);
            found = 1;
        } else {
            const BoxPose reference = best_owner == 0 ? box : other_box;
            const BoxPose incident = best_owner == 0 ? other_box : box;
            float3 face_normal = reference.Axis[best_axis];
            if (dot(face_normal, incident.Center - reference.Center) < 0) face_normal = -face_normal;
            normal = best_owner == 0 ? -face_normal : face_normal;

            uint incident_axis = 0;
            float incident_side = 1, most_opposed = INFINITY;
            for (uint i = 0; i < 3; ++i) {
                for (uint s = 0; s < 2; ++s) {
                    const float side = s == 0 ? 1 : -1;
                    const float alignment = dot(incident.Axis[i] * side, face_normal);
                    if (alignment < most_opposed) {
                        most_opposed = alignment;
                        incident_axis = i;
                        incident_side = side;
                    }
                }
            }

            float3 poly[MaxFacePoints], clipped[MaxFacePoints];
            uint names[MaxFacePoints], clipped_names[MaxFacePoints];
            FaceCorners(incident, incident_axis, incident_side, poly);
            for (uint i = 0; i < 4; ++i) names[i] = (1u << i) | (1u << ((i + 3) % 4));
            uint poly_count = 4;
            const float clip_tolerance = 1e-5f * max(1.f, length(reference.Center) + length(reference.Half));
            for (uint edge = 0; edge < 2; ++edge) {
                const uint side_axis = (best_axis + 1 + edge) % 3;
                for (uint s = 0; s < 2; ++s) {
                    const float3 side_normal = reference.Axis[side_axis] * (s == 0 ? 1 : -1);
                    const float side_offset = dot(side_normal, reference.Center) + reference.Half[side_axis];
                    const uint plane = 1u << (4 + edge * 2 + s);
                    poly_count = ClipAgainst(poly, names, poly_count, side_normal, side_offset, plane, clip_tolerance, MaxFacePoints, clipped, clipped_names);
                }
            }

            const float face_offset = dot(face_normal, reference.Center) + reference.Half[best_axis];
            for (uint i = 0; i < poly_count && found < MaxFacePoints; ++i) {
                const float depth = dot(face_normal, poly[i]) - face_offset;
                if (depth >= q.Reach) continue;
                const float3 on_reference = poly[i] - depth * face_normal;
                points_here[found] = best_owner == 0 ? on_reference : poly[i];
                points_there[found] = best_owner == 0 ? poly[i] : on_reference;
                // Feature identity encodes reference-face ownership, face axes and clipping geometry.
                features[found] = (best_owner << 13) | (best_axis << 11) | (incident_axis << 9) |
                    ((incident_side > 0 ? 1u : 0u) << 8) | names[i];
                ++found;
            }
        }
    } else {
        found = ConvexManifold(q.OwnPoly, MakePoly(q.TargetShapePose, q.Target), hull_vertices, hull_faces, q.Reach, float3(0), points_here, points_there, features, normal, lane);
    }

#if !SENSOR_PASS
    // Reject internal contacts using the resulting normal against suppressed face directions.
    if (found > 0 && (BuriedAlong(q.Target, q.TargetPose, hull_faces, normal) || BuriedAlong(q.Shape, q.Pose, hull_faces, -normal)))
        found = 0;
#endif

    // Merge duplicate geometry before four-point reduction.
    found = WeldManifold(points_here, points_there, features, found, q.Weld);
#if !SENSOR_PASS
    if (q.Report && patch.x < 0) patch = ManifoldGeometry(points_here, found, normal, q.Pose, q.TargetPose, q.OwnVelocity, q.OtherVelocity);
#endif
    found = ReduceManifold(points_here, points_there, features, found, normal);
    GeometryManifold result{};
    result.Visited = true;
    result.Count = found;
    result.Weld = q.Weld;
    result.SubShape = sub_shape;
    result.SubShapeA = sub_shape_a;
    result.Patch = patch;
    if (found) result.Normal = normal;
    for (uint i = 0; i < found; ++i) {
        result.Here[i] = points_here[i];
        result.There[i] = points_there[i];
        result.Feature[i] = features[i];
    }
    return result;
}

kernel void EvaluateQueries(device uint *pool [[buffer(11)]], constant StepParams &p [[buffer(7)]], device const float3 *vertices [[buffer(27)]], device const HullFace *faces [[buffer(30)]], device const Triangle *triangles [[buffer(28)]], uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    if (QueryHeader(pool).Reuse || group >= min(QueryHeader(pool).TaskCount, QueryHeader(pool).TaskCapacity)) return;
    device QueuedQuery &task = QueryRecords(pool)[group];
    if (task.Batch >= QueryHeader(pool).BatchCapacity) return;
    device QueryBatch &batch = QueryBatches(pool)[task.Batch];
    device QueryContext &context = QueryContexts(pool)[batch.Context];
    GeometryQuery query = context.Query;
    const bool cooperate = COLLECT_LANES >= 32 && ((ConvexLeaf(query.Shape.Kind) && ConvexLeaf(query.Target.Kind) && (query.Shape.Kind == ShapeHull || query.Target.Kind == ShapeHull)) || (query.Shape.Kind == ShapeHull && (MESH_SHAPES && query.Target.Kind == ShapeMesh)));
    if (!cooperate && lane != 0) return;
    query.Weld = task.Weld;
    if (query.MeshPair) query.OwnPoly = MakeTriangle(query.ShapePose, triangles[task.OwnTriangle]);
    const GeometryManifold result = QueryGeometry(query, task.Candidate, task.OwnTriangle, task.Previous, vertices, faces, triangles, cooperate ? lane : NoIndex);
    if (lane == 0 && result.Visited && (result.Count || context.Cached)) {
        const uint at = atomic_fetch_add_explicit((device atomic_uint *)(&QueryHeader(pool).ResultCount), 1u, memory_order_relaxed);
        if (at >= QueryHeader(pool).ResultCapacity) atomic_fetch_add_explicit((device atomic_uint *)(pool + QueryOwnerOffset(p.BodyCount) + context.Owner), 1u, memory_order_relaxed);
        else {
            QueryResults(pool)[at] = result;
            task.Result = at;
            const uint within = group - batch.First;
            atomic_fetch_or_explicit((device atomic_uint *)(batch.Present + within / 32), 1u << (within % 32), memory_order_relaxed);
        }
    }
}

// Lane zero owns contact slots; cooperative lanes share history and geometry scratch.
struct ContactHistory {
    uint was_feature[ContactsPerBody], was_stick[ContactsPerBody];
    Index was_other[ContactsPerBody];
    Index was_sub_a[ContactsPerBody];
    Index was_sub[ContactsPerBody];
    ulong was_children[ContactsPerBody];
    float3 was_lambda[ContactsPerBody], was_penalty[ContactsPerBody];
    float3 was_anchor_a[ContactsPerBody], was_anchor_b[ContactsPerBody];
};

static void SaveContactHistory(COLLECT_STORAGE ContactHistory &history, device Contact *slots, uint i) {
    history.was_feature[i] = slots[i].Feature;
    history.was_other[i] = slots[i].BodyB;
    history.was_sub[i] = slots[i].SubShape;
    history.was_sub_a[i] = slots[i].SubShapeA;
    history.was_children[i] = slots[i].Children;
    history.was_lambda[i] = slots[i].Lambda;
    history.was_penalty[i] = slots[i].Penalty;
    history.was_stick[i] = slots[i].Stick;
    history.was_anchor_a[i] = slots[i].AnchorA;
    history.was_anchor_b[i] = slots[i].AnchorB;
    if (!PREPARE_QUERIES) slots[i].Active = false;
}

static void CollectManifold(
    const thread GeometryManifold &geometry, const thread GeometryQuery &q,
    uint body, uint other, uint own_leaf, uint target_leaf, bool cached_pair,
    const thread Shape &body_shape, const thread Shape &other_body_shape,
    device const Material *materials, device Contact *slots, device uint *contact_refusals,
    constant StepParams &p, thread uint &count, thread uint *inherited,
    COLLECT_STORAGE const ContactHistory &history
) {
    const Shape shape = q.Shape, target = q.Target;
    const Pose pose = q.Pose, target_pose = q.TargetPose, shape_pose = q.ShapePose, target_shape_pose = q.TargetShapePose;
    const Velocity own_velocity = q.OwnVelocity, other_velocity = q.OtherVelocity;
    const uint own_leaf_count = body_shape.Kind == ShapeCompound ? body_shape.VertexCount : 1;
    const uint target_leaf_count = other_body_shape.Kind == ShapeCompound ? other_body_shape.VertexCount : 1;
    const float3 penalty_floor(p.PenaltyMin);
    const float weld = geometry.Weld;
    const ulong children = ChildPair(own_leaf, target_leaf);
    if (!geometry.Visited) return;
    const Index sub_shape = geometry.SubShape, sub_shape_a = geometry.SubShapeA;
    const uint found = geometry.Count;
    const float3 normal = geometry.Normal;
    const float2 patch = geometry.Patch;
    const thread float3 *points_here = geometry.Here, *points_there = geometry.There;
    const thread uint *features = geometry.Feature;
    if (cached_pair) {
        for (uint k = 0; k < count; ++k)
            if (slots[k].BodyB == other && slots[k].Children == children && slots[k].SubShape == sub_shape && slots[k].SubShapeA == sub_shape_a) {
                slots[k].NominalArea = patch.x;
                slots[k].NominalExtent = patch.y;
            }
        return;
    }

#if !SENSOR_PASS
    // Weld matching positions and normals across leaves or triangles without merging unrelated features.
    const bool siblings = own_leaf_count > 1 || target_leaf_count > 1 || sub_shape_a != NoIndex || sub_shape != NoIndex;
#endif

    for (uint i = 0; i < found; ++i) {
#if !SENSOR_PASS
        const float3 anchor_a = LocalPoint(pose, points_here[i]);
        const float3 anchor_b = LocalPoint(target_pose, points_there[i]);
        // The first matching geometric contact retains ownership.
        bool held = false;
        for (uint k = 0; k < count && siblings && !held; ++k)
            held = slots[k].Active && slots[k].BodyB == other && dot(slots[k].Normal, normal) > 0.99999f &&
                distance(anchor_a, slots[k].PointA) <= weld && distance(anchor_b, slots[k].PointB) <= weld;
        if (held) continue;
#endif

        // When full, replace the shallowest retained contact with a deeper candidate.
        // Maintain a dense run and account for every refused point.
        const float separation = dot(normal, points_here[i] - points_there[i]) + p.ContactMargin;
#if SENSOR_PASS
        if (dot(normal, points_here[i] - points_there[i]) > 0) continue;
        bool already = false;
        for (uint k = 0; k < count; ++k)
            already |= slots[k].BodyB == other && slots[k].Children == children;
        if (already) continue;
#endif
        uint at = count;
        if (count == ContactsPerBody) {
            uint shallowest = 0;
            for (uint k = 1; k < ContactsPerBody; ++k)
                if (slots[k].C0.x > slots[shallowest].C0.x) shallowest = k;
            ++contact_refusals[body];
            if (separation >= slots[shallowest].C0.x) continue;
            at = shallowest;
        }
        device Contact &contact = slots[at];
        contact.BodyA = body;
        contact.BodyB = other;
        contact.Feature = features[i];
        contact.Children = children;
        contact.Active = true;
#if SENSOR_PASS
        // Sensors retain one identity per overlapping leaf pair, without solver state.
        contact.C0.x = separation;
#else
        contact.AnchorA = anchor_a;
        contact.AnchorB = anchor_b;
        contact.PointA = anchor_a;
        contact.PointB = anchor_b;
        contact.NominalArea = patch.x;
        contact.NominalExtent = patch.y;
        contact.Normal = normal;
        const Material material_a = shape.HasMaterial ? shape.Surface : (body_shape.HasMaterial ? body_shape.Surface : materials[body]);
        const Material material_b = target.HasMaterial ? target.Surface : (other_body_shape.HasMaterial ? other_body_shape.Surface : materials[other]);
        const float3 relative_velocity = (own_velocity.Linear + cross(own_velocity.Angular, points_here[i] - pose.Position)) - (other_velocity.Linear + cross(other_velocity.Angular, points_there[i] - target_pose.Position));
        const bool resting = length(relative_velocity - normal * dot(relative_velocity, normal)) < 1e-3f;
        contact.Friction = Combine(resting ? material_a.StaticFriction : material_a.DynamicFriction, resting ? material_b.StaticFriction : material_b.DynamicFriction, material_a.FrictionCombine, material_b.FrictionCombine);
        contact.Restitution = Combine(material_a.Restitution, material_b.Restitution, material_a.RestitutionCombine, material_b.RestitutionCombine);
        contact.SubShape = sub_shape;
        contact.SubShapeA = sub_shape_a;

        contact.Approach = -dot(normal, relative_velocity);
        contact.BounceImpulse = 0;
        contact.BounceDelta = 0;

        contact.Penalty = penalty_floor;
        contact.Lambda = float3(0);
        contact.Stick = false;
        inherited[at] = NoIndex;
        for (uint j = 0; j < ContactsPerBody; ++j) {
            if (history.was_feature[j] == NoIndex) break;
            if (history.was_feature[j] != contact.Feature || history.was_other[j] != other || history.was_sub[j] != sub_shape || history.was_sub_a[j] != sub_shape_a || history.was_children[j] != children) continue;
            inherited[at] = j;
            contact.Penalty = clamp(history.was_penalty[j] * p.Gamma, penalty_floor, float3(p.PenaltyMax));
            contact.Lambda = history.was_lambda[j];
            // Contacts that remained inside the friction cone keep their sticking anchors.
            // Sliding contacts use fresh anchors so tangential displacement does not accumulate across slip.
            const float3 want_a = WorldPoint(pose, history.was_anchor_a[j]);
            const float3 want_b = WorldPoint(target_pose, history.was_anchor_b[j]);
            bool onto_another = false;
            for (uint k = 0; k < count && !onto_another; ++k) {
                if (k == at || !slots[k].Active || slots[k].BodyB != other || slots[k].Children != children || dot(slots[k].Normal, normal) <= 0.99999f) continue;
                onto_another = distance(history.was_anchor_a[j], slots[k].AnchorA) <= weld &&
                    distance(history.was_anchor_b[j], slots[k].AnchorB) <= weld;
            }
            for (uint k = i + 1; k < found && !onto_another; ++k)
                onto_another = distance(want_a, points_here[k]) <= weld && distance(want_b, points_there[k]) <= weld;
            const bool curved = CurvedSurface(shape.Kind, shape_pose.Orientation, normal) || CurvedSurface(target.Kind, target_shape_pose.Orientation, normal);
            if (history.was_stick[j] && !curved && !onto_another) {
                contact.AnchorA = history.was_anchor_a[j];
                contact.AnchorB = history.was_anchor_b[j];
                contact.Stick = true;
            }
            break;
        }
        const ContactBasis basis = MakeContactBasis(normal);
        const float3 gap = WorldPoint(pose, contact.AnchorA) - WorldPoint(target_pose, contact.AnchorB);
        contact.C0 = float3(dot(basis.Axis[0], gap), dot(basis.Axis[1], gap), dot(basis.Axis[2], gap)) + float3(p.ContactMargin, 0, 0);
#endif
        if (at == count) ++count;
    }
}

kernel void CollectContacts(
    device uint *query_pool [[buffer(11)]],
    device Contact *contacts [[buffer(5)]], device const Pose *poses [[buffer(0)]],
    device const BodyMass *masses [[buffer(4)]], device const Index *body_shapes [[buffer(6)]],
    device const Shape *shapes [[buffer(8)]], device const Material *materials [[buffer(10)]],
    device const Index *compound_children [[buffer(15)]],
    device const Velocity *velocities [[buffer(3)]],
    device const Filter *filters [[buffer(19)]], device const Index *jointed_to [[buffer(20)]],
    device ContactEvent *contact_events [[buffer(24)]], device uint *contact_event_counts [[buffer(25)]],
    device uint *contact_refusals [[buffer(26)]], device const float3 *hull_vertices [[buffer(27)]],
    device const Triangle *mesh_triangles [[buffer(28)]], device const BvhNode *bvh_nodes [[buffer(29)]],
    device const HullFace *hull_faces [[buffer(30)]], device const uint *quiet [[buffer(21)]],
    device BroadPhaseNode *broad_phase [[buffer(14)]],
    device Adjacency *incoming [[buffer(17)]],
    constant StepParams &p [[buffer(7)]],
#if COLLECT_LANES > 1
    uint body [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]
#else
    uint body [[thread_position_in_grid]]
#endif
) {
#if COLLECT_LANES == 1
    const uint lane = 0;
#endif
#if PREPARE_QUERIES
    const uint partitions = QueryPartitions(p.BodyCount);
    const uint partition = body % partitions;
    body /= partitions;
#endif
    if (body >= p.BodyCount) return;
#if PREPARE_QUERIES
    if (QueryHeader(query_pool).Reuse) return;
    const Index partition_shape = body_shapes[body];
    if (partition_shape == NoIndex) return;
    const Shape partition_root = shapes[partition_shape];
    if (partition_root.Kind == ShapePlane) return;
    const uint partition_leaves = partition_root.Kind == ShapeCompound ? partition_root.VertexCount : 1;
    const uint partner_partitions = max(partitions / max(partition_leaves, 1u), 1u);
    const uint leaf_partitions = partitions / partner_partitions;
    const uint partition_width = partition_leaves / leaf_partitions + uint(partition_leaves % leaf_partitions != 0);
    const uint first_leaf = (partition / partner_partitions) * partition_width;
    if (first_leaf >= partition_leaves) return;
    const uint last_leaf = first_leaf + min(partition_width, partition_leaves - first_leaf);
    const uint partner_width = p.BodyCount / partner_partitions + uint(p.BodyCount % partner_partitions != 0);
    const uint first_partner = (partition % partner_partitions) * partner_width;
    if (first_partner >= p.BodyCount) return;
    const uint last_partner = first_partner + min(partner_width, p.BodyCount - first_partner);
#endif
#if QUEUED_QUERIES
    if (query_pool[QueryOwnerOffset(p.BodyCount) + body]) return;
#elif RECOMPUTE_QUERIES
    if (!query_pool[QueryOwnerOffset(p.BodyCount) + body]) return;
#endif

    if (!PREPARE_QUERIES && lane == 0) contact_refusals[body] = 0;
#if PREPARE_QUERIES || QUEUED_QUERIES
    device QueuedQuery *query_records = QueryRecords(query_pool);
    COLLECT_STORAGE uint query_base, context_base, batch_base;
    COLLECT_STORAGE bool first_context, query_failed;
    device QueryBatch *query_batches = QueryBatches(query_pool);
    device QueryContext *query_contexts = QueryContexts(query_pool);
    uint query_tail = NoIndex;
#endif
#if !PREPARE_QUERIES && !QUEUED_QUERIES
    COLLECT_STORAGE GeometryManifold geometries[COLLECT_LANES];
#endif
    device Contact *slots = contacts + body * ContactsPerBody;
    device ContactEvent *events = contact_events + body * EventsPerBody;
    // Track claimed prior slots and emitted events separately from the current contact count.
    ulong claimed = 0;
    uint reported = 0;

    uint inherited[ContactsPerBody];

    COLLECT_STORAGE ContactHistory history;
    // One SIMD group copies the dense history run without changing its sentinel.
#if COLLECT_LANES >= 32
    if (lane < 32) {
        uint first_inactive = ContactsPerBody;
        for (uint i = lane; i < ContactsPerBody; i += 32)
            if (!slots[i].Active) first_inactive = min(first_inactive, i);
        first_inactive = simd_min(first_inactive);
        for (uint i = lane; i < first_inactive; i += 32) SaveContactHistory(history, slots, i);
        if (lane == 0 && first_inactive < ContactsPerBody) history.was_feature[first_inactive] = NoIndex;
    }
#else

    if (lane == 0) {
        for (uint i = 0; i < ContactsPerBody; ++i) {
            if (!slots[i].Active) {
                history.was_feature[i] = NoIndex;
                break;
            }
            SaveContactHistory(history, slots, i);
        }
    }
#endif
    if (COLLECT_LANES > 1) threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    const Index shape_index = body_shapes[body];
    if (shape_index == NoIndex || shapes[shape_index].Kind == ShapePlane) {
        if (!PREPARE_QUERIES && lane == 0) EndUnclaimed(events, contact_event_counts, body, claimed, reported, history.was_feature, history.was_other, history.was_sub, history.was_sub_a, history.was_children);
        return;
    }
    const Shape body_shape = shapes[shape_index];
    const Pose pose = poses[body];
    const Filter own_filter = filters[body];
    const float own_inverse_mass = masses[body].InvMass;
    const bool i_move = Moves(masses[body]);

    const uint own_leaf_count = body_shape.Kind == ShapeCompound ? body_shape.VertexCount : 1;

    uint count = 0;
    // Carry frozen pairs forward unchanged; remeasure reporting geometry when reporting is enabled later.
    COLLECT_STORAGE bool measure_cached;
    if (lane == 0) measure_cached = false;
#if SENSOR_PASS
    const bool frozen = false;
#else
    const bool frozen = Frozen(masses[body], velocities[body], quiet[body], p);
#endif
    if (frozen && lane == 0) {
        for (uint j = 0; j < ContactsPerBody; ++j) {
            if (history.was_feature[j] == NoIndex) break;
            const Index partner = history.was_other[j];
            if (body_shapes[partner] == NoIndex) continue;
            if (!Frozen(masses[partner], velocities[partner], quiet[partner], p)) continue;
            const Shape other_root = shapes[body_shapes[partner]];
            if (own_filter.Sensor || filters[partner].Sensor ||
                !Allows(LeafFilter(body_shape, OwnChild(history.was_children[j]), own_filter, shapes, compound_children), LeafFilter(other_root, OtherChild(history.was_children[j]), filters[partner], shapes, compound_children))) continue;

#if !PREPARE_QUERIES
            slots[count] = slots[j];
            slots[count].Active = true;
            slots[count].Approach = slots[count].BounceImpulse = slots[count].BounceDelta = 0;
#endif
            // Reporting enabled after sleep measures geometry without replacing cached solver rows.
            if (p.ReportContacts && slots[PREPARE_QUERIES ? j : count].NominalArea < 0) {
                if (!PREPARE_QUERIES) slots[count].NominalArea = slots[count].NominalExtent = 0;
                measure_cached = true;
            }
            inherited[count] = j;
            ++count;
        }
    }
    if (COLLECT_LANES > 1) threadgroup_barrier(mem_flags::mem_threadgroup);
#if QUEUED_QUERIES
    if (lane == 0) {
        for (uint part = 0; part < QueryPartitions(p.BodyCount); ++part) {
            for (Index context_at = query_pool[QueryHeadOffset(p.BodyCount, body, part)]; context_at != NoIndex; context_at = query_batches[context_at].Next) {
                device const QueryBatch &batch = query_batches[context_at];
                device const QueryContext &context = query_contexts[batch.Context];
                for (uint word = 0; word < 2; ++word) {
                    uint present = batch.Present[word];
                    while (present) {
                        const uint query_at = batch.First + 32 * word + ctz(present);
                        present &= present - 1;
                        const QueuedQuery record = query_records[query_at];
                        const GeometryQuery q = context.Query;
                        const Index other = context.Other;
                        const uint own_leaf = context.OwnLeaf, target_leaf = context.TargetLeaf;
                        const Shape other_body_shape = shapes[body_shapes[other]];
                        const bool cached_pair = frozen && Frozen(masses[other], velocities[other], quiet[other], p);

                        const GeometryManifold geometry = QueryResults(query_pool)[record.Result];
                        CollectManifold(geometry, q, body, other, own_leaf, target_leaf, cached_pair, body_shape, other_body_shape, materials, slots, contact_refusals, p, count, inherited, history);
                    }
                }
            }
        }
    }
#else
    // Visit every leaf pair even after the contact run fills.
#if PREPARE_QUERIES
    for (uint own_leaf = first_leaf; own_leaf < last_leaf; ++own_leaf) {
#else
    for (uint own_leaf = 0; own_leaf < own_leaf_count; ++own_leaf) {
#endif
        Shape shape = shapes[body_shape.Kind == ShapeCompound ? ChildOf(body_shape, own_leaf, compound_children) : shape_index];
        if (body_shape.Kind == ShapeCompound) shape.Local = ComposePose(body_shape.Local, shape.Local);
        if (shape.Kind == ShapePlane) continue;
        const CollisionMask leaf_filter = ResolveFilter(shape, ResolveFilter(body_shape, {own_filter.Layer, own_filter.Collides}));
        if (own_filter.Mixed && InternalFaces(shape) != 0) shape.FirstTriangle = 0;
        const Pose shape_pose = ComposePose(pose, shape.Local);
        Poly leaf_poly = MakePoly(shape_pose, shape);
        // A mesh against a half-space queries its vertex run, with no hull cook or face list.
        if ((MESH_SHAPES && shape.Kind == ShapeMesh)) leaf_poly.Kind = ShapeHull;
        const uint geometry_lane = COLLECT_LANES >= 32 ? lane % 32 : NoIndex;
        float own_reach = (MESH_SHAPES && shape.Kind == ShapeMesh) ? -1 : PolyReach(leaf_poly, hull_vertices, geometry_lane) + shape.Radius;
#if BROAD_PHASE_MODE == 1
#if PREPARE_QUERIES
        for (uint other = first_partner; other < last_partner; ++other) {
#else
        for (uint other = 0; other < p.BodyCount; ++other) {
#endif
            if (!(broad_phase[body].Candidates & (1u << other))) continue;
#elif !MESH_SHAPES && COLLECT_LANES == 1
        BodyCandidateCursor candidates{};
        for (uint other = NextBodyCandidateBatch<2>(broad_phase, p.BodyCount, body, candidates); other != NoIndex; other = NextBodyCandidateBatch<2>(broad_phase, p.BodyCount, body, candidates)) {
#else
    for (uint other = NextBodyCandidate<2>(broad_phase, p.BodyCount, body, 0); other != NoIndex; other = NextBodyCandidate<2>(broad_phase, p.BodyCount, body, other + 1)) {
#endif
            const Index other_shape = body_shapes[other];
            if (other == body || other_shape == NoIndex) continue;

            const bool cached_pair = frozen && Frozen(masses[other], velocities[other], quiet[other], p);
            if (cached_pair && !measure_cached) continue;
            const bool they_move = Moves(masses[other]);
#if SENSOR_PASS
            if (!(own_filter.Sensor || filters[other].Sensor)) continue;

#else
            if (own_filter.Sensor || filters[other].Sensor) continue;
            if (!i_move && !they_move) continue;

#endif

            const Filter theirs = filters[other];
            if (!Allows(leaf_filter, theirs.Aggregate)) continue;
            bool jointed = false;
            for (uint i = jointed_to[body]; i < jointed_to[body + 1] && !jointed; ++i) jointed = jointed_to[i] == other;
            if (jointed) continue;

            const Shape other_body_shape = shapes[other_shape];
            if ((MESH_SHAPES && shape.Kind == ShapeMesh) && other_body_shape.Kind != ShapePlane && (!MESH_PAIRS || (!MESH_SHAPES || other_body_shape.Kind != ShapeMesh)) && other_body_shape.Kind != ShapeCompound) continue;
            const Pose target_pose = poses[other];
            const Velocity own_velocity = velocities[body], other_velocity = velocities[other];
#if SENSOR_PASS
            const float reach = 0;
#else
            const float reach = p.ContactMargin +
                min(p.DeltaTime * (length(own_velocity.Linear - other_velocity.Linear) + length(p.Gravity) * p.DeltaTime),
                    p.MaxContactReach);
#endif

            const uint target_leaf_count = other_body_shape.Kind == ShapeCompound ? other_body_shape.VertexCount : 1;
            for (uint target_leaf = 0; target_leaf < target_leaf_count; ++target_leaf) {
#if PREPARE_QUERIES
                if (lane == 0) context_base = NoIndex;
                if (COLLECT_LANES > 1) threadgroup_barrier(mem_flags::mem_threadgroup);
#endif
                Shape target = shapes[other_body_shape.Kind == ShapeCompound ? ChildOf(other_body_shape, target_leaf, compound_children) : other_shape];
                if ((MESH_SHAPES && shape.Kind == ShapeMesh) && target.Kind != ShapePlane && (!MESH_PAIRS || (!MESH_SHAPES || target.Kind != ShapeMesh))) continue;
                if (other_body_shape.Kind == ShapeCompound) target.Local = ComposePose(other_body_shape.Local, target.Local);
                if (!Allows(leaf_filter, ResolveFilter(target, ResolveFilter(other_body_shape, {theirs.Layer, theirs.Collides})))) continue;
                if (theirs.Mixed && InternalFaces(target) != 0) target.FirstTriangle = 0;
                const bool mesh_pair = MESH_PAIRS && (MESH_SHAPES && shape.Kind == ShapeMesh) && (MESH_SHAPES && target.Kind == ShapeMesh);
                if (ConvexLeaf(target.Kind) || mesh_pair) {
#if SENSOR_PASS
                    if (other < body) continue;
#else
                    if (!i_move || (they_move && other < body)) continue;
#endif
                }
                const Poly own_bounds = mesh_pair ? MeshBounds(shape, shape_pose, bvh_nodes) : leaf_poly;
                const Poly target_bounds = mesh_pair ? MeshBounds(target, ComposePose(target_pose, target.Local), bvh_nodes) : leaf_poly;
                uint2 own_candidates[MaxMeshTriangles], own_walk[2 * MeshStackDepth];
                uint own_depth = mesh_pair ? 1 : 0;
                own_walk[0] = uint2(0);
                for (bool own_walking = true; own_walking;) {
                    uint own_count = 1;
                    if (mesh_pair) {
                        const float4 inverse = QuatConjugate(target_bounds.Orientation);
                        const float3 offset = Rotate(inverse, shape_pose.Position - ComposePose(target_pose, target.Local).Position);
                        const float3x3 rotation = QuatToMatrix(QuatMul(inverse, shape_pose.Orientation));
                        own_count = GatherMeshPairs(shape, target, offset, rotation, reach, bvh_nodes, mesh_triangles, hull_vertices, cached_pair, lane, own_walk, own_depth, own_candidates);
                        own_walking = own_depth > 0;
                    } else {
                        own_walking = false;
                    }
                    for (uint own_base = 0; own_base < own_count; own_base += mesh_pair ? COLLECT_LANES : 1) {
                        // Padded mesh-pair lanes repeat the last query to keep traversal uniform.
// Collection discards padded results.
                        const uint own_at = mesh_pair ? min(own_base + lane, own_count - 1) : own_base;
                        const Index own_triangle = mesh_pair ? shape.FirstTriangle + own_candidates[own_at].x : NoIndex;
                        Poly own_poly = mesh_pair ? MakeTriangle(shape_pose, mesh_triangles[own_triangle]) : leaf_poly;
                        bool plane_back = false;
                        if (target.Kind == ShapePlane) {
                            const Pose local = ComposePose(target_pose, target.Local);
                            target.Normal = Rotate(local.Orientation, target.Normal);
                            target.Offset += dot(target.Normal, local.Position);
                            float3 centre = shape_pose.Position;
                            if ((MESH_SHAPES && shape.Kind == ShapeMesh) && target.DoubleSided) {
                                const BvhNode bounds = bvh_nodes[shape.RootNode];
                                centre = WorldPoint(shape_pose, (bounds.Low + bounds.High) * 0.5f);
                            }
                            plane_back = target.DoubleSided && dot(target.Normal, centre) < target.Offset;
                            // Retain the previous side during mesh-plane penetration.
                            // The vertex feature encodes the side to invalidate the dual when it changes.
                            if ((MESH_SHAPES && shape.Kind == ShapeMesh) && target.DoubleSided && abs(dot(target.Normal, centre) - target.Offset) <= reach) {
                                for (uint j = 0; j < ContactsPerBody && history.was_feature[j] != NoIndex; ++j) {
                                    if (history.was_other[j] != other || history.was_children[j] != ChildPair(own_leaf, target_leaf)) continue;
                                    plane_back = (history.was_feature[j] & PlaneBackFeature) != 0;
                                    break;
                                }
                            }
                            if (plane_back) {
                                target.Normal = -target.Normal;
                                target.Offset = -target.Offset;
                            }
                            if ((MESH_SHAPES && shape.Kind == ShapeMesh)) {
                                const BvhNode bounds = bvh_nodes[shape.RootNode];
                                const float3 axis = Rotate(QuatConjugate(shape_pose.Orientation), target.Normal);
                                const float offset = target.Offset - dot(target.Normal, shape_pose.Position);
                                const float nearest = dot(axis, (bounds.Low + bounds.High) * 0.5f) - dot(abs(axis), (bounds.High - bounds.Low) * 0.5f);
                                const float roundoff = 1e-5f * max(1.f, abs(target.Offset) + length(shape_pose.Position) + length(bounds.Low) + length(bounds.High));
                                if (nearest > offset + reach + roundoff) continue;
                            }
                        }
                        const Pose target_shape_pose = ComposePose(target_pose, target.Local);
                        Poly plane_patch{};
                        float pair_reach = own_reach;
                        if (BoundedPlane(target)) {
                            Poly query = own_poly;
                            if ((MESH_SHAPES && shape.Kind == ShapeMesh)) {
                                const BvhNode bounds = bvh_nodes[shape.RootNode];
                                query.Kind = ShapeBox;
                                query.Center = WorldPoint(shape_pose, (bounds.Low + bounds.High) * 0.5f);
                                query.Half = (bounds.High - bounds.Low) * 0.5f;
                            }
                            if (!PlanePatch(query, target_shape_pose, target, hull_vertices, reach, plane_patch, geometry_lane)) continue;
                            pair_reach = PolyReach(query, hull_vertices, geometry_lane) + query.Radius;
                            plane_patch.Center -= shape_pose.Position;
                        } else if (own_reach < 0) {
                            pair_reach = PolyReach(own_poly, hull_vertices, mesh_pair ? NoIndex : geometry_lane);
                            if (!mesh_pair) own_reach = pair_reach;
                        }

                        const float geometry = max(pair_reach, BoundedPlane(target) ? PolyReach(plane_patch, hull_vertices) : PolyReach(MakePoly(target_shape_pose, target), hull_vertices, mesh_pair ? NoIndex : geometry_lane) + target.Radius);
                        const float weld = 1e-3f * geometry + 1e-6f;

                        uint candidates[MaxMeshTriangles], walk[MeshStackDepth], depth = 0;
                        float3 low = 0, high = 0;
                        if ((MESH_SHAPES && target.Kind == ShapeMesh) && !mesh_pair) {
                            PolyBounds(own_poly, target_shape_pose, hull_vertices, low, high, geometry_lane);
                            const float let_out = own_poly.Radius + reach;
                            low -= let_out;
                            high += let_out;
                            walk[depth++] = 0;
                        } else if (BoundedPlane(target) && (MESH_SHAPES && shape.Kind == ShapeMesh)) {
                            walk[depth++] = 0;
                        }

                        for (bool walking = true; walking;) {
                            uint manifolds = 1;
                            if (mesh_pair) {
                                candidates[0] = own_candidates[own_at].y;
                                walking = false;
                            } else if ((MESH_SHAPES && target.Kind == ShapeMesh)) {
                                const float4 inverse = QuatConjugate(own_poly.Orientation);
                                const float3 offset = Rotate(inverse, target_shape_pose.Position - own_poly.Center);
                                const float3x3 rotation = QuatToMatrix(QuatMul(inverse, target_shape_pose.Orientation));
                                manifolds = GatherTriangles(target, low, high, own_poly, offset, rotation, hull_faces, reach, bvh_nodes, walk, depth, candidates, lane);
                                walking = depth > 0;
                                if (manifolds == 0) break;
                            } else if (BoundedPlane(target) && (MESH_SHAPES && shape.Kind == ShapeMesh)) {
                                const float4 inverse = QuatConjugate(plane_patch.Orientation);
                                const float3 offset = Rotate(inverse, -plane_patch.Center);
                                const float3x3 rotation = QuatToMatrix(QuatMul(inverse, shape_pose.Orientation));
                                manifolds = GatherTriangles(shape, float3(-INFINITY), float3(INFINITY), plane_patch, offset, rotation, hull_faces, reach, bvh_nodes, walk, depth, candidates, lane);
                                walking = depth > 0;
                                if (manifolds == 0) break;
                            } else {
                                walking = false;
                            }

                            for (uint batch = 0; batch < manifolds; batch += COLLECT_LANES) {
#if PREPARE_QUERIES
                                const uint query_count = min(uint(COLLECT_LANES), mesh_pair ? own_count - own_base : manifolds - batch);
                                if (lane == 0) {
                                    query_base = ReserveQueries((device atomic_uint *)&QueryHeader(query_pool).TaskCount, QueryHeader(query_pool).TaskCapacity, query_count);
                                    first_context = context_base == NoIndex;
                                    if (first_context) context_base = ReserveQueries((device atomic_uint *)(&QueryHeader(query_pool).ContextCount), QueryHeader(query_pool).ContextCapacity, 1);
                                    batch_base = ReserveQueries((device atomic_uint *)(&QueryHeader(query_pool).BatchCount), QueryHeader(query_pool).BatchCapacity, 1);
                                    query_failed = query_base == NoIndex || context_base >= QueryHeader(query_pool).ContextCapacity || batch_base >= QueryHeader(query_pool).BatchCapacity;
                                    if (query_failed) atomic_fetch_add_explicit((device atomic_uint *)(query_pool + QueryOwnerOffset(p.BodyCount) + body), query_count, memory_order_relaxed);
                                    else {
                                        if (query_tail == NoIndex) query_pool[QueryHeadOffset(p.BodyCount, body, partition)] = batch_base;
                                        else query_batches[query_tail].Next = batch_base;
                                        query_tail = batch_base;
                                    }
                                }
                                if (COLLECT_LANES > 1) threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
                                if (lane < query_count && query_base != NoIndex && query_base + lane < QueryHeader(query_pool).TaskCapacity) query_records[query_base + lane].Batch = NoIndex;
                                // Use a group-local exit decision to keep barriers uniform when other partitions fail concurrently.
                                if (query_failed) return;
#endif
                                const uint manifold = mesh_pair ? 0 : batch + lane;
                                const bool cooperate = COLLECT_LANES >= 32 && ConvexLeaf(shape.Kind) && ConvexLeaf(target.Kind) && (shape.Kind == ShapeHull || target.Kind == ShapeHull);
                                const GeometryQuery query{shape, target, pose, target_pose, shape_pose, target_shape_pose, own_poly, plane_patch, own_bounds.Center, target_bounds.Center, own_velocity, other_velocity, reach, weld, mesh_pair, plane_back, bool(p.ReportContacts)};
                                if (manifold < manifolds || (cooperate && lane < 32)) {
                                    const ulong children = ChildPair(own_leaf, target_leaf);
                                    uint previous = NoIndex;
                                    if (mesh_pair) {
                                        const Index sub_shape = target.FirstTriangle + candidates[manifold];
                                        for (uint j = 0; j < ContactsPerBody && history.was_feature[j] != NoIndex; ++j)
                                            if (history.was_other[j] == other && history.was_children[j] == children && history.was_sub_a[j] == own_triangle && history.was_sub[j] == sub_shape) {
                                                previous = history.was_feature[j];
                                                break;
                                            }
                                    }
                                    const uint candidate = mesh_pair || (MESH_SHAPES && target.Kind == ShapeMesh) || (BoundedPlane(target) && (MESH_SHAPES && shape.Kind == ShapeMesh)) ? candidates[manifold] : 0;
#if PREPARE_QUERIES
                                    if (lane < query_count && query_base != NoIndex && query_base + query_count <= QueryHeader(query_pool).TaskCapacity && context_base < QueryHeader(query_pool).ContextCapacity && batch_base < QueryHeader(query_pool).BatchCapacity) {
                                        if (lane == 0) {
                                            if (first_context) query_contexts[context_base] = {query, body, other, own_leaf, target_leaf, uint(cached_pair)};
                                            query_batches[batch_base] = {context_base, query_base, query_count, NoIndex, {0, 0}};
                                        }
                                        query_records[query_base + lane] = {batch_base, candidate, own_triangle, previous, weld, NoIndex};
                                    }
#else
                                    geometries[lane] = QueryGeometry(query, candidate, own_triangle, previous, hull_vertices, hull_faces, mesh_triangles, cooperate ? lane % 32 : NoIndex);
#endif
                                }
                                if (COLLECT_LANES > 1) threadgroup_barrier(mem_flags::mem_threadgroup);
#if !PREPARE_QUERIES
                                if (lane == 0) {
                                    for (uint query_at = 0; query_at < min(uint(COLLECT_LANES), mesh_pair ? own_count - own_base : manifolds - batch); ++query_at) {
                                        const GeometryManifold geometry = geometries[query_at];
                                        CollectManifold(geometry, query, body, other, own_leaf, target_leaf, cached_pair, body_shape, other_body_shape, materials, slots, contact_refusals, p, count, inherited, history);
                                    }
                                }
#endif
                                if (COLLECT_LANES > 1) threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
                            }
                        }
                    }
                }
            }
        }
    }

#endif

    if (PREPARE_QUERIES) return;
#if !SENSOR_PASS && COLLECT_LANES >= 32
    threadgroup_barrier(mem_flags::mem_device);
    if (lane >= 32) return;
    const uint final_count = simd_shuffle(count, 0);
    constexpr uint final_stride = 32;
#else
    if (lane != 0) return;
    const uint final_count = count;
    constexpr uint final_stride = 1;
#endif
#if !SENSOR_PASS
    for (uint first = 0; lane == 0 && first < count;) {
        const Index other = slots[first].BodyB;
        uint end = first + 1;
        while (end < count && slots[end].BodyB == other) ++end;
        atomic_fetch_add_explicit((device atomic_uint *)&incoming[other].Count, end - first, memory_order_relaxed);
        first = end;
    }
    // Divide the pair's inertial stiffness among points sharing a normal to avoid multiplying support stiffness.
    for (uint k = lane; k < final_count; k += final_stride) {
        device Contact &contact = slots[k];
        uint points = 0;
        for (uint j = 0; j < final_count; ++j)
            points += slots[j].BodyB == contact.BodyB && dot(slots[j].Normal, contact.Normal) > 0.99999f;
        const uint other = contact.BodyB;
        float inverse = own_inverse_mass + masses[other].InvMass;
        if (inverse == 0) {
            const float3 a = cross(Rotate(pose.Orientation, contact.AnchorA), contact.Normal);
            const float3 b = cross(Rotate(poses[other].Orientation, contact.AnchorB), contact.Normal);
            inverse = dot(a, WorldInverseInertia(pose.Orientation, masses[body].InvInertiaLocal) * a) +
                dot(b, WorldInverseInertia(poses[other].Orientation, masses[other].InvInertiaLocal) * b);
        }
        contact.StiffnessScale = PairStiffness(inverse, p.DeltaTime) / float(points);
        contact.Penalty.x = max(contact.Penalty.x, contact.StiffnessScale);
    }
    if (lane != 0) return;
    // Emit events after replacement and welding finish so only retained contacts are reported.
    for (uint k = 0; k < count; ++k) {
        if (inherited[k] != NoIndex) claimed |= 1ul << inherited[k];
        events[reported++] = ContactEvent{body, slots[k].BodyB, slots[k].Feature, slots[k].SubShape, slots[k].Children, uint(inherited[k] != NoIndex ? ContactPersisted : ContactAdded), slots[k].SubShapeA};
    }
    EndUnclaimed(events, contact_event_counts, body, claimed, reported, history.was_feature, history.was_other, history.was_sub, history.was_sub_a, history.was_children);
#endif
}

static float4 JointFrame(float4 orientation, float4 frame) { return QuatMul(orientation, frame); }

static float4 RelativeFrame(float4 frame_a, float4 frame_b) { return QuatMul(QuatConjugate(frame_b), frame_a); }

static float4 TwistPart(float4 relative, float3 axis) {
    const float4 along = MakeFloat4(dot(relative.xyz, axis) * axis, relative.w);
    const float size = length(along);

    return size > 1e-7f ? along / size : float4(0, 0, 0, 1);
}

static float TwistAngle(float4 relative, float3 axis, float near) {
    const float4 twist = TwistPart(relative, axis);
    const float turn = 2 * atan2(dot(twist.xyz, axis), twist.w);
    const float full = 2 * M_PI_F;
    return near + (turn - near) - full * round((turn - near) / full);
}

static float3 AngularError(float4 relative, uint twist_axis, float unwrapped) {
    if (twist_axis > 2) return RotationVector(relative);
    const float3 axis = UnitAxis(twist_axis);
    float3 error = RotationVector(QuatMul(relative, QuatConjugate(TwistPart(relative, axis))));
    error[twist_axis] = unwrapped;
    return error;
}

static float3 TwistGradient(float4 relative, float3 axis, float projected) {
    const float denominator = relative.w * relative.w + projected * projected;
    return denominator > 1e-8f ?
        (relative.w * relative.w * axis + relative.w * cross(relative.xyz, axis) + projected * relative.xyz) / denominator :
        axis;
}

static float3 AngularGradient(float4 relative, uint twist_axis, uint row) {
    const float3 basis = UnitAxis(row);
    if (twist_axis > 2) return LogGradient(RotationVector(relative), basis);
    const float3 axis = UnitAxis(twist_axis);
    const float3 twist_gradient = TwistGradient(relative, axis, dot(relative.xyz, axis));
    if (row == twist_axis) return twist_gradient;
    const float4 swing = QuatMul(relative, QuatConjugate(TwistPart(relative, axis)));
    const float3 gradient = LogGradient(RotationVector(swing), basis);
    return gradient - twist_gradient * dot(Rotate(swing, axis), gradient);
}

// Remeasure joint rows at the current iterate without advancing the stored twist.
struct JointMeasure {
    float4 FrameB;
    float3 Reach, Error, Turned;
    float4 Relative;
};

template<typename JointPointer>
static JointMeasure MeasureJoint(JointPointer joint, Pose a, Pose b, device const Pose *initial) {
    const float4 frame_b = JointFrame(b.Orientation, joint->FrameB);
    const float4 relative = RelativeFrame(JointFrame(a.Orientation, joint->FrameA), frame_b);
    const uint twist_axis = TwistAxis(joint->AngularModes);
    const float unwrapped = twist_axis <= 2 ? TwistAngle(relative, UnitAxis(twist_axis), joint->Twist) : 0;
    return {frame_b, WorldPoint(a, joint->AnchorA) - WorldPoint(b, joint->AnchorB), AngularError(relative, twist_axis, unwrapped), RotationVector(QuatMul(a.Orientation, QuatConjugate(initial[joint->BodyA].Orientation))) - RotationVector(QuatMul(b.Orientation, QuatConjugate(initial[joint->BodyB].Orientation))), relative};
}

struct AxisSetup {
    uint Mode;
    float Stiffness, Damping, Speed, Target, MaxForce, Low, High;
    float3 Axis;
    // Value is the current coordinate; Began is its value at the start of the step.
    float Value, Began, Moved;
    float Lambda, Penalty;
};

// Six base rows followed by six independent drives, each in linear XYZ then angular XYZ order.
template<typename JointPointer>
static AxisSetup JointRowAt(JointPointer joint, JointMeasure measured, uint row) {
    const uint r = row % 3;
    const bool linear = row % 6 < 3;
    const bool drive = row >= 6;
    AxisSetup setup;
    if (!drive) setup = linear ? AxisSetup{AxisMode(joint->LinearModes, r), joint->LinearStiffness[r], joint->LinearDamping[r], joint->LinearMotorSpeed[r], joint->LinearMotorTarget[r], joint->LinearMotorMaxForce[r], joint->LinearLimitLow[r], joint->LinearLimitHigh[r]} : AxisSetup{AxisMode(joint->AngularModes, r), joint->AngularStiffness[r], joint->AngularDamping[r], joint->MotorSpeed[r], joint->MotorTarget[r], joint->MotorMaxTorque[r], joint->LimitLow[r], joint->LimitHigh[r]};
    else {
        const JointDrive d = joint->Drives[row - 6];
        setup = AxisSetup{d.Enabled ? uint(AxisPositioned) : uint(AxisFree), d.Stiffness, d.Damping, d.Speed, d.Target, d.MaxForce, 0, 0};
    }
    setup.Lambda = drive ? joint->Drives[row - 6].Lambda : (linear ? joint->LambdaLinear[r] : joint->LambdaAngular[r]);
    setup.Penalty = drive ? joint->Drives[row - 6].Penalty : (linear ? joint->PenaltyLinear[r] : joint->PenaltyAngular[r]);
    const float3 frame_axis = Rotate(measured.FrameB, UnitAxis(r));
    setup.Axis = linear ? frame_axis : Rotate(measured.FrameB, AngularGradient(measured.Relative, TwistAxis(joint->AngularModes), r));
    setup.Value = linear ? dot(measured.Reach, setup.Axis) : measured.Error[r];
    setup.Began = drive ? joint->Drives[row - 6].Began : (linear ? joint->C0Linear[r] : joint->C0Angular[r]);
    setup.Moved = linear ? setup.Value - setup.Began : dot(measured.Turned, frame_axis);
    const uint mask = drive ? 0 : (linear ? joint->LinearLimitAxes[r] : joint->AngularLimitAxes[r]);
    if (mask && (!linear || popcount(mask) > 1)) {
        float3 local = linear ? Rotate(QuatConjugate(measured.FrameB), measured.Reach) : RotationVector(measured.Relative);
        if (linear || mask == 7) {
            for (uint i = 0; linear && i < 3; ++i)
                if (!(mask & (1u << i))) local[i] = 0;
            setup.Value = length(local);
            setup.Axis = Rotate(measured.FrameB, setup.Value > 1e-8f ? local / setup.Value : UnitAxis(r));
        } else if (popcount(mask) == 1) {
            float4 q = measured.Relative;
            if (q.w < 0) q = -q;
            setup.Value = 2 * atan2(q[r], q.w);
            setup.Axis = Rotate(measured.FrameB, TwistGradient(q, UnitAxis(r), q[r]));
        } else {
            const float3 axis = UnitAxis(ctz(7u ^ mask));
            const float3 turned = Rotate(measured.Relative, axis);
            const float3 normal = cross(axis, turned);
            const float sine = length(normal);
            setup.Value = atan2(sine, clamp(dot(axis, turned), -1.f, 1.f));
            setup.Axis = Rotate(measured.FrameB, sine > 1e-8f ? normal / sine : UnitAxis(r));
        }
        setup.Moved = setup.Value - setup.Began;
    }
    return setup;
}

static bool JointRow(
    AxisSetup axis, float dt, thread float &c, thread float &damped, thread float &low, thread float &high
) {
    const float alpha = IsHard(axis.Stiffness) ? ConstraintAlpha : 0;
    c = axis.Value - axis.Began * alpha;
    // Damping acts on displacement relative to the requested velocity over this step.
    damped = axis.Value - axis.Began;
    low = -INFINITY;
    high = INFINITY;
    if (axis.Mode == AxisDriven) {
        c = axis.Moved - axis.Speed * dt;
        damped = c;
        low = -axis.MaxForce;
        high = axis.MaxForce;
    } else if (axis.Mode == AxisPositioned) {
        c -= axis.Target * (1 - alpha);
        damped -= axis.Speed * dt;
        low = -axis.MaxForce;
        high = axis.MaxForce;
    } else if (axis.Mode == AxisLimited) {
        // Choose a limit side from the initial pose so the row cannot switch stops during a solve.
        if (axis.Began > axis.High) {
            c -= axis.High * (1 - alpha);
            low = 0;
        } else if (axis.Began < axis.Low) {
            c -= axis.Low * (1 - alpha);
            high = 0;
        } else {
            return false;
        }
    }
    return true;
}

// Combine elastic and viscous forces with the dual in the shifted force interval.
// The stiffness returned to the body block follows the unclamped force.
static float RowForce(
    float penalty, float rate, float c, float damped, float lambda, float low, float high, thread float &stiffness
) {
    const float requested = penalty * c + lambda + rate * damped;
    const float force = clamp(requested, low, high);
    stiffness = penalty + rate;
    if (requested != force && abs(c) > 1e-9f) stiffness = clamp((force - lambda) / c, 0.f, penalty + rate);
    return force;
}

static void PrepareJointsBody(
    device Joint *joints, device const Pose *poses,
    constant StepParams &p, uint index
) {
    if (index >= p.JointCount) return;
    device Joint &joint = joints[index];
    if (!joint.Active) return;
    const Pose a = poses[joint.BodyA], b = poses[joint.BodyB];

    const float4 frame_b = JointFrame(b.Orientation, joint.FrameB);
    const float3 reach = WorldPoint(a, joint.AnchorA) - WorldPoint(b, joint.AnchorB);
    for (uint r = 0; r < 3; ++r) joint.C0Linear[r] = dot(reach, Rotate(frame_b, UnitAxis(r)));
    // Advance accumulated twist once per step.
    const float4 relative = RelativeFrame(JointFrame(a.Orientation, joint.FrameA), frame_b);
    const uint twist_axis = TwistAxis(joint.AngularModes);
    if (twist_axis <= 2) joint.Twist = TwistAngle(relative, UnitAxis(twist_axis), joint.Twist);
    joint.C0Angular = AngularError(relative, twist_axis, joint.Twist);
    const JointMeasure measure{frame_b, reach, joint.C0Angular, float3(0), relative};
    const Joint before = joint;
    for (uint r = 0; r < 3; ++r) {
        if (popcount(joint.LinearLimitAxes[r]) > 1) joint.C0Linear[r] = JointRowAt(&before, measure, r).Value;
        if (joint.AngularLimitAxes[r]) joint.C0Angular[r] = JointRowAt(&before, measure, r + 3).Value;
    }
    for (uint r = 0; r < 6; ++r) {
        device JointDrive &drive = joint.Drives[r];
        drive.Began = r < 3 ? dot(reach, Rotate(frame_b, UnitAxis(r))) : before.C0Angular[r - 3];
        drive.Penalty = min(clamp(drive.Penalty * p.Gamma, p.PenaltyMin, p.PenaltyMax), drive.Stiffness);
        if (!IsHard(drive.Stiffness)) drive.Lambda = 0;
    }

    joint.PenaltyLinear = min(clamp(joint.PenaltyLinear * p.Gamma, p.PenaltyMin, p.PenaltyMax), joint.LinearStiffness);
    joint.PenaltyAngular = min(clamp(joint.PenaltyAngular * p.Gamma, p.PenaltyMin, p.PenaltyMax), joint.AngularStiffness);

    for (uint r = 0; r < 3; ++r) {
        if (!IsHard(joint.LinearStiffness[r])) joint.LambdaLinear[r] = 0;
        if (!IsHard(joint.AngularStiffness[r])) joint.LambdaAngular[r] = 0;
    }
}

kernel void PrepareJoints(
    device Joint *joints [[buffer(16)]], device const Pose *poses [[buffer(0)]],
    constant StepParams &p [[buffer(7)]], uint index [[thread_position_in_grid]]
) {
    PrepareJointsBody(joints, poses, p, index);
}

static bool LargeIsland(device const uint *pool, constant StepParams &p, uint body) {
    uint root = pool[IslandHeaderWords + body];
    while (root != pool[IslandHeaderWords + root]) root = pool[IslandHeaderWords + root];
    return pool[IslandHeaderWords + p.BodyCount + root] > IslandBodyLimit;
}

// A joint with two unsolved endpoints still updates its dual once, owned by A.
static uint JointDualOwner(device const Joint &joint, device const BodyMass *masses, device const uint *quiet, constant StepParams &p) {
    return Solved(masses[joint.BodyA], quiet[joint.BodyA], p) ? joint.BodyA :
        Solved(masses[joint.BodyB], quiet[joint.BodyB], p)    ? joint.BodyB :
                                                                joint.BodyA;
}

static bool UpdateJointDual(
    device Joint *joints, device const Pose *poses,
    device const Pose *initial, constant StepParams &p,
    threadgroup float2 *updated, uint index, uint lane
) {
    if (index >= p.JointCount) return false;
    device Joint &joint = joints[index];
    if (!joint.Active) return false;
    bool changed = false;
    if (lane < 12) {
        device const Joint &state = joint;
        const JointMeasure measured = MeasureJoint(&state, poses[state.BodyA], poses[state.BodyB], initial);
        const uint row = lane;
        const AxisSetup setup = JointRowAt(&state, measured, row);

        float lambda = setup.Lambda, penalty = setup.Penalty;
        const uint old_lambda = as_type<uint>(lambda), old_penalty = as_type<uint>(penalty);
        float c, damped, low, high;

        if (setup.Mode == AxisFree || !JointRow(setup, p.DeltaTime, c, damped, low, high)) {
            lambda = 0;
        } else if (!IsHard(setup.Stiffness)) {
            // Soft rows have no dual and cap the penalty at material stiffness.
            penalty = min(penalty + p.Beta * abs(c), setup.Stiffness);
        } else {
            const float damping_force = setup.Damping / p.DeltaTime * damped;
            const float requested = penalty * c + lambda + damping_force;
            lambda = clamp(requested, low, high) - damping_force;

            if (requested > low && requested < high) penalty = min(penalty + p.Beta * abs(c), p.PenaltyMax);
        }
        changed = old_lambda != as_type<uint>(lambda) || old_penalty != as_type<uint>(penalty);
        updated[row] = float2(lambda, penalty);
    }
    const bool any_changed = simd_any(changed);
    // All rows finish reading the joint before its single writer publishes the updates.
    simdgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    if (lane != 0) return false;
    for (uint row = 0; row < 12; ++row) {
        const uint r = row % 3;
        const bool linear = row % 6 < 3;
        const float lambda = updated[row].x, penalty = updated[row].y;
        if (row >= 6) {
            joint.Drives[row - 6].Lambda = lambda;
            joint.Drives[row - 6].Penalty = penalty;
        } else if (linear) {
            joint.LambdaLinear[r] = lambda;
            joint.PenaltyLinear[r] = penalty;
        } else {
            joint.LambdaAngular[r] = lambda;
            joint.PenaltyAngular[r] = penalty;
        }
    }
    return any_changed;
}

kernel void UpdateJointDuals(
    device Joint *joints [[buffer(16)]], device const Pose *poses [[buffer(0)]],
    device const Pose *initial [[buffer(1)]], constant StepParams &p [[buffer(7)]],
#if MIXED_ISLAND_SOLVE
    device const uint *islands [[buffer(25)]], device const BodyMass *masses [[buffer(4)]], device const uint *quiet [[buffer(21)]],
#endif
    uint index [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]
) {
#if MIXED_ISLAND_SOLVE
    if (p.BodyCount > SolveLanes && islands[0]) {
        if (index >= p.JointCount || !joints[index].Active) return;
        device const Joint &joint = joints[index];
        const uint owner = JointDualOwner(joint, masses, quiet, p);
        if (!LargeIsland(islands, p, owner)) return;
    }
#endif
    threadgroup float2 updated[12];
    UpdateJointDual(joints, poses, initial, p, updated, index, lane);
}

kernel void ScanIncoming(
    device Adjacency *incoming [[buffer(17)]], device uint *blocks [[buffer(14)]],
    constant StepParams &p [[buffer(7)]], uint body [[thread_position_in_grid]],
    uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]],
    uint width [[threads_per_threadgroup]]
) {
    const uint count = incoming[body].Count;
    uint prefix = simd_prefix_exclusive_sum(count);
    const uint total = simd_sum(count);
    threadgroup uint waves[RadixWaves];
    if (lane == 0) waves[simd] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint wave = 0; wave < simd; ++wave) prefix += waves[wave];
    incoming[body].Start = prefix;
    incoming[body].Cursor = prefix;
    if (tid == 0) {
        uint sum = 0;
        for (uint wave = 0; wave < (width + RadixSimdWidth - 1) / RadixSimdWidth; ++wave) sum += waves[wave];
        blocks[group] = sum;
    }
}

kernel void ScanIncomingBlocks(
    device uint *blocks [[buffer(14)]], constant StepParams &p [[buffer(7)]],
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]
) {
    const uint count = RadixBlocks(p.BodyCount), chunk = (count + RadixBlockSize - 1) / RadixBlockSize;
    const uint begin = tid * chunk, end = min(begin + chunk, count);
    uint total = 0;
    for (uint i = begin; i < end; ++i) total += blocks[i];
    uint prefix = simd_prefix_exclusive_sum(total);
    const uint wave_total = simd_sum(total);
    threadgroup uint waves[RadixWaves];
    if (lane == 0) waves[simd] = wave_total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint wave = 0; wave < simd; ++wave) prefix += waves[wave];
    for (uint i = begin; i < end; ++i) {
        const uint value = blocks[i];
        blocks[i] = prefix;
        prefix += value;
    }
}

kernel void OffsetIncoming(
    device Adjacency *incoming [[buffer(17)]], device const uint *blocks [[buffer(14)]],
    constant StepParams &p [[buffer(7)]], uint body [[thread_position_in_grid]]
) {
    if (body >= p.BodyCount) return;
    const uint start = incoming[body].Start + blocks[body / RadixBlockSize];
    incoming[body].Start = start;
    incoming[body].Cursor = start;
}

static void FillIncomingBody(
    device Adjacency *incoming, device uint *slots,
    device const Contact *contacts, constant StepParams &p,
    uint body
) {
    if (body >= p.BodyCount) return;
    for (uint i = 0; i < ContactsPerBody; ++i) {
        const uint slot = body * ContactsPerBody + i;
        const Contact contact = contacts[slot];
        if (!contact.Active) break;
        device atomic_uint *cursor = (device atomic_uint *)&incoming[contact.BodyB].Cursor;
        slots[atomic_fetch_add_explicit(cursor, 1u, memory_order_relaxed)] = slot;
    }
}

kernel void FillIncoming(
    device Adjacency *incoming [[buffer(17)]], device uint *slots [[buffer(18)]],
    device const Contact *contacts [[buffer(5)]], constant StepParams &p [[buffer(7)]],
    uint body [[thread_position_in_grid]]
) {
    FillIncomingBody(incoming, slots, contacts, p, body);
}

// Solved bodies require contacts in slot order.
// Unsolved bodies use this list only for order-independent waking writes.
static void SortIncomingBody(
    device const Adjacency *incoming, device uint *slots,
    device const BodyMass *masses,
    constant StepParams &p, uint body
) {
    if (body >= p.BodyCount || !Moves(masses[body])) return;
    const uint start = incoming[body].Start, count = incoming[body].Count;
    for (uint i = 1; i < count; ++i) {
        const uint value = slots[start + i];
        uint j = i;
        for (; j > 0 && slots[start + j - 1] > value; --j) slots[start + j] = slots[start + j - 1];
        slots[start + j] = value;
    }
}

kernel void SortIncoming(
    device const Adjacency *incoming [[buffer(17)]], device uint *slots [[buffer(18)]],
    device const BodyMass *masses [[buffer(4)]],
    constant StepParams &p [[buffer(7)]], uint body [[thread_position_in_grid]]
) {
    SortIncomingBody(incoming, slots, masses, p, body);
}

static float NormalOffset(Contact contact, constant StepParams &p) {
#if STABILIZE
    return contact.C0[0];
#else
    return max(contact.C0[0] - p.ContactMargin, 0.f);
#endif
}

static float3 ContactConstraint(
    Contact contact, constant StepParams &p, ContactBasis basis, float3 arm_a, float3 arm_b,
    Displacement moved_a, Displacement moved_b
) {
    const float offset = NormalOffset(contact, p);
    float3 c;
    for (uint r = 0; r < 3; ++r) {
        const float3 axis = basis.Axis[r];
        c[r] = (r == 0 ? offset : contact.C0[r] * (1 - ConstraintAlpha)) +
            dot(axis, moved_a.Linear) + dot(cross(arm_a, axis), moved_a.Angular) -
            dot(axis, moved_b.Linear) - dot(cross(arm_b, axis), moved_b.Angular);
    }
    return c;
}

static float3 ContactForce(float3 requested, float friction) {
    float3 force = requested;
    force[0] = min(force[0], 0.f);
    const float bound = abs(force[0]) * friction;
    const float tangential = length(float2(force[1], force[2]));
    if (tangential > bound && tangential > 0) {
        force[1] *= bound / tangential;
        force[2] *= bound / tangential;
    }
    return force;
}

static Index ContactPartner(Contact contact, uint body) { return contact.BodyA == body ? contact.BodyB : contact.BodyA; }

static Index JointPartner(Joint joint, uint body, device const BodyMass *masses) {
    if (!joint.Active || (joint.BodyA != body && joint.BodyB != body)) return NoIndex;
    const Index other = joint.BodyA == body ? joint.BodyB : joint.BodyA;
    return Moves(masses[other]) ? other : NoIndex;
}

static uint ContactSlot(thread uint &i, uint own, uint start, device const uint *incoming_slots, device const Contact *contacts) {
    const uint slot = i < ContactsPerBody ? own + i : incoming_slots[start + i - ContactsPerBody];
    if (contacts[slot].Active) return slot;
    if (i < ContactsPerBody) i = ContactsPerBody - 1;
    return NoIndex;
}

static uint LowestFree(uint taken) {
    return ctz(~taken | (1u << 31));
}

static void NoteNeighbour(
    uint sweep, uint other, uint other_word, uint body, bool both_quiet,
    thread uint &degree, thread uint &taken, thread uint &taken_all
) {
    if (sweep == 0) {
        ++degree;
        return;
    }
    const uint held = 1u << min(ColorOf(other_word), 31u);
    taken_all |= held;
    if (Prioritized(other_word, other, degree, body, both_quiet)) taken |= held;
}

// Incremental coloring uses degree and body identity for deterministic priority.
// Unresolved conflicts retain snapshot reads and relaxed body updates.
static void UpdateColorsBody(
    device const uint *colors, device uint *next,
    device const Contact *contacts, device const BodyMass *masses,
    device const Joint *joints, device const Adjacency *incoming,
    device const uint *incoming_slots, device const uint *quiet,
    device const uint *joint_incidence,
    constant StepParams &p, uint body
) {
    if (body >= p.BodyCount) return;
    const uint mine = ColorOf(colors[body]);
    next[body] = colors[body];
    if (!Moves(masses[body])) return;

    if (Asleep(quiet[body], p)) return;
    const bool my_quiet = quiet[body] > 0;

    // Complete degree counts before comparing priorities.
    uint degree = 0, taken = 0, taken_all = 0;
    const Adjacency neighbours = incoming[body];
    for (uint sweep = 0; sweep < 2; ++sweep) {
        for (uint i = 0; i < ContactsPerBody + neighbours.Count; ++i) {
            const uint slot = ContactSlot(i, body * ContactsPerBody, neighbours.Start, incoming_slots, contacts);
            if (slot == NoIndex) continue;
            const Index other = ContactPartner(contacts[slot], body);
            if (!Moves(masses[other])) continue;
            NoteNeighbour(sweep, other, colors[other], body, my_quiet && quiet[other] > 0, degree, taken, taken_all);
        }

        for (uint at = joint_incidence[body]; at < joint_incidence[body + 1]; ++at) {
            const uint index = joint_incidence[at];
            const Index other = JointPartner(joints[index], body, masses);
            if (other == NoIndex) continue;
            NoteNeighbour(sweep, other, colors[other], body, my_quiet && quiet[other] > 0, degree, taken, taken_all);
        }
        degree = min(degree, MaxColorDegree);
    }

    uint chosen = mine;
    if ((taken & (1u << min(mine, 31u))) != 0) {
        const uint best = LowestFree(taken);
        if (best < p.MaxColors) chosen = best;
    } else if (my_quiet) {
        const uint best = LowestFree(taken_all);
        if (best < mine) chosen = best;
    }
    next[body] = (degree << ColorDegreeShift) | chosen;
}

kernel void UpdateColors(
    device const uint *colors [[buffer(12)]], device uint *next [[buffer(13)]],
    device const Contact *contacts [[buffer(5)]], device const BodyMass *masses [[buffer(4)]],
    device const Joint *joints [[buffer(16)]], device const Adjacency *incoming [[buffer(17)]],
    device const uint *incoming_slots [[buffer(18)]], device const uint *quiet [[buffer(21)]],
    device const uint *joint_incidence [[buffer(20)]],
    device ColorWork &color_work [[buffer(26)]],
    constant StepParams &p [[buffer(7)]], uint body [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]
) {
    if (p.BodyCount > SolveLanes && !color_work.ColoringActive) return;
    UpdateColorsBody(colors, next, contacts, masses, joints, incoming, incoming_slots, quiet, joint_incidence, p, body);
    // One store per changed SIMD group avoids a contended atomic from every body.
    const bool changed = simd_any(body < p.BodyCount && next[body] != colors[body]);
    if (p.BodyCount > SolveLanes && changed && lane == 0)
        atomic_store_explicit((device atomic_uint *)&color_work.ColoringChanged, 1u, memory_order_relaxed);
}

static void PublishColorsBody(
    device uint *colors, device const uint *next,
    constant StepParams &p, uint body
) {
    if (body >= p.BodyCount) return;
    colors[body] = next[body];
}

kernel void PublishColors(
    device uint *colors [[buffer(12)]], device const uint *next [[buffer(13)]],
    device ColorWork &color_work [[buffer(26)]],
    constant StepParams &p [[buffer(7)]], uint body [[thread_position_in_grid]]
) {
    PublishColorsBody(colors, next, p, body);
    if (p.BodyCount > SolveLanes && body == 0) {
        color_work.ColoringActive = color_work.ColoringChanged;
        color_work.ColoringChanged = 0;
    }
}

struct BodyContactRows {
    float3 Axis[3], Angular[3], Force, Stiffness;
    float Side;
    bool Shared, Valid;
};

struct BodyJointRows {
    float3 Arm;
    float Side;
    float3 Axis[2];
    float Force[2], Stiffness[2];
    bool Valid, Shared, MineIsA, Active[2];
};

static bool SolveBody(
    device const Pose *poses, device BodyIterate *iterates, device const Pose *initial,
    device const Pose *inertial, device const BodyMass *masses,
    device const Displacement *displacements,
    device const Contact *contacts, device const uint *colors,
    uint color, device const Joint *joints,
    device const Adjacency *incoming, device const uint *incoming_slots,
    device const uint *joint_incidence,
    device const uint *quiet, constant StepParams &p,
    uint body, uint lane
) {
    constexpr uint WorkLanes = SOLVE_BODIES_PER_GROUP == 1 ? 32 : Dof;
    constexpr uint JointsPerBatch = WorkLanes / Dof;
    if (lane >= WorkLanes * SOLVE_BODIES_PER_GROUP) return false;
    const uint worker = lane % WorkLanes;
    if (body >= p.BodyCount) return false;
    const BodyMass mass = masses[body];

    if (!Solved(mass, quiet[body], p) || ColorOf(colors[body]) % p.MaxColors != color) return false;

    const uint block_row = worker % Dof, base = lane - worker;
    Pose pose = poses[body];
    const Pose start = initial[body], target = inertial[body];
    const float inv_dt2 = 1 / (p.DeltaTime * p.DeltaTime);

    float2 H[Dof], g = 0;
#pragma unroll
    for (uint j = 0; j < Dof; ++j) H[j] = 0;

    const bool3 rigid = mass.InvInertiaLocal == float3(0);
    const float3 heavy = select(1 / select(mass.InvInertiaLocal, float3(1), rigid), float3(0), rigid);
    const float m = Translates(mass) ? 1 / mass.InvMass : 0;
    const float3 offset = pose.Position - target.Position;
#pragma unroll
    for (uint i = 0; i < 3; ++i) {
        if (block_row == i) {
            H[i] = WideMul(float2(m, 0), float2(inv_dt2, 0));
            g = WideMul(H[i], float2(offset[i], 0));
        }
    }
    const float3x3 rotation = QuatToMatrix(pose.Orientation);
    const float3x3 world_inertia = WorldTensor(rotation, heavy);
    const float3 twist = RotationVector(QuatMul(pose.Orientation, QuatConjugate(target.Orientation)));
    const float3 torque = world_inertia * twist * inv_dt2;
    if (block_row >= 3) {
        g = float2(torque[block_row - 3], 0);
#pragma unroll
        for (uint j = 0; j < 3; ++j) H[3 + j] = WideMul(float2(world_inertia[j][block_row - 3], 0), float2(inv_dt2, 0));
    }

    const Adjacency neighbours = incoming[body];
    bool shared_color = false;
    uint own_count = 0;
    while (own_count < ContactsPerBody && contacts[body * ContactsPerBody + own_count].Active) ++own_count;
    const uint contact_count = own_count + neighbours.Count;
    constexpr uint ContactsPerBatch = SOLVE_BODIES_PER_GROUP == 1 ? WorkLanes / Dof : WorkLanes;
    for (uint first = 0; first < contact_count; first += ContactsPerBatch) {
        const uint i = first + (SOLVE_BODIES_PER_GROUP == 1 ? worker / Dof : worker);
        const uint slot = i >= contact_count ? NoIndex : i < own_count ? body * ContactsPerBody + i :
                                                                         incoming_slots[neighbours.Start + i - own_count];
        BodyContactRows rows{};
        rows.Valid = (SOLVE_BODIES_PER_GROUP != 1 || worker < ContactsPerBatch * Dof) && slot != NoIndex && contacts[slot].Active;
        if (rows.Valid) {
            const Contact contact = contacts[slot];
            const bool mine_is_a = contact.BodyA == body;
            const Index other = ContactPartner(contact, body);
            rows.Shared = Solved(masses[other], quiet[other], p) && ColorOf(colors[other]) % p.MaxColors == color;

            // Contact anchors and axes remain fixed during sweeps; map angular displacement through LogGradient.
            const Pose start_a = mine_is_a ? start : initial[contact.BodyA];
            const Pose start_b = mine_is_a ? initial[contact.BodyB] : start;
            const float3 arm_a = Rotate(start_a.Orientation, contact.AnchorA);
            const float3 arm_b = Rotate(start_b.Orientation, contact.AnchorB);
            // Same-colored neighbours read the iteration's snapshot; other colors read the latest completed update.
            const Displacement moved_other = rows.Shared ? displacements[other] : iterates[other].Moved;
            const Displacement moved_a = mine_is_a ? displacements[body] : moved_other;
            const Displacement moved_b = mine_is_a ? moved_other : displacements[body];
            const float3 arm = mine_is_a ? arm_a : arm_b;
            const float side = mine_is_a ? 1 : -1;
            const ContactBasis basis = MakeContactBasis(contact.Normal);
            const float3 constraint = ContactConstraint(contact, p, basis, arm_a, arm_b, moved_a, moved_b);

            const float3 requested = contact.Penalty * constraint + contact.Lambda;
            const float3 force = ContactForce(requested, contact.Friction);

            // The normal secant stays in [0, penalty] to keep the body block definite.
            float3 stiffness = contact.Penalty;
            if (requested[0] != force[0] && abs(constraint[0]) >= 1e-9f)
                stiffness[0] = clamp((force[0] - contact.Lambda[0]) / constraint[0], 0.f, contact.Penalty[0]);
            // The friction secant is centered at C = -lambda / penalty, where the augmented force vanishes.
            // Its radial projection scales force and curvature together, including the cached dual.
            const float tangent = length(requested.yz);
            if (tangent > 0) stiffness.yz *= min(1.f, abs(force.x) * contact.Friction / tangent);

            rows.Side = side;
            rows.Force = force;
            rows.Stiffness = stiffness;
            for (uint r = 0; r < 3; ++r) {
                rows.Axis[r] = basis.Axis[r];
                rows.Angular[r] = LogGradient(mine_is_a ? moved_a.Angular : moved_b.Angular, cross(arm, basis.Axis[r]));
            }
        }
#if SOLVE_BODIES_PER_GROUP == 1
        // Normalize each addition, then merge blocks in contact order.
        float2 partial[Dof], partial_g = 0;
#pragma unroll
        for (uint j = 0; j < Dof; ++j) partial[j] = 0;
        if (rows.Valid)
            for (uint r = 0; r < 3; ++r)
                AddRow(partial, partial_g, block_row, rows.Axis[r], rows.Angular[r], rows.Side, rows.Force[r], rows.Stiffness[r]);
        for (uint prepared = 0; prepared < min(ContactsPerBatch, contact_count - first); ++prepared) {
            const uint source = base + prepared * Dof;
            if (!simd_shuffle(uint(rows.Valid), source)) continue;
            shared_color |= bool(simd_shuffle(uint(rows.Shared), source));
            MergePreparedBlock(H, g, partial, partial_g, source + block_row);
        }
#else
        for (uint prepared = 0; prepared < min(WorkLanes, contact_count - first); ++prepared) {
            const uint source = base + prepared;
            if (!simd_shuffle(uint(rows.Valid), source)) continue;
            shared_color |= bool(simd_shuffle(uint(rows.Shared), source));
            const float side = simd_shuffle(rows.Side, source);
            const float3 force = simd_shuffle(rows.Force, source), stiffness = simd_shuffle(rows.Stiffness, source);
            float2 partial[Dof], partial_g = 0;
#pragma unroll
            for (uint j = 0; j < Dof; ++j) partial[j] = 0;
            for (uint r = 0; r < 3; ++r)
                AddRow(partial, partial_g, block_row, simd_shuffle(rows.Axis[r], source), simd_shuffle(rows.Angular[r], source), side, force[r], stiffness[r]);
            g = WideAdd(g, partial_g);
#pragma unroll
            for (uint j = 0; j < Dof; ++j) H[j] = WideAdd(H[j], partial[j]);
        }
#endif
    }

    const uint joint_start = joint_incidence[body], joint_count = joint_incidence[body + 1] - joint_start;
    for (uint first_joint = 0; first_joint < joint_count; first_joint += JointsPerBatch) {
        BodyJointRows rows{};
        const uint at = first_joint + worker / Dof;
        const uint index = at < joint_count ? joint_incidence[joint_start + at] : NoIndex;
        if (worker < JointsPerBatch * Dof && index < p.JointCount) {
            device const Joint &joint = joints[index];
            rows.Valid = joint.Active && (joint.BodyA == body || joint.BodyB == body);
            if (rows.Valid) {
                const bool mine_is_a = joint.BodyA == body;
                const Index other_body = mine_is_a ? joint.BodyB : joint.BodyA;
                rows.Shared = Solved(masses[other_body], quiet[other_body], p) && ColorOf(colors[other_body]) % p.MaxColors == color;
                const Pose other = rows.Shared ? poses[other_body] : iterates[other_body].Current;
                const Pose a = mine_is_a ? pose : other, b = mine_is_a ? other : pose;
                const float side = mine_is_a ? 1 : -1;
                float3 arm = Rotate(pose.Orientation, mine_is_a ? joint.AnchorA : joint.AnchorB);
                const BodyMass other_mass = masses[mine_is_a ? joint.BodyB : joint.BodyA];

                // Hard rows use a minimum inertial stiffness during stabilization.
                const float linear_floor = Stabilizing * PairStiffness(mass.InvMass + other_mass.InvMass, p.DeltaTime);

                const JointMeasure measured = MeasureJoint(&joint, a, b, initial);
                // Rotation of B changes the measurement axes and the torque arm to A's anchor.
                if (!mine_is_a) arm += measured.Reach;
                const float3x3 inverse_inertia = WorldInverseInertia(a.Orientation, masses[joint.BodyA].InvInertiaLocal) +
                    WorldInverseInertia(b.Orientation, masses[joint.BodyB].InvInertiaLocal);

                rows.Arm = arm;
                rows.Side = side;
                rows.MineIsA = mine_is_a;
                for (uint family = 0; family < 2; ++family) {
                    const uint row = family * Dof + block_row;
                    const bool is_linear = row % 6 < 3;
                    const AxisSetup setup = JointRowAt(&joint, measured, row);
                    float c, damped, low, high;
                    const bool active = setup.Mode != AxisFree && JointRow(setup, p.DeltaTime, c, damped, low, high);
                    float force = 0, stiffness = 0;
                    if (active) {
                        const bool hard = IsHard(setup.Stiffness);

                        const float floored = is_linear ? linear_floor : Stabilizing * PairStiffness(dot(setup.Axis, inverse_inertia * setup.Axis), p.DeltaTime);
                        const float penalty = hard ? max(setup.Penalty, floored) : setup.Penalty;
                        const float lambda = hard ? setup.Lambda : 0;
                        force = RowForce(penalty, setup.Damping / p.DeltaTime, c, damped, lambda, low, high, stiffness);
                    }
                    rows.Axis[family] = setup.Axis;
                    rows.Active[family] = active;
                    rows.Force[family] = force;
                    rows.Stiffness[family] = stiffness;
                }
            }
        }
        for (uint prepared_joint = 0; prepared_joint < min(JointsPerBatch, joint_count - first_joint); ++prepared_joint) {
            const uint joint_source = base + prepared_joint * Dof;
            if (!simd_shuffle(uint(rows.Valid), joint_source)) continue;
            shared_color |= bool(simd_shuffle(uint(rows.Shared), joint_source));
            const bool mine_is_a = bool(simd_shuffle(uint(rows.MineIsA), joint_source));
            const float3 arm = simd_shuffle(rows.Arm, joint_source);
            const float side = simd_shuffle(rows.Side, joint_source);
            float3 applied{0, 0, 0};
            for (uint family = 0; family < 2; ++family) {
                for (uint prepared_row = 0; prepared_row < Dof; ++prepared_row) {
                    const uint source = joint_source + prepared_row;
                    if (!simd_shuffle(uint(rows.Active[family]), source)) continue;
                    const float3 axis = simd_shuffle(rows.Axis[family], source);
                    const float row_force = simd_shuffle(rows.Force[family], source), row_stiffness = simd_shuffle(rows.Stiffness[family], source);
                    if (prepared_row < 3) {
                        AddRow(H, g, block_row, axis, cross(arm, axis), side, row_force, row_stiffness);
                        applied += row_force * axis;
                    } else if (block_row >= 3) {
                        const uint i = block_row - 3;
                        g = WideAdd(g, WideMul(WideMul(float2(side, 0), float2(axis[i], 0)), float2(row_force, 0)));
#pragma unroll
                        for (uint j = 0; j < 3; ++j) H[3 + j] = WideAdd(H[3 + j], WideMul(WideMul(float2(row_stiffness, 0), float2(axis[i], 0)), float2(axis[j], 0)));
                    }
                }
            }
            // Joint geometric stiffness follows section 3.5; contact blocks omit that term.
            const float3x3 outer = float3x3(arm * applied.x, arm * applied.y, arm * applied.z);
            const float3x3 geometric = (mine_is_a ? outer : transpose(outer)) - float3x3(dot(arm, applied));
#pragma unroll
            for (uint i = 0; i < 3; ++i)
                if (block_row == 3 + i) H[3 + i] = WideAdd(H[3 + i], float2(length(geometric[i]), 0));
        }
    }
    if (worker >= Dof) return false;

    if (!Translates(mass)) {
        for (uint i = 0; i < 3; ++i) LockDirection(H, g, block_row, base, 0, UnitAxis(i));
    }
    for (uint i = 0; i < 3; ++i) {
        if (mass.InvInertiaLocal[i] == 0) LockDirection(H, g, block_row, base, 3, rotation[i]);
    }

    const float step = SolveBlock(H, g, block_row, base);
    const float3 linear{simd_shuffle(step, base + 0), simd_shuffle(step, base + 1), simd_shuffle(step, base + 2)};
    const float3 angular{simd_shuffle(step, base + 3), simd_shuffle(step, base + 4), simd_shuffle(step, base + 5)};
    if (block_row != 0) return false;
    if (!isfinite(linear.x + linear.y + linear.z + angular.x + angular.y + angular.z)) return false;
    // Half steps stabilize simultaneous updates when adjacent bodies share a color.
    const float relaxation = shared_color ? 0.5f : 1.f;
    pose.Position += relaxation * linear;
    pose.Orientation = normalize(QuatMul(QuatFromRotationVector(relaxation * angular), pose.Orientation));
    const BodyIterate before = iterates[body];
    const Displacement moved = Since(pose, start);
    const bool changed = any(as_type<uint3>(pose.Position) != as_type<uint3>(before.Current.Position)) ||
        any(as_type<uint4>(pose.Orientation) != as_type<uint4>(before.Current.Orientation)) ||
        any(as_type<uint3>(moved.Linear) != as_type<uint3>(before.Moved.Linear)) ||
        any(as_type<uint3>(moved.Angular) != as_type<uint3>(before.Moved.Angular));
    iterates[body] = {pose, moved};
    return changed;
}

kernel void SolveBodies(
    device const Pose *poses [[buffer(0)]], device BodyIterate *iterates [[buffer(11)]], device const Pose *initial [[buffer(1)]],
    device const Pose *inertial [[buffer(2)]], device const BodyMass *masses [[buffer(4)]],
    device const Displacement *displacements [[buffer(10)]],
    device const Contact *contacts [[buffer(5)]], device const uint *colors [[buffer(12)]],
    device const uint *cursor [[buffer(14)]], device const Joint *joints [[buffer(16)]],
    device const Adjacency *incoming [[buffer(17)]], device const uint *incoming_slots [[buffer(18)]],
    device const uint *joint_incidence [[buffer(20)]],
#if COMPACT_BODY_WORK
    device const uint *body_ids [[buffer(13)]], device const ColorWork &color_work [[buffer(26)]],
#endif
    device const uint *quiet [[buffer(21)]], constant StepParams &p [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint WorkLanes = SOLVE_BODIES_PER_GROUP == 1 ? 32 : Dof;
    if (lane >= WorkLanes * SOLVE_BODIES_PER_GROUP) return;
    const uint work = group * SOLVE_BODIES_PER_GROUP + lane / WorkLanes;
#if COMPACT_BODY_WORK
    const uint first = color_work.Offsets[cursor[0]], end = color_work.Offsets[cursor[0] + 1];
    if (work >= end - first) return;
    const uint body = body_ids[first + work];
#else
    const uint body = work;
    if (body >= p.BodyCount) return;
#endif
    SolveBody(poses, iterates, initial, inertial, masses, displacements, contacts, colors, cursor[0], joints, incoming, incoming_slots, joint_incidence, quiet, p, body, lane);
}

// Publish after every color finishes; keep the iteration snapshot unchanged within the sweep.
static void PublishBody(
    device Pose *poses, device const BodyIterate *iterates,
    device Displacement *displacements,
    device const BodyMass *masses, device const uint *quiet,
    constant StepParams &p, uint body
) {
    if (body >= p.BodyCount || !Solved(masses[body], quiet[body], p)) return;
    poses[body] = iterates[body].Current;
    displacements[body] = iterates[body].Moved;
}

kernel void PublishPoses(
    device Pose *poses [[buffer(0)]], device const BodyIterate *iterates [[buffer(11)]],
    device Displacement *displacements [[buffer(10)]],
    device const BodyMass *masses [[buffer(4)]], device const uint *quiet [[buffer(21)]],
    constant StepParams &p [[buffer(7)]], uint body [[thread_position_in_grid]]
) {
    PublishBody(poses, iterates, displacements, masses, quiet, p, body);
}

static bool UpdateContactDualAtSlot(
    device Contact *contacts, device const Displacement *displacements,
    device const Pose *initial, device const BodyMass *masses,
    device const uint *quiet, constant StepParams &p, uint slot
) {
    device Contact &contact = contacts[slot];

    const Index body = contact.BodyA, other_body = contact.BodyB;
    // Rows with no solved endpoint retain their dual; unchanged displacement carries no new violation.
    if (!Solved(masses[body], quiet[body], p) && !Solved(masses[other_body], quiet[other_body], p)) return false;
    const uint3 old_lambda = as_type<uint3>(contact.Lambda), old_penalty = as_type<uint3>(contact.Penalty);
    const uint old_stick = contact.Stick;
    const Pose start = initial[body], other_start = initial[other_body];
    const Displacement own = displacements[body], other = displacements[other_body];
    const float3 arm = Rotate(start.Orientation, contact.AnchorA);
    const float3 other_arm = Rotate(other_start.Orientation, contact.AnchorB);
    const ContactBasis basis = MakeContactBasis(contact.Normal);
    const float3 c = ContactConstraint(contact, p, basis, arm, other_arm, own, other);

    // Measure penalty saturation before clamping force to the friction cone.
    const float3 requested = contact.Penalty * c + contact.Lambda;
    const float3 force = ContactForce(requested, contact.Friction);
    contact.Lambda = force;
    // Scale the penalty ramp with supported force.
    // Closing travel and gravity displacement bound impact scaling; ContactMargin bounds the denominator.
    const float travel = max(p.ContactMargin, (abs(contact.Approach) + abs(dot(p.Gravity, contact.Normal)) * p.DeltaTime) * p.DeltaTime);
    const float beta = p.ContactBeta * contact.StiffnessScale;
    const float normal_beta = p.ContactBeta * max(contact.StiffnessScale, abs(force.x) / max(travel, 1e-9f));
    if (force[0] < 0) contact.Penalty[0] = min(contact.Penalty[0] + normal_beta * abs(c[0]), p.PenaltyMax);
    const float bound = abs(force[0]) * contact.Friction;
    if (length(float2(requested[1], requested[2])) <= bound) {
        contact.Penalty[1] = min(contact.Penalty[1] + beta * abs(c[1]), p.PenaltyMax);
        contact.Penalty[2] = min(contact.Penalty[2] + beta * abs(c[2]), p.PenaltyMax);
        contact.Stick = length(float2(c[1], c[2])) < p.ContactMargin;
    }
    return any(old_lambda != as_type<uint3>(contact.Lambda)) || any(old_penalty != as_type<uint3>(contact.Penalty)) || old_stick != contact.Stick;
}

static bool UpdateContactDual(
    device Contact *contacts, device const Displacement *displacements,
    device const Pose *initial, device const BodyMass *masses,
    device const uint *quiet, device const Adjacency *incoming,
    device const uint *incoming_slots, constant StepParams &p,
    uint id
) {
    const Adjacency last = incoming[p.BodyCount - 1];
    if (id >= last.Start + last.Count) return false;
    const uint slot = incoming_slots[id];
    return UpdateContactDualAtSlot(contacts, displacements, initial, masses, quiet, p, slot);
}

kernel void UpdateDuals(
    device Contact *contacts [[buffer(5)]], device const Displacement *displacements [[buffer(10)]],
    device const Pose *initial [[buffer(1)]], device const BodyMass *masses [[buffer(4)]],
    device const uint *quiet [[buffer(21)]], device const Adjacency *incoming [[buffer(17)]],
    device const uint *incoming_slots [[buffer(18)]], constant StepParams &p [[buffer(7)]],
#if MIXED_ISLAND_SOLVE
    device const uint *islands [[buffer(25)]],
#endif
    uint id [[thread_position_in_grid]]
) {
#if MIXED_ISLAND_SOLVE
    if (p.BodyCount > SolveLanes && islands[0]) {
        const Adjacency last = incoming[p.BodyCount - 1];
        if (id >= last.Start + last.Count) return;
        device const Contact &contact = contacts[incoming_slots[id]];
        const uint owner = Solved(masses[contact.BodyA], quiet[contact.BodyA], p) ? contact.BodyA : contact.BodyB;
        if (!LargeIsland(islands, p, owner)) return;
    }
#endif
    UpdateContactDual(contacts, displacements, initial, masses, quiet, incoming, incoming_slots, p, id);
}

kernel void Finalize(
    device const Displacement *displacements [[buffer(10)]],
    device Velocity *velocities [[buffer(3)]], device const BodyMass *masses [[buffer(4)]],
    device const uint *quiet [[buffer(21)]], constant StepParams &p [[buffer(7)]],
    uint body [[thread_position_in_grid]]
) {
    if (body >= p.BodyCount || !Solved(masses[body], quiet[body], p)) return;
    const Displacement moved = displacements[body];
    const float inv_dt = 1 / p.DeltaTime;
    velocities[body] = {moved.Linear * inv_dt, moved.Angular * inv_dt};
}

// Restitution requires a compressive normal force and initial approach speed above MinBounceSpeed.
kernel void Restitution(
    device Contact *contacts [[buffer(5)]], device const Pose *poses [[buffer(0)]],
    device const Velocity *velocities [[buffer(3)]], device const BodyMass *masses [[buffer(4)]],
    device const Adjacency *incoming [[buffer(17)]], device const uint *incoming_slots [[buffer(18)]],
    constant StepParams &p [[buffer(7)]], uint id [[thread_position_in_grid]]
) {
    const Adjacency last = incoming[p.BodyCount - 1];
    if (id >= last.Start + last.Count) return;
    const uint slot = incoming_slots[id];
    device Contact &contact = contacts[slot];
    contact.BounceDelta = 0;
    if (contact.Lambda[0] >= 0 || contact.Approach <= p.MinBounceSpeed) return;
    const Index a = contact.BodyA, b = contact.BodyB;
    const float restitution = contact.Restitution;
    if (restitution <= 0) return;

    const BodyMass mass_a = masses[a], mass_b = masses[b];
    const float3 arm_a = Rotate(poses[a].Orientation, contact.AnchorA);
    const float3 arm_b = Rotate(poses[b].Orientation, contact.AnchorB);
    const float3 normal = contact.Normal;

    const float3 turn_a = WorldInverseInertia(poses[a].Orientation, mass_a.InvInertiaLocal) * cross(arm_a, normal);
    const float3 turn_b = WorldInverseInertia(poses[b].Orientation, mass_b.InvInertiaLocal) * cross(arm_b, normal);
    const float weight = mass_a.InvMass + mass_b.InvMass +
        dot(normal, cross(turn_a, arm_a)) + dot(normal, cross(turn_b, arm_b));
    if (weight <= 0) return;

    const Velocity va = velocities[a], vb = velocities[b];
    const float3 closing = (va.Linear + cross(va.Angular, arm_a)) - (vb.Linear + cross(vb.Angular, arm_b));
    // Accumulate impulses toward the requested separating speed and clamp the total to remain repulsive.
    const float total = max(0.f, contact.BounceImpulse + (restitution * contact.Approach - dot(normal, closing)) / weight);
    contact.BounceDelta = total - contact.BounceImpulse;
    contact.BounceImpulse = total;
}

kernel void ApplyRestitution(
    device Velocity *velocities [[buffer(3)]], device const Contact *contacts [[buffer(5)]],
    device const Pose *poses [[buffer(0)]], device const BodyMass *masses [[buffer(4)]],
    device const Adjacency *incoming [[buffer(17)]], device const uint *incoming_slots [[buffer(18)]],
    device const uint *quiet [[buffer(21)]], constant StepParams &p [[buffer(7)]],
    uint body [[thread_position_in_grid]]
) {
    if (body >= p.BodyCount) return;
    const BodyMass mass = masses[body];
    if (!Solved(mass, quiet[body], p)) return;

    const float4 orientation = poses[body].Orientation;
    const float3x3 inverse_inertia = WorldInverseInertia(orientation, mass.InvInertiaLocal);
    float3 linear = float3(0), angular = float3(0);
    const Adjacency neighbours = incoming[body];
    for (uint i = 0; i < ContactsPerBody + neighbours.Count; ++i) {
        const uint slot = ContactSlot(i, body * ContactsPerBody, neighbours.Start, incoming_slots, contacts);
        if (slot == NoIndex) continue;
        const Contact contact = contacts[slot];
        if (contact.BounceDelta == 0) continue;
        const bool mine_is_a = contact.BodyA == body;
        const float3 arm = Rotate(orientation, mine_is_a ? contact.AnchorA : contact.AnchorB);

        const float3 impulse = contact.Normal * (mine_is_a ? contact.BounceDelta : -contact.BounceDelta);
        linear += impulse * mass.InvMass;
        angular += inverse_inertia * cross(arm, impulse);
    }
    velocities[body].Linear += linear;
    velocities[body].Angular += angular;
}

// Publish all quiet counts before FinishWaking reads neighboring counts.
kernel void FinishPoses(
    device Pose *poses [[buffer(0)]], device const Velocity *velocities [[buffer(3)]],
    device Displacement *displacements [[buffer(10)]], device const BodyIterate *iterates [[buffer(11)]],
    device const BodyMass *masses [[buffer(4)]], device const uint *quiet [[buffer(21)]], device uint *next [[buffer(23)]],
    device Pose *rest [[buffer(22)]], constant StepParams &p [[buffer(7)]],
    uint body [[thread_position_in_grid]]
) {
    if (body >= p.BodyCount) return;
    PublishBody(poses, iterates, displacements, masses, quiet, p, body);
    next[body] = quiet[body];
    if (!Solved(masses[body], quiet[body], p)) return;
    const Velocity now = velocities[body];
    const bool slow = length(now.Linear) <= p.SleepSpeed && length(now.Angular) <= p.SleepSpeed;
    uint counted = slow ? quiet[body] + 1 : 0;
    if (counted == p.SleepSteps) {
        const Displacement since = Since(poses[body], rest[body]);
        if (length(since.Linear) > p.SleepDrift || length(since.Angular) > p.SleepDrift) counted = 0;
    }
    next[body] = counted;
    if (counted == 0) rest[body] = poses[body];
}

// Propagate the minimum quiet count across one contact per step to synchronize sleeping.
static void FinishWakingBody(
    device const uint *counted, device uint *quiet,
    device const Contact *contacts, device const Joint *joints,
    device const Adjacency *incoming, device const uint *incoming_slots,
    device const BodyMass *masses, device const Velocity *velocities,
    device const uint *joint_incidence,
    constant StepParams &p, uint body
) {
    if (body >= p.BodyCount) return;
    uint least = counted[body];
    quiet[body] = least;
    if (!Moves(masses[body]) || least == 0) return;

    const Adjacency neighbours = incoming[body];
    for (uint i = 0; i < ContactsPerBody + neighbours.Count; ++i) {
        const uint slot = ContactSlot(i, body * ContactsPerBody, neighbours.Start, incoming_slots, contacts);
        if (slot == NoIndex) continue;
        const Index other = ContactPartner(contacts[slot], body);
        if (Moves(masses[other])) least = min(least, counted[other]);
        // Moving unsolved bodies wake sleeping neighbors.
        else if (Driven(masses[other], velocities[other], p)) least = 0;
    }
    for (uint at = joint_incidence[body]; at < joint_incidence[body + 1]; ++at) {
        const uint index = joint_incidence[at];
        const Index other = JointPartner(joints[index], body, masses);
        if (other != NoIndex) least = min(least, counted[other]);
    }
    quiet[body] = least;
}

kernel void FinishWaking(
    device const uint *counted [[buffer(23)]], device uint *quiet [[buffer(21)]],
    device const Contact *contacts [[buffer(5)]], device const Joint *joints [[buffer(16)]],
    device const Adjacency *incoming [[buffer(17)]], device const uint *incoming_slots [[buffer(18)]],
    device const BodyMass *masses [[buffer(4)]], device const Velocity *velocities [[buffer(3)]],
    device const uint *joint_incidence [[buffer(20)]],
    constant StepParams &p [[buffer(7)]], uint body [[thread_position_in_grid]]
) {
    FinishWakingBody(counted, quiet, contacts, joints, incoming, incoming_slots, masses, velocities, joint_incidence, p, body);
}

kernel void FollowSensors(
    device Pose *poses [[buffer(0)]], device Velocity *velocities [[buffer(3)]],
    device const SensorFollower *followers [[buffer(26)]], uint index [[thread_position_in_grid]]
) {
    const SensorFollower follower = followers[index];
    const Pose owner = poses[follower.Owner];
    const Pose pose = ComposePose(owner, follower.Local);
    Velocity velocity = velocities[follower.Owner];
    velocity.Linear += cross(velocity.Angular, pose.Position - owner.Position);
    poses[follower.Sensor] = pose;
    velocities[follower.Sensor] = velocity;
}

kernel void ReduceStepColors(
    constant StepParams &p [[buffer(7)]], device const uint *colors [[buffer(12)]],
    device const BodyMass *masses [[buffer(4)]], device atomic_uint *used [[buffer(26)]],
    uint body [[thread_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]
) {
    threadgroup uint partial[4];
    const uint local = body < p.BodyCount && Moves(masses[body]) ? ColorOf(colors[body]) + 1 : 1;
    const uint maximum = simd_max(local);
    if (lane == 0) partial[simd] = maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
        const uint group_maximum = simd_max(lane < 4 ? partial[lane] : 1);
        if (lane == 0) atomic_fetch_max_explicit(used, group_maximum, memory_order_relaxed);
    }
}

static void PublishStepColors(device StepParams &p, device uint *groups, uint count, uint lane) {
    // All lanes read the old cap before lane zero replaces it with the current count.
    simdgroup_barrier(mem_flags::mem_device);
    if (lane == 0) p.MaxColors = count;
    const uint packed = FallbackBodiesPerGroup(p.BodyCount);
    groups[lane * 3] = lane < count ? (p.BodyCount + packed - 1) / packed : 0;
    groups[lane * 3 + 1] = groups[lane * 3 + 2] = 1;
}

kernel void FinishStepColors(
    device StepParams &p [[buffer(7)]], device uint *groups [[buffer(26)]],
    uint lane [[thread_index_in_simdgroup]]
) {
    PublishStepColors(p, groups, clamp(groups[0] + 1, 1u, p.MaxColors), lane);
}

kernel void PrepareStepColors(
    device StepParams &p [[buffer(7)]], device const uint *colors [[buffer(12)]],
    device const BodyMass *masses [[buffer(4)]], device uint *groups [[buffer(26)]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint used = 1;
    for (uint body = lane; body < p.BodyCount; body += 32)
        if (Moves(masses[body])) used = max(used, ColorOf(colors[body]) + 1);
    PublishStepColors(p, groups, clamp(simd_max(used) + 1, 1u, p.MaxColors), lane);
}

kernel void CaptureStep(
    device const Pose *poses [[buffer(0)]], device const Pose *initial [[buffer(1)]],
    device const Velocity *velocities [[buffer(3)]], device const Contact *contacts [[buffer(5)]],
    device const Contact *sensors [[buffer(6)]], constant StepParams &p [[buffer(7)]],
    device ContactReport *out_contacts [[buffer(8)]], device const ContactEvent *events [[buffer(9)]],
    device const uint *event_counts [[buffer(10)]], device Pose *out_initial [[buffer(11)]],
    device const BroadPhaseNode *nodes [[buffer(13)]], device const uint *sensor_refusals [[buffer(14)]],
    device Pose *out_poses [[buffer(15)]], device Velocity *out_velocities [[buffer(16)]],
    device SensorPair *out_sensors [[buffer(17)]], device StepCounts *out_counts [[buffer(18)]],
    device StepCompletion *completion [[buffer(19)]], constant StepOutputFlags &output [[buffer(20)]],
    device ContactEvent *out_removed [[buffer(21)]],
    device const uint *refusals [[buffer(26)]],
#if FINISH_WAKING_CAPTURE
    device const BodyMass *masses [[buffer(4)]], device uint *quiet [[buffer(22)]],
    device const uint *counted [[buffer(23)]], device const Joint *joints [[buffer(27)]],
    device const Adjacency *incoming [[buffer(28)]], device const uint *incoming_slots [[buffer(29)]],
    device const uint *joint_incidence [[buffer(30)]],
#endif
    uint body [[thread_position_in_grid]]
) {
    if (body >= p.BodyCount) return;
#if FINISH_WAKING_CAPTURE
    FinishWakingBody(counted, quiet, contacts, joints, incoming, incoming_slots, masses, velocities, joint_incidence, p, body);
#endif
    if (body == 0) {
        const BroadPhaseNode root = nodes[BroadPhaseRoot(p.BodyCount)];
        completion[0] = {p.MaxColors, root.Ready, root.Errors};
    }
    StepCounts count{0, 0, 0, refusals[body], output.Sensors ? sensor_refusals[body] : 0};
    if (output.Poses) {
        out_poses[body] = poses[body];
        out_velocities[body] = velocities[body];
    }
    if (p.ReportContacts) {
        out_initial[body] = initial[body];
        for (uint i = 0; i < event_counts[body]; ++i) {
            const ContactEvent event = events[body * EventsPerBody + i];
            if (event.Kind == ContactRemoved) out_removed[body * ContactsPerBody + count.RemovedContacts++] = event;
            else {
                const uint slot = body * ContactsPerBody + count.Contacts++;
                out_contacts[slot] = ReportContact(event, contacts[slot]);
            }
        }
    }
    if (output.Sensors) {
        for (uint i = 0; i < ContactsPerBody; ++i) {
            const Contact contact = sensors[body * ContactsPerBody + i];
            if (!contact.Active) break;
            out_sensors[body * ContactsPerBody + count.Sensors++] = {contact.BodyA, contact.BodyB, contact.Children};
        }
    }
    out_counts[body] = count;
}

kernel void CountColorWork(
    device const BodyMass *masses [[buffer(4)]], device const uint *colors [[buffer(12)]],
    device const uint *quiet [[buffer(21)]], constant StepParams &p [[buffer(7)]],
    device ColorWork &work [[buffer(26)]], uint body [[thread_position_in_grid]]
) {
    if (body < p.BodyCount && Solved(masses[body], quiet[body], p))
        atomic_fetch_add_explicit((device atomic_uint *)&work.Counts[ColorOf(colors[body]) % p.MaxColors], 1u, memory_order_relaxed);
}

kernel void PrefixColorWork(
    device ColorWork &work [[buffer(26)]], device uint *groups [[buffer(14)]],
    constant StepParams &p [[buffer(7)]], uint lane [[thread_index_in_simdgroup]]
) {
    const uint count = work.Counts[lane], first = simd_prefix_exclusive_sum(count);
    work.Offsets[lane] = work.Cursors[lane] = first;
    if (lane == MaxSupportedColors - 1) work.Offsets[MaxSupportedColors] = first + count;
    groups[lane * 3] = (count + FallbackBodiesPerGroup(p.BodyCount) - 1) / FallbackBodiesPerGroup(p.BodyCount);
    groups[lane * 3 + 1] = groups[lane * 3 + 2] = 1;
}

kernel void FillColorWork(
    device const BodyMass *masses [[buffer(4)]], device const uint *colors [[buffer(12)]],
    device const uint *quiet [[buffer(21)]], constant StepParams &p [[buffer(7)]],
    device ColorWork &work [[buffer(26)]], device uint *body_ids [[buffer(13)]],
    uint body [[thread_position_in_grid]]
) {
    if (body < p.BodyCount && Solved(masses[body], quiet[body], p)) {
        const uint at = atomic_fetch_add_explicit((device atomic_uint *)&work.Cursors[ColorOf(colors[body]) % p.MaxColors], 1u, memory_order_relaxed);
        body_ids[at] = body;
    }
}

// Only solved endpoints connect components.
static void FindSmallIslandsBody(
    device const BodyMass *masses, device const Contact *contacts,
    constant StepParams &p, device const Joint *joints,
    device const Adjacency *incoming, device const uint *incoming_slots,
    device const uint *joint_incidence, device const uint *quiet,
    device uint *members, uint lane
) {
    const bool solved = lane < p.BodyCount && Solved(masses[lane], quiet[lane], p);
    const uint movable = uint(ulong(simd_ballot(solved)));
    uint reached = 1u << lane;
    if (solved) {
        const Adjacency neighbours = incoming[lane];
        for (uint i = 0; i < ContactsPerBody + neighbours.Count; ++i) {
            const uint slot = ContactSlot(i, lane * ContactsPerBody, neighbours.Start, incoming_slots, contacts);
            if (slot != NoIndex) reached |= (1u << ContactPartner(contacts[slot], lane)) & movable;
        }
        for (uint at = joint_incidence[lane]; at < joint_incidence[lane + 1]; ++at) {
            device const Joint &joint = joints[joint_incidence[at]];
            if (joint.Active) reached |= (1u << (joint.BodyA == lane ? joint.BodyB : joint.BodyA)) & movable;
        }
    }
    // Each round doubles the reachable path length; five cover all 32 bodies.
    for (uint round = 0; round < 5; ++round) {
        uint next = reached;
        for (uint other = 0; other < 32; ++other) {
            const uint linked = simd_shuffle(reached, other);
            if (reached & (1u << other)) next |= linked;
        }
        reached = next;
    }
    members[lane] = reached;
}

kernel void FindSmallIslands(
    device const BodyMass *masses [[buffer(4)]], device const Contact *contacts [[buffer(5)]],
    constant StepParams &p [[buffer(7)]], device const Joint *joints [[buffer(16)]],
    device const Adjacency *incoming [[buffer(17)]], device const uint *incoming_slots [[buffer(18)]],
    device const uint *joint_incidence [[buffer(20)]], device const uint *quiet [[buffer(21)]],
    device uint *members [[buffer(26)]], uint lane [[thread_index_in_simdgroup]]
) {
    FindSmallIslandsBody(masses, contacts, p, joints, incoming, incoming_slots, joint_incidence, quiet, members, lane);
}

static void PackBodyColors(uint body, uint color, uint count, threadgroup uint *body_ids, threadgroup uint *offsets, uint lane) {
    uint first = 0;
    for (uint at_color = 0; at_color < count; ++at_color) {
        const uint kept = uint(color == at_color), at = simd_prefix_exclusive_sum(kept);
        const uint size = simd_sum(kept);
        if (kept) body_ids[first + at] = body;
        if (lane == 0) offsets[at_color] = first;
        first += size;
    }
    if (lane == 0) offsets[count] = first;
}

#if FUSED_SMALL_SOLVE

kernel void SolveSmallWorld(
    device Pose *poses [[buffer(0)]], device BodyIterate *iterates [[buffer(11)]], device const Pose *initial [[buffer(1)]],
    device const Pose *inertial [[buffer(2)]], device const BodyMass *masses [[buffer(4)]],
    device Displacement *displacements [[buffer(10)]], device Contact *contacts [[buffer(5)]],
    device const uint *colors [[buffer(12)]], device const uint *iterations [[buffer(14)]],
    device Joint *joints [[buffer(16)]], device const Adjacency *incoming [[buffer(17)]],
    device const uint *incoming_slots [[buffer(18)]], device const uint *joint_incidence [[buffer(20)]],
    device const uint *quiet [[buffer(21)]], constant StepParams &p [[buffer(7)]],
    device const uint *islands [[buffer(26)]], uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint wave [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
#if FUSED_GENERAL_SOLVE
    if (group >= islands[0]) return;
    const uint root = islands[IslandHeaderWords + (2 + IslandBodyLimit) * p.BodyCount + group];
    const uint count = islands[IslandHeaderWords + p.BodyCount + root];
    device const uint *members = islands + IslandHeaderWords + 2 * p.BodyCount + root * IslandBodyLimit;
    const uint waves = SmallSolveWaves;
#else
    const uint members = islands[group];
    if (ctz(members) != group) return;
    const uint waves = min(p.BodyCount, SmallSolveWaves);
#endif
    const uint threads = waves * SolveLanes;
    threadgroup uint body_ids[MaxSupportedColors], offsets[MaxSupportedColors + 1];
    if (wave == 0) {
#if FUSED_GENERAL_SOLVE
        const uint body = lane < count ? members[lane] : NoIndex;
        const bool solved = body != NoIndex && Solved(masses[body], quiet[body], p);
#else
        const uint body = lane;
        const bool solved = lane < p.BodyCount && (members & (1u << lane)) && Solved(masses[lane], quiet[lane], p);
#endif
        PackBodyColors(body, solved ? ColorOf(colors[body]) % p.MaxColors : NoIndex, p.MaxColors, body_ids, offsets, lane);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float2 updated[SmallSolveWaves * 12];
    threadgroup uint changed_waves[SmallSolveWaves];
    for (uint iteration = 0; iteration < iterations[0]; ++iteration) {
        bool changed = false;
        for (uint color = 0; color < p.MaxColors; ++color) {
            const uint first = offsets[color];
            for (uint at = wave; at < offsets[color + 1] - first; at += waves) {
                const uint body = body_ids[first + at];
                changed |= SolveBody(poses, iterates, initial, inertial, masses, displacements, contacts, colors, color, joints, incoming, incoming_slots, joint_incidence, quiet, p, body, lane);
            }
            threadgroup_barrier(mem_flags::mem_device);
        }
#if FUSED_GENERAL_SOLVE
        if (tid < count) PublishBody(poses, iterates, displacements, masses, quiet, p, members[tid]);
#else
        if (tid < p.BodyCount && (members & (1u << tid))) PublishBody(poses, iterates, displacements, masses, quiet, p, tid);
#endif
        threadgroup_barrier(mem_flags::mem_device);
#if FUSED_GENERAL_SOLVE
        for (uint at = wave; at < count; at += waves) {
            const uint body = members[at];
            if (Solved(masses[body], quiet[body], p)) {
                const Adjacency neighbours = incoming[body];
                for (uint i = lane; i < ContactsPerBody + neighbours.Count; i += SolveLanes) {
                    // ContactSlot skips a serial iterator to the end of the owned run.
                    // SIMD lanes must keep their stride to cover every incoming contact once.
                    const uint slot = i < ContactsPerBody ? body * ContactsPerBody + i : incoming_slots[neighbours.Start + i - ContactsPerBody];
                    if (!contacts[slot].Active) continue;
                    device const Contact &contact = contacts[slot];
                    const uint owner = Solved(masses[contact.BodyA], quiet[contact.BodyA], p) ? contact.BodyA : contact.BodyB;
                    if (owner == body)
                        changed |= UpdateContactDualAtSlot(contacts, displacements, initial, masses, quiet, p, slot);
                }
            }
            for (uint incidence = joint_incidence[body]; incidence < joint_incidence[body + 1]; ++incidence) {
                const uint index = joint_incidence[incidence];
                device const Joint &joint = joints[index];
                if (!joint.Active) continue;
                const uint owner = JointDualOwner(joint, masses, quiet, p);
                if (owner == body)
                    changed |= UpdateJointDual(joints, poses, initial, p, updated + wave * 12, index, lane);
            }
#else
        const Adjacency last = incoming[p.BodyCount - 1];
        for (uint id = tid; id < last.Start + last.Count; id += threads) {
            device const Contact &contact = contacts[incoming_slots[id]];
            const uint owner = Solved(masses[contact.BodyA], quiet[contact.BodyA], p) ? contact.BodyA : contact.BodyB;
            if (ctz(islands[owner]) == group)
                changed |= UpdateContactDual(contacts, displacements, initial, masses, quiet, incoming, incoming_slots, p, id);
        }
        for (uint index = wave; index < p.JointCount; index += waves) {
            device const Joint &joint = joints[index];
            if (!joint.Active) continue;
            const uint owner = JointDualOwner(joint, masses, quiet, p);
            if (ctz(islands[owner]) == group)
                changed |= UpdateJointDual(joints, poses, initial, p, updated + wave * 12, index, lane);
#endif
        }
        const bool wave_changed = simd_any(changed);
        if (lane == 0) changed_waves[wave] = wave_changed;
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        bool any_changed = false;
        for (uint at = 0; at < waves; ++at) any_changed |= changed_waves[at] != 0;
        if (!any_changed) break;
    }
}
#endif

static uint IslandRoot(device atomic_uint *roots, uint body) {
    uint parent = atomic_load_explicit(roots + body, memory_order_relaxed);
    while (parent != body) {
        const uint next = atomic_load_explicit(roots + parent, memory_order_relaxed);
        atomic_store_explicit(roots + body, next, memory_order_relaxed);
        body = parent;
        parent = next;
    }
    return body;
}
static void JoinIsland(device atomic_uint *roots, uint a, uint b) {
    for (;;) {
        a = IslandRoot(roots, a);
        b = IslandRoot(roots, b);
        if (a == b) return;
        const uint low = min(a, b), high = max(a, b);
        uint expected = high;
        if (atomic_compare_exchange_weak_explicit(roots + high, &expected, low, memory_order_relaxed, memory_order_relaxed)) return;
    }
}
kernel void InitializeIslands(device uint *pool [[buffer(26)]], constant StepParams &p [[buffer(7)]], uint id [[thread_position_in_grid]]) {
    if (id >= p.BodyCount) return;
    pool[IslandHeaderWords + id] = id;
    pool[IslandHeaderWords + p.BodyCount + id] = 0;
    if (id == 0) pool[0] = pool[IslandLargeAt] = 0;
}
kernel void UnionIslands(
    device uint *pool [[buffer(26)]], constant StepParams &p [[buffer(7)]],
    device const BodyMass *masses [[buffer(4)]], device const Contact *contacts [[buffer(5)]],
    device const Joint *joints [[buffer(16)]], device const uint *quiet [[buffer(21)]],
    uint id [[thread_position_in_grid]]
) {
    device atomic_uint *roots = (device atomic_uint *)(pool + IslandHeaderWords);
    if (id < p.BodyCount) {
        Index previous = NoIndex;
        for (uint i = 0; i < ContactsPerBody; ++i) {
            device const Contact &contact = contacts[id * ContactsPerBody + i];
            if (!contact.Active) break;
            const Index other = contact.BodyB;
            if (other != previous && Solved(masses[contact.BodyA], quiet[contact.BodyA], p) && Solved(masses[other], quiet[other], p))
                JoinIsland(roots, contact.BodyA, other);
            previous = other;
        }
    }
    if (id < p.JointCount) {
        device const Joint &joint = joints[id];
        if (joint.Active && Solved(masses[joint.BodyA], quiet[joint.BodyA], p) && Solved(masses[joint.BodyB], quiet[joint.BodyB], p))
            JoinIsland(roots, joint.BodyA, joint.BodyB);
    }
}
kernel void PackIslands(device uint *pool [[buffer(26)]], constant StepParams &p [[buffer(7)]], uint body [[thread_position_in_grid]]) {
    if (body >= p.BodyCount) return;
    device atomic_uint *roots = (device atomic_uint *)(pool + IslandHeaderWords);
    const uint root = IslandRoot(roots, body);
    const uint at = atomic_fetch_add_explicit((device atomic_uint *)(pool + IslandHeaderWords + p.BodyCount + root), 1u, memory_order_relaxed);
    if (at < IslandBodyLimit) pool[IslandHeaderWords + 2 * p.BodyCount + root * IslandBodyLimit + at] = body;
    else atomic_store_explicit((device atomic_uint *)(pool + IslandLargeAt), 1u, memory_order_relaxed);
}
kernel void FinishIslands(
    device uint *pool [[buffer(26)]], device const uint *groups [[buffer(14)]], constant StepParams &p [[buffer(7)]],
    device const BodyMass *masses [[buffer(4)]], device const uint *quiet [[buffer(21)]],
    device const uint *body_ids [[buffer(13)]], device const ColorWork &color_work [[buffer(25)]],
    device const Adjacency *incoming [[buffer(17)]],
    device const Joint *joints [[buffer(16)]], device const uint *incidence [[buffer(20)]], uint id [[thread_position_in_grid]]
) {
    bool keep = false;
    if (id < p.BodyCount && pool[IslandHeaderWords + id] == id && pool[IslandHeaderWords + p.BodyCount + id] <= IslandBodyLimit) {
        keep = Solved(masses[id], quiet[id], p);
        // Unsolved bodies update only joints whose other endpoint is also unsolved.
        if (!keep) {
            for (uint at = incidence[id]; at < incidence[id + 1]; ++at) {
                device const Joint &joint = joints[incidence[at]];
                keep |= joint.Active && joint.BodyA == id && !Solved(masses[joint.BodyB], quiet[joint.BodyB], p);
            }
        }
    }
    if (keep) {
        const uint at = atomic_fetch_add_explicit((device atomic_uint *)pool, 1u, memory_order_relaxed);
        pool[IslandHeaderWords + (2 + IslandBodyLimit) * p.BodyCount + at] = id;
    }
    const uint large = pool[IslandLargeAt];
    if (large && id < color_work.Offsets[MaxSupportedColors]) {
        const uint body = body_ids[id];
        pool[IslandHeaderWords + (3 + IslandBodyLimit) * p.BodyCount + id] = LargeIsland(pool, p, body) ? body : NoIndex;
    }
    if (id == 0) {
        pool[1] = pool[2] = 1;
        pool[3] = large ? (p.BodyCount + 127) / 128 : 0;
        pool[4] = pool[5] = 1;
        const Adjacency last = incoming[p.BodyCount - 1];
        pool[6] = large ? (last.Start + last.Count + 63) / 64 : 0;
        pool[7] = pool[8] = 1;
        pool[9] = large ? p.JointCount : 0;
        pool[10] = pool[11] = 1;
    }
    if (id < MaxSupportedColors) {
        pool[IslandColorsAt + 3 * id] = large ? groups[3 * id] : 0;
        pool[IslandColorsAt + 3 * id + 1] = pool[IslandColorsAt + 3 * id + 2] = 1;
    }
}

// Device barriers synchronize the dependency graph within one SIMD group.
kernel void PrepareSmallWorld(
    device Pose *poses [[buffer(0)]], device const Pose *initial [[buffer(1)]],
    device const Velocity *velocities [[buffer(3)]], device const BodyMass *masses [[buffer(4)]],
    device const Contact *contacts [[buffer(5)]], constant StepParams &p [[buffer(7)]],
    device Velocity *previous [[buffer(9)]], device Displacement *displacements [[buffer(10)]],
    device BodyIterate *iterates [[buffer(11)]], device uint *colors [[buffer(12)]], device uint *next [[buffer(13)]],
    device const uint *budgets [[buffer(14)]], device Joint *joints [[buffer(16)]],
    device Adjacency *incoming [[buffer(17)]], device uint *slots [[buffer(18)]],
    device const uint *joint_incidence [[buffer(20)]], device const uint *quiet [[buffer(21)]],
    device uint *islands [[buffer(26)]], uint body [[thread_index_in_simdgroup]]
) {
    const uint count = body < p.BodyCount ? incoming[body].Count : 0;
    const uint start = simd_prefix_exclusive_sum(count);
    if (body < p.BodyCount) {
        incoming[body].Start = start;
        incoming[body].Cursor = start;
    }
    threadgroup_barrier(mem_flags::mem_device);
    FillIncomingBody(incoming, slots, contacts, p, body);
    threadgroup_barrier(mem_flags::mem_device);
    SortIncomingBody(incoming, slots, masses, p, body);
    threadgroup_barrier(mem_flags::mem_device);
    for (uint joint = body; joint < p.JointCount; joint += SolveLanes)
        PrepareJointsBody(joints, poses, p, joint);
    threadgroup_barrier(mem_flags::mem_device);
    WarmStartBody(poses, initial, velocities, previous, masses, displacements, iterates, p, body);
    threadgroup_barrier(mem_flags::mem_device);
    for (uint pass = 0; pass < budgets[1]; ++pass) {
        UpdateColorsBody(colors, next, contacts, masses, joints, incoming, slots, quiet, joint_incidence, p, body);
        threadgroup_barrier(mem_flags::mem_device);
        // Degree participates in priority, so the entire color word must be fixed.
        const bool changed = simd_any(body < p.BodyCount && next[body] != colors[body]);
        PublishColorsBody(colors, next, p, body);
        threadgroup_barrier(mem_flags::mem_device);
        if (!changed) break;
    }
    if (budgets[0]) FindSmallIslandsBody(masses, contacts, p, joints, incoming, slots, joint_incidence, quiet, islands, body);
}
