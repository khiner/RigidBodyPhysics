// AVBD for rigid bodies, following Augmented Vertex Block Descent (Giles et al., SIGGRAPH 2025).
// Ported from the reference implementations rather than re-derived: ../avbd-demo2d/source/solver.cpp
// for the algorithm and its equation numbering, ../MetalAVBD/MetalAVBD/AVBDCompute.metal for the
// three-dimensional contact basis and friction cone. See NOTICE.md for their notices.
//
// One step is: integrate to an inertial target, build the contacts against the pose at the start of
// the step, then alternate a primal sweep (each body solves its own 6x6 block) with a dual update
// (each contact's force and penalty adapt). Velocity is read off the motion after the last of those,
// and a final stabilization sweep removes leftover penetration without touching velocity - which is
// how error correction happens without adding energy to the system.
//
// Sign conventions, which are the references': Normal points out of B towards A, C is the separation
// along it, so penetration is negative and a contact's normal force is never positive.

constant uint Dof = 6;

// A body that has been too slow to be worth solving for long enough stops being solved. It keeps its
// contacts, so whatever is resting on it stays held up - it simply behaves as static geometry does
// until something disturbs it.
static bool Asleep(uint quiet, constant StepParams &p) { return quiet >= p.SleepSteps; }

// Box2D-lite's bias towards keeping the reference face already chosen, which both references carry.
constant float RelativeTolerance = 0.95f;
constant float AbsoluteTolerance = 0.01f;
// The same bias between an edge pair and a face, at MetalAVBD's value. Absolute rather than scaled by
// a half extent, since an edge pair has no one box's extent to scale by.
constant float EdgeTolerance = 0.01f;

// Compiled a second time with Stabilize set, for the final pass that keeps the C0 term.
#ifndef STABILIZE
#define STABILIZE 0
#endif
// How much of the error a step began with a hard row is allowed to ignore. Eq. 18 makes it a parameter
// and spreads the correction over several steps. Post-stabilization instead runs the whole solve at 1
// and then one pass at 0, which is the cheaper of the two ways to add no energy. A soft row takes 0
// always, since a spring answers for the extension it has.
#if STABILIZE
constant float ConstraintAlpha = 0; // keep all of the accumulated error, and correct it
#else
constant float ConstraintAlpha = 1; // ignore it, and only resist error added during this step
#endif

// The inertial stiffness a pair of bodies brings to a row measuring a distance between them: their
// reduced mass over h squared, which is the M/h^2 the penalty shares a Hessian with and so the scale a
// penalty is meaningful against (Sec. 3.4). Two static bodies bring none and want no penalty.
static float PairStiffness(float inverse, float dt) { return inverse > 0 ? 1 / (inverse * dt * dt) : 0; }

// Axis `i` of whatever frame the caller is working in, which several rows here are built one at a time.
static float3 UnitAxis(uint i) { return float3(i == 0 ? 1.f : 0.f, i == 1 ? 1.f : 0.f, i == 2 ? 1.f : 0.f); }

static float3x3 Diagonal(float3 d) {
    return float3x3(float3(d.x, 0, 0), float3(0, d.y, 0), float3(0, 0, d.z));
}

static float3x3 QuatToMatrix(float4 q) {
    return float3x3(Rotate(q, float3(1, 0, 0)), Rotate(q, float3(0, 1, 0)), Rotate(q, float3(0, 0, 1)));
}

// Any orthonormal frame whose first axis is the normal. Which tangents come out does not matter, only
// that the same normal always produces the same pair, so warm-started friction stays meaningful.
static float3x3 ContactBasis(float3 normal) {
    float3 tangent = abs(normal.x) > abs(normal.z) ? float3(-normal.y, normal.x, 0) : float3(0, -normal.z, normal.y);
    const float len = length(tangent);
    tangent = len > 1e-8f ? tangent / len : float3(1, 0, 0);
    return float3x3(normal, tangent, cross(normal, tangent));
}

// Solve H x = -g for a symmetric positive definite H, by LDL^T without pivoting. H is definite by
// construction: the inertial term puts mass on the whole diagonal before any contact adds to it.
static void SolveBlock(thread float H[Dof][Dof], thread float g[Dof], thread float out[Dof]) {
    for (uint j = 0; j < Dof; ++j) {
        for (uint k = 0; k < j; ++k) H[j][j] -= H[j][k] * H[j][k] * H[k][k];
        for (uint i = j + 1; i < Dof; ++i) {
            for (uint k = 0; k < j; ++k) H[i][j] -= H[i][k] * H[j][k] * H[k][k];
            H[i][j] /= H[j][j];
        }
    }
    float y[Dof];
    for (uint i = 0; i < Dof; ++i) {
        y[i] = -g[i];
        for (uint k = 0; k < i; ++k) y[i] -= H[i][k] * y[k];
    }
    for (uint i = 0; i < Dof; ++i) y[i] /= H[i][i];
    for (uint i = Dof; i-- > 0;) {
        out[i] = y[i];
        for (uint k = i + 1; k < Dof; ++k) out[i] -= H[k][i] * out[k];
    }
}

// The displacement of a body since the step began, which is the variable the constraints are
// expressed in.
struct Displacement {
    float3 Linear, Angular;
};

static Displacement Since(Pose now, Pose start) {
    return {now.Position - start.Position, RotationVector(QuatMul(now.Orientation, QuatConjugate(start.Orientation)))};
}

// Where one step of the body's own motion carries it, with `share` of gravity added to that. The
// inertial target of Eq. 2 takes the whole of gravity and the warm start below takes the part of it the
// body's recent acceleration actually showed, and they are otherwise the same flight.
static Pose FreeFlight(Pose pose, Velocity v, float3 gravity, float share, float dt) {
    return {pose.Position + dt * v.Linear + (share * dt * dt) * gravity,
            normalize(QuatMul(QuatFromRotationVector(dt * v.Angular), pose.Orientation))};
}

// Eq. 2: where free flight would put the body, which is what the inertial term pulls back towards. The
// pose itself is left where it is, since collision and every C0 and Jacobian are taken at the pose the
// step began from and only then is the body moved to its starting guess. The other way round measures
// C0 one warm start ahead of the pose the Taylor series is expanded about, and the stabilization pass
// then counts the guess twice.
kernel void Integrate(
    device Pose *poses [[buffer(0)]], device Pose *initial [[buffer(1)]], device Pose *inertial [[buffer(2)]],
    device Velocity *velocities [[buffer(3)]],
    device const BodyMass *masses [[buffer(4)]], device Adjacency *incoming [[buffer(17)]],
    device uint *quiet [[buffer(21)]], constant StepParams &p [[buffer(7)]],
    uint body [[thread_position_in_grid]]
) {
    if (body >= p.BodyCount) return;
    incoming[body].Count = 0; // before CountIncoming adds to it, and every body is here anyway
    const Pose pose = poses[body];
    initial[body] = pose;
    if (masses[body].InvMass == 0) {
        inertial[body] = pose;
        return;
    }

    Velocity v = velocities[body];
    // Anything that moves a sleeping body wakes it, and the host setting a velocity is the main one -
    // a sleeping body's velocity is zero, so anything else in there came from outside.
    if (Asleep(quiet[body], p) && (length(v.Linear) > p.SleepSpeed || length(v.Angular) > p.SleepSpeed)) quiet[body] = 0;
    if (Asleep(quiet[body], p)) {
        inertial[body] = pose;
        velocities[body] = {float3(0), float3(0)};
        return;
    }

    const float spin = length(v.Angular);
    if (spin > p.MaxAngularSpeed) v.Angular *= p.MaxAngularSpeed / spin;
    velocities[body] = v;

    inertial[body] = FreeFlight(pose, v, p.Gravity, 1, p.DeltaTime);
}

// The starting guess the sweeps begin from, which is not the inertial target. Gravity enters it only in
// proportion to how much the body's recent acceleration actually matched gravity, so a body already
// resting on something is not guessed into the floor every step and then pushed back out. The adaptive
// warm start from the original VBD paper, run after collision for the reason Integrate gives.
kernel void WarmStart(
    device Pose *poses [[buffer(0)]], device const Pose *initial [[buffer(1)]],
    device const Velocity *velocities [[buffer(3)]], device Velocity *previous [[buffer(9)]],
    device const BodyMass *masses [[buffer(4)]], constant StepParams &p [[buffer(7)]],
    uint body [[thread_position_in_grid]]
) {
    if (body >= p.BodyCount || masses[body].InvMass == 0) return;
    const Pose pose = initial[body];
    const Velocity v = velocities[body];
    const float dt = p.DeltaTime;
    const float gravity = length(p.Gravity);
    const float3 fall = gravity > 1e-6f ? p.Gravity / gravity : float3(0);
    const float along = dot((v.Linear - previous[body].Linear) / dt, fall);
    const float weight = gravity > 1e-6f ? clamp(along / gravity, 0.f, 1.f) : 0.f;
    previous[body] = v;
    poses[body] = FreeFlight(pose, v, p.Gravity, weight, dt);
}

// An oriented box, with its axes already in world space.
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

// How far the box reaches from its centre along an axis.
static float Reach(BoxPose box, float3 axis) {
    return abs(dot(box.Axis[0], axis)) * box.Half[0] + abs(dot(box.Axis[1], axis)) * box.Half[1] +
        abs(dot(box.Axis[2], axis)) * box.Half[2];
}

// Positive when the two boxes overlap along the axis, negative by the gap between them when they do
// not. This is the separating axis test: one negative axis and they cannot be touching.
static float Overlap(BoxPose a, BoxPose b, float3 axis) {
    return Reach(a, axis) + Reach(b, axis) - abs(dot(b.Center - a.Center, axis));
}

// The corners of one face of a box, wound so consecutive pairs are edges.
static void FaceCorners(BoxPose box, uint axis, float side, thread float3 *out) {
    const uint u = (axis + 1) % 3, v = (axis + 2) % 3;
    const float3 centre = box.Center + box.Axis[axis] * (side * box.Half[axis]);
    const float3 du = box.Axis[u] * box.Half[u], dv = box.Axis[v] * box.Half[v];
    out[0] = centre - du - dv;
    out[1] = centre + du - dv;
    out[2] = centre + du + dv;
    out[3] = centre - du + dv;
}

// A convex polytope as the narrowphase addresses it: a pose, and a way to ask for its vertices. A box
// is one of these too - eight corners it computes rather than stores - which is what lets one support
// function serve every convex algorithm below. A sphere and a capsule are here as the one or two points
// of the core they are the neighbourhood of, with the radius applied to the answer rather than to the
// search: the Minkowski difference of two cores is that of the two solids moved in by the two radii, so
// everything below works on cores and takes the radii off at the end.
struct Poly {
    float3 Center;
    float4 Orientation;
    float3 Half; // a box's half extents, or in y the half length of the segment a capsule surrounds
    float Radius; // a sphere's or a capsule's, and zero for a polytope
    uint First, Count; // a hull's run of the vertex pool
    uint FirstFace, FaceCount; // and of the face pool, the faces its cook recovered
    uint3 Corner; // and a triangle's three, which are anywhere in it rather than a run
    uint Kind; // the ShapeKind it was made from, which is what says how to read the fields above
};

static Poly MakePoly(Pose pose, Shape shape) {
    return {pose.Position, pose.Orientation, shape.HalfExtents, shape.Radius, shape.FirstVertex, shape.VertexCount,
            shape.FirstFace, shape.FaceCount, uint3(0), shape.Kind};
}

// One triangle of a mesh, as a polytope with no thickness. Everything below works on it unchanged. What
// it is missing is a side, which is the caller's business.
static Poly MakeTriangle(Pose pose, Triangle triangle) {
    return {pose.Position, pose.Orientation, float3(0), 0, 0, 0, 0, 0, uint3(triangle.A, triangle.B, triangle.C), ShapeMesh};
}

static uint PolyCount(Poly poly) {
    if (poly.Kind == ShapeBox) return 8;
    if (poly.Kind == ShapeHull) return poly.Count;
    if (poly.Kind == ShapeMesh) return 3;
    return poly.Kind == ShapeCapsule ? 2 : 1; // a sphere's core is a single point
}

// Vertex `i` in world space. A box's eight are the sign patterns of its half extents, in the order the
// bits of `i` give them, which is the order the box-vs-plane manifold names its corners in.
static float3 LocalVertex(Poly poly, device const float3 *pool, uint i) {
    float3 local = float3(0);
    if (poly.Kind == ShapeBox) local = poly.Half * float3((i & 1) ? 1.f : -1.f, (i & 2) ? 1.f : -1.f, (i & 4) ? 1.f : -1.f);
    else if (poly.Kind == ShapeHull) local = pool[poly.First + i];
    else if (poly.Kind == ShapeMesh) local = pool[poly.Corner[i]];
    else if (poly.Kind == ShapeCapsule) local = float3(0, i == 0 ? -poly.Half.y : poly.Half.y, 0);
    return local;
}

static float3 PolyVertex(Poly poly, device const float3 *pool, uint i) {
    return poly.Center + Rotate(poly.Orientation, LocalVertex(poly, pool, i));
}

// The vertex reaching furthest along `direction`, as an index rather than a point: a contact has to be
// named by the geometry that made it, and for a polytope that name is which corner it was.
static uint PolySupport(Poly poly, device const float3 *pool, float3 direction) {
    const float3 local = Rotate(QuatConjugate(poly.Orientation), direction);
    if (poly.Kind == ShapeBox) return (local.x > 0 ? 1u : 0u) | (local.y > 0 ? 2u : 0u) | (local.z > 0 ? 4u : 0u);
    if (poly.Kind == ShapeCapsule) return local.y > 0 ? 1u : 0u;
    if (poly.Kind == ShapeMesh) {
        uint best = 0;
        float furthest = -INFINITY;
        for (uint i = 0; i < 3; ++i) {
            const float reach = dot(pool[poly.Corner[i]], local);
            if (reach > furthest) {
                furthest = reach;
                best = i;
            }
        }
        return best;
    }
    if (poly.Kind != ShapeHull) return 0;
    uint best = 0;
    float furthest = -INFINITY;
    for (uint i = 0; i < poly.Count; ++i) {
        const float reach = dot(pool[poly.First + i], local);
        if (reach > furthest) {
            furthest = reach;
            best = i;
        }
    }
    return best;
}

// And that vertex itself, where the caller wants the point rather than the name of it.
static float3 PolySupportPoint(Poly poly, device const float3 *pool, float3 direction) {
    return PolyVertex(poly, pool, PolySupport(poly, pool, direction));
}

// How far the polytope's furthest vertex sits from its own centre, which is the scale a face tolerance
// has to be measured against - a face is flat only to the precision its own size is known to.
static float PolyReach(Poly poly, device const float3 *pool) {
    if (poly.Kind == ShapeBox) return length(poly.Half);
    if (poly.Kind == ShapeCapsule) return poly.Half.y;
    // A triangle's own size, not its distance from the mesh's origin: a hull is centred on its centre
    // of mass so a vertex's length is the shape's radius, and a triangle is wherever the mesh put it.
    if (poly.Kind == ShapeMesh)
        return max(max(distance(pool[poly.Corner[0]], pool[poly.Corner[1]]), distance(pool[poly.Corner[1]], pool[poly.Corner[2]])),
                   distance(pool[poly.Corner[2]], pool[poly.Corner[0]]));
    if (poly.Kind != ShapeHull) return 0;
    float reach = 0;
    for (uint i = 0; i < poly.Count; ++i) reach = max(reach, length(pool[poly.First + i]));
    return reach;
}

// A sphere and a capsule are the same shape with a different core: every point within Radius of a
// segment. A sphere's segment has zero length, which is why one path serves both.
struct Core {
    float3 From, To;
    float Radius;
};

static Core MakeCore(Pose pose, Shape shape) {
    const float3 along = Rotate(pose.Orientation, float3(0, shape.HalfExtents.y, 0)); // zero for a sphere
    return {pose.Position - along, pose.Position + along, shape.Radius};
}

static bool IsRound(uint kind) { return kind == ShapeSphere || kind == ShapeCapsule; }

// The point of a box nearest `at`, and how far outside the box that leaves `at`. A point inside has no
// nearest surface point in the useful sense, so it comes out through the face it is least deep under
// and the distance comes back negative to say so.
static float3 ClosestOnBox(BoxPose box, float3 at, thread float &distance, thread uint &feature) {
    const float3 offset = at - box.Center;
    float3 local = float3(dot(offset, box.Axis[0]), dot(offset, box.Axis[1]), dot(offset, box.Axis[2]));
    const float3 clamped = clamp(local, -box.Half, box.Half);
    const float3 outside = local - clamped;
    // Which of the three axes had to be clamped, and to which side. Together they name the face, edge
    // or vertex the point landed on, which is a name the geometry owns rather than the search order.
    feature = 0;
    for (uint i = 0; i < 3; ++i) feature |= (local[i] != clamped[i] ? (local[i] > 0 ? 1u : 2u) : 0u) << (2 * i);
    if (dot(outside, outside) > 1e-14f) {
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

// The edge of a box running along `axis` that reaches furthest in `direction`, as its two endpoints.
// Which of the four parallel edges that is comes back too, since the contact it makes has to be named
// by the edge itself rather than by the axis it runs along - the other three would inherit its dual.
static void SupportEdge(BoxPose box, uint axis, float3 direction, thread float3 *ends, thread uint &which) {
    const uint u = (axis + 1) % 3, v = (axis + 2) % 3;
    const float su = dot(direction, box.Axis[u]) >= 0 ? 1.f : -1.f;
    const float sv = dot(direction, box.Axis[v]) >= 0 ? 1.f : -1.f;
    const float3 centre = box.Center + box.Axis[u] * (box.Half[u] * su) + box.Axis[v] * (box.Half[v] * sv);
    ends[0] = centre - box.Axis[axis] * box.Half[axis];
    ends[1] = centre + box.Axis[axis] * box.Half[axis];
    which = (su > 0 ? 2u : 0u) | (sv > 0 ? 1u : 0u);
}

// The point of a segment nearest `at`.
static float3 ClosestOnSegment(float3 from, float3 to, float3 at) {
    const float3 along = to - from;
    const float length_squared = dot(along, along);
    return length_squared < 1e-12f ? from : from + along * clamp(dot(at - from, along) / length_squared, 0.f, 1.f);
}

// The closest pair of points on two segments, clamped to their ends.
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

// Sutherland-Hodgman against one half-space, keeping what is behind the plane, and carrying each
// point's name along with it.
//
// A name is the set of planes the point lies on: bits 0 to 3 the four edges of the incident face, bits
// 4 to 7 the four side planes of the reference face. Corner k of the incident face sits on edges k-1
// and k, and a point cut from the segment between two others sits on whatever both of them sat on,
// plus the plane that cut it - which is the whole rule, and it holds however many times a point has
// been through the clip. Two of those bits are always set and the pair names the point uniquely, so
// the same geometry gets the same name next step whatever order the points came out in.
//
// `tolerance` is how far outside the plane still counts as on it. Two faces that meet exactly have
// every corner of one *on* a side plane of the other, and the two sides compute that number by
// different arithmetic, so it lands either side of zero on rounding alone. Called exactly, a corner a
// rounding error outside is dropped and replaced by a cut point in the same place under a different
// name, so each corner appears twice and the pair fight over one piece of geometry.
static uint ClipAgainst(
    thread float3 *poly, thread uint *names, uint count, float3 normal, float offset, uint plane,
    float tolerance, uint limit, thread float3 *out, thread uint *out_names
) {
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
        // A cut at either end of the edge is that endpoint under another name, and whichever of the
        // two the clip keeps is the same point, so only a cut strictly between them is a new one.
        const float at = in_from / (in_from - in_to);
        if (at <= 1e-4f || at >= 1 - 1e-4f) continue;
        out[kept] = from + (to - from) * at;
        out_names[kept] = (names[i] & names[next]) | plane;
        ++kept;
    }
    return kept;
}

// -------------------------------------------------------------------------------------------------
// Convex against convex, the one part of the engine with no reference to diff against: none of
// avbd-demo2d, MetalAVBD or webphysics has a hull at all. It follows the sources LiteratureReview.md
// section "Convex queries" names instead - GJK with Montanari's signed-volume subalgorithm for the
// direction the two are apart along, bounded EPA in fixed scratch for the one they overlap along, and a
// clipped one-shot manifold from the faces that direction names.
//
// Nothing here knows a polytope's topology, only its vertices through Poly's support function, which is
// what lets one path serve hull against hull, hull against box, and hull against a capsule's segment.
// -------------------------------------------------------------------------------------------------

// What clipping one face into another can produce, which is more than either of them holds: a convex
// polygon cut by a half-plane gains a vertex, so an eight point face clipped against an eight edge one
// is a sixteen-gon. Bounding the clip at MaxFacePoints instead drops the overflow as a *contiguous run*
// of the perimeter - half the shape, which does not carry the body's centre - and a stack of two
// octagonal prisms twisted against each other falls over. Jolt guards the same thing from the other
// side, filling only half of its supporting-face buffer "since extra edges will be generated by
// clipping". Reduction takes this down to four immediately, so the width costs registers, not slots.
constant uint MaxClipPoints = 2 * MaxFacePoints;
// EPA's scratch. Bounded on purpose: a growing face heap is the worst-behaved thing a GPU kernel can
// be asked to hold, so it stops expanding and answers with the best face it has reached.
constant uint MaxEpaVertices = 16;
constant uint MaxEpaFaces = 32;

// A point of the Minkowski difference of the two cores, remembering which vertex of each made it - so
// a contact that comes out of the simplex can still be named by the geometry behind it.
struct Mink {
    float3 At;
    uint IndexA, IndexB;
};

static Mink MinkSupport(Poly a, Poly b, device const float3 *pool, float3 direction) {
    const uint ia = PolySupport(a, pool, direction), ib = PolySupport(b, pool, -direction);
    return {PolyVertex(a, pool, ia) - PolyVertex(b, pool, ib), ia, ib};
}

// -------------------------------------------------------------------------------------------------
// The signed-volume distance subalgorithm (Montanari, Petrinic and Barbieri, TOG 36(3) 2017, kept at
// ~/acoustic_solver_papers/2017_montanari-petrinic-barbieri_improving-gjk-signed-volumes.pdf), which
// replaces both Johnson's subalgorithm and the backup procedure GJK otherwise needs. Written from the
// paper - openGJK was not read, and nothing here is derived from it.
//
// Three things it does that the textbook Voronoi-region version does not:
//
// - It searches from the inside out, inspecting 2^m regions rather than 2^(m+1)-1: only the vertices
//   whose own barycentric coordinate has the wrong sign can be dropped.
// - It never drops the newest vertex, since the origin can never be in the region of the older ones
//   alone. That needs the simplex ordered newest-first, which is why Gjk below prepends.
// - It projects onto whichever Cartesian plane or axis the simplex shades most of, which is what makes
//   it accurate on a simplex that has gone nearly flat - and a hull's coplanar face points produce
//   those constantly. A needle simplex then yields a NaN the sign comparison rejects rather than a
//   division that poisons the answer, which is the whole of the backup procedure, gone.
// -------------------------------------------------------------------------------------------------

// The paper's CompareSigns. Zero and NaN both answer no, which is what makes a degenerate simplex
// exclude the region it came from instead of being believed.
static bool SameSign(float a, float b) { return (a > 0 && b > 0) || (a < 0 && b < 0); }

// Six times the signed volume of the tetrahedron, which is the determinant the barycentric
// coordinates are ratios of. Replacing one corner by the origin gives that corner's coordinate.
static float Signed4(float3 a, float3 b, float3 c, float3 d) { return dot(b - a, cross(c - a, d - a)); }

// And twice the signed area of the triangle projected onto the (u, v) plane, the same way.
static float Signed2(float3 a, float3 b, float3 c, uint u, uint v) {
    return (b[u] - a[u]) * (c[v] - a[v]) - (b[v] - a[v]) * (c[u] - a[u]);
}

// The point of a segment nearest the origin, with a mask of which ends carry it. s1 is the newest.
static float3 SignedVolume1(float3 s1, float3 s2, thread uint &mask) {
    const float3 along = s2 - s1;
    const float3 projected = s2 - along * (dot(s2, along) / dot(along, along));
    // The axis the segment shades the longest, which is the one its coordinates are best conditioned on.
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
    mask = 1; // the origin is past the newest vertex, and the older one cannot be the nearest alone
    return s1;
}

// The point of a triangle nearest the origin, likewise.
static float3 SignedVolume2(float3 s1, float3 s2, float3 s3, thread uint &mask) {
    const float3 turn = cross(s2 - s1, s3 - s1);
    // The origin projected onto the plane of the triangle. A needle triangle makes this a NaN, which
    // every sign comparison below then answers no to.
    const float3 projected = turn * (dot(s1, turn) / dot(turn, turn));
    // The Cartesian plane the triangle shades the largest area on, taken in cyclic order so the sign
    // of an area means the same thing whichever plane won.
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
    for (uint j = 1; j < 3; ++j) { // never the newest vertex, which is why j starts at one
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

// And of a tetrahedron. All four coordinates agreeing with its volume is the origin inside it, which
// is the answer GJK is really asking for: the two overlap, and EPA takes the simplex from here.
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
        const uint from[3] = {0, j == 1 ? 2u : 1u, j == 3 ? 2u : 3u}; // where each of the three came from
        mask = 0;
        for (uint b = 0; b < 3; ++b) mask |= ((sub >> b) & 1) << from[b];
    }
    return closest;
}

// The point of the simplex nearest the origin, with the simplex cut back to just the vertices that
// carry it - which is what keeps it a simplex as it walks.
static float3 ReduceSimplex(thread Mink *simplex, thread uint &count, thread bool &inside) {
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

// GJK, Algorithm 1 of the same paper. Comes back with the direction the two cores are apart along -
// out of b towards a, which is the sign the solve wants - and how far, or says they overlap and
// leaves the tetrahedron it finished with for EPA to grow.
//
// The new support point goes in at the front, since the subalgorithm above takes s1 for the newest.
static bool Gjk(
    Poly a, Poly b, device const float3 *pool, thread Mink *simplex, thread uint &count,
    thread float3 &direction, thread float &distance
) {
    float3 search = a.Center - b.Center;
    if (dot(search, search) < 1e-12f) search = float3(1, 0, 0);
    direction = normalize(search);
    distance = 0;
    simplex[0] = MinkSupport(a, b, pool, search);
    count = 1;
    float3 closest = simplex[0].At;
    for (uint iteration = 0; iteration < 32; ++iteration) {
        const float squared = dot(closest, closest);
        // Eq. 9, relative to the simplex itself rather than to an absolute length: the origin is on
        // the difference, so the two are touching exactly and the last direction is as good as any.
        float scale = 0;
        for (uint i = 0; i < count; ++i) scale = max(scale, dot(simplex[i].At, simplex[i].At));
        if (squared <= 1e-10f * scale) return false;
        const Mink next = MinkSupport(a, b, pool, -closest);
        // Eq. 10, and its companion: a vertex already in the simplex, or one that reaches no further
        // towards the origin than the simplex already does, means this is the face facing the origin.
        bool repeated = false;
        for (uint i = 0; i < count; ++i) repeated = repeated || (simplex[i].IndexA == next.IndexA && simplex[i].IndexB == next.IndexB);
        if (repeated || squared - dot(next.At, closest) <= 1e-8f * squared) break;
        for (uint i = count; i > 0; --i) simplex[i] = simplex[i - 1];
        simplex[0] = next;
        ++count;
        bool inside = false;
        closest = ReduceSimplex(simplex, count, inside);
        if (inside) return true;
    }
    distance = length(closest);
    if (distance > 1e-9f) direction = closest / distance;
    return false;
}

// The expanding polytope: grow the tetrahedron GJK finished inside towards the face of the Minkowski
// difference nearest the origin. That face's outward normal is the least direction the two have to
// move apart along, and its distance is by how much - so the contact normal is its opposite.
static bool Epa(Poly a, Poly b, device const float3 *pool, thread Mink *simplex, thread float3 &normal, thread float &depth) {
    Mink vertices[MaxEpaVertices];
    uint faces[MaxEpaFaces][3];
    float3 planes[MaxEpaFaces];
    float offsets[MaxEpaFaces];
    uint vertex_count = 4, face_count = 0;
    for (uint i = 0; i < 4; ++i) vertices[i] = simplex[i];

    const uint corners[4][3] = {{0, 1, 2}, {0, 2, 3}, {0, 3, 1}, {1, 3, 2}};
    const uint apexes[4] = {3, 1, 2, 0};
    for (uint f = 0; f < 4; ++f) {
        uint i = corners[f][0], j = corners[f][1], k = corners[f][2];
        float3 turn = cross(vertices[j].At - vertices[i].At, vertices[k].At - vertices[i].At);
        if (dot(turn, vertices[apexes[f]].At - vertices[i].At) > 0) { // wound away from the fourth corner, so it faces out
            const uint swap = j;
            j = k;
            k = swap;
            turn = -turn;
        }
        const float area = length(turn);
        if (area < 1e-18f) return false; // a flat simplex has no inside to expand
        faces[f][0] = i;
        faces[f][1] = j;
        faces[f][2] = k;
        planes[f] = turn / area;
        offsets[f] = dot(planes[f], vertices[i].At);
        ++face_count;
    }

    for (uint iteration = 0; iteration < 24; ++iteration) {
        uint best = 0;
        float least = INFINITY;
        for (uint f = 0; f < face_count; ++f) {
            if (offsets[f] >= least) continue;
            least = offsets[f];
            best = f;
        }
        normal = planes[best];
        depth = max(least, 0.f);
        // Out of room to grow, so answer with the best face reached - which is what bounded means.
        if (vertex_count == MaxEpaVertices || face_count + 8 > MaxEpaFaces) return true;
        const Mink next = MinkSupport(a, b, pool, normal);
        if (dot(next.At, normal) - least < 1e-6f) return true; // nothing further out: this is the face

        // The faces the new point can see come out, and the rim they leave is filled in from it. An
        // edge shared by two departing faces is interior to the hole rather than on its rim, which is
        // what cancelling a directed edge against its reverse finds.
        uint2 rim[MaxEpaFaces];
        uint rim_count = 0, kept = 0;
        for (uint f = 0; f < face_count; ++f) {
            if (dot(planes[f], next.At) - offsets[f] > 1e-9f) {
                for (uint e = 0; e < 3; ++e) {
                    const uint2 edge = uint2(faces[f][e], faces[f][(e + 1) % 3]);
                    bool cancelled = false;
                    for (uint r = 0; r < rim_count && !cancelled; ++r) {
                        if (rim[r].x != edge.y || rim[r].y != edge.x) continue;
                        rim[r] = rim[--rim_count];
                        cancelled = true;
                    }
                    if (!cancelled && rim_count < MaxEpaFaces) rim[rim_count++] = edge;
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
        for (uint r = 0; r < rim_count && face_count < MaxEpaFaces; ++r) {
            const uint i = rim[r].x, j = rim[r].y;
            const float3 turn = cross(vertices[j].At - vertices[i].At, vertices[apex].At - vertices[i].At);
            const float area = length(turn);
            if (area < 1e-18f) continue; // a sliver, which contributes no direction worth having
            faces[face_count][0] = i;
            faces[face_count][1] = j;
            faces[face_count][2] = apex;
            planes[face_count] = turn / area;
            offsets[face_count] = dot(planes[face_count], vertices[i].At);
            ++face_count;
        }
        if (face_count == 0) return false;
    }
    return true;
}

// At most `limit` of the polytope's vertices lying at or past `threshold` along `axis`, each named by
// its own index in the polytope. Where that many or fewer qualify it is every one of them.
//
// Where more do, which a coin lying flat has as soon as it is more than eight sided, taking whichever
// come first takes a contiguous run of the rim - half a disc, which does not carry the body's own
// centre, so it leans off the uncovered side a little further every step. So they are picked by spread
// instead: the lowest-indexed qualifier anchors the set, that being a name that does not move, and each
// one after it is whichever qualifier stands furthest from everything already kept.
static uint SpreadSupport(
    Poly poly, device const float3 *pool, float3 axis, float threshold, uint limit, thread float3 *out, thread uint *names
) {
    const uint count = PolyCount(poly);
    uint qualify = 0;
    for (uint i = 0; i < count; ++i) qualify += dot(PolyVertex(poly, pool, i), axis) >= threshold ? 1 : 0;

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
            if (nearest <= widest) continue; // ties to the lower index, so the set is a fixed one
            widest = nearest;
            pick = i;
        }
        out[found] = PolyVertex(poly, pool, pick);
        names[found] = pick;
        ++found;
    }
    return found;
}

// The face `poly` presents along `direction`: its vertices in the order that makes consecutive pairs
// its edges, with the face's own plane normal, out of the polytope. One vertex is a corner and two are
// an edge, and either is a face this cannot hold a manifold on.
//
// Which face is a question the shape answers rather than a question about heights. Every kind has its
// faces by construction - a box's from the dominant axis of the direction, a triangle's being the
// triangle, a capsule core's being its segment - and a hull's are recovered by its cook, so all four
// are exact and none needs a tolerance. See HullFace in Shared.h for why a height tolerance cannot
// stand in for the question.
//
// Which point comes first is taken from the lowest vertex index rather than from the search, so edge
// zero of a face is the same edge every step and a point clipped against it keeps its name.
static uint SupportFace(
    Poly poly, device const float3 *pool, device const HullFace *hull_faces, float3 direction,
    thread float3 *out, thread uint *names, thread float3 &plane
) {
    // In the polytope's own frame, where its faces are, so the choice is free of where the body stands.
    const float3 local = normalize(Rotate(QuatConjugate(poly.Orientation), direction));
    float3 local_plane = local;
    uint corners[MaxFacePoints], found = 0;

    if (poly.Kind == ShapeHull) {
        uint best = 0;
        float most = -INFINITY;
        for (uint f = 0; f < poly.FaceCount; ++f) {
            const float along = dot(hull_faces[poly.FirstFace + f].Normal, local);
            if (along <= most) continue; // ties to the lower face, so a face is a fixed answer
            most = along;
            best = f;
        }
        if (poly.FaceCount == 0) return 0;
        const HullFace face = hull_faces[poly.FirstFace + best];
        local_plane = face.Normal;
        // The cook wound it about that normal, started it at its lowest vertex, and sampled it around
        // its rim if it was wider than this may hold - so it arrives named, wound and the right width.
        for (uint i = 0; i < face.Count && i < MaxFacePoints; ++i) corners[found++] = face.Corner[i];
    } else if (poly.Kind == ShapeBox) {
        // The face the direction points most strongly out of, and its four corners wound about that
        // normal. A corner's name is the same bitmask PolyVertex reads, so the two agree by construction.
        const float3 magnitude = abs(local);
        const uint axis = magnitude.x >= magnitude.y && magnitude.x >= magnitude.z ? 0u : (magnitude.y >= magnitude.z ? 1u : 2u);
        const bool positive = local[axis] > 0;
        local_plane = float3(0);
        local_plane[axis] = positive ? 1 : -1;
        const uint across = (axis + 1) % 3, along = (axis + 2) % 3;
        // Wound so the turn from `across` to `along` is positive about the face normal on the positive
        // side, and reversed on the negative one, which is what keeps the loop outward either way.
        const uint2 order[4] = {uint2(0, 0), uint2(1, 0), uint2(1, 1), uint2(0, 1)};
        for (uint i = 0; i < 4; ++i) {
            const uint2 corner = order[positive ? i : 3 - i];
            corners[found++] = (positive ? (1u << axis) : 0u) | (corner.x << across) | (corner.y << along);
        }
    } else if (poly.Kind == ShapeMesh) {
        // A triangle is one face and has two sides. Wound so its normal is the one being asked for.
        const float3 a = pool[poly.Corner[0]], b = pool[poly.Corner[1]], c = pool[poly.Corner[2]];
        const float3 turn = cross(b - a, c - a);
        if (length(turn) < 1e-18f) return 0;
        const bool forwards = dot(turn, local) >= 0;
        local_plane = forwards ? normalize(turn) : -normalize(turn);
        for (uint i = 0; i < 3; ++i) corners[found++] = forwards ? i : 2 - i;
    } else {
        // A capsule's core is a segment and a sphere's is a point, neither of which is a face. They come
        // back as they are and the caller takes the closest-pair branch.
        const uint count = PolyCount(poly);
        for (uint i = 0; i < count; ++i) corners[found++] = i;
    }

    // How much of that face is presented: whether the body meets it flat or stands on one of its edges
    // or corners. That is a question about the direction and so needs a tolerance, but one that can
    // only ever take a subset of the one face topology already chose, never union two. Fewer than three
    // left is an edge or a corner and the caller answers for it.
    //
    // A triangle keeps all three whatever the direction, since a half-metre triangle loses its third
    // corner to a twentieth of a degree of error in it. A capsule's core is not a face at all.
    if (poly.Kind == ShapeHull || poly.Kind == ShapeBox) {
        float furthest = -INFINITY;
        for (uint i = 0; i < found; ++i) furthest = max(furthest, dot(LocalVertex(poly, pool, corners[i]), local));
        const float tolerance = 1e-3f * PolyReach(poly, pool) + 1e-6f;
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

    // Rotated so the lowest name comes first. A hull's cook already did it and a box's winding starts
    // wherever the axis put it, so this is what makes the two agree on where edge zero of a face is.
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

// The unit normal of the side plane along edge `e` of a face lying in `plane`, out of the face - so
// what is behind it is inside it. False where the edge is too short to name a direction at all.
static bool SidePlane(thread const float3 *face, uint count, uint e, float3 plane, thread float3 &unit) {
    const float3 side = cross(face[(e + 1) % count] - face[e], plane);
    const float span = length(side);
    if (span < 1e-12f) return false;
    unit = side / span;
    return true;
}

// The manifold between two convex polytopes, `a` this body's and `b` the other's. Fills the pair of
// points each contact holds - where it sits on a, and where on b - with the name of the geometry that
// made it, and the normal out of b towards a.
//
// A name is the planes the point lies on, in the same bitmask sense the box path uses, with bits 0-7 an
// edge of the incident face and bits 8-15 a side plane of the reference one, then which face of each
// polytope those were as its lowest vertex index, then which body presented the reference face. Nothing
// in it is a position in the output array, which is what lets a dual find its point again.
//
// `known` is a direction out of b towards a that the caller already knows the two are apart along, or
// zero to go and find one. A triangle is what it is for: it has exactly one face, and a search that
// comes back a thousandth of a radian off is enough to stop a perfectly flat triangle presenting a face
// at all. Given the direction, the triangle presents its whole face and always wins the reference,
// which is the way round that names a contact after the geometry under it.
static uint ConvexManifold(
    Poly a, Poly b, device const float3 *pool, device const HullFace *hull_faces, float margin, float3 known, thread float3 *here,
    thread float3 *there, thread uint *names, thread float3 &normal
) {
    Mink simplex[4];
    uint simplex_count = 0;
    float3 axis; // out of b towards a
    float distance; // between the cores, and negative by the overlap when they are inside each other
    const bool told = any(known != 0);
    if (told) {
        axis = known;
        // How far apart they are along it, which is between the point of each that faces the other.
        const float3 near_a = PolySupportPoint(a, pool, -axis);
        const float3 near_b = PolySupportPoint(b, pool, axis);
        distance = dot(near_a - near_b, axis);
    } else if (Gjk(a, b, pool, simplex, simplex_count, axis, distance)) {
        float depth;
        float3 out_of_a;
        if (simplex_count < 4 || !Epa(a, b, pool, simplex, out_of_a, depth)) return 0;
        // EPA's face faces out of the difference, which is the way a has to move to leave b, so the
        // normal out of b towards a is its opposite.
        axis = -out_of_a;
        distance = -depth;
    }
    const float gap = distance - a.Radius - b.Radius;
    if (gap >= margin) return 0;

    float3 face_a[MaxFacePoints], face_b[MaxFacePoints];
    uint name_a[MaxFacePoints], name_b[MaxFacePoints];
    float3 plane_a, plane_b; // each out of its own polytype, towards the other
    const uint count_a = SupportFace(a, pool, hull_faces, -axis, face_a, name_a, plane_a);
    const uint count_b = SupportFace(b, pool, hull_faces, axis, face_b, name_b, plane_b);

    if (count_a < 3 && count_b < 3) {
        // Neither presents a face: an edge across an edge, or a corner against something, which holds
        // one point wherever it is. Both sets are a point or a segment, so this is the nearest pair.
        float3 on_a, on_b;
        ClosestOnSegments(face_a[0], face_a[count_a - 1], face_b[0], face_b[count_b - 1], on_a, on_b);
        normal = axis;
        here[0] = on_a - a.Radius * axis;
        there[0] = on_b + b.Radius * axis;
        names[0] = (1u << 29) | name_a[0] | (name_a[count_a - 1] << 6) | (name_b[0] << 12) | (name_b[count_b - 1] << 18);
        return 1;
    }

    // Whichever face lies flatter against the normal holds the contact, and the other is clipped into
    // it - except that a caller who handed in the direction handed in a's own face normal with it, so a
    // takes the reference outright. Left to the comparison that case is a tie between two numbers both
    // one to within the rounding of a normalize: a box lying flat across a mesh would get the reference
    // on the triangle under one part of it and on itself for the next, so half its manifold would be
    // points at the corners of the tessellation, named after the body rather than the geometry under
    // them, and out of reach of the seam rule that would have dropped them.
    const bool reference_is_a = count_a >= 3 && (told || count_b < 3 || abs(dot(plane_a, axis)) >= abs(dot(plane_b, axis)));
    const float3 reference_plane = reference_is_a ? plane_a : plane_b; // out of the reference, towards the incident
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
        // A corner of the incident face sits on the two of its edges that meet there, and a lone point
        // or the end of a segment answers to itself.
        poly_names[i] = incident_count >= 3 ? ((1u << i) | (1u << ((i + incident_count - 1) % incident_count))) : (1u << i);
    }
    // Relative to where the face is, since the rounding in a dot product is relative to what went into it.
    float scale = 1;
    for (uint i = 0; i < reference_count; ++i) scale = max(scale, length(reference[i]));
    const float tolerance = 1e-5f * scale;
    if (incident_count >= 3) {
        for (uint e = 0; e < reference_count; ++e) {
            float3 unit; // out of the reference face, so what is kept is inside it
            if (!SidePlane(reference, reference_count, e, reference_plane, unit)) continue;
            poly_count = ClipAgainst(poly, poly_names, poly_count, unit, dot(unit, reference[e]), 1u << (8 + e), tolerance, MaxClipPoints, clipped, clipped_names);
            for (uint i = 0; i < poly_count; ++i) {
                poly[i] = clipped[i];
                poly_names[i] = clipped_names[i];
            }
        }
    } else if (incident_count == 1) {
        // One point, which the loop below would otherwise take on trust. It is on the reference face
        // only if it is inside every one of its side planes - a corner level with a face but off to
        // the side of it is not touching it, and against a triangle that is most of the mesh.
        for (uint e = 0; e < reference_count; ++e) {
            float3 unit;
            if (!SidePlane(reference, reference_count, e, reference_plane, unit)) continue;
            if (dot(unit, poly[0] - reference[e]) > tolerance) poly_count = 0;
        }
    } else if (incident_count == 2) {
        // A segment, which Sutherland-Hodgman cannot clip without emitting its cut point twice - once
        // going out and once coming back. So it is clipped as an interval instead, each limit keeping
        // the name of whichever side plane set it, exactly as a capsule against a box is.
        const float3 from = poly[0], along = poly[1] - poly[0];
        float low = 0, high = 1;
        uint low_name = 1u << 0, high_name = 1u << 1;
        for (uint e = 0; e < reference_count; ++e) {
            float3 unit;
            if (!SidePlane(reference, reference_count, e, reference_plane, unit)) continue;
            const float offset = dot(unit, reference[e]);
            const float at_from = dot(unit, from) - offset, at_to = dot(unit, poly[1]) - offset;
            const float slope = at_to - at_from;
            if (abs(slope) < 1e-12f) { // parallel to the plane, so it is either wholly in or wholly out
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
            // Clipped down to a point, which is one contact rather than two in the same place.
            poly_count = high - low > 1e-6f ? 2 : 1;
        } else {
            poly_count = 0;
        }
    }

    uint reference_face = 63, incident_face = 63; // each named by its lowest vertex, which is geometry
    for (uint i = 0; i < reference_count; ++i) reference_face = min(reference_face, reference_names[i]);
    for (uint i = 0; i < incident_count; ++i) incident_face = min(incident_face, incident_names[i]);
    const float reference_offset = dot(reference_plane, reference[0]);
    const float reference_radius = reference_is_a ? a.Radius : b.Radius;
    const float incident_radius = reference_is_a ? b.Radius : a.Radius;

    uint found = 0;
    for (uint i = 0; i < poly_count && found < MaxClipPoints; ++i) {
        // How far the incident core point stands off the reference core's face plane, and what is left
        // of that once both surfaces have been let out to their radii.
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

// One piece of geometry, one row. A clip emits the same point twice wherever the geometry is degenerate
// against it - a corner landing exactly on a cutting plane, an edge clipped down to a sliver, a vertex
// meeting the ridge two faces share - and each of the two then carries its own dual and its own
// penalty, so the pair applies twice the force at one spot. Neither is wrong about where the body goes,
// which is why it shows as ringing and as a slot budget that will not fit rather than as a wrong
// position, and why the ones found before were found by hand rather than by a test.
//
// Welded here rather than in each clipper, since every path ends here and their degeneracies are not
// the same list. The scale is the one SupportFace resolves geometry at: points it would call one
// feature are one point. First kept wins.
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

// The four points of a manifold with the largest area between them, which is Gregorius's reduction
// (GDC 2015, kept at ~/acoustic_solver_papers/2015_gregorius_robust-contact-creation.pdf). Four points
// hold a face contact: they resist the body turning as well as sliding, and a fifth adds nothing a
// solver can use while costing a slot some other pair of this body needs.
//
// It is area the choice maximizes, not distance. Two points a long way apart on one edge of a face
// leave the body free to rock about that edge however deep they are.
//
// The set is picked from the geometry rather than from the order the points came out in, so a settled
// contact keeps the same four every step and their duals with them - which RbpScenes stack reports as
// churn if it ever stops being true.
static uint ReduceManifold(thread float3 *here, thread float3 *there, thread uint *names, uint found, float3 normal) {
    if (found <= ManifoldPoints) return found;
    // Every comparison below has to be settled by the geometry rather than by the last bits of a world
    // position, because a tie that falls the other way renames all four points and throws their duals
    // away. An octagon lying flat is exactly that: eight points at one depth and two squares of
    // identical area to choose between, so rounding alone flips the answer every few steps as the body
    // turns. So a challenger has to beat the incumbent by a margin relative to the manifold's own size,
    // which leaves every tie with the lowest-indexed point - the same relative-plus-absolute bias the
    // box SAT keeps on its reference axis, for the same reason.
    float extent = 0;
    for (uint i = 0; i < found; ++i) extent = max(extent, distance(here[i], here[0]));
    const float slack = 1e-5f * extent + 1e-9f, area_slack = 1e-5f * extent * extent + 1e-9f;

    uint keep[4]; // the four are what this picks - deepest, furthest, largest triangle, most area
    // The deepest, so whichever point is actually resolving penetration is always in the set.
    float deepest = INFINITY;
    keep[0] = 0;
    for (uint i = 0; i < found; ++i) {
        const float separation = dot(normal, here[i] - there[i]);
        if (separation >= deepest - slack) continue;
        deepest = separation;
        keep[0] = i;
    }
    // The furthest from it, which is the longest the manifold reaches.
    float furthest = -1;
    keep[1] = keep[0];
    for (uint i = 0; i < found; ++i) {
        const float3 span = here[i] - here[keep[0]];
        if (dot(span, span) <= furthest + area_slack) continue;
        furthest = dot(span, span);
        keep[1] = i;
    }
    // The one making the largest triangle on that edge. Its sign says which way the triangle is wound
    // about the normal, and swapping the pair to wind it positively is what lets the fourth point be
    // found by sign alone.
    float widest = 0;
    keep[2] = keep[0];
    for (uint i = 0; i < found; ++i) {
        const float area = dot(cross(here[keep[1]] - here[keep[0]], here[i] - here[keep[0]]), normal);
        if (abs(area) <= abs(widest) + area_slack) continue;
        widest = area;
        keep[2] = i;
    }
    if (widest == 0) return 4; // every point on one line, so there is no area to choose by
    if (widest < 0) {
        const uint swap = keep[1];
        keep[1] = keep[2];
        keep[2] = swap;
    }
    // And the point adding the most to that triangle: outside one of its edges, which is the side the
    // signed area comes out negative on, and the largest such area rather than the largest distance.
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

    // In the order they were found in, so which slot a point lands in does not depend on which of the
    // four it happened to be chosen as.
    float3 kept_here[4], kept_there[4];
    uint kept_names[4], kept = 0;
    for (uint i = 0; i < found; ++i) {
        bool wanted = false;
        for (uint k = 0; k < 4; ++k) wanted = wanted || keep[k] == i;
        if (!wanted) continue;
        kept_here[kept] = here[i];
        kept_there[kept] = there[i];
        kept_names[kept] = names[i];
        ++kept;
    }
    for (uint i = 0; i < kept; ++i) {
        here[i] = kept_here[i];
        there[i] = kept_there[i];
        names[i] = kept_names[i];
    }
    return kept;
}

// How many triangles of one mesh a single body may be collided against in a step, and how deep the
// walk down the tree may go. Both fixed, since a kernel cannot grow anything: what a body reaches past
// these is counted as a refusal, the same as a contact it had no slot for.
constant uint MaxMeshTriangles = 32;
constant uint MeshStackDepth = 32;

// The next batch of triangles of a mesh whose bounds a body's own reach into, taken by walking the tree
// over them. `low` and `high` are the body's box in the *mesh's* frame, since that is the frame the tree
// was built in and moving one box across is cheaper than moving every triangle back.
//
// The walk's stack belongs to the caller and survives between calls, so `MaxMeshTriangles` bounds how
// many triangles are held at once rather than how much of the mesh a body may touch. It has to: a body
// wide against the cut of the floor under it reaches past any fixed number - a four metre slab over
// third-of-a-metre quads is three hundred triangles - and stopping the walk there did not leave the
// body resting on the ones it had, it left it with no contacts at all and it fell through the floor.
//
// A leaf's `First` counts from the mesh's own first triangle and an interior node's from its own root,
// so a shape's tree says nothing about where in the pools it was put.
static uint GatherTriangles(
    Shape mesh, float3 low, float3 high, device const BvhNode *nodes, thread uint *stack, thread uint &depth, thread uint *out
) {
    uint found = 0;
    while (depth > 0) {
        const uint at = stack[--depth];
        const BvhNode node = nodes[mesh.RootNode + at];
        if (any(node.High < low) || any(node.Low > high)) continue;
        if (node.Count > 0) {
            // A leaf holds four, so a batch that cannot take all of them puts the node back for the next
            // one rather than splitting it - which keeps the walk's whole state in the stack.
            if (found + node.Count > MaxMeshTriangles) {
                stack[depth++] = at;
                return found;
            }
            for (uint i = 0; i < node.Count; ++i) out[found++] = node.First + i;
            continue;
        }
        // The left child was written straight after this node, so only the right one needs an index.
        // Two more than the stack holds is a tree deeper than four billion leaves, so it cannot happen -
        // but a kernel that walked off the end of an array would not say so, and this does.
        if (depth + 2 > MeshStackDepth) continue;
        stack[depth++] = at + 1;
        stack[depth++] = node.First;
    }
    return found;
}

// Whether a point lies along the line of edge `e` of a triangle, to the tolerance the clip that made
// the point measured with - which is what the two seam rules in CollectContacts both ask.
static bool OnEdgeLine(Poly face, device const float3 *pool, float3 outward, uint e, float3 at_point, float seam) {
    const float3 at = PolyVertex(face, pool, e);
    const float3 side = cross(PolyVertex(face, pool, (e + 1) % 3) - at, outward);
    const float span = length(side);
    return span > 1e-12f && abs(dot(side / span, at_point - at)) <= seam;
}

// The contacts a body held last step that nothing has claimed this one, and the total it reported.
// Every exit from CollectContacts goes through this, because a body that has stopped colliding at all -
// it went static, or lost its shape - has still ended every contact it was holding.
static void EndUnclaimed(
    device ContactEvent *events, device uint *counts, uint body, uint claimed, uint reported,
    thread const uint *was_feature, thread const Index *was_other, thread const Index *was_sub
) {
    for (uint j = 0; j < ContactsPerBody; ++j) {
        if (was_feature[j] == NoIndex) break; // the sentinel: the run was dense, so nothing follows
        if ((claimed & (1u << j)) != 0) continue;
        events[reported++] = ContactEvent{body, was_other[j], was_feature[j], was_sub[j], uint(ContactRemoved)};
    }
    counts[body] = reported;
}

// Filtered N^2 broadphase and narrowphase, one thread per body. That thread owns all of its body's
// contact slots, so it can read the previous step's before overwriting them and carry the dual across
// by feature - and nothing is appended by anyone else, so the pool is identical on every run.
kernel void CollectContacts(
    device Contact *contacts [[buffer(5)]], device const Pose *poses [[buffer(0)]],
    device const BodyMass *masses [[buffer(4)]], device const Index *body_shapes [[buffer(6)]],
    device const Shape *shapes [[buffer(8)]], device const float *frictions [[buffer(10)]],
    device const Velocity *velocities [[buffer(3)]],
    device const Filter *filters [[buffer(19)]], device const Index *jointed_to [[buffer(20)]],
    device ContactEvent *contact_events [[buffer(24)]], device uint *contact_event_counts [[buffer(25)]],
    device uint *contact_refusals [[buffer(26)]], device const float3 *hull_vertices [[buffer(27)]],
    device const Triangle *mesh_triangles [[buffer(28)]], device const BvhNode *bvh_nodes [[buffer(29)]],
    device const HullFace *hull_faces [[buffer(30)]], device const uint *quiet [[buffer(21)]],
    constant StepParams &p [[buffer(7)]], uint body [[thread_position_in_grid]]
) {
    if (body >= p.BodyCount) return;
    contact_refusals[body] = 0;
    device Contact *slots = contacts + body * ContactsPerBody;
    device ContactEvent *events = contact_events + body * EventsPerBody;
    // Which of last step's slots a point has claimed, and how many events this body has written. The
    // claim has to accumulate over every body this one touches before anything can be called removed,
    // since the slot a feature lands in has nothing to do with which pair produced it.
    uint claimed = 0, reported = 0;
    // Which of last step's slots each live contact inherited from, or NoIndex for one that is new.
    // Kept per slot rather than as a mask because a contact can lose its place later in the step to a
    // deeper one, and what it had claimed has to go back with it.
    uint inherited[ContactsPerBody];

    // The previous step's state, kept only so a matching feature can inherit it.
    uint was_feature[ContactsPerBody], was_stick[ContactsPerBody];
    Index was_other[ContactsPerBody]; // who the contact was against, which only a removal still needs
    Index was_sub[ContactsPerBody]; // and which part of it, which for a mesh is which triangle
    float3 was_lambda[ContactsPerBody], was_penalty[ContactsPerBody];
    float3 was_anchor_a[ContactsPerBody], was_anchor_b[ContactsPerBody];
    // A run is dense from zero - the fill below appends and evictions replace within the filled part -
    // so the first inactive slot ends it. One NoIndex sentinel marks that end and every reader of was_*
    // stops there, so the slots past it are neither loaded here nor read there.
    for (uint i = 0; i < ContactsPerBody; ++i) {
        if (!slots[i].Active) {
            was_feature[i] = NoIndex;
            break;
        }
        was_feature[i] = slots[i].Feature;
        was_other[i] = slots[i].BodyB;
        was_sub[i] = slots[i].SubShape;
        was_lambda[i] = slots[i].Lambda;
        was_penalty[i] = slots[i].Penalty;
        was_stick[i] = slots[i].Stick;
        was_anchor_a[i] = slots[i].AnchorA;
        was_anchor_b[i] = slots[i].AnchorB;
        slots[i].Active = false;
    }

    const Index shape_index = body_shapes[body];
    if (masses[body].InvMass == 0 || shape_index == NoIndex) {
        EndUnclaimed(events, contact_event_counts, body, claimed, reported, was_feature, was_other, was_sub);
        return;
    }
    const Shape shape = shapes[shape_index];
    if (shape.Kind == ShapePlane || shape.Kind == ShapeMesh) { // neither of the two static surfaces owns a manifold
        EndUnclaimed(events, contact_event_counts, body, claimed, reported, was_feature, was_other, was_sub);
        return;
    }
    const Pose pose = poses[body];
    const BoxPose box = MakeBox(pose, shape);
    const Poly own_poly = MakePoly(pose, shape);
    // This body's own lanes, which every partner below reads and none of them changes.
    const Filter own_filter = filters[body];
    const float own_friction = frictions[body], own_inverse_mass = masses[body].InvMass;
    // How far this body's own geometry reaches, which is a scan over a hull's every vertex and does not
    // depend on who it is being collided against.
    const float own_reach = PolyReach(own_poly, hull_vertices) + shape.Radius;

    uint count = 0;
    // A sleeping body's pairs against partners just as frozen - asleep themselves, or static - are
    // carried forward verbatim instead of re-collided: neither pose has moved, so every anchor, C0 and
    // dual is still exact and the narrowphase would reproduce them to the bit at full price. Coverage
    // survives by construction, since an approaching body is awake and its pairs are re-collided from
    // whichever side owns them, and the kept contacts are exactly the ones SpreadWaking travels
    // through. Sleep state is settled once a step, which is what lets both sides of a pair agree on
    // whether it is frozen without asking.
    const bool asleep = Asleep(quiet[body], p);
    if (asleep) {
        for (uint j = 0; j < ContactsPerBody; ++j) {
            if (was_feature[j] == NoIndex) break; // the sentinel: the run was dense
            const Index partner = was_other[j];
            if (body_shapes[partner] == NoIndex) continue; // removed, and its contacts end with it
            if (masses[partner].InvMass > 0 && !Asleep(quiet[partner], p)) continue; // awake: re-collide
            // The full slot is still in device memory - only its Active flag was cleared above - so
            // carrying it forward is a compacting copy. j never runs ahead of count, so nothing is
            // overwritten before it is read.
            slots[count] = slots[j];
            slots[count].Active = true;
            inherited[count] = j;
            ++count;
        }
    }
    // Every other body, whether or not the run is already full: which contacts a body keeps must not
    // be decided by which of them happened to be looked at first. It also makes the refusal count
    // below exact rather than a lower bound, since nothing goes uncollided.
    for (uint other = 0; other < p.BodyCount; ++other) {
        const Index other_shape = body_shapes[other];
        if (other == body || other_shape == NoIndex) continue;
        // The pairs the carry above already holds, and the mirror of it in the partner's own run:
        // two frozen poses have nothing new to say to each other.
        if (asleep && (masses[other].InvMass == 0 || Asleep(quiet[other], p))) continue;
        // One manifold per pair, owned by the lower-indexed body, as the references do. Generating it
        // from both sides instead gives an interface two independent constraint sets with two sets of
        // duals for one physical contact, which Jacobi's symmetry hides and Gauss-Seidel does not.
        // Static bodies never solve, so a box still owns its contact against a plane whatever the
        // indices are.
        //
        // It spreads the pairs unevenly - a scene is built from the ground up, so the bodies holding
        // everything else up own most of it - and splitting them by parity instead was measured and is
        // worse: in a stack consecutive indices always sum odd, so parity flips every pair rather than
        // alternating, and the middle of the stack then creeps about three hundredths of a millimetre a
        // step, under SleepSpeed and over SleepDrift, so it wakes itself for ever. Budgeting the run in
        // manifolds took away the reason to spread them at all - RbpScenes raft asks for no slot it
        // cannot have once settled, at either size it is built at.
        if (masses[other].InvMass > 0 && other < body) continue;

        // Each has to be in the other's mask, and a joint already holding the two together means the
        // overlap is by design and a contact fighting it is not a physical force.
        const Filter theirs = filters[other];
        if (!(own_filter.Layer & theirs.Collides) || !(theirs.Layer & own_filter.Collides)) continue;
        bool jointed = false;
        for (uint i = 0; i < JointsPerBody && !jointed; ++i) jointed = jointed_to[body * JointsPerBody + i] == other;
        if (jointed) continue;

        const Shape target = shapes[other_shape];
        const Pose target_pose = poses[other];
        // The softest the normal row of this pair may be: the inertial stiffness of the mass it has to
        // move, the pair's reduced mass over h squared. Sec. 3.4 is why. On a settled contact the dual
        // has absorbed the load, so C goes to zero, the beta ramp goes with it, and nothing opposes
        // Eq. 19's Gamma decay down to PenaltyMin - the penalty ends up under a hundredth of M/h^2, the
        // contact all but vanishes from the 6x6 block, ten iterations stop converging, and a rocking
        // mode too slow for implicit Euler to damp rings on.
        //
        // The paper says PenaltyMin "virtually has no impact, except for the very first iteration of
        // the very first frame", which holds for the loaded pendulum it measures - a joint under load
        // keeps C nonzero, so the ramp never lets the penalty return to the floor. A resting contact
        // does return, and there the floor is the whole of it.
        //
        // Measured over four decades of body mass, a six-box stack is quietest at M/h^2 itself, usable
        // over about [0.3, 3] of it, and unstable past 10. The band is a ratio, not an absolute, which
        // is why this cannot be a constant.
        //
        // The normal row only. Friction's penalty is the algorithmic one behind the stick constraint
        // rather than a material stiffness, and the cone already bounds what it may apply, so flooring
        // it there locks the stick-slip transition early and a sliding box stops short of mu g.
        const float pair_stiffness = PairStiffness(own_inverse_mass + masses[other].InvMass, p.DeltaTime);
        const float3 penalty_floor{max(p.PenaltyMin, pair_stiffness), p.PenaltyMin, p.PenaltyMin};
        const float friction = sqrt(own_friction * frictions[other]);
        // How close two manifold points have to be before they are one, which is the scale the
        // narrowphase resolves geometry at - the same number SupportFace decides which vertices make
        // up one face with. A mesh contributes nothing here and does not need to: the points sit on
        // the body, so the body's own size is the scale they are measured against.
        const float geometry = max(own_reach, PolyReach(MakePoly(target_pose, target), hull_vertices) + target.Radius);
        const float weld = 1e-3f * geometry + 1e-6f;
        // How far apart the pair may be and still be given contacts. A contact built between the
        // closest features while the bodies are still apart carries the gap it measured as slack: it
        // does no work unless the step's motion would more than consume that gap, and then removes
        // exactly the excess, so the step ends at touch. Continuous collision approximated by an
        // ordinary constraint, so the reach covers a step of motion - Avian's velocity-scaled margin
        // with MetalAVBD's gravity term at a horizon of one step.
        //
        // The gravity term is a whole g h^2 rather than the half free fall covers, because Integrate's
        // inertial target is x + h v + h^2 g and that is the motion this has to reach across. The
        // textbook half leaves it short by g h^2 / 2, which at 1/60 is 1.4 mm - three times the margin,
        // landing as depth a body is pushed back out of rather than as a gap it stopped across.
        //
        // Rotation is deliberately not in it: a contact built at the pose the step began from cannot
        // see swept orientation, so a thin plate spinning fast tunnels at any reach at all. That is the
        // sweep's job, later, and a term pretending to cover it would be worse than the gap.
        //
        // It replaces the margin in every *generation* test below and nowhere else. C0 keeps
        // ContactMargin, which is the resting depth a contact holds at rather than how far it reaches.
        const Velocity own_velocity = velocities[body], other_velocity = velocities[other];
        const float reach = p.ContactMargin +
            min(p.DeltaTime * (length(own_velocity.Linear - other_velocity.Linear) + length(p.Gravity) * p.DeltaTime),
                p.MaxContactReach);

        // Whether either surface at this contact is round. A contact point on one does not stay with
        // the material under it, which is what static friction's inherited anchors assume.
        const bool curved = IsRound(shape.Kind) || IsRound(target.Kind);

        // How many manifolds this pair has. Every shape but a mesh is one piece and presents one, and a
        // mesh presents one per triangle the body reaches, a batch at a time. The walk's stack lives out
        // here so it survives between batches - see GatherTriangles.
        uint candidates[MaxMeshTriangles], walk[MeshStackDepth], depth = 0;
        float3 low = 0, high = 0;
        if (target.Kind == ShapeMesh) {
            // The body's own box, taken across into the mesh's frame, which is the frame its tree was
            // built in. The corners of the body's polytope are exactly its box, and a round shape's
            // radius and the margin are what has to be let out beyond them.
            const uint corners = PolyCount(own_poly);
            low = INFINITY;
            high = -INFINITY;
            for (uint i = 0; i < corners; ++i) {
                const float3 at = Rotate(QuatConjugate(target_pose.Orientation), PolyVertex(own_poly, hull_vertices, i) - target_pose.Position);
                low = min(low, at);
                high = max(high, at);
            }
            const float let_out = own_poly.Radius + reach;
            low -= let_out;
            high += let_out;
            walk[depth++] = 0;
        }

        for (bool walking = true; walking;) {
            uint manifolds = 1;
            if (target.Kind == ShapeMesh) {
                manifolds = GatherTriangles(target, low, high, bvh_nodes, walk, depth, candidates);
                walking = depth > 0;
                if (manifolds == 0) break;
            } else {
                walking = false;
            }

            for (uint manifold = 0; manifold < manifolds; ++manifold) {
                // Which part of the other body's shape this manifold is against, which only a mesh has.
                Index sub_shape = NoIndex;
                // A manifold point is a pair: where it sits on this body, and where on the other. Their
                // separation along the normal is what the constraint measures, so they must not be the same
                // point - one of them is always a projection onto the other body's surface.
                float3 points_here[MaxClipPoints], points_there[MaxClipPoints], normal;
                // A feature names the geometry that produced a point - which corner, or which pair of faces
                // and which vertex of the clip - never its position in the output, or warm starting would
                // hand a dual to the wrong point whenever the set of touching corners changed.
                uint features[MaxClipPoints];
                uint found = 0;

                const bool hulled = shape.Kind == ShapeHull || target.Kind == ShapeHull;
                if (target.Kind == ShapeMesh) {
                    const Index index = target.FirstTriangle + candidates[manifold];
                    const Triangle triangle = mesh_triangles[index];
                    sub_shape = index;
                    const Poly face = MakeTriangle(target_pose, triangle);
                    const float3 first = PolyVertex(face, hull_vertices, 0);
                    const float3 turn = cross(PolyVertex(face, hull_vertices, 1) - first, PolyVertex(face, hull_vertices, 2) - first);
                    const float area = length(turn);
                    if (area < 1e-18f) continue; // a sliver the cook let through has no side to speak of
                    const float3 outward = turn / area; // out of the surface, which is what the winding says

                    // A surface has a side. A mesh has no inside for a body to be in, so a body wholly
                    // behind a triangle is past it rather than through it and nothing pushes it back -
                    // while one still reaching through from the front is pushed back out of the front.
                    const float3 top = PolySupportPoint(own_poly, hull_vertices, outward);
                    if (dot(top - first, outward) + own_poly.Radius <= 0) continue;

                    // The triangle goes in first and with its own normal for the direction, which is what
                    // makes it the reference face every time - so the manifold is the body's face clipped
                    // into the triangle, and every point is named after the geometry under it. The answer
                    // comes back the other way round, since the normal has to come out of the mesh.
                    found = ConvexManifold(face, own_poly, hull_vertices, hull_faces, reach, -outward, points_there,
                                           points_here, features, normal);

                    // And where that finds nothing while the body is right there, the direction was the
                    // wrong question. A body over a crease presents, to each slope's normal, the feature of
                    // itself that is over the *other* slope - an oblique direction picks an edge or a corner
                    // rather than a face - and that feature lies outside the triangle it is then clipped
                    // into, so the clip drops it, both slopes come back empty, and the body falls through a
                    // ridge it should be resting on. The contact there is against the crease itself and its
                    // direction is neither slope's, so it has to be searched for.
                    //
                    // Only as a fallback, since the search is what handing the direction in avoids, and only
                    // where this triangle has an active edge: what the search finds is a contact against an
                    // edge, and an inactive edge is a seam of the tessellation, where the triangle across is
                    // flat with this one and holds the body on its own face. A flat floor never reaches here.
                    const float3 bottom = PolySupportPoint(own_poly, hull_vertices, -outward);
                    const bool within = dot(bottom - first, outward) - own_poly.Radius < reach;
                    const bool searched = found == 0 && within && triangle.ActiveEdges != 0;
                    if (searched)
                        found = ConvexManifold(face, own_poly, hull_vertices, hull_faces, reach, float3(0), points_there,
                                               points_here, features, normal);
                    normal = -normal;

                    // What is left is the seams. A point one of them cut is a point the triangle across it
                    // cuts out of its own side too, so both would hold one piece of geometry with a dual
                    // each. An edge that is a feature is not a seam: the rim of an open surface or a crease
                    // genuinely cuts there, and what it cut stays. Bits 8 to 10 name which of the reference
                    // face's edges cut a point, which is only the triangle's business while the triangle is
                    // the reference (bit 28 clear) - where the search put the reference on the body those
                    // bits mean nothing here, and that case is about an active edge anyway.
                    //
                    // The tolerance is relative to where the triangle is, since the rounding in a dot
                    // product is relative to what went into it - the scale the clip measured with.
                    float scale = 1;
                    for (uint v = 0; v < 3; ++v) scale = max(scale, length(PolyVertex(face, hull_vertices, v)));
                    const float seam = 1e-5f * scale;

                    uint kept = 0;
                    for (uint i = 0; i < found; ++i) {
                        const bool triangle_led = ((features[i] >> 28) & 1) == 0;
                        if (triangle_led && (((features[i] >> 8) & 7) & ~triangle.ActiveEdges) != 0) continue;
                        // A searched point that is not on an active edge belongs to the triangle across,
                        // which holds it on its own face. Without this every triangle of a flat mesh with a
                        // rim answers for whatever is near its plane, which is all of them, and a box
                        // sliding down the middle is caught by edges nowhere near it.
                        if (searched) {
                            bool on_feature = false;
                            for (uint e = 0; e < 3 && !on_feature; ++e)
                                on_feature = (triangle.ActiveEdges & (1u << e)) != 0 &&
                                    OnEdgeLine(face, hull_vertices, outward, e, points_there[i], seam);
                            if (!on_feature) continue;
                        }
                        // A point the seam did not cut but that landed along it anyway: a corner of the
                        // body's face exactly on the line where two triangles meet. A vertex on a plane is
                        // inside it and keeps its own name, so both triangles hold it and the cook's owner
                        // settles which one answers.
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
                    // One of the two is round, which makes the whole pair a distance problem: find the point
                    // of the other shape nearest the core, and the contact is that direction with the radius
                    // taken off it. A sphere touches at one point whatever it meets. A capsule's core is a
                    // segment and can lie along what it touches, so it takes a sample per end - the whole of
                    // the manifold it ever has, since between the ends its surface is a straight ruling.
                    const bool mine_is_round = IsRound(shape.Kind);
                    const Shape round_shape = mine_is_round ? shape : target;
                    const Shape against = mine_is_round ? target : shape;
                    const Core core = MakeCore(mine_is_round ? pose : target_pose, round_shape);
                    const Pose other_pose = mine_is_round ? target_pose : pose;
                    const bool other_is_round = IsRound(against.Kind);
                    const Core other_core = other_is_round ? MakeCore(other_pose, against) : Core{};

                    // Where along each core to take a contact, and the name each sample answers to. Two ends
                    // meeting two ends is still at most two contacts: the pair either meets at a point, or
                    // lies along a common stretch whose two limits are what hold it steady.
                    float3 samples[2], others[2];
                    uint names[2];
                    uint taken = 0;
                    if (other_is_round) {
                        const float3 mine_along = core.To - core.From, theirs_along = other_core.To - other_core.From;
                        const float mine_length = length(mine_along), theirs_length = length(theirs_along);
                        const bool parallel = mine_length > 1e-6f && theirs_length > 1e-6f &&
                            abs(dot(mine_along / mine_length, theirs_along / theirs_length)) > 0.999f;
                        if (parallel) {
                            // Side by side, so they touch along the stretch both cores cover. Its two limits
                            // are each an end of one core or the other, which is the name each one takes.
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
                                    // Whichever core's end bounded this limit is what names it.
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
                        // A plane is flat everywhere, so the ends of the core are the whole of it.
                        samples[0] = core.From;
                        names[0] = 0;
                        taken = 1;
                        if (distance(core.From, core.To) > 1e-6f) {
                            samples[1] = core.To;
                            names[1] = 1;
                            taken = 2;
                        }
                    } else {
                        // Against a box, where along the core to sample is a question the core's own ends
                        // cannot answer: a capsule can rest with its middle across a box and both ends out
                        // over nothing, and sampling only the ends walks it straight through. So find where
                        // the core comes nearest the box - alternating projection, which converges from
                        // either side because both shapes are convex - and take the face that lands on. The
                        // stretch of the core over that face is what the capsule rests on, and its two
                        // limits are the manifold, each named by whichever thing bounded it.
                        const BoxPose target_box = MakeBox(other_pose, against);
                        float3 on_core = ClosestOnSegment(core.From, core.To, target_box.Center), on_box;
                        float away;
                        uint face = 0;
                        for (uint round = 0; round < 3; ++round) {
                            on_box = ClosestOnBox(target_box, on_core, away, face);
                            on_core = ClosestOnSegment(core.From, core.To, on_box);
                        }
                        on_box = ClosestOnBox(target_box, on_core, away, face);

                        const float3 out_of = away < 0 ? normalize(on_box - on_core)
                                                       : (away > 1e-9f ? (on_core - on_box) / away : float3(0, 1, 0));
                        uint axis = 0;
                        float most = 0;
                        for (uint i = 0; i < 3; ++i) {
                            const float aligned = abs(dot(out_of, target_box.Axis[i]));
                            if (aligned > most) {
                                most = aligned;
                                axis = i;
                            }
                        }

                        // Clip the core to the two slabs across that face. Both limits keep the name of what
                        // set them, so the pair is the same pair next step wherever the search happened to
                        // start from.
                        const float3 along = core.To - core.From;
                        float low = 0, high = 1;
                        uint low_name = 0, high_name = 1;
                        for (uint side = 0; side < 2; ++side) {
                            const uint slab = (axis + 1 + side) % 3;
                            const float direction = dot(along, target_box.Axis[slab]);
                            const float from = dot(core.From - target_box.Center, target_box.Axis[slab]);
                            for (uint face_side = 0; face_side < 2; ++face_side) {
                                const float edge = face_side == 0 ? target_box.Half[slab] : -target_box.Half[slab];
                                // Where the core crosses this slab face, keeping the inside of it.
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

                        // The stretch has to be a stretch of something: a sphere's core is a single point, so
                        // its two limits are the same place and taking both is one contact written twice.
                        if (high - low > 1e-5f && length(along) * (high - low) > 1e-5f) {
                            samples[0] = core.From + along * low;
                            samples[1] = core.From + along * high;
                            names[0] = low_name;
                            names[1] = high_name;
                            taken = 2;
                        } else {
                            // Nothing of the core lies over a face - a cap against an edge or a corner - so
                            // the single nearest point is the contact, named by the feature it found.
                            samples[0] = on_core;
                            names[0] = 8 + face;
                            taken = 1;
                        }
                    }

                    for (uint sample = 0; sample < taken && found < MaxFacePoints; ++sample) {
                        const float3 at = samples[sample];
                        // `out_of` points away from the other shape, and `gap` is the distance between the
                        // two surfaces along it.
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
                            out_of = span > 1e-9f ? apart / span : float3(0, 1, 0); // coincident: any direction will do
                            nearest = on_theirs + out_of * other_core.Radius;
                            gap = span - other_core.Radius - core.Radius;
                        } else {
                            float away;
                            uint face;
                            nearest = ClosestOnBox(MakeBox(other_pose, against), at, away, face);
                            const float3 apart = at - nearest;
                            // A negative distance means the core point is inside the box, and `nearest` is
                            // then the surface it came out through, so the direction is the other way round.
                            out_of = away < 0 ? normalize(nearest - at) : (away > 1e-9f ? apart / away : float3(0, 1, 0));
                            gap = away - core.Radius;
                        }

                        if (gap >= reach) continue;
                        // Convention is out of the other body towards this one, and `out_of` points away from
                        // whichever of the two is not the round one being sampled.
                        normal = mine_is_round ? out_of : -out_of;
                        const float3 on_round = at - out_of * core.Radius;
                        points_here[found] = mine_is_round ? on_round : nearest;
                        points_there[found] = mine_is_round ? nearest : on_round;
                        features[found] = names[sample];
                        ++found;
                    }
                } else if (target.Kind == ShapePlane) {
                    // A plane is flat everywhere, so nothing has to be searched for: every vertex of the
                    // polytope either reaches through it or does not, and the ones that do are the manifold.
                    normal = target.Normal;
                    // Every vertex that reaches through is a candidate, and eight of them is what one pair
                    // may report, so the same spread the recovered faces take rather than the first eight.
                    found = SpreadSupport(own_poly, hull_vertices, -normal, -(target.Offset + reach), MaxFacePoints, points_here, features);
                    for (uint i = 0; i < found; ++i)
                        points_there[i] = points_here[i] - (dot(normal, points_here[i]) - target.Offset) * normal;
                } else if (!hulled) {
                    const BoxPose other_box = MakeBox(target_pose, target);

                    // Separating axis test over all fifteen axes. Any one apart and there is no contact. The
                    // nine cross products are the edge-on-edge axes and the shallowest is kept: two boxes
                    // crossing at an angle touch along one pair of edges and no face axis describes that -
                    // taking a face there gives the wrong normal and more penetration than there really is.
                    bool apart = false;
                    uint edge_i = 0, edge_j = 0;
                    float3 edge_normal = float3(0);
                    float least_edge = INFINITY;
                    for (uint i = 0; i < 3 && !apart; ++i) {
                        for (uint j = 0; j < 3; ++j) {
                            const float3 axis = cross(box.Axis[i], other_box.Axis[j]);
                            const float len = length(axis);
                            if (len < 1e-6f) continue; // parallel edges, already covered by the face axes
                            const float3 unit = axis / len;
                            const float overlap = Overlap(box, other_box, unit);
                            // Apart by more than the reach, rather than apart at all: a projection gap is
                            // a lower bound on the distance between the two, so the early-out stays sound
                            // and everything closer than the reach goes on to be given its slack.
                            if (overlap < -reach) {
                                apart = true;
                                break;
                            }
                            if (overlap < least_edge) {
                                least_edge = overlap;
                                edge_i = i;
                                edge_j = j;
                                // Out of the other body towards this one, which is the sign the solve wants.
                                edge_normal = dot(unit, box.Center - other_box.Center) < 0 ? -unit : unit;
                            }
                        }
                    }
                    // The face axis they overlap along least is the one to separate them on. A challenger has
                    // to beat the incumbent by a margin, relative and absolute both, which is the tolerance
                    // Box2D-lite carries and both references keep: two faces of a stacked box overlap by
                    // almost the same amount, so a bare minimum flips between them on noise, and a flipped
                    // reference axis renames every point in the manifold and throws away its warm start.
                    //
                    // The least overlap is the most separated axis when the boxes are disjoint, which is
                    // the face a speculative contact between them belongs on. The relative part of the
                    // tolerance is written as a fraction of the incumbent's own size for that reason:
                    // multiplying a negative incumbent by 0.95 moves it the wrong way and turns the bias
                    // towards the incumbent into a bias against it.
                    uint best_axis = 0, best_owner = 0;
                    float least = INFINITY;
                    for (uint owner = 0; owner < 2 && !apart; ++owner) {
                        for (uint i = 0; i < 3; ++i) {
                            const float3 axis = owner == 0 ? box.Axis[i] : other_box.Axis[i];
                            const float overlap = Overlap(box, other_box, axis);
                            if (overlap < -reach) {
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
                    if (apart || least > 1e18f) continue;

                    // Both measured as separation - negative while the boxes overlap - so this reads as the
                    // reference writes it: the edge pair has to separate them by a clear margin more than the
                    // best face does before it wins, which leaves ties to the faces. A stack of axis-aligned
                    // boxes never gets here, since its cross products are all degenerate.
                    const bool on_edge = least_edge < 1e18f && RelativeTolerance * -least_edge > -least + EdgeTolerance;
                    if (on_edge) {
                        float3 mine[2], theirs[2];
                        uint which_mine, which_theirs;
                        // The normal points out of the other body, so this body's edge is the one reaching
                        // back along it and the other body's is the one reaching along it.
                        SupportEdge(box, edge_i, -edge_normal, mine, which_mine);
                        SupportEdge(other_box, edge_j, edge_normal, theirs, which_theirs);
                        normal = edge_normal;
                        ClosestOnSegments(mine[0], mine[1], theirs[0], theirs[1], points_here[0], points_there[0]);
                        // Which two edges made it, named so that the other three parallel to each are not
                        // handed its dual when the pair changes.
                        features[0] = (1u << 15) | (edge_i << 13) | (edge_j << 11) | (which_mine << 9) | (which_theirs << 7);
                        found = 1;
                    } else {
                        // Normal points out of the other body towards this one, which is the sign the solve wants.
                        const BoxPose reference = best_owner == 0 ? box : other_box;
                        const BoxPose incident = best_owner == 0 ? other_box : box;
                        float3 face_normal = reference.Axis[best_axis];
                        if (dot(face_normal, incident.Center - reference.Center) < 0) face_normal = -face_normal;
                        normal = best_owner == 0 ? -face_normal : face_normal;

                        // The incident face is whichever of the other box's faces points most against the normal.
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

                        // Four corners cut by four side planes is eight points and never more, so this
                        // path needs none of the width the hull path's clip does.
                        float3 poly[MaxFacePoints], clipped[MaxFacePoints];
                        uint names[MaxFacePoints], clipped_names[MaxFacePoints];
                        FaceCorners(incident, incident_axis, incident_side, poly);
                        for (uint i = 0; i < 4; ++i) names[i] = (1u << i) | (1u << ((i + 3) % 4)); // its two edges
                        uint poly_count = 4;
                        for (uint edge = 0; edge < 2; ++edge) {
                            const uint side_axis = (best_axis + 1 + edge) % 3;
                            for (uint s = 0; s < 2; ++s) {
                                const float3 side_normal = reference.Axis[side_axis] * (s == 0 ? 1 : -1);
                                const float side_offset = dot(side_normal, reference.Center) + reference.Half[side_axis];
                                const uint plane = 1u << (4 + edge * 2 + s);
                                // No tolerance: a box's side planes and the incident corners clipped against
                                // them are both built from the same half extents, so an incidence there is
                                // exact rather than rounded.
                                poly_count = ClipAgainst(poly, names, poly_count, side_normal, side_offset, plane, 0, MaxFacePoints, clipped, clipped_names);
                                for (uint i = 0; i < poly_count; ++i) {
                                    poly[i] = clipped[i];
                                    names[i] = clipped_names[i];
                                }
                            }
                        }

                        // Keep the clipped points that are actually against the reference face. face_normal is
                        // already the outward direction, so the face plane sits one half-extent along it.
                        const float face_offset = dot(face_normal, reference.Center) + reference.Half[best_axis];
                        for (uint i = 0; i < poly_count && found < MaxFacePoints; ++i) {
                            const float depth = dot(face_normal, poly[i]) - face_offset;
                            if (depth >= reach) continue;
                            // The clipped point lies on the incident body. Its partner is where it projects onto
                            // the reference face.
                            const float3 on_reference = poly[i] - depth * face_normal;
                            points_here[found] = best_owner == 0 ? on_reference : poly[i];
                            points_there[found] = best_owner == 0 ? poly[i] : on_reference;
                            // Which body owned the reference face, which axes made the two faces, and where the
                            // point sits on them. Nothing here is an index into the output array.
                            features[found] = (best_owner << 13) | (best_axis << 11) | (incident_axis << 9) |
                                ((incident_side > 0 ? 1u : 0u) << 8) | names[i];
                            ++found;
                        }
                    }
                } else {
                    // At least one of the two is a hull, so neither has a face list to test axes against and
                    // the separating axis test has nothing to enumerate. Support functions do instead.
                    found = ConvexManifold(own_poly, MakePoly(target_pose, target), hull_vertices, hull_faces, reach,
                                           float3(0), points_here, points_there, features, normal);
                }

                // No two rows on one piece of geometry, whatever produced them, and only then the four
                // that are worth keeping.
                found = WeldManifold(points_here, points_there, features, found, weld);
                found = ReduceManifold(points_here, points_there, features, found, normal);

                // One slot per manifold point, its feature naming where the point came from so the dual can
                // find it again next step.
                for (uint i = 0; i < found; ++i) {
                    // Where this point goes. While the run has room it takes the next slot, and once the
                    // run is full it has to earn a place: the shallowest contact held gives way, so a
                    // contact at a positive separation is the first to go. That one is speculative -
                    // nothing rests on it, and if it matters it comes back the moment it is the deeper of
                    // the two. Deciding by body order instead leaves a box in a lattice holding four
                    // contacts with a neighbour it merely touches and none with the box standing on it.
                    const float separation = dot(normal, points_here[i] - points_there[i]) + p.ContactMargin;
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
                    contact.AnchorA = Rotate(QuatConjugate(pose.Orientation), points_here[i] - pose.Position);
                    contact.AnchorB = Rotate(QuatConjugate(target_pose.Orientation), points_there[i] - target_pose.Position);
                    contact.Normal = normal;
                    contact.BodyA = body;
                    contact.BodyB = other;
                    contact.Friction = friction;
                    contact.Feature = features[i];
                    contact.SubShape = sub_shape;

                    // What the two were closing at here when the step began, which is what a bounce is
                    // measured against. Recorded whole and ungated, since the threshold and the
                    // coefficient belong to the velocity pass rather than to the row.
                    const float3 closing = (own_velocity.Linear + cross(own_velocity.Angular, points_here[i] - pose.Position)) -
                        (other_velocity.Linear + cross(other_velocity.Angular, points_there[i] - target_pose.Position));
                    contact.Approach = -dot(normal, closing); // positive while they are coming together
                    contact.BounceImpulse = 0;
                    contact.BounceDelta = 0;
                    contact.Active = true;

                    contact.Penalty = penalty_floor;
                    contact.Lambda = float3(0);
                    contact.Stick = false;
                    inherited[at] = NoIndex;
                    for (uint j = 0; j < ContactsPerBody; ++j) {
                        if (was_feature[j] == NoIndex) break; // the sentinel: nothing to inherit past it
                        if (was_feature[j] != contact.Feature || was_other[j] != other || was_sub[j] != sub_shape) continue;
                        inherited[at] = j;
                        contact.Penalty = clamp(was_penalty[j] * p.Gamma, penalty_floor, float3(p.PenaltyMax));
                        contact.Lambda = was_lambda[j];
                        // Static friction. A contact that stayed inside the cone last step keeps the anchor
                        // pair it was holding, so C0's friction rows below measure the drift since it stuck
                        // rather than starting from zero. Recomputing the anchors every step leaves friction
                        // with nothing to pull back towards - it can only resist motion added during the
                        // step, so a loaded box creeps a little further every step and never settles.
                        //
                        // Only where the feature names geometry that stays put on the body: a rolling
                        // sphere's contact sweeps across its surface, so last step's material point has
                        // turned away, and holding friction to it brakes the ball and spins it backwards.
                        //
                        // And only where the anchors it wants back do not land on a row this pair has
                        // already written. WeldManifold only sees the points this step's clip produced, and
                        // a stuck contact is held where it stuck, which may have been several steps ago -
                        // so two that stuck at different times can drift onto each other carrying all but
                        // the same Jacobian and C0, each with its own dual. The point is worth keeping, so
                        // what gives way is the inheritance.
                        //
                        // Checked against the rows already written and against the fresh points still to
                        // come, which covers every pair from one side or the other: for i before j, j is
                        // compared against i's final anchors, and i against j's fresh point, which is where
                        // j ends up whenever it does not inherit.
                        const float3 want_a = pose.Position + Rotate(pose.Orientation, was_anchor_a[j]);
                        const float3 want_b = target_pose.Position + Rotate(target_pose.Orientation, was_anchor_b[j]);
                        bool onto_another = false;
                        for (uint k = 0; k < count && !onto_another; ++k) {
                            if (k == at || !slots[k].Active || slots[k].BodyB != other || slots[k].SubShape != sub_shape) continue;
                            onto_another = distance(was_anchor_a[j], slots[k].AnchorA) <= weld &&
                                distance(was_anchor_b[j], slots[k].AnchorB) <= weld;
                        }
                        for (uint k = i + 1; k < found && !onto_another; ++k)
                            onto_another = distance(want_a, points_here[k]) <= weld && distance(want_b, points_there[k]) <= weld;
                        if (was_stick[j] && !curved && !onto_another) {
                            contact.AnchorA = was_anchor_a[j];
                            contact.AnchorB = was_anchor_b[j];
                            contact.Stick = true;
                        }
                        break;
                    }
                    // Eq. 15: separation resolved in the contact basis, plus a margin on the normal row so
                    // contacts engage just before they touch rather than just after.
                    const float3x3 basis = ContactBasis(normal);
                    const float3 gap = (pose.Position + Rotate(pose.Orientation, contact.AnchorA)) -
                        (target_pose.Position + Rotate(target_pose.Orientation, contact.AnchorB));
                    contact.C0 = float3(dot(basis[0], gap), dot(basis[1], gap), dot(basis[2], gap)) + float3(p.ContactMargin, 0, 0);
                    if (at == count) ++count;
                }
            }
        }
    }

    // The events, once the run has settled rather than as each point is written. A contact that lost
    // its place to a deeper one later in the step was never held, so a listener must not hear that it
    // arrived - nor must the slot it had claimed count as claimed, or the contact that really did end
    // last step goes unreported. Added or persisted was decided when the contact was written: a point
    // that found a feature to inherit from is the same contact as last step's.
    for (uint k = 0; k < count; ++k) {
        if (inherited[k] != NoIndex) claimed |= 1u << inherited[k];
        events[reported++] = ContactEvent{body, slots[k].BodyB, slots[k].Feature, slots[k].SubShape,
                                          uint(inherited[k] != NoIndex ? ContactPersisted : ContactAdded)};
    }
    EndUnclaimed(events, contact_event_counts, body, claimed, reported, was_feature, was_other, was_sub);
}

// How far a joint's two bodies have turned from the rotation it holds, as a world-frame rotation
// vector. Driving it to zero brings them back into line, and its derivative is the identity on each
// body's angular degrees of freedom to first order, which is what makes the rows above so plain.
static float3 AngularError(float4 a, float4 b, float4 rest) {
    // Taken back into B's frame, since that is the frame the joint's three axes are named in.
    return Rotate(QuatConjugate(b), RotationVector(QuatMul(a, QuatConjugate(QuatMul(b, rest)))));
}

// Which stop a limited axis is against, if any, and what its row then asks for. The choice is made
// from the angle the step *began* at rather than the one the sweep has reached, so a row cannot change
// which side of the joint it is arguing for halfway through a solve - the same discipline that fixes a
// contact's feature at the start of the step.
//
// Inside the range there is nothing to do. Outside it the row is one-sided exactly as a contact's
// normal row is: at the high stop it may only turn the axis back down, at the low stop only back up.
static bool LimitedRow(float began, float low, float high, float held, float alpha, thread float &c, thread float &sign) {
    if (began > high) {
        sign = 1; // may only push back down
        c = held - high * (1 - alpha);
        return true;
    }
    if (began < low) {
        sign = -1;
        c = held - low * (1 - alpha);
        return true;
    }
    return false;
}

// What a row about a joint axis may apply. A motor has only so much to give whether it is chasing a
// speed or an angle, a stop may push only the way that puts the axis back inside its range, and a
// locked axis is a hard constraint with nothing bounding it either way. The primal and the dual both
// take their bounds from here, because Eq. 16 ramps a row only while it is strictly inside them and a
// row the two passes disagree about would be clamped in one and free in the other.
static void AngularBounds(uint mode, float max_torque, float sign, thread float &low, thread float &high) {
    low = -INFINITY;
    high = INFINITY;
    if (mode == AxisDriven || mode == AxisPositioned) {
        low = -max_torque;
        high = max_torque;
    } else if (mode == AxisLimited) {
        if (sign > 0) low = 0; // at the low stop, and may only turn back up
        else high = 0;
    }
}

// What a driven axis asks for: the two bodies' relative turn over this step, along that axis, against
// the turn its speed would make in a step. Expressed as an angle rather than a rate because that is
// what the solve moves, which is what the reference's Motor does too.
static float DrivenError(float3 turned, float3 axis, float speed, float dt) {
    return dot(turned, axis) - speed * dt;
}

// And what a positioned one asks for: how far the axis is from the angle it is being turned to, from
// the same zero the limits are measured against. Against the whole error and not the part added this
// step, because a motor drives the axis to where it is told rather than resisting drift away from
// where it was - which is the same reason a driven axis reads a raw turn.
static float PositionedError(float error, float target) { return error - target; }

// What one angular row of a joint asks for this iteration, the world axis it acts about, and the
// bounds on what it may apply doing so. The primal and the dual both come through here, since a row
// the two of them disagreed about would be clamped in one and free in the other.
//
// False where the row is holding nothing, which is a limited axis inside its range - the one case a
// caller has to answer for itself. A free axis never reaches here.
static bool AngularRow(
    Joint joint, uint r, float3 error, float3 turned, float4 b_orientation, float dt,
    thread float3 &axis, thread float &c, thread float &low, thread float &high
) {
    axis = Rotate(b_orientation, UnitAxis(r));
    // A spring answers for the extension it actually has, so no part of its error is held back for a
    // later step the way a hard row's is. That is the whole of what alpha does here.
    const float alpha = IsHard(joint.AngularStiffness[r]) ? ConstraintAlpha : 0;
    const float held = error[r] - joint.C0Angular[r] * alpha;
    const uint mode = AxisMode(joint.AxisModes, r);
    c = held;
    float sign = 0;
    if (mode == AxisDriven) c = DrivenError(turned, axis, joint.MotorSpeed[r], dt);
    else if (mode == AxisPositioned) c = PositionedError(error[r], joint.MotorTarget[r]);
    else if (mode == AxisLimited && !LimitedRow(joint.C0Angular[r], joint.LimitLow[r], joint.LimitHigh[r], held, alpha, c, sign)) return false;
    AngularBounds(mode, joint.MotorMaxTorque[r], sign, low, high);
    return true;
}

// Joints are re-measured every iteration, so all the start of a step has to record is the error that
// alpha spreads the correction of, and the penalty decay that warm starts them. The dual carries over
// in full, as a contact's does, since post-stabilization means no correction energy went into it.
kernel void PrepareJoints(
    device Joint *joints [[buffer(16)]], device const Pose *poses [[buffer(0)]],
    constant StepParams &p [[buffer(7)]], uint index [[thread_position_in_grid]]
) {
    if (index >= p.JointCount) return;
    device Joint &joint = joints[index];
    if (!joint.Active) return;
    const Pose a = poses[joint.BodyA], b = poses[joint.BodyB];
    joint.C0Linear = (a.Position + Rotate(a.Orientation, joint.AnchorA)) -
        (b.Position + Rotate(b.Orientation, joint.AnchorB));
    joint.C0Angular = AngularError(a.Orientation, b.Orientation, joint.RestRotation);
    // Eq. 19's decay, then Eq. 16's cap: a soft row's penalty ramps up to the material stiffness and no
    // further, which is what keeps it a spring of that stiffness rather than a hard constraint that
    // took a while to arrive. Infinite stiffness makes the min a no-op, so a hard row is untouched.
    joint.PenaltyLinear = min(clamp(joint.PenaltyLinear * p.Gamma, p.PenaltyMin, p.PenaltyMax), joint.LinearStiffness);
    joint.PenaltyAngular = min(clamp(joint.PenaltyAngular * p.Gamma, p.PenaltyMin, p.PenaltyMax), joint.AngularStiffness);
    // And it carries no dual, so nothing stale is left where one would have been.
    for (uint r = 0; r < 3; ++r) {
        if (!IsHard(joint.LinearStiffness[r])) joint.LambdaLinear[r] = 0;
        if (!IsHard(joint.AngularStiffness[r])) joint.LambdaAngular[r] = 0;
    }
}

// Eqs. 11 and 16 again, for joints. Every row is hard and unbounded, so there is nothing to clamp and
// the penalty ramps on every one of them.
kernel void UpdateJointDuals(
    device Joint *joints [[buffer(16)]], device const Pose *poses [[buffer(0)]],
    device const Pose *initial [[buffer(1)]], constant StepParams &p [[buffer(7)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= p.JointCount) return;
    device Joint &joint = joints[index];
    if (!joint.Active) return;
    const Pose a = poses[joint.BodyA], b = poses[joint.BodyB];
    const float3 reach = (a.Position + Rotate(a.Orientation, joint.AnchorA)) -
        (b.Position + Rotate(b.Orientation, joint.AnchorB));
    for (uint r = 0; r < 3; ++r) {
        const bool hard = IsHard(joint.LinearStiffness[r]);
        const float linear = reach[r] - joint.C0Linear[r] * (hard ? ConstraintAlpha : 0);
        if (hard) {
            joint.LambdaLinear[r] = joint.PenaltyLinear[r] * linear + joint.LambdaLinear[r];
            joint.PenaltyLinear[r] = min(joint.PenaltyLinear[r] + p.Beta * abs(linear), p.PenaltyMax);
        } else {
            joint.PenaltyLinear[r] = min(joint.PenaltyLinear[r] + p.Beta * abs(linear), joint.LinearStiffness[r]);
        }
    }

    const float3 error = AngularError(a.Orientation, b.Orientation, joint.RestRotation);
    const float3 turned = RotationVector(QuatMul(a.Orientation, QuatConjugate(initial[joint.BodyA].Orientation))) -
        RotationVector(QuatMul(b.Orientation, QuatConjugate(initial[joint.BodyB].Orientation)));
    const Joint state = joint; // the rows below write only the dual and the penalty, which none of them reads
    for (uint r = 0; r < 3; ++r) {
        if (AxisMode(joint.AxisModes, r) == AxisFree) continue;
        const bool hard = IsHard(joint.AngularStiffness[r]);
        float3 axis;
        float c, low, high;
        if (!AngularRow(state, r, error, turned, b.Orientation, p.DeltaTime, axis, c, low, high)) {
            // Off its stops, so it is holding nothing and carries nothing over.
            joint.LambdaAngular[r] = 0;
            continue;
        }
        if (!hard) {
            // Eq. 16's other branch and Algorithm 1 line 33: no dual, and the ramp stops at the
            // material stiffness. Ramping up to it rather than starting there is the point - it is what
            // keeps the stiffness ratio on a body small in the early iterations, per Sec. 3.4.
            joint.PenaltyAngular[r] = min(joint.PenaltyAngular[r] + p.Beta * abs(c), joint.AngularStiffness[r]);
            continue;
        }
        const float requested = joint.PenaltyAngular[r] * c + joint.LambdaAngular[r];
        joint.LambdaAngular[r] = clamp(requested, low, high);
        // Eq. 16 ramps a row only while it is strictly inside its bounds, tested against what the row
        // asked for rather than what it was allowed, since a clamped row sits exactly on its bound.
        if (requested > low && requested < high)
            joint.PenaltyAngular[r] = min(joint.PenaltyAngular[r] + p.Beta * abs(c), p.PenaltyMax);
    }
}

// Counts how many contacts name each body as B. One thread per slot, and integer atomics, so the
// counts come out the same however the threads interleave.
kernel void CountIncoming(
    device Adjacency *incoming [[buffer(17)]], device const Contact *contacts [[buffer(5)]],
    constant StepParams &p [[buffer(7)]], uint slot [[thread_position_in_grid]]
) {
    if (slot >= p.BodyCount * ContactsPerBody) return;
    const Contact contact = contacts[slot];
    if (!contact.Active) return;
    device atomic_uint *count = (device atomic_uint *)&incoming[contact.BodyB].Count;
    atomic_fetch_add_explicit(count, 1u, memory_order_relaxed);
}

// Turns those counts into where each body's run starts. One thread walking the bodies in order, which
// is a serial scan once a step in place of a pass over the whole pool per body per colour per iteration.
kernel void ScanIncoming(
    device Adjacency *incoming [[buffer(17)]], constant StepParams &p [[buffer(7)]],
    uint id [[thread_position_in_grid]]
) {
    if (id != 0) return;
    uint at = 0;
    for (uint body = 0; body < p.BodyCount; ++body) {
        incoming[body].Start = at;
        incoming[body].Cursor = at;
        at += incoming[body].Count;
    }
}

kernel void FillIncoming(
    device Adjacency *incoming [[buffer(17)]], device uint *slots [[buffer(18)]],
    device const Contact *contacts [[buffer(5)]], constant StepParams &p [[buffer(7)]],
    uint slot [[thread_position_in_grid]]
) {
    if (slot >= p.BodyCount * ContactsPerBody) return;
    const Contact contact = contacts[slot];
    if (!contact.Active) return;
    device atomic_uint *cursor = (device atomic_uint *)&incoming[contact.BodyB].Cursor;
    slots[atomic_fetch_add_explicit(cursor, 1u, memory_order_relaxed)] = slot;
}

// The scatter above lands in whatever order the threads got there, so each body's run is sorted back
// into slot order. Determinism is not a nicety here - it is what makes a replay a replay.
kernel void SortIncoming(
    device const Adjacency *incoming [[buffer(17)]], device uint *slots [[buffer(18)]],
    constant StepParams &p [[buffer(7)]], uint body [[thread_position_in_grid]]
) {
    if (body >= p.BodyCount) return;
    const uint start = incoming[body].Start, count = incoming[body].Count;
    for (uint i = 1; i < count; ++i) { // insertion sort, over a run a body's neighbours keep short
        const uint value = slots[start + i];
        uint j = i;
        for (; j > 0 && slots[start + j - 1] > value; --j) slots[start + j] = slots[start + j - 1];
        slots[start + j] = value;
    }
}

// What the normal row measures its separation from, before this step's motion is added to it.
//
// Through the main iterations a contact resists only what the step adds, which is what alpha is for -
// except that a contact generated while the bodies were still apart has a gap to give before it resists
// anything. That slack is the *separation* alone: C0's normal row is the separation plus ContactMargin,
// and the margin is not slack but the depth a contact comes to rest at, which the stabilization pass
// answers for. Carrying the margin here too would step by a whole margin as a contact crossed from
// apart to touching - a discontinuity a settled stack feels as chatter of that size, which at 1/60 is
// three centimetres a second and enough to keep a twelve-box stack from ever sleeping. Written as a
// positive part rather than a branch so it is continuous at touch, where the two readings agree at zero.
//
// The stabilization pass keeps the whole of C0 for every contact, apart or touching alike.
//
// Everything else about a separated row already behaves: the force clamp gives it no force while its
// gap is unspent, Eq. 14's secant then gives it no stiffness either so nothing damps free flight, the
// ramp gate skips a row applying nothing, and Stick cannot set while the cone bound is zero.
static float NormalOffset(Contact contact, constant StepParams &p) {
#if STABILIZE
    return contact.C0[0];
#else
    return max(contact.C0[0] - p.ContactMargin, 0.f);
#endif
}

// What a contact row asked for, clamped to what a contact can actually do: row 0 can only push, and
// rows 1 and 2 together cannot exceed the friction cone. Takes the requested force rather than the
// terms it is made of, since both callers need that unclamped force for Eq. 16's ramp gate as well.
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

// Walking a body's contacts: its own run, which is every contact where it is A, then the list gathered
// this step, which is every contact where it is B. Between them that is all of them, without a look at
// anybody else's slots, and neither is longer than the bodies this one touches.
//
// NoIndex where there is nothing at `i`. The own run is dense from zero, so its first inactive slot
// ends it, and `i` is advanced past the rest of the run rather than loading it to learn the same thing.
static uint ContactSlot(thread uint &i, uint own, uint start, device const uint *incoming_slots, device const Contact *contacts) {
    const uint slot = i < ContactsPerBody ? own + i : incoming_slots[start + i - ContactsPerBody];
    if (contacts[slot].Active) return slot;
    if (i < ContactsPerBody) i = ContactsPerBody - 1;
    return NoIndex;
}

// One neighbour of a body being coloured, counted in the first sweep and read for its colour in the
// second - degree has to be whole before any neighbour's priority can be judged against it. Two masks:
// conflicts are judged against the prioritized neighbours, the way the paper's scheme always has, and a
// compacting body moves only to a colour no neighbour holds at all, since stealing one a higher-indexed
// neighbour is using would just push the conflict onto it.
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

// Algorithm 1 line 2, "update colorization". Bodies of the same colour touch nothing in common, so a
// colour can be solved in parallel while the colours in sequence give Gauss-Seidel propagation - which
// is what a stack needs, since load has to travel down it.
//
// The paper's scheme: incremental, so it starts from last step's answer and needs only a few passes,
// and allowed to come out imperfect, since two neighbours sharing a colour fall back to Jacobi for that
// step, which the pose snapshot underneath already provides.
//
// Two amendments, measured on RbpScenes raft, both gated on the pair having gone quiet - see
// Prioritized for why the gate is what keeps a collapsing pile off Jacobi. Priority goes by contact
// degree before index, Welsh-Powell's ordering, because index order is add order and a scene built from
// the ground up hands the busiest bodies whatever their lower-indexed neighbours left over: the raft
// settled at eight colours where the same graph six-colours under degree order. And a quiet body also
// moves *down* to the lowest colour no neighbour holds, because without that no settled colouring ever
// improves - a proper colouring has no conflicts, and conflict was the only trigger. Two quiet
// neighbours can adopt one free colour at once, which is a conflict the next pass resolves by priority.
kernel void UpdateColors(
    device const uint *colors [[buffer(12)]], device uint *next [[buffer(13)]],
    device const Contact *contacts [[buffer(5)]], device const BodyMass *masses [[buffer(4)]],
    device const Joint *joints [[buffer(16)]], device const Adjacency *incoming [[buffer(17)]],
    device const uint *incoming_slots [[buffer(18)]], device const uint *quiet [[buffer(21)]],
    constant StepParams &p [[buffer(7)]], uint body [[thread_position_in_grid]]
) {
    if (body >= p.BodyCount) return;
    const uint mine = ColorOf(colors[body]);
    next[body] = colors[body];
    if (masses[body].InvMass == 0) return;
    // Asleep keeps its word as-is: the sweeps skip it, so its colour constrains nothing it does, and an
    // awake neighbour at the island's edge reads the frozen word exactly as it would a live one. What
    // this saves is the gather, which a resting world was paying for bodies whose colours cannot change.
    if (Asleep(quiet[body], p)) return;
    const bool my_quiet = quiet[body] > 0;

    // Two sweeps over everything this body touches, since degree has to be whole before any
    // neighbour's priority can be judged against it. See NoteNeighbour for the two masks.
    uint degree = 0, taken = 0, taken_all = 0;
    const uint own = body * ContactsPerBody, incoming_count = incoming[body].Count, incoming_start = incoming[body].Start;
    for (uint sweep = 0; sweep < 2; ++sweep) {
        for (uint i = 0; i < ContactsPerBody + incoming_count; ++i) {
            const uint slot = ContactSlot(i, own, incoming_start, incoming_slots, contacts);
            if (slot == NoIndex) continue;
            const Contact contact = contacts[slot];
            const Index other = contact.BodyA == body ? contact.BodyB : contact.BodyA;
            if (masses[other].InvMass == 0) continue;
            NoteNeighbour(sweep, other, colors[other], body, my_quiet && quiet[other] > 0, degree, taken, taken_all);
        }
        // A joint couples two bodies exactly as a contact does, so it constrains the colouring the
        // same way. Colouring from contacts alone lets a jointed pair share a colour and race in place.
        for (uint index = 0; index < p.JointCount; ++index) {
            const Joint joint = joints[index];
            if (!joint.Active || (joint.BodyA != body && joint.BodyB != body)) continue;
            const Index other = joint.BodyA == body ? joint.BodyB : joint.BodyA;
            if (masses[other].InvMass == 0) continue;
            NoteNeighbour(sweep, other, colors[other], body, my_quiet && quiet[other] > 0, degree, taken, taken_all);
        }
        degree = min(degree, MaxColorDegree);
    }

    uint chosen = mine;
    if ((taken & (1u << min(mine, 31u))) != 0) {
        // Conflicted: off to the lowest colour no prioritized neighbour holds, and where the cap
        // leaves nothing, stay - the pair aliases onto a colour being solved and is Jacobi for the
        // step, which the snapshot allows.
        uint best = 0;
        while (best < 31 && (taken & (1u << best)) != 0) ++best;
        if (best < p.MaxColors) chosen = best;
    } else if (my_quiet) {
        // Quiet and unconflicted: compact downwards where a colour sits entirely free, which is what
        // lets a settled colouring improve at all - a proper colouring has no conflicts, and conflict
        // used to be the only reason a body ever moved.
        uint best = 0;
        while (best < 31 && (taken_all & (1u << best)) != 0) ++best;
        if (best < mine) chosen = best;
    }
    next[body] = (degree << ColorDegreeShift) | chosen;
}

kernel void PublishColors(
    device uint *colors [[buffer(12)]], device const uint *next [[buffer(13)]],
    constant StepParams &p [[buffer(7)]], uint body [[thread_position_in_grid]]
) {
    if (body >= p.BodyCount) return;
    colors[body] = next[body];
}

// The primal update, Eqs. 4 to 6 and 13: one body per thread, gathering its own contacts, building
// its 6x6 block and moving. Nothing scatters, so there is no atomic accumulation anywhere.
kernel void SolveBodies(
    device const Pose *poses [[buffer(0)]], device Pose *solved [[buffer(11)]], device const Pose *initial [[buffer(1)]],
    device const Pose *inertial [[buffer(2)]], device const BodyMass *masses [[buffer(4)]],
    device const Contact *contacts [[buffer(5)]], device const uint *colors [[buffer(12)]],
    device const uint *cursor [[buffer(14)]], device const Joint *joints [[buffer(16)]],
    device const Adjacency *incoming [[buffer(17)]], device const uint *incoming_slots [[buffer(18)]],
    device const uint *quiet [[buffer(21)]], constant StepParams &p [[buffer(7)]],
    uint body [[thread_position_in_grid]]
) {
    if (body >= p.BodyCount) return;
    const BodyMass mass = masses[body];
    // The colour this pass is for is the slot the cursor binding was pointed at, so nothing has to be
    // dispatched between colours to advance it. Taken modulo the count in case a body still holds a
    // colour from a step that allowed more of them, which at worst makes that pair Jacobi for the step.
    if (mass.InvMass == 0 || Asleep(quiet[body], p) || ColorOf(colors[body]) % p.MaxColors != cursor[0]) return;

    Pose pose = poses[body];
    const Pose start = initial[body], target = inertial[body];
    const float inv_dt2 = 1 / (p.DeltaTime * p.DeltaTime);

    float H[Dof][Dof];
    float g[Dof];
    for (uint i = 0; i < Dof; ++i) {
        g[i] = 0;
        for (uint j = 0; j < Dof; ++j) H[i][j] = 0;
    }

    // Eqs. 5 and 6: M / h^2 on the block diagonal, and a gradient pulling back towards free flight.
    const float m = 1 / mass.InvMass;
    const float3 offset = pose.Position - target.Position;
    for (uint i = 0; i < 3; ++i) {
        H[i][i] = m * inv_dt2;
        g[i] = m * inv_dt2 * offset[i];
    }
    const float3x3 rotation = QuatToMatrix(pose.Orientation);
    const float3x3 world_inertia = rotation * Diagonal(1 / mass.InvInertiaLocal) * transpose(rotation);
    const float3 twist = RotationVector(QuatMul(pose.Orientation, QuatConjugate(target.Orientation)));
    const float3 torque = world_inertia * twist * inv_dt2;
    for (uint i = 0; i < 3; ++i) {
        g[3 + i] = torque[i];
        for (uint j = 0; j < 3; ++j) H[3 + i][3 + j] = world_inertia[j][i] * inv_dt2;
    }

    // Everything this body is in, which neither depends on how many bodies there are - see ContactSlot.
    const uint own = body * ContactsPerBody, incoming_count = incoming[body].Count, incoming_start = incoming[body].Start;
    for (uint i = 0; i < ContactsPerBody + incoming_count; ++i) {
        const uint slot = ContactSlot(i, own, incoming_start, incoming_slots, contacts);
        if (slot == NoIndex) continue;
        const Contact contact = contacts[slot];
        const bool mine_is_a = contact.BodyA == body;

        // Jacobians are taken at the pose the step began from and held fixed across the sweeps, which
        // is what makes the constraint below a Taylor series rather than a moving target. C is always
        // measured A minus B, whichever side is being solved.
        const Pose start_a = mine_is_a ? start : initial[contact.BodyA];
        const Pose start_b = mine_is_a ? initial[contact.BodyB] : start;
        const float3 arm_a = Rotate(start_a.Orientation, contact.AnchorA);
        const float3 arm_b = Rotate(start_b.Orientation, contact.AnchorB);
        const Displacement moved_a = Since(mine_is_a ? pose : poses[contact.BodyA], start_a);
        const Displacement moved_b = Since(mine_is_a ? poses[contact.BodyB] : pose, start_b);
        const float3 arm = mine_is_a ? arm_a : arm_b;
        const float side = mine_is_a ? 1 : -1;
        const float3x3 basis = ContactBasis(contact.Normal);
        const float offset = NormalOffset(contact, p);

        float jacobian[3][Dof];
        float c[3];
        for (uint r = 0; r < 3; ++r) {
            const float3 axis = basis[r], angular = cross(arm, axis);
            for (uint k = 0; k < 3; ++k) {
                jacobian[r][k] = side * axis[k];
                jacobian[r][3 + k] = side * angular[k];
            }
            c[r] = (r == 0 ? offset : contact.C0[r] * (1 - ConstraintAlpha)) +
                dot(axis, moved_a.Linear) + dot(cross(arm_a, axis), moved_a.Angular) -
                dot(axis, moved_b.Linear) - dot(cross(arm_b, axis), moved_b.Angular);
        }

        const float3 constraint = float3(c[0], c[1], c[2]);
        const float3 requested = contact.Penalty * constraint + contact.Lambda;
        const float3 force = ContactForce(requested, contact.Friction);

        // Eq. 14, the paper's stiffness rescaling, which neither reference implements. A row the bounds
        // clamped applies less force than its penalty asks for, so the raw penalty tells the block the
        // row is stiffer than the force it delivers and damps the step for nothing. The secant instead:
        // the stiffness that would have produced the clamped force at this C, held in [0, penalty] to
        // keep H definite for an LDL without pivoting. It is worth little at ten iterations and a great
        // deal below that - at two a five-box stack holds up with it and falls through itself without.
        float3 stiffness = contact.Penalty;
        for (uint r = 0; r < 3; ++r) {
            if (requested[r] == force[r] || abs(constraint[r]) < 1e-9f) continue;
            stiffness[r] = clamp((force[r] - contact.Lambda[r]) / constraint[r], 0.f, contact.Penalty[r]);
        }

        for (uint r = 0; r < 3; ++r) {
            for (uint row = 0; row < Dof; ++row) {
                g[row] += jacobian[r][row] * force[r];
                // Eq. 17. The second-order term is dropped, as the reference drops it for contacts.
                for (uint col = 0; col < Dof; ++col) H[row][col] += stiffness[r] * jacobian[r][row] * jacobian[r][col];
            }
        }
    }

    // Joints, which unlike contacts are measured at the pose the sweep has reached rather than
    // expanded about the one it started from, and which keep the second-order term contacts drop.
    for (uint index = 0; index < p.JointCount; ++index) {
        const Joint joint = joints[index];
        if (!joint.Active || (joint.BodyA != body && joint.BodyB != body)) continue;
        const bool mine_is_a = joint.BodyA == body;
        const Pose other = poses[mine_is_a ? joint.BodyB : joint.BodyA];
        const Pose a = mine_is_a ? pose : other, b = mine_is_a ? other : pose;
        const float side = mine_is_a ? 1 : -1;
        const float3 arm = Rotate(pose.Orientation, mine_is_a ? joint.AnchorA : joint.AnchorB);

        // Three rows holding the anchors together, one per world axis. A soft row measures the reach it
        // actually has and answers with Eq. 7 on it, with no dual and no part of the error deferred.
        const float3 reach = (a.Position + Rotate(a.Orientation, joint.AnchorA)) -
            (b.Position + Rotate(b.Orientation, joint.AnchorB));
        float3 linear, force;
        for (uint r = 0; r < 3; ++r) {
            const bool hard = IsHard(joint.LinearStiffness[r]);
            linear[r] = reach[r] - joint.C0Linear[r] * (hard ? ConstraintAlpha : 0);
            force[r] = joint.PenaltyLinear[r] * linear[r] + (hard ? joint.LambdaLinear[r] : 0);
        }
        for (uint r = 0; r < 3; ++r) {
            const float3 axis = UnitAxis(r);
            const float3 angular_part = cross(arm, axis);
            float row[Dof];
            for (uint k = 0; k < 3; ++k) {
                row[k] = side * axis[k];
                row[3 + k] = side * angular_part[k];
            }
            for (uint i = 0; i < Dof; ++i) {
                g[i] += row[i] * force[r];
                for (uint j = 0; j < Dof; ++j) H[i][j] += joint.PenaltyLinear[r] * row[i] * row[j];
            }
        }

        // Sec. 3.5's geometric stiffness, which the reference keeps for joints and drops for contacts:
        // the arm turns as the body does, so the same force asks for a different torque, and that
        // second derivative is real here. Lumped onto the diagonal by column length, as the paper says.
        float3x3 geometric = float3x3(0);
        for (uint r = 0; r < 3; ++r) {
            const float3 unit = UnitAxis(r);
            geometric += (float3x3(-arm[r]) + float3x3(arm * unit.x, arm * unit.y, arm * unit.z)) * force[r];
        }
        for (uint i = 0; i < 3; ++i) H[3 + i][3 + i] += length(geometric[i]);

        // And up to three more about body B's own axes, each either holding the rotation the joint
        // found or turning it towards a speed or an angle. Their Jacobian is that axis on the angular
        // block and nothing on the linear one, so no geometric stiffness comes with them.
        const float3 error = AngularError(a.Orientation, b.Orientation, joint.RestRotation);
        const float3 turned = RotationVector(QuatMul(a.Orientation, QuatConjugate(initial[joint.BodyA].Orientation))) -
            RotationVector(QuatMul(b.Orientation, QuatConjugate(initial[joint.BodyB].Orientation)));
        for (uint r = 0; r < 3; ++r) {
            if (AxisMode(joint.AxisModes, r) == AxisFree) continue;
            const bool hard = IsHard(joint.AngularStiffness[r]);
            float3 axis;
            float c, low, high;
            if (!AngularRow(joint, r, error, turned, b.Orientation, p.DeltaTime, axis, c, low, high)) continue;
            const float requested = joint.PenaltyAngular[r] * c + (hard ? joint.LambdaAngular[r] : 0);
            const float torque = clamp(requested, low, high);
            // Eq. 14 again, as the contact rows above. A motor asked for an angle it is nowhere near
            // shows why it matters: the first step ramps the penalty on a large c, and from then on
            // the row is saturated and never comes back down, so a bounded motor reaches half the rate
            // its torque and inertia say it should.
            float stiffness = joint.PenaltyAngular[r];
            if (requested != torque && abs(c) > 1e-9f)
                stiffness = clamp((torque - (hard ? joint.LambdaAngular[r] : 0)) / c, 0.f, joint.PenaltyAngular[r]);
            for (uint i = 0; i < 3; ++i) {
                g[3 + i] += side * axis[i] * torque;
                for (uint j = 0; j < 3; ++j) H[3 + i][3 + j] += stiffness * axis[i] * axis[j];
            }
        }
    }

    float step[Dof];
    SolveBlock(H, g, step);
    const float3 linear = float3(step[0], step[1], step[2]), angular = float3(step[3], step[4], step[5]);
    if (!isfinite(linear.x + linear.y + linear.z + angular.x + angular.y + angular.z)) return;
    pose.Position += linear;
    pose.Orientation = normalize(QuatMul(QuatFromRotationVector(angular), pose.Orientation));
    solved[body] = pose;
}

// Publishes a sweep's results. Two bodies sharing a contact each read the other's pose, so a sweep
// reads one buffer and writes another and this makes the swap: every body sees the same snapshot,
// which is what stops the answer depending on which thread ran first. It is also what lets two
// same-coloured neighbours fall back to Jacobi rather than race.
kernel void PublishPoses(
    device Pose *poses [[buffer(0)]], device const Pose *solved [[buffer(11)]],
    device const BodyMass *masses [[buffer(4)]], device const uint *colors [[buffer(12)]],
    device const uint *cursor [[buffer(14)]], constant StepParams &p [[buffer(7)]],
    uint body [[thread_position_in_grid]]
) {
    if (body >= p.BodyCount || masses[body].InvMass == 0) return;
    if (ColorOf(colors[body]) % p.MaxColors != cursor[0]) return;
    poses[body] = solved[body];
}

// Eqs. 11 and 16: the dual absorbs whatever violation the sweep left, and the penalty ramps in
// proportion to that violation. A row already at its limit - a contact separating, or friction
// saturated at the cone - does not ramp, since a stiffer penalty would not buy anything there.
kernel void UpdateDuals(
    device Contact *contacts [[buffer(5)]], device const Pose *poses [[buffer(0)]],
    device const Pose *initial [[buffer(1)]], constant StepParams &p [[buffer(7)]],
    uint slot [[thread_position_in_grid]]
) {
    device Contact &contact = contacts[slot];
    if (slot >= p.BodyCount * ContactsPerBody || !contact.Active) return;

    const Index body = contact.BodyA, other_body = contact.BodyB;
    const Pose start = initial[body], other_start = initial[other_body];
    const Displacement own = Since(poses[body], start), other = Since(poses[other_body], other_start);
    const float3 arm = Rotate(start.Orientation, contact.AnchorA);
    const float3 other_arm = Rotate(other_start.Orientation, contact.AnchorB);
    const float3x3 basis = ContactBasis(contact.Normal);
    const float offset = NormalOffset(contact, p); // identically, as in the primal, or Eq. 16 ramps
                                                   // the penalty against a constraint nothing is solving

    float3 c;
    for (uint r = 0; r < 3; ++r) {
        const float3 axis = basis[r];
        c[r] = (r == 0 ? offset : contact.C0[r] * (1 - ConstraintAlpha)) +
            dot(axis, own.Linear) + dot(cross(arm, axis), own.Angular) -
            dot(axis, other.Linear) - dot(cross(other_arm, axis), other.Angular);
    }

    // What the rows asked for before the cone clamped them. Eq. 16 ramps a row only while it is
    // strictly inside its bounds, and the clamp puts a sliding contact exactly on the cone, so asking
    // the clamped force says yes forever: the row ramps until its penalty dwarfs the inertial term and
    // the damped primal step can no longer reach the friction impulse the dual is reporting in full.
    const float3 requested = contact.Penalty * c + contact.Lambda;
    const float3 force = ContactForce(requested, contact.Friction);
    contact.Lambda = force;
    if (force[0] < 0) contact.Penalty[0] = min(contact.Penalty[0] + p.Beta * abs(c[0]), p.PenaltyMax);
    const float bound = abs(force[0]) * contact.Friction;
    if (length(float2(requested[1], requested[2])) <= bound) {
        contact.Penalty[1] = min(contact.Penalty[1] + p.Beta * abs(c[1]), p.PenaltyMax);
        contact.Penalty[2] = min(contact.Penalty[2] + p.Beta * abs(c[2]), p.PenaltyMax);
        contact.Stick = length(float2(c[1], c[2])) < p.ContactMargin;
    }
}

// Velocity is read back out of the motion the sweeps produced, rather than integrated alongside it.
// It is taken before the stabilization pass, so removing accumulated penetration moves bodies without
// handing them the energy that motion would otherwise imply.
kernel void Finalize(
    device const Pose *poses [[buffer(0)]], device const Pose *initial [[buffer(1)]],
    device Velocity *velocities [[buffer(3)]], device const BodyMass *masses [[buffer(4)]],
    device const uint *quiet [[buffer(21)]], constant StepParams &p [[buffer(7)]],
    uint body [[thread_position_in_grid]]
) {
    if (body >= p.BodyCount || masses[body].InvMass == 0 || Asleep(quiet[body], p)) return;
    const Displacement moved = Since(poses[body], initial[body]);
    const float inv_dt = 1 / p.DeltaTime;
    velocities[body] = {moved.Linear * inv_dt, moved.Angular * inv_dt};
}

// The inverse inertia in world, which the two passes below need and the primal sweep builds inverted.
static float3x3 InverseInertia(float4 orientation, float3 inverse_local) {
    const float3x3 rotation = QuatToMatrix(orientation);
    return rotation * Diagonal(inverse_local) * transpose(rotation);
}

// Restitution, as a velocity pass after the solve rather than as a distance the normal row asks for.
//
// One displacement per step is all AVBD produces and velocity is read off it, so that displacement
// cannot carry both an approach and a rebound - its average is neither. As soon as a contact can be
// built across a gap, a row asking to end the step at e v h and a row asking not to penetrate are one
// row asking for two things. Box2D v3 and the XPBD rigid-body paper both answer with a pass after the
// solve.
//
// XPBD Eq. 34 wearing Box2D's gates: nothing bounces unless the solve's own normal dual actually
// pushed, which keeps a speculative contact that never arrived from bouncing anything, and unless the
// approach beat the threshold, which lets a settling body settle. Contact-parallel, so what it computes
// is an impulse and the gather below moves the bodies - no atomics, and symmetric over a manifold's
// points where a sequential pass admits it is approximate.
kernel void Restitution(
    device Contact *contacts [[buffer(5)]], device const Pose *poses [[buffer(0)]],
    device const Velocity *velocities [[buffer(3)]], device const BodyMass *masses [[buffer(4)]],
    device const float *restitutions [[buffer(15)]], constant StepParams &p [[buffer(7)]],
    uint slot [[thread_position_in_grid]]
) {
    if (slot >= p.BodyCount * ContactsPerBody) return;
    device Contact &contact = contacts[slot];
    contact.BounceDelta = 0;
    if (!contact.Active || contact.Lambda[0] >= 0 || contact.Approach <= p.MinBounceSpeed) return;
    // The larger of the two, so a bouncy body stays bouncy whatever it lands on.
    const Index a = contact.BodyA, b = contact.BodyB;
    const float restitution = max(restitutions[a], restitutions[b]);
    if (restitution <= 0) return;

    const BodyMass mass_a = masses[a], mass_b = masses[b];
    const float3 arm_a = Rotate(poses[a].Orientation, contact.AnchorA);
    const float3 arm_b = Rotate(poses[b].Orientation, contact.AnchorB);
    const float3 normal = contact.Normal;
    // Eqs. 2 and 3: what a unit impulse along the normal at these two anchors does to the normal speed
    // between them, which is the effective mass the pass divides by.
    const float3 turn_a = InverseInertia(poses[a].Orientation, mass_a.InvInertiaLocal) * cross(arm_a, normal);
    const float3 turn_b = InverseInertia(poses[b].Orientation, mass_b.InvInertiaLocal) * cross(arm_b, normal);
    const float weight = mass_a.InvMass + mass_b.InvMass +
        dot(normal, cross(turn_a, arm_a)) + dot(normal, cross(turn_b, arm_b));
    if (weight <= 0) return; // nothing here can move, which a pair of static bodies is

    const Velocity va = velocities[a], vb = velocities[b];
    const float3 closing = (va.Linear + cross(va.Angular, arm_a)) - (vb.Linear + cross(vb.Angular, arm_b));
    // Positive along the normal is separating, and the pass drives it to e times the speed the step
    // began closing at. Accumulated so the pass can be iterated - Box2D notes that iterating only
    // matters where a manifold has more than one point - and clamped at zero so the total can only
    // ever push the two apart.
    const float total = max(0.f, contact.BounceImpulse + (restitution * contact.Approach - dot(normal, closing)) / weight);
    contact.BounceDelta = total - contact.BounceImpulse;
    contact.BounceImpulse = total;
}

// And the gather that moves the bodies: each one sums what the pass just asked for over the contacts it
// is in, exactly as the primal sweep gathers forces and for the same reason.
kernel void ApplyRestitution(
    device Velocity *velocities [[buffer(3)]], device const Contact *contacts [[buffer(5)]],
    device const Pose *poses [[buffer(0)]], device const BodyMass *masses [[buffer(4)]],
    device const Adjacency *incoming [[buffer(17)]], device const uint *incoming_slots [[buffer(18)]],
    device const uint *quiet [[buffer(21)]], constant StepParams &p [[buffer(7)]],
    uint body [[thread_position_in_grid]]
) {
    if (body >= p.BodyCount) return;
    const BodyMass mass = masses[body];
    if (mass.InvMass == 0 || Asleep(quiet[body], p)) return;

    const float4 orientation = poses[body].Orientation;
    const float3x3 inverse_inertia = InverseInertia(orientation, mass.InvInertiaLocal);
    float3 linear = float3(0), angular = float3(0);
    const uint own = body * ContactsPerBody, count = incoming[body].Count, start = incoming[body].Start;
    for (uint i = 0; i < ContactsPerBody + count; ++i) {
        const uint slot = ContactSlot(i, own, start, incoming_slots, contacts);
        if (slot == NoIndex) continue;
        const Contact contact = contacts[slot];
        if (contact.BounceDelta == 0) continue;
        const bool mine_is_a = contact.BodyA == body;
        const float3 arm = Rotate(orientation, mine_is_a ? contact.AnchorA : contact.AnchorB);
        // The normal points out of B towards A, so a positive impulse drives the two apart.
        const float3 impulse = contact.Normal * (mine_is_a ? contact.BounceDelta : -contact.BounceDelta);
        linear += impulse * mass.InvMass;
        angular += inverse_inertia * cross(arm, impulse);
    }
    velocities[body].Linear += linear;
    velocities[body].Angular += angular;
}

// Whether a body has been still long enough to stop being solved. Counted after the pass above rather
// than alongside the velocity it reads, because a body that has just been handed a rebound is not
// still and must not be allowed to fall asleep holding it.
kernel void CountQuiet(
    device const Pose *poses [[buffer(0)]], device const Velocity *velocities [[buffer(3)]],
    device const BodyMass *masses [[buffer(4)]], device uint *quiet [[buffer(21)]],
    device Pose *rest [[buffer(22)]], constant StepParams &p [[buffer(7)]],
    uint body [[thread_position_in_grid]]
) {
    if (body >= p.BodyCount || masses[body].InvMass == 0 || Asleep(quiet[body], p)) return;
    const Velocity now = velocities[body];
    const bool slow = length(now.Linear) <= p.SleepSpeed && length(now.Angular) <= p.SleepSpeed;
    uint counted = slow ? quiet[body] + 1 : 0;
    if (counted == p.SleepSteps) {
        // The step it would fall asleep on. Slow is not the same as arrived - a settling stack is slow
        // long before it stops moving - so what decides it is how far the body actually got over the
        // window, and still travelling starts the count again from here.
        const Displacement since = Since(poses[body], rest[body]);
        if (length(since.Linear) > p.SleepDrift || length(since.Angular) > p.SleepDrift) counted = 0;
    }
    quiet[body] = counted;
    if (counted == 0) rest[body] = poses[body];
}

// No body is quieter than what it is touching. A body takes the smallest count among its neighbours
// and its own, which makes sleeping a property of a group rather than of a body: a stack settles
// together and falls asleep together, and one link of it still moving holds all of it awake.
//
// It has to be the whole count and not just a test for zero. A body that reaches the threshold while a
// neighbour is five steps behind goes to sleep mid-settle, and the neighbour carries on pressing
// against something that has stopped answering - a stack converging to a tenth of a millimetre ended
// three times further out for the three times it happened.
//
// It spreads a hop a step, which costs nothing: the count has to climb to the threshold regardless.
kernel void SpreadWaking(
    device const uint *quiet [[buffer(21)]], device uint *next [[buffer(23)]],
    device const Contact *contacts [[buffer(5)]], device const Joint *joints [[buffer(16)]],
    device const Adjacency *incoming [[buffer(17)]], device const uint *incoming_slots [[buffer(18)]],
    device const BodyMass *masses [[buffer(4)]], constant StepParams &p [[buffer(7)]],
    uint body [[thread_position_in_grid]]
) {
    if (body >= p.BodyCount) return;
    uint least = quiet[body];
    next[body] = least;
    if (masses[body].InvMass == 0 || least == 0) return;

    const uint own = body * ContactsPerBody, count = incoming[body].Count, start = incoming[body].Start;
    for (uint i = 0; i < ContactsPerBody + count; ++i) {
        const uint slot = ContactSlot(i, own, start, incoming_slots, contacts);
        if (slot == NoIndex) continue;
        const Contact contact = contacts[slot];
        const Index other = contact.BodyA == body ? contact.BodyB : contact.BodyA;
        if (masses[other].InvMass > 0) least = min(least, quiet[other]);
    }
    for (uint index = 0; index < p.JointCount; ++index) {
        const Joint joint = joints[index];
        if (!joint.Active || (joint.BodyA != body && joint.BodyB != body)) continue;
        const Index other = joint.BodyA == body ? joint.BodyB : joint.BodyA;
        if (masses[other].InvMass > 0) least = min(least, quiet[other]);
    }
    next[body] = least;
}

kernel void PublishWaking(
    device uint *quiet [[buffer(21)]], device const uint *next [[buffer(23)]],
    constant StepParams &p [[buffer(7)]], uint body [[thread_position_in_grid]]
) {
    if (body >= p.BodyCount) return;
    quiet[body] = next[body];
}
