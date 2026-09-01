// AVBD for rigid bodies, following Augmented Vertex Block Descent (Giles et al., SIGGRAPH 2025).
// Ported from ../avbd-demo2d/source/solver.cpp for the algorithm and its equation numbering.
// The 3D contact basis and friction cone come from ../MetalAVBD/MetalAVBD/AVBDCompute.metal. See NOTICE.md.
//
// One step: integrate to an inertial target, build contacts against the pose the step began from, then alternate primal sweeps with dual updates.
// Velocity is read from the motion before a final stabilization sweep removes leftover penetration, so error correction adds no energy.
//
// Sign convention: the normal points out of B towards A, and C is the separation along it.
// Penetration is therefore negative, and a contact's normal force is at most zero.

constant uint Dof = 6;

// A body too slow for long enough stops being solved and keeps its contacts, so loads stay up.
static bool Asleep(uint quiet, constant StepParams &p) { return quiet >= p.SleepSteps; }

// The single threshold for movement, used everywhere in a step.
static bool Moving(Velocity v, constant StepParams &p) { return length(v.Linear) > p.SleepSpeed || length(v.Angular) > p.SleepSpeed; }

// Kinematic: a body the solve cannot move that the host is moving.
// Keyed on Moves rather than on inverse mass, because a body pinned in space with an inertia of its own is dynamic and the solve still turns it.
static bool Driven(BodyMass mass, Velocity v, constant StepParams &p) {
    return !Moves(mass) && Moving(v, p);
}

// Whether this body's pose is unchanged since the last step, which lets a sleeping body carry its pairs forward rather than re-collide them.
// A kinematic body is not frozen.
static bool Frozen(BodyMass mass, Velocity v, uint quiet, constant StepParams &p) {
    return !Moves(mass) ? !Driven(mass, v, p) : Asleep(quiet, p);
}

// Whether the primal sweeps move this body. A pair with neither side solved has nothing for Eq. 11.
static bool Solved(BodyMass mass, uint quiet, constant StepParams &p) {
    return Moves(mass) && !Asleep(quiet, p);
}

// Whether a shape presents a manifold of its own, which decides which body owns a pair.
// A plane and a mesh are reference surfaces with no support function, so the convex side always owns the pair, and two of them produce no contact.
static bool Presents(uint kind) { return kind != ShapePlane && kind != ShapeMesh; }

// Box2D-lite's bias towards keeping the reference face already chosen, kept by both references.
constant float RelativeTolerance = 0.95f;
constant float AbsoluteTolerance = 0.01f;
// The same bias between an edge pair and a face, at MetalAVBD's value.
// Absolute rather than scaled by a half extent, an edge pair having no single box's extent to scale by.
constant float EdgeTolerance = 0.01f;

// Compiled a second time with STABILIZE set, for the final pass that keeps the C0 term.
#ifndef STABILIZE
#define STABILIZE 0
#endif
// How much of the error a step began with a hard row may ignore.
// Eq. 18 spreads the correction over several steps, and post-stabilization instead runs the solve at 1 and one pass at 0.
// A soft row takes 0 always.
#if STABILIZE
constant float ConstraintAlpha = 0; // keep all of the accumulated error, and correct it
constant float Stabilizing = 1; // and a joint row is stiff enough to move it - see SolveBodies
#else
constant float ConstraintAlpha = 1; // ignore it, and only resist error added during this step
constant float Stabilizing = 0;
#endif

// The inertial stiffness a pair brings to a row measuring a distance between them: their reduced mass over h squared.
// That is the scale a penalty is meaningful against (Sec. 3.4).
// `inverse` is the sum of the inverse masses, so an infinite mass drops out and two static bodies floor to zero penalty rather than an infinite one.
static float PairStiffness(float inverse, float dt) { return inverse > 0 ? 1 / (inverse * dt * dt) : 0; }

static float3 UnitAxis(uint i) { return float3(i == 0 ? 1.f : 0.f, i == 1 ? 1.f : 0.f, i == 2 ? 1.f : 0.f); }

static float3x3 Diagonal(float3 d) {
    return float3x3(float3(d.x, 0, 0), float3(0, d.y, 0), float3(0, 0, d.z));
}

static float3x3 QuatToMatrix(float4 q) {
    return float3x3(Rotate(q, float3(1, 0, 0)), Rotate(q, float3(0, 1, 0)), Rotate(q, float3(0, 0, 1)));
}

// R D R^T: a body-frame diagonal tensor in the world. Takes the matrix, not the quaternion.
static float3x3 WorldTensor(float3x3 rotation, float3 diagonal) {
    return rotation * Diagonal(diagonal) * transpose(rotation);
}

// Zero for a static body, as its inverse mass is, so a pair with one takes only the other's.
static float3x3 WorldInverseInertia(float4 orientation, float3 inverse_local) {
    return WorldTensor(QuatToMatrix(orientation), inverse_local);
}

// Solves H x = -g by LDL^T without pivoting, for a symmetric positive definite H.
// H is definite by construction, the inertial term putting mass on the whole diagonal before any contact adds to it.
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

// Adds one constraint row to a body's 6x6 block: its force onto the gradient, and its outer product scaled by the row's stiffness onto the Hessian.
// The Jacobian is `axis` together with the moment `arm` makes with it.
// `side` is +1 where the row measures this body as A and -1 as B.
static void AddRow(
    thread float H[Dof][Dof], thread float g[Dof], float3 axis, float3 arm, float side, float force, float stiffness
) {
    const float3 angular = cross(arm, axis);
    float row[Dof];
    for (uint k = 0; k < 3; ++k) {
        row[k] = side * axis[k];
        row[3 + k] = side * angular[k];
    }
    for (uint i = 0; i < Dof; ++i) {
        g[i] += row[i] * force;
        for (uint j = 0; j < Dof; ++j) H[i][j] += stiffness * row[i] * row[j];
    }
}

// Removes one direction from a body's block, for a degree of freedom it does not have.
// Call after every row is in, with `first` 0 for a direction in space and 3 for one to turn about.
static void LockDirection(thread float H[Dof][Dof], thread float g[Dof], uint first, float3 axis) {
    // Projected out rather than given a large number, which would cost the LDL its conditioning and still let the body creep.
    // The diagonal keeps the block's own scale, deliberately not a unit.
    // Projection leaves an epsilon of the block's largest term coupled in, and a unit divisor turns that into a visible step or a zero pivot.
    float a[Dof], product[Dof], along = 0, gradient = 0, held = 1;
    for (uint i = 0; i < Dof; ++i) a[i] = i >= first && i < first + 3 ? axis[i - first] : 0;
    for (uint i = 0; i < Dof; ++i) {
        product[i] = 0;
        for (uint j = 0; j < Dof; ++j) product[i] += H[i][j] * a[j];
    }
    for (uint i = 0; i < Dof; ++i) {
        along += a[i] * product[i];
        gradient += a[i] * g[i];
        held = max(held, H[i][i]); // the block is positive definite, so its scale is on its diagonal
    }
    for (uint i = 0; i < Dof; ++i) {
        g[i] -= a[i] * gradient;
        for (uint j = 0; j < Dof; ++j) H[i][j] += a[i] * (along * a[j] - product[j] + held * a[j]) - product[i] * a[j];
    }
}

// The displacement of a body since the step began, the variable the constraints are expressed in.
struct Displacement {
    float3 Linear, Angular;
};

static Displacement Since(Pose now, Pose start) {
    return {now.Position - start.Position, RotationVector(QuatMul(now.Orientation, QuatConjugate(start.Orientation)))};
}

// Where one step of the body's own motion carries it, with `share` of gravity added.
static Pose FreeFlight(Pose pose, Velocity v, float3 gravity, float share, float dt) {
    return {pose.Position + dt * v.Linear + (share * dt * dt) * gravity,
            normalize(QuatMul(QuatFromRotationVector(dt * v.Angular), pose.Orientation))};
}

// Eq. 2: where free flight would put the body, the target the inertial term pulls back towards.
// The pose is deliberately left where it is, so collision and every C0 and Jacobian are taken at the pose the step began from.
// WarmStart then moves the body to its starting guess.
kernel void Integrate(
    device Pose *poses [[buffer(0)]], device Pose *initial [[buffer(1)]], device Pose *inertial [[buffer(2)]],
    device Velocity *velocities [[buffer(3)]],
    device const BodyMass *masses [[buffer(4)]], device Adjacency *incoming [[buffer(17)]],
    device uint *quiet [[buffer(21)]], constant StepParams &p [[buffer(7)]],
    uint body [[thread_position_in_grid]]
) {
    if (body >= p.BodyCount) return;
    incoming[body].Count = 0; // before CountIncoming adds to it, and every body has a thread here
    const Pose pose = poses[body];
    initial[body] = pose;
    const BodyMass mass = masses[body];
    if (!Moves(mass)) {
        inertial[body] = pose;
        return;
    }

    Velocity v = velocities[body];
    // A sleeping body's velocity is zero, so any velocity here came from outside and wakes it.
    if (Asleep(quiet[body], p) && Moving(v, p)) quiet[body] = 0;
    if (Asleep(quiet[body], p)) {
        inertial[body] = pose;
        velocities[body] = {float3(0), float3(0)};
        return;
    }

    // Damping as the fraction of velocity removed per second, Jolt's form, applied before the flight so the step integrates the damped body.
    // Floored at zero, because a coefficient past one over the step would reverse the velocity.
    // Applied only to the half of the motion the body has: no medium slows an infinite mass, and the linear velocity on a pinned body is the host carrying it.
    if (Translates(mass)) v.Linear *= max(0.f, 1 - mass.LinearDamping * p.DeltaTime);
    if (Turns(mass)) v.Angular *= max(0.f, 1 - mass.AngularDamping * p.DeltaTime);

    const float spin = length(v.Angular);
    if (spin > p.MaxAngularSpeed) v.Angular *= p.MaxAngularSpeed / spin;
    velocities[body] = v;

    // Eq. 2 with the body's own gravity: 1 is the world's, 0 leaves it in free flight, and a negative value lifts it.
    // The contact reach and the bounce threshold stay on the world's gravity.
    // An infinite mass takes no gravity, so its target is where the host's velocity carries it.
    inertial[body] = FreeFlight(pose, v, (Translates(mass) ? mass.GravityScale : 0) * p.Gravity, 1, p.DeltaTime);
}

// The starting guess the sweeps begin from, which is not the inertial target.
// Gravity enters in proportion to how much recent acceleration matched it, so a resting body is not guessed into the floor each step.
// VBD's adaptive warm start, run after collision. See Integrate.
kernel void WarmStart(
    device Pose *poses [[buffer(0)]], device const Pose *initial [[buffer(1)]],
    device const Velocity *velocities [[buffer(3)]], device Velocity *previous [[buffer(9)]],
    device const BodyMass *masses [[buffer(4)]], constant StepParams &p [[buffer(7)]],
    uint body [[thread_position_in_grid]]
) {
    if (body >= p.BodyCount) return;
    const BodyMass mass = masses[body];
    const Pose pose = initial[body];
    const Velocity v = velocities[body];
    const float dt = p.DeltaTime;
    if (!Moves(mass)) {
        // A kinematic body is moved to where the host is taking it, for the same reason a dynamic body is moved to its guess.
        // Eq. 15 measures constraints in the displacement of the two bodies.
        // A body left where it started therefore contributes nothing to any row and produces no velocity.
        // The flight is x + h v, with no gravity and no adaptive weight.
        if (Driven(mass, v, p)) poses[body] = FreeFlight(pose, v, float3(0), 0, dt);
        return;
    }
    // The body's own gravity on both sides of the weighting: the direction recent acceleration is compared against, and the amount the guess then carries.
    // Both velocities are the damped ones Integrate wrote, so this weighs the acceleration the body had.
    // A body at terminal velocity is not accelerating and must not be guessed a step of gravity further.
    const float3 pull = (Translates(mass) ? mass.GravityScale : 0) * p.Gravity;
    const float gravity = length(pull);
    const float3 fall = gravity > 1e-6f ? pull / gravity : float3(0);
    const float along = dot((v.Linear - previous[body].Linear) / dt, fall);
    const float weight = gravity > 1e-6f ? clamp(along / gravity, 0.f, 1.f) : 0.f;
    previous[body] = v;
    poses[body] = FreeFlight(pose, v, pull, weight, dt);
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

static float Reach(BoxPose box, float3 axis) {
    return abs(dot(box.Axis[0], axis)) * box.Half[0] + abs(dot(box.Axis[1], axis)) * box.Half[1] +
        abs(dot(box.Axis[2], axis)) * box.Half[2];
}

// The separating axis test: positive when the two boxes overlap along the axis, and negative by the gap otherwise.
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

// A convex polytope as the narrowphase addresses it: a pose, and an indexed vertex accessor.
// A box is one too, its eight corners computed rather than stored.
// A sphere or capsule is the one or two points of its core, because the Minkowski difference of two cores is that of the solids moved in by their radii.
// Everything below therefore works on cores and takes the radii off at the end.
struct Poly {
    float3 Center;
    float4 Orientation;
    float3 Half; // a box's half extents, or in y the half length of the segment a capsule surrounds
    float Radius; // a sphere's or a capsule's, and zero for a polytope
    uint First, Count; // a hull's run of the vertex pool
    uint FirstFace, FaceCount; // and of the face pool, the faces its cook produced
    uint3 Corner; // and a triangle's three, which are anywhere in the pool rather than a run
    uint Kind; // the ShapeKind it was made from, which selects the fields above
};

static Poly MakePoly(Pose pose, Shape shape) {
    return {pose.Position, pose.Orientation, shape.HalfExtents, shape.Radius, shape.FirstVertex, shape.VertexCount,
            shape.FirstFace, shape.FaceCount, uint3(0), shape.Kind};
}

// One triangle of a mesh, as a polytope with no thickness. The caller resolves which side is out.
static Poly MakeTriangle(Pose pose, Triangle triangle) {
    return {pose.Position, pose.Orientation, float3(0), 0, 0, 0, 0, 0, uint3(triangle.A, triangle.B, triangle.C), ShapeMesh};
}

static uint PolyCount(Poly poly) {
    if (poly.Kind == ShapeBox) return 8;
    if (poly.Kind == ShapeHull) return poly.Count;
    if (poly.Kind == ShapeMesh) return 3;
    return poly.Kind == ShapeCapsule ? 2 : 1; // a sphere's core is a single point
}

// Vertex `i` in world. A box's eight are its half-extent sign patterns, in the order `i`'s bits give.
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

// The vertex furthest along `direction`, as an index, because a contact is named by its geometry.
static uint PolySupport(Poly poly, device const float3 *pool, float3 direction) {
    const float3 local = Rotate(QuatConjugate(poly.Orientation), direction);
    if (poly.Kind == ShapeBox) return (local.x > 0 ? 1u : 0u) | (local.y > 0 ? 2u : 0u) | (local.z > 0 ? 4u : 0u);
    if (poly.Kind == ShapeCapsule) return local.y > 0 ? 1u : 0u;
    uint best = 0;
    float furthest = -INFINITY;
    for (uint i = 0, count = PolyCount(poly); i < count; ++i) {
        const float reach = dot(LocalVertex(poly, pool, i), local);
        if (reach > furthest) {
            furthest = reach;
            best = i;
        }
    }
    return best;
}

static float3 PolySupportPoint(Poly poly, device const float3 *pool, float3 direction) {
    return PolyVertex(poly, pool, PolySupport(poly, pool, direction));
}

// How far the furthest vertex sits from the centre, the scale a face tolerance is measured against.
static float PolyReach(Poly poly, device const float3 *pool) {
    if (poly.Kind == ShapeBox) return length(poly.Half);
    if (poly.Kind == ShapeCapsule) return poly.Half.y;
    // A triangle's own size, not its distance from the mesh origin, a hull being centred on its centre of mass.
    if (poly.Kind == ShapeMesh)
        return max(max(distance(pool[poly.Corner[0]], pool[poly.Corner[1]]), distance(pool[poly.Corner[1]], pool[poly.Corner[2]])),
                   distance(pool[poly.Corner[2]], pool[poly.Corner[0]]));
    if (poly.Kind != ShapeHull) return 0;
    float reach = 0;
    for (uint i = 0; i < poly.Count; ++i) reach = max(reach, length(pool[poly.First + i]));
    return reach;
}

// How far apart two corners must be before the polytope rests on one. Relative, with a floor.
static float FaceTolerance(Poly poly, device const float3 *pool) { return 1e-3f * PolyReach(poly, pool) + 1e-6f; }

// A sphere and a capsule are one shape with a different core: every point within Radius of a segment.
struct Core {
    float3 From, To;
    float Radius;
};

static Core MakeCore(Pose pose, Shape shape) {
    const float3 along = Rotate(pose.Orientation, float3(0, shape.HalfExtents.y, 0)); // zero for a sphere
    return {pose.Position - along, pose.Position + along, shape.Radius};
}

static bool IsRound(uint kind) { return kind == ShapeSphere || kind == ShapeCapsule; }

// The point of a box nearest `at`, and how far outside the box `at` is. Negative for a point inside.
static float3 ClosestOnBox(BoxPose box, float3 at, thread float &distance, thread uint &feature) {
    const float3 offset = at - box.Center;
    float3 local = float3(dot(offset, box.Axis[0]), dot(offset, box.Axis[1]), dot(offset, box.Axis[2]));
    const float3 clamped = clamp(local, -box.Half, box.Half);
    const float3 outside = local - clamped;
    if (dot(outside, outside) > 1e-14f) {
        // Which axes were clamped and to which side, naming the face, edge or vertex the point landed on.
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

// The outward direction from a box to a point. Reversed for a point inside, and arbitrary on the surface.
static float3 OutOfBox(float3 at, float3 nearest, float away) {
    return away < 0 ? normalize(nearest - at) : (away > 1e-9f ? (at - nearest) / away : float3(0, 1, 0));
}

// The box edge along `axis` furthest in `direction`, with `which` naming it so its parallels do not inherit its dual.
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

// Sutherland-Hodgman against one half-space, keeping the part behind the plane and carrying each point's name.
// Clips in place through the caller's scratch.
//
// A name is the set of planes the point lies on: bits 0-3 the four edges of the incident face, and bits 4-7 the four side planes of the reference face.
// Two bits are always set, and the pair names the point uniquely, so the same geometry takes the same name next step whatever order the points came out in.
//
// `tolerance` must be nonzero.
// Two faces that meet exactly have every corner of one on a side plane of the other, computed differently on the two sides.
// The result then lands either side of zero on rounding alone, and the pair contend over one piece of geometry.
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
        // A cut at either end is that endpoint renamed, so only a cut strictly between them is new.
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

// Convex against convex, following the sources LiteratureReview.md names under "Convex queries".
// GJK with Montanari's signed-volume subalgorithm gives the direction the two are apart along, and bounded EPA in fixed scratch the one they overlap along.
// The manifold is then a one-shot clip.
// Nothing here reads a polytope's topology, only its vertices through Poly's support function.
// One path therefore serves hull against hull, hull against box, and hull against a capsule's segment.

// The most points clipping one face into another can produce, which is more than either face holds.
// A convex polygon cut by a half-plane gains a vertex, so an eight point face against an eight edge one is a sixteen-gon.
// Bounding at MaxFacePoints instead drops a contiguous run of the perimeter, half the shape, which does not carry the body's centre.
// Jolt fills only half its buffer for the same reason.
constant uint MaxClipPoints = 2 * MaxFacePoints;
// EPA's scratch, bounded on purpose: it stops expanding and returns the best face it reached.
constant uint MaxEpaVertices = 16;
constant uint MaxEpaFaces = 32;

// A point of the Minkowski difference, carrying which vertex of each polytope produced it, so it can be named.
struct Mink {
    float3 At;
    uint IndexA, IndexB;
};

static Mink MinkSupport(Poly a, Poly b, device const float3 *pool, float3 direction) {
    const uint ia = PolySupport(a, pool, direction), ib = PolySupport(b, pool, -direction);
    return {PolyVertex(a, pool, ia) - PolyVertex(b, pool, ib), ia, ib};
}

// The signed-volume distance subalgorithm (Montanari, Petrinic and Barbieri, TOG 36(3) 2017).
// The paper is kept at ~/acoustic_solver_papers/2017_montanari-petrinic-barbieri_improving-gjk-signed-volumes.pdf.
// It replaces both Johnson's subalgorithm and the backup procedure GJK otherwise needs.
// It never drops the newest vertex, which requires the simplex ordered newest-first, so Gjk prepends.
// It projects onto whichever plane or axis the simplex shades most of.
// A needle simplex therefore yields a NaN the sign comparisons reject rather than a poisoned division.

// The paper's CompareSigns. Zero and NaN both give false, so a degenerate simplex is rejected.
static bool SameSign(float a, float b) { return (a > 0 && b > 0) || (a < 0 && b < 0); }

// Six times the signed tetrahedron volume. Replacing a corner by the origin gives that corner's coordinate.
static float Signed4(float3 a, float3 b, float3 c, float3 d) { return dot(b - a, cross(c - a, d - a)); }

// And twice the signed area of the triangle projected onto the (u, v) plane, the same way.
static float Signed2(float3 a, float3 b, float3 c, uint u, uint v) {
    return (b[u] - a[u]) * (c[v] - a[v]) - (b[v] - a[v]) * (c[u] - a[u]);
}

// The point of a segment nearest the origin, with a mask of which ends carry it. s1 is the newest vertex.
static float3 SignedVolume1(float3 s1, float3 s2, thread uint &mask) {
    const float3 along = s2 - s1;
    const float3 projected = s2 - along * (dot(s2, along) / dot(along, along));
    // The axis the segment shades longest, which its coordinates are best conditioned on.
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
    mask = 1; // the origin is past the newest vertex, and the older one cannot be nearest on its own
    return s1;
}

// The point of a triangle nearest the origin, likewise.
static float3 SignedVolume2(float3 s1, float3 s2, float3 s3, thread uint &mask) {
    const float3 turn = cross(s2 - s1, s3 - s1);
    // The origin projected onto the triangle's plane. A needle triangle makes this a NaN.
    const float3 projected = turn * (dot(s1, turn) / dot(turn, turn));
    // The Cartesian plane the triangle shades most, in cyclic order so an area's sign means one thing.
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

// And of a tetrahedron. All four coordinates agreeing in sign with its volume means the origin is inside it.
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

// The point of the simplex nearest the origin, cut back to the vertices that carry it.
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

// GJK, Algorithm 1 of the same paper.
// Returns false with the direction the two cores are apart along, out of b towards a, and the distance.
// Returns true where they overlap, leaving its tetrahedron for EPA.
// The new support point goes in at the front, the subalgorithm above taking s1 for the newest.
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
        // Eq. 9, relative to the simplex: the origin is on the difference, so the two touch exactly.
        float scale = 0;
        for (uint i = 0; i < count; ++i) scale = max(scale, dot(simplex[i].At, simplex[i].At));
        if (squared <= 1e-10f * scale) return false;
        const Mink next = MinkSupport(a, b, pool, -closest);
        // Eq. 10: with no vertex reaching further towards the origin, this is the face facing it.
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

// Adds one face to the polytope EPA is growing. False where the turn has no area to normalize by.
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

// The expanding polytope: grows the tetrahedron GJK finished inside towards the nearest face of the Minkowski difference.
// That face's outward normal is the least separating direction, so the contact normal is its opposite.
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
        // A flat simplex has no inside to expand.
        if (!PushFace(vertices, faces, planes, offsets, face_count, i, j, k, turn)) return false;
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
        // Out of room to grow, so return the best face reached, which is the bound.
        if (vertex_count == MaxEpaVertices || face_count + 8 > MaxEpaFaces) return true;
        const Mink next = MinkSupport(a, b, pool, normal);
        if (dot(next.At, normal) - least < 1e-6f) return true; // nothing further out, so this is the face

        // The faces the new point can see are removed, and the rim they leave is filled in from it.
        // An edge shared by two departing faces is interior to the hole, found by cancelling it against its reverse.
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
            // A sliver contributes no usable direction and is left out.
            PushFace(vertices, faces, planes, offsets, face_count, i, j, apex,
                     cross(vertices[j].At - vertices[i].At, vertices[apex].At - vertices[i].At));
        }
        if (face_count == 0) return false;
    }
    return true;
}

// At most `limit` of the polytope's vertices at or past `threshold` along `axis`, named by index.
// Where more qualify, taking the first of them would take a contiguous run of the rim, half a disc, which does not carry the body's centre.
// They are picked by spread instead: the lowest-indexed qualifier anchors the set, and each one after it is the furthest from everything already kept.
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

    // More corners reach than may be kept, and which ones go is constrained.
    // A plane carries no face list, so the level of the deepest corner is the presented face.
    // A corner further back than the resolution of the geometry is behind it.
    // Left to the spread alone, a crowned plate keeps three untouching rim corners over three of the four it stands on.
    // A body reduced to one point then rocks on nothing but I/h^2.
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
            if (nearest <= widest) continue; // ties go to the lower index, so the set is deterministic
            widest = nearest;
            pick = i;
        }
        out[found] = PolyVertex(poly, pool, pick);
        names[found] = pick;
        ++found;
    }
    return found;
}

// The face `poly` presents along `direction`: its vertices ordered so consecutive pairs are its edges, with the face's plane normal out of the polytope.
// Fewer than three vertices is a corner or an edge, which cannot hold a manifold.
// Which face it is comes from the shape's own construction, and a hull's from its cook, so no tolerance is needed. See HullFace in Shared.h.
// The first point is the lowest vertex index, so edge zero is the same edge every step and a clipped point keeps its name.
static uint SupportFace(
    Poly poly, device const float3 *pool, device const HullFace *hull_faces, float3 direction,
    thread float3 *out, thread uint *names, thread float3 &plane
) {
    // In the polytope's own frame, where its faces are, so the choice does not depend on where the body stands.
    const float3 local = normalize(Rotate(QuatConjugate(poly.Orientation), direction));
    float3 local_plane = local;
    uint corners[MaxFacePoints], found = 0;

    if (poly.Kind == ShapeHull) {
        uint best = 0;
        float most = -INFINITY;
        for (uint f = 0; f < poly.FaceCount; ++f) {
            const float along = dot(hull_faces[poly.FirstFace + f].Normal, local);
            if (along <= most) continue; // ties go to the lower face index, so the choice is deterministic
            most = along;
            best = f;
        }
        if (poly.FaceCount == 0) return 0;
        const HullFace face = hull_faces[poly.FirstFace + best];
        local_plane = face.Normal;
        // The cook wound it about that normal, started it at its lowest vertex, and sampled it around its rim where it was wider than this may hold.
        // It therefore arrives named, wound and within the width.
        for (uint i = 0; i < face.Count && i < MaxFacePoints; ++i) corners[found++] = face.Corner[i];
    } else if (poly.Kind == ShapeBox) {
        // The face the direction points most strongly out of, and its four corners wound about that normal.
        // A corner's name is the same bitmask PolyVertex reads, so the two agree by construction.
        const float3 magnitude = abs(local);
        const uint axis = magnitude.x >= magnitude.y && magnitude.x >= magnitude.z ? 0u : (magnitude.y >= magnitude.z ? 1u : 2u);
        const bool positive = local[axis] > 0;
        local_plane = float3(0);
        local_plane[axis] = positive ? 1 : -1;
        const uint across = (axis + 1) % 3, along = (axis + 2) % 3;
        // Wound so the turn from `across` to `along` is positive about the face normal on the positive side, and reversed on the negative one.
        // The loop is then outward either way.
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
        // A capsule's core is a segment and a sphere's is a point, neither of which is a face.
        // Both come back unchanged and the caller takes the closest-pair branch.
        const uint count = PolyCount(poly);
        for (uint i = 0; i < count; ++i) corners[found++] = i;
    }

    // How much of that face is presented: whether the body meets it flat or stands on one of its edges or corners.
    // That depends on the direction and so needs a tolerance, but the tolerance can only take a subset of the one face topology already chose, never union two.
    // Fewer than three left is an edge or a corner, which the caller handles.
    //
    // A triangle keeps all three corners whatever the direction, a half-metre triangle losing its third corner to a twentieth of a degree of error.
    // A capsule's core is not a face at all.
    if (poly.Kind == ShapeHull || poly.Kind == ShapeBox) {
        float furthest = -INFINITY;
        for (uint i = 0; i < found; ++i) furthest = max(furthest, dot(LocalVertex(poly, pool, corners[i]), local));
        const float tolerance = FaceTolerance(poly, pool);
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

    // Rotated so the lowest name comes first.
    // A hull's cook already did this and a box's winding starts wherever the axis put it, so this makes the two agree on where edge zero of a face is.
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

// The unit normal of side plane `e`, out of the face. False where the edge is too short to give one.
static bool SidePlane(thread const float3 *face, uint count, uint e, float3 plane, thread float3 &unit) {
    const float3 side = cross(face[(e + 1) % count] - face[e], plane);
    const float span = length(side);
    if (span < 1e-12f) return false;
    unit = side / span;
    return true;
}

// How closely a normal must line up with a face to be that face's own. Loose, the match being unambiguous.
constant float BuriedAlignment = 0.999f;

// Whether `direction`, out of this shape, is a face buried against a sibling of its compound.
// A contact there is inside the body's own solid, the join two coplanar children share being interior rather than surface. See InternalFaces.
static bool BuriedAlong(Shape shape, Pose pose, device const HullFace *hull_faces, float3 direction) {
    const uint internal = InternalFaces(shape);
    if (internal == 0) return false; // every shape that is not a compound's child, which is nearly all of them
    const float3 local = normalize(Rotate(QuatConjugate(ComposePose(pose, shape.Local).Orientation), direction));
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

// The manifold between two convex polytopes, `a` this body's and `b` the other's.
// Fills each contact's pair of points, on a and on b, with the geometry's name and the normal out of b towards a.
//
// A name is the planes the point lies on: bits 0-7 an edge of the incident face and bits 8-15 a side plane of the reference one.
// Then comes each polytope's face as its lowest vertex index, then which body presented the reference.
// No part of a name is a position in the output array.
//
// `known` is a direction the caller has established the two are apart along, or zero to search.
// A triangle requires it: a search a thousandth of a radian off stops a flat triangle presenting a face.
// Given the direction, the triangle always takes the reference.
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
        const float3 near_a = PolySupportPoint(a, pool, -axis);
        const float3 near_b = PolySupportPoint(b, pool, axis);
        distance = dot(near_a - near_b, axis);
    } else if (Gjk(a, b, pool, simplex, simplex_count, axis, distance)) {
        float depth;
        float3 out_of_a;
        if (simplex_count < 4 || !Epa(a, b, pool, simplex, out_of_a, depth)) return 0;
        // EPA's face faces out of the difference, so the normal out of b towards a is its opposite.
        axis = -out_of_a;
        distance = -depth;
    }
    const float gap = distance - a.Radius - b.Radius;
    if (gap >= margin) return 0;

    float3 face_a[MaxFacePoints], face_b[MaxFacePoints];
    uint name_a[MaxFacePoints], name_b[MaxFacePoints];
    float3 plane_a, plane_b; // each out of its own polytope, towards the other
    const uint count_a = SupportFace(a, pool, hull_faces, -axis, face_a, name_a, plane_a);
    const uint count_b = SupportFace(b, pool, hull_faces, axis, face_b, name_b, plane_b);

    if (count_a < 3 && count_b < 3) {
        // Neither presents a face, so this is the nearest pair of two points or segments.
        float3 on_a, on_b;
        ClosestOnSegments(face_a[0], face_a[count_a - 1], face_b[0], face_b[count_b - 1], on_a, on_b);
        normal = axis;
        here[0] = on_a - a.Radius * axis;
        there[0] = on_b + b.Radius * axis;
        names[0] = (1u << 29) | name_a[0] | (name_a[count_a - 1] << 6) | (name_b[0] << 12) | (name_b[count_b - 1] << 18);
        return 1;
    }

    // Whichever face lies flatter against the normal holds the contact, and the other is clipped into it.
    // A caller that handed in the direction handed in a's own face normal, so a takes the reference outright.
    // Left to the comparison that is a tie within the rounding of a normalize.
    // A box lying flat across a mesh would then name half its manifold after the body rather than the geometry.
    // Those points would then be out of reach of the seam rule.
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
        // A corner sits on its two edges, and a lone point or segment end is named by itself alone.
        poly_names[i] = incident_count >= 3 ? ((1u << i) | (1u << ((i + incident_count - 1) % incident_count))) : (1u << i);
    }
    // Relative to where the face is, a dot product's rounding scaling with its inputs.
    float scale = 1;
    for (uint i = 0; i < reference_count; ++i) scale = max(scale, length(reference[i]));
    const float tolerance = 1e-5f * scale;
    if (incident_count >= 3) {
        for (uint e = 0; e < reference_count; ++e) {
            float3 unit; // out of the reference face, so the kept part is inside it
            if (!SidePlane(reference, reference_count, e, reference_plane, unit)) continue;
            poly_count = ClipAgainst(poly, poly_names, poly_count, unit, dot(unit, reference[e]), 1u << (8 + e), tolerance, MaxClipPoints, clipped, clipped_names);
        }
    } else if (incident_count == 1) {
        // One point, which the loop below would keep unconditionally.
        // It is on the reference face only when inside every side plane, a corner level with a face but off to the side not being in contact.
        for (uint e = 0; e < reference_count; ++e) {
            float3 unit;
            if (!SidePlane(reference, reference_count, e, reference_plane, unit)) continue;
            if (dot(unit, poly[0] - reference[e]) > tolerance) poly_count = 0;
        }
    } else if (incident_count == 2) {
        // A segment, clipped as an interval because Sutherland-Hodgman would emit its cut point twice.
        const float3 from = poly[0], along = poly[1] - poly[0];
        float low = 0, high = 1;
        uint low_name = 1u << 0, high_name = 1u << 1;
        for (uint e = 0; e < reference_count; ++e) {
            float3 unit;
            if (!SidePlane(reference, reference_count, e, reference_plane, unit)) continue;
            const float offset = dot(unit, reference[e]);
            const float at_from = dot(unit, from) - offset, at_to = dot(unit, poly[1]) - offset;
            const float slope = at_to - at_from;
            if (abs(slope) < 1e-12f) { // parallel to the plane, so wholly in or wholly out
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

    uint reference_face = 63, incident_face = 63; // each named by its lowest vertex index, which is geometry
    for (uint i = 0; i < reference_count; ++i) reference_face = min(reference_face, reference_names[i]);
    for (uint i = 0; i < incident_count; ++i) incident_face = min(incident_face, incident_names[i]);
    const float reference_offset = dot(reference_plane, reference[0]);
    const float reference_radius = reference_is_a ? a.Radius : b.Radius;
    const float incident_radius = reference_is_a ? b.Radius : a.Radius;

    uint found = 0;
    for (uint i = 0; i < poly_count && found < MaxClipPoints; ++i) {
        // How far the incident point stands off the reference face plane, before the radii come off.
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

// One piece of geometry, one row.
// A clip emits the same point twice wherever the geometry is degenerate against it.
// Each copy carries its own dual and penalty, which doubles the force at one spot.
// Welded here rather than in each clipper, at the scale SupportFace resolves geometry at, keeping the first of each pair.
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

// The four points of a manifold with the largest area between them, Gregorius's reduction (GDC 2015).
// The talk is kept at ~/acoustic_solver_papers/2015_gregorius_robust-contact-creation.pdf.
// Four points hold a face contact against turning as well as sliding.
// Area rather than distance, because two points far apart on one edge leave the body free to rock about it.
static uint ReduceManifold(thread float3 *here, thread float3 *there, thread uint *names, uint found, float3 normal) {
    if (found <= ManifoldPoints) return found;
    // Every comparison below is settled by the geometry rather than by the last bits of a world position.
    // A tie falling the other way renames all four points and discards their duals.
    // An octagon lying flat is such a case: eight points at one depth and two squares of identical area.
    // Rounding alone then flips the choice every few steps as the body turns.
    // A challenger therefore has to beat the incumbent by a margin relative to the manifold's own size, which leaves every tie with the lowest-indexed point.
    // This is the same relative-plus-absolute bias the box SAT keeps on its reference axis, for the same reason.
    float extent = 0;
    for (uint i = 0; i < found; ++i) extent = max(extent, distance(here[i], here[0]));
    const float slack = 1e-5f * extent + 1e-9f, area_slack = 1e-5f * extent * extent + 1e-9f;

    uint keep[4]; // the four picked here: deepest, furthest, largest triangle, most added area
    // The deepest, so the point resolving penetration is always in the set.
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
    // The one making the largest triangle on that edge.
    // Its sign gives the winding of the triangle about the normal, and swapping the pair to wind it positively lets the fourth point be found by sign alone.
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
    // And the point adding the most to that triangle: outside one of its edges, the side the signed area comes out negative on.
    // The largest such area decides, rather than the largest distance.
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

    // In the order they were found in, so a point's slot does not depend on which of the four roles it was chosen for.
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

// How many triangles of one mesh a body may be collided against in a batch, and how deep the walk may go.
// Fixed, a kernel being unable to grow anything, and anything a body reaches past these counts as a refusal.
constant uint MaxMeshTriangles = 32;
constant uint MeshStackDepth = 32;

// The next batch of triangles of a mesh whose bounds the body's own box reaches into.
// `low` and `high` are the body's box in the mesh's frame, the frame the tree was built in.
//
// The walk's stack belongs to the caller and survives between calls.
// `MaxMeshTriangles` therefore bounds how many are held at once rather than how much of the mesh a body may touch.
// A body cut off at a fixed number gets no contacts at all and falls through the floor.
//
// A leaf's `First` counts from the mesh's first triangle and an interior node's from its own root.
static uint GatherTriangles(
    Shape mesh, float3 low, float3 high, device const BvhNode *nodes, thread uint *stack, thread uint &depth, thread uint *out
) {
    uint found = 0;
    while (depth > 0) {
        const uint at = stack[--depth];
        const BvhNode node = nodes[mesh.RootNode + at];
        if (any(node.High < low) || any(node.Low > high)) continue;
        if (node.Count > 0) {
            // A leaf holds four, so a short batch puts the node back rather than splitting it.
            if (found + node.Count > MaxMeshTriangles) {
                stack[depth++] = at;
                return found;
            }
            for (uint i = 0; i < node.Count; ++i) out[found++] = node.First + i;
            continue;
        }
        // The left child was written straight after this node, so only the right needs an index.
        // The depth guard covers a tree deeper than four billion leaves, and it keeps a stray write out of the caller's stack.
        if (depth + 2 > MeshStackDepth) continue;
        stack[depth++] = at + 1;
        stack[depth++] = node.First;
    }
    return found;
}

// Whether a point lies along edge `e`'s line, to the tolerance the clip that made it measured with.
static bool OnEdgeLine(Poly face, device const float3 *pool, float3 outward, uint e, float3 at_point, float seam) {
    const float3 at = PolyVertex(face, pool, e);
    const float3 side = cross(PolyVertex(face, pool, (e + 1) % 3) - at, outward);
    const float span = length(side);
    return span > 1e-12f && abs(dot(side / span, at_point - at)) <= seam;
}

// Reports the contacts a body held last step that were not claimed this one.
// Every exit from CollectContacts goes through this, a body that stopped colliding having still ended every contact it held.
static void EndUnclaimed(
    device ContactEvent *events, device uint *counts, uint body, ulong claimed, uint reported,
    thread const uint *was_feature, thread const Index *was_other, thread const Index *was_sub,
    thread const uint *was_children
) {
    for (uint j = 0; j < ContactsPerBody; ++j) {
        if (was_feature[j] == NoIndex) break; // the run is dense, so this sentinel ends it
        if ((claimed & (1ul << j)) != 0) continue;
        events[reported++] = ContactEvent{body, was_other[j], was_feature[j], was_sub[j], was_children[j], uint(ContactRemoved)};
    }
    counts[body] = reported;
}

// The pieces a body collides with: its own shape, or a compound's children, each with its own Local.
static uint ShapeLeaves(Shape shape, Index shape_index, thread Index *out) {
    if (shape.Kind != ShapeCompound) {
        out[0] = shape_index;
        return 1;
    }
    uint count = 0;
    for (uint i = 0; i < ChildrenPerCompound; ++i) {
        const Index child = ChildOf(shape, i);
        if (child == NoIndex) break; // the run's terminator, as Shape describes it
        out[count++] = child;
    }
    return count;
}

// Filtered N^2 broadphase and narrowphase, one thread per body.
// That thread owns all of its body's contact slots, so it reads the previous step's before overwriting them and carries the dual across by feature.
// No other thread appends to the run, so the pool is identical on every run.
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
    // claim must accumulate over every body this one touches before anything can be called removed.
    // One bit a slot, in a word wide enough for the whole run, which caps ContactsPerBody at 64.
    ulong claimed = 0;
    uint reported = 0;
    // Which slot each live contact inherited from, per slot since a contact can lose its place later.
    uint inherited[ContactsPerBody];

    // The previous step's state, kept only so a matching feature can inherit it.
    uint was_feature[ContactsPerBody], was_stick[ContactsPerBody];
    Index was_other[ContactsPerBody]; // the partner body, which only a removal still needs
    Index was_sub[ContactsPerBody]; // and which part of it, which for a mesh is the triangle
    uint was_children[ContactsPerBody]; // and which leaf of each shape, which for a compound is the child
    float3 was_lambda[ContactsPerBody], was_penalty[ContactsPerBody];
    float3 was_anchor_a[ContactsPerBody], was_anchor_b[ContactsPerBody];
    // A run is dense from zero and one NoIndex sentinel ends it, where every reader of was_* stops.
    for (uint i = 0; i < ContactsPerBody; ++i) {
        if (!slots[i].Active) {
            was_feature[i] = NoIndex;
            break;
        }
        was_feature[i] = slots[i].Feature;
        was_other[i] = slots[i].BodyB;
        was_sub[i] = slots[i].SubShape;
        was_children[i] = slots[i].Children;
        was_lambda[i] = slots[i].Lambda;
        was_penalty[i] = slots[i].Penalty;
        was_stick[i] = slots[i].Stick;
        was_anchor_a[i] = slots[i].AnchorA;
        was_anchor_b[i] = slots[i].AnchorB;
        slots[i].Active = false;
    }

    const Index shape_index = body_shapes[body];
    // A body with no shape is a removed body's slot, and a reference surface presents no manifold and never owns a pair.
    // Deliberately no test for a static body, a dynamic mesh's partner being able to be static.
    // Exiting on mass alone would leave no owner for the pair, and the mesh falls through.
    if (shape_index == NoIndex) {
        EndUnclaimed(events, contact_event_counts, body, claimed, reported, was_feature, was_other, was_sub, was_children);
        return;
    }
    const Shape body_shape = shapes[shape_index];
    if (!Presents(body_shape.Kind)) {
        EndUnclaimed(events, contact_event_counts, body, claimed, reported, was_feature, was_other, was_sub, was_children);
        return;
    }
    const Pose pose = poses[body];
    const Filter own_filter = filters[body];
    const float own_friction = frictions[body], own_inverse_mass = masses[body].InvMass;
    // Whether the solve moves this body at all, which decides which body owns a pair.
    // Not the inverse mass alone, because a body pinned in space with an inertia of its own is turned by every contact.
    const bool i_move = Moves(masses[body]);
    // The pieces this body presents, each already at its pose within the body frame. See Shape.
    Index own_leaves[ChildrenPerCompound];
    const uint own_leaf_count = ShapeLeaves(body_shape, shape_index, own_leaves);

    uint count = 0;
    // A sleeping body's pairs against equally frozen partners are carried forward verbatim.
    // Neither pose has moved, so every anchor, C0 and dual is still exact.
    // Coverage survives because an approaching body is awake and its pairs are re-collided from whichever side owns them.
    // Sleep state is settled once a step, so both sides agree on frozen without communicating.
    // Frozen rather than static or asleep: a kinematic body is moved by nothing here yet is moving, and an undriven static body has not moved either.
    const bool frozen = Frozen(masses[body], velocities[body], quiet[body], p);
    if (frozen) {
        for (uint j = 0; j < ContactsPerBody; ++j) {
            if (was_feature[j] == NoIndex) break; // the sentinel ending the dense run
            const Index partner = was_other[j];
            if (body_shapes[partner] == NoIndex) continue; // removed, and its contacts end with it
            if (!Frozen(masses[partner], velocities[partner], quiet[partner], p)) continue; // moving, so re-collide
            // A compacting copy, and j never runs ahead of count.
            slots[count] = slots[j];
            slots[count].Active = true;
            inherited[count] = j;
            ++count;
        }
    }
    // Every leaf of this body against every leaf of every other, run full or not.
    // Which contacts a body keeps must not depend on the order the partners were visited in, and the refusal count below is then exact.
    for (uint own_leaf = 0; own_leaf < own_leaf_count; ++own_leaf) {
        const Shape shape = shapes[own_leaves[own_leaf]];
        // Where this leaf's geometry is.
        // Everything reading geometry works from this, and everything naming a point - anchors, lever arms, C0 - works from `pose`.
        // A contact belongs to the body frame whatever pose the shape sits at.
        const Pose shape_pose = ComposePose(pose, shape.Local);
        const BoxPose box = MakeBox(shape_pose, shape);
        const Poly own_poly = MakePoly(shape_pose, shape);
        // Hoisted out of the partner loop, being a scan over every vertex of a hull.
        const float own_reach = PolyReach(own_poly, hull_vertices) + shape.Radius;
        for (uint other = 0; other < p.BodyCount; ++other) {
            const Index other_shape = body_shapes[other];
            if (other == body || other_shape == NoIndex) continue;
            // The pairs the carry above holds, two frozen poses producing nothing new.
            if (frozen && Frozen(masses[other], velocities[other], quiet[other], p)) continue;
            // One manifold per pair, owned by the lower-indexed body, as the references do.
            // Generating it from both sides gives two independent constraint sets with two sets of duals for one physical contact.
            // Jacobi's symmetry hides that and Gauss-Seidel does not.
            // A body may defer only to a partner that will actually present the manifold.
            // A pair with a plane or a mesh therefore belongs to the convex side, whatever the index order.
            // Of two convex bodies the one the solve moves comes first.
            // A body of infinite mass that turns counts as moving, or nothing spins a pinned wheel.
            // Deliberately not split by parity to even the load.
            // Consecutive indices in a stack always sum odd, so parity flips every pair rather than alternating.
            // The middle of the stack then creeps under SleepSpeed and over SleepDrift for ever.
            const bool they_move = Moves(masses[other]);
            if (!i_move && !they_move) continue;
            if (Presents(shapes[other_shape].Kind)) {
                if (!i_move) continue; // they move and this body does not, so the pair is theirs
                if (they_move && other < body) continue; // both move, so the lower index owns it
            }

            // Each must be in the other's mask, and a joint between them makes the overlap by design.
            const Filter theirs = filters[other];
            if (!(own_filter.Layer & theirs.Collides) || !(theirs.Layer & own_filter.Collides)) continue;
            bool jointed = false;
            for (uint i = 0; i < JointsPerBody && !jointed; ++i) jointed = jointed_to[body * JointsPerBody + i] == other;
            if (jointed) continue;

            const Shape other_body_shape = shapes[other_shape];
            const Pose target_pose = poses[other];
            // The softest the normal row of this pair may be: the pair's reduced mass over h squared (Sec. 3.4).
            // On a settled contact the dual has absorbed the load, so C goes to zero and nothing opposes Eq. 19's decay to PenaltyMin.
            // The contact then all but vanishes from the block and a slow rocking mode rings on.
            // A six-box stack is quietest at M/h^2 and unstable past ten times it, so this is a ratio rather than a constant.
            // The normal row only: friction's penalty is algorithmic and the cone already bounds it, so a floor there locks the stick-slip transition early.
            const float pair_stiffness = PairStiffness(own_inverse_mass + masses[other].InvMass, p.DeltaTime);
            const float3 penalty_floor{max(p.PenaltyMin, pair_stiffness), p.PenaltyMin, p.PenaltyMin};
            const float friction = sqrt(own_friction * frictions[other]);
            // How far apart the pair may be and still be given contacts.
            // A contact built while the bodies are apart carries the gap as slack and does no work until the step's motion consumes it.
            // The step therefore ends at touch.
            // Avian's velocity-scaled margin with MetalAVBD's gravity term, a whole g h^2, since Integrate's target is x + h v + h^2 g.
            // Rotation is deliberately left out, collision at the pose the step began from being blind to swept orientation.
            // This replaces the margin in every generation test and nowhere else, C0 keeping ContactMargin.
            const Velocity own_velocity = velocities[body], other_velocity = velocities[other];
            const float reach = p.ContactMargin +
                min(p.DeltaTime * (length(own_velocity.Linear - other_velocity.Linear) + length(p.Gravity) * p.DeltaTime),
                    p.MaxContactReach);

            // And the other body's pieces, each against this one.
            // A leaf pair carries its own manifold, duals and name, a point on leaf 3 against leaf 5 being different geometry from leaf 2 against the same 5.
            Index target_leaves[ChildrenPerCompound];
            const uint target_leaf_count = ShapeLeaves(other_body_shape, other_shape, target_leaves);
            for (uint target_leaf = 0; target_leaf < target_leaf_count; ++target_leaf) {
                const Shape target = shapes[target_leaves[target_leaf]];
                const Pose target_shape_pose = ComposePose(target_pose, target.Local); // as above, on the other side
                // How close two points must be to be one, the scale SupportFace resolves at.
                const float geometry = max(own_reach, PolyReach(MakePoly(target_shape_pose, target), hull_vertices) + target.Radius);
                const float weld = 1e-3f * geometry + 1e-6f;
                // Whether either surface is round, which inherited friction anchors cannot assume.
                const bool curved = IsRound(shape.Kind) || IsRound(target.Kind);

                // How many manifolds this pair has: one, or one per mesh triangle the body reaches, a batch at a time.
                // The walk's stack lives out here so it survives between batches.
                uint candidates[MaxMeshTriangles], walk[MeshStackDepth], depth = 0;
                float3 low = 0, high = 0;
                if (target.Kind == ShapeMesh) {
                    // The body's box in the mesh's frame, let out by radius and margin.
                    const uint corners = PolyCount(own_poly);
                    low = INFINITY;
                    high = -INFINITY;
                    for (uint i = 0; i < corners; ++i) {
                        const float3 at = LocalPoint(target_shape_pose, PolyVertex(own_poly, hull_vertices, i));
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
                        // Which part of the other shape this is against, which only a mesh has.
                        Index sub_shape = NoIndex;
                        // And which leaf of each produced it.
                        const uint children = ChildPair(own_leaf, target_leaf);
                        // A manifold point is a pair: where it sits on this body, and where on the other.
                        // The two are distinct points, one always being a projection onto the other's surface.
                        float3 points_here[MaxClipPoints], points_there[MaxClipPoints], normal;
                        // A feature names the geometry that produced a point - which corner, or which
                        // pair of faces and which vertex of the clip - never its position in the
                        // output, or warm starting hands a dual to the wrong point when the touching
                        // set changes.
                        uint features[MaxClipPoints];
                        uint found = 0;

                        const bool hulled = shape.Kind == ShapeHull || target.Kind == ShapeHull;
                        if (target.Kind == ShapeMesh) {
                            const Index index = target.FirstTriangle + candidates[manifold];
                            const Triangle triangle = mesh_triangles[index];
                            sub_shape = index;
                            const Poly face = MakeTriangle(target_shape_pose, triangle);
                            const float3 first = PolyVertex(face, hull_vertices, 0);
                            const float3 turn = cross(PolyVertex(face, hull_vertices, 1) - first, PolyVertex(face, hull_vertices, 2) - first);
                            const float area = length(turn);
                            if (area < 1e-18f) continue; // a sliver the cook let through has no usable side
                            const float3 outward = turn / area; // out of the surface, as the winding defines it

                            // A mesh has no interior, so a body wholly behind a triangle is past it.
                            const float3 top = PolySupportPoint(own_poly, hull_vertices, outward);
                            if (dot(top - first, outward) + own_poly.Radius <= 0) continue;

                            // The triangle goes in first and with its own normal, which makes it the reference face every time.
                            // The manifold is then the body's face clipped into the triangle, and every point is named after the geometry under it.
                            // The result comes back the other way round, out of the mesh.
                            found = ConvexManifold(face, own_poly, hull_vertices, hull_faces, reach, -outward, points_there,
                                                   points_here, features, normal);

                            // Where that finds nothing while the body is in range, the given direction is wrong for this geometry.
                            // A body over a crease presents, to each slope's normal, the feature of itself over the other slope, which the clip drops.
                            // Both triangles then come back empty and the body falls through the ridge.
                            // The contact is against the crease itself and has to be searched for.
                            // Only as a fallback, and only where this triangle has an active edge.
                            // An inactive edge is a seam whose neighbour holds the body on its own face.
                            const float3 bottom = PolySupportPoint(own_poly, hull_vertices, -outward);
                            const bool within = dot(bottom - first, outward) - own_poly.Radius < reach;
                            const bool searched = found == 0 && within && triangle.ActiveEdges != 0;
                            if (searched)
                                found = ConvexManifold(face, own_poly, hull_vertices, hull_faces, reach, float3(0), points_there,
                                                       points_here, features, normal);
                            normal = -normal;

                            // A point one seam cut is cut by the triangle across it too, so both would hold one piece of geometry with a dual each.
                            // Dropping it from both loses nothing, only the tessellation having put it there.
                            // An edge that is a feature, a rim or a crease, is not a seam, and the points it cut stay.
                            // Bits 8 to 10 name which reference-face edges cut a point.
                            // Those are the triangle's own only while it is the reference (bit 28 clear).
                            // The tolerance is the scale the clip measured with.
                            float scale = 1;
                            for (uint v = 0; v < 3; ++v) scale = max(scale, length(PolyVertex(face, hull_vertices, v)));
                            const float seam = 1e-5f * scale;

                            uint kept = 0;
                            for (uint i = 0; i < found; ++i) {
                                // A surface has a side, and a body still in front of one is at most a step of motion behind it.
                                // The reach covers exactly a step.
                                // A point reading deeper is a body beside the triangle rather than through it.
                                // The cull above covers only the whole body being past the triangle.
                                // A body wider than the mesh piece it stands on gets its far face clipped in.
                                // That is a row whose ends are metres apart.
                                // It holds no force until post-stabilization takes it all back at once along contradictory normals.
                                if (dot(normal, points_here[i] - points_there[i]) < -reach) continue;
                                const bool triangle_led = ((features[i] >> 28) & 1) == 0;
                                // Which edges cut the point, and which were seams.
                                const uint cut_by = triangle_led ? (features[i] >> 8) & 7 : 0u;
                                const uint cut_by_seam = cut_by & ~triangle.ActiveEdges;
                                // Two of its edges cutting one point put it at a corner of the triangle.
                                // Where one of the two is a feature it is a corner of the surface as well.
                                // A body can rest there, so it is not a point both triangles may drop.
                                // Dropped by both, a cube on a two-triangle face keeps 2 rows on the diagonal instead of 4 on the corners.
                                // The seam's owner keeps it, which is the lower-numbered triangle as the cook wrote the bit.
                                // The two threads therefore read opposite results from one name.
                                const bool corner_is_mine = (cut_by & triangle.ActiveEdges) != 0 && (cut_by_seam & ~triangle.OwnedEdges) == 0;
                                if (cut_by_seam != 0 && !corner_is_mine) continue;
                                // A searched point not on an active edge belongs to the triangle across, which holds it on its own face.
                                // Without this every triangle of a flat mesh takes whatever is near its plane.
                                // A box sliding down the middle is then caught by edges nowhere near it.
                                if (searched) {
                                    bool on_feature = false;
                                    for (uint e = 0; e < 3 && !on_feature; ++e)
                                        on_feature = (triangle.ActiveEdges & (1u << e)) != 0 &&
                                            OnEdgeLine(face, hull_vertices, outward, e, points_there[i], seam);
                                    if (!on_feature) continue;
                                }
                                // A point the seam did not cut but that landed along it anyway.
                                // A vertex on a plane is inside it and keeps its own name, so both triangles hold it.
                                // The cook's owner decides which one reports it.
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
                            // One of the two is round, which makes the pair a distance problem.
                            // The contact is the nearest point of the other shape, with the radius taken off.
                            // A capsule's core is a segment and can lie along what it touches, so it takes one sample per end.
                            // Those two are its whole manifold, its surface between them being a straight ruling.
                            const bool mine_is_round = IsRound(shape.Kind);
                            const Shape round_shape = mine_is_round ? shape : target;
                            const Shape against = mine_is_round ? target : shape;
                            const Core core = MakeCore(mine_is_round ? shape_pose : target_shape_pose, round_shape);
                            const Pose other_pose = mine_is_round ? target_shape_pose : shape_pose;
                            const bool other_is_round = IsRound(against.Kind);
                            const Core other_core = other_is_round ? MakeCore(other_pose, against) : Core{};

                            // Where along each core to sample, and each sample's name.
                            float3 samples[2], others[2];
                            uint names[2];
                            uint taken = 0;
                            if (other_is_round) {
                                const float3 mine_along = core.To - core.From, theirs_along = other_core.To - other_core.From;
                                const float mine_length = length(mine_along), theirs_length = length(theirs_along);
                                const bool parallel = mine_length > 1e-6f && theirs_length > 1e-6f &&
                                    abs(dot(mine_along / mine_length, theirs_along / theirs_length)) > 0.999f;
                                if (parallel) {
                                    // Side by side: the stretch both cores cover.
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
                                            // The core end that bounded this limit names it.
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
                                // A plane is flat everywhere, so the core's ends are all of it.
                                samples[0] = core.From;
                                names[0] = 0;
                                taken = 1;
                                if (distance(core.From, core.To) > 1e-6f) {
                                    samples[1] = core.To;
                                    names[1] = 1;
                                    taken = 2;
                                }
                            } else {
                                // Against a box, the core's own ends do not give where to sample.
                                // A capsule can rest with its middle across a box and both ends over nothing.
                                // Alternating projection finds where the core comes nearest the box, converging because both shapes are convex.
                                // That gives the face the core lands on.
                                // The stretch of core over that face is where the capsule rests, and its two limits are the manifold.
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

                                // Clip the core to the two slabs across that face. Both limits keep
                                // the name of what set them, so the pair is the same pair next step
                                // wherever the search started.
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

                                // A sphere's core is a point, so both limits are one contact.
                                if (high - low > 1e-5f && length(along) * (high - low) > 1e-5f) {
                                    samples[0] = core.From + along * low;
                                    samples[1] = core.From + along * high;
                                    names[0] = low_name;
                                    names[1] = high_name;
                                    taken = 2;
                                } else {
                                    // No part of the core lies over a face, so the nearest point is the whole sample.
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
                                    out_of = span > 1e-9f ? apart / span : float3(0, 1, 0); // coincident, so any direction serves
                                    nearest = on_theirs + out_of * other_core.Radius;
                                    gap = span - other_core.Radius - core.Radius;
                                } else {
                                    float away;
                                    uint face;
                                    nearest = ClosestOnBox(MakeBox(other_pose, against), at, away, face);
                                    out_of = OutOfBox(at, nearest, away);
                                    gap = away - core.Radius;
                                }

                                if (gap >= reach) continue;
                                // The convention is out of the other body towards this one.
                                normal = mine_is_round ? out_of : -out_of;
                                const float3 on_round = at - out_of * core.Radius;
                                points_here[found] = mine_is_round ? on_round : nearest;
                                points_there[found] = mine_is_round ? nearest : on_round;
                                features[found] = names[sample];
                                ++found;
                            }
                        } else if (target.Kind == ShapePlane) {
                            // The vertices reaching through the plane are the manifold.
                            normal = target.Normal;
                            // Spread rather than the first eight, MaxFacePoints being the most one pair may report.
                            found = SpreadSupport(own_poly, hull_vertices, -normal, -(target.Offset + reach), MaxFacePoints, points_here, features);
                            for (uint i = 0; i < found; ++i)
                                points_there[i] = points_here[i] - (dot(normal, points_here[i]) - target.Offset) * normal;
                        } else if (!hulled) {
                            const BoxPose other_box = MakeBox(target_shape_pose, target);

                            // Separating axis test over all fifteen axes.
                            // The nine cross products are the edge-on-edge axes, and the shallowest is kept.
                            // Two boxes crossing at an angle touch along one pair of edges.
                            // Taking a face there gives the wrong normal and too much penetration.
                            bool apart = false;
                            uint edge_i = 0, edge_j = 0;
                            float3 edge_normal = float3(0);
                            float least_edge = INFINITY;
                            for (uint i = 0; i < 3 && !apart; ++i) {
                                for (uint j = 0; j < 3; ++j) {
                                    const float3 axis = cross(box.Axis[i], other_box.Axis[j]);
                                    const float len = length(axis);
                                    if (len < 1e-6f) continue; // parallel edges, covered by the face axes
                                    const float3 unit = axis / len;
                                    const float overlap = Overlap(box, other_box, unit);
                                    // Apart by more than the reach, rather than merely apart.
                                    // A projection gap is a lower bound on the distance.
                                    // The early-out therefore stays sound, and everything within reach is given its slack.
                                    if (overlap < -reach) {
                                        apart = true;
                                        break;
                                    }
                                    if (overlap < least_edge) {
                                        least_edge = overlap;
                                        edge_i = i;
                                        edge_j = j;
                                        // Out of the other body towards this one.
                                        edge_normal = dot(unit, box.Center - other_box.Center) < 0 ? -unit : unit;
                                    }
                                }
                            }
                            // The face axis they overlap along least is the one to separate them on.
                            // A challenger must beat the incumbent by a margin, relative and absolute both.
                            // That is the tolerance Box2D-lite carries and both references keep.
                            // Two faces of a stacked box overlap by almost the same amount.
                            // A flipped reference axis renames every point and discards its warm start.
                            // The relative part is a fraction of the incumbent's absolute size, the incumbent going negative when the boxes are disjoint.
                            uint best_axis = 0, best_owner = 0;
                            float least = INFINITY;
                            // The `!apart` guard is composed into the loop conditions rather than run as an early continue.
                            // A continue here miscompiled under fast math, taking the branch with its condition provably false.
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

                            // Both measured as separation, negative while the boxes overlap, so this reads as the reference writes it.
                            // The edge pair must beat the best face by a clear margin, which leaves ties to the faces.
                            // A stack of axis-aligned boxes never reaches here, its cross products all being degenerate.
                            const bool on_edge = least_edge < 1e18f && RelativeTolerance * -least_edge > -least + EdgeTolerance;
                            if (on_edge) {
                                float3 mine[2], theirs[2];
                                uint which_mine, which_theirs;
                                // This body's edge is the one reaching back along the normal.
                                SupportEdge(box, edge_i, -edge_normal, mine, which_mine);
                                SupportEdge(other_box, edge_j, edge_normal, theirs, which_theirs);
                                normal = edge_normal;
                                ClosestOnSegments(mine[0], mine[1], theirs[0], theirs[1], points_here[0], points_there[0]);
                                // Named by the two edges, so their parallels do not inherit its dual.
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

                                // Four corners cut by four planes is eight points, so no extra width.
                                float3 poly[MaxFacePoints], clipped[MaxFacePoints];
                                uint names[MaxFacePoints], clipped_names[MaxFacePoints];
                                FaceCorners(incident, incident_axis, incident_side, poly);
                                for (uint i = 0; i < 4; ++i) names[i] = (1u << i) | (1u << ((i + 3) % 4)); // its two edges
                                uint poly_count = 4;
                                // The relative clip tolerance the hull path uses. See ClipAgainst.
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

                                // The face plane sits one half-extent along the outward face_normal.
                                const float face_offset = dot(face_normal, reference.Center) + reference.Half[best_axis];
                                for (uint i = 0; i < poly_count && found < MaxFacePoints; ++i) {
                                    const float depth = dot(face_normal, poly[i]) - face_offset;
                                    if (depth >= reach) continue;
                                    // The point is on the incident body, its partner the projection.
                                    const float3 on_reference = poly[i] - depth * face_normal;
                                    points_here[found] = best_owner == 0 ? on_reference : poly[i];
                                    points_there[found] = best_owner == 0 ? poly[i] : on_reference;
                                    // Which body owned the reference face, which axes made the two faces, and where the point sits.
                                    // No part of this is an index into the output array.
                                    features[found] = (best_owner << 13) | (best_axis << 11) | (incident_axis << 9) |
                                        ((incident_side > 0 ? 1u : 0u) << 8) | names[i];
                                    ++found;
                                }
                            }
                        } else {
                            // A hull has no face list for the SAT, so this path uses support functions.
                            found = ConvexManifold(own_poly, MakePoly(target_shape_pose, target), hull_vertices, hull_faces, reach,
                                                   float3(0), points_here, points_there, features, normal);
                        }

                        // A manifold on a face buried against a sibling is inside that body's own solid, so there is no contact.
                        // Tested against the normal that came out rather than the faces each path chose between.
                        // The box test names one direction from either side.
                        if (found > 0 && (BuriedAlong(target, target_pose, hull_faces, normal) ||
                                          BuriedAlong(shape, pose, hull_faces, -normal)))
                            found = 0;

                        // No two rows on one piece of geometry, then the four worth keeping.
                        found = WeldManifold(points_here, points_there, features, found, weld);
                        found = ReduceManifold(points_here, points_there, features, found, normal);

                        // Whether either side has siblings, the only way two manifolds of one pair can land on the same geometry.
                        // Two children sharing an edge present the same corner, and WeldManifold sees only one leaf pair.
                        // The test below costs a scan of the run per point.
                        const bool siblings = own_leaf_count > 1 || target_leaf_count > 1;

                        // One slot per manifold point, its feature naming where it came from.
                        for (uint i = 0; i < found; ++i) {
                            const float3 anchor_a = LocalPoint(pose, points_here[i]);
                            const float3 anchor_b = LocalPoint(target_pose, points_there[i]);
                            // One piece of geometry, one row, across leaf pairs too: the first leaf to write a place keeps it.
                            // That is deterministic because leaves are walked in the compound's own order.
                            bool held = false;
                            for (uint k = 0; k < count && siblings && !held; ++k)
                                held = slots[k].Active && slots[k].BodyB == other && slots[k].SubShape == sub_shape &&
                                    distance(anchor_a, slots[k].AnchorA) <= weld && distance(anchor_b, slots[k].AnchorB) <= weld;
                            if (held) continue;

                            // Where this point goes. With room it takes the next slot, and once the
                            // run is full it must earn a place: the shallowest contact gives way, so
                            // a speculative contact at positive separation goes first and returns
                            // the moment it is the deeper. Deciding by body order leaves a box in a
                            // lattice holding four contacts with a neighbour it merely touches and
                            // none with the box on it.
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
                            contact.AnchorA = anchor_a;
                            contact.AnchorB = anchor_b;
                            contact.Normal = normal;
                            contact.BodyA = body;
                            contact.BodyB = other;
                            contact.Friction = friction;
                            contact.Feature = features[i];
                            contact.SubShape = sub_shape;
                            contact.Children = children;

                            // The closing speed when the step began, which a bounce is measured against.
                            // Ungated, the threshold and coefficient belonging to the velocity pass.
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
                                if (was_feature[j] == NoIndex) break; // the sentinel, with nothing to inherit past it
                                if (was_feature[j] != contact.Feature || was_other[j] != other || was_sub[j] != sub_shape || was_children[j] != children) continue;
                                inherited[at] = j;
                                contact.Penalty = clamp(was_penalty[j] * p.Gamma, penalty_floor, float3(p.PenaltyMax));
                                contact.Lambda = was_lambda[j];
                                // Static friction: a contact that stayed inside the cone last step keeps the anchor pair it held.
                                // C0's friction rows then measure the drift since it stuck.
                                // Recomputed anchors would leave a loaded box creeping every step.
                                // Not on a curved surface, whose contact sweeps across the material.
                                // And not where the anchors land on a row this pair already wrote.
                                // Two contacts that stuck at different times can drift onto each other with all but the same Jacobian and C0.
                                const float3 want_a = WorldPoint(pose, was_anchor_a[j]);
                                const float3 want_b = WorldPoint(target_pose, was_anchor_b[j]);
                                bool onto_another = false;
                                for (uint k = 0; k < count && !onto_another; ++k) {
                                    if (k == at || !slots[k].Active || slots[k].BodyB != other || slots[k].SubShape != sub_shape || slots[k].Children != children) continue;
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
                            // Eq. 15: separation in the contact basis, plus the normal row's margin.
                            const ContactBasis basis = MakeContactBasis(normal);
                            const float3 gap = WorldPoint(pose, contact.AnchorA) - WorldPoint(target_pose, contact.AnchorB);
                            contact.C0 = float3(dot(basis.Axis[0], gap), dot(basis.Axis[1], gap), dot(basis.Axis[2], gap)) + float3(p.ContactMargin, 0, 0);
                            if (at == count) ++count;
                        }
                    }
                }
            }
        }
    }

    // The events, once the run has settled rather than as each point is written.
    // A contact that lost its place to a deeper one was never held, so it reports no addition and its inherited slot does not count as claimed.
    for (uint k = 0; k < count; ++k) {
        if (inherited[k] != NoIndex) claimed |= 1ul << inherited[k];
        events[reported++] = ContactEvent{body, slots[k].BodyB, slots[k].Feature, slots[k].SubShape, slots[k].Children,
                                          uint(inherited[k] != NoIndex ? ContactPersisted : ContactAdded)};
    }
    EndUnclaimed(events, contact_event_counts, body, claimed, reported, was_feature, was_other, was_sub, was_children);
}

// A joint's frame on a body: the frame it recorded, turned by the body's current orientation.
static float4 JointFrame(float4 orientation, float4 frame) { return QuatMul(orientation, frame); }

// How far the frames have turned apart, back in the frame on B, where the joint's axes are named.
static float4 RelativeFrame(float4 frame_a, float4 frame_b) { return QuatMul(QuatConjugate(frame_b), frame_a); }

// The twist half of that rotation about one of the frame's axes.
// The remainder is the swing, whose axis is perpendicular, so the two together give the rotation with nothing double counted (Sec. 3.3).
static float4 TwistPart(float4 relative, float3 axis) {
    const float4 along = MakeFloat4(dot(relative.xyz, axis) * axis, relative.w);
    const float size = length(along);
    // Zero only where the pair has swung a half turn about a perpendicular axis, which is a hinge already come apart.
    return size > 1e-7f ? along / size : float4(0, 0, 0, 1);
}

// The twist angle, within a half turn of `near`, so a wheel's angle stays continuous.
static float TwistAngle(float4 relative, float3 axis, float near) {
    const float4 twist = TwistPart(relative, axis);
    const float turn = 2 * atan2(dot(twist.xyz, axis), twist.w);
    const float full = 2 * M_PI_F;
    return near + (turn - near) - full * round((turn - near) / full);
}

// How far the two frames have turned apart, resolved along the joint frame's own three axes.
// With a twist axis it is swing-twist: the two locked rows read the swing's rotation-vector components and the twist row the unwrapped angle.
// Neither goes near the log map's seam, where every component flips sign at once and the gain collapses.
// Without a twist axis the whole misalignment is the plain rotation vector, which does reach that seam.
// Both forms have the same first-order derivative, so the Jacobians are the frame's axes either way.
static float3 AngularError(float4 relative, uint twist_axis, float unwrapped) {
    if (twist_axis > 2) return RotationVector(relative);
    const float3 axis = UnitAxis(twist_axis);
    float3 error = RotationVector(QuatMul(relative, QuatConjugate(TwistPart(relative, axis))));
    error[twist_axis] = unwrapped; // exactly where the swing has nothing, its axis being perpendicular to this one
    return error;
}

// And that error at whatever pose an iteration has reached.
// The twist is unwrapped against the one the step began with rather than advanced.
// Only PrepareJoints runs once a step, and primal and dual must read one value.
static float3 JointAngularError(Joint joint, float4 frame_a, float4 frame_b) {
    const float4 relative = RelativeFrame(frame_a, frame_b);
    const uint twist_axis = TwistAxis(joint.AngularModes);
    const float unwrapped = twist_axis <= 2 ? TwistAngle(relative, UnitAxis(twist_axis), joint.Twist) : 0;
    return AngularError(relative, twist_axis, unwrapped);
}

// Everything a joint's six rows are measured from at the pose an iteration has reached.
// The frame on B, the reach between the anchors, the angular error about those axes, and how far the pair has turned since the step began.
// Primal and dual must agree here.
struct JointMeasure {
    float4 FrameB;
    float3 Reach, Error, Turned;
};

static JointMeasure MeasureJoint(Joint joint, Pose a, Pose b, device const Pose *initial) {
    const float4 frame_b = JointFrame(b.Orientation, joint.FrameB);
    return {frame_b,
            WorldPoint(a, joint.AnchorA) - WorldPoint(b, joint.AnchorB),
            JointAngularError(joint, JointFrame(a.Orientation, joint.FrameA), frame_b),
            RotationVector(QuatMul(a.Orientation, QuatConjugate(initial[joint.BodyA].Orientation))) -
                RotationVector(QuatMul(b.Orientation, QuatConjugate(initial[joint.BodyB].Orientation)))};
}

// The configuration of one axis of a joint, in its row's units: metres and newtons for a linear axis, radians and newton metres for an angular one.
// One struct, because the two are the same row twice over with only the Jacobian differing.
// The last four fields are where the row stands this iteration.
struct AxisSetup {
    uint Mode;
    float Stiffness, Damping, Speed, Target, MaxForce, Low, High;
    float3 Axis; // the direction the row acts along or about, in world
    // `Value` is the row's coordinate now and `Began` its value when the step began.
    // `Moved` is how far the bodies travelled along the row since, which differs from the error.
    // A motor turning for ever wraps its error at half a turn but never its travel.
    float Value, Began, Moved;
};

// One of the six rows: three linear axes along the frame, then three angular ones about it.
static AxisSetup JointRowAt(Joint joint, JointMeasure measured, uint row) {
    const uint r = row % 3;
    const bool linear = row < 3;
    AxisSetup setup = linear
        ? AxisSetup{AxisMode(joint.LinearModes, r), joint.LinearStiffness[r], joint.LinearDamping[r], joint.LinearMotorSpeed[r],
                    joint.LinearMotorTarget[r], joint.LinearMotorMaxForce[r], joint.LinearLimitLow[r], joint.LinearLimitHigh[r]}
        : AxisSetup{AxisMode(joint.AngularModes, r), joint.AngularStiffness[r], joint.AngularDamping[r], joint.MotorSpeed[r],
                    joint.MotorTarget[r], joint.MotorMaxTorque[r], joint.LimitLow[r], joint.LimitHigh[r]};
    setup.Axis = Rotate(measured.FrameB, UnitAxis(r));
    setup.Value = linear ? dot(measured.Reach, setup.Axis) : measured.Error[r];
    setup.Began = linear ? joint.C0Linear[r] : joint.C0Angular[r];
    setup.Moved = linear ? setup.Value - setup.Began : dot(measured.Turned, setup.Axis);
    return setup;
}

// The correction one row of a joint requests this iteration, and the bounds on the force it may apply.
// Primal and dual both come through here, because Eq. 16 ramps a row only while strictly inside its bounds.
// A row they disagreed about would be clamped in one and free in the other.
// Returns false where the row holds nothing, which is a limited axis inside its range, handled by the caller.
// A free axis never reaches here.
static bool JointRow(
    AxisSetup axis, float dt, thread float &c, thread float &damped, thread float &low, thread float &high
) {
    // A spring carries its whole extension, so alpha holds nothing back for it.
    const float alpha = IsHard(axis.Stiffness) ? ConstraintAlpha : 0;
    c = axis.Value - axis.Began * alpha;
    // What the damper acts on: how far the row moved this step against the rate asked of it.
    // For every mode but one that rate is zero.
    // A driven row's value is already travel measured against its speed's travel, so a brake damps the whole velocity error.
    damped = axis.Value - axis.Began;
    low = -INFINITY;
    high = INFINITY;
    if (axis.Mode == AxisDriven) {
        // Against the travel its speed makes in a step, the solve moving distances rather than velocities.
        c = axis.Moved - axis.Speed * dt;
        damped = c;
        low = -axis.MaxForce;
        high = axis.MaxForce;
    } else if (axis.Mode == AxisPositioned) {
        // How far the row is from where it is driven to, the same shift a stop makes below.
        // The target is this row's zero, and the rest follows a locked row's rule.
        // A hard row spreads its error by alpha exactly as a lock does, and post-stabilization removes the error a step begins with after velocity is read.
        // Taken whole it is Sec. 3.6's explosive correction.
        // A soft row takes alpha zero and carries its whole extension.
        c -= axis.Target * (1 - alpha);
        low = -axis.MaxForce;
        high = axis.MaxForce;
    } else if (axis.Mode == AxisLimited) {
        // Which stop the row is against is taken from where the step began rather than from where the sweep reached, so a row cannot change sides mid-solve.
        // That is the same discipline that fixes a contact's feature.
        // Outside the range the row is one-sided exactly as a contact's normal row is.
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

// The force a row applies, and the stiffness reported to the block: two Eq. 7 forces plus Eq. 13's dual.
// The viscous one is `rate`, the damping coefficient over the step.
// Backwards Euler on -c dC/dt gives energy (c/2h)(C - C0)^2 with Hessian (c/h) J^T J, so the row's stiffness is penalty + c/h.
// The damper needs no dual and no ramp, its stiffness being exact.
//
// Eq. 14's secant applies where the bounds bind, held in [0, penalty + rate] to keep H definite for an LDL without pivoting.
// Without it a bounded motor asked for a distant angle saturates and reaches half its due rate.
static float RowForce(
    float penalty, float rate, float c, float damped, float lambda, float low, float high, thread float &stiffness
) {
    const float requested = penalty * c + lambda + rate * damped;
    const float force = clamp(requested, low, high);
    stiffness = penalty + rate;
    if (requested != force && abs(c) > 1e-9f) stiffness = clamp((force - lambda) / c, 0.f, penalty + rate);
    return force;
}

// Joints are re-measured every iteration, so a step records only C0 and the penalty decay.
kernel void PrepareJoints(
    device Joint *joints [[buffer(16)]], device const Pose *poses [[buffer(0)]],
    constant StepParams &p [[buffer(7)]], uint index [[thread_position_in_grid]]
) {
    if (index >= p.JointCount) return;
    device Joint &joint = joints[index];
    if (!joint.Active) return;
    const Pose a = poses[joint.BodyA], b = poses[joint.BodyB];
    // Both errors as the step found them, resolved along the joint frame's own three axes.
    const float4 frame_b = JointFrame(b.Orientation, joint.FrameB);
    const float3 reach = WorldPoint(a, joint.AnchorA) - WorldPoint(b, joint.AnchorB);
    for (uint r = 0; r < 3; ++r) joint.C0Linear[r] = dot(reach, Rotate(frame_b, UnitAxis(r)));
    // The one place the accumulated twist advances, this being the only joint kernel that runs once a step.
    const float4 relative = RelativeFrame(JointFrame(a.Orientation, joint.FrameA), frame_b);
    const uint twist_axis = TwistAxis(joint.AngularModes);
    if (twist_axis <= 2) joint.Twist = TwistAngle(relative, UnitAxis(twist_axis), joint.Twist);
    joint.C0Angular = AngularError(relative, twist_axis, joint.Twist);
    // Eq. 19's decay then Eq. 16's cap: a soft row ramps to its material stiffness and no further.
    joint.PenaltyLinear = min(clamp(joint.PenaltyLinear * p.Gamma, p.PenaltyMin, p.PenaltyMax), joint.LinearStiffness);
    joint.PenaltyAngular = min(clamp(joint.PenaltyAngular * p.Gamma, p.PenaltyMin, p.PenaltyMax), joint.AngularStiffness);
    // A soft row carries no dual, so any stale value is cleared here.
    for (uint r = 0; r < 3; ++r) {
        if (!IsHard(joint.LinearStiffness[r])) joint.LambdaLinear[r] = 0;
        if (!IsHard(joint.AngularStiffness[r])) joint.LambdaAngular[r] = 0;
    }
}

// Eqs. 11 and 16 for joints.
kernel void UpdateJointDuals(
    device Joint *joints [[buffer(16)]], device const Pose *poses [[buffer(0)]],
    device const Pose *initial [[buffer(1)]], constant StepParams &p [[buffer(7)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= p.JointCount) return;
    device Joint &joint = joints[index];
    if (!joint.Active) return;
    const Joint state = joint; // the rows below write only the dual and the penalty, neither of which any of them reads
    const JointMeasure measured = MeasureJoint(state, poses[joint.BodyA], poses[joint.BodyB], initial);

    for (uint row = 0; row < 6; ++row) {
        const uint r = row % 3;
        const bool linear = row < 3;
        const AxisSetup setup = JointRowAt(state, measured, row);
        // Read out and written back at the end, MSL having no reference over two device lanes.
        float lambda = linear ? state.LambdaLinear[r] : state.LambdaAngular[r];
        float penalty = linear ? state.PenaltyLinear[r] : state.PenaltyAngular[r];
        float c, damped, low, high;
        // A free row and a row off its stops hold nothing, so neither carries a dual forward.
        if (setup.Mode == AxisFree || !JointRow(setup, p.DeltaTime, c, damped, low, high)) {
            lambda = 0;
        } else if (!IsHard(setup.Stiffness)) {
            // Eq. 16's other branch and Algorithm 1 line 33: no dual, and the ramp stops at the material stiffness.
            // Ramping up to it rather than starting there keeps the stiffness ratio on a body small in the early iterations (Sec. 3.4).
            penalty = min(penalty + p.Beta * abs(c), setup.Stiffness);
        } else {
            // The dual is the elastic half alone, clamped in the shifted frame. See RowForce.
            const float damping_force = setup.Damping / p.DeltaTime * damped;
            const float requested = penalty * c + lambda + damping_force;
            lambda = clamp(requested, low, high) - damping_force;
            // Ramps only while strictly inside the bounds, tested against the requested force.
            if (requested > low && requested < high) penalty = min(penalty + p.Beta * abs(c), p.PenaltyMax);
        }
        if (linear) {
            joint.LambdaLinear[r] = lambda;
            joint.PenaltyLinear[r] = penalty;
        } else {
            joint.LambdaAngular[r] = lambda;
            joint.PenaltyAngular[r] = penalty;
        }
    }
}

// Counts how many contacts name each body as B. Integer atomics, so the counts do not depend on order.
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

// Turns those counts into the start of each body's run, as a serial scan once a step.
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

// Sorted back into slot order after the scatter, so the list does not depend on the order FillIncoming's atomic handed out cursors in.
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

// The separation the normal row measures from, before this step's motion.
// A contact generated while the bodies were apart has a gap to consume first, and that slack is the separation alone.
// C0's normal row is separation plus ContactMargin, and the margin is the depth a contact rests at rather than slack.
// Carrying the margin here too would step by a whole margin as a contact crosses from apart to touching.
// Post-stabilization keeps the whole of C0.
static float NormalOffset(Contact contact, constant StepParams &p) {
#if STABILIZE
    return contact.C0[0];
#else
    return max(contact.C0[0] - p.ContactMargin, 0.f);
#endif
}

// Eq. 15's constraint in the contact basis: the separation the step began with, plus the two bodies' displacement since.
// `arm_a` and `arm_b` are the anchors turned by the pose the step began from, so the Jacobian is the fixed one.
// Primal and dual both come through here and must agree exactly, or Eq. 16 ramps against a constraint nothing is solving.
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

// The requested contact force, clamped: row 0 only pushes and rows 1 and 2 stay inside the cone.
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

// And the body at the other end of a joint, NoIndex where the joint is not this body's or the far end cannot move.
static Index JointPartner(Joint joint, uint body, device const BodyMass *masses) {
    if (!joint.Active || (joint.BodyA != body && joint.BodyB != body)) return NoIndex;
    const Index other = joint.BodyA == body ? joint.BodyB : joint.BodyA;
    return Moves(masses[other]) ? other : NoIndex;
}

// Walks a body's contacts: its own run, where it is A, then the list gathered this step, where it is B.
// The two together cover all of them without reading another body's slots.
// Returns NoIndex where there is nothing at `i`.
// The own run is dense from zero, so its first inactive slot ends it and `i` skips the rest.
static uint ContactSlot(thread uint &i, uint own, uint start, device const uint *incoming_slots, device const Contact *contacts) {
    const uint slot = i < ContactsPerBody ? own + i : incoming_slots[start + i - ContactsPerBody];
    if (contacts[slot].Active) return slot;
    if (i < ContactsPerBody) i = ContactsPerBody - 1;
    return NoIndex;
}

// The lowest color the mask does not hold, capped at the width of the mask itself.
static uint LowestFree(uint taken) {
    uint at = 0;
    while (at < 31 && (taken & (1u << at)) != 0) ++at;
    return at;
}

// One neighbour of a body being coloured, counted in the first sweep and read for its colour in the
// second, since degree must be whole before priority can be judged. Conflicts are judged against
// prioritized neighbours, and a compacting body moves only to a colour no neighbour holds.
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

// Algorithm 1 line 2, "update colorization".
// Bodies of one color share no constraint, so a color solves in parallel, and the sequence of colors gives the Gauss-Seidel propagation a stack needs.
// The paper's scheme, incremental and allowed to come out imperfect, because two neighbours sharing a color fall back to Jacobi.
//
// Two amendments, both gated on the pair having gone quiet. See Prioritized.
// Priority goes by contact degree before index, Welsh-Powell's ordering, because index order is add order.
// And a quiet body moves down to the lowest free color, which lets a settled coloring improve.
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
    if (!Moves(masses[body])) return;
    // A sleeping body keeps its color, the sweeps skipping it, so that color constrains nothing.
    if (Asleep(quiet[body], p)) return;
    const bool my_quiet = quiet[body] > 0;

    // Two sweeps, because degree must be complete before priority is judged. See NoteNeighbour.
    uint degree = 0, taken = 0, taken_all = 0;
    const Adjacency neighbours = incoming[body];
    for (uint sweep = 0; sweep < 2; ++sweep) {
        for (uint i = 0; i < ContactsPerBody + neighbours.Count; ++i) {
            const uint slot = ContactSlot(i, body * ContactsPerBody, neighbours.Start, incoming_slots, contacts);
            if (slot == NoIndex) continue;
            const Index other = ContactPartner(contacts[slot], body);
            if (!Moves(masses[other])) continue; // the sweeps do not move it, so this body cannot race it
            NoteNeighbour(sweep, other, colors[other], body, my_quiet && quiet[other] > 0, degree, taken, taken_all);
        }
        // A joint couples two bodies as a contact does, and without this a jointed pair shares a color and races.
        for (uint index = 0; index < p.JointCount; ++index) {
            const Index other = JointPartner(joints[index], body, masses);
            if (other == NoIndex) continue;
            NoteNeighbour(sweep, other, colors[other], body, my_quiet && quiet[other] > 0, degree, taken, taken_all);
        }
        degree = min(degree, MaxColorDegree);
    }

    uint chosen = mine;
    if ((taken & (1u << min(mine, 31u))) != 0) {
        // Conflicted: move to the lowest color no prioritized neighbour holds, or stay and solve Jacobi.
        const uint best = LowestFree(taken);
        if (best < p.MaxColors) chosen = best;
    } else if (my_quiet) {
        // Quiet and unconflicted: compact down to a color that is entirely free.
        const uint best = LowestFree(taken_all);
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

// The primal update, Eqs. 4 to 6 and 13: one body per thread, gathering and moving. Nothing scatters.
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
    // The color this pass runs is the slot the cursor was pointed at, modulo the color count.
    if (!Solved(mass, quiet[body], p) || ColorOf(colors[body]) % p.MaxColors != cursor[0]) return;

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
    // An infinite mass or inertia contributes nothing and is dropped from the block at the end. See LockDirection.
    // A rigid axis is held out of the reciprocal on the way in and zeroed on the way out, and never divided by.
    // Fast math may assume no infinity arises, and an ordinary body must come out bit for bit as before.
    const bool3 rigid = mass.InvInertiaLocal == float3(0);
    const float3 heavy = select(1 / select(mass.InvInertiaLocal, float3(1), rigid), float3(0), rigid);
    const float m = Translates(mass) ? 1 / mass.InvMass : 0;
    const float3 offset = pose.Position - target.Position;
    for (uint i = 0; i < 3; ++i) {
        H[i][i] = m * inv_dt2;
        g[i] = m * inv_dt2 * offset[i];
    }
    const float3x3 rotation = QuatToMatrix(pose.Orientation);
    const float3x3 world_inertia = WorldTensor(rotation, heavy);
    const float3 twist = RotationVector(QuatMul(pose.Orientation, QuatConjugate(target.Orientation)));
    const float3 torque = world_inertia * twist * inv_dt2;
    for (uint i = 0; i < 3; ++i) {
        g[3 + i] = torque[i];
        for (uint j = 0; j < 3; ++j) H[3 + i][3 + j] = world_inertia[j][i] * inv_dt2;
    }

    // Every contact this body is party to, at a cost independent of the body count. See ContactSlot.
    const Adjacency neighbours = incoming[body];
    for (uint i = 0; i < ContactsPerBody + neighbours.Count; ++i) {
        const uint slot = ContactSlot(i, body * ContactsPerBody, neighbours.Start, incoming_slots, contacts);
        if (slot == NoIndex) continue;
        const Contact contact = contacts[slot];
        const bool mine_is_a = contact.BodyA == body;

        // Jacobians are taken at the pose the step began from and held fixed across the sweeps.
        // The constraint is then a Taylor series rather than a moving target.
        // C is always A minus B.
        const Pose start_a = mine_is_a ? start : initial[contact.BodyA];
        const Pose start_b = mine_is_a ? initial[contact.BodyB] : start;
        const float3 arm_a = Rotate(start_a.Orientation, contact.AnchorA);
        const float3 arm_b = Rotate(start_b.Orientation, contact.AnchorB);
        const Displacement moved_a = Since(mine_is_a ? pose : poses[contact.BodyA], start_a);
        const Displacement moved_b = Since(mine_is_a ? poses[contact.BodyB] : pose, start_b);
        const float3 arm = mine_is_a ? arm_a : arm_b;
        const float side = mine_is_a ? 1 : -1;
        const ContactBasis basis = MakeContactBasis(contact.Normal);
        const float3 constraint = ContactConstraint(contact, p, basis, arm_a, arm_b, moved_a, moved_b);

        const float3 requested = contact.Penalty * constraint + contact.Lambda;
        const float3 force = ContactForce(requested, contact.Friction);

        // Eq. 14's rescaling, held in [0, penalty] to keep H definite for the LDL. See RowForce.
        float3 stiffness = contact.Penalty;
        for (uint r = 0; r < 3; ++r) {
            if (requested[r] == force[r] || abs(constraint[r]) < 1e-9f) continue;
            stiffness[r] = clamp((force[r] - contact.Lambda[r]) / constraint[r], 0.f, contact.Penalty[r]);
        }

        // The second-order term of Eq. 17 is dropped, as the reference drops it for contacts.
        for (uint r = 0; r < 3; ++r) AddRow(H, g, basis.Axis[r], arm, side, force[r], stiffness[r]);
    }

    // Joints, measured at the pose the sweep has reached and keeping the second-order term contacts drop.
    for (uint index = 0; index < p.JointCount; ++index) {
        const Joint joint = joints[index];
        if (!joint.Active || (joint.BodyA != body && joint.BodyB != body)) continue;
        const bool mine_is_a = joint.BodyA == body;
        const Pose other = poses[mine_is_a ? joint.BodyB : joint.BodyA];
        const Pose a = mine_is_a ? pose : other, b = mine_is_a ? other : pose;
        const float side = mine_is_a ? 1 : -1;
        const float3 arm = Rotate(pose.Orientation, mine_is_a ? joint.AnchorA : joint.AnchorB);
        const BodyMass other_mass = masses[mine_is_a ? joint.BodyB : joint.BodyA];

        // What a hard row is stiffened to in post-stabilization, and only there.
        // That pass alone removes the error a step began with, and an unloaded error has ramped no penalty, so it would sit at PenaltyMin for good.
        // Moving the pair one step's worth costs the pair's own inertial stiffness (Sec. 3.4), in the row's own units.
        //
        // Deliberately not applied to the main iterations, whose ramp from PenaltyMin is Sec. 3.4's device.
        // Starting them at this floor costs a swinging jointed pair 0.5% of its linear momentum against a 0.1% bar, and lets a hinge past the seam come apart.
        const float linear_floor = Stabilizing * PairStiffness(mass.InvMass + other_mass.InvMass, p.DeltaTime);

        // Six rows under one rule. A soft row applies Eq. 7 on its extension, with no dual.
        const JointMeasure measured = MeasureJoint(joint, a, b, initial);
        const float3x3 inverse_inertia = WorldInverseInertia(a.Orientation, masses[joint.BodyA].InvInertiaLocal) +
            WorldInverseInertia(b.Orientation, masses[joint.BodyB].InvInertiaLocal);

        float3 applied{0, 0, 0}; // the sum of the linear rows' forces, which the geometric term is taken from
        for (uint row = 0; row < 6; ++row) {
            const uint r = row % 3;
            const bool is_linear = row < 3;
            const AxisSetup setup = JointRowAt(joint, measured, row);
            if (setup.Mode == AxisFree) continue;
            const float3 axis = setup.Axis;
            float c, damped, low, high;
            if (!JointRow(setup, p.DeltaTime, c, damped, low, high)) continue;

            const bool hard = IsHard(setup.Stiffness);
            // The stabilization floor, in this row's units. See linear_floor above.
            const float floored = is_linear ? linear_floor
                                            : Stabilizing * PairStiffness(dot(axis, inverse_inertia * axis), p.DeltaTime);
            const float held = is_linear ? joint.PenaltyLinear[r] : joint.PenaltyAngular[r];
            const float penalty = hard ? max(held, floored) : held;
            const float lambda = hard ? (is_linear ? joint.LambdaLinear[r] : joint.LambdaAngular[r]) : 0;
            float stiffness;
            const float force = RowForce(penalty, setup.Damping / p.DeltaTime, c, damped, lambda, low, high, stiffness);
            if (is_linear) {
                AddRow(H, g, axis, arm, side, force, stiffness);
                applied += force * axis;
            } else {
                // An angular row touches the angular block alone, so it adds no geometric stiffness.
                for (uint i = 0; i < 3; ++i) {
                    g[3 + i] += side * axis[i] * force;
                    for (uint j = 0; j < 3; ++j) H[3 + i][3 + j] += stiffness * axis[i] * axis[j];
                }
            }
        }

        // Sec. 3.5's geometric stiffness, which the reference keeps for joints and drops for contacts.
        // The arm turns as the body does, so the same force produces a different torque.
        // d(arm x f)/dtheta is arm (x) f - (arm . f) I, the same matrix whatever axes the rows used.
        // It is taken once from their summed force and lumped onto the diagonal by column length.
        const float3x3 geometric = float3x3(arm * applied.x, arm * applied.y, arm * applied.z) - float3x3(dot(arm, applied));
        for (uint i = 0; i < 3; ++i) H[3 + i][3 + i] += length(geometric[i]);
    }

    // And the degrees of freedom this body does not have, now that every row is gathered.
    // An infinite inertia about a body axis locks turning about wherever that axis now points, and KHR's pinned wheel authors (0, 1, 0).
    if (!Translates(mass)) {
        for (uint i = 0; i < 3; ++i) LockDirection(H, g, 0, UnitAxis(i));
    }
    for (uint i = 0; i < 3; ++i) {
        if (mass.InvInertiaLocal[i] == 0) LockDirection(H, g, 3, rotation[i]);
    }

    float step[Dof];
    SolveBlock(H, g, step);
    const float3 linear = float3(step[0], step[1], step[2]), angular = float3(step[3], step[4], step[5]);
    if (!isfinite(linear.x + linear.y + linear.z + angular.x + angular.y + angular.z)) return;
    pose.Position += linear;
    pose.Orientation = normalize(QuatMul(QuatFromRotationVector(angular), pose.Orientation));
    solved[body] = pose;
}

// Publishes a sweep's results.
// A sweep reads one buffer and writes another, and this makes the swap.
// Every body then sees the same snapshot, and the result does not depend on thread completion order.
// It also lets two same-colored neighbours fall back to Jacobi rather than race.
//
// A sleeping body is skipped, nothing having written `solved` for it.
// Publishing it would revert a pose the host wrote between steps, and a teleported sleeping body would snap back.
kernel void PublishPoses(
    device Pose *poses [[buffer(0)]], device const Pose *solved [[buffer(11)]],
    device const BodyMass *masses [[buffer(4)]], device const uint *colors [[buffer(12)]],
    device const uint *cursor [[buffer(14)]], device const uint *quiet [[buffer(21)]],
    constant StepParams &p [[buffer(7)]], uint body [[thread_position_in_grid]]
) {
    if (body >= p.BodyCount || !Solved(masses[body], quiet[body], p)) return;
    if (ColorOf(colors[body]) % p.MaxColors != cursor[0]) return;
    poses[body] = solved[body];
}

// Eqs. 11 and 16: the dual absorbs the sweep's violation and the penalty ramps, unless at a limit.
kernel void UpdateDuals(
    device Contact *contacts [[buffer(5)]], device const Pose *poses [[buffer(0)]],
    device const Pose *initial [[buffer(1)]], device const BodyMass *masses [[buffer(4)]],
    device const uint *quiet [[buffer(21)]], constant StepParams &p [[buffer(7)]],
    uint slot [[thread_position_in_grid]]
) {
    if (slot >= p.BodyCount * ContactsPerBody) return;
    device Contact &contact = contacts[slot];
    if (!contact.Active) return;

    const Index body = contact.BodyA, other_body = contact.BodyB;
    // A row neither side solved has no violation to absorb.
    // The sweeps skip a sleeping body and a body with no mass.
    // Eq. 11 against a displacement one side had while the other went unsolved ramps a dual on an error nothing corrected.
    // A sleeping ball then leaves a kinematic paddle faster than the paddle was going.
    if (!Solved(masses[body], quiet[body], p) && !Solved(masses[other_body], quiet[other_body], p)) return;
    const Pose start = initial[body], other_start = initial[other_body];
    const Displacement own = Since(poses[body], start), other = Since(poses[other_body], other_start);
    const float3 arm = Rotate(start.Orientation, contact.AnchorA);
    const float3 other_arm = Rotate(other_start.Orientation, contact.AnchorB);
    const ContactBasis basis = MakeContactBasis(contact.Normal);
    const float3 c = ContactConstraint(contact, p, basis, arm, other_arm, own, other);

    // The force the rows requested before the cone clamped it.
    // Eq. 16 ramps only while strictly inside the bounds, and the clamp puts a sliding contact exactly on the cone.
    // Testing the clamped force would therefore pass for ever.
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

// Velocity is read from the sweeps' motion, before post-stabilization, so error correction adds no energy.
kernel void Finalize(
    device const Pose *poses [[buffer(0)]], device const Pose *initial [[buffer(1)]],
    device Velocity *velocities [[buffer(3)]], device const BodyMass *masses [[buffer(4)]],
    device const uint *quiet [[buffer(21)]], constant StepParams &p [[buffer(7)]],
    uint body [[thread_position_in_grid]]
) {
    if (body >= p.BodyCount || !Solved(masses[body], quiet[body], p)) return;
    const Displacement moved = Since(poses[body], initial[body]);
    const float inv_dt = 1 / p.DeltaTime;
    velocities[body] = {moved.Linear * inv_dt, moved.Angular * inv_dt};
}

// Restitution, as a velocity pass after the solve rather than as a distance requested by the normal row.
// One displacement per step cannot carry both an approach and a rebound, and Box2D v3 and the XPBD rigid-body paper both use a pass after the solve.
//
// XPBD Eq. 34 with Box2D's gates: a contact bounces only where the solve's normal dual pushed and the approach beat the threshold.
// Contact-parallel, so it computes an impulse and the gather below moves the bodies, with no atomics and symmetric over a manifold's points.
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
    // Eqs. 2 and 3: the change a unit normal impulse makes to the normal speed, which is the effective mass.
    const float3 turn_a = WorldInverseInertia(poses[a].Orientation, mass_a.InvInertiaLocal) * cross(arm_a, normal);
    const float3 turn_b = WorldInverseInertia(poses[b].Orientation, mass_b.InvInertiaLocal) * cross(arm_b, normal);
    const float weight = mass_a.InvMass + mass_b.InvMass +
        dot(normal, cross(turn_a, arm_a)) + dot(normal, cross(turn_b, arm_b));
    if (weight <= 0) return; // neither body can move, which is a pair of static bodies

    const Velocity va = velocities[a], vb = velocities[b];
    const float3 closing = (va.Linear + cross(va.Angular, arm_a)) - (vb.Linear + cross(vb.Angular, arm_b));
    // Positive along the normal is separating, and the pass drives it to e times the speed the step began closing at.
    // Accumulated so the pass can be iterated, and clamped at zero so the total only ever pushes the two apart.
    const float total = max(0.f, contact.BounceImpulse + (restitution * contact.Approach - dot(normal, closing)) / weight);
    contact.BounceDelta = total - contact.BounceImpulse;
    contact.BounceImpulse = total;
}

// And the gather that moves the bodies, exactly as the primal sweep gathers forces.
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
        // Each half takes its own inverse quantity, so a pinned body takes the whole impulse into spin.
        const float3 impulse = contact.Normal * (mine_is_a ? contact.BounceDelta : -contact.BounceDelta);
        linear += impulse * mass.InvMass;
        angular += inverse_inertia * cross(arm, impulse);
    }
    velocities[body].Linear += linear;
    velocities[body].Angular += angular;
}

// Counts how long a body has been still, which decides whether it stops being solved. Runs after restitution.
kernel void CountQuiet(
    device const Pose *poses [[buffer(0)]], device const Velocity *velocities [[buffer(3)]],
    device const BodyMass *masses [[buffer(4)]], device uint *quiet [[buffer(21)]],
    device Pose *rest [[buffer(22)]], constant StepParams &p [[buffer(7)]],
    uint body [[thread_position_in_grid]]
) {
    if (body >= p.BodyCount || !Solved(masses[body], quiet[body], p)) return;
    const Velocity now = velocities[body];
    const bool slow = length(now.Linear) <= p.SleepSpeed && length(now.Angular) <= p.SleepSpeed;
    uint counted = slow ? quiet[body] + 1 : 0;
    if (counted == p.SleepSteps) {
        // The step it would sleep on, where the drift over the window decides, slow motion alone not meaning settled.
        const Displacement since = Since(poses[body], rest[body]);
        if (length(since.Linear) > p.SleepDrift || length(since.Angular) > p.SleepDrift) counted = 0;
    }
    quiet[body] = counted;
    if (counted == 0) rest[body] = poses[body];
}

// No body is quieter than what it is touching, which makes sleeping a property of a group.
// A stack settles and sleeps together, and one link still moving holds all of it awake.
// The whole count spreads rather than a test for zero, because a body reaching the threshold while a neighbour is five steps behind would sleep mid-settle.
// The neighbour would then press on against a body no longer being solved.
// The count spreads one hop per step, which costs nothing.
kernel void SpreadWaking(
    device const uint *quiet [[buffer(21)]], device uint *next [[buffer(23)]],
    device const Contact *contacts [[buffer(5)]], device const Joint *joints [[buffer(16)]],
    device const Adjacency *incoming [[buffer(17)]], device const uint *incoming_slots [[buffer(18)]],
    device const BodyMass *masses [[buffer(4)]], device const Velocity *velocities [[buffer(3)]],
    constant StepParams &p [[buffer(7)]], uint body [[thread_position_in_grid]]
) {
    if (body >= p.BodyCount) return;
    uint least = quiet[body];
    next[body] = least;
    if (!Moves(masses[body]) || least == 0) return;

    const Adjacency neighbours = incoming[body];
    for (uint i = 0; i < ContactsPerBody + neighbours.Count; ++i) {
        const uint slot = ContactSlot(i, body * ContactsPerBody, neighbours.Start, incoming_slots, contacts);
        if (slot == NoIndex) continue;
        const Index other = ContactPartner(contacts[slot], body);
        if (Moves(masses[other])) least = min(least, quiet[other]);
        // A body the solve skips has no quiet count, and its motion still has to wake whatever sleeps on it.
        else if (Driven(masses[other], velocities[other], p)) least = 0;
    }
    for (uint index = 0; index < p.JointCount; ++index) {
        const Index other = JointPartner(joints[index], body, masses);
        if (other != NoIndex) least = min(least, quiet[other]);
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
