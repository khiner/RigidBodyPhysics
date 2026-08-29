// The engine's GPU vocabulary, compiled by both clang and the Metal compiler from this one text, so a
// layout can never drift between host and device. The spellings are the MSL ones and the host aliases
// exist to match them, not the other way round.
//
// Conventions from LiteratureReview.md section 1.1: SI units, Y-up right-handed, transforms centered
// on the center of mass, inertia diagonal in the body frame.

#ifndef RBP_GPU_SHARED_H // a guard rather than #pragma once, since this text is prepended into MSL as the main file
#define RBP_GPU_SHARED_H

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;

#define GPU_CONSTANT constant constexpr // MSL puts program-scope constants in the constant address space

// MSL spells this float4(xyz, w) and simd needs a call, so both sides get the name the shared text uses.
inline float4 MakeFloat4(float3 xyz, float w) { return float4(xyz, w); }
#else
#include <cmath>
#include <cstdint>
#include <simd/simd.h>

using uint = uint32_t;
using uchar = uint8_t;
using float3 = simd::float3; // 16-byte aligned on both sides, which is why the layouts agree
using float4 = simd::float4;
using simd::cross;
using simd::dot;
using simd::length;
using simd::normalize;
using std::abs; // the scalar one, so `abs` in the shared text below means what MSL means by it

#define GPU_CONSTANT constexpr

inline float4 MakeFloat4(float3 xyz, float w) { return simd::make_float4(xyz, w); }
#endif

// An index into a pool. Pools never move their contents, so an index is a stable handle, and it is the
// only kind of reference that crosses to the GPU.
using Index = uint;
GPU_CONSTANT Index NoIndex = ~0u;

// Orientation is a unit quaternion stored (x, y, z, w). MSL has no quaternion type, so both sides
// carry the raw four floats and rotate through Rotate() below.
struct Pose {
    float3 Position;
    float4 Orientation;
};

struct Velocity {
    float3 Linear;
    float3 Angular;
};

// Inverse quantities, because that is what the solve divides by, and a zero inverse mass is exactly
// what makes a body static without a branch.
struct BodyMass {
    float3 InvInertiaLocal; // diagonal of the inverse inertia tensor, in the body frame
    float InvMass;
};

enum ShapeKind : uint {
    ShapeBox,
    ShapePlane,
    ShapeSphere,
    ShapeCapsule,
    ShapeHull,
    ShapeMesh,
};

// The most vertices one hull may have. A support search scans all of them, so this bounds the
// narrowphase's inner loop, and PhysX picks 64 for the same reason. It also makes a vertex index six
// bits, which is what lets a clipped contact point be named by the two hull vertices that made it.
GPU_CONSTANT uint MaxHullVertices = 64;

GPU_CONSTANT uint MaxFacePoints = 8; // per face, which covers a box and most hulls

// One face of a cooked hull: the plane it lies in, and a run of its vertices in the face-vertex pool,
// each an index into the shape's own run of the vertex pool. Wound about the normal and starting at the
// lowest of them, so a face is named by the geometry rather than by the order the cook found it in.
//
// Recovered by the cook because the builder triangulates, and because gathering a face back from
// vertices at query time needs a height tolerance that no value satisfies: loose enough to present a
// whole face when the contact normal is a fraction of a degree off it is also loose enough to swallow a
// separate facet meeting it at a shallow angle. Gregorius (GDC 2015) and Jolt both answer this in the
// cook. Corners are inline and capped, so a wider face is sampled around its rim by the cook - which
// has the whole face in hand, pays once, and keeps a hull to one of Metal's thirty-one bindings.
struct HullFace {
    float3 Normal;
    float Offset;
    uint Count;
    uchar Corner[MaxFacePoints]; // into the shape's own run, which a byte holds at MaxHullVertices
};

// A flat tagged union: each kind reads the fields that mean something to it and leaves the rest zero.
// Box uses HalfExtents. Sphere uses Radius. Capsule uses Radius and HalfExtents.y, the half length of
// the segment its two caps sit on, so it runs along the body's local y and reaches HalfExtents.y +
// Radius from the centre. Plane uses Normal and Offset and is always static: points with
// dot(Normal, p) < Offset are inside it. Hull uses a run of the world's one vertex pool, already in the
// frame World::AddHull cooked it into - centred on the centre of mass and turned onto its principal
// axes, which is what lets a body carry a diagonal inertia.
//
// A mesh adds a run of triangles and the root of the tree over them. It is static geometry and nothing
// else: no inside, so no volume, no centre of mass and no inertia, and a body given one never moves. A
// moving concave shape is a set of hulls, which is what convex decomposition is for.
struct Shape {
    float3 HalfExtents;
    float3 Normal;
    float Offset;
    float Radius;
    Index FirstVertex;
    uint VertexCount;
    Index FirstFace; // a hull's run of the face pool
    uint FaceCount;
    Index FirstTriangle, RootNode; // a mesh's, into the pools below
    // Nothing on the device reads these - a walk starts at the root and a leaf names its own triangles
    // - but a shape that can be removed has to be able to give its runs back.
    uint TriangleCount, NodeCount;
    uint Kind;
};

// One triangle of a mesh, by absolute index into the vertex pool, wound so that the normal of
// (B - A) x (C - A) points out of the surface. Which side is out is the whole of what a mesh says
// about solidity: nothing is ever pushed out of the back of one.
//
// An edge is *active* when it is a feature something can actually hit - where the triangle across it
// folds away, leaving a convex crease. Where two triangles meet flat, or fold towards each other, the
// edge is a seam of the tessellation rather than a shape, and a body sliding over it must not catch on
// it. Baked by the cook, since it is a property of the mesh and never changes.
struct Triangle {
    Index A, B, C;
    uint ActiveEdges; // bit i is edge i, from corner i to corner (i + 1) % 3
    // And which of them this triangle answers for what lies exactly along. A point on an edge is inside
    // both triangles as far as a clip is concerned, so without an owner one piece of geometry gets two
    // rows and two duals. The cook names the lower-numbered of the pair, and an edge nobody shares is
    // its own.
    uint OwnedEdges;
};

// A node of the tree over a mesh's triangles: the box around everything below it, and either a run of
// triangles or a pair of children. Leaves have a nonzero count, and an interior node's two children
// are adjacent, so one index names both.
//
// Plain float bounds and a binary tree, rather than the quantized four-wide nodes LiteratureReview.md
// section 7 recommends: that is a bandwidth argument, and so a measurement rather than a default.
struct BvhNode {
    float3 Low, High;
    Index First; // the first triangle of a leaf, relative to the shape's run, or the left child of a node
    uint Count; // triangles in a leaf, and zero in an interior node
};

// What a body will touch: two bodies collide only when each one's layer is in the other's mask. The
// bitmask scheme KHR_physics_rigid_bodies uses and MeshEditor already speaks.
struct Filter {
    uint Layer; // the bits this body is
    uint Collides; // the bits it will touch
};

// A body's jointed partners, which by default it does not collide with - two bodies a joint already
// holds together will overlap by design, and contacts fighting the joint is not a physical force. A
// fixed run per body with a NoIndex terminator, since a body in a ragdoll has a few and no more.
GPU_CONSTANT uint JointsPerBody = 8;

// A body's colour word carries its colour in the low byte and its contact degree above, so a
// neighbour's colouring priority - degree first, index as the tie - arrives in the load the conflict
// test already does, and no binding is spent on a second per-body lane.
GPU_CONSTANT uint ColorDegreeShift = 8;
GPU_CONSTANT uint MaxColorDegree = 255;
inline uint ColorOf(uint word) { return word & 0xFFu; }
// Degree order only stands between two bodies that have both gone quiet. While either is still moving
// its degree churns step to step, and a priority that follows the churn reshuffles colours
// mid-collapse - every reshuffle is a pass of transient conflicts, and a conflicted pair solves Jacobi
// for the step, which is what the colouring exists to prevent (a 5x3 raft blew itself to 23 m/s this
// way). Index order is churn-proof, so it keeps the loud regions. The quiet gate is symmetric, so the
// pair always agrees on which rule it is under.
inline bool Prioritized(uint other_word, uint other, uint degree, uint body, bool both_quiet) {
    const uint other_degree = other_word >> ColorDegreeShift;
    if (both_quiet && other_degree != degree) return other_degree > degree;
    return other < body;
}

// A body owns a fixed run of contact slots, which is what makes the pool addressable without an atomic
// append: one thread fills its own body's run, in a fixed order over the other bodies. The run is
// budgeted in manifolds, since reduction leaves four points a manifold whatever shape made it, so a
// partner is one manifold. Eight is what RbpScenes' raft measures - a lower box there is touched by the
// plane, four neighbours and four more standing on it - and what the claimed-slot mask has room for.
//
// The run is allocated rather than filled: every partner is collided whether or not the run is already
// full, and the shallowest contact held gives up its place to anything deeper, so going over budget
// loses the contact least worth having rather than whichever body happened to be numbered last. A mesh
// is left uncapped deliberately: it presents one manifold per triangle the body reaches, and a body
// resting across a seam is held by both of the triangles under it.
GPU_CONSTANT uint ManifoldPoints = 4; // Gregorius's four, which is what ReduceManifold leaves
GPU_CONSTANT uint ManifoldsPerBody = 8;
GPU_CONSTANT uint ContactsPerBody = ManifoldPoints * ManifoldsPerBody;

// A contact is written by the thread that owns the pair, which makes it body A, so every contact where
// a body is A sits in that body's own run of slots. The ones where it is B are scattered through other
// bodies' runs, and sweeping the pool for them would cost every body a look at every other body's slots.
// So they are gathered once a step into a list per body: count, scan to offsets, scatter, and sort each
// body's run so the order does not depend on which thread got there first.
struct Adjacency {
    uint Count, Start, Cursor;
};

// One contact point, at the slot its feature owns. Normal points out of body B towards body A, and
// the three constraint rows live in an orthonormal basis built from it: row 0 resists penetration,
// rows 1 and 2 are friction. Row 0's force is never positive, since a contact can only push.
//
// C0 is the separation at the start of the step. The constraint is a truncated Taylor series about
// that pose - C = C0 (1 - alpha) + J dq - rather than a gap re-measured from scratch every sweep,
// which is both cheaper and what keeps the sweeps from fighting each other.
struct Contact {
    float3 AnchorA; // in body A's frame
    float3 AnchorB; // in body B's frame
    float3 Normal; // world, out of B towards A
    float3 C0;
    float3 Lambda; // the force each row is applying, and the dual the solve converges
    float3 Penalty;
    float Friction;
    // How fast the two bodies were closing at this point when the step began, without the restitution
    // or the threshold folded in: whether it bounces is the velocity pass's question rather than the
    // row's, since one displacement per step cannot carry both an approach and a rebound.
    float Approach;
    // What the restitution pass has applied along the normal so far this step, and what its last
    // iteration added - which is what the per-body gather reads. Clamped at zero, so the pass can only
    // ever push the two apart.
    float BounceImpulse, BounceDelta;
    Index BodyA, BodyB;
    uint Feature; // names where the point came from, so next step's point can inherit its dual
    // And which part of body B's shape it came from, which for a mesh is the triangle, and NoIndex for
    // a shape made of one piece. A contact is matched on the pair of this and Feature together.
    Index SubShape;
    uint Stick; // held the friction cone last step, so its anchors are kept for static friction
    uint Active;
};

// The frame a contact's three rows are resolved in: its normal, then the two tangents friction acts
// along. Which pair of tangents comes out does not matter, only that the same normal always produces
// the same pair - a stuck contact's dual is held in this frame from one step to the next, and a basis
// that turned between steps would hand it back arguing along a different direction.
//
// Shared rather than the solve's own, because anything reading a contact's force back out as a vector
// has to resolve it in the frame the solve applied it in. RbpScenes slide reports exactly that, and a
// tool that merely looks like it agrees is worse than no tool.
struct ContactBasis {
    float3 Axis[3];
};

inline ContactBasis MakeContactBasis(float3 normal) {
    float3 tangent = abs(normal.x) > abs(normal.z) ? float3{-normal.y, normal.x, 0} : float3{0, -normal.z, normal.y};
    const float len = length(tangent);
    tangent = len > 1e-8f ? tangent / len : float3{1, 0, 0};
    return {{normal, tangent, cross(normal, tangent)}};
}

// What happened to a contact between one step and the next, which is the distinction warm starting
// already draws: this step's points are matched against last step's by feature, so an unmatched point
// is new and a last-step feature nothing claims is a contact that has ended.
enum ContactEventKind : uint {
    ContactAdded,
    ContactPersisted,
    ContactRemoved,
};

// One such change, named exactly the way the contact itself is named, and deliberately without
// geometry: a removed contact no longer has any, and for the other two the live contact is still in
// body A's run under this feature, so anything geometric is a lookup away.
struct ContactEvent {
    Index BodyA, BodyB;
    uint Feature;
    Index SubShape; // which triangle of a mesh, and NoIndex for every shape that is one piece
    uint Kind;
};

// A body can end a step having reported at most one event for each slot it filled and one for each slot
// it held last step and did not refill, so twice the slot count is exact and the run cannot overflow.
// Each body's thread writes its own run in slot order, so events need no atomic and no sort to come out
// the same on every run.
GPU_CONSTANT uint EventsPerBody = 2 * ContactsPerBody;

// CollectContacts tracks which of last step's slots a point has claimed in one word of bits, and
// reports what nothing claims as gone. Widen the mask before widening the run.
static_assert(ContactsPerBody <= 32, "the claimed-slot mask in CollectContacts is a single uint");

// What a joint does with each of its three angular axes, taken in body B's frame. An axis does exactly
// one of these, which is why every mode shares the same row of dual and penalty.
enum JointAxisMode : uint {
    AxisFree,
    AxisLocked,
    AxisDriven, // turned towards a relative speed, within a torque bound
    AxisPositioned, // turned towards a relative angle, within the same bound
    AxisLimited, // free between two angles and stopped outside them, which is a contact's one-sided row
};

// A joint holds two bodies' anchor points together, and when asked also the rotation between them.
// Unlike a contact it is re-measured at the current pose every iteration rather than expanded once
// about the pose the step began from, since a body hanging off one sweeps an arc the Taylor series does
// not survive. What the step's start records is only the error, which alpha spreads over several steps.
// A hard row also has no force bounds, so nothing is ever clamped and the penalty ramps every iteration.
struct Joint {
    float3 AnchorA, AnchorB; // in each body's frame
    float4 RestRotation; // A's orientation in B's frame, which a locked axis holds it to
    float3 C0Linear, C0Angular;
    float3 LambdaLinear, LambdaAngular; // the force and torque each row is applying
    float3 PenaltyLinear, PenaltyAngular;
    float3 MotorSpeed; // per axis, the relative angular speed a driven one turns towards
    float3 MotorTarget; // and the relative angle a positioned one turns towards, measured from rest
    float3 MotorMaxTorque; // and the most either may use doing so
    float3 LimitLow, LimitHigh; // per axis, the angles a limited one turns between, measured from rest
    // How stiff each row is as a material. Infinite makes it a hard constraint. Finite makes it a
    // spring, which Sec. 3.4 says is a different thing: its penalty ramps to this rather than to
    // PenaltyMax, it carries no dual, and its force is Eq. 7 on the extension it actually has rather
    // than on the error added this step.
    float3 LinearStiffness, AngularStiffness;
    Index BodyA, BodyB;
    uint AxisModes; // three bits per axis, one JointAxisMode each
    uint Active;
    // Whether this joint wrote itself into both bodies' Jointed runs, which is what suppresses contacts
    // between them. Removal has to undo exactly what was written, or clearing an entry this joint never
    // wrote would un-suppress the contacts of another joint between the same pair.
    uint Suppresses;
};

inline uint AxisMode(uint modes, uint axis) { return (modes >> (3 * axis)) & 7u; }

// A row with nothing bounding how stiff it may become is a hard constraint, and only a hard one gets a
// dual, a stabilized constraint and a penalty free to ramp past its material stiffness.
inline bool IsHard(float stiffness) { return isinf(stiffness); }

// Named for the paper's symbols, since the whole point is to be diffable against the references.
struct StepParams {
    float3 Gravity;
    float DeltaTime;
    float Beta; // how fast the penalty ramps per unit of constraint violation
    float Gamma; // how much of the penalty survives into the next step
    float PenaltyMin;
    float PenaltyMax;
    float ContactMargin;
    // The most a pair's contact reach may grow past that margin. See CollectContacts: reach covers a
    // step of motion so a contact exists before the bodies touch, and the clamp is the dial between
    // ghost collisions at speed and crossing when it is set too low. INFINITY leaves it unclamped.
    float MaxContactReach;
    float MaxAngularSpeed; // a body spinning faster than this makes its own contacts meaningless
    float MinBounceSpeed; // slower impacts than this do not bounce, so a settling body can settle
    float SleepSpeed; // a body slower than this everywhere is a candidate for sleeping
    uint SleepSteps; // and sleeps once it has been that slow for this many in a row
    float SleepDrift; // provided it has also actually got nowhere over them
    uint BodyCount;
    uint JointCount;
    uint MaxColors; // a body that cannot find a free colour below this keeps its own, and falls back
                    // to Jacobi against its neighbour, which is what double buffering is there for
};

inline float4 QuatConjugate(float4 q) { return MakeFloat4(-q.xyz, q.w); }

// The rotation vector (axis times angle) a quaternion represents. q and -q are the same rotation, so
// flip to the near side first and the result is always the shortest arc.
inline float3 RotationVector(float4 q) {
    if (q.w < 0) q = -q;
    const float sine = length(q.xyz);
    if (sine < 1e-7f) return 2 * q.xyz; // the small-angle limit of the line below
    return q.xyz * (2 * atan2(sine, q.w) / sine);
}

// The quaternion a rotation vector represents.
inline float4 QuatFromRotationVector(float3 v) {
    const float angle = length(v);
    if (angle < 1e-7f) return normalize(MakeFloat4(0.5f * v, 1));
    return MakeFloat4(v * (sin(0.5f * angle) / angle), cos(0.5f * angle));
}

// Rotate v by the unit quaternion q, without materializing a matrix.
inline float3 Rotate(float4 q, float3 v) {
    const float3 axis = q.xyz;
    return v + 2 * cross(axis, cross(axis, v) + q.w * v);
}

inline float4 QuatMul(float4 a, float4 b) {
    return MakeFloat4(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz), a.w * b.w - dot(a.xyz, b.xyz));
}

#endif // RBP_GPU_SHARED_H
