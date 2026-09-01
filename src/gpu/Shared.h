// The engine's GPU vocabulary, compiled by both clang and the Metal compiler from this one text, so host and device layouts cannot drift apart.
// The spellings are the MSL ones.
//
// Conventions from LiteratureReview.md section 1.1: SI units, Y-up right-handed, transforms centered on the center of mass, inertia diagonal in the body frame.

#ifndef RBP_GPU_SHARED_H // a guard rather than #pragma once, since this text is prepended into MSL as the main file
#define RBP_GPU_SHARED_H

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;

#define GPU_CONSTANT constant constexpr // MSL puts program-scope constants in the constant address space
#else
#include <cmath>
#include <cstdint>
#include <simd/simd.h>

#define GPU_CONSTANT constexpr
#endif

// One namespace, so common names like Contact, Shape and World do not collide with a host's.
// Kernels use them unqualified through the using-directive at the bottom of this file.
namespace rbp {

#ifdef __METAL_VERSION__
// MSL spells this float4(xyz, w) and simd needs a call, so both sides define one name for the shared text below.
inline float4 MakeFloat4(float3 xyz, float w) { return float4(xyz, w); }
#else
using uint = uint32_t;
using uchar = uint8_t;
using float3 = simd::float3; // 16-byte aligned on both sides, so the layouts agree
using float4 = simd::float4;
using simd::cross;
using simd::dot;
using simd::length;
using simd::normalize;
using std::abs; // the scalar overload, matching MSL's `abs` in the shared text below

inline float4 MakeFloat4(float3 xyz, float w) { return simd::make_float4(xyz, w); }
#endif

// An index into a pool.
// Pools never move their contents, so an index is a stable handle, and it is the only kind of reference that crosses to the GPU.
using Index = uint;
GPU_CONSTANT Index NoIndex = ~0u;

// Orientation is a unit quaternion stored (x, y, z, w).
// MSL has no quaternion type, so both sides carry the raw four floats and rotate through Rotate() below.
struct Pose {
    float3 Position;
    float4 Orientation;
};

// At the origin, with no rotation. Named because a zeroed Pose carries a zero quaternion, which is not a rotation.
GPU_CONSTANT Pose IdentityPose{{0, 0, 0}, {0, 0, 0, 1}};

struct Velocity {
    float3 Linear;
    float3 Angular;
};

// Inverse quantities, because the solve divides by them, and zero is exactly infinite: a locked axis, expressed with no branch and no flag.
// The three motion properties are stored here rather than in per-body buffers of their own, all thirty-one Metal bindings being spent.
// They belong to the body rather than to its shape. See World::SetBodyShape.
struct BodyMass {
    float3 InvInertiaLocal; // diagonal of the inverse inertia tensor, in the body frame
    float InvMass;
    // The fraction of the world's gravity applied to this body: 1 is all of it, 0 none, and -1 reverses it.
    // KHR physics calls it motion.gravityFactor.
    float GravityScale = 1;
    // The fraction of linear and angular velocity removed per second, modelling a medium rather than a contact.
    // Zero by default, so an undamped body matches its closed form.
    float LinearDamping = 0, AngularDamping = 0;
};

// Translation and rotation are independent: infinite mass with finite inertia is a body that cannot be pushed and still turns freely.
// KHR physics rigid bodies Sec. 128 gives that meaning to an explicit mass of zero.
// Moves is the test for whether the solve touches a body at all, and it is false only when every inverse is zero.
inline bool Translates(BodyMass mass) { return mass.InvMass > 0; }
inline bool Turns(BodyMass mass) {
    return mass.InvInertiaLocal.x > 0 || mass.InvInertiaLocal.y > 0 || mass.InvInertiaLocal.z > 0;
}
inline bool Moves(BodyMass mass) { return Translates(mass) || Turns(mass); }

enum ShapeKind : uint {
    ShapeBox,
    ShapePlane,
    ShapeSphere,
    ShapeCapsule,
    ShapeHull,
    ShapeMesh,
    ShapeCompound,
};

// The most pieces one body may be made of, matching KHR's body of a motion and several collider descendants.
// Eight is the number of run-descriptor fields in Shape, and the range of the three bits a contact's name spends on a child.
// More than that is a convex decomposition, which would need a pool of its own. See Shape.
GPU_CONSTANT uint ChildrenPerCompound = 8;

// The most vertices one hull may have.
// A support search scans all of them, so this bounds the narrowphase's inner loop, and PhysX picks 64 for the same reason.
// It also fits a vertex index in six bits, so a clipped contact point is named by the two hull vertices that produced it.
GPU_CONSTANT uint MaxHullVertices = 64;

GPU_CONSTANT uint MaxFacePoints = 8; // corners per face, covering a box and most hulls

// One face of a cooked hull: its plane, and a run of corners in the face-vertex pool, each an index into the shape's own run of the vertex pool.
// Wound about the normal and starting at the lowest corner index, so a face is named by its geometry rather than by the order the cook produced it in.
//
// Faces are merged in the cook, as Gregorius (GDC 2015) and Jolt both do, because gathering a face at query time needs a height tolerance no value satisfies.
// A tolerance loose enough for a whole face when the normal is a fraction of a degree off also admits a facet meeting it shallowly.
// Corners are inline and capped, so the cook samples a wider face around its rim.
struct HullFace {
    float3 Normal;
    float Offset;
    uint Count;
    uchar Corner[MaxFacePoints]; // into the shape's own run, which a byte holds at MaxHullVertices
};

// A flat tagged union: each kind reads the fields that apply to it and leaves the rest zero.
// Box uses HalfExtents.
// Sphere uses Radius.
// Capsule uses Radius and HalfExtents.y, the half length of the segment its two caps sit on.
// A capsule runs along the body's local y and reaches HalfExtents.y + Radius from the centre.
// Plane uses Normal and Offset and is always static, with points at dot(Normal, p) < Offset inside it.
// Hull uses a run of the world's one vertex pool, in the frame World::AddHull cooked it into.
// That frame is centred on the centre of mass and turned onto the principal axes, which gives the body a diagonal inertia.
//
// A mesh adds a run of triangles and the root of the tree over them.
// It is one-sided and carries no mass properties, so a moving body wearing one requires an explicit mass from the host (BodyDesc::Mass).
// Jolt, Havok and the KHR reference loader all require the same.
// One-sided means nothing is ever pushed out of the back of a triangle.
// A mesh presents no manifold of its own, so mesh against mesh and mesh against plane produce no contact.
// A moving solid concave shape is a set of hulls instead.
//
// A compound is several of the kinds above, each at its own Local, and is flat with convex leaves only.
// A child that is itself a compound, a mesh or a plane is refused.
// World::AddCompound re-expresses each child's pose in the body frame, so a leaf's Local reads exactly as a lone shape's does.
// The children are shape indices held in the eight run-descriptor fields below, FirstVertex through NodeCount, because a compound's run is its children.
// Read them through ChildOf, never by field name.
// The run carries no count and ends at a NoIndex, as a body's Jointed run does.
// A children pool would cost a buffer binding, and all thirty-one are spent.
// FirstTriangle carries one further use of the same fields. See InternalFaces.
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
    // Unread on the device, and required so removing a shape can return its runs to the pools.
    uint TriangleCount, NodeCount;
    uint Kind;
    // The pose of the shape's own frame within the body frame, which is centred on the centre of mass with the inertia diagonal.
    // The narrowphase composes body pose with this only where it reads the geometry, so anchors, features and manifold points stay in the body frame.
    // A contact on an offset shape therefore has anchors carrying the offset.
    // A plane leaves this unread, its Normal and Offset being world quantities rather than geometry in a frame.
    //
    // Mass properties are the shape's own, about its own centre. See MassProperties.
    // A body wearing a shape at any other pose needs an explicit mass.
    // Moving the tensor onto the shifted, turned origin and diagonalizing it is the host's arithmetic.
    // World::AddBody refuses such a body without one.
    Pose Local = IdentityPose;
};

// Child `i` of a compound, or NoIndex past the end of its run.
// The eight run-descriptor fields, in declaration order.
inline Index ChildOf(Shape shape, uint i) {
    switch (i) {
        case 0: return shape.FirstVertex;
        case 1: return shape.VertexCount;
        case 2: return shape.FirstFace;
        case 3: return shape.FaceCount;
        case 4: return shape.FirstTriangle;
        case 5: return shape.RootNode;
        case 6: return shape.TriangleCount;
        default: return shape.NodeCount;
    }
}

// Which of a child's own faces are buried against a sibling, one bit each, and zero for every shape that is not a compound's child.
// Two coplanar siblings share a face inside the solid: a floor made of two boxes, or a leg's top under a slab.
// A separating axis is a direction a body can be pushed along, so a face left unmarked here acts as a wall at the join.
// A mesh's inactive edge marks the same thing.
//
// A bit is indexed by the face's own name: BoxFaceIndex below for a box, and the face's place in the shape's run of the face pool for a hull.
// A face past the thirty-two a word holds is left unmarked, because a wrong mark is worse than the wall.
// Stored in FirstTriangle, and read through this function rather than by field name.
GPU_CONSTANT uint MaxInternalFaces = 32;
inline uint InternalFaces(Shape shape) {
    return shape.Kind == ShapeBox || shape.Kind == ShapeHull ? shape.FirstTriangle : 0u;
}

// A box's six faces, indexed by axis and then by end, on host and device alike.
inline uint BoxFaceIndex(uint axis, bool positive) { return 2 * axis + (positive ? 1u : 0u); }

#ifndef __METAL_VERSION__
inline void SetInternalFaces(Shape &shape, uint mask) { shape.FirstTriangle = mask; }

// The host's half of the same mapping, beside ChildOf so the two cannot drift apart.
inline void SetChild(Shape &shape, uint i, Index child) {
    switch (i) {
        case 0: shape.FirstVertex = child; break;
        case 1: shape.VertexCount = child; break;
        case 2: shape.FirstFace = child; break;
        case 3: shape.FaceCount = child; break;
        case 4: shape.FirstTriangle = child; break;
        case 5: shape.RootNode = child; break;
        case 6: shape.TriangleCount = child; break;
        default: shape.NodeCount = child; break;
    }
}
#endif

// One triangle of a mesh, by absolute index into the vertex pool, wound so that (B - A) x (C - A) points out of the surface.
// That winding is the only solidity a mesh carries.
//
// An edge is active where the triangle across it folds away, leaving a convex crease a body can strike.
// Where two triangles meet flat, or fold towards each other, the edge is a seam of the tessellation, and a body sliding over it passes without catching.
// Computed in the cook, because it never changes.
struct Triangle {
    Index A, B, C;
    uint ActiveEdges; // bit i is edge i, from corner i to corner (i + 1) % 3
    // Which of those edges this triangle owns contact points on.
    // A point exactly on an edge clips inside both triangles, and a single owner keeps it from taking two rows and two duals.
    uint OwnedEdges;
};

// A node of the tree over a mesh's triangles: the box around everything below it, and either a run of triangles or a pair of children.
// A leaf has a nonzero count, and an interior node's two children are adjacent, so one index names both.
// Plain float bounds and a binary tree, rather than the quantized four-wide nodes of LiteratureReview.md section 7.
// That case is a bandwidth argument, so adopting it needs a measurement.
struct BvhNode {
    float3 Low, High;
    Index First; // the first triangle of a leaf, relative to the shape's run, or the left child of a node
    uint Count; // triangles in a leaf, and zero in an interior node
};

// Two bodies collide only when each one's layer is in the other's mask.
// The bitmask scheme of KHR_physics_rigid_bodies, which MeshEditor already uses.
struct Filter {
    uint Layer; // the bits this body belongs to
    uint Collides; // the bits it collides with
};

// A body's jointed partners, which by default it does not collide with, because two bodies a joint holds together overlap by design.
// A fixed run per body, terminated by a NoIndex.
GPU_CONSTANT uint JointsPerBody = 8;

// A body's color word holds its color in the low byte and its contact degree above.
// One load therefore supplies both the conflict test and the neighbour's coloring priority, which is degree first and index as the tie.
GPU_CONSTANT uint ColorDegreeShift = 8;
GPU_CONSTANT uint MaxColorDegree = 255;
inline uint ColorOf(uint word) { return word & 0xFFu; }
// Degree order applies only between two bodies that have both gone quiet.
// While either is still moving its degree changes step to step, and colors reshuffled mid-collapse leave conflicted pairs solving Jacobi.
// The coloring exists to prevent that.
// Index order is stable under the churn, so it governs the moving regions.
// The quiet gate is symmetric, so both bodies of a pair select the same rule.
inline bool Prioritized(uint other_word, uint other, uint degree, uint body, bool both_quiet) {
    const uint other_degree = other_word >> ColorDegreeShift;
    if (both_quiet && other_degree != degree) return other_degree > degree;
    return other < body;
}

// Each body has a fixed run of contact slots, so the pool is addressable without an atomic append.
// One thread fills its own body's run, in a fixed order over the other bodies.
// The budget is in manifolds, because reduction leaves four points per manifold whatever shape produced it.
// Ten manifolds cover the busiest measured scene, which puts 33 points on one body, with a whole manifold of headroom.
//
// Every partner is collided whether or not the run is full, and the shallowest contact held is replaced by any deeper one.
// Going over budget therefore drops the shallowest contact rather than the highest-numbered body.
// A mesh is deliberately left uncapped, presenting one manifold per triangle, so a body resting across a seam is held by both triangles.
GPU_CONSTANT uint ManifoldPoints = 4; // Gregorius's four, the count ReduceManifold leaves
GPU_CONSTANT uint ManifoldsPerBody = 10;
GPU_CONSTANT uint ContactsPerBody = ManifoldPoints * ManifoldsPerBody;

// A contact is written by the thread owning the pair, which makes that body A, so the contacts where a body is B are scattered through other bodies' runs.
// They are gathered once a step into a list per body, and each run is sorted so the order does not depend on thread completion order.
struct Adjacency {
    uint Count, Start, Cursor;
};

// One contact point, in the slot its feature owns.
// Normal points out of body B towards body A, and the three constraint rows use an orthonormal basis built from it.
// Row 0 resists penetration and rows 1 and 2 are friction.
// Row 0's force is at most zero, a contact being able only to push.
// C0 is the separation at the start of the step.
// The constraint is a truncated Taylor series about that pose, C = C0 (1 - alpha) + J dq, rather than a gap re-measured every sweep.
struct Contact {
    float3 AnchorA; // in body A's frame
    float3 AnchorB; // in body B's frame
    float3 Normal; // world, out of B towards A
    float3 C0;
    float3 Lambda; // the force each row is applying, and the dual the solve converges
    float3 Penalty;
    float Friction;
    // The closing speed at this point when the step began, with neither restitution nor the threshold folded in.
    // One displacement per step cannot carry both an approach and a rebound, so the velocity pass decides the bounce rather than the row.
    float Approach;
    // The normal impulse the restitution pass has applied so far this step, and the amount its last iteration added.
    // Clamped at zero, so the pass only ever pushes the two apart.
    float BounceImpulse, BounceDelta;
    Index BodyA, BodyB;
    uint Feature; // identifies the geometry the point came from, so next step's point inherits its dual
    // The part of body B's shape it came from: the triangle for a mesh, and NoIndex for a shape of one piece.
    // A contact is matched on this, Feature and Children together.
    Index SubShape;
    // The leaf of each body's shape that produced it: this body's in bits 0-2, the other's in bits 3-5, both zero for a body of one piece.
    // Leaf 3 against leaf 5 is different geometry from leaf 2 against leaf 5, even when both name the same face and corner.
    // The warm start needs this to give each leaf its own dual.
    // Stored here rather than in Feature, which has two bits left.
    uint Children;
    uint Stick; // inside the friction cone last step, so its anchors are kept for static friction
    uint Active;
};

inline uint ChildPair(uint own, uint other) { return own | (other << 3); }
inline uint OwnChild(uint children) { return children & 7u; }
inline uint OtherChild(uint children) { return (children >> 3) & 7u; }

// The frame a contact's three rows resolve in: its normal, then the two tangents friction acts along.
// The particular tangent pair is arbitrary, but one normal must always give the same pair: a stuck contact's dual is carried in this frame between steps.
// Shared rather than private to the solve, because reading a contact's force back out as a vector requires the frame it was applied in.
struct ContactBasis {
    float3 Axis[3];
};

inline ContactBasis MakeContactBasis(float3 normal) {
    float3 tangent = abs(normal.x) > abs(normal.z) ? float3{-normal.y, normal.x, 0} : float3{0, -normal.z, normal.y};
    const float len = length(tangent);
    tangent = len > 1e-8f ? tangent / len : float3{1, 0, 0};
    return {{normal, tangent, cross(normal, tangent)}};
}

// A contact's change between one step and the next, taken from the match warm starting already performs.
// This step's points are matched against last step's by feature, so an unmatched new point is added and an unmatched last-step feature is removed.
enum ContactEventKind : uint {
    ContactAdded,
    ContactPersisted,
    ContactRemoved,
};

// One such change, named the way the contact itself is named, and deliberately without geometry.
// A removed contact has none, and for the other two kinds the live contact is one lookup away.
struct ContactEvent {
    Index BodyA, BodyB;
    uint Feature;
    Index SubShape; // which triangle of a mesh, and NoIndex for every shape that is one piece
    uint Children; // and which leaf of each compound, packed as Contact::Children is
    uint Kind;
};

// A body reports at most one event per slot it filled, plus one per slot it held last step and did not refill, so twice the slot count is an exact bound.
// Each body's thread writes its own run in slot order, so events need no atomic and no sort to come out identical on every run.
GPU_CONSTANT uint EventsPerBody = 2 * ContactsPerBody;

// CollectContacts tracks which of last step's slots have been claimed, in one word of bits.
// Widen the mask before widening the run.
static_assert(ContactsPerBody <= 64, "the claimed-slot mask in CollectContacts is a single ulong");

// The mode of one of a joint's six axes, three linear and three angular, taken in the joint's own frame.
// An axis is in exactly one mode, so every mode shares the same row of dual and penalty.
// Linear and angular are the same row, in metres and newtons against radians and newton metres.
enum JointAxisMode : uint {
    AxisFree,
    AxisLocked,
    AxisDriven, // moved towards a relative speed, within a force bound
    AxisPositioned, // moved towards a relative offset or angle, within the same bound
    AxisLimited, // free between two stops and held outside them, as a contact's one-sided row
};

// A joint holds two bodies' anchor points together, and where configured the rotation between them as well.
// Unlike a contact it is re-measured at the current pose every iteration rather than expanded once about the pose the step began from.
// A body hanging off a joint sweeps an arc the Taylor series does not survive.
// The start of the step records only the error, which alpha spreads over several steps.
//
// The joint carries a frame of its own rather than reusing body B's axes.
// KHR gives the joint node's world transform as the frame on A and the connected node's as the frame on B.
// Rest is the two frames coinciding, so frames differing at creation are an initial error the joint closes.
//
// The axis modes determine how that rotation resolves into three numbers.
// Exactly one angular axis off Locked is a hinge: the rotation splits into the twist about that axis and the swing left over. See AngularError.
// Twist below accumulates that twist across steps.
// Any other combination has no single axis of rotation, so the whole misalignment stays one rotation vector.
// Its limits are small by construction, which keeps the log map's seam out of reach.
struct Joint {
    float3 AnchorA, AnchorB; // in each body's frame
    float4 FrameA, FrameB; // and the joint's own frame, likewise
    float3 C0Linear, C0Angular;
    // The twist angle a hinge-like joint has turned through since creation, unwrapped against last step's value.
    // A limit at three half-turns therefore means three half-turns rather than folding into the half turn a quaternion can name.
    // Zero at creation, and every angular target is measured from that zero.
    float Twist;
    float3 LambdaLinear, LambdaAngular; // the force and torque each row is applying
    float3 PenaltyLinear, PenaltyAngular;
    float3 MotorSpeed; // per angular axis, the relative angular speed a driven one turns towards
    float3 MotorTarget; // and the relative angle a positioned one turns towards, measured from rest
    float3 MotorMaxTorque; // and the torque bound for both of those modes
    float3 LimitLow, LimitHigh; // per angular axis, the angles a limited one turns between, from rest
    // The same five for the linear axes, in metres and newtons.
    // Separate fields rather than a shared set, because a joint is routinely both at once and the units never mix.
    float3 LinearMotorSpeed, LinearMotorTarget, LinearMotorMaxForce;
    float3 LinearLimitLow, LinearLimitHigh;
    // The material stiffness of each row.
    // Infinite makes the row a hard constraint.
    // Finite makes it a spring, which Sec. 3.4 treats separately: the penalty ramps to this rather than to PenaltyMax, and the row carries no dual.
    // A spring's force is Eq. 7 on the extension the row currently has, rather than on the error added this step.
    // Zero is a brake, leaving only the damper below.
    float3 LinearStiffness, AngularStiffness;
    // The viscous coefficient of each row, in N s/m and N m s/rad.
    // Backwards Euler evaluates the viscous force -c times the row's rate at the end of the step, making it -(c/h) times the distance the row moved.
    // That is a second Eq. 7 force at the known stiffness c/h, acting on the change rather than on the error.
    // It carries no dual and no ramp, its stiffness already being in closed form. See RowForce.
    float3 LinearDamping, AngularDamping;
    Index BodyA, BodyB;
    uint LinearModes, AngularModes; // three bits per axis, one JointAxisMode each
    uint Active;
    // Whether this joint wrote itself into both bodies' Jointed runs, which suppresses contacts between them.
    // Removal must undo exactly what was written, or it lifts the suppression another joint between the same pair installed.
    uint Suppresses;
};

inline uint AxisMode(uint modes, uint axis) { return (modes >> (3 * axis)) & 7u; }

// The angular axis a joint turns about, or 3 when it has none.
// One axis off Locked is the hinge case, and every other combination is measured as a single rotation vector.
// Derived from the modes rather than stored, so changing an axis mode changes this.
inline uint TwistAxis(uint angular_modes) {
    uint found = 3, count = 0;
    for (uint axis = 0; axis < 3; ++axis) {
        if (AxisMode(angular_modes, axis) == AxisLocked) continue;
        found = axis;
        ++count;
    }
    return count == 1 ? found : 3;
}

// A row with unbounded stiffness is a hard constraint.
// Only a hard row gets a dual, a stabilized constraint, and a penalty free to ramp past its material stiffness.
inline bool IsHard(float stiffness) { return isinf(stiffness); }

// Named for the paper's symbols, so the kernels diff against the references.
struct StepParams {
    float3 Gravity;
    float DeltaTime;
    float Beta; // how fast the penalty ramps per unit of constraint violation
    float Gamma; // the fraction of a penalty carried into the next step
    float PenaltyMin;
    float PenaltyMax;
    float ContactMargin;
    // The most a pair's contact reach may grow past that margin, or INFINITY for unclamped.
    // See StepSettings::MaxContactReach.
    float MaxContactReach;
    float MaxAngularSpeed; // a body spinning faster than this has meaningless contacts
    float MinBounceSpeed; // impacts slower than this do not bounce. See StepSettings::BounceSpeedFactor.
    float SleepSpeed; // a body slower than this at every point is a sleep candidate
    uint SleepSteps; // and sleeps after this many consecutive slow steps
    float SleepDrift; // provided it also moved less than this over them
    uint BodyCount;
    uint JointCount;
    // A body with no free color below this keeps its own and solves Jacobi against its neighbour, which the double buffering supports.
    uint MaxColors;
};

inline float4 QuatConjugate(float4 q) { return MakeFloat4(-q.xyz, q.w); }

// The rotation vector (axis times angle) a quaternion represents.
// q and -q are the same rotation, and flipping to the near side first makes the result the shortest arc.
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

// A point in a frame's own space, mapped into the space that frame is posed in, and the inverse.
inline float3 WorldPoint(Pose pose, float3 local) { return pose.Position + Rotate(pose.Orientation, local); }
inline float3 LocalPoint(Pose pose, float3 world) { return Rotate(QuatConjugate(pose.Orientation), world - pose.Position); }

// `inner` expressed in the space `outer` is posed in.
// A shape's Local composed into its body's pose is the pose of the geometry. See World::AddHull.
inline Pose ComposePose(Pose outer, Pose inner) {
    return {WorldPoint(outer, inner.Position), QuatMul(outer.Orientation, inner.Orientation)};
}

} // namespace rbp

#ifdef __METAL_VERSION__
// The kernels are compiled with this text prepended and use these names unqualified.
using namespace rbp;
#endif

#endif // RBP_GPU_SHARED_H
