// Host and device layouts use SI units, right-handed Y-up coordinates and body-frame diagonal inertia.

#ifndef RBP_GPU_SHARED_H // An include guard supports prepending this header to MSL source.
#define RBP_GPU_SHARED_H

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;

#define GPU_CONSTANT constant constexpr
#else
#include <cmath>
#include <cstdint>
#include <simd/simd.h>

#define GPU_CONSTANT constexpr
#endif

namespace rbp {

#ifdef __METAL_VERSION__

inline float4 MakeFloat4(float3 xyz, float w) { return float4(xyz, w); }
#else
using uint = uint32_t;
using uchar = uint8_t;
using ulong = uint64_t;
using float3 = simd::float3;
using float4 = simd::float4;
using simd::cross;
using simd::dot;
using simd::length;
using simd::normalize;
using std::abs;

inline float4 MakeFloat4(float3 xyz, float w) { return simd::make_float4(xyz, w); }
#endif

// Pool index; NoIndex is the absent-value sentinel.
using Index = uint;
GPU_CONSTANT Index NoIndex = ~0u;

// Orientation is a unit quaternion stored (x, y, z, w).
struct Pose {
    float3 Position;
    float4 Orientation;
};

GPU_CONSTANT Pose IdentityPose{{0, 0, 0}, {0, 0, 0, 1}};

struct Velocity {
    float3 Linear;
    float3 Angular;
};

struct SensorFollower {
    Index Sensor, Owner;
    Pose Local;
};

struct Displacement {
    float3 Linear, Angular;
};

struct BodyIterate {
    Pose Current;
    Displacement Moved;
};

// Inverse mass and inertia; zero locks the corresponding degree of freedom.
struct BodyMass {
    float3 InvInertiaLocal; // Inverse inertia is diagonal in the body frame.
    float InvMass;
    // Gravity multiplier: 1 applies gravity, 0 disables it, and -1 reverses it.
    float GravityScale = 1;
    // Fractions of linear and angular velocity removed per second.
    float LinearDamping = 0, AngularDamping = 0;
};

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
    ShapeCylinder,
};

// Both collision masks must accept the pair.
struct CollisionMask {
    uint Layer = ~0u, Collides = ~0u;
};
inline bool Allows(CollisionMask a, CollisionMask b) {
    return (a.Layer & b.Collides) != 0 && (b.Layer & a.Collides) != 0;
}
inline bool SameMask(CollisionMask a, CollisionMask b) { return a.Layer == b.Layer && a.Collides == b.Collides; }

enum CombineMode : uint { CombineAverage,
                          CombineMinimum,
                          CombineMaximum,
                          CombineMultiply,
                          CombineGeometricMean };
struct Material {
    float StaticFriction = 0.5f, DynamicFriction = 0.5f, Restitution = 0;
    uint FrictionCombine = CombineAverage, RestitutionCombine = CombineAverage;
};
inline float Combine(float a, float b, uint mode_a, uint mode_b) {
    const uint mode = mode_a < mode_b ? mode_a : mode_b;
    if (mode == CombineAverage) return (a + b) * 0.5f;
    if (mode == CombineMinimum) return a < b ? a : b;
    if (mode == CombineMaximum) return a > b ? a : b;
    if (mode == CombineMultiply) return a * b;
    return sqrt(a * b);
}

// Hull limits bound support-query and clipping scratch storage.
GPU_CONSTANT uint MaxHullVertices = 64;

GPU_CONSTANT uint MaxFacePoints = 8;

// Cooked hull face with shape-local vertex indices, wound around its outward normal.
struct HullFace {
    float3 Normal;
    float Offset;
    uint Count;
    uchar Corner[MaxFacePoints];
};

struct BodyBounds {
    float3 Low, High;
};

struct BroadPhaseNode {
    BodyBounds Bounds;
    uint Left, Right, Parent, MinBody, MaxBody, Ready, Errors, Candidates;
};
struct MortonKey {
    uint Code, Body;
};
GPU_CONSTANT uint RadixBlockSize = 256;
GPU_CONSTANT uint RadixBins = 256;
GPU_CONSTANT uint RadixSimdWidth = 32;
GPU_CONSTANT uint CollisionLanes = 64;
GPU_CONSTANT uint SolveLanes = 32;
GPU_CONSTANT uint SolveDof = 6;
GPU_CONSTANT uint SolveBodiesPerGroup = SolveLanes / SolveDof;
GPU_CONSTANT uint WideSolveBodyLimit = 512;
inline uint FallbackBodiesPerGroup(uint bodies) { return bodies <= WideSolveBodyLimit ? 1 : SolveBodiesPerGroup; }
GPU_CONSTANT uint SmallSolveWaves = 8;
// Jacobi impulse sweeps distribute the approach speed across a manifold.
GPU_CONSTANT uint RestitutionPasses = 4;
GPU_CONSTANT uint RadixWaves = RadixBlockSize / RadixSimdWidth;
inline uint RadixBlocks(uint bodies) { return (bodies + RadixBlockSize - 1) / RadixBlockSize; }
inline uint BroadPhaseRoot(uint bodies) { return bodies == 1 ? 0 : bodies; }

#ifdef __METAL_VERSION__
inline bool BoundsOverlap(BodyBounds a, BodyBounds b) {
    return all(a.Low <= a.High) && all(b.Low <= b.High) && !any(a.Low > b.High) && !any(b.Low > a.High);
}

// Return the first candidate index at least first, or NoIndex.
template<uint Mode = 0>
inline uint NextBodyCandidate(device BroadPhaseNode *nodes, uint bodies, uint body, uint first) {
    if (first >= bodies) return NoIndex;
    if (Mode == 1 || (Mode == 0 && bodies <= RadixSimdWidth)) {
        const uint candidates = nodes[body].Candidates & (~0u << first);
        return candidates ? ctz(candidates) : NoIndex;
    }
    const uint root = BroadPhaseRoot(bodies);
    const BodyBounds query = nodes[body].Bounds;
    uint best = bodies, at = root, from = NoIndex;
    for (uint visited = 0; at != NoIndex && visited < 4 * bodies; ++visited) {
        const BroadPhaseNode node = nodes[at];
        const bool possible = node.MinBody < best && node.MaxBody >= first && BoundsOverlap(query, node.Bounds);
        uint next = node.Parent;
        if (from == node.Parent && possible) {
            if (node.Left == NoIndex) {
                if (node.MinBody != body && node.MinBody >= first) best = node.MinBody;
            } else next = node.Left;
        } else if (from == node.Left && possible) next = node.Right;
        from = at;
        at = next;
    }
    if (at != NoIndex) atomic_fetch_or_explicit((device atomic_uint *)&nodes[root].Errors, 2u, memory_order_relaxed);
    return best < bodies ? best : NoIndex;
}
GPU_CONSTANT uint BodyCandidateCapacity = 4;
struct BodyCandidateCursor {
    uint Values[BodyCandidateCapacity];
    uint Count, At, First;
};

// Return the next candidate index in ascending order, or NoIndex when exhausted.
template<uint Mode = 0>
inline uint NextBodyCandidateBatch(device BroadPhaseNode *nodes, uint bodies, uint body, thread BodyCandidateCursor &cursor) {
    if (Mode == 1 || (Mode == 0 && bodies <= RadixSimdWidth)) {
        const uint next = NextBodyCandidate<1>(nodes, bodies, body, cursor.First);
        cursor.First = next == NoIndex ? bodies : next + 1;
        return next;
    }
    if (cursor.At == cursor.Count) {
        if (cursor.First >= bodies) return NoIndex;
        cursor.Count = cursor.At = 0;
        const uint root = BroadPhaseRoot(bodies);
        const BodyBounds query = nodes[body].Bounds;
        uint at = root, from = NoIndex;
        for (uint visited = 0; at != NoIndex && visited < 4 * bodies; ++visited) {
            const BroadPhaseNode node = nodes[at];
            const uint best = cursor.Count == BodyCandidateCapacity ? cursor.Values[BodyCandidateCapacity - 1] : bodies;
            const bool possible = node.MinBody < best && node.MaxBody >= cursor.First && BoundsOverlap(query, node.Bounds);
            uint next = node.Parent;
            if (from == node.Parent && possible) {
                if (node.Left == NoIndex) {
                    if (node.MinBody != body && node.MinBody >= cursor.First) {
                        uint insert = min(cursor.Count, BodyCandidateCapacity - 1);
                        cursor.Count = min(cursor.Count + 1, BodyCandidateCapacity);
                        while (insert && cursor.Values[insert - 1] > node.MinBody) {
                            cursor.Values[insert] = cursor.Values[insert - 1];
                            --insert;
                        }
                        cursor.Values[insert] = node.MinBody;
                    }
                } else next = node.Left;
            } else if (from == node.Left && possible) next = node.Right;
            from = at;
            at = next;
        }
        if (at != NoIndex) atomic_fetch_or_explicit((device atomic_uint *)&nodes[root].Errors, 2u, memory_order_relaxed);
        cursor.First = cursor.Count < BodyCandidateCapacity ? bodies : cursor.Values[BodyCandidateCapacity - 1] + 1;
    }
    return cursor.At < cursor.Count ? cursor.Values[cursor.At++] : NoIndex;
}

#endif

// ShapeKind selects the active fields.
// Boxes use HalfExtents; spheres use Radius; capsules use Radius and Y half-segment length.
// Cylinder: X radius and Y half-height in HalfExtents; Radius is zero.
// Plane: dot(Normal, p) < Offset is inside; positive X/Z half-extents bound its footprint.
// Bounded planes use local Normal (0, +/-1, 0).
// Hulls use vertex and face ranges; meshes use vertex, triangle and BVH ranges.
// Compounds: FirstVertex and VertexCount index a flat run of child shape indices.
struct Shape {
    float3 HalfExtents;
    float3 Normal;
    float Offset;
    float Radius;
    Index FirstVertex;
    uint VertexCount;
    Index FirstFace;
    uint FaceCount;
    Index FirstTriangle, RootNode;

    uint TriangleCount, NodeCount;
    uint Kind;
    // Collider-local pose relative to the body's center-of-mass frame.
    // Non-identity poses on movable bodies require authored body-frame mass properties.
    Pose Local = IdentityPose;
    Material Surface{};
    uint HasMaterial = 0;
    CollisionMask Mask{};
    uint HasFilter = 0;
    ulong UserData = 0;
    uint DoubleSided = 0; // Planes and meshes accept contacts on both sides.
};

inline bool IsBoundedPlane(Shape shape) { return shape.Kind == ShapePlane && (shape.HalfExtents.x > 0 || shape.HalfExtents.z > 0); }
inline CollisionMask ResolveFilter(Shape shape, CollisionMask inherited) { return shape.HasFilter ? shape.Mask : inherited; }

#ifdef __METAL_VERSION__
inline Index ChildOf(Shape shape, uint i, device const Index *children) {
#else
inline Index ChildOf(Shape shape, uint i, const Index *children) {
#endif
    return i < shape.VertexCount ? children[shape.FirstVertex + i] : NoIndex;
}

// Internal face bits stored in FirstTriangle: box face index, hull face ordinal, or cylinder cap 0/1.
// Faces beyond the 32-bit mask remain unmarked.
GPU_CONSTANT uint MaxInternalFaces = 32;
inline uint InternalFaces(Shape shape) {
    return shape.Kind == ShapeBox || shape.Kind == ShapeHull || shape.Kind == ShapeCylinder ? shape.FirstTriangle : 0u;
}

inline uint BoxFaceIndex(uint axis, bool positive) { return 2 * axis + (positive ? 1u : 0u); }

#ifndef __METAL_VERSION__
inline void SetInternalFaces(Shape &shape, uint mask) { shape.FirstTriangle = mask; }

#endif

// Absolute vertex indices with outward winding (B - A) x (C - A).
// Active edges mark convex creases; coplanar and concave edges are inactive.
struct Triangle {
    Index A, B, C;
    uint ActiveEdges; // Bit i represents the edge from corner i to corner (i + 1) % 3.
    // Owned edges prevent duplicate contact points across adjacent triangles.
    uint OwnedEdges;
    uint BackActiveEdges; // Reverse-winding edge flags retain inactive coplanar edges.
};

struct BvhNode {
    float3 Low, High;
    Index First;
    uint Count;
};

struct Filter {
    uint Layer;
    uint Collides;
    uint Sensor = 0;
    // The union of leaf masks provides conservative body-pair rejection.
    CollisionMask Aggregate{};
    uint Mixed = 0; // Different leaf masks prevent unconditional face suppression.
};

// Color in the low byte, contact degree in the remaining bits.
GPU_CONSTANT uint ColorDegreeShift = 8;
GPU_CONSTANT uint MaxColorDegree = 255;

GPU_CONSTANT uint MaxSupportedColors = 32;
// Island scratch begins with indirect dispatch arguments, then roots, counts and bounded member lists.
GPU_CONSTANT uint IslandBodyLimit = SolveLanes;
GPU_CONSTANT uint IslandPublishAt = 3, IslandContactDualAt = 6, IslandJointDualAt = 9;
GPU_CONSTANT uint IslandLargeAt = 12, IslandColorsAt = 13;
GPU_CONSTANT uint IslandHeaderWords = IslandColorsAt + MaxSupportedColors * 3;
GPU_CONSTANT uint IslandWordsPerBody = 4 + IslandBodyLimit;

struct ColorWork {
    uint Counts[MaxSupportedColors], Offsets[MaxSupportedColors + 1], Cursors[MaxSupportedColors];

    uint ColoringActive, ColoringChanged;
};
inline uint ColorOf(uint word) { return word & 0xFFu; }
// Degree breaks priority ties only between quiet bodies; moving bodies use index order.
inline bool Prioritized(uint other_word, uint other, uint degree, uint body, bool both_quiet) {
    const uint other_degree = other_word >> ColorDegreeShift;
    if (both_quiet && other_degree != degree) return other_degree > degree;
    return other < body;
}

// Fixed per-body contact runs avoid atomic append.
// Collection visits every candidate and replaces shallower contacts when the run is full.
GPU_CONSTANT uint ManifoldPoints = 4;
GPU_CONSTANT uint ManifoldsPerBody = 10;
GPU_CONSTANT uint ContactsPerBody = ManifoldPoints * ManifoldsPerBody;

// Adjacency indexes contacts owned by other bodies, where this body is B.
struct Adjacency {
    uint Count, Start, Cursor;
};

// Normal points from B to A; row 0 has a nonpositive normal dual, rows 1/2 are friction.
// Contacts linearize about the initial pose: C = C0 (1 - alpha) + J dq.
struct Contact {
    float3 AnchorA;
    float3 AnchorB;
    float3 Normal; // The world-space normal points from B to A.
    float3 PointA, PointB; // Geometric points use body frames independently of retained friction anchors.
    float NominalArea = -1, NominalExtent = 0; // Unreduced patch area is -1 when reporting is disabled.
    float StiffnessScale = 0; // Pair inertial stiffness is divided among points sharing the normal.
    float3 C0;
    float3 Lambda;
    float3 Penalty;
    float Friction;
    float Restitution;
    // Initial closing speed before restitution or its threshold is applied.
    float Approach;
    // Accumulated normal restitution impulse and the latest iteration's increment.
    float BounceImpulse, BounceDelta;
    Index BodyA, BodyB;
    uint Feature;
    // Triangle index on B, or NoIndex for non-mesh geometry.
    Index SubShape;
    Index SubShapeA = NoIndex;
    // Child ordinals occupy the low and high 32 bits and are zero for single colliders.
    ulong Children;
    uint Stick;
    uint Active;
};

inline ulong ChildPair(uint own, uint other) { return ulong(own) | (ulong(other) << 32); }
inline uint OwnChild(ulong children) { return uint(children); }
inline uint OtherChild(ulong children) { return uint(children >> 32); }

struct ContactBasis {
    float3 Axis[3];
};

inline ContactBasis MakeContactBasis(float3 normal) {
    float3 tangent = abs(normal.x) > abs(normal.z) ? float3{-normal.y, normal.x, 0} : float3{0, -normal.z, normal.y};
    const float len = length(tangent);
    tangent = len > 1e-8f ? tangent / len : float3{1, 0, 0};
    return {{normal, tangent, cross(normal, tangent)}};
}

enum ContactEventKind : uint {
    ContactAdded,
    ContactPersisted,
    ContactRemoved,
};

struct ContactEvent {
    Index BodyA, BodyB;
    uint Feature;
    Index SubShape;
    ulong Children;
    uint Kind;
    Index SubShapeA = NoIndex;
};

// Events retain solved fields after subsequent substeps overwrite the contact.
struct ContactReport {
    ContactEvent Event;
    float3 Lambda{};
    float Approach = 0, BounceImpulse = 0;
    float3 PointA{}, PointB{}, AnchorA{}, AnchorB{}, Normal{};
    float Friction = 0, Restitution = 0, NominalArea = 0, NominalExtent = 0;
};

inline ContactReport ReportContact(ContactEvent event, Contact contact) {
    ContactReport report{.Event = event};
    if (event.Kind == ContactRemoved) return report;
    report.Lambda = contact.Lambda;
    report.Approach = contact.Approach;
    report.BounceImpulse = contact.BounceImpulse;
    report.PointA = contact.PointA;
    report.PointB = contact.PointB;
    report.AnchorA = contact.AnchorA;
    report.AnchorB = contact.AnchorB;
    report.Normal = contact.Normal;
    report.Friction = contact.Friction;
    report.Restitution = contact.Restitution;
    report.NominalArea = contact.NominalArea;
    report.NominalExtent = contact.NominalExtent;
    return report;
}

struct SensorPair {
    Index BodyA, BodyB;
    ulong Children;
};

struct StepCounts {
    uint Contacts, RemovedContacts, Sensors, ContactRefusals, SensorRefusals;
};

struct StepCompletion {
    uint Colors, Ready, Errors;
};

struct StepOutputFlags {
    uint Poses, Sensors;
};

// At most one event per current point plus one per removed point.
GPU_CONSTANT uint EventsPerBody = 2 * ContactsPerBody;

// One claim bit per contact slot.
static_assert(ContactsPerBody <= 64, "the claimed-slot mask in CollectContacts is a single ulong");

enum JointAxisMode : uint {
    AxisFree,
    AxisLocked,
    AxisDriven,
    AxisPositioned,
    AxisLimited,
};

// An independent drive row, which may act on an axis that also has a limit.
struct JointDrive {
    uint Enabled = 0;
    float Speed = 0, Target = 0, MaxForce = 0;
    float Stiffness = 0, Damping = 0;
    float Lambda = 0, Penalty = 1, Began = 0;
};

// Joints remeasure anchors and frames at every iteration.
// Exactly one unlocked angular axis uses unwrapped twist and the remaining swing.
// Other angular configurations use the relative rotation vector.
struct Joint {
    float3 AnchorA, AnchorB;
    float4 FrameA, FrameB;
    float3 C0Linear, C0Angular;
    // Continuous hinge twist, unwrapped against the previous step.
    float Twist;
    float3 LambdaLinear, LambdaAngular;
    float3 PenaltyLinear, PenaltyAngular;
    float3 MotorSpeed;
    float3 MotorTarget;
    float3 MotorMaxTorque;
    float3 LimitLow, LimitHigh;
    // Linear drives and limits, in metres and newtons.
    float3 LinearMotorSpeed, LinearMotorTarget, LinearMotorMaxForce;
    float3 LinearLimitLow, LinearLimitHigh;
    // Infinite stiffness is hard; finite stiffness is a spring with no dual.
    // Zero stiffness leaves only damping.
    float3 LinearStiffness, AngularStiffness;
    // Viscous coefficients in N s/m and N m s/rad.
    // Backward Euler contributes stiffness c/h on displacement, without a dual or penalty ramp.
    float3 LinearDamping, AngularDamping;
    Index BodyA, BodyB;
    uint LinearModes, AngularModes;
    uint Active;
    JointDrive Drives[6]{};
    // A nonzero mask groups a limit's axes into a distance or cone, stored on its first axis.
    uint LinearLimitAxes[3]{}, AngularLimitAxes[3]{};

    uint Suppresses;
};

inline uint AxisMode(uint modes, uint axis) { return (modes >> (3 * axis)) & 7u; }

// Return the unique unlocked angular axis, or 3 if its count differs from one.
inline uint TwistAxis(uint angular_modes) {
    uint found = 3, count = 0;
    for (uint axis = 0; axis < 3; ++axis) {
        if (AxisMode(angular_modes, axis) == AxisLocked) continue;
        found = axis;
        ++count;
    }
    return count == 1 ? found : 3;
}

inline bool IsHard(float stiffness) { return isinf(stiffness); }

// The first three words are Metal's indirect threadgroup count.
struct QueryArenaHeader {
    uint TaskCount, DispatchY, DispatchZ, TaskCapacity;
    uint ContextCount, ResultCount, ContextCapacity, ResultCapacity;
    uint ContextOffset, TaskOffset, ResultOffset, BatchOffset;
    uint BatchCount, BatchCapacity, Bytes;
    uint Reuse, Valid, GeometryValid, DynamicSame;
    uint GeometryX, GeometryY, GeometryZ;
    uint InputX, InputY, InputZ;
    uint GeometryChecked;
};
GPU_CONSTANT uint QueryHeaderWords = sizeof(QueryArenaHeader) / sizeof(uint);
// Small worlds partition query preparation to increase parallelism.
inline uint QueryPartitions(uint bodies) { return bodies <= 32 ? 8u : 1u; }
inline uint QueryOwnerOffset(uint bodies) { return QueryHeaderWords + bodies * QueryPartitions(bodies); }
inline uint QueryHeadOffset(uint bodies, uint body, uint part) { return QueryHeaderWords + body * QueryPartitions(bodies) + part; }

static_assert(sizeof(QueryArenaHeader) == 104);
struct QueryInputSpec {
    uint Offsets[16], Words;
};
GPU_CONSTANT uint QueryScratchBytes = 32u * 1024u * 1024u;

struct StepParams {
    float3 Gravity;
    float DeltaTime;
    float Beta;
    float ContactBeta;
    float Gamma;
    float PenaltyMin;
    float PenaltyMax;
    float ContactMargin;
    float MaxContactReach;
    float MaxAngularSpeed;
    float MinBounceSpeed; // See StepSettings::BounceSpeedFactor for the threshold scale.
    float SleepSpeed;
    uint SleepSteps;
    float SleepDrift;
    uint BodyCount;
    uint JointCount;
    uint MaxColors;
    uint ReportContacts; // Enable unreduced manifold geometry for contact reporting.
};

inline float4 QuatConjugate(float4 q) { return MakeFloat4(-q.xyz, q.w); }

// Quaternion logarithm as an axis-angle rotation vector.
inline float3 RotationVector(float4 q) {
    if (q.w < 0) q = -q;
    const float sine = length(q.xyz);
    if (sine < 1e-7f) return 2 * q.xyz;
    return q.xyz * (2 * atan2(sine, q.w) / sine);
}

inline float4 QuatFromRotationVector(float3 v) {
    const float angle = length(v);
    if (angle < 1e-7f) return normalize(MakeFloat4(0.5f * v, 1));
    return MakeFloat4(v * (sin(0.5f * angle) / angle), cos(0.5f * angle));
}

inline float3 Rotate(float4 q, float3 v) {
    const float3 axis = q.xyz;
    return v + 2 * cross(axis, cross(axis, v) + q.w * v);
}

inline float4 QuatMul(float4 a, float4 b) {
    return MakeFloat4(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz), a.w * b.w - dot(a.xyz, b.xyz));
}

inline float3 WorldPoint(Pose pose, float3 local) { return pose.Position + Rotate(pose.Orientation, local); }
inline float3 LocalPoint(Pose pose, float3 world) { return Rotate(QuatConjugate(pose.Orientation), world - pose.Position); }

// Compose inner into outer's parent frame.
inline Pose ComposePose(Pose outer, Pose inner) {
    return {WorldPoint(outer, inner.Position), QuatMul(outer.Orientation, inner.Orientation)};
}

} // namespace rbp

#ifdef __METAL_VERSION__

using namespace rbp;
#endif

#endif
