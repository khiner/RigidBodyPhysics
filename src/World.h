#pragma once

#include "gpu/Shared.h"

#include "metal/Buffer.h"
#include "metal/Context.h"
#include <cmath>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace rbp::mtl {
struct Context;
}

namespace rbp {

// Fixed capacities; additions beyond a pool limit are refused and counted.
struct WorldLimits {
    uint32_t Bodies = 4096;
    uint32_t Shapes = 4096;
    uint32_t Joints = 4096;
    uint32_t ShapeVertices = 65536;
    uint32_t HullFaces = 16384;
    uint32_t Triangles = 65536;
    uint32_t BvhNodes = 65536;
    uint32_t CompoundChildren = 65536;
};

inline Pose At(float3 position, float4 turn = {0, 0, 0, 1}) { return {.Position = position, .Orientation = turn}; }

// Mass uses kg and diagonal body-frame inertia uses kg m^2.
// Zero locks the corresponding motion.
struct AuthoredMass {
    float Mass = 1;
    float3 Inertia{1, 1, 1};
};

// Density uses kg/m^3.
// Zero makes the body static.
struct BodyDesc {
    Pose Pose{.Position = {0, 0, 0}, .Orientation = {0, 0, 0, 1}};
    Velocity Velocity{};
    Index Shape = NoIndex;
    float Density = 1000;
    // Authored body-frame mass and diagonal inertia override Density.
    // Movable meshes and offset colliders require these values.
    std::optional<AuthoredMass> Mass{};
    float Friction = 0.5f; // Friction combines by geometric mean.
    // Restitution combines by maximum unless Surface specifies another policy.
    float Restitution = 0;

    // Overrides the body coefficients; collider materials take precedence.
    std::optional<Material> Surface;
    // Gravity multiplier and fractions of velocity removed per second.
    float GravityScale = 1;
    float LinearDamping = 0, AngularDamping = 0;
    // Collider masks override body defaults, and both masks must accept the pair.
    uint32_t Layer = ~0u, CollidesWith = ~0u;
    bool Sensor = false;
};

// Joint anchors and frames use world coordinates.
// A static body provides a world attachment.
struct JointDesc {
    Index BodyA = NoIndex, BodyB = NoIndex;
    float3 At{0, 0, 0};
    // Independent endpoint anchors override At.
    std::optional<float3> AtA, AtB;
    // World orientation of the joint axes; defaults to body B's orientation.
    std::optional<float4> Frame;
    // Independently authored world orientations; override Frame at the corresponding endpoint.
    std::optional<float4> FrameA, FrameB;
    // A single unlocked angular axis uses continuous hinge twist.
    JointAxisMode Angular[3]{AxisFree, AxisFree, AxisFree};
    // Linear axis modes use metres and newtons.
    JointAxisMode Linear[3]{AxisLocked, AxisLocked, AxisLocked};
    float3 MotorSpeed{0, 0, 0};
    // Target angle relative to the joint's rest frame; hinge twist is unwrapped.
    float3 MotorTarget{0, 0, 0};
    float3 MotorMaxTorque{0, 0, 0};
    // Angular limits relative to the rest frame.
    float3 LimitLow{0, 0, 0}, LimitHigh{0, 0, 0};
    // Linear drives and limits relative to the rest anchors.
    float3 LinearMotorSpeed{0, 0, 0}, LinearMotorTarget{0, 0, 0}, LinearMotorMaxForce{0, 0, 0};
    float3 LinearLimitLow{0, 0, 0}, LinearLimitHigh{0, 0, 0};
    // Stiffness uses N/m and N m/rad.
    // Infinity selects a hard constraint.
    float3 LinearStiffness{INFINITY, INFINITY, INFINITY}, AngularStiffness{INFINITY, INFINITY, INFINITY};
    // Viscous coefficients in N s/m and N m s/rad, independent of spring stiffness.
    float3 LinearDamping{0, 0, 0}, AngularDamping{0, 0, 0};
    bool Collide = false;
    JointDrive Drives[6]{};
    // Grouped limits use XYZ masks stored on the lowest selected axis.
    // Other selected axes must be free.
    // Linear groups constrain radial distance; angular groups constrain cone or rotation angle.
    uint32_t LinearLimitAxes[3]{}, AngularLimitAxes[3]{};
};

// Slot and generation identify a body across slot reuse.
struct BodyId {
    Index Slot = NoIndex;
    uint32_t Spawn = 0;
    bool operator==(const BodyId &) const = default;
};

// A geometric manifold, independent of point features and solver ownership order.
// The adapter assigns lifetime IDs; this key can recur after separation.
struct ContactManifold {
    BodyId A, B;
    uint32_t ChildA, ChildB;
    Index SubShapeA, SubShapeB; // Triangle indices are NoIndex for other geometry.
    bool operator==(const ContactManifold &) const = default;
};

// One side's state copied at the reporting boundary, independent of later world mutations.
struct ContactSide {
    Pose InitialPose = IdentityPose; // Contact points and forces use the initial body frame.
    Pose Pose = IdentityPose; // COM pose is measured at the end of the step.
    Velocity Velocity{}; // World velocity is measured at the end of the step.
    float3 Point{}, Anchor{}; // Point and retained friction anchor use the local COM frame.
    uint64_t UserData = 0; // Leaf Shape::UserData remains valid after shape removal or replacement.
    float InvMass = 0;
};

// Contact identity survives body-slot reuse.
struct ContactChange {
    BodyId A, B;
    uint32_t Feature = 0;
    Index SubShape = NoIndex;
    Index SubShapeA = NoIndex;
    uint64_t Children = 0;
    ContactEventKind Kind = ContactAdded;
    // The contact basis contains one nonpositive normal force and two friction forces.
    float3 Lambda{};
    float Approach = 0; // Closing speed is measured at the start of the step.
    float BounceImpulse = 0; // Restitution impulse includes only this step.
    // Removals carry identity and timing only; solved data are zero/default.
    uint64_t Step = 0; // Completed steps are one-based; host removals use the last completed step.
    float DeltaTime = 0; // Duration uses seconds and is zero for host removals.
    ContactSide SideA, SideB;
    float3 Normal{}; // The world-space normal points into A.
    float Friction = 0, Restitution = 0;
    // Patch area and extent use m^2 and m before four-point reduction.
    // Extent follows initial centroid slip or equals the diameter at rest.
    // All points of a manifold share these values.
    float NominalArea = 0, NominalExtent = 0;

    ContactManifold Manifold() const {
        return A.Slot < B.Slot ? ContactManifold{A, B, OwnChild(Children), OtherChild(Children), SubShapeA, SubShape} : ContactManifold{B, A, OtherChild(Children), OwnChild(Children), SubShape, SubShapeA};
    }
    // Return the constraint force on A in newtons, including support.
// B receives the opposite force.
    float3 ForceOnA() const {
        const auto basis = MakeContactBasis(Normal);
        return -(Lambda.x * basis.Axis[0] + Lambda.y * basis.Axis[1] + Lambda.z * basis.Axis[2]);
    }
    // Constraint impulse estimate plus the velocity pass's impulse, in N s.
    // Momentum agreement depends on solver convergence; post-stabilization contributes no impulse.
    float3 ImpulseOnA() const { return DeltaTime * ForceOnA() + BounceImpulse * Normal; }
};

struct SensorOverlap {
    BodyId A, B;
    uint64_t Children{};
    bool operator==(const SensorOverlap &) const = default;
};
struct SensorChange {
    SensorOverlap Pair;
    bool Entered;
};

struct StepSnapshot {
    std::span<const Pose> InitialPoses, Poses;
    std::span<const Velocity> Velocities;
    std::span<const ContactReport> Contacts;
    std::span<const ContactEvent> RemovedContacts;
    std::span<const SensorPair> Sensors;
    std::span<const StepCounts> Counts;
};

// Fixed-capacity structure-of-arrays storage shared by the CPU and GPU.
struct World {
    explicit World(const mtl::Context &, WorldLimits = {});
    World(World &&) noexcept = default;
    World &operator=(World &&) noexcept = default;
    // The world owns its residency set and drains its queued resources on destruction.
    ~World();

    // Return NoIndex and increment the corresponding overflow counter when a pool is full.
    Index AddShape(const Shape &);
    Index AddBody(const BodyDesc &);
    Index AddJoint(const JointDesc &);
    // Cook a solid convex hull; return NoIndex for degenerate input.
    // frame returns the cooked center-of-mass and principal-axis frame.
    // Without local, the body uses that cooked frame and Shape::Local is identity.
    // With local, Shape::Local composes local with the cook frame; authored body mass is then required.
    Index AddHull(std::span<const float3> points, Pose *frame = nullptr, std::optional<Pose> local = {});
    // Cook a one-sided triangle surface in the supplied local frame.
    // Return NoIndex for empty surfaces.
    // Movable mesh bodies require authored mass.
    Index AddMesh(std::span<const float3> points, std::span<const uint32_t> indices, Pose local = IdentityPose);
    // Flatten compounds into owned collider copies and compose their local poses.
    // frame returns the cooked mass frame.
    // Return NoIndex for empty or invalid child lists.
    Index AddCompound(std::span<const Index> children, Pose *frame = nullptr);

    // Mutations require a completed advance.
    // Removal returns false for an inactive index.
    // Raw indices do not carry generations; callers must track their lifetime.
    bool RemoveBody(Index);
    bool RemoveJoint(Index);

    bool RemoveShape(Index);
    // Replace geometry while preserving identity, pose and motion properties.
    // Recompute mass from density unless authored mass is supplied.
    bool SetBodyShape(Index body, Index shape, float density = 1000, std::optional<AuthoredMass> mass = {});
    bool Alive(Index body) const { return body < NumBodies && LiveBodies[body]; }

    // Mark internal faces between stationary boxes or hulls with identical effective masks.
    // Every call recomputes welding and stores masks on private shape copies.
    // Repeat after topology, shape or motion changes.
    // Compounds weld during cooking.
    uint32_t WeldStatic();

    // Wake the body and its connected contacts after host edits, including teleports.
    void Wake(Index body);

    void OnStepped(float delta_time, const StepSnapshot & = {});

    // Opt-in contact queues are populated before Step or Advance returns.
    // Host removals append immediately; consume the queue to bound its growth.
    bool TrackContacts = false;
    std::vector<ContactChange> TakeContactChanges() { return std::exchange(Changes, {}); }

    BodyId IdOf(Index body) const { return {body, Spawns[body]}; }

    uint32_t BodyCount() const { return NumBodies; }
    uint32_t ShapeCount() const { return NumShapes; }
    uint32_t JointCount() const { return NumJoints; }

    struct Overflows {
        uint32_t Bodies{}, Shapes{}, Joints{}, ShapeVertices{}, HullFaces{}, Triangles{}, BvhNodes{}, CompoundChildren{};
    };
    Overflows Overflow{};

    // Offset colliders refused because authored mass was missing.
    uint32_t OffsetsWithoutMass{};

    // Compounds refused because their child list was empty or invalid.
    uint32_t RefusedCompounds{};

    mtl::Buffer<Pose> Poses;
    mtl::Buffer<Velocity> Velocities;
    mtl::Buffer<BodyMass> Masses;
    mtl::Buffer<Index> BodyShapes;
    mtl::Buffer<Shape> Shapes;
    mtl::Buffer<float3> ShapeVertices;
    // Inline face corners use one Metal buffer binding for the hull face pool.
    mtl::Buffer<HullFace> HullFaces;
    mtl::Buffer<Triangle> Triangles;
    mtl::Buffer<BvhNode> BvhNodes;

    mtl::Buffer<Index> CompoundChildren;
    Index Child(Index compound, uint32_t i) const { return ChildOf(Shapes[compound], i, CompoundChildren.All().data()); }
    mtl::Buffer<Material> Materials;
    mtl::Buffer<Filter> Filters;
    mtl::Buffer<Index> Jointed; // Body offsets precede two partner indices per collision-suppressing joint.
    mtl::Buffer<Index> JointIncidence; // Body offsets precede incident joint indices in ascending order.

    mtl::Buffer<BodyBounds> Bounds;
    mtl::Buffer<BodyBounds> BoundsReductions;
    mtl::Buffer<BroadPhaseNode> BroadPhaseNodes;
    mtl::Buffer<MortonKey> BroadPhaseKeys;
    mtl::Buffer<uint32_t> BroadPhaseScratch;
    mtl::Buffer<Pose> InitialPoses;
    mtl::Buffer<Displacement> Displacements;
    mtl::Buffer<Pose> InertialPoses;
    mtl::Buffer<Velocity> PreviousVelocities;
    mtl::Buffer<BodyIterate> Iterates;
    mtl::Buffer<Pose> RestPoses;
    mtl::Buffer<uint32_t> Quiet, NextQuiet;
    mtl::Buffer<uint32_t> Colors;
    mtl::Buffer<uint32_t> NextColors;
    mtl::Buffer<Contact> Contacts;
    // Sensor storage is allocated on first use with one point per overlapping leaf pair.
    // Only Active, pair identity, Feature and C0.x are populated.
    mtl::Buffer<Contact> SensorContacts;
    mtl::Buffer<uint32_t> SensorRefusals;
    std::span<const SensorOverlap> Overlaps() const { return SensorOverlaps; }
    std::vector<SensorChange> TakeSensorChanges() { return std::exchange(SensorChanges, {}); }
    bool TrackSensors = false; // Overlap state updates every step; queued changes are opt-in.

    mtl::Buffer<Adjacency> Incoming;
    mtl::Buffer<uint32_t> IncomingSlots;

    // Per-body event runs, with ContactEventCounts giving the live length.
    mtl::Buffer<ContactEvent> ContactEvents;
    mtl::Buffer<uint32_t> ContactEventCounts;

    mtl::Buffer<uint32_t> ContactRefusals;
    mtl::Buffer<Joint> Joints;

    NS::SharedPtr<MTL::ResidencySet> Residency;
    NS::SharedPtr<MTL4::CommandQueue> Queue;

private:
    friend struct Solver;
    template<typename T> void MakeBuffer(mtl::Buffer<T> &, uint32_t capacity);
    void EnsureSensorBuffers();
    void RefreshFilters();
    // Variable-length geometry runs with reusable gaps.
    struct RunPool {
        struct Run {
            uint32_t Start, Count;
        };
        uint32_t Used{}, Capacity{};
        std::vector<Run> Free;

        Index Take(uint32_t count);
        void Give(Index start, uint32_t count);
    };

    BodyMass ShapeOrAuthoredMass(Index shape, float density, std::optional<AuthoredMass>) const;
    bool OffsetNeedsAuthoredMass(Index shape, const BodyMass &, bool authored) const;
    Index CopyShape(Shape);

    void DropWeld(Index body);
    void EndContacts(Index body);

    void DrainContactEvents(float delta_time, const StepSnapshot &);
    void UpdateSensorOverlaps(const StepSnapshot &);
    void EndSensorOverlaps(Index);
    void ReleaseShape(Index);

    void RebuildJointed();
    bool RetireJoint(Index joint);

    uint32_t NumBodies{}, NumShapes{}, NumJoints{};
    RunPool VertexPool, FacePool, TrianglePool, NodePool, ChildPool;
    std::vector<uint8_t> LiveBodies, LiveShapes;
    // Private welded shape per body, or NoIndex.
    std::vector<Index> WeldedShapes;
    std::vector<Index> FreeBodies, FreeShapes, FreeJoints;

    std::vector<Index> RetiredBodies;
    std::vector<uint32_t> Spawns;
    std::vector<ContactChange> Changes;
    uint64_t CompletedSteps = 0;
    std::vector<SensorOverlap> SensorOverlaps;
    std::vector<SensorChange> SensorChanges;
};

inline constexpr BodyMass StaticMass{.InvInertiaLocal = {0, 0, 0}, .InvMass = 0};

// Mass and diagonal inertia about the shape's center of mass.
BodyMass MassProperties(const Shape &, float density, std::span<const float3> shape_vertices = {}, std::span<const Shape> shapes = {}, std::span<const Index> children = {});

} // namespace rbp
