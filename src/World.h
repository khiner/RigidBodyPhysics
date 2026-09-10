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

// Capacities are fixed at construction and no pool grows.
// A full pool refuses the add and counts the refusal, rather than reallocating under a solve holding indices into it.
struct WorldLimits {
    uint32_t Bodies = 4096;
    uint32_t Shapes = 4096;
    uint32_t Joints = 4096;
    uint32_t ShapeVertices = 65536; // the single pool holding every hull's and every mesh's vertices
    uint32_t HullFaces = 16384;
    uint32_t Triangles = 65536;
    uint32_t BvhNodes = 65536;
    uint32_t CompoundChildren = 65536;
};

// A pose at a position, with an optional rotation.
// Named because a designated initializer giving only the position leaves the orientation at zero, which is not a rotation and collapses the body.
inline Pose At(float3 position, float4 turn = {0, 0, 0, 1}) { return {.Position = position, .Orientation = turn}; }

// A body's mass in kg and the diagonal of its inertia about its own centre of mass in kg m^2, supplied by the host.
// A triangle mesh has no volume to integrate, so only the host can supply these, as Jolt, Havok and the KHR reference loader also require.
//
// Zero in either field means infinite rather than absent.
// Zero mass is a body that cannot be pushed, and zero inertia about an axis is a body that cannot be turned about it.
// Zero mass with a finite inertia is pinned in space and spins freely (KHR physics rigid bodies Sec. 128).
// All of it zero is a static body.
struct AuthoredMass {
    float Mass = 1;
    float3 Inertia{1, 1, 1};
};

// Density is in kg/m^3, water being 1000.
// Zero density makes the body static: it keeps its pose and collides, and the solve does not move it.
struct BodyDesc {
    Pose Pose{.Position = {0, 0, 0}, .Orientation = {0, 0, 0, 1}};
    Velocity Velocity{};
    Index Shape = NoIndex;
    float Density = 1000;
    // Set, the mass properties come from here and Density is not read.
    // Unset, a mesh has no volume to integrate and the body stays static, so setting this field is the only way to make a mesh body move.
    //
    // Also required of a movable body whose shape sits at a non-identity Shape::Local, the shape's own integral being about the shape's centre.
    // Carrying the tensor to the body origin is the host's arithmetic.
    // AddBody and SetBodyShape refuse such a body and count World::OffsetsWithoutMass, and a static body is never refused.
    std::optional<AuthoredMass> Mass{};
    float Friction = 0.5f; // combined with the other body's as their geometric mean
    // The fraction of an impact's closing speed returned as separating speed.
    // Combined with the other body's by taking the larger (LiteratureReview.md section 1.1), so a bouncy ball stays bouncy against anything.
    float Restitution = 0;
    // Overrides the legacy coefficients above. Shape materials override this per collider.
    std::optional<Material> Surface;
    // The fraction of the world's gravity applied to this body, and the fraction of its linear and angular velocity removed per second.
    // Motion properties rather than mass properties. See BodyMass.
    float GravityScale = 1;
    float LinearDamping = 0, AngularDamping = 0;
    // The bits this body belongs to, and the bits it collides with. Everything collides with everything by default.
    // Collider and compound Shape::Mask overrides replace these defaults.
    uint32_t Layer = ~0u, CollidesWith = ~0u;
    bool Sensor = false;
};

// Two bodies pinned together at a point in the world, stored in each body's own frame.
// A joint to the world is a joint to a zero-density body, so there is no null-body case.
struct JointDesc {
    Index BodyA = NoIndex, BodyB = NoIndex;
    float3 At{0, 0, 0}; // the world point they are pinned at, as they stand now
    // Or each body's own end of it, where the two are not the same point, as KHR authors one frame per node.
    // A joint made across a gap closes it over the steps alpha spreads the error across rather than in one step.
    std::optional<float3> AtA, AtB;
    // The joint's own frame, in world, as it stands now: the axes the modes below are named in.
    // For a KHR joint that is the joint node's transform, generally not an axis of either body.
    // Unset takes body B's own axes.
    std::optional<float4> Frame;
    // Independently authored world orientations; override Frame at the corresponding endpoint.
    std::optional<float4> FrameA, FrameB;
    // The mode of each of the frame's three angular axes.
    // All free is a ball joint, all locked a fixed one, and two locked with the third free or driven is a hinge about that axis.
    JointAxisMode Angular[3]{AxisFree, AxisFree, AxisFree};
    // The mode of the frame's three linear axes, the same list in metres and newtons.
    // An axis is locked at the anchor, free, sliding between stops, driven to a speed, or held at an offset, and all locked is a ball joint.
    // KHR names a linear axis by its limit: min == max == 0 is Locked, min < max is Limited, and min == max elsewhere is Positioned at that offset.
    // An axis KHR never mentions is Free.
    JointAxisMode Linear[3]{AxisLocked, AxisLocked, AxisLocked};
    float3 MotorSpeed{0, 0, 0}; // for a driven axis, the relative speed it turns towards
    // For a positioned axis, the relative angle it turns to, measured from the pose the joint was made in, the zero LimitLow and LimitHigh also use.
    // Unbounded on the axis a hinge turns about, whose angle accumulates rather than being read off a quaternion, so twenty revolutions means twenty.
    // On a joint with no such axis the error is one rotation vector and folds back at a half turn.
    float3 MotorTarget{0, 0, 0};
    float3 MotorMaxTorque{0, 0, 0};
    // For a limited axis, how far either side of the pose the joint was made in it may turn.
    // Past a half turn where that axis is the one a hinge turns about, on the same terms as MotorTarget.
    float3 LimitLow{0, 0, 0}, LimitHigh{0, 0, 0};
    // The same five for the linear axes, measured along the frame from where the two ends met when the joint was made.
    // The speed a driven axis slides at, the offset a positioned one holds, the force bound for both, and how far either way a limited one may slide.
    float3 LinearMotorSpeed{0, 0, 0}, LinearMotorTarget{0, 0, 0}, LinearMotorMaxForce{0, 0, 0};
    float3 LinearLimitLow{0, 0, 0}, LinearLimitHigh{0, 0, 0};
    // The material stiffness of each row, in N/m and N m/rad.
    // Infinite is a hard constraint, the default for a joint.
    // Finite makes the row a spring: a limited axis then stops softly at its stops, and a locked one holds with give.
    float3 LinearStiffness{INFINITY, INFINITY, INFINITY}, AngularStiffness{INFINITY, INFINITY, INFINITY};
    // The viscous coefficient of each row, in N s/m and N m s/rad, the c of KHR's k (xT - x) + c (vT - v), whose k is the stiffness above.
    // Damping with no stiffness is a velocity motor approaching its target, and with a target of zero it is a brake.
    // A drive with neither is a hard row, which spreads a step's initial error by alpha rather than closing it in one step.
    // Acceleration mode is the host's arithmetic on the reduced inertia, deliberately not a mode here.
    float3 LinearDamping{0, 0, 0}, AngularDamping{0, 0, 0};
    // Two bodies a joint holds together overlap by design, so by default they are not also collided.
    bool Collide = false;
    JointDrive Drives[6]{}; // independent linear XYZ and angular XYZ drives
    // Grouped limit masks (bits XYZ), placed on the lowest selected axis. Other selected axes must be free.
    // Linear groups constrain radial distance; angular groups constrain cone/rotation angle.
    uint32_t LinearLimitAxes[3]{}, AngularLimitAxes[3]{};
};

// A body's identity across time: its slot, and which tenancy of that slot.
// An Index alone is ambiguous once a slot is reused, and a contact stream is read after the fact.
// AddBody bumps the spawn, so a held copy of an old identity stops matching. World::IdOf gives the current one.
struct BodyId {
    Index Slot = NoIndex;
    uint32_t Spawn = 0;
    bool operator==(const BodyId &) const = default;
};

// A geometric manifold, independent of point features and solver ownership order.
// The adapter assigns lifetime IDs; this key can recur after separation.
struct ContactManifold {
    BodyId A, B; // sorted by body slot
    uint32_t ChildA, ChildB;
    Index SubShapeA, SubShapeB; // triangle on each side, or NoIndex
    bool operator==(const ContactManifold &) const = default;
};

// One side's state copied at the reporting boundary, independent of later world mutations.
struct ContactSide {
    Pose InitialPose = IdentityPose; // collision geometry and AVBD force frame
    Pose Pose = IdentityPose; // end-of-step COM pose
    Velocity Velocity{}; // end-of-step world velocity
    float3 Point{}, Anchor{}; // local COM frame: geometric point and possibly retained friction anchor
    uint64_t UserData = 0; // leaf Shape::UserData, valid after shape removal/replacement
    float InvMass = 0;
};

// One change to one contact, in host terms: the kernel's ContactEvent in identities that survive slot reuse.
// It carries the excitation record read back from the contact as the step left it.
// Body A owns the manifold, so a pair is reported exactly once. See World::TrackContacts.
struct ContactChange {
    BodyId A, B;
    uint32_t Feature = 0; // identifies the geometry the point came from, stable while the contact persists
    Index SubShape = NoIndex;
    Index SubShapeA = NoIndex;
    uint64_t Children = 0; // and which leaf of each compound, packed as Contact::Children is
    ContactEventKind Kind = ContactAdded;
    // The force each row is applying, in the contact's own basis: the normal row first, then the two friction rows.
    // A normal row only pushes, so Lambda[0] < 0 marks an engaged contact.
    // A contact that never pushed excites no mode, however long it persists.
    float3 Lambda{};
    float Approach = 0; // how fast the pair was closing when the step began
    float BounceImpulse = 0; // the normal impulse the restitution pass applied this step
    // Removals carry identity and timing only; solved data are zero/default.
    uint64_t Step = 0; // one-based completed step, or last completed step for a host-side removal
    float DeltaTime = 0; // seconds; zero for host-side removals
    ContactSide SideA, SideB;
    float3 Normal{}; // world, into A
    float Friction = 0, Restitution = 0;
    // Projected patch before four-point reduction, in m^2 and m. Extent follows initial slip
    // at the patch centroid, or is its diameter when stationary. Shared by all points of a manifold.
    float NominalArea = 0, NominalExtent = 0;

    ContactManifold Manifold() const {
        return A.Slot < B.Slot ? ContactManifold{A, B, OwnChild(Children), OtherChild(Children), SubShapeA, SubShape} : ContactManifold{B, A, OtherChild(Children), OwnChild(Children), SubShape, SubShapeA};
    }
    // AVBD constraint force in newtons, including support. B receives its negative.
    float3 ForceOnA() const {
        const auto basis = MakeContactBasis(Normal);
        return -(Lambda.x * basis.Axis[0] + Lambda.y * basis.Axis[1] + Lambda.z * basis.Axis[2]);
    }
    // Constraint impulse estimate plus the velocity pass's impulse, in N s.
    // Momentum agreement depends on solver convergence; post-stabilization contributes no impulse.
    float3 ImpulseOnA() const { return DeltaTime * ForceOnA() + BounceImpulse * Normal; }
};

// One overlapping collider pair. Identities remain valid across body-slot reuse.
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

// Engine-owned struct-of-arrays in shared buffers.
// Every per-body array is indexed by the same body index, so a body's handle is that index and the engine holds no pointers.
// The arrays are split by access pattern rather than by concept, so a kernel loads only the lanes it reads.
struct World {
    explicit World(const mtl::Context &, WorldLimits = {});
    World(World &&) noexcept = default;
    World &operator=(World &&) noexcept = default;
    // Returns the residency set to the queue.
    // A queue holds at most thirty-two residency sets, so past that limit a further world's buffers are never made resident and its next step never returns.
    ~World();

    // Each returns NoIndex when the pool is full, and bumps the matching overflow counter.
    Index AddShape(const Shape &);
    Index AddBody(const BodyDesc &);
    Index AddJoint(const JointDesc &);
    // The convex hull of these points, cooked into the frame the engine holds a shape in. See CookedHull.
    // Points inside the hull are dropped.
    // Returns NoIndex when the points make no solid, which is a shape that cannot exist rather than a pool that ran out, and so is not an overflow.
    //
    // The cook moves the points, so a body given this shape has a pose in the cooked frame.
    // `frame` returns where that frame sits in the caller's, which turns the body's pose back into the pose of the geometry as given.
    //
    // `local` gives where the points as handed in sit in the body frame, and Local comes back as `local` composed with the cook's frame, in that order.
    // Omitting `local` differs from passing identity: the body frame becomes the cooked frame, the one frame the engine can integrate the hull in.
    // Local then comes back identity.
    // An identity `local` instead puts the body frame on the points as handed in, and the host then owes an AuthoredMass about that frame.
    Index AddHull(std::span<const float3> points, Pose *frame = nullptr, std::optional<Pose> local = {});
    // A triangle mesh: a one-sided surface with no interior and no mass properties. See CookedMesh.
    // A body given one moves only when the host supplies BodyDesc::Mass.
    // Returns NoIndex when the input holds no surface.
    // `local` is where those points sit in the body frame.
    // A surface stays in the frame it arrived in, so unlike a hull there is no cook frame under it to compose with.
    Index AddMesh(std::span<const float3> points, std::span<const uint32_t> indices, Pose local = IdentityPose);
    // Copies collider leaves into a compound, flattening nested compounds and composing their poses.
    // `frame` returns the cooked centre-of-mass and principal-axis frame in the supplied frame.
    // A compound owns its copied leaves. Input shapes remain independently usable.
    // Empty lists and dead children are refused and counted. Surface-only compounds need authored mass to move.
    Index AddCompound(std::span<const Index> children, Pose *frame = nullptr);

    // Mutation happens between steps: Solver::Step commits and waits for its own completion before returning, so outside a step the host owns every buffer.
    // Nothing here queues, defers or flushes.
    //
    // Each returns false when the index does not name something live, which catches a double removal.
    // An index carries no generation, so a handle used after its slot was reused is the caller's lifetime problem.
    bool RemoveBody(Index);
    bool RemoveJoint(Index);
    // Refused while any live body still uses it, so a shape cannot be removed under a body.
    bool RemoveShape(Index);
    // New geometry for a body that keeps its index, its pose and its motion.
    // Mass properties are recomputed at `density`, the world keeping no record of the density the body was made with.
    // The motion properties beside them in BodyMass are kept, belonging to the body rather than to the shape.
    // Given `mass`, the mass properties come from there and `density` is not read. See BodyDesc::Mass.
    bool SetBodyShape(Index body, Index shape, float density = 1000, std::optional<AuthoredMass> mass = {});
    bool Alive(Index body) const { return body < NumBodies && LiveBodies[body]; }

    // Welds static geometry with identical effective collision masks, returning the number of internal faces.
    // A face of one static body wholly covered by a coplanar touching face of another is inside the solid the two make together.
    // Such a face is marked through the same InternalFaces bits AddCompound sets for a compound's siblings.
    // Unmarked, that join acts as a wall.
    // A contact rests ContactMargin inside the face holding it, and the box test then reads the far tile's near face as the least overlap.
    // A box slid at the join of two static boxes travels 0.042 m of a seamless floor's 0.392 (RbpScenes join).
    // No per-pair rule resolves this, Jolt's EnhancedInternalEdgeRemoval included.
    //
    // Static here means a body the solve does not move (Moves) and the host is not moving.
    // A kinematic body the host has not started moving reads as static, so call this with the world standing as it will run.
    // Boxes and hulls only: a plane is unbounded, a mesh is a surface, and a compound's children are marked by its own cook.
    //
    // Every call recomputes the whole result, so there is nothing incremental to unwind.
    // A body removed, reshaped or given a mass since the last call has its neighbours' faces back at the next call.
    // A body added afterwards is welded at the next call.
    // Shapes are shared, so the mask goes on a private copy the body wears from then on, and BodyShapes names that copy afterwards.
    // The copy is released when the body is removed or reshaped.
    uint32_t WeldStatic();

    // Wakes this body and everything it is touching.
    // The host calls it after writing Poses, Masses or any other lane that changes the world.
    // Waking otherwise spreads only from a moving body, and a teleported body has no velocity to spread from.
    //
    // Deliberately not called by the pose write itself.
    // A host restoring a cached pose puts a body back where it already was, and waking there would start a scrubbed timeline running.
    void Wake(Index body);

    void OnStepped(float delta_time, const StepSnapshot & = {});

    // While set, OnStepped copies each step's event runs into a CPU-side queue.
    // RemoveBody and SetBodyShape append the removals they synthesize.
    // Off by default, because an untaken queue only grows.
    bool TrackContacts = false;
    std::vector<ContactChange> TakeContactChanges() { return std::exchange(Changes, {}); }
    // The current identity of a body slot, in the terms the contact stream uses to name bodies.
    // A held copy stops matching once the slot is reused.
    BodyId IdOf(Index body) const { return {body, Spawns[body]}; }

    uint32_t BodyCount() const { return NumBodies; }
    uint32_t ShapeCount() const { return NumShapes; }
    uint32_t JointCount() const { return NumJoints; }

    // Adds refused because a pool was full, a scene sizing problem rather than a runtime one.
    struct Overflows {
        uint32_t Bodies{}, Shapes{}, Joints{}, ShapeVertices{}, HullFaces{}, Triangles{}, BvhNodes{}, CompoundChildren{};
    };
    Overflows Overflow{};

    // Bodies refused for carrying a shape at a non-identity pose within the body frame with no authored mass.
    // Not an overflow, the pools having had room. See BodyDesc::Mass.
    uint32_t OffsetsWithoutMass{};

    // Compounds refused as a shape the engine does not build: an empty list or a dead child.
    // Not an overflow either, this limit being one the host cannot raise. See AddCompound.
    uint32_t RefusedCompounds{};

    mtl::Buffer<Pose> Poses;
    mtl::Buffer<Velocity> Velocities;
    mtl::Buffer<BodyMass> Masses;
    mtl::Buffer<Index> BodyShapes;
    mtl::Buffer<Shape> Shapes;
    mtl::Buffer<float3> ShapeVertices; // every hull's corners and every mesh's points, a run per shape
    // The faces the hull cook produced, each holding its corners inline, which keeps a hull to one of Metal's thirty-one buffer bindings.
    mtl::Buffer<HullFace> HullFaces;
    mtl::Buffer<Triangle> Triangles; // and every mesh's triangles, indexing absolutely into that pool
    mtl::Buffer<BvhNode> BvhNodes; // and the tree over them, each shape's nodes indexed from its root

    mtl::Buffer<Index> CompoundChildren;
    Index Child(Index compound, uint32_t i) const { return ChildOf(Shapes[compound], i, CompoundChildren.All().data()); }
    mtl::Buffer<Material> Materials;
    mtl::Buffer<Filter> Filters;
    mtl::Buffer<Index> Jointed; // Body offsets precede two partner indices per collision-suppressing joint.
    mtl::Buffer<Index> JointIncidence; // Body offsets precede incident joint indices in ascending order.

    // Solver state, per body and per contact slot, named for the algorithm.
    // Kept with everything else the GPU addresses, being indexed the same way and living exactly as long.
    mtl::Buffer<BodyBounds> Bounds;
    mtl::Buffer<BodyBounds> BoundsReductions;
    mtl::Buffer<BroadPhaseNode> BroadPhaseNodes;
    mtl::Buffer<MortonKey> BroadPhaseKeys;
    mtl::Buffer<uint32_t> BroadPhaseScratch;
    mtl::Buffer<Pose> InitialPoses; // the pose the step began at, which velocity is measured against
    mtl::Buffer<Displacement> Displacements;
    mtl::Buffer<Pose> InertialPoses; // where free flight would have ended it
    mtl::Buffer<Velocity> PreviousVelocities; // last step's, for the adaptive warm start
    mtl::Buffer<BodyIterate> Iterates;
    mtl::Buffer<Pose> RestPoses; // where a body was when it last moved, for the sleep drift test
    mtl::Buffer<uint32_t> Quiet, NextQuiet; // consecutive steps a body has been slower than SleepSpeed
    mtl::Buffer<uint32_t> Colors, NextColors; // kept across steps, since the coloring is incremental
    mtl::Buffer<Contact> Contacts;
    // Allocated on first sensor step. One point per overlapping leaf pair, separate from solid contacts.
    // Only Active, pair identity, Feature and C0.x are populated.
    mtl::Buffer<Contact> SensorContacts;
    mtl::Buffer<uint32_t> SensorRefusals;
    std::span<const SensorOverlap> Overlaps() const { return SensorOverlaps; }
    std::vector<SensorChange> TakeSensorChanges() { return std::exchange(SensorChanges, {}); }
    bool TrackSensors = false; // overlap state is always refreshed; queued changes are opt-in

    mtl::Buffer<Adjacency> Incoming; // per body, where its contacts-as-B are in the list below
    mtl::Buffer<uint32_t> IncomingSlots;

    // The changes to this body's contacts over the last step: a fixed run of EventsPerBody per body, of which the first ContactEventCounts[body] are live.
    // Per body rather than compacted into one list, so the layout is fixed before a step starts.
    // Two runs therefore report the same events in the same order, with no atomic append and no sort.
    // Only the body a contact names as A reports it, so a pair is reported once and the event names both sides.
    mtl::Buffer<ContactEvent> ContactEvents;
    mtl::Buffer<uint32_t> ContactEventCounts;

    // Manifold points produced last step with no slot left to hold them, which is a scene outgrowing its world.
    // Exact rather than a lower bound, because every partner is collided even once the run is full.
    mtl::Buffer<uint32_t> ContactRefusals;
    mtl::Buffer<Joint> Joints;

    NS::SharedPtr<MTL::ResidencySet> Residency;
    NS::SharedPtr<MTL4::CommandQueue> Queue; // held rather than borrowed, so the destructor does not depend on the context

private:
    friend struct Solver;
    template<typename T> void MakeBuffer(mtl::Buffer<T> &, uint32_t capacity);
    void EnsureSensorBuffers();
    void RefreshFilters();
    // A pool of variable-length runs: a hull's vertices, a mesh's triangles, the tree over them.
    // A bump pointer plus the freed runs, kept sorted and merged with their neighbours.
    // A mesh replaced by a similar one therefore fits where the first was rather than fragmenting the pool.
    struct RunPool {
        struct Run {
            uint32_t Start, Count;
        };
        uint32_t Used{}, Capacity{};
        std::vector<Run> Free;

        Index Take(uint32_t count); // NoIndex when nothing free fits and the tail has no room either
        void Give(Index start, uint32_t count);
    };

    // The body's mass properties, from the authored mass where the host gave one and from the shape otherwise.
    // See BodyDesc::Mass, the only source for a shape with no volume.
    BodyMass ShapeOrAuthoredMass(Index shape, float density, std::optional<AuthoredMass>) const;
    // Whether those properties are about the wrong point.
    // The shape sits off the body's origin, no mass was authored, and the shape was integrated about itself.
    bool OffsetNeedsAuthoredMass(Index shape, const BodyMass &, bool authored) const;
    // One shape duplicated into a slot of its own at a new pose within the body frame, runs included.
    // Returns NoIndex when a pool or the slot table refused, releasing whatever was taken.
    Index CopyShape(Shape);
    // Releases the private copy of its shape the weld gave this body. See WeldStatic.
    void DropWeld(Index body);
    // Reports every live contact of this body as removed and clears them, so the stream carries the end with the mutation that caused it, not a step later.
    // A sustained excitation that ends without a removal rings for ever.
    void EndContacts(Index body);
    // The step's event runs, translated and appended to the queue. See TrackContacts.
    void DrainContactEvents(float delta_time, const StepSnapshot &);
    void UpdateSensorOverlaps(const StepSnapshot &);
    void EndSensorOverlaps(Index);
    // Releases the pool runs and the slot itself, without RemoveShape's checks on remaining users.
    // A compound removing its own children requires skipping those checks.
    void ReleaseShape(Index);

    void RebuildJointed();
    bool RetireJoint(Index joint);

    uint32_t NumBodies{}, NumShapes{}, NumJoints{};
    RunPool VertexPool, FacePool, TrianglePool, NodePool, ChildPool;
    // Which slots are handed out and not yet released.
    // A body with no shape is a legal live body, so liveness is not derivable from the buffers and is kept here.
    // A joint's liveness is its own Active flag.
    std::vector<uint8_t> LiveBodies, LiveShapes;
    // The private copy of its shape the weld gave this body, NoIndex where it has none.
    // This is the only record that a shape carrying a mask belongs to the weld and is released with the body. See WeldStatic.
    std::vector<Index> WeldedShapes;
    std::vector<Index> FreeBodies, FreeShapes, FreeJoints;
    // Bodies removed since the last step ran, which are not free yet. See RemoveBody.
    std::vector<Index> RetiredBodies;
    // The current tenancy of each body slot (see BodyId), and the queue the stream accumulates in until the host takes it.
    // Host-side state, read by no kernel.
    std::vector<uint32_t> Spawns;
    std::vector<ContactChange> Changes;
    uint64_t CompletedSteps = 0;
    std::vector<SensorOverlap> SensorOverlaps;
    std::vector<SensorChange> SensorChanges;
};

// The mass of a static body: a plane, a mesh, no shape at all, or a density of zero.
// Zero inverse quantities hold it fixed with no flag for the solve to branch on.
// Every one of them must be zero, not the mass alone. See Moves in Shared.h.
inline constexpr BodyMass StaticMass{.InvInertiaLocal = {0, 0, 0}, .InvMass = 0};

// The mass and diagonal inertia a shape of this density has about its center of mass.
// A hull's geometry is in the vertex pool and a compound's in its children, so both pools are passed in and read here rather than cached.
BodyMass MassProperties(const Shape &, float density, std::span<const float3> shape_vertices = {}, std::span<const Shape> shapes = {}, std::span<const Index> children = {});

} // namespace rbp
