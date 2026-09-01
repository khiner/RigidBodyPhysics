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
    // The fraction of the world's gravity applied to this body, and the fraction of its linear and angular velocity removed per second.
    // Motion properties rather than mass properties. See BodyMass.
    float GravityScale = 1;
    float LinearDamping = 0, AngularDamping = 0;
    // The bits this body belongs to, and the bits it collides with. Everything collides with everything by default.
    uint32_t Layer = ~0u, CollidesWith = ~0u;
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
};

// A body's identity across time: its slot, and which tenancy of that slot.
// An Index alone is ambiguous once a slot is reused, and a contact stream is read after the fact.
// AddBody bumps the spawn, so a held copy of an old identity stops matching. World::IdOf gives the current one.
struct BodyId {
    Index Slot = NoIndex;
    uint32_t Spawn = 0;
    bool operator==(const BodyId &) const = default;
};

// One change to one contact, in host terms: the kernel's ContactEvent in identities that survive slot reuse.
// It carries the excitation record read back from the contact as the step left it.
// Body A owns the manifold, so a pair is reported exactly once. See World::TrackContacts.
struct ContactChange {
    BodyId A, B;
    uint32_t Feature = 0; // identifies the geometry the point came from, stable while the contact persists
    Index SubShape = NoIndex; // which triangle of a mesh, NoIndex for a shape that is one piece
    uint32_t Children = 0; // and which leaf of each compound, packed as Contact::Children is
    ContactEventKind Kind = ContactAdded;
    // The force each row is applying, in the contact's own basis: the normal row first, then the two friction rows.
    // A normal row only pushes, so Lambda[0] < 0 marks an engaged contact.
    // A contact that never pushed excites no mode, however long it persists.
    float3 Lambda{};
    float Approach = 0; // how fast the pair was closing when the step began
    float BounceImpulse = 0; // the normal impulse the restitution pass applied this step
    // All three are zero on a removal, which has no contact left to read them from.
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
    // A body made of several pieces: the shapes named here, each kept at the Local it already carries, as KHR authors a node with several collider descendants.
    // Returns NoIndex, counted in RefusedCompounds, for more than ChildrenPerCompound children.
    // Also NoIndex for a child that is not live, or a child that is itself a compound, a mesh or a plane.
    // A leaf must be a convex solid the narrowphase already collides, and the tree stays one deep.
    //
    // The body frame produced is the frame every other shape is held in: the centre of mass with the inertia diagonal.
    // A compound therefore moves the geometry under the caller as a hull cook does, and `frame` returns where it sits in the frame the children were given in.
    //
    // The children are copied into shape slots of their own at poses re-expressed in that frame, so the shapes handed in are untouched.
    // A child then belongs to its compound.
    // RemoveShape refuses a child while the compound holds it, and releases every child with the compound.
    // No density is taken here, a compound being one material whose frame does not depend on it, and the body's density is read in AddBody and SetBodyShape.
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

    // Welds static geometry shut, and returns the number of faces marked internal.
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

    // Called by Solver::Step once the step's commands have completed. See RemoveBody.
    void OnStepped();

    // While set, OnStepped copies each step's event runs into a CPU-side queue.
    // That copy happens inside Step, so the queue holds every event of the step before the host can mutate anything.
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
        uint32_t Bodies{}, Shapes{}, Joints{}, Jointed{}, ShapeVertices{}, HullFaces{}, Triangles{}, BvhNodes{};
    };
    Overflows Overflow{};

    // Bodies refused for carrying a shape at a non-identity pose within the body frame with no authored mass.
    // Not an overflow, the pools having had room. See BodyDesc::Mass.
    uint32_t OffsetsWithoutMass{};

    // Compounds refused as a shape the engine does not build: too many children, a child that is not live, or a child that is a compound, a mesh or a plane.
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

    mtl::Buffer<float> Frictions, Restitutions;
    mtl::Buffer<Filter> Filters;
    mtl::Buffer<Index> Jointed; // JointsPerBody slots per body, NoIndex past the end of the run

    // Solver state, per body and per contact slot, named for the algorithm.
    // Kept with everything else the GPU addresses, being indexed the same way and living exactly as long.
    mtl::Buffer<Pose> InitialPoses; // the pose the step began at, which velocity is measured against
    mtl::Buffer<Pose> InertialPoses; // where free flight would have ended it
    mtl::Buffer<Velocity> PreviousVelocities; // last step's, for the adaptive warm start
    mtl::Buffer<Pose> SolvedPoses; // a sweep writes here and is published, so bodies read one snapshot
    mtl::Buffer<Pose> RestPoses; // where a body was when it last moved, for the sleep drift test
    mtl::Buffer<uint32_t> Quiet, NextQuiet; // consecutive steps a body has been slower than SleepSpeed
    mtl::Buffer<uint32_t> Colors, NextColors; // kept across steps, since the coloring is incremental
    mtl::Buffer<Contact> Contacts;
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
    Index CopyShape(Index source, Pose local);
    // Releases the private copy of its shape the weld gave this body. See WeldStatic.
    void DropWeld(Index body);
    // Reports every live contact of this body as removed and clears them, so the stream carries the end with the mutation that caused it, not a step later.
    // A sustained excitation that ends without a removal rings for ever.
    void EndContacts(Index body);
    // The step's event runs, translated and appended to the queue. See TrackContacts.
    void DrainContactEvents();
    // Releases the pool runs and the slot itself, without RemoveShape's checks on remaining users.
    // A compound removing its own children requires skipping those checks.
    void ReleaseShape(Index);

    // The slots naming the bodies this one is jointed to, and so does not collide with.
    // A fixed run per body, filled at the first gap and cleared back to a gap, because the kernel reading it sweeps the whole run.
    std::span<Index> JointedRun(Index body) const { return Jointed.All().subspan(body * JointsPerBody, JointsPerBody); }

    uint32_t NumBodies{}, NumShapes{}, NumJoints{};
    RunPool VertexPool, FacePool, TrianglePool, NodePool;
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
};

// The mass of a static body: a plane, a mesh, no shape at all, or a density of zero.
// Zero inverse quantities hold it fixed with no flag for the solve to branch on.
// Every one of them must be zero, not the mass alone. See Moves in Shared.h.
inline constexpr BodyMass StaticMass{.InvInertiaLocal = {0, 0, 0}, .InvMass = 0};

// The mass and diagonal inertia a shape of this density has about its center of mass.
// A hull's geometry is in the vertex pool and a compound's in its children, so both pools are passed in and read here rather than cached.
BodyMass MassProperties(const Shape &, float density, std::span<const float3> shape_vertices = {}, std::span<const Shape> shapes = {});

} // namespace rbp
