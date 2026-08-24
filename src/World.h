#pragma once

#include "gpu/Shared.h"

#include <cmath>
#include <span>
#include <vector>
#include "metal/Buffer.h"
#include "metal/Context.h"

namespace mtl {
struct Context;
}

// Capacities are fixed at construction and nothing here grows. A full pool refuses the add and counts
// the refusal, so a scene that outgrows its world says so plainly instead of reallocating out from
// under a solve that is holding indices into it.
struct WorldLimits {
    uint32_t Bodies = 4096;
    uint32_t Shapes = 4096;
    uint32_t Joints = 4096;
    uint32_t ShapeVertices = 65536; // the one pool every hull's and every mesh's vertices live in
    uint32_t HullFaces = 16384;
    uint32_t Triangles = 65536;
    uint32_t BvhNodes = 65536;
};

// Density in kg/m^3 - water is 1000. Zero density makes the body static: it keeps its pose and takes
// part in collision, but nothing moves it.
struct BodyDesc {
    Pose Pose{.Position = {0, 0, 0}, .Orientation = {0, 0, 0, 1}};
    Velocity Velocity{};
    Index Shape = NoIndex;
    float Density = 1000;
    float Friction = 0.5f; // combined with the other body's as their geometric mean
    // How much of an impact's closing speed comes back as separating speed. Combined with the other
    // body's by taking the larger, which is the convention in LiteratureReview.md section 1.1: a
    // bouncy ball stays bouncy whatever it lands on.
    float Restitution = 0;
    // Which bits this body is, and which it will touch. Everything touches everything by default.
    uint32_t Layer = ~0u, CollidesWith = ~0u;
};

// Two bodies pinned together at a point in the world, which each of them remembers in its own frame.
// A joint to nothing in particular is a joint to a zero-density body, so there is no null-body case.
struct JointDesc {
    Index BodyA = NoIndex, BodyB = NoIndex;
    float3 At{0, 0, 0}; // the world point they are pinned at, as they stand now
    // What each of body B's three axes does. All free is a ball joint, all locked a fixed one, and two
    // locked with the third free or driven is a hinge about that axis.
    JointAxisMode Angular[3]{AxisFree, AxisFree, AxisFree};
    float3 MotorSpeed{0, 0, 0}; // for a driven axis, the relative speed it turns towards
    // For a positioned axis, the relative angle it turns to, measured from the pose the joint was made
    // in - the same zero LimitLow and LimitHigh use. Within a half turn either way, since the error a
    // row measures comes out of a quaternion and wraps.
    float3 MotorTarget{0, 0, 0};
    float3 MotorMaxTorque{0, 0, 0};
    // For a limited axis, how far either side of the pose the joint was made in it may turn.
    float3 LimitLow{0, 0, 0}, LimitHigh{0, 0, 0};
    // How stiff each row is as a material. Infinite is a hard constraint, which is what a joint is by
    // default. Finite makes that row a spring: a limited axis given one stops at its stops softly, and
    // a locked one given one holds with give rather than exactly. In N/m and N m/rad respectively.
    float3 LinearStiffness{INFINITY, INFINITY, INFINITY}, AngularStiffness{INFINITY, INFINITY, INFINITY};
    // Two bodies a joint holds together overlap by design, so by default they are not also collided.
    bool Collide = false;
};

// Engine-owned struct-of-arrays in shared buffers. Every per-body array is indexed by the same body
// index, so a body's handle *is* that index and nothing in the engine holds a pointer.
//
// The split is by access pattern rather than by concept: integration touches Poses and Velocities,
// the solve touches Velocities and Masses, collision touches Poses and Shapes. Keeping them apart
// means a kernel pulls in only the lanes it reads.
struct World {
    explicit World(const mtl::Context &, WorldLimits = {});
    World(World &&) noexcept = default;
    World &operator=(World &&) noexcept = default;
    // Which exists only to give the residency set back. A queue holds thirty-two ever, so a context
    // that had built that many worlds, alive or dead, stopped making any further world's
    // buffers resident and the next step never returned. Anything building a world in a loop reaches it.
    ~World();

    // Both return NoIndex when the pool is full, and bump the matching overflow counter.
    Index AddShape(const Shape &);
    Index AddBody(const BodyDesc &);
    Index AddJoint(const JointDesc &);
    // The convex hull of these points, cooked into the frame the engine holds a shape in - see
    // CookedHull. Points inside the hull are dropped. NoIndex when the points make no solid, which is a
    // shape that could not exist rather than a pool that ran out, and so is not counted as an overflow.
    //
    // Cooking moves the points, so a body given this shape has a pose in the cooked frame rather than
    // in the one the points arrived in. `frame`, when a caller passes one, comes back with where that
    // frame sits in theirs, which is what turns the body's pose back into the pose of the geometry as
    // it was given.
    Index AddHull(std::span<const float3> points, Pose *frame = nullptr);
    // A triangle mesh - static collision geometry, and the one shape a body cannot be given and still
    // move. See CookedMesh: the points are welded, the seams of the tessellation told from the creases,
    // and a tree built over the triangles. NoIndex when there is no surface in what was given.
    Index AddMesh(std::span<const float3> points, std::span<const uint32_t> indices);

    // Mutation, which is a between-steps operation and needs to be nothing more than that: Solver::Step
    // commits and waits for its own completion before it returns, so whenever the host is not inside
    // one it owns every buffer outright. Nothing here queues, defers or flushes.
    //
    // Each returns false when the index does not name something live, which catches a removal done
    // twice. An index carries no generation, so a handle kept past its body and used after the slot was
    // handed out again is the caller's lifetime problem.
    bool RemoveBody(Index);
    bool RemoveJoint(Index);
    // Refused while any live body still has it, so a shape cannot go out from under one.
    bool RemoveShape(Index);
    // New geometry for a body that keeps its index, its pose and its motion. Mass properties are
    // recomputed at this density, since the body does not remember the one it was made with.
    bool SetBodyShape(Index body, Index shape, float density = 1000);
    bool Alive(Index body) const { return body < NumBodies && LiveBodies[body]; }

    // Called by Solver::Step once the step's commands have completed, and the reason removal needs no
    // machinery of its own. See RemoveBody.
    void OnStepped();

    uint32_t BodyCount() const { return NumBodies; }
    uint32_t ShapeCount() const { return NumShapes; }
    uint32_t JointCount() const { return NumJoints; }

    // Adds refused for want of capacity, which is a scene problem rather than a runtime one.
    struct Overflows {
        uint32_t Bodies{}, Shapes{}, Joints{}, Jointed{}, ShapeVertices{}, HullFaces{}, Triangles{}, BvhNodes{};
    };
    Overflows Overflow{};

    mtl::Buffer<Pose> Poses;
    mtl::Buffer<Velocity> Velocities;
    mtl::Buffer<BodyMass> Masses;
    mtl::Buffer<Index> BodyShapes;
    mtl::Buffer<Shape> Shapes;
    mtl::Buffer<float3> ShapeVertices; // every hull's corners and every mesh's points, a run per shape
    // The faces a hull's cook recovered, each naming its own corners inline - which is what keeps a
    // hull to one buffer binding, and Metal gives thirty-one in all.
    mtl::Buffer<HullFace> HullFaces;
    mtl::Buffer<Triangle> Triangles; // and every mesh's triangles, indexing absolutely into that pool
    mtl::Buffer<BvhNode> BvhNodes; // and the tree over them, each shape's nodes indexed from its root

    mtl::Buffer<float> Frictions, Restitutions;
    mtl::Buffer<Filter> Filters;
    mtl::Buffer<Index> Jointed; // JointsPerBody slots per body, NoIndex where it runs out

    // Solver state, per body and per contact slot, named for the algorithm. Kept here with everything
    // else the GPU addresses, since it is indexed the same way and lives exactly as long.
    mtl::Buffer<Pose> InitialPoses; // where the step began, which is what velocity comes out of
    mtl::Buffer<Pose> InertialPoses; // where free flight would have ended it
    mtl::Buffer<Velocity> PreviousVelocities; // last step's, for the adaptive warm start
    mtl::Buffer<Pose> SolvedPoses; // a sweep writes here and is published, so bodies read one snapshot
    mtl::Buffer<Pose> RestPoses; // where a body was when it last moved, to see whether it has since
    mtl::Buffer<uint32_t> Quiet, NextQuiet; // consecutive steps a body has been too slow to be worth solving
    mtl::Buffer<uint32_t> Colors, NextColors; // kept across steps, since the coloring is incremental
    mtl::Buffer<Contact> Contacts;
    mtl::Buffer<Adjacency> Incoming; // per body, where its contacts-as-B are in the list below
    mtl::Buffer<uint32_t> IncomingSlots;

    // What changed about this body's contacts over the last step: a fixed run of EventsPerBody per
    // body, of which the first ContactEventCounts[body] are live. Kept per body rather than compacted
    // into one list so the layout is fixed before a step starts, which is what makes two runs report
    // the same events in the same order without an atomic append or a sort to undo it.
    //
    // Only the body that owns a manifold reports it - the body a contact names as A - so a pair is
    // reported once and the event names both sides. A settled stack goes on reporting its contacts as
    // persisted, since a body solved as asleep still collides.
    mtl::Buffer<ContactEvent> ContactEvents;
    mtl::Buffer<uint32_t> ContactEventCounts;

    // Manifold points a body found last step and had no slot left to keep, per body - a scene
    // outgrowing its world, the same thing the pools above count a refusal for. Exact rather than a
    // lower bound, since every partner is collided whether or not the run is already full.
    mtl::Buffer<uint32_t> ContactRefusals;
    mtl::Buffer<Joint> Joints;

    NS::SharedPtr<MTL::ResidencySet> Residency;
    NS::SharedPtr<MTL4::CommandQueue> Queue; // held rather than borrowed, so the destructor above stands alone

private:
    // A pool of variable-length runs - a hull's vertices, a mesh's triangles, the tree over them. A
    // bump pointer plus the runs handed back, kept sorted and merged with their neighbours, so a mesh
    // removed and replaced by a similar one fits where the first was instead of leaving the pool in
    // pieces none of which is long enough.
    struct RunPool {
        struct Run {
            uint32_t Start, Count;
        };
        uint32_t Used{}, Capacity{};
        std::vector<Run> Free;

        Index Take(uint32_t count); // NoIndex when nothing free fits and the tail has no room either
        void Give(Index start, uint32_t count);
    };

    // Everything a removed body was touching, woken. Nothing else will: waking spreads from a body
    // that is moving to the ones it touches, and what has just gone is not moving.
    void Wake(Index body);

    uint32_t NumBodies{}, NumShapes{}, NumJoints{};
    RunPool VertexPool, FacePool, TrianglePool, NodePool;
    // Which slots have been handed out and not given back. A body with no shape is a legal live body,
    // so liveness is not readable from the buffers and is kept here. A joint's is its own Active flag.
    std::vector<uint8_t> LiveBodies, LiveShapes;
    std::vector<Index> FreeBodies, FreeShapes, FreeJoints;
    // Bodies removed since the last step ran, which are not free yet. See RemoveBody.
    std::vector<Index> RetiredBodies;
};

// What a body nothing can move has: a plane, a mesh, no shape at all, or a density of zero. Zero
// inverse quantities are what keep it fixed without the solve needing to branch on a flag.
inline constexpr BodyMass StaticMass{.InvInertiaLocal = {0, 0, 0}, .InvMass = 0};

// Mass and the diagonal inertia a shape of this density has about its center of mass. Exposed because
// the analytic tests check it directly against the closed forms. A hull is the one kind whose answer
// is not in the shape itself, so it is handed its own run of the vertex pool as well.
BodyMass MassProperties(const Shape &, float density, std::span<const float3> shape_vertices = {});
