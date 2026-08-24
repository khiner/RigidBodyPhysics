#include "World.h"

#include "Hull.h"
#include "Mesh.h"

#include <algorithm>
#include <utility>

namespace {
// A slot of one of the fixed pools: whatever was given back first, then the tail, and a refusal
// counted when neither has room. Every pool here is a bump pointer and a free list of bare indices,
// so the three of them differ only in which arrays they are.
Index TakeSlot(std::vector<Index> &free, uint32_t &used, uint32_t capacity, uint32_t &overflow) {
    if (!free.empty()) {
        const Index index = free.back();
        free.pop_back();
        return index;
    }
    if (used < capacity) return used++;
    ++overflow;
    return NoIndex;
}
} // namespace

BodyMass MassProperties(const Shape &shape, float density, std::span<const float3> shape_vertices) {
    // A plane is unbounded, a mesh is a surface with no inside, and a zero-density body is static by
    // request. All three get zero inverse quantities, which is what keeps them fixed without the solve
    // needing to branch on a flag.
    if (shape.Kind == ShapePlane || shape.Kind == ShapeMesh || density <= 0) return StaticMass;

    if (shape.Kind == ShapeHull) {
        // Taken from the vertices rather than remembered from the cook, so there is one answer to this
        // question and no cache of it to fall out of step. They are already in the cook's frame, so
        // this is the integral over the tetrahedra and nothing more.
        const CookedHull cooked = CookHull(shape_vertices.subspan(shape.FirstVertex, shape.VertexCount));
        if (cooked.Vertices.empty()) return StaticMass;
        return {.InvInertiaLocal = 1 / (cooked.Inertia * density), .InvMass = 1 / (cooked.Volume * density)};
    }

    if (shape.Kind == ShapeCapsule) {
        // A cylinder with a hemisphere on each end, which together make one sphere. About the long
        // axis both parts are what they would be alone. Across it, the two hemispheres shift out to
        // the ends: their own second moment, plus the parallel axis carry, comes to h^2 + 3hr/4 once
        // the offset of a hemisphere's center of mass from its flat face cancels out of the algebra.
        const float radius = shape.Radius, half = shape.HalfExtents.y;
        const float pi = 3.14159265358979f;
        const float cylinder = density * pi * radius * radius * 2 * half;
        const float caps = density * 4.f / 3 * pi * radius * radius * radius;
        const float along = cylinder * radius * radius / 2 + caps * 2.f / 5 * radius * radius;
        const float across = cylinder * (radius * radius / 4 + half * half / 3) +
            caps * (2.f / 5 * radius * radius + half * half + 3.f / 4 * half * radius);
        return {.InvInertiaLocal = 1 / float3{across, along, across}, .InvMass = 1 / (cylinder + caps)};
    }

    if (shape.Kind == ShapeSphere) {
        // Solid sphere about its center: m = rho 4/3 pi r^3, and I = 2/5 m r^2 about every axis.
        const float radius = shape.Radius;
        const float mass = density * 4.f / 3 * 3.14159265358979f * radius * radius * radius;
        const float inertia = 2.f / 5 * mass * radius * radius;
        return {.InvInertiaLocal = 1 / float3{inertia, inertia, inertia}, .InvMass = 1 / mass};
    }

    const float3 extents = 2 * shape.HalfExtents;
    const float mass = density * extents.x * extents.y * extents.z;
    // Solid box about its center: I_x = m (e_y^2 + e_z^2) / 12, and cyclically.
    const float3 squared = extents * extents;
    const float3 inertia = mass / 12 * float3{squared.y + squared.z, squared.x + squared.z, squared.x + squared.y};
    return {.InvInertiaLocal = 1 / inertia, .InvMass = 1 / mass};
}

World::World(const mtl::Context &context, WorldLimits limits) {
    auto *device = context.Device.get();
    Poses = {device, limits.Bodies};
    Velocities = {device, limits.Bodies};
    Masses = {device, limits.Bodies};
    BodyShapes = {device, limits.Bodies};
    Shapes = {device, limits.Shapes};
    ShapeVertices = {device, limits.ShapeVertices};
    HullFaces = {device, limits.HullFaces};
    Triangles = {device, limits.Triangles};
    BvhNodes = {device, limits.BvhNodes};
    Frictions = {device, limits.Bodies};
    Restitutions = {device, limits.Bodies};
    Filters = {device, limits.Bodies};
    Jointed = {device, limits.Bodies * JointsPerBody};
    InitialPoses = {device, limits.Bodies};
    InertialPoses = {device, limits.Bodies};
    PreviousVelocities = {device, limits.Bodies};
    SolvedPoses = {device, limits.Bodies};
    RestPoses = {device, limits.Bodies};
    Quiet = {device, limits.Bodies};
    NextQuiet = {device, limits.Bodies};
    Colors = {device, limits.Bodies};
    NextColors = {device, limits.Bodies};
    std::ranges::fill(Colors.All(), 0u);
    Contacts = {device, limits.Bodies * ContactsPerBody};
    std::ranges::fill(Contacts.All(), Contact{});
    Incoming = {device, limits.Bodies};
    IncomingSlots = {device, limits.Bodies * ContactsPerBody};
    ContactEvents = {device, limits.Bodies * EventsPerBody};
    ContactEventCounts = {device, limits.Bodies};
    std::ranges::fill(ContactEventCounts.All(), 0u);
    ContactRefusals = {device, limits.Bodies};
    std::ranges::fill(ContactRefusals.All(), 0u);
    Joints = {device, limits.Joints};
    std::ranges::fill(Joints.All(), Joint{});
    // Read by Wake before the first step has written one, so it has to mean something from the start.
    std::ranges::fill(Incoming.All(), Adjacency{});

    VertexPool.Capacity = limits.ShapeVertices;
    FacePool.Capacity = limits.HullFaces;
    TrianglePool.Capacity = limits.Triangles;
    NodePool.Capacity = limits.BvhNodes;
    LiveBodies.assign(limits.Bodies, 0);
    LiveShapes.assign(limits.Shapes, 0);

    // Metal 4 has no implicit residency tracking, so everything a kernel can reach lives in one set
    // attached to the queue for the world's lifetime - and taken off it again when that ends, which is
    // what the destructor is for.
    NS::Error *error{};
    Residency = NS::TransferPtr(device->newResidencySet(mtl::Make<MTL::ResidencySetDescriptor>().get(), &error));
    for (auto *buffer : {Poses.Handle.get(), Velocities.Handle.get(), Masses.Handle.get(), BodyShapes.Handle.get(), Shapes.Handle.get(), ShapeVertices.Handle.get(), HullFaces.Handle.get(), Triangles.Handle.get(), BvhNodes.Handle.get(), Frictions.Handle.get(), Restitutions.Handle.get(), Filters.Handle.get(), Jointed.Handle.get(), InitialPoses.Handle.get(), InertialPoses.Handle.get(), PreviousVelocities.Handle.get(), SolvedPoses.Handle.get(), RestPoses.Handle.get(), Quiet.Handle.get(), NextQuiet.Handle.get(), Colors.Handle.get(), NextColors.Handle.get(), Contacts.Handle.get(), Joints.Handle.get(), Incoming.Handle.get(), IncomingSlots.Handle.get(), ContactEvents.Handle.get(), ContactEventCounts.Handle.get(), ContactRefusals.Handle.get()})
        Residency->addAllocation(buffer);
    Residency->commit();
    Residency->requestResidency();
    Queue = context.Queue;
    Queue->addResidencySet(Residency.get());
}

World::~World() {
    // Null in a world that has been moved from, which owns nothing and has nothing to give back.
    if (Queue && Residency) Queue->removeResidencySet(Residency.get());
}

// First fit, and the tail when nothing fits. First fit rather than best fit because the runs given
// back are the runs asked for again - a mesh replaced by an edited version of itself wants the same
// length - and searching for a tighter hole would only find the same one further down the list.
Index World::RunPool::Take(uint32_t count) {
    for (auto run = Free.begin(); run != Free.end(); ++run) {
        if (run->Count < count) continue;
        const Index start = run->Start;
        if (run->Count == count) Free.erase(run);
        else *run = {start + count, run->Count - count};
        return start;
    }
    if (Used + count > Capacity) return NoIndex;
    const Index start = Used;
    Used += count;
    return start;
}

void World::RunPool::Give(Index start, uint32_t count) {
    if (count == 0) return;
    // Back onto the tail if that is where it came from, so a shape added and taken away again leaves
    // the pool exactly as it was rather than as a hole at the end of it.
    if (start + count == Used) {
        Used = start;
        while (!Free.empty() && Free.back().Start + Free.back().Count == Used) {
            Used = Free.back().Start;
            Free.pop_back();
        }
        return;
    }
    const auto at = std::ranges::lower_bound(Free, start, {}, &Run::Start);
    const auto run = Free.insert(at, {start, count});
    // Merge with the neighbour on each side, the later one first so the earlier merge does not move it.
    const auto next = run + 1;
    if (next != Free.end() && run->Start + run->Count == next->Start) {
        run->Count += next->Count;
        Free.erase(next);
    }
    if (run != Free.begin() && (run - 1)->Start + (run - 1)->Count == run->Start) {
        (run - 1)->Count += run->Count;
        Free.erase(run);
    }
}

// Everything this body is touching, woken, in both directions: its own run holds the contacts where
// it is A, and the list gathered by the last step holds the ones where it is B. Without this a body
// pulled out from under a sleeping stack leaves the stack asleep on nothing - waking spreads from
// whatever is moving, and a body that has been removed is not moving.
void World::Wake(Index body) {
    Quiet[body] = 0;
    for (uint32_t i = 0; i < ContactsPerBody; ++i) {
        const Contact &contact = Contacts[body * ContactsPerBody + i];
        if (contact.Active) Quiet[contact.BodyB] = 0;
    }
    const Adjacency incoming = Incoming[body];
    for (uint32_t i = 0; i < incoming.Count; ++i) {
        const Contact &contact = Contacts[IncomingSlots[incoming.Start + i]];
        if (contact.Active) Quiet[contact.BodyA] = 0;
    }
}

Index World::AddShape(const Shape &shape) {
    const Index index = TakeSlot(FreeShapes, NumShapes, Shapes.Capacity, Overflow.Shapes);
    if (index == NoIndex) return NoIndex;
    Shapes[index] = shape;
    LiveShapes[index] = 1;
    return index;
}

Index World::AddHull(std::span<const float3> points, Pose *frame) {
    const CookedHull cooked = CookHull(points);
    const uint32_t count = cooked.Vertices.size();
    if (count == 0) return NoIndex; // no solid, so no shape to make of it
    if (frame != nullptr) *frame = cooked.Frame;
    if (count > MaxHullVertices) return ++Overflow.ShapeVertices, NoIndex;
    const Index first = VertexPool.Take(count);
    if (first == NoIndex) return ++Overflow.ShapeVertices, NoIndex;
    const uint32_t face_count = cooked.Faces.size();
    const Index first_face = FacePool.Take(face_count);
    if (first_face == NoIndex) {
        VertexPool.Give(first, count);
        return ++Overflow.HullFaces, NoIndex;
    }
    for (uint32_t i = 0; i < count; ++i) ShapeVertices[first + i] = cooked.Vertices[i];
    for (uint32_t i = 0; i < face_count; ++i) HullFaces[first_face + i] = cooked.Faces[i];
    const Index shape = AddShape({.FirstVertex = first, .VertexCount = count, .FirstFace = first_face, .FaceCount = face_count, .Kind = ShapeHull});
    if (shape == NoIndex) { // the shape pool refused it, so give both runs back
        VertexPool.Give(first, count);
        FacePool.Give(first_face, face_count);
    }
    return shape;
}

Index World::AddMesh(std::span<const float3> points, std::span<const uint32_t> indices) {
    const CookedMesh cooked = CookMesh(points, indices);
    if (cooked.Triangles.empty()) return NoIndex; // no surface, so no shape to make of it
    const uint32_t vertices = cooked.Vertices.size(), triangles = cooked.Triangles.size(), nodes = cooked.Nodes.size();
    const Index first_vertex = VertexPool.Take(vertices);
    if (first_vertex == NoIndex) return ++Overflow.ShapeVertices, NoIndex;
    const Index first_triangle = TrianglePool.Take(triangles);
    if (first_triangle == NoIndex) {
        VertexPool.Give(first_vertex, vertices);
        return ++Overflow.Triangles, NoIndex;
    }
    const Index root = NodePool.Take(nodes);
    if (root == NoIndex) {
        VertexPool.Give(first_vertex, vertices);
        TrianglePool.Give(first_triangle, triangles);
        return ++Overflow.BvhNodes, NoIndex;
    }

    for (uint32_t i = 0; i < vertices; ++i) ShapeVertices[first_vertex + i] = cooked.Vertices[i];
    // Triangles index the pool absolutely, so a kernel reads a corner without knowing whose mesh it is.
    for (uint32_t i = 0; i < triangles; ++i) {
        Triangle triangle = cooked.Triangles[i];
        triangle.A += first_vertex;
        triangle.B += first_vertex;
        triangle.C += first_vertex;
        Triangles[first_triangle + i] = triangle;
    }
    // Nodes stay relative to their own root, since a traversal starts there and knows where it is.
    for (uint32_t i = 0; i < nodes; ++i) BvhNodes[root + i] = cooked.Nodes[i];

    const Index shape = AddShape({.FirstVertex = first_vertex,
                                  .VertexCount = vertices,
                                  .FirstTriangle = first_triangle,
                                  .RootNode = root,
                                  .TriangleCount = triangles,
                                  .NodeCount = nodes,
                                  .Kind = ShapeMesh});
    if (shape == NoIndex) { // the shape pool refused it, so give back everything it took
        VertexPool.Give(first_vertex, vertices);
        TrianglePool.Give(first_triangle, triangles);
        NodePool.Give(root, nodes);
    }
    return shape;
}

Index World::AddBody(const BodyDesc &desc) {
    const Index index = TakeSlot(FreeBodies, NumBodies, Poses.Capacity, Overflow.Bodies);
    if (index == NoIndex) return NoIndex;
    LiveBodies[index] = 1;
    // A slot handed out again has to arrive in the state a fresh one would, or the world a scene ends
    // up in depends on what stood in that slot before it. Most per-body lanes are written by a step
    // before anything reads them, and the contact run was emptied by the removed body's own pass
    // through CollectContacts, which is what the wait in OnStepped is for. The colour is the one thing
    // that is neither, colouring being incremental.
    Colors[index] = 0;
    Incoming[index] = {};
    Poses[index] = desc.Pose;
    Velocities[index] = desc.Velocity;
    PreviousVelocities[index] = desc.Velocity;
    BodyShapes[index] = desc.Shape;
    Quiet[index] = 0;
    RestPoses[index] = desc.Pose;
    Frictions[index] = desc.Friction;
    Restitutions[index] = desc.Restitution;
    Filters[index] = {.Layer = desc.Layer, .Collides = desc.CollidesWith};
    for (uint32_t i = 0; i < JointsPerBody; ++i) Jointed[index * JointsPerBody + i] = NoIndex;
    Masses[index] = desc.Shape == NoIndex ? StaticMass : MassProperties(Shapes[desc.Shape], desc.Density, ShapeVertices.All());
    return index;
}

Index World::AddJoint(const JointDesc &desc) {
    if (!Alive(desc.BodyA) || !Alive(desc.BodyB)) return NoIndex;
    const Index index = TakeSlot(FreeJoints, NumJoints, Joints.Capacity, Overflow.Joints);
    if (index == NoIndex) return NoIndex;
    const Pose a = Poses[desc.BodyA], b = Poses[desc.BodyB];
    // The world point, remembered in each body's own frame, and the rotation between them as it
    // stands - so a joint holds whatever pose the bodies were in when it was made.
    if (!desc.Collide) { // remember each as the other's, so neither generates contacts against it
        for (const auto pair : {std::pair{desc.BodyA, desc.BodyB}, std::pair{desc.BodyB, desc.BodyA}}) {
            uint32_t at = 0;
            while (at < JointsPerBody && Jointed[pair.first * JointsPerBody + at] != NoIndex) ++at;
            if (at == JointsPerBody) ++Overflow.Jointed;
            else Jointed[pair.first * JointsPerBody + at] = pair.second;
        }
    }
    Joints[index] = {
        .AnchorA = Rotate(QuatConjugate(a.Orientation), desc.At - a.Position),
        .AnchorB = Rotate(QuatConjugate(b.Orientation), desc.At - b.Position),
        .RestRotation = QuatMul(QuatConjugate(b.Orientation), a.Orientation),
        .LambdaLinear = {0, 0, 0},
        .LambdaAngular = {0, 0, 0},
        .PenaltyLinear = {1, 1, 1},
        .PenaltyAngular = {1, 1, 1},
        .MotorSpeed = desc.MotorSpeed,
        .MotorTarget = desc.MotorTarget,
        .MotorMaxTorque = desc.MotorMaxTorque,
        .LimitLow = desc.LimitLow,
        .LimitHigh = desc.LimitHigh,
        .LinearStiffness = desc.LinearStiffness,
        .AngularStiffness = desc.AngularStiffness,
        .BodyA = desc.BodyA,
        .BodyB = desc.BodyB,
        .AxisModes = uint(desc.Angular[0]) | (uint(desc.Angular[1]) << 3) | (uint(desc.Angular[2]) << 6),
        .Active = 1,
        .Suppresses = desc.Collide ? 0u : 1u,
    };
    return index;
}

bool World::RemoveBody(Index body) {
    if (!Alive(body)) return false;
    Wake(body);
    // Its joints go with it. A joint whose other end has been removed reads as a joint to a static
    // body, since a dead one has no mass - so it would go on holding the live end to a pose nothing
    // is keeping, and say nothing about it.
    for (Index joint = 0; joint < NumJoints; ++joint) {
        const Joint &held = Joints[joint];
        if (held.Active && (held.BodyA == body || held.BodyB == body)) RemoveJoint(joint);
    }
    // The tombstone is a body the engine already knows how to leave alone: no shape and no mass is what
    // every per-body kernel early-outs on, so nothing on the device has to be told a body has gone.
    //
    // Its contacts are deliberately left exactly as they are. Next step's CollectContacts reads them
    // to find what this body was holding, reports every one of them removed through EndUnclaimed, and
    // clears the run itself. Clearing it here instead would swallow those events.
    BodyShapes[body] = NoIndex;
    Masses[body] = StaticMass;
    Velocities[body] = {};
    LiveBodies[body] = 0;
    RetiredBodies.push_back(body);
    return true;
}

bool World::RemoveJoint(Index joint) {
    if (joint >= NumJoints || !Joints[joint].Active) return false;
    const Joint &held = Joints[joint];
    if (held.Suppresses) { // undo exactly what AddJoint wrote, which is one entry each way and no more
        for (const auto pair : {std::pair{held.BodyA, held.BodyB}, std::pair{held.BodyB, held.BodyA}}) {
            for (uint32_t at = 0; at < JointsPerBody; ++at) {
                if (Jointed[pair.first * JointsPerBody + at] != pair.second) continue;
                // A hole rather than a compaction: the kernel that reads this run sweeps all of it and
                // AddJoint fills at the first gap, so neither cares where the gap is.
                Jointed[pair.first * JointsPerBody + at] = NoIndex;
                break;
            }
        }
    }
    // Whatever the joint was holding up is now falling, and neither end will hear it from anything else.
    Quiet[held.BodyA] = 0;
    Quiet[held.BodyB] = 0;
    Joints[joint].Active = 0;
    FreeJoints.push_back(joint);
    // A joint carries nothing from one step to the next that a kernel has to read first, so its slot
    // is free at once. Give the tail back while it is free: every body scans the whole joint pool once
    // a colour once an iteration, which is the most expensive place a dead slot could sit.
    while (NumJoints > 0 && !Joints[NumJoints - 1].Active) {
        std::erase(FreeJoints, NumJoints - 1);
        --NumJoints;
    }
    return true;
}

bool World::RemoveShape(Index shape) {
    if (shape >= NumShapes || !LiveShapes[shape]) return false;
    for (Index body = 0; body < NumBodies; ++body)
        if (LiveBodies[body] && BodyShapes[body] == shape) return false; // a live body still has it
    const Shape &held = Shapes[shape];
    if (held.Kind == ShapeHull || held.Kind == ShapeMesh) VertexPool.Give(held.FirstVertex, held.VertexCount);
    if (held.Kind == ShapeHull) FacePool.Give(held.FirstFace, held.FaceCount);
    if (held.Kind == ShapeMesh) {
        TrianglePool.Give(held.FirstTriangle, held.TriangleCount);
        NodePool.Give(held.RootNode, held.NodeCount);
    }
    LiveShapes[shape] = 0;
    FreeShapes.push_back(shape);
    while (NumShapes > 0 && !LiveShapes[NumShapes - 1]) {
        std::erase(FreeShapes, NumShapes - 1);
        --NumShapes;
    }
    return true;
}

bool World::SetBodyShape(Index body, Index shape, float density) {
    if (!Alive(body)) return false;
    if (shape != NoIndex && (shape >= NumShapes || !LiveShapes[shape])) return false;
    Wake(body);
    // The contacts this body owns go with the geometry that made them: a warm-started contact is found
    // by a feature naming faces and vertices of the shape that has just been replaced, so a feature of
    // the new shape named the same would be handed the old one's dual, penalty and stick anchors. The
    // contacts where this body is B need nothing, since their owners re-collide against the new shape.
    //
    // What it costs is a spurious ContactRemoved for each, EndUnclaimed reporting a run that is empty.
    for (uint32_t i = 0; i < ContactsPerBody; ++i) Contacts[body * ContactsPerBody + i].Active = false;
    BodyShapes[body] = shape;
    Masses[body] = shape == NoIndex ? StaticMass : MassProperties(Shapes[shape], density, ShapeVertices.All());
    return true;
}

void World::OnStepped() {
    // A body removed during a step is not recycled until the step after it has run. That step is what
    // reports everything it was holding as gone, attributed to the body that was holding it rather
    // than to whoever moved into the slot, and it is what leaves the contact run empty for the next
    // tenant. One step of a slot standing idle buys both.
    for (const Index body : RetiredBodies) FreeBodies.push_back(body);
    RetiredBodies.clear();
    // And a free slot at the end of the pool is a thread in every dispatch, so give the tail back while
    // it is free. After the promotion above and not before it: a body still waiting for its removals to
    // be reported has to keep its place in the dispatch to report them.
    while (NumBodies > 0 && !LiveBodies[NumBodies - 1]) {
        if (std::erase(FreeBodies, NumBodies - 1) == 0) break; // retired rather than free
        --NumBodies;
    }
}
