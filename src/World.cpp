#include "World.h"

#include "Hull.h"
#include "Mesh.h"

#include <algorithm>
#include <bit>
#include <numbers>
#include <utility>

namespace rbp {

namespace {
// A freed slot first, then the tail, with a refusal counted when neither has room.
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

// A free slot at the end of a pool still costs a thread in every dispatch, so the tail is released while it is free.
// A slot that is neither live nor on the free list is a body retired this step, which keeps its place until the next step reports its removals.
// The trim stops there.
void TrimTail(uint32_t &used, std::vector<Index> &free, const auto &live) {
    while (used > 0 && !live(used - 1) && std::erase(free, used - 1) != 0) --used;
}

using double3 = simd::double3;

// A unit quaternion as the matrix whose columns are the axes it turns onto.
// In double because the tensor sum below is a difference of much larger numbers wherever a piece sits well off the whole's centre, as with CookHull's integral.
void RotationMatrix(float4 q, double (&m)[3][3]) {
    const double x = q.x, y = q.y, z = q.z, w = q.w;
    m[0][0] = 1 - 2 * (y * y + z * z), m[0][1] = 2 * (x * y - z * w), m[0][2] = 2 * (x * z + y * w);
    m[1][0] = 2 * (x * y + z * w), m[1][1] = 1 - 2 * (x * x + z * z), m[1][2] = 2 * (y * z - x * w);
    m[2][0] = 2 * (x * z - y * w), m[2][1] = 2 * (y * z + x * w), m[2][2] = 1 - 2 * (x * x + y * y);
}

// A compound's children taken together, in the frame their poses are written in: the volume, its centre, and the inertia tensor about that centre.
// At unit density, because a compound is one material and density then cancels out of the centre and out of the frame the tensor is diagonal in.
struct Aggregate {
    double Volume{};
    double3 Center{0, 0, 0};
    double Tensor[3][3]{};
};

Aggregate WeighChildren(const Shape &compound, std::span<const float3> vertices, std::span<const Shape> shapes) {
    double volume[ChildrenPerCompound]{}, inertia[ChildrenPerCompound][3][3]{};
    double3 centre[ChildrenPerCompound]{};
    uint32_t count = 0;
    Aggregate whole;
    double3 moment{0, 0, 0};
    for (uint32_t i = 0; i < ChildrenPerCompound; ++i) {
        const Index child = ChildOf(compound, i);
        if (child == NoIndex || child >= shapes.size()) break; // the run's terminator, as Shape describes it
        const Shape &piece = shapes[child];
        const BodyMass own = MassProperties(piece, 1, vertices, shapes);
        if (!(own.InvMass > 0)) continue; // no volume, which AddCompound has already refused
        volume[count] = 1 / double(own.InvMass);
        // A shape's own frame is centred on its centre of mass, so Local's position is where that centre sits in the compound's frame.
        const float3 at = piece.Local.Position;
        centre[count] = double3{at.x, at.y, at.z};
        double turn[3][3];
        RotationMatrix(piece.Local.Orientation, turn);
        const float3 inverse = own.InvInertiaLocal;
        const double3 diagonal{inverse.x > 0 ? 1 / double(inverse.x) : 0, inverse.y > 0 ? 1 / double(inverse.y) : 0,
                               inverse.z > 0 ? 1 / double(inverse.z) : 0};
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                for (int k = 0; k < 3; ++k) inertia[count][r][c] += turn[r][k] * diagonal[k] * turn[c][k];
        moment += volume[count] * centre[count];
        whole.Volume += volume[count];
        ++count;
    }
    if (whole.Volume <= 0) return whole;
    whole.Center = moment / whole.Volume;
    for (uint32_t c = 0; c < count; ++c) {
        const double3 offset = centre[c] - whole.Center;
        const double square = dot(offset, offset);
        for (int r = 0; r < 3; ++r)
            for (int k = 0; k < 3; ++k)
                whole.Tensor[r][k] += inertia[c][r][k] + volume[c] * ((r == k ? square : 0) - offset[r] * offset[k]);
    }
    return whole;
}
// One face of a shape, in the frame the caller placed it in.
// Indexed as the device indexes it: BoxFaceIndex for a box, and its place in the run for a hull. See InternalFaces.
struct ShapeFace {
    uint32_t Index;
    float3 Normal;
    float Offset; // dot(Normal, a point on it), so two coplanar faces facing each other sum to zero
    std::vector<float3> Corner;
};

// `local` is where this shape's geometry sits in the frame the faces are wanted in.
// That is a compound child's Local, or a body's pose composed with it, which puts two static bodies in the world frame.
std::vector<ShapeFace> ShapeFaces(const Shape &shape, Pose local, std::span<const float3> vertices, std::span<const HullFace> faces) {
    std::vector<ShapeFace> out;
    if (shape.Kind == ShapeBox) {
        for (uint32_t axis = 0; axis < 3; ++axis) {
            const uint32_t u = (axis + 1) % 3, v = (axis + 2) % 3;
            for (const bool positive : {false, true}) {
                float3 outward{0, 0, 0};
                outward[axis] = positive ? 1.f : -1.f;
                ShapeFace face{.Index = BoxFaceIndex(axis, positive), .Normal = Rotate(local.Orientation, outward)};
                for (const auto [a, b] : {std::pair{-1.f, -1.f}, std::pair{1.f, -1.f}, std::pair{1.f, 1.f}, std::pair{-1.f, 1.f}}) {
                    float3 corner = shape.HalfExtents * outward;
                    corner[u] = a * shape.HalfExtents[u];
                    corner[v] = b * shape.HalfExtents[v];
                    face.Corner.push_back(WorldPoint(local, corner));
                }
                face.Offset = dot(face.Normal, face.Corner[0]);
                out.push_back(std::move(face));
            }
        }
    } else if (shape.Kind == ShapeHull) {
        for (uint32_t f = 0; f < shape.FaceCount && f < MaxInternalFaces; ++f) {
            const HullFace &held = faces[shape.FirstFace + f];
            ShapeFace face{.Index = f, .Normal = Rotate(local.Orientation, held.Normal)};
            for (uint32_t i = 0; i < held.Count; ++i) face.Corner.push_back(WorldPoint(local, vertices[shape.FirstVertex + held.Corner[i]]));
            if (face.Corner.size() < 3) continue;
            face.Offset = dot(face.Normal, face.Corner[0]);
            out.push_back(std::move(face));
        }
    }
    // A sphere and a capsule have no faces, so nothing of theirs is ever buried.
    return out;
}

// Whether every corner of `inner` lies within `outer`, both convex and in the same plane.
// The inside of an edge is taken from the centroid rather than from a winding, because a box's faces and a hull cook's are wound by different rules.
bool Within(const ShapeFace &inner, const ShapeFace &outer, float tolerance) {
    float3 centre{0, 0, 0};
    for (const float3 corner : outer.Corner) centre += corner;
    centre /= float(outer.Corner.size());
    for (size_t e = 0; e < outer.Corner.size(); ++e) {
        const float3 from = outer.Corner[e], to = outer.Corner[(e + 1) % outer.Corner.size()];
        const float3 inward = cross(outer.Normal, to - from);
        const float span = simd::length(inward);
        if (span < 1e-12f) continue; // an edge too short to give an inward direction
        const float sign = dot(inward, centre - from) >= 0 ? 1.f : -1.f;
        for (const float3 corner : inner.Corner)
            if (sign * dot(inward, corner - from) / span < -tolerance) return false;
    }
    return true;
}

// Which faces of each piece another piece has buried, one bit each, indexed as the device indexes the face, with the pieces all in one frame.
// A face counts only where the other's covers the whole of it, so a leg's top is buried in a slab and the slab's own bottom is not.
// Symmetric, and so independent of the order the pieces come in.
std::vector<uint32_t> BuriedFaces(std::span<const std::vector<ShapeFace>> pieces) {
    float scale = 1e-6f;
    for (const auto &piece : pieces)
        for (const ShapeFace &face : piece)
            for (const float3 corner : face.Corner) scale = std::max(scale, simd::length(corner));
    // Relative to where the pieces are, because the rounding in a dot product scales with its inputs, as the hull cook's coplanarity epsilon does.
    const float tolerance = 1e-5f * scale;
    std::vector<uint32_t> masks(pieces.size(), 0);
    for (size_t i = 0; i < pieces.size(); ++i)
        for (size_t j = 0; j < pieces.size(); ++j) {
            if (j == i) continue;
            for (const ShapeFace &mine : pieces[i])
                for (const ShapeFace &theirs : pieces[j]) {
                    if (dot(mine.Normal, theirs.Normal) > -0.99999f) continue; // not facing each other
                    if (std::abs(mine.Offset + theirs.Offset) > tolerance) continue; // not in one plane
                    if (Within(mine, theirs, tolerance)) masks[i] |= 1u << mine.Index;
                }
        }
    return masks;
}
} // namespace

BodyMass MassProperties(const Shape &shape, float density, std::span<const float3> shape_vertices, std::span<const Shape> shapes) {
    constexpr float Pi = std::numbers::pi_v<float>;
    // A plane is unbounded, a mesh is a surface with no interior, and a zero density is static by request. See StaticMass.
    if (shape.Kind == ShapePlane || shape.Kind == ShapeMesh || density <= 0) return StaticMass;

    if (shape.Kind == ShapeCompound) {
        // AddCompound left the children in the frame this diagonalizes to, so the moments come out in the order of the body frame's own axes.
        const Aggregate whole = WeighChildren(shape, shape_vertices, shapes);
        if (whole.Volume <= 0) return StaticMass;
        const double3 moments = DiagonalizeSymmetric(whole.Tensor).Values;
        return {.InvInertiaLocal = 1 / (float3{float(moments.x), float(moments.y), float(moments.z)} * density),
                .InvMass = 1 / float(whole.Volume * density)};
    }

    if (shape.Kind == ShapeHull) {
        // The vertices are already in the cook's frame, so this is the tetrahedra integral and no more.
        const CookedHull cooked = CookHull(shape_vertices.subspan(shape.FirstVertex, shape.VertexCount));
        if (cooked.Vertices.empty()) return StaticMass;
        return {.InvInertiaLocal = 1 / (cooked.Inertia * density), .InvMass = 1 / (cooked.Volume * density)};
    }

    if (shape.Kind == ShapeCapsule) {
        // A cylinder with a hemisphere on each end, the two caps making one sphere.
        // About the long axis each part contributes the inertia it would have alone.
        // Across it the caps shift out to the ends, and their second moment plus the parallel axis carry comes to h^2 + 3hr/4.
        const float radius = shape.Radius, half = shape.HalfExtents.y;
        const float cylinder = density * Pi * radius * radius * 2 * half;
        const float caps = density * 4.f / 3 * Pi * radius * radius * radius;
        const float along = cylinder * radius * radius / 2 + caps * 2.f / 5 * radius * radius;
        const float across = cylinder * (radius * radius / 4 + half * half / 3) +
            caps * (2.f / 5 * radius * radius + half * half + 3.f / 4 * half * radius);
        return {.InvInertiaLocal = 1 / float3{across, along, across}, .InvMass = 1 / (cylinder + caps)};
    }

    if (shape.Kind == ShapeSphere) {
        // Solid sphere about its center: m = rho 4/3 pi r^3, and I = 2/5 m r^2 about every axis.
        const float radius = shape.Radius;
        const float mass = density * 4.f / 3 * Pi * radius * radius * radius;
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

// An authored mass takes precedence, only the host being able to supply one for a shape with no volume.
// An authored zero passes through as a zero inverse, which is a locked axis. See AuthoredMass.
BodyMass World::ShapeOrAuthoredMass(Index shape, float density, std::optional<AuthoredMass> authored) const {
    if (authored) {
        const auto [mass, inertia] = *authored;
        return {.InvInertiaLocal = {inertia.x > 0 ? 1 / inertia.x : 0, inertia.y > 0 ? 1 / inertia.y : 0, inertia.z > 0 ? 1 / inertia.z : 0},
                .InvMass = mass > 0 ? 1 / mass : 0};
    }
    return shape == NoIndex ? StaticMass : MassProperties(Shapes[shape], density, ShapeVertices.All(), Shapes.All());
}

// See BodyDesc::Mass.
// The computed mass decides this rather than the shape's kind, a static body having no mass properties about a point to get wrong.
// Only the quaternion's vector part is checked, a zero vector part being the identity whichever sign w carries.
bool World::OffsetNeedsAuthoredMass(Index shape, const BodyMass &mass, bool authored) const {
    if (authored || shape == NoIndex || !Moves(mass)) return false;
    const auto [at, turn] = Shapes[shape].Local;
    return at.x != 0 || at.y != 0 || at.z != 0 || turn.x != 0 || turn.y != 0 || turn.z != 0;
}

World::World(const mtl::Context &context, WorldLimits limits) {
    auto *device = context.Device.get();
    // Metal 4 has no implicit residency tracking, so everything a kernel can reach is in one set attached to the queue for the world's lifetime.
    // The destructor takes it off again.
    NS::Error *error{};
    Residency = NS::TransferPtr(device->newResidencySet(mtl::Make<MTL::ResidencySetDescriptor>().get(), &error));
    // Every buffer starts zeroed, whatever the pages held before.
    // A world built after another gets that other's pages back.
    // A lane the solver reads before writing would otherwise tie run-to-run determinism to the allocations repeating.
    // The fill happens where the buffer is made.
    const auto make = [&]<typename T>(mtl::Buffer<T> &buffer, uint32_t capacity) {
        buffer = {device, capacity};
        Residency->addAllocation(buffer.Handle.get());
        std::ranges::fill(buffer.All(), T{});
    };
    // In the order World declares them, so the header's grouping by access pattern reads the same here.
    make(Poses, limits.Bodies);
    make(Velocities, limits.Bodies);
    make(Masses, limits.Bodies);
    make(BodyShapes, limits.Bodies);
    make(Shapes, limits.Shapes);
    make(ShapeVertices, limits.ShapeVertices);
    make(HullFaces, limits.HullFaces);
    make(Triangles, limits.Triangles);
    make(BvhNodes, limits.BvhNodes);
    make(Frictions, limits.Bodies);
    make(Restitutions, limits.Bodies);
    make(Filters, limits.Bodies);
    make(Jointed, limits.Bodies * JointsPerBody);
    make(InitialPoses, limits.Bodies);
    make(InertialPoses, limits.Bodies);
    make(PreviousVelocities, limits.Bodies);
    make(SolvedPoses, limits.Bodies);
    make(RestPoses, limits.Bodies);
    make(Quiet, limits.Bodies);
    make(NextQuiet, limits.Bodies);
    make(Colors, limits.Bodies);
    make(NextColors, limits.Bodies);
    make(Contacts, limits.Bodies * ContactsPerBody);
    make(Incoming, limits.Bodies);
    make(IncomingSlots, limits.Bodies * ContactsPerBody);
    make(ContactEvents, limits.Bodies * EventsPerBody);
    make(ContactEventCounts, limits.Bodies);
    make(ContactRefusals, limits.Bodies);
    make(Joints, limits.Joints);
    Residency->commit();
    Residency->requestResidency();
    Queue = context.Queue;
    Queue->addResidencySet(Residency.get());

    // The lanes whose empty value is NoIndex rather than zero, which is every lane naming something.
    for (auto *buffer : {&BodyShapes, &Jointed, &IncomingSlots}) std::ranges::fill(buffer->All(), NoIndex);

    VertexPool.Capacity = limits.ShapeVertices;
    FacePool.Capacity = limits.HullFaces;
    TrianglePool.Capacity = limits.Triangles;
    NodePool.Capacity = limits.BvhNodes;
    LiveBodies.assign(limits.Bodies, 0);
    LiveShapes.assign(limits.Shapes, 0);
    WeldedShapes.assign(limits.Bodies, NoIndex);
    Spawns.assign(limits.Bodies, 0);
}

World::~World() {
    // Null in a world that has been moved from, which owns nothing.
    if (Queue && Residency) {
        Queue->removeResidencySet(Residency.get());
        mtl::Drain(Queue.get()); // the removal reaches the driver before this returns. See mtl::Drain.
    }
}

// First fit rather than best fit, because a freed run is usually the size asked for next: a mesh replaced by an edited version of itself needs the same length.
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
    if (count == 0 || start == NoIndex) return; // a run the pool refused, so there is nothing to release
    // Back onto the tail if that is where it came from, so a shape added and taken away leaves nothing.
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

// Both directions: this body's own run holds the contacts where it is A, and last step's incoming list the ones where it is B.
// Both are needed because a plane and a mesh own no manifold, so everything standing on one names it as B and only the incoming list reaches those.
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

// This body's appearances in other bodies' runs are found by scanning the pool, deliberately not by walking Incoming.
// Incoming is last step's gather, and an earlier mutation in the same between-steps window may have compacted slots since.
// The list can therefore miss contacts the pool still holds.
//
// A run another body owns is compacted from its tail as this empties slots out of its middle, because a run must stay dense from zero.
// Every reader stops at the first inactive slot.
// The surviving order changes, which is safe because a contact is matched by feature rather than by slot.
void World::EndContacts(Index body) {
    const auto end = [this](Contact &contact) {
        if (TrackContacts)
            Changes.push_back({
                .A = {contact.BodyA, Spawns[contact.BodyA]},
                .B = {contact.BodyB, Spawns[contact.BodyB]},
                .Feature = contact.Feature,
                .SubShape = contact.SubShape,
                .Children = contact.Children,
                .Kind = ContactRemoved,
            });
        contact.Active = false;
    };
    for (uint32_t i = 0; i < ContactsPerBody; ++i) {
        Contact &contact = Contacts[body * ContactsPerBody + i];
        if (!contact.Active) break;
        end(contact);
    }
    for (Index owner = 0; owner < NumBodies; ++owner) {
        if (owner == body) continue;
        const auto run = Contacts.All().subspan(owner * ContactsPerBody, ContactsPerBody);
        uint32_t count = 0;
        while (count < ContactsPerBody && run[count].Active) ++count;
        for (uint32_t i = 0; i < count;) {
            if (run[i].BodyB != body) {
                ++i;
                continue;
            }
            end(run[i]);
            run[i] = run[--count];
            run[count].Active = false;
        }
    }
}

// A live contact's event takes its excitation record by position rather than by search.
// CollectContacts writes one event per live slot in slot order before EndUnclaimed appends the removals.
// A body's added and persisted events are therefore its contact run, in order.
void World::DrainContactEvents() {
    if (!TrackContacts) return;
    for (Index body = 0; body < NumBodies; ++body) {
        uint32_t live = 0;
        for (uint32_t i = 0; i < ContactEventCounts[body]; ++i) {
            const ContactEvent &event = ContactEvents[body * EventsPerBody + i];
            ContactChange change{
                .A = {event.BodyA, Spawns[event.BodyA]},
                .B = {event.BodyB, Spawns[event.BodyB]},
                .Feature = event.Feature,
                .SubShape = event.SubShape,
                .Children = event.Children,
                .Kind = ContactEventKind(event.Kind),
            };
            if (change.Kind != ContactRemoved) {
                const Contact &contact = Contacts[body * ContactsPerBody + live++];
                change.Lambda = contact.Lambda;
                change.Approach = contact.Approach;
                change.BounceImpulse = contact.BounceImpulse;
            }
            Changes.push_back(change);
        }
    }
}

Index World::AddShape(const Shape &shape) {
    const Index index = TakeSlot(FreeShapes, NumShapes, Shapes.Capacity, Overflow.Shapes);
    if (index == NoIndex) return NoIndex;
    Shapes[index] = shape;
    LiveShapes[index] = 1;
    return index;
}

Index World::AddHull(std::span<const float3> points, Pose *frame, std::optional<Pose> local) {
    const CookedHull cooked = CookHull(points);
    const uint32_t count = cooked.Vertices.size(), face_count = cooked.Faces.size();
    if (count == 0) return NoIndex; // no solid, so no shape to make of it
    if (frame != nullptr) *frame = cooked.Frame;
    // The caller's frame first and the cook's underneath it.
    // The other order would move the caller's offset by however far the cook shifted the centroid, a distance the caller never saw.
    const Pose shape_local = local ? ComposePose(*local, cooked.Frame) : IdentityPose;
    // Both runs up front, and the refusal is counted against whichever pool refused first.
    // The cook has already brought the corner count under MaxHullVertices, so only a full pool can refuse here.
    const Index first = VertexPool.Take(count);
    const Index first_face = FacePool.Take(face_count);
    uint32_t *refused = nullptr;
    if (first == NoIndex) refused = &Overflow.ShapeVertices;
    else if (first_face == NoIndex) refused = &Overflow.HullFaces;
    const Index shape = refused != nullptr ? NoIndex : AddShape({.FirstVertex = first, .VertexCount = count, .FirstFace = first_face, .FaceCount = face_count, .Kind = ShapeHull, .Local = shape_local});
    if (shape == NoIndex) { // on any refusal, release every run this took
        VertexPool.Give(first, count);
        FacePool.Give(first_face, face_count);
        if (refused != nullptr) ++*refused;
        return NoIndex;
    }
    std::ranges::copy(cooked.Vertices, ShapeVertices.All().begin() + first);
    std::ranges::copy(cooked.Faces, HullFaces.All().begin() + first_face);
    return shape;
}

Index World::AddMesh(std::span<const float3> points, std::span<const uint32_t> indices, Pose local) {
    const CookedMesh cooked = CookMesh(points, indices);
    if (cooked.Triangles.empty()) return NoIndex; // no surface, so no shape to make of it
    const uint32_t vertices = cooked.Vertices.size(), triangles = cooked.Triangles.size(), nodes = cooked.Nodes.size();
    // All three runs up front, as AddHull takes its two, so one exit releases whatever was taken.
    const Index first_vertex = VertexPool.Take(vertices);
    const Index first_triangle = TrianglePool.Take(triangles);
    const Index root = NodePool.Take(nodes);
    uint32_t *refused = nullptr;
    if (first_vertex == NoIndex) refused = &Overflow.ShapeVertices;
    else if (first_triangle == NoIndex) refused = &Overflow.Triangles;
    else if (root == NoIndex) refused = &Overflow.BvhNodes;
    const Index shape = refused != nullptr ? NoIndex : AddShape({.FirstVertex = first_vertex, .VertexCount = vertices, .FirstTriangle = first_triangle, .RootNode = root, .TriangleCount = triangles, .NodeCount = nodes, .Kind = ShapeMesh, .Local = local});
    if (shape == NoIndex) {
        VertexPool.Give(first_vertex, vertices);
        TrianglePool.Give(first_triangle, triangles);
        NodePool.Give(root, nodes);
        if (refused != nullptr) ++*refused;
        return NoIndex;
    }

    std::ranges::copy(cooked.Vertices, ShapeVertices.All().begin() + first_vertex);
    // Triangles index the pool absolutely, so a kernel reads a corner without resolving which mesh it belongs to.
    for (uint32_t i = 0; i < triangles; ++i) {
        Triangle triangle = cooked.Triangles[i];
        triangle.A += first_vertex;
        triangle.B += first_vertex;
        triangle.C += first_vertex;
        Triangles[first_triangle + i] = triangle;
    }
    // Nodes stay relative to their own root, because a traversal starts there.
    std::ranges::copy(cooked.Nodes, BvhNodes.All().begin() + root);
    return shape;
}

Index World::CopyShape(Index source, Pose local) {
    Shape copy = Shapes[source];
    copy.Local = local;
    // A hull's geometry is in the pools, and a copy sharing those runs could not be released on its own.
    // A face names its corners by their place in the shape's own run, so both runs copy unchanged.
    if (copy.Kind == ShapeHull) {
        const Index first = VertexPool.Take(copy.VertexCount);
        const Index first_face = FacePool.Take(copy.FaceCount);
        uint32_t *refused = nullptr;
        if (first == NoIndex) refused = &Overflow.ShapeVertices;
        else if (first_face == NoIndex) refused = &Overflow.HullFaces;
        if (refused != nullptr) {
            VertexPool.Give(first, copy.VertexCount);
            FacePool.Give(first_face, copy.FaceCount);
            ++*refused;
            return NoIndex;
        }
        const auto vertices = ShapeVertices.All();
        const auto faces = HullFaces.All();
        std::ranges::copy(vertices.subspan(copy.FirstVertex, copy.VertexCount), vertices.begin() + first);
        std::ranges::copy(faces.subspan(copy.FirstFace, copy.FaceCount), faces.begin() + first_face);
        copy.FirstVertex = first;
        copy.FirstFace = first_face;
    }
    const Index shape = AddShape(copy);
    if (shape == NoIndex && copy.Kind == ShapeHull) {
        VertexPool.Give(copy.FirstVertex, copy.VertexCount);
        FacePool.Give(copy.FirstFace, copy.FaceCount);
    }
    return shape;
}

void World::ReleaseShape(Index shape) {
    const Shape held = Shapes[shape];
    if (held.Kind == ShapeHull || held.Kind == ShapeMesh) VertexPool.Give(held.FirstVertex, held.VertexCount);
    if (held.Kind == ShapeHull) FacePool.Give(held.FirstFace, held.FaceCount);
    if (held.Kind == ShapeMesh) {
        TrianglePool.Give(held.FirstTriangle, held.TriangleCount);
        NodePool.Give(held.RootNode, held.NodeCount);
    }
    // A compound owns its children, so they are released with it. One level deep, a child never being a compound.
    if (held.Kind == ShapeCompound) {
        for (uint32_t i = 0; i < ChildrenPerCompound; ++i) {
            const Index child = ChildOf(held, i);
            if (child == NoIndex) break;
            ReleaseShape(child);
        }
    }
    LiveShapes[shape] = 0;
    FreeShapes.push_back(shape);
}

Index World::AddCompound(std::span<const Index> children, Pose *frame) {
    if (children.empty() || children.size() > ChildrenPerCompound) {
        ++RefusedCompounds;
        return NoIndex;
    }
    for (const Index child : children) {
        if (child >= NumShapes || !LiveShapes[child]) {
            ++RefusedCompounds;
            return NoIndex;
        }
        const uint32_t kind = Shapes[child].Kind;
        // Flat, and convex leaves only: a plane and a mesh have no volume and no manifold of their own.
        if (kind == ShapeCompound || kind == ShapeMesh || kind == ShapePlane) {
            ++RefusedCompounds;
            return NoIndex;
        }
    }

    // The pieces' mass properties in the frame they were handed in, which places the body frame.
    Shape given{.Kind = ShapeCompound};
    for (uint32_t i = 0; i < ChildrenPerCompound; ++i) SetChild(given, i, i < children.size() ? children[i] : NoIndex);
    const Aggregate whole = WeighChildren(given, ShapeVertices.All(), Shapes.All());
    if (whole.Volume <= 0) { // pieces enclosing nothing, which is a shape that cannot exist
        ++RefusedCompounds;
        return NoIndex;
    }
    const Diagonalized principal = DiagonalizeSymmetric(whole.Tensor);
    const Pose body{.Position = float3{float(whole.Center.x), float(whole.Center.y), float(whole.Center.z)},
                    .Orientation = principal.Orientation};
    if (frame != nullptr) *frame = body;

    // And every child at its pose within that frame.
    // The caller's shapes are untouched, the compound holding its own copies.
    const Pose inverse{.Position = Rotate(QuatConjugate(body.Orientation), -body.Position),
                       .Orientation = QuatConjugate(body.Orientation)};
    Shape compound{.Kind = ShapeCompound};
    for (uint32_t i = 0; i < ChildrenPerCompound; ++i) SetChild(compound, i, NoIndex);
    uint32_t made = 0;
    for (; made < children.size(); ++made) {
        const Index copy = CopyShape(children[made], ComposePose(inverse, Shapes[children[made]].Local));
        if (copy == NoIndex) break; // a pool or the slot table refused, and has counted it already
        SetChild(compound, made, copy);
    }
    const Index shape = made == children.size() ? AddShape(compound) : NoIndex;
    if (shape == NoIndex) { // on any refusal, release every slot and run this took
        for (uint32_t i = 0; i < made; ++i) ReleaseShape(ChildOf(compound, i));
        TrimTail(NumShapes, FreeShapes, [this](Index at) { return LiveShapes[at] != 0; });
        return NoIndex;
    }

    // And which of each child's faces are buried against a sibling, in the frame the children now share.
    // See InternalFaces, and World::WeldStatic, which computes the same for two static bodies.
    std::vector<std::vector<ShapeFace>> faces;
    for (uint32_t i = 0; i < made; ++i) {
        const Shape &child = Shapes[ChildOf(compound, i)];
        faces.push_back(ShapeFaces(child, child.Local, ShapeVertices.All(), HullFaces.All()));
    }
    const std::vector<uint32_t> masks = BuriedFaces(faces);
    for (uint32_t i = 0; i < made; ++i) SetInternalFaces(Shapes[ChildOf(compound, i)], masks[i]);
    return shape;
}

uint32_t World::WeldStatic() {
    // The bodies the weld covers. See the header for each exclusion.
    std::vector<Index> resting;
    std::vector<std::vector<ShapeFace>> faces;
    for (Index body = 0; body < NumBodies; ++body) {
        if (!LiveBodies[body] || Moves(Masses[body])) continue;
        const Velocity motion = Velocities[body];
        // Kinematic is a velocity and nothing more anywhere in this engine.
        // A driven slab may leave, and the face it was covering has to be a face again the moment it does.
        if (simd::length(motion.Linear) > 0 || simd::length(motion.Angular) > 0) continue;
        const Index shape = BodyShapes[body];
        if (shape == NoIndex) continue;
        const Shape &held = Shapes[shape];
        if (held.Kind != ShapeBox && held.Kind != ShapeHull) continue;
        resting.push_back(body);
        // In world space, the one frame two separate bodies share.
        faces.push_back(ShapeFaces(held, ComposePose(Poses[body], held.Local), ShapeVertices.All(), HullFaces.All()));
    }
    std::vector<uint32_t> wanted(NumBodies, 0);
    const std::vector<uint32_t> masks = BuriedFaces(faces);
    for (uint32_t i = 0; i < resting.size(); ++i) wanted[resting[i]] = masks[i];

    // Every live body rather than only the ones above.
    // A body given a mass or a velocity since the last call has left that list and is exactly the one needing its faces back.
    uint32_t buried = 0;
    for (Index body = 0; body < NumBodies; ++body) {
        if (!LiveBodies[body]) continue;
        const uint32_t mask = wanted[body];
        Index copy = WeldedShapes[body];
        if (copy == NoIndex) {
            if (mask == 0) continue; // nothing buried, and no earlier call's mark to clear
            // A shape is shared, so the mark goes on a copy this body owns. Refused, it stays unwelded.
            copy = CopyShape(BodyShapes[body], Shapes[BodyShapes[body]].Local);
            if (copy == NoIndex) continue;
            WeldedShapes[body] = copy;
            BodyShapes[body] = copy;
        }
        if (InternalFaces(Shapes[copy]) != mask) {
            SetInternalFaces(Shapes[copy], mask);
            // A body standing on a face just buried, or on one just restored, holds contacts about to be dropped or made.
            Wake(body);
        }
        buried += uint32_t(std::popcount(mask));
    }
    return buried;
}

// A weld copy belongs to the weld rather than to the host, so it is released with the body that wore it.
// The slot is cleared first so the release goes through RemoveShape, which refuses a shape while a second body wears it.
void World::DropWeld(Index body) {
    const Index copy = WeldedShapes[body];
    if (copy == NoIndex) return;
    WeldedShapes[body] = NoIndex;
    BodyShapes[body] = NoIndex;
    RemoveShape(copy);
}

Index World::AddBody(const BodyDesc &desc) {
    // Before a slot is taken, so a body the engine cannot integrate about its own frame is refused rather than half made. See BodyDesc::Mass.
    BodyMass mass = ShapeOrAuthoredMass(desc.Shape, desc.Density, desc.Mass);
    if (OffsetNeedsAuthoredMass(desc.Shape, mass, desc.Mass.has_value())) {
        ++OffsetsWithoutMass;
        return NoIndex;
    }
    const Index index = TakeSlot(FreeBodies, NumBodies, Poses.Capacity, Overflow.Bodies);
    if (index == NoIndex) return NoIndex;
    LiveBodies[index] = 1;
    // A new tenancy, so anything holding the last tenant's identity stops matching. See BodyId.
    ++Spawns[index];
    // A reused slot has to arrive in the state a fresh one would, or a scene's outcome depends on the body that held the slot before.
    // Most per-body lanes are written by a step before anything reads them, and EndContacts emptied the contact run.
    // The color and the incoming list are neither, so they are reset here.
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
    std::ranges::fill(JointedRun(index), NoIndex);
    // Mass properties come from the shape and motion properties from the body, sharing one lane because Integrate reads that lane.
    mass.GravityScale = desc.GravityScale;
    mass.LinearDamping = desc.LinearDamping;
    mass.AngularDamping = desc.AngularDamping;
    Masses[index] = mass;
    return index;
}

namespace {
// Three axis modes packed into the word the kernels read them out of, three bits each.
uint32_t Modes(const JointAxisMode (&axes)[3]) {
    return uint32_t(axes[0]) | (uint32_t(axes[1]) << 3) | (uint32_t(axes[2]) << 6);
}
} // namespace

Index World::AddJoint(const JointDesc &desc) {
    if (!Alive(desc.BodyA) || !Alive(desc.BodyB)) return NoIndex;
    const Index index = TakeSlot(FreeJoints, NumJoints, Joints.Capacity, Overflow.Joints);
    if (index == NoIndex) return NoIndex;
    const Pose a = Poses[desc.BodyA], b = Poses[desc.BodyB];
    // Each end's world point and the joint's world frame, stored in each body's own frame, so a joint holds the pose the bodies were in when it was made.
    // Two ends that do not coincide are an initial error the joint closes rather than a refusal.
    const float3 at_a = desc.AtA.value_or(desc.At), at_b = desc.AtB.value_or(desc.At);
    const float4 frame = desc.Frame.value_or(b.Orientation);
    if (!desc.Collide) { // record each as the other's partner, so neither generates contacts against it
        for (const auto [owner, partner] : {std::pair{desc.BodyA, desc.BodyB}, std::pair{desc.BodyB, desc.BodyA}}) {
            const auto run = JointedRun(owner);
            const auto gap = std::ranges::find(run, NoIndex);
            if (gap == run.end()) ++Overflow.Jointed;
            else *gap = partner;
        }
    }
    Joints[index] = {
        .AnchorA = LocalPoint(a, at_a),
        .AnchorB = LocalPoint(b, at_b),
        .FrameA = QuatMul(QuatConjugate(a.Orientation), frame),
        .FrameB = QuatMul(QuatConjugate(b.Orientation), frame),
        .LambdaLinear = {0, 0, 0},
        .LambdaAngular = {0, 0, 0},
        .PenaltyLinear = {1, 1, 1},
        .PenaltyAngular = {1, 1, 1},
        .MotorSpeed = desc.MotorSpeed,
        .MotorTarget = desc.MotorTarget,
        .MotorMaxTorque = desc.MotorMaxTorque,
        .LimitLow = desc.LimitLow,
        .LimitHigh = desc.LimitHigh,
        .LinearMotorSpeed = desc.LinearMotorSpeed,
        .LinearMotorTarget = desc.LinearMotorTarget,
        .LinearMotorMaxForce = desc.LinearMotorMaxForce,
        .LinearLimitLow = desc.LinearLimitLow,
        .LinearLimitHigh = desc.LinearLimitHigh,
        .LinearStiffness = desc.LinearStiffness,
        .AngularStiffness = desc.AngularStiffness,
        .LinearDamping = desc.LinearDamping,
        .AngularDamping = desc.AngularDamping,
        .BodyA = desc.BodyA,
        .BodyB = desc.BodyB,
        .LinearModes = Modes(desc.Linear),
        .AngularModes = Modes(desc.Angular),
        .Active = 1,
        .Suppresses = desc.Collide ? 0u : 1u,
    };
    return index;
}

bool World::RemoveBody(Index body) {
    if (!Alive(body)) return false;
    Wake(body);
    // Its joints go with it.
    // A removed body has no mass, so a joint to it would read as a joint to static geometry and go on holding the live end to a pose nothing maintains.
    for (Index joint = 0; joint < NumJoints; ++joint) {
        const Joint &held = Joints[joint];
        if (held.Active && (held.BodyA == body || held.BodyB == body)) RemoveJoint(joint);
    }
    // After Wake, which reads the runs this empties.
    // The removed body's slot is left with no shape and no mass, the condition every per-body kernel early-outs on.
    EndContacts(body);
    DropWeld(body); // the weld's copy is released with it, and the faces it buried are faces again
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
    if (held.Suppresses) { // undo exactly what AddJoint wrote: one entry each way and no more
        for (const auto [owner, partner] : {std::pair{held.BodyA, held.BodyB}, std::pair{held.BodyB, held.BodyA}}) {
            const auto run = JointedRun(owner);
            if (const auto at = std::ranges::find(run, partner); at != run.end()) *at = NoIndex;
        }
    }
    // Whatever the joint was holding up is now falling, and nothing else wakes either end.
    Quiet[held.BodyA] = 0;
    Quiet[held.BodyB] = 0;
    Joints[joint].Active = 0;
    FreeJoints.push_back(joint);
    // A joint carries nothing across steps that a kernel must read first, so its slot is free at once.
    // Every body scans the whole joint pool once per color per iteration.
    TrimTail(NumJoints, FreeJoints, [this](Index at) { return Joints[at].Active != 0; });
    return true;
}

bool World::RemoveShape(Index shape) {
    if (shape >= NumShapes || !LiveShapes[shape]) return false;
    for (Index body = 0; body < NumBodies; ++body)
        if (LiveBodies[body] && BodyShapes[body] == shape) return false; // a live body still uses it
    // And a child belongs to its compound, which would be left naming a reassigned slot.
    for (Index other = 0; other < NumShapes; ++other) {
        if (!LiveShapes[other] || Shapes[other].Kind != ShapeCompound) continue;
        const Shape parent = Shapes[other];
        for (uint32_t i = 0; i < ChildrenPerCompound; ++i) {
            const Index child = ChildOf(parent, i);
            if (child == NoIndex) break;
            if (child == shape) return false;
        }
    }
    ReleaseShape(shape);
    TrimTail(NumShapes, FreeShapes, [this](Index at) { return LiveShapes[at] != 0; });
    return true;
}

bool World::SetBodyShape(Index body, Index shape, float density, std::optional<AuthoredMass> authored) {
    if (!Alive(body)) return false;
    if (shape != NoIndex && (shape >= NumShapes || !LiveShapes[shape])) return false;
    // Computed before anything is written, so a shape the engine cannot integrate about the body's frame leaves the body exactly as it was. See BodyDesc::Mass.
    BodyMass mass = ShapeOrAuthoredMass(shape, density, authored);
    if (OffsetNeedsAuthoredMass(shape, mass, authored.has_value())) {
        ++OffsetsWithoutMass;
        return false;
    }
    Wake(body);
    // Its contacts go with the geometry that made them, on both sides of the manifold.
    // A warm-started contact is matched by a feature naming faces and vertices of the replaced shape.
    // An identically named feature of the new shape would otherwise inherit the old dual, penalty and anchors.
    EndContacts(body);
    // And the weld's copy of the replaced geometry is released with it, unless the body has been handed the copy it already wears.
    // A host that read BodyShapes back passes that copy.
    if (shape != WeldedShapes[body]) DropWeld(body);
    BodyShapes[body] = shape;
    const BodyMass held = Masses[body];
    mass.GravityScale = held.GravityScale;
    mass.LinearDamping = held.LinearDamping;
    mass.AngularDamping = held.AngularDamping;
    Masses[body] = mass;
    return true;
}

void World::OnStepped() {
    // The step's events first, so the queue holds every event of the step before RemoveBody or SetBodyShape can append a synthesized removal.
    DrainContactEvents();
    // A body removed between steps is not recycled until a step has run.
    // The event runs the previous step wrote still name it until this step overwrites them, and a slot standing idle for the step keeps those readable.
    // BodyId's spawn counter covers anything held longer.
    for (const Index body : RetiredBodies) FreeBodies.push_back(body);
    RetiredBodies.clear();
    // And only then the tail, since a body waiting to report its removals keeps its place to do so.
    TrimTail(NumBodies, FreeBodies, [this](Index at) { return LiveBodies[at] != 0; });
}

} // namespace rbp
