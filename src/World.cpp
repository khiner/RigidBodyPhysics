#include "World.h"

#include "Hull.h"
#include "Mesh.h"

#include <algorithm>
#include <bit>
#include <numbers>
#include <stdexcept>
#include <utility>

namespace rbp {

namespace {

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

// Trim free tail slots so GPU dispatches do not visit them.
void TrimTail(uint32_t &used, std::vector<Index> &free, const auto &live) {
    while (used > 0 && !live(used - 1) && std::erase(free, used - 1) != 0) --used;
}

using double3 = simd::double3;

void RotationMatrix(float4 q, double (&m)[3][3]) {
    const double x = q.x, y = q.y, z = q.z, w = q.w;
    m[0][0] = 1 - 2 * (y * y + z * z), m[0][1] = 2 * (x * y - z * w), m[0][2] = 2 * (x * z + y * w);
    m[1][0] = 2 * (x * y + z * w), m[1][1] = 1 - 2 * (x * x + z * z), m[1][2] = 2 * (y * z - x * w);
    m[2][0] = 2 * (x * z - y * w), m[2][1] = 2 * (y * z + x * w), m[2][2] = 1 - 2 * (x * x + y * y);
}

struct Aggregate {
    double Volume{};
    double3 Center{0, 0, 0};
    double Tensor[3][3]{};
};

Aggregate WeighChildren(const Shape &compound, std::span<const float3> vertices, std::span<const Shape> shapes, std::span<const Index> children) {
    if (compound.FirstVertex > children.size() || compound.VertexCount > children.size() - compound.FirstVertex)
        throw std::invalid_argument("Compound mass properties require its child-index pool.");
    std::vector<Aggregate> pieces;
    pieces.reserve(compound.VertexCount);
    Aggregate whole;
    double3 moment{0, 0, 0};
    for (uint32_t i = 0; i < compound.VertexCount; ++i) {
        const Index child = ChildOf(compound, i, children.data());
        if (child == NoIndex || child >= shapes.size()) break;
        const Shape &piece = shapes[child];
        const BodyMass own = MassProperties(piece, 1, vertices, shapes);
        if (!(own.InvMass > 0)) continue;
        Aggregate &mass = pieces.emplace_back();
        mass.Volume = 1 / double(own.InvMass);

        const float3 at = piece.Local.Position;
        mass.Center = double3{at.x, at.y, at.z};
        double turn[3][3];
        RotationMatrix(piece.Local.Orientation, turn);
        const float3 inverse = own.InvInertiaLocal;
        const double3 diagonal{inverse.x > 0 ? 1 / double(inverse.x) : 0, inverse.y > 0 ? 1 / double(inverse.y) : 0, inverse.z > 0 ? 1 / double(inverse.z) : 0};
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                for (int k = 0; k < 3; ++k) mass.Tensor[r][c] += turn[r][k] * diagonal[k] * turn[c][k];
        moment += mass.Volume * mass.Center;
        whole.Volume += mass.Volume;
    }
    if (whole.Volume <= 0) return whole;
    whole.Center = moment / whole.Volume;
    for (const Aggregate &mass : pieces) {
        const double3 offset = mass.Center - whole.Center;
        const double square = dot(offset, offset);
        for (int r = 0; r < 3; ++r)
            for (int k = 0; k < 3; ++k)
                whole.Tensor[r][k] += mass.Tensor[r][k] + mass.Volume * ((r == k ? square : 0) - offset[r] * offset[k]);
    }
    return whole;
}

struct ShapeFace {
    uint32_t Index;
    float3 Normal;
    float Offset;
    std::vector<float3> Corner;
    float Radius = 0;
};

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
    } else if (shape.Kind == ShapeCylinder) {
        for (uint32_t end = 0; end < 2; ++end) {
            const float3 normal = Rotate(local.Orientation, float3{0, end ? 1.f : -1.f, 0});
            const float3 center = local.Position + shape.HalfExtents.y * normal;
            out.push_back({.Index = end, .Normal = normal, .Offset = dot(normal, center), .Corner = {center}, .Radius = shape.HalfExtents.x});
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

    return out;
}

bool Within(const ShapeFace &inner, const ShapeFace &outer, float tolerance) {
    if (outer.Radius > 0) {
        for (const float3 corner : inner.Corner)
            if (simd::length(corner - outer.Corner[0]) + inner.Radius > outer.Radius + tolerance) return false;
        return true;
    }
    float3 centre{0, 0, 0};
    for (const float3 corner : outer.Corner) centre += corner;
    centre /= float(outer.Corner.size());
    for (size_t e = 0; e < outer.Corner.size(); ++e) {
        const float3 from = outer.Corner[e], to = outer.Corner[(e + 1) % outer.Corner.size()];
        const float3 inward = cross(outer.Normal, to - from);
        const float span = simd::length(inward);
        if (span < 1e-12f) continue;
        const float sign = dot(inward, centre - from) >= 0 ? 1.f : -1.f;
        for (const float3 corner : inner.Corner)
            if (sign * dot(inward, corner - from) / span < inner.Radius - tolerance) return false;
    }
    return true;
}

std::vector<uint32_t> BuriedFaces(std::span<const std::vector<ShapeFace>> pieces, std::span<const CollisionMask> filters = {}) {
    float scale = 1e-6f;
    for (const auto &piece : pieces)
        for (const ShapeFace &face : piece)
            for (const float3 corner : face.Corner) scale = std::max(scale, simd::length(corner) + face.Radius);

    const float tolerance = 1e-5f * scale;
    std::vector<uint32_t> masks(pieces.size(), 0);
    for (size_t i = 0; i < pieces.size(); ++i)
        for (size_t j = 0; j < pieces.size(); ++j) {
            if (j == i || (!filters.empty() && !SameMask(filters[i], filters[j]))) continue;
            for (const ShapeFace &mine : pieces[i])
                for (const ShapeFace &theirs : pieces[j]) {
                    if (dot(mine.Normal, theirs.Normal) > -0.99999f) continue;
                    if (std::abs(mine.Offset + theirs.Offset) > tolerance) continue;
                    if (Within(mine, theirs, tolerance)) masks[i] |= 1u << mine.Index;
                }
        }
    return masks;
}
} // namespace

BodyMass MassProperties(const Shape &shape, float density, std::span<const float3> shape_vertices, std::span<const Shape> shapes, std::span<const Index> children) {
    constexpr float Pi = std::numbers::pi_v<float>;

    if (shape.Kind == ShapePlane || shape.Kind == ShapeMesh || density <= 0) return StaticMass;

    if (shape.Kind == ShapeCompound) {
        const Aggregate whole = WeighChildren(shape, shape_vertices, shapes, children);
        if (whole.Volume <= 0) return StaticMass;
        if (shape.VertexCount == 1) {
            const Shape &child = shapes[ChildOf(shape, 0, children.data())];
            if (simd::all(child.Local.Position == IdentityPose.Position) && simd::all(child.Local.Orientation == IdentityPose.Orientation))
                return MassProperties(child, density, shape_vertices, shapes, children);
        }
        const double3 moments = DiagonalizeSymmetric(whole.Tensor).Values;
        return {.InvInertiaLocal = 1 / (float3{float(moments.x), float(moments.y), float(moments.z)} * density), .InvMass = 1 / float(whole.Volume * density)};
    }

    if (shape.Kind == ShapeHull) {
        const CookedHull cooked = CookHull(shape_vertices.subspan(shape.FirstVertex, shape.VertexCount));
        if (cooked.Vertices.empty()) return StaticMass;
        return {.InvInertiaLocal = 1 / (cooked.Inertia * density), .InvMass = 1 / (cooked.Volume * density)};
    }

    if (shape.Kind == ShapeCylinder) {
        const float radius = shape.HalfExtents.x, half = shape.HalfExtents.y;
        const float mass = density * Pi * radius * radius * 2 * half;
        const float along = mass * radius * radius / 2;
        const float across = mass * (radius * radius / 4 + half * half / 3);
        return {.InvInertiaLocal = 1 / float3{across, along, across}, .InvMass = 1 / mass};
    }

    if (shape.Kind == ShapeCapsule) {
        const float radius = shape.Radius, half = shape.HalfExtents.y;
        const float cylinder = density * Pi * radius * radius * 2 * half;
        const float caps = density * 4.f / 3 * Pi * radius * radius * radius;
        const float along = cylinder * radius * radius / 2 + caps * 2.f / 5 * radius * radius;
        const float across = cylinder * (radius * radius / 4 + half * half / 3) +
            caps * (2.f / 5 * radius * radius + half * half + 3.f / 4 * half * radius);
        return {.InvInertiaLocal = 1 / float3{across, along, across}, .InvMass = 1 / (cylinder + caps)};
    }

    if (shape.Kind == ShapeSphere) {
        const float radius = shape.Radius;
        const float mass = density * 4.f / 3 * Pi * radius * radius * radius;
        const float inertia = 2.f / 5 * mass * radius * radius;
        return {.InvInertiaLocal = 1 / float3{inertia, inertia, inertia}, .InvMass = 1 / mass};
    }

    const float3 extents = 2 * shape.HalfExtents;
    const float mass = density * extents.x * extents.y * extents.z;

    const float3 squared = extents * extents;
    const float3 inertia = mass / 12 * float3{squared.y + squared.z, squared.x + squared.z, squared.x + squared.y};
    return {.InvInertiaLocal = 1 / inertia, .InvMass = 1 / mass};
}

BodyMass World::ShapeOrAuthoredMass(Index shape, float density, std::optional<AuthoredMass> authored) const {
    if (authored) {
        const auto [mass, inertia] = *authored;
        return {.InvInertiaLocal = {inertia.x > 0 ? 1 / inertia.x : 0, inertia.y > 0 ? 1 / inertia.y : 0, inertia.z > 0 ? 1 / inertia.z : 0}, .InvMass = mass > 0 ? 1 / mass : 0};
    }
    return shape == NoIndex ? StaticMass : MassProperties(Shapes[shape], density, ShapeVertices.All(), Shapes.All(), CompoundChildren.All());
}

// Offset collider frames require authored body-frame mass properties.
bool World::OffsetNeedsAuthoredMass(Index shape, const BodyMass &mass, bool authored) const {
    if (authored || shape == NoIndex || !Moves(mass)) return false;
    const auto [at, turn] = Shapes[shape].Local;
    return at.x != 0 || at.y != 0 || at.z != 0 || turn.x != 0 || turn.y != 0 || turn.z != 0;
}

template<typename T> void World::MakeBuffer(mtl::Buffer<T> &buffer, uint32_t capacity) {
    buffer = {Queue->device(), capacity};
    std::ranges::fill(buffer.All(), T{});
    Residency->addAllocation(buffer.Handle.get());
}

World::World(const mtl::Context &context, WorldLimits limits) : Queue(context.Queue) {
    auto *device = context.Device.get();
    // Metal 4 requires explicit residency for every resource a shader can reach.
    NS::Error *error{};
    Residency = NS::TransferPtr(device->newResidencySet(mtl::Make<MTL::ResidencySetDescriptor>().get(), &error));

    MakeBuffer(Poses, limits.Bodies);
    MakeBuffer(Velocities, limits.Bodies);
    MakeBuffer(Masses, limits.Bodies);
    MakeBuffer(BodyShapes, limits.Bodies);
    MakeBuffer(Shapes, limits.Shapes);
    MakeBuffer(ShapeVertices, limits.ShapeVertices);
    MakeBuffer(HullFaces, limits.HullFaces);
    MakeBuffer(Triangles, limits.Triangles);
    MakeBuffer(BvhNodes, limits.BvhNodes);
    MakeBuffer(Materials, limits.Bodies);
    MakeBuffer(CompoundChildren, limits.CompoundChildren);
    MakeBuffer(Filters, limits.Bodies);
    MakeBuffer(Jointed, limits.Bodies + 1 + 2 * limits.Joints);
    MakeBuffer(JointIncidence, limits.Bodies + 1 + 2 * limits.Joints);
    MakeBuffer(Bounds, limits.Bodies);
    MakeBuffer(BoundsReductions, RadixBlocks(limits.Bodies) + 1);
    MakeBuffer(BroadPhaseNodes, 2 * limits.Bodies);
    MakeBuffer(BroadPhaseKeys, 2 * limits.Bodies);
    MakeBuffer(BroadPhaseScratch, limits.Bodies + RadixBlocks(limits.Bodies) * RadixBins);
    MakeBuffer(InitialPoses, limits.Bodies);
    MakeBuffer(Displacements, limits.Bodies);
    MakeBuffer(InertialPoses, limits.Bodies);
    MakeBuffer(PreviousVelocities, limits.Bodies);
    MakeBuffer(Iterates, limits.Bodies);
    MakeBuffer(RestPoses, limits.Bodies);
    MakeBuffer(Quiet, limits.Bodies);
    MakeBuffer(NextQuiet, limits.Bodies);
    MakeBuffer(Colors, limits.Bodies);
    MakeBuffer(NextColors, limits.Bodies);
    MakeBuffer(Contacts, limits.Bodies * ContactsPerBody);
    MakeBuffer(Incoming, limits.Bodies);
    MakeBuffer(IncomingSlots, limits.Bodies * ContactsPerBody);
    MakeBuffer(ContactEvents, limits.Bodies * EventsPerBody);
    MakeBuffer(ContactEventCounts, limits.Bodies);
    MakeBuffer(ContactRefusals, limits.Bodies);
    MakeBuffer(Joints, limits.Joints);
    Residency->commit();
    Residency->requestResidency();
    Queue->addResidencySet(Residency.get());

    std::ranges::fill(Jointed.All().first(limits.Bodies + 1), limits.Bodies + 1);
    std::ranges::fill(JointIncidence.All().first(limits.Bodies + 1), limits.Bodies + 1);

    for (auto *buffer : {&BodyShapes, &IncomingSlots}) std::ranges::fill(buffer->All(), NoIndex);

    VertexPool.Capacity = limits.ShapeVertices;
    FacePool.Capacity = limits.HullFaces;
    TrianglePool.Capacity = limits.Triangles;
    NodePool.Capacity = limits.BvhNodes;
    ChildPool.Capacity = limits.CompoundChildren;
    LiveBodies.assign(limits.Bodies, 0);
    LiveShapes.assign(limits.Shapes, 0);
    WeldedShapes.assign(limits.Bodies, NoIndex);
    Spawns.assign(limits.Bodies, 0);
}

World::~World() {
    if (Queue && Residency) {
        Queue->removeResidencySet(Residency.get());
        mtl::Drain(Queue.get());
    }
}

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
    if (count == 0 || start == NoIndex) return;

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

// Host removals can invalidate incoming adjacency, so scan contact storage directly.
void World::EndContacts(Index body) {
    EndSensorOverlaps(body);
    const auto end = [this](Contact &contact) {
        if (TrackContacts)
            Changes.push_back({
                .A = IdOf(contact.BodyA),
                .B = IdOf(contact.BodyB),
                .Feature = contact.Feature,
                .SubShape = contact.SubShape,
                .SubShapeA = contact.SubShapeA,
                .Children = contact.Children,
                .Kind = ContactRemoved,
                .Step = CompletedSteps,
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

void World::RefreshFilters() {
    for (Index body = 0; body < NumBodies; ++body) {
        if (!Alive(body)) continue;
        Filter &filter = Filters[body];
        filter.Aggregate = {filter.Layer, filter.Collides};
        filter.Mixed = 0;
        const Index index = BodyShapes[body];
        if (index == NoIndex) continue;
        const Shape &shape = Shapes[index];
        const CollisionMask inherited = ResolveFilter(shape, filter.Aggregate);
        filter.Aggregate = inherited;
        if (shape.Kind != ShapeCompound) continue;
        filter.Aggregate = {0, 0};
        CollisionMask first{};
        for (uint32_t i = 0; i < shape.VertexCount; ++i) {
            const CollisionMask leaf = ResolveFilter(Shapes[Child(index, i)], inherited);
            if (i == 0) first = leaf;
            filter.Mixed |= !SameMask(first, leaf);
            filter.Aggregate.Layer |= leaf.Layer;
            filter.Aggregate.Collides |= leaf.Collides;
        }
    }
}

void World::EnsureSensorBuffers() {
    if (SensorContacts.Handle) return;
    MakeBuffer(SensorContacts, Poses.Capacity * ContactsPerBody);
    MakeBuffer(SensorRefusals, Poses.Capacity);
    Residency->commit();
}

void World::EndSensorOverlaps(Index body) {
    if (!SensorContacts.Handle) return;
    std::erase_if(SensorOverlaps, [&](const SensorOverlap &pair) {
        if (pair.A.Slot != body && pair.B.Slot != body) return false;
        if (TrackSensors) SensorChanges.push_back({pair, false});
        return true;
    });
    for (Index owner = 0; owner < NumBodies; ++owner) {
        const auto run = SensorContacts.All().subspan(owner * ContactsPerBody, ContactsPerBody);
        const auto removed = std::ranges::remove_if(run, [body](const Contact &contact) {
            return !contact.Active || contact.BodyA == body || contact.BodyB == body;
        });
        for (Contact &contact : removed) contact.Active = 0;
    }
}

void World::UpdateSensorOverlaps(const StepSnapshot &snapshot) {
    if (SensorOverlaps.empty() && (snapshot.Counts.empty() ? !SensorContacts.Handle : snapshot.Sensors.empty())) return;
    std::vector<SensorOverlap> current;
    current.reserve(SensorOverlaps.size());
    for (Index body = 0; body < NumBodies; ++body) {
        if (!snapshot.Counts.empty()) {
            for (uint32_t i = 0; i < snapshot.Counts[body].Sensors; ++i) {
                const auto pair = snapshot.Sensors[body * ContactsPerBody + i];
                current.push_back({IdOf(pair.BodyA), IdOf(pair.BodyB), pair.Children});
            }
        } else if (SensorContacts.Handle) {
            for (uint32_t i = 0; i < ContactsPerBody; ++i) {
                const Contact &contact = SensorContacts[body * ContactsPerBody + i];
                if (!contact.Active) break;
                current.push_back({IdOf(contact.BodyA), IdOf(contact.BodyB), contact.Children});
            }
        }
    }
    if (TrackSensors) {
        for (const auto &pair : SensorOverlaps)
            if (std::ranges::find(current, pair) == current.end()) SensorChanges.push_back({pair, false});
        for (const auto &pair : current)
            if (std::ranges::find(SensorOverlaps, pair) == SensorOverlaps.end()) SensorChanges.push_back({pair, true});
    }
    SensorOverlaps = std::move(current);
}

// Live event positions correspond to the finalized contact run.
void World::DrainContactEvents(float delta_time, const StepSnapshot &snapshot) {
    if (!TrackContacts) return;
    const auto initial = snapshot.InitialPoses.empty() ? InitialPoses.All() : snapshot.InitialPoses;
    const auto poses = snapshot.Poses.empty() ? Poses.All() : snapshot.Poses;
    const auto velocities = snapshot.Velocities.empty() ? Velocities.All() : snapshot.Velocities;
    const auto side = [&](Index body, uint32_t child, float3 point, float3 anchor) -> ContactSide {
        const Index root = BodyShapes[body];
        const auto user_data = Shapes[Shapes[root].Kind == ShapeCompound ? Child(root, child) : root].UserData;
        return {initial[body], poses[body], velocities[body], point, anchor, user_data, Masses[body].InvMass};
    };
    for (Index body = 0; body < NumBodies; ++body) {
        uint32_t live = 0;
        const uint32_t count = snapshot.Counts.empty() ? ContactEventCounts[body] : snapshot.Counts[body].Contacts + snapshot.Counts[body].RemovedContacts;
        for (uint32_t i = 0; i < count; ++i) {
            ContactReport report;
            if (snapshot.Counts.empty()) {
                const ContactEvent event = ContactEvents[body * EventsPerBody + i];
                report = ReportContact(event, event.Kind == ContactRemoved ? Contact{} : Contacts[body * ContactsPerBody + live++]);
            } else if (i < snapshot.Counts[body].Contacts) report = snapshot.Contacts[body * ContactsPerBody + i];
            else report = {.Event = snapshot.RemovedContacts[body * ContactsPerBody + i - snapshot.Counts[body].Contacts]};
            const ContactEvent &event = report.Event;
            ContactChange change{
                .A = IdOf(event.BodyA),
                .B = IdOf(event.BodyB),
                .Feature = event.Feature,
                .SubShape = event.SubShape,
                .SubShapeA = event.SubShapeA,
                .Children = event.Children,
                .Kind = ContactEventKind(event.Kind),
                .Step = CompletedSteps,
                .DeltaTime = delta_time,
            };
            if (change.Kind != ContactRemoved) {
                const ContactReport &contact = report;
                change.Lambda = contact.Lambda;
                change.Approach = contact.Approach;
                change.BounceImpulse = contact.BounceImpulse;
                change.SideA = side(event.BodyA, OwnChild(event.Children), contact.PointA, contact.AnchorA);
                change.SideB = side(event.BodyB, OtherChild(event.Children), contact.PointB, contact.AnchorB);
                change.Normal = contact.Normal;
                change.Friction = contact.Friction;
                change.Restitution = contact.Restitution;
                change.NominalArea = contact.NominalArea;
                change.NominalExtent = contact.NominalExtent;
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
    if (count == 0) return NoIndex;
    if (frame != nullptr) *frame = cooked.Frame;
    const Pose shape_local = local ? ComposePose(*local, cooked.Frame) : IdentityPose;

    const Index first = VertexPool.Take(count);
    const Index first_face = FacePool.Take(face_count);
    uint32_t *refused = nullptr;
    if (first == NoIndex) refused = &Overflow.ShapeVertices;
    else if (first_face == NoIndex) refused = &Overflow.HullFaces;
    const Index shape = refused != nullptr ? NoIndex : AddShape({.FirstVertex = first, .VertexCount = count, .FirstFace = first_face, .FaceCount = face_count, .Kind = ShapeHull, .Local = shape_local});
    if (shape == NoIndex) {
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
    if (cooked.Triangles.empty()) return NoIndex;
    const uint32_t vertices = cooked.Vertices.size(), triangles = cooked.Triangles.size(), nodes = cooked.Nodes.size();

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

    for (uint32_t i = 0; i < triangles; ++i) {
        Triangle triangle = cooked.Triangles[i];
        triangle.A += first_vertex;
        triangle.B += first_vertex;
        triangle.C += first_vertex;
        Triangles[first_triangle + i] = triangle;
    }

    std::ranges::copy(cooked.Nodes, BvhNodes.All().begin() + root);
    return shape;
}

Index World::CopyShape(Shape copy) {
    if (copy.Kind == ShapeMesh) {
        std::vector<uint32_t> indices;
        for (const Triangle &triangle : Triangles.All().subspan(copy.FirstTriangle, copy.TriangleCount)) {
            indices.insert(indices.end(), {triangle.A - copy.FirstVertex, triangle.B - copy.FirstVertex, triangle.C - copy.FirstVertex});
        }
        const Index result = AddMesh(ShapeVertices.All().subspan(copy.FirstVertex, copy.VertexCount), indices, copy.Local);
        if (result != NoIndex) {
            Shapes[result].UserData = copy.UserData;
            Shapes[result].Surface = copy.Surface;
            Shapes[result].HasMaterial = copy.HasMaterial;
            Shapes[result].Mask = copy.Mask;
            Shapes[result].HasFilter = copy.HasFilter;
            Shapes[result].DoubleSided = copy.DoubleSided;
        }
        return result;
    }
    // Copies own their geometry runs so either shape can be released independently.
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

    if (held.Kind == ShapeCompound) {
        for (Index child : CompoundChildren.All().subspan(held.FirstVertex, held.VertexCount)) ReleaseShape(child);
        ChildPool.Give(held.FirstVertex, held.VertexCount);
    }
    LiveShapes[shape] = 0;
    FreeShapes.push_back(shape);
}

Index World::AddCompound(std::span<const Index> children, Pose *frame) {
    std::vector<Shape> leaves;
    for (Index child : children) {
        if (child >= NumShapes || !LiveShapes[child]) {
            ++RefusedCompounds;
            return NoIndex;
        }
        const Shape &shape = Shapes[child];
        if (shape.Kind == ShapeCompound) {
            for (uint32_t i = 0; i < shape.VertexCount; ++i) {
                Shape leaf = Shapes[Child(child, i)];
                leaf.Local = ComposePose(shape.Local, leaf.Local);
                if (!leaf.HasMaterial && shape.HasMaterial) leaf.Surface = shape.Surface;
                if (!leaf.HasFilter && shape.HasFilter) leaf.Mask = shape.Mask;
                leaf.HasMaterial |= shape.HasMaterial;
                leaf.HasFilter |= shape.HasFilter;
                leaves.push_back(leaf);
            }
        } else leaves.push_back(shape);
    }
    if (leaves.empty()) {
        ++RefusedCompounds;
        return NoIndex;
    }
    if (leaves.size() > CompoundChildren.Capacity) {
        ++Overflow.CompoundChildren;
        return NoIndex;
    }
    const uint32_t count = uint32_t(leaves.size());
    const Index first = ChildPool.Take(count);
    if (first == NoIndex) {
        ++Overflow.CompoundChildren;
        return NoIndex;
    }
    Shape compound{.FirstVertex = first, .VertexCount = count, .Kind = ShapeCompound};
    uint32_t made = 0;
    for (; made < count; ++made) {
        const Index copy = CopyShape(leaves[made]);
        if (copy == NoIndex) break;
        CompoundChildren[first + made] = copy;
    }
    const auto discard = [&] {
        for (uint32_t i = 0; i < made; ++i) ReleaseShape(CompoundChildren[first + i]);
        ChildPool.Give(first, count);
        TrimTail(NumShapes, FreeShapes, [this](Index at) { return LiveShapes[at] != 0; });
    };
    if (made != count) {
        discard();
        return NoIndex;
    }
    const Aggregate whole = WeighChildren(compound, ShapeVertices.All(), Shapes.All(), CompoundChildren.All());
    const bool single = count == 1 && whole.Volume > 0;
    const Pose body = single ? leaves[0].Local : (whole.Volume > 0 ? Pose{.Position = float3{float(whole.Center.x), float(whole.Center.y), float(whole.Center.z)}, .Orientation = DiagonalizeSymmetric(whole.Tensor).Orientation} : IdentityPose);
    const Pose inverse{.Position = Rotate(QuatConjugate(body.Orientation), -body.Position), .Orientation = QuatConjugate(body.Orientation)};
    std::vector<std::vector<ShapeFace>> faces;
    for (uint32_t i = 0; i < count; ++i) {
        Shape &child = Shapes[CompoundChildren[first + i]];
        child.Local = single ? IdentityPose : ComposePose(inverse, child.Local);
        if (child.Kind == ShapeBox || child.Kind == ShapeHull || child.Kind == ShapeCylinder) SetInternalFaces(child, 0);
        faces.push_back(ShapeFaces(child, child.Local, ShapeVertices.All(), HullFaces.All()));
    }
    const auto masks = BuriedFaces(faces);
    for (uint32_t i = 0; i < count; ++i) {
        Shape &child = Shapes[CompoundChildren[first + i]];
        if (child.Kind == ShapeBox || child.Kind == ShapeHull || child.Kind == ShapeCylinder) SetInternalFaces(child, masks[i]);
    }
    const Index result = AddShape(compound);
    if (result == NoIndex) {
        discard();
        return NoIndex;
    }
    if (frame) *frame = body;
    return result;
}

uint32_t World::WeldStatic() {
    std::vector<Index> resting;
    std::vector<std::vector<ShapeFace>> faces;
    std::vector<CollisionMask> filters;
    for (Index body = 0; body < NumBodies; ++body) {
        if (!LiveBodies[body] || Moves(Masses[body]) || Filters[body].Sensor) continue;
        const Velocity motion = Velocities[body];
        if (simd::length(motion.Linear) > 0 || simd::length(motion.Angular) > 0) continue;
        const Index shape = BodyShapes[body];
        if (shape == NoIndex) continue;
        const Shape &held = Shapes[shape];
        if (held.Kind != ShapeBox && held.Kind != ShapeHull && held.Kind != ShapeCylinder) continue;
        resting.push_back(body);
        filters.push_back(ResolveFilter(held, {Filters[body].Layer, Filters[body].Collides}));

        faces.push_back(ShapeFaces(held, ComposePose(Poses[body], held.Local), ShapeVertices.All(), HullFaces.All()));
    }
    std::vector<uint32_t> wanted(NumBodies, 0);
    const std::vector<uint32_t> masks = BuriedFaces(faces, filters);
    for (uint32_t i = 0; i < resting.size(); ++i) wanted[resting[i]] = masks[i];

    // Revisit all live bodies so previous welds can be undone after edits.
    uint32_t buried = 0;
    for (Index body = 0; body < NumBodies; ++body) {
        if (!LiveBodies[body]) continue;
        const uint32_t mask = wanted[body];
        Index copy = WeldedShapes[body];
        if (copy == NoIndex) {
            if (mask == 0) continue;

            copy = CopyShape(Shapes[BodyShapes[body]]);
            if (copy == NoIndex) continue;
            WeldedShapes[body] = copy;
            BodyShapes[body] = copy;
        }
        if (InternalFaces(Shapes[copy]) != mask) {
            SetInternalFaces(Shapes[copy], mask);

            Wake(body);
        }
        buried += uint32_t(std::popcount(mask));
    }
    return buried;
}

void World::DropWeld(Index body) {
    const Index copy = WeldedShapes[body];
    if (copy == NoIndex) return;
    WeldedShapes[body] = NoIndex;
    BodyShapes[body] = NoIndex;
    RemoveShape(copy);
}

Index World::AddBody(const BodyDesc &desc) {
    BodyMass mass = ShapeOrAuthoredMass(desc.Shape, desc.Density, desc.Mass);
    if (OffsetNeedsAuthoredMass(desc.Shape, mass, desc.Mass.has_value())) {
        ++OffsetsWithoutMass;
        return NoIndex;
    }
    const Index index = TakeSlot(FreeBodies, NumBodies, Poses.Capacity, Overflow.Bodies);
    if (index == NoIndex) return NoIndex;
    LiveBodies[index] = 1;

    ++Spawns[index];
    // Reset reused slots to fresh-body state so retirement cannot leak solver history.
    Colors[index] = 0;
    Incoming[index] = {};
    Poses[index] = desc.Pose;
    Velocities[index] = desc.Velocity;
    PreviousVelocities[index] = desc.Velocity;
    BodyShapes[index] = desc.Shape;
    Quiet[index] = 0;
    RestPoses[index] = desc.Pose;
    Materials[index] = desc.Surface.value_or(Material{desc.Friction, desc.Friction, desc.Restitution, CombineGeometricMean, CombineMaximum});
    Filters[index] = {.Layer = desc.Layer, .Collides = desc.CollidesWith, .Sensor = desc.Sensor};
    if (index + 1 == NumBodies) {
        Jointed[index + 1] = Jointed[index];
        JointIncidence[index + 1] = JointIncidence[index];
    }

    mass.GravityScale = desc.GravityScale;
    mass.LinearDamping = desc.LinearDamping;
    mass.AngularDamping = desc.AngularDamping;
    Masses[index] = mass;
    return index;
}

namespace {

uint32_t Modes(const JointAxisMode (&axes)[3]) {
    return uint32_t(axes[0]) | (uint32_t(axes[1]) << 3) | (uint32_t(axes[2]) << 6);
}
} // namespace

Index World::AddJoint(const JointDesc &desc) {
    if (!Alive(desc.BodyA) || !Alive(desc.BodyB)) return NoIndex;
    for (uint32_t row = 0; row < 6; ++row) {
        const uint32_t axis = row % 3;
        const auto *modes = row < 3 ? desc.Linear : desc.Angular;
        const uint32_t mask = row < 3 ? desc.LinearLimitAxes[axis] : desc.AngularLimitAxes[axis];
        if (!mask) continue;
        if (mask > 7 || std::countr_zero(mask) != axis || modes[axis] != AxisLimited) return NoIndex;
        for (uint32_t other = axis + 1; other < 3; ++other)
            if ((mask & (1u << other)) && modes[other] != AxisFree) return NoIndex;
    }
    const Index index = TakeSlot(FreeJoints, NumJoints, Joints.Capacity, Overflow.Joints);
    if (index == NoIndex) return NoIndex;
    const Pose a = Poses[desc.BodyA], b = Poses[desc.BodyB];
    const float3 at_a = desc.AtA.value_or(desc.At), at_b = desc.AtB.value_or(desc.At);
    const float4 frame = desc.Frame.value_or(b.Orientation);
    Joints[index] = {
        .AnchorA = LocalPoint(a, at_a),
        .AnchorB = LocalPoint(b, at_b),
        .FrameA = QuatMul(QuatConjugate(a.Orientation), desc.FrameA.value_or(frame)),
        .FrameB = QuatMul(QuatConjugate(b.Orientation), desc.FrameB.value_or(frame)),
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
    for (uint32_t i = 0; i < 6; ++i) {
        Joints[index].Drives[i] = desc.Drives[i];
        Joints[index].Drives[i].Lambda = 0;
        Joints[index].Drives[i].Penalty = 1;
    }
    for (uint32_t i = 0; i < 3; ++i) {
        Joints[index].LinearLimitAxes[i] = desc.LinearLimitAxes[i];
        Joints[index].AngularLimitAxes[i] = desc.AngularLimitAxes[i];
    }
    RebuildJointed();
    return index;
}

void World::RebuildJointed() {
    TrimTail(NumJoints, FreeJoints, [this](Index at) { return Joints[at].Active != 0; });
    std::vector<Index> cursor(NumBodies, 0);
    for (const bool suppressing : {false, true}) {
        auto &links = suppressing ? Jointed : JointIncidence;
        std::ranges::fill(cursor, 0);
        for (Index i = 0; i < NumJoints; ++i) {
            const Joint &joint = Joints[i];
            if (!joint.Active || (suppressing && !joint.Suppresses)) continue;
            ++cursor[joint.BodyA];
            ++cursor[joint.BodyB];
        }
        Index end = Poses.Capacity + 1;
        for (Index body = 0; body < NumBodies; ++body) {
            links[body] = end;
            end += cursor[body];
            cursor[body] = links[body];
        }
        links[NumBodies] = end;
        for (Index i = 0; i < NumJoints; ++i) {
            const Joint &joint = Joints[i];
            if (!joint.Active || (suppressing && !joint.Suppresses)) continue;
            links[cursor[joint.BodyA]++] = suppressing ? joint.BodyB : i;
            links[cursor[joint.BodyB]++] = suppressing ? joint.BodyA : i;
        }
    }
}

bool World::RemoveBody(Index body) {
    if (!Alive(body)) return false;
    Wake(body);

    bool removed_joints = false;
    for (Index joint = 0; joint < NumJoints; ++joint) {
        const Joint &held = Joints[joint];
        if (held.Active && (held.BodyA == body || held.BodyB == body)) removed_joints |= RetireJoint(joint);
    }
    if (removed_joints) RebuildJointed();
    // Wake before clearing the contact runs it traverses.
    EndContacts(body);
    DropWeld(body);
    BodyShapes[body] = NoIndex;
    Masses[body] = StaticMass;
    Velocities[body] = {};
    LiveBodies[body] = 0;
    RetiredBodies.push_back(body);
    return true;
}

bool World::RetireJoint(Index joint) {
    if (joint >= NumJoints || !Joints[joint].Active) return false;
    const Joint &held = Joints[joint];

    Quiet[held.BodyA] = 0;
    Quiet[held.BodyB] = 0;
    Joints[joint].Active = 0;
    FreeJoints.push_back(joint);
    return true;
}

bool World::RemoveJoint(Index joint) {
    if (!RetireJoint(joint)) return false;
    RebuildJointed();
    return true;
}

bool World::RemoveShape(Index shape) {
    if (shape >= NumShapes || !LiveShapes[shape]) return false;
    for (Index body = 0; body < NumBodies; ++body)
        if (LiveBodies[body] && BodyShapes[body] == shape) return false;

    for (Index other = 0; other < NumShapes; ++other) {
        if (!LiveShapes[other] || Shapes[other].Kind != ShapeCompound) continue;
        const Shape parent = Shapes[other];
        for (uint32_t i = 0; i < parent.VertexCount; ++i) {
            const Index child = ChildOf(parent, i, CompoundChildren.All().data());
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

    BodyMass mass = ShapeOrAuthoredMass(shape, density, authored);
    if (OffsetNeedsAuthoredMass(shape, mass, authored.has_value())) {
        ++OffsetsWithoutMass;
        return false;
    }
    Wake(body);
    EndContacts(body);

    if (shape != WeldedShapes[body]) DropWeld(body);
    BodyShapes[body] = shape;
    const BodyMass held = Masses[body];
    mass.GravityScale = held.GravityScale;
    mass.LinearDamping = held.LinearDamping;
    mass.AngularDamping = held.AngularDamping;
    Masses[body] = mass;
    return true;
}

void World::OnStepped(float delta_time, const StepSnapshot &snapshot) {
    ++CompletedSteps;
    UpdateSensorOverlaps(snapshot);

    DrainContactEvents(delta_time, snapshot);
    // Delay body-slot reuse until completed reporting no longer refers to its previous occupant.
    for (const Index body : RetiredBodies) FreeBodies.push_back(body);
    RetiredBodies.clear();

    TrimTail(NumBodies, FreeBodies, [this](Index at) { return LiveBodies[at] != 0; });
}

} // namespace rbp
