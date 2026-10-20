
#include "World.h"
#include "Shapes.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <numbers>
#include <utility>
#include <vector>

#include <doctest/doctest.h>

using namespace rbp;

namespace {

struct Device {
    const mtl::Context context;
};
struct OneWorld : Device {
    World world{context};
};

Index AddQuadMesh(World &world, Pose local = IdentityPose) {
    const std::vector<float3> points{float3{-1, 0, -1}, float3{1, 0, -1}, float3{1, 0, 1}, float3{-1, 0, 1}};
    const std::vector<uint32_t> indices{0, 1, 2, 0, 2, 3};
    return world.AddMesh(points, indices, local);
}
} // namespace

TEST_CASE_FIXTURE(OneWorld, "world: fixed topology mesh updates preserve shared identity and reject atomically") {
    const Index mesh = AddQuadMesh(world);
    REQUIRE(mesh != NoIndex);
    const Shape shape = world.Shapes[mesh];
    const Index first = world.AddBody({.Shape = mesh, .Density = 0, .GeometryActive = true});
    const Index second = world.AddBody({.Pose = At(float3{3, 0, 0}), .Shape = mesh, .Density = 0});
    const auto first_id = world.IdOf(first), second_id = world.IdOf(second);
    const auto first_pose = world.Poses[first], second_pose = world.Poses[second];
    const auto first_mass = world.Masses[first], second_mass = world.Masses[second];
    const auto vertices = world.ShapeVertices.All().subspan(shape.FirstVertex, shape.VertexCount);
    const auto triangles = world.Triangles.All().subspan(shape.FirstTriangle, shape.TriangleCount);
    const auto nodes = world.BvhNodes.All().subspan(shape.RootNode, shape.NodeCount);
    const std::vector<Triangle> topology(triangles.begin(), triangles.end());
    const std::vector<BvhNode> tree(nodes.begin(), nodes.end());
    std::vector<float3> deformed(vertices.begin(), vertices.end());
    for (auto &point : deformed) point.y = 0.2f * point.x;
    world.Quiet[first] = world.Quiet[second] = 100;
    REQUIRE(world.SetMeshVertices(mesh, deformed));
    CHECK(world.Quiet[first] == 0);
    CHECK(world.Quiet[second] == 0);
    CHECK(world.IdOf(first) == first_id);
    CHECK(world.IdOf(second) == second_id);
    CHECK(std::memcmp(&shape, &world.Shapes[mesh], sizeof(Shape)) == 0);
    CHECK(std::memcmp(&first_pose, &world.Poses[first], sizeof(Pose)) == 0);
    CHECK(std::memcmp(&second_pose, &world.Poses[second], sizeof(Pose)) == 0);
    CHECK(std::memcmp(&first_mass, &world.Masses[first], sizeof(BodyMass)) == 0);
    CHECK(std::memcmp(&second_mass, &world.Masses[second], sizeof(BodyMass)) == 0);
    for (std::size_t i = 0; i < vertices.size(); ++i) CHECK(simd::all(vertices[i] == deformed[i]));
    for (std::size_t i = 0; i < triangles.size(); ++i) {
        CHECK(triangles[i].A == topology[i].A);CHECK(triangles[i].B == topology[i].B);
        CHECK(triangles[i].C == topology[i].C);CHECK(triangles[i].OwnedEdges == topology[i].OwnedEdges);
    }
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        CHECK(nodes[i].First == tree[i].First);CHECK(nodes[i].Count == tree[i].Count);
    }
    CHECK(nodes[0].Low.y == doctest::Approx(-.2f));
    CHECK(nodes[0].High.y == doctest::Approx(.2f));
    const std::vector<Triangle> accepted_triangles(triangles.begin(), triangles.end());
    const std::vector<BvhNode> accepted_nodes(nodes.begin(), nodes.end());
    for (int invalid = 0; invalid < 4; ++invalid) {
        auto bad = deformed;
        if (invalid == 0) bad.pop_back();
        if (invalid == 1) bad.back().x = std::numeric_limits<float>::quiet_NaN();
        if (invalid == 2) bad[topology[0].C - shape.FirstVertex] = bad[topology[0].A - shape.FirstVertex];
        CHECK_FALSE(world.SetMeshVertices(invalid == 3 ? NoIndex : mesh, bad));
        CHECK(std::memcmp(vertices.data(), deformed.data(), vertices.size_bytes()) == 0);
        CHECK(std::memcmp(triangles.data(), accepted_triangles.data(), triangles.size_bytes()) == 0);
        CHECK(std::memcmp(nodes.data(), accepted_nodes.data(), nodes.size_bytes()) == 0);
    }
    CHECK(world.SetGeometryActive(first, false));
    CHECK_FALSE(world.Filters[first].GeometryActive);
    CHECK_FALSE(world.SetGeometryActive(NoIndex, true));
    CHECK(world.RemoveBody(first));
    CHECK_FALSE(world.SetGeometryActive(first, true));
}

namespace {
constexpr float TileHalf = 2.5f, TileTop = 0.25f;

Index AddTile(World &world, Index shape, float side) {
    return world.AddBody({.Pose = At(float3{side * TileHalf, 0, 0}), .Shape = shape, .Density = 0});
}
} // namespace

TEST_CASE_FIXTURE(OneWorld, "world: retirement preserves joint topology and reusable slots") {
    const auto hub = world.AddBody({});
    const auto a = world.AddBody({}), b = world.AddBody({});
    const auto first = world.AddJoint({.BodyA = hub, .BodyB = a});
    const auto survivor = world.AddJoint({.BodyA = a, .BodyB = b});
    const auto gap = world.AddJoint({.BodyA = hub, .BodyB = b, .Collide = true});
    const auto last_survivor = world.AddJoint({.BodyA = a, .BodyB = b, .Collide = true});
    for (uint32_t i = 0; i < 24; ++i)
        REQUIRE(world.AddJoint({.BodyA = hub, .BodyB = i % 2 ? a : b, .Collide = bool(i % 3)}) != NoIndex);
    REQUIRE(world.RemoveBody(hub));
    REQUIRE(world.JointCount() == last_survivor + 1);
    CHECK(world.Joints[survivor].Active);
    CHECK(world.Joints[last_survivor].Active);
    CHECK(world.AddJoint({.BodyA = a, .BodyB = b}) == gap);
    CHECK(world.AddJoint({.BodyA = a, .BodyB = b, .Collide = true}) == first);
    CHECK(world.AddJoint({.BodyA = a, .BodyB = b}) == last_survivor + 1);
    REQUIRE(world.SetJoint(survivor, {.BodyA = b, .BodyB = a, .Collide = true, .Drives = {{.Enabled = 1, .Target = 2, .Lambda = 9, .Penalty = 7, .Began = 5}}}));
    CHECK(world.JointCount() == last_survivor + 2);
    const auto &drive = world.Joints[survivor].Drives[0];
    CHECK(drive.Target == 2);
    CHECK(drive.Lambda == 0);
    CHECK(drive.Penalty == 1);
    CHECK(drive.Began == 0);

    for (Index body = 0; body < world.BodyCount(); ++body) {
        std::vector<Index> incident, suppressed;
        for (Index index = 0; index < world.JointCount(); ++index) {
            const Joint &joint = world.Joints[index];
            if (!joint.Active) continue;
            for (const auto [owner, partner] : {std::pair{joint.BodyA, joint.BodyB}, std::pair{joint.BodyB, joint.BodyA}})
                if (owner == body) {
                    incident.push_back(index);
                    if (joint.Suppresses) suppressed.push_back(partner);
                }
        }
        const auto run = [body](const auto &links) { return links.All().subspan(links[body], links[body + 1] - links[body]); };
        CHECK(std::ranges::equal(run(world.JointIncidence), incident));
        CHECK(std::ranges::equal(run(world.Jointed), suppressed));
    }
    const auto retired = world.IdOf(hub);
    world.ResetDynamics();
    REQUIRE(world.AddBody({}) == hub);
    CHECK(world.IdOf(hub) != retired);
}

TEST_CASE_FIXTURE(OneWorld, "world: joints engage only while their bodies share a scope") {
    const auto a = world.AddBody({.Scope = 1}), b = world.AddBody({.Scope = 3});
    const auto joint = world.AddJoint({.BodyA = a, .BodyB = b});
    REQUIRE(joint != NoIndex);
    CHECK(world.Joints[joint].Active);
    REQUIRE(world.SetBodyScope(b, 2));
    CHECK(!world.Joints[joint].Active);
    CHECK(world.JointCount() == joint + 1);
    REQUIRE(world.SetJoint(joint, {.BodyA = a, .BodyB = b, .Collide = true}));
    CHECK(!world.Joints[joint].Active);
    REQUIRE(world.SetBodyScope(a, 2));
    CHECK(world.Joints[joint].Active);
    CHECK(world.AddJoint({.BodyA = a, .BodyB = b}) == joint + 1);
    REQUIRE(world.RemoveJoint(joint));
    CHECK(world.AddJoint({.BodyA = a, .BodyB = b}) == joint);
}

TEST_CASE_FIXTURE(OneWorld, "world: geometry edits preserve body properties and release owned storage") {
    SUBCASE("new geometry recomputes what the shape decides and keeps what the body decides") {
        const auto box = world.AddShape(UnitBox);
        constexpr Shape BigBox{.HalfExtents = {1, 1, 1}, .Kind = ShapeBox};
        const auto body = world.AddBody({.Shape = box, .GravityScale = 0.5f, .LinearDamping = 0.04f, .AngularDamping = 0.1f});
        REQUIRE(world.SetBodyShape(body, world.AddShape(BigBox)));

        CHECK(world.Masses[body].InvMass == doctest::Approx(1.f / 8000));
        CHECK(world.Masses[body].GravityScale == 0.5f);
        CHECK(world.Masses[body].LinearDamping == 0.04f);
        CHECK(world.Masses[body].AngularDamping == 0.1f);
    }
    SUBCASE("a shape a live body still has is not removed") {
        const auto box = world.AddShape(UnitBox);
        const auto body = world.AddBody({.Shape = box});

        CHECK(!world.RemoveShape(box));
        REQUIRE(world.RemoveBody(body));
        CHECK(!world.Alive(body));
        CHECK(!world.RemoveBody(body));

        CHECK(world.RemoveShape(box));
        CHECK(!world.RemoveShape(box));
    }
    SUBCASE("runs given back next to each other are one run again") {
        World world{context, {.ShapeVertices = 16}};
        const std::vector<float3> cube = CubeCorners(2);
        const std::vector<float3> prism = PrismPoints(6, 1, 1);

        const auto first = world.AddHull(cube), second = world.AddHull(cube);
        REQUIRE(first != NoIndex);
        REQUIRE(second != NoIndex);
        CHECK(world.AddHull(prism) == NoIndex);
        REQUIRE(world.RemoveShape(first));
        REQUIRE(world.RemoveShape(second));

        const auto big = world.AddHull(prism);
        REQUIRE(big != NoIndex);
        CHECK(world.Shapes[big].VertexCount == 12);
        CHECK(world.Shapes[big].FirstVertex == 0);
    }
}

TEST_CASE_FIXTURE(OneWorld, "world: bounded allocation refuses and rolls back atomically") {
    SUBCASE("a full pool refuses the add and counts it") {
        World world{context, {.Bodies = 2, .Shapes = 1}};

        const auto box = world.AddShape(UnitBox);
        CHECK(box == 0);
        CHECK(world.AddShape(GroundPlane) == NoIndex);
        CHECK(world.Overflow.Shapes == 1);
        CHECK(world.ShapeCount() == 1);

        CHECK(world.AddBody({.Shape = box}) == 0);
        CHECK(world.AddBody({.Shape = box}) == 1);
        CHECK(world.AddBody({.Shape = box}) == NoIndex);
        CHECK(world.Overflow.Bodies == 1);
        CHECK(world.BodyCount() == 2);
    }
    SUBCASE("a compound the engine will not make is refused and counted") {
        const auto box = world.AddShape(UnitBox);
        const auto plane = world.AddShape(GroundPlane);
        const auto mesh = AddQuadMesh(world);
        REQUIRE(mesh != NoIndex);
        const auto pair = world.AddCompound(std::vector<Index>{box, box});
        REQUIRE(pair != NoIndex);

        uint32_t refused = 0;
        CHECK(world.AddCompound(std::vector<Index>(9, box)) != NoIndex);
        CHECK(world.AddCompound(std::vector<Index>{box, pair}) != NoIndex);

        CHECK(world.AddCompound(std::vector<Index>{box, mesh}) != NoIndex);
        CHECK(world.AddCompound(std::vector<Index>{box, plane}) != NoIndex);

        CHECK(world.AddCompound({}) == NoIndex);
        CHECK(world.RefusedCompounds == ++refused);
        CHECK(world.AddCompound(std::vector<Index>{box, world.ShapeCount() + 7}) == NoIndex);
        CHECK(world.RefusedCompounds == ++refused);

        CHECK(world.Overflow.Shapes == 0);

        CHECK(world.AddCompound(std::vector<Index>(8, box)) != NoIndex);
        CHECK(world.RefusedCompounds == refused);
    }
    SUBCASE("integration: compound child pool exhaustion rolls back and can be reused") {
        World world{context, {.Bodies = 4, .Shapes = 32, .CompoundChildren = 12}};
        const auto box = world.AddShape(UnitBox);
        const auto first = world.AddCompound(std::vector<Index>(12, box));
        REQUIRE(first != NoIndex);
        const auto count = world.ShapeCount();
        CHECK(world.AddCompound(std::vector<Index>{box}) == NoIndex);
        CHECK(world.Overflow.CompoundChildren == 1);
        CHECK(world.ShapeCount() == count);
        REQUIRE(world.RemoveShape(first));
        CHECK(world.AddCompound(std::vector<Index>(12, box)) != NoIndex);
    }
}

TEST_CASE_FIXTURE(OneWorld, "world: authored mass and compound frames obey the parallel axis theorem") {
    SUBCASE("a host-supplied mass is what a shape with no volume cannot say for itself") {
        const auto mesh = AddQuadMesh(world);
        REQUIRE(mesh != NoIndex);

        const auto scenery = world.AddBody({.Shape = mesh});
        CHECK(world.Masses[scenery].InvMass == 0);

        const auto moving = world.AddBody({.Shape = mesh, .Mass = {{.Mass = 4, .Inertia = {2, 8, 16}}}});
        CHECK(world.Masses[moving].InvMass == doctest::Approx(0.25f));
        CHECK(world.Masses[moving].InvInertiaLocal[0] == doctest::Approx(0.5f));
        CHECK(world.Masses[moving].InvInertiaLocal[1] == doctest::Approx(0.125f));
        CHECK(world.Masses[moving].InvInertiaLocal[2] == doctest::Approx(0.0625f));

        const auto box = world.AddShape(UnitBox);
        const auto authored = world.AddBody({.Shape = box, .Density = 1000, .Mass = {{.Mass = 2, .Inertia = {1, 1, 1}}}});
        CHECK(world.Masses[authored].InvMass == doctest::Approx(0.5f));

        const auto floaty = world.AddBody({.Shape = mesh, .Mass = {{.Mass = 4, .Inertia = {1, 1, 1}}}, .GravityScale = 0.5f, .LinearDamping = 0.25f});
        CHECK(world.Masses[floaty].GravityScale == doctest::Approx(0.5f));
        CHECK(world.Masses[floaty].LinearDamping == doctest::Approx(0.25f));

        REQUIRE(world.SetBodyShape(scenery, mesh, 1000, AuthoredMass{.Mass = 5, .Inertia = {1, 2, 4}}));
        CHECK(world.Masses[scenery].InvMass == doctest::Approx(0.2f));
        CHECK(world.Masses[scenery].InvInertiaLocal[2] == doctest::Approx(0.25f));
        REQUIRE(world.SetBodyShape(floaty, mesh));
        CHECK(world.Masses[floaty].InvMass == 0);
        CHECK(world.Masses[floaty].GravityScale == doctest::Approx(0.5f));
    }
    SUBCASE("a body wearing an offset shape is refused unless the host says what it weighs") {
        constexpr Shape OffsetBox{.HalfExtents = {Half, Half, Half}, .Kind = ShapeBox, .Local = {{0.3f, 0, 0}, {0, 0, 0, 1}}};
        const auto offset = world.AddShape(OffsetBox);
        const auto centred = world.AddShape(UnitBox);

        CHECK(world.AddBody({.Shape = offset}) == NoIndex);
        CHECK(world.OffsetsWithoutMass == 1);
        CHECK(world.BodyCount() == 0);
        CHECK(world.Overflow.Bodies == 0);

        const auto authored = world.AddBody({.Shape = offset, .Mass = {{.Mass = 1000, .Inertia = {200, 250, 250}}}});
        REQUIRE(authored != NoIndex);
        CHECK(world.Masses[authored].InvMass == doctest::Approx(1.f / 1000));

        const auto fixed = world.AddBody({.Shape = offset, .Density = 0});
        REQUIRE(fixed != NoIndex);
        CHECK(world.Masses[fixed].InvMass == 0);
        const auto mesh = AddQuadMesh(world, At(float3{0, 0, 0}, QuatFromRotationVector(float3{0, 0.7f, 0})));
        REQUIRE(mesh != NoIndex);
        CHECK(world.AddBody({.Shape = mesh}) != NoIndex);
        CHECK(world.OffsetsWithoutMass == 1);

        const auto body = world.AddBody({.Shape = centred});
        REQUIRE(body != NoIndex);
        CHECK(!world.SetBodyShape(body, offset));
        CHECK(world.OffsetsWithoutMass == 2);
        CHECK(world.BodyShapes[body] == centred);
        CHECK(world.Masses[body].InvMass == doctest::Approx(1.f / 1000));
        CHECK(world.SetBodyShape(body, offset, 1000, AuthoredMass{.Mass = 2, .Inertia = {1, 1, 1}}));
        CHECK(world.BodyShapes[body] == offset);
        CHECK(world.Masses[body].InvMass == doctest::Approx(0.5f));
    }

    SUBCASE("a dumbbell's inertia is its pieces carried by the parallel axis theorem") {
        constexpr float Radius = 0.2f, Rod = 0.05f, Reach = 0.6f, Density = 1000;
        constexpr Shape RodShape{.HalfExtents = {0, Reach, 0}, .Radius = Rod, .Kind = ShapeCapsule};
        constexpr Shape BallShape{.Radius = Radius, .Kind = ShapeSphere};
        std::vector<Index> parts{world.AddShape(RodShape)};
        for (const float side : {-1.f, 1.f})
            parts.push_back(world.AddShape({.Radius = Radius, .Kind = ShapeSphere, .Local = At(float3{0, side * Reach, 0})}));

        Pose frame{};
        const auto dumbbell = world.AddCompound(parts, &frame);
        REQUIRE(dumbbell != NoIndex);
        CHECK(simd::length(frame.Position) < 1e-6f);

        const auto bar = MassProperties(RodShape, Density);
        const auto ball = MassProperties(BallShape, Density);
        const float ball_mass = 1 / ball.InvMass;
        const float along = 1 / bar.InvInertiaLocal.y + 2 / ball.InvInertiaLocal.y;
        const float across = 1 / bar.InvInertiaLocal.x + 2 * (1 / ball.InvInertiaLocal.x + ball_mass * Reach * Reach);

        const auto built = MassOf(world, dumbbell, Density);
        CHECK(1 / built.InvMass == doctest::Approx(1 / bar.InvMass + 2 * ball_mass).epsilon(1e-5));
        CHECK(1 / built.InvInertiaLocal.x == doctest::Approx(across).epsilon(1e-5));
        CHECK(1 / built.InvInertiaLocal.y == doctest::Approx(along).epsilon(1e-5));
        CHECK(1 / built.InvInertiaLocal.z == doctest::Approx(across).epsilon(1e-5));
    }
    SUBCASE("a compound off the origin says where it put the body frame") {
        constexpr float TopY = 0.75f, LegHigh = 0.35f;
        std::vector<Index> parts{world.AddShape({.HalfExtents = {1, 0.05f, 0.6f}, .Kind = ShapeBox, .Local = At(float3{0, TopY, 0})})};
        for (const float x : {-0.9f, 0.9f})
            for (const float z : {-0.5f, 0.5f})
                parts.push_back(world.AddShape({.HalfExtents = {0.05f, LegHigh, 0.05f}, .Kind = ShapeBox, .Local = At(float3{x, LegHigh, z})}));

        Pose frame{};
        const auto table = world.AddCompound(parts, &frame);
        REQUIRE(table != NoIndex);
        CHECK(std::abs(frame.Position.x) < 1e-6f);
        CHECK(std::abs(frame.Position.z) < 1e-6f);
        CHECK(frame.Position.y > 0);
        CHECK(frame.Position.y < TopY);

        const Index top = world.Child(table, 0);
        REQUIRE(top != NoIndex);
        const Pose local = world.Shapes[top].Local;
        const float3 placed = WorldPoint(frame, local.Position);
        CHECK(simd::distance(placed, float3{0, TopY, 0}) < 1e-5f);
    }
}

TEST_CASE_FIXTURE(OneWorld, "world: welding respects coverage motion and collider filters") {
    SUBCASE("static faces that cover each other are buried, on copies the weld owns") {
        const auto tile = world.AddShape({.HalfExtents = {TileHalf, TileTop, TileHalf}, .Kind = ShapeBox});
        const auto near = AddTile(world, tile, -1), far = AddTile(world, tile, 1);
        REQUIRE(far != NoIndex);
        const uint32_t before = world.ShapeCount();

        CHECK(world.WeldStatic() == 2);

        CHECK(world.BodyShapes[near] != tile);
        CHECK(world.BodyShapes[far] != tile);
        CHECK(world.ShapeCount() == before + 2);
        CHECK(InternalFaces(world.Shapes[tile]) == 0);
        CHECK(InternalFaces(world.Shapes[world.BodyShapes[near]]) == 1u << BoxFaceIndex(0, true));
        CHECK(InternalFaces(world.Shapes[world.BodyShapes[far]]) == 1u << BoxFaceIndex(0, false));

        const Index copy = world.BodyShapes[near];
        CHECK(world.WeldStatic() == 2);
        CHECK(world.ShapeCount() == before + 2);
        CHECK(world.BodyShapes[near] == copy);

        REQUIRE(world.SetBodyShape(near, world.BodyShapes[near], 0));
        CHECK(world.BodyShapes[near] == copy);
        CHECK(InternalFaces(world.Shapes[copy]) == 1u << BoxFaceIndex(0, true));

        REQUIRE(world.RemoveBody(far));
        CHECK(world.WeldStatic() == 0);
        CHECK(InternalFaces(world.Shapes[world.BodyShapes[near]]) == 0);
        CHECK(world.ShapeCount() == before + 1);
    }
    SUBCASE("the weld speaks only for what nothing can move") {
        const auto tile = world.AddShape({.HalfExtents = {TileHalf, TileTop, TileHalf}, .Kind = ShapeBox});
        const auto near = AddTile(world, tile, -1);
        const auto far = AddTile(world, tile, 1);
        REQUIRE(far != NoIndex);

        world.Velocities[far].Linear = {1, 0, 0};
        CHECK(world.WeldStatic() == 0);
        CHECK(InternalFaces(world.Shapes[world.BodyShapes[near]]) == 0);
        world.Velocities[far] = {};
        CHECK(world.WeldStatic() == 2);

        world.Masses[far].InvMass = 1e-3f;
        CHECK(world.WeldStatic() == 0);
        CHECK(InternalFaces(world.Shapes[world.BodyShapes[near]]) == 0);
        CHECK(InternalFaces(world.Shapes[world.BodyShapes[far]]) == 0);
    }
    SUBCASE("a partly covered face is not buried") {
        const auto leg = world.AddBody({.Pose = At(float3{0, 0.5f, 0}), .Shape = world.AddShape({.HalfExtents = {0.5f, 0.5f, 0.5f}, .Kind = ShapeBox}), .Density = 0});
        const auto slab = world.AddBody({.Pose = At(float3{0, 1.25f, 0}), .Shape = world.AddShape({.HalfExtents = {2, 0.25f, 2}, .Kind = ShapeBox}), .Density = 0});
        REQUIRE(slab != NoIndex);
        const uint32_t before = world.ShapeCount();

        CHECK(world.WeldStatic() == 1);
        CHECK(InternalFaces(world.Shapes[world.BodyShapes[leg]]) == 1u << BoxFaceIndex(1, true));
        CHECK(world.ShapeCount() == before + 1);

        REQUIRE(world.RemoveBody(slab));
        CHECK(world.WeldStatic() == 0);
        CHECK(InternalFaces(world.Shapes[world.BodyShapes[leg]]) == 0);
        CHECK(world.ShapeCount() == before + 1);
    }
    SUBCASE("collider filters: static welds require matching collider masks") {
        const auto shape = world.AddShape(UnitBox);
        const auto a = world.AddBody({.Pose = At(float3{-Half, 0, 0}), .Shape = shape, .Density = 0, .Layer = 1, .CollidesWith = 2});
        const auto b = world.AddBody({.Pose = At(float3{Half, 0, 0}), .Shape = shape, .Density = 0, .Layer = 4, .CollidesWith = 8});
        CHECK(world.WeldStatic() == 0);
        world.Filters[b].Layer = 1;
        world.Filters[b].Collides = 2;
        CHECK(world.WeldStatic() == 2);
        world.Filters[a].Collides = 0;
        CHECK(world.WeldStatic() == 0);
    }
}

TEST_CASE_FIXTURE(OneWorld, "world: nested compounds flatten and release owned storage") {
    const auto box = world.AddShape(UnitBox);
    const auto inner = world.AddCompound(std::vector<Index>(12, box));
    REQUIRE(inner != NoIndex);
    world.Shapes[inner].Local = At(float3{3, 0, 0});
    Pose frame;
    const auto outer = world.AddCompound(std::vector<Index>{inner, inner}, &frame);
    REQUIRE(outer != NoIndex);
    CHECK(world.Shapes[outer].VertexCount == 24);
    CHECK(frame.Position.x == doctest::Approx(3));
    CHECK(world.Child(outer, 24) == NoIndex);
    const auto first = world.Shapes[outer].FirstVertex;
    REQUIRE(world.RemoveShape(outer));
    const auto again = world.AddCompound(std::vector<Index>{inner, inner});
    REQUIRE(again != NoIndex);
    CHECK(world.Shapes[again].FirstVertex == first);
    CHECK(OwnChild(ChildPair(100000, 200000)) == 100000);
    CHECK(OtherChild(ChildPair(100000, 200000)) == 200000);
}

TEST_CASE("cooking: primitive mass inertia and scale match solid integrals") {
    constexpr float Pi = std::numbers::pi_v<float>;
    struct Case {
        Shape ShapeDesc;
        float Volume;
        float3 InertiaPerMass;
    };
    const Case cases[]{
        {UnitBox, 1, {1.f / 6, 1.f / 6, 1.f / 6}},
        {{.Radius = Half, .Kind = ShapeSphere}, Pi / 6, {0.1f, 0.1f, 0.1f}},
        {{.HalfExtents = {Half, 0.7f, 0}, .Kind = ShapeCylinder}, Pi * 0.25f * 1.4f, {0.25f / 4 + 0.49f / 3, 0.25f / 2, 0.25f / 4 + 0.49f / 3}},
        {{.Radius = Half, .Kind = ShapeCapsule}, Pi / 6, {0.1f, 0.1f, 0.1f}}
    };
    for (const auto &c : cases) {
        CAPTURE(c.ShapeDesc.Kind);
        const auto mass = MassProperties(c.ShapeDesc, 1000);
        CHECK(1 / mass.InvMass == doctest::Approx(1000 * c.Volume));
        auto twice = c.ShapeDesc;
        twice.HalfExtents *= 2;
        twice.Radius *= 2;
        const auto scaled = MassProperties(twice, 1000);
        CHECK(mass.InvMass / scaled.InvMass == doctest::Approx(8));
        for (uint32_t axis = 0; axis < 3; ++axis) {
            CHECK(1 / mass.InvInertiaLocal[axis] == doctest::Approx(1000 * c.Volume * c.InertiaPerMass[axis]));
            CHECK(mass.InvInertiaLocal[axis] / scaled.InvInertiaLocal[axis] == doctest::Approx(32));
        }
        CHECK(MassProperties(c.ShapeDesc, 0).InvMass == 0);
    }
    const auto capsule = MassProperties({.HalfExtents = {0, 1, 0}, .Radius = Half, .Kind = ShapeCapsule}, 1000);
    CHECK(1 / capsule.InvMass == doctest::Approx(1000 * (Pi / 2 + Pi / 6)));
    CHECK(1 / capsule.InvInertiaLocal.x > 4 / capsule.InvInertiaLocal.y);
    CHECK(capsule.InvInertiaLocal.x == doctest::Approx(capsule.InvInertiaLocal.z));
    CHECK(MassProperties(GroundPlane, 1000).InvMass == 0);
}

TEST_CASE_FIXTURE(OneWorld, "world: swept mesh BVH returns every triangle pair and preserves identity") {
    std::vector<float3> points;
    std::vector<uint32_t> indices;
    for (uint32_t i = 0; i < 10; ++i) {
        const uint32_t start = uint32_t(points.size());
        const float x = 2.f * i;
        points.push_back(float3{x, 0, 0}); points.push_back(float3{x + 1, 0, 0}); points.push_back(float3{x, 0, 1});
        indices.insert(indices.end(), {start, start + 1, start + 2});
    }
    const Index mesh = world.AddMesh(points, indices);
    const Index a = world.AddBody({.Shape = mesh, .Density = 0, .GeometryActive = true});
    const Index b = world.AddBody({.Pose = At(float3{0, 2, 0}), .Shape = mesh, .Density = 0});
    REQUIRE(a != NoIndex); REQUIRE(b != NoIndex);
    const Shape shape = world.Shapes[mesh];
    REQUIRE(shape.NodeCount > 1);
    std::vector<SweptBounds> ba(shape.VertexCount), bb(shape.VertexCount);
    for (uint32_t i = 0; i < shape.VertexCount; ++i) {
        const auto v = world.ShapeVertices[shape.FirstVertex + i];
        ba[i] = {{v.x, 0, v.z}, {v.x, 0, v.z}};
        // Endpoints y=2 are clear; the caller's conservative motion envelope
        // contains an interior excursion to y=-1. No endpoint sampling here.
        bb[i] = {{v.x + .25, -1, v.z}, {v.x + .25, 2, v.z}};
    }
    std::vector<SweptMesh> inputs{{world.IdOf(a), mesh, NoIndex, ba}, {world.IdOf(b), mesh, NoIndex, bb}};
    const auto before = world.BvhNodes[shape.RootNode];
    const auto pairs = world.CollectSweptTriangles(inputs, 0, 100);
    REQUIRE(pairs.size() == 10);
    for (const auto &pair : pairs) {
        CHECK(pair.A == world.IdOf(a)); CHECK(pair.B == world.IdOf(b));
        CHECK(pair.ShapeA == mesh); CHECK(pair.ShapeB == mesh);
        CHECK(pair.ChildA == NoIndex); CHECK(pair.ChildB == NoIndex);
        CHECK(pair.TriangleA == pair.TriangleB);
    }
    std::reverse(inputs.begin(), inputs.end());
    CHECK(world.CollectSweptTriangles(inputs, 0, 100) == pairs);
    CHECK_THROWS_AS(world.CollectSweptTriangles(inputs, 0, 9), std::overflow_error);
    CHECK(std::memcmp(&before, &world.BvhNodes[shape.RootNode], sizeof(before)) == 0);
    CHECK(world.CollectSweptTriangles(inputs, 0, 100) == pairs);
    for (auto &bound : bb) bound.Low[1] = bound.High[1] = 2;
    CHECK(world.CollectSweptTriangles(inputs, 0, 100).empty());
    CHECK(world.CollectSweptTriangles(inputs, 2, 100).size() >= 10);
    bb[0].Low[0] = std::numeric_limits<double>::quiet_NaN();
    CHECK_THROWS_AS(world.CollectSweptTriangles(inputs, 0, 100), std::invalid_argument);
}

TEST_CASE_FIXTURE(OneWorld, "world: swept mesh query honors leaf filters activity joints and generations") {
    const Index mesh = AddQuadMesh(world);
    const Index compound = world.AddCompound(std::vector<Index>{mesh, mesh});
    REQUIRE(compound != NoIndex);
    const Index a = world.AddBody({.Shape = compound, .Density = 0, .GeometryActive = true});
    const Index b = world.AddBody({.Shape = mesh, .Density = 0});
    REQUIRE(a != NoIndex); REQUIRE(b != NoIndex);
    const SweptBounds broad{{-2, -2, -2}, {2, 2, 2}};
    std::vector<SweptBounds> vertices(world.Shapes[mesh].VertexCount, broad);
    std::vector<SweptMesh> inputs{{world.IdOf(a), world.Child(compound, 0), 0, vertices},
        {world.IdOf(a), world.Child(compound, 1), 1, vertices}, {world.IdOf(b), mesh, NoIndex, vertices}};
    const auto pairs = world.CollectSweptTriangles(inputs, 0, 32);
    CHECK(pairs.size() == 8); // Four triangle pairs per leaf; never self-body pairs.
    world.Shapes[world.Child(compound, 1)].HasFilter = true;
    world.Shapes[world.Child(compound, 1)].Mask = {0, 0};
    const auto filtered = world.CollectSweptTriangles(inputs, 0, 32);
    REQUIRE(filtered.size() == 4);
    for (const auto &pair : filtered) CHECK(pair.ChildA == 0);
    REQUIRE(world.SetGeometryActive(a, false));
    CHECK(world.CollectSweptTriangles(inputs, 0, 32).empty());
    REQUIRE(world.SetGeometryActive(a, true));
    const Index joint = world.AddJoint({.BodyA = a, .BodyB = b});
    REQUIRE(joint != NoIndex);
    CHECK(world.CollectSweptTriangles(inputs, 0, 32).empty());
    REQUIRE(world.RemoveJoint(joint));
    CHECK(world.CollectSweptTriangles(inputs, 0, 32) == filtered);
    CHECK_THROWS_AS(world.CollectSweptTriangles(std::span(inputs).first(2), 0, 32), std::invalid_argument);
    auto duplicate = inputs; duplicate.push_back(inputs.back());
    CHECK_THROWS_AS(world.CollectSweptTriangles(duplicate, 0, 32), std::invalid_argument);
    CHECK_THROWS_AS(world.CollectSweptTriangles(inputs, -1, 32), std::invalid_argument);
    CHECK_THROWS_AS(world.CollectSweptTriangles(inputs, 0, 0), std::invalid_argument);
    world.Filters[b].Sensor = true;
    CHECK(world.CollectSweptTriangles(std::span(inputs).first(2), 0, 32).empty());
    world.Filters[b].Sensor = false;
    const auto old = world.IdOf(b);
    REQUIRE(world.RemoveBody(b));
    // Retirement intentionally delays slot recycling until reporting completes
    // or dynamics are explicitly reset. Reclaim before exercising generation.
    world.ResetDynamics();
    const Index reused = world.AddBody({.Shape = mesh, .Density = 0});
    REQUIRE(reused == b); CHECK(world.IdOf(reused) != old);
    CHECK_THROWS_AS(world.CollectSweptTriangles(inputs, 0, 32), std::invalid_argument);
    inputs.back().Body = world.IdOf(reused);
    const auto refreshed = world.CollectSweptTriangles(inputs, 0, 32);
    REQUIRE(refreshed.size() == filtered.size());
    for (const auto &pair : refreshed) CHECK(pair.B == world.IdOf(reused));
}

TEST_CASE_FIXTURE(OneWorld, "world: swept interval axes reject disjoint coplanar triangles without hiding motion") {
    const std::vector<float3> pa{float3{0, 0, 0}, float3{2, 0, 0}, float3{0, 0, 2}};
    const std::vector<float3> pb{float3{2, 0, 2}, float3{2, 0, .5}, float3{.5, 0, 2}};
    const std::vector<uint32_t> indices{0, 1, 2};
    const Index sa = world.AddMesh(pa, indices), sb = world.AddMesh(pb, indices);
    const Index a = world.AddBody({.Shape = sa, .Density = 0, .GeometryActive = true});
    const Index b = world.AddBody({.Shape = sb, .Density = 0});
    REQUIRE(a != NoIndex); REQUIRE(b != NoIndex);
    const auto bounds = [&](Index shape) {
        std::vector<SweptBounds> result;
        const auto &s = world.Shapes[shape];
        for (uint32_t i = 0; i < s.VertexCount; ++i) {
            const auto p = world.ShapeVertices[s.FirstVertex + i];
            result.push_back({{p.x, p.y, p.z}, {p.x, p.y, p.z}});
        }
        return result;
    };
    auto ba = bounds(sa), bb = bounds(sb);
    const std::vector<SweptMesh> inputs{{world.IdOf(a), sa, NoIndex, ba}, {world.IdOf(b), sb, NoIndex, bb}};
    // A has x+z<=2; B has x+z>=2.5, despite overlapping XYZ boxes.
    CHECK(world.CollectSweptTriangles(inputs, 0, 1).empty());
    CHECK(world.CollectSweptTriangles(inputs, .3, 1).empty());
    CHECK(world.CollectSweptTriangles(inputs, .36, 1).size() == 1);
    // Closed interval contains a translated touching configuration (x+z=2).
    for (auto &v : bb) { v.Low[0] -= .25; v.Low[2] -= .25; }
    CHECK(world.CollectSweptTriangles(inputs, 0, 1).size() == 1);
    // Widen to an interior crossing, with initial/final triangles still clear.
    for (auto &v : bb) { v.Low[0] -= .5; v.Low[2] -= .5; }
    CHECK(world.CollectSweptTriangles(inputs, 0, 1).size() == 1);
}
