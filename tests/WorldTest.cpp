
#include "World.h"
#include "Shapes.h"

#include <algorithm>
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
