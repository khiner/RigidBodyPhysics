#include "Shapes.h"
#include "Solver.h"

#include <array>
#include <vector>

#include <doctest/doctest.h>

using namespace rbp;

TEST_CASE("collision: bounded query storage preserves contacts and sensor transitions") {
    CHECK(QueuedQueriesFit(32));
    CHECK(QueuedQueriesFit(4'194'284));
    CHECK_FALSE(QueuedQueriesFit(4'194'285));
    CHECK_FALSE(QueuedQueriesFit(NoIndex));
    SUBCASE("collision storage: dense mesh queries preserve contacts and sensor transitions") {
        mtl::Context context;
        Solver solver{context};

        for (bool sensor : {false, true}) {
            for (uint32_t copies : {3u, 240u, 640u, 480u}) {
                const bool compound = copies == 480;
                const uint32_t leaves = compound ? 2 : 1;
                CAPTURE(sensor);
                CAPTURE(copies);
                World world{context, {.Bodies = 4, .Shapes = 8, .Joints = 1, .ShapeVertices = 16, .HullFaces = 1, .Triangles = (compound ? 5 : 2) * copies, .BvhNodes = (compound ? 10 : 4) * copies, .CompoundChildren = 2}};
                std::vector<uint32_t> indices;
                for (uint32_t i = 0; i < copies; ++i) indices.insert(indices.end(), {0, 1, 2});
                const float3 floor_points[]{{-2, 0, -2}, {0, 0, 2}, {2, 0, -2}};
                const std::array solid_points{float3{-1, 0, -1}, float3{1, 0, -1}, float3{0, 0, 1}};
                const std::array crossing_points{float3{-1, -1, 0}, float3{1, -1, 0}, float3{0, 1, 0}};
                const auto floor_shape = world.AddMesh(floor_points, indices);
                const auto shape = world.AddMesh(sensor ? crossing_points : solid_points, indices);
                REQUIRE(floor_shape != NoIndex);
                REQUIRE(shape != NoIndex);
                REQUIRE(world.Shapes[shape].TriangleCount == copies);
                world.Shapes[floor_shape].DoubleSided = world.Shapes[shape].DoubleSided = true;
                Pose frame = IdentityPose;
                Index body_shape = shape;
                if (compound) {
                    world.Shapes[shape].Local.Position.x = -0.25f;
                    const Index second = world.AddMesh(sensor ? crossing_points : solid_points, indices, At(float3{0.25f, 0, 0}));
                    REQUIRE(second != NoIndex);
                    world.Shapes[second].DoubleSided = true;
                    body_shape = world.AddCompound(std::array{shape, second}, &frame);
                }
                REQUIRE(body_shape != NoIndex);

                Index floor = NoIndex, body = NoIndex;
                if (compound) body = world.AddBody({.Pose = frame, .Shape = body_shape, .Mass = AuthoredMass{1, {1, 1, 1}}});
                floor = world.AddBody({.Shape = floor_shape, .Density = 0, .Sensor = sensor});
                if (!compound) body = world.AddBody({.Shape = body_shape, .Mass = AuthoredMass{1, {1, 1, 1}}});
                REQUIRE(floor != NoIndex);
                REQUIRE(body != NoIndex);
                world.TrackContacts = world.TrackSensors = true;
                const StepSettings settings{.Gravity = {0, 0, 0}, .Iterations = 1, .SleepSteps = ~0u};
                solver.Step(world, settings);
                const auto contacts = world.TakeContactChanges();
                const auto overlaps = world.TakeSensorChanges();
                CHECK(contacts.size() == (sensor ? 0 : 3 * leaves));
                REQUIRE(overlaps.size() == (sensor ? leaves : 0));
                uint32_t sensor_leaves = 0, contact_leaves = 0;
                for (const auto &overlap : overlaps) {
                    CHECK(overlap.Entered);
                    CHECK(overlap.Pair.A.Slot == (compound ? body : floor));
                    CHECK(overlap.Pair.B.Slot == (compound ? floor : body));
                    const uint32_t leaf = compound ? OwnChild(overlap.Pair.Children) : OtherChild(overlap.Pair.Children);
                    REQUIRE(leaf < leaves);
                    sensor_leaves |= 1u << leaf;
                }
                CHECK(sensor_leaves == (sensor ? (1u << leaves) - 1 : 0));
                for (const auto &contact : contacts) {
                    CHECK(contact.Kind == ContactAdded);
                    CHECK(contact.NominalArea > 0);
                    CHECK(length(contact.Lambda) == 0);
                    REQUIRE(contact.A.Slot == body);
                    REQUIRE(OwnChild(contact.Children) < leaves);
                    contact_leaves |= 1u << OwnChild(contact.Children);
                }
                CHECK(contact_leaves == (sensor ? 0 : (1u << leaves) - 1));
                CHECK(length(world.Poses[body].Position) == 0);
                CHECK(length(world.Velocities[body].Linear) == 0);
                CHECK(world.ContactRefusals[body] == 0);
                if (sensor) CHECK(world.SensorRefusals[compound ? body : floor] == 0);

                world.Poses[body].Position.y = 10;
                world.Wake(body);
                solver.Step(world, settings);
                const auto ended_contacts = world.TakeContactChanges();
                const auto ended_overlaps = world.TakeSensorChanges();
                CHECK(ended_contacts.size() == contacts.size());
                for (const auto &contact : ended_contacts) CHECK(contact.Kind == ContactRemoved);
                REQUIRE(ended_overlaps.size() == overlaps.size());
                for (uint32_t i = 0; i < ended_overlaps.size(); ++i) {
                    CHECK(!ended_overlaps[i].Entered);
                    CHECK(ended_overlaps[i].Pair == overlaps[i].Pair);
                }
            }
        }
    }
    SUBCASE("collision storage: compound partitions retain every leaf across body-count changes") {
        mtl::Context context;
        Solver solver{context};
        const StepSettings settings{.Gravity = {0, 0, 0}, .Iterations = 1, .SleepSteps = ~0u};
        constexpr float radius = 0.125f;
        for (const uint32_t leaves : {1u, 3u, 9u, 17u, 33u}) {
            CAPTURE(leaves);
            World world{context, {.Bodies = 40, .Shapes = 80, .Joints = 1, .ShapeVertices = 8, .HullFaces = 1, .Triangles = 2, .BvhNodes = 4, .CompoundChildren = 40}};
            const float3 points[]{{-12, 0, -2}, {-12, 0, 2}, {12, 0, 2}, {12, 0, -2}};
            const Index mesh = world.AddMesh(points, std::array<uint32_t, 6>{0, 1, 2, 0, 2, 3});
            REQUIRE(mesh != NoIndex);
            REQUIRE(world.AddBody({.Shape = mesh, .Density = 0}) != NoIndex);
            std::vector<Index> children;
            for (uint32_t i = 0; i < leaves; ++i) {
                const float x = 0.5f * (float(i) - 0.5f * float(leaves - 1));
                children.push_back(world.AddShape({.Radius = radius, .Kind = ShapeSphere, .Local = At(float3{x, 0, 0})}));
                REQUIRE(children.back() != NoIndex);
            }
            Pose frame;
            const Index compound = world.AddCompound(children, &frame);
            REQUIRE(compound != NoIndex);
            const Pose pose = ComposePose(At(float3{0, radius, 0.23f}), frame);
            const Index body = world.AddBody({.Pose = pose, .Shape = compound});
            REQUIRE(body != NoIndex);
            uint32_t floors = 1;
            auto settle = [&] {
                for (uint32_t step = 0; step < 6; ++step) {
                    solver.Step(world, settings);
                    uint64_t seen = 0;
                    uint32_t count = 0;
                    std::array<uint32_t, 33> partners{};
                    for (uint32_t slot = 0; slot < ContactsPerBody; ++slot) {
                        const auto &contact = world.Contacts[body * ContactsPerBody + slot];
                        if (!contact.Active) continue;
                        REQUIRE(OwnChild(contact.Children) < leaves);
                        const uint32_t leaf = OwnChild(contact.Children);
                        seen |= uint64_t{1} << leaf;
                        REQUIRE(contact.BodyB < 32);
                        CHECK((partners[leaf] & (1u << contact.BodyB)) == 0);
                        partners[leaf] |= 1u << contact.BodyB;
                        ++count;
                        CHECK(contact.Normal.y == doctest::Approx(1));
                        CHECK(length(contact.Lambda) == doctest::Approx(0).epsilon(1e-6));
                    }
                    CHECK(count == leaves * floors);
                    CHECK(seen == (uint64_t{1} << leaves) - 1);
                    const uint32_t expected = floors == 1 ? 1u : 0x11111111u;
                    for (uint32_t leaf = 0; leaf < leaves; ++leaf) CHECK(partners[leaf] == expected);
                    CHECK(length(world.Poses[body].Position - pose.Position) == doctest::Approx(0).epsilon(1e-6));
                    CHECK(length(world.Velocities[body].Linear) == doctest::Approx(0).epsilon(1e-6));
                    CHECK(world.ContactRefusals[body] == 0);
                }
            };
            settle();
            const Index dummy = world.AddShape({.Radius = radius, .Kind = ShapeSphere});
            REQUIRE(dummy != NoIndex);
            std::vector<Index> added;
            while (world.BodyCount() < 32) {
                // Each surface copy requires a contact, including copies across a range boundary.
                const bool near = leaves == 3 && world.BodyCount() % 4 == 0;
                added.push_back(world.AddBody({.Pose = near ? IdentityPose : At(float3{100 + float(world.BodyCount()), 0, 0}), .Shape = near ? mesh : dummy, .Density = 0}));
                REQUIRE(added.back() != NoIndex);
                floors += near;
            }
            settle();
            added.push_back(world.AddBody({.Pose = At(float3{200, 0, 0}), .Shape = dummy, .Density = 0}));
            REQUIRE(added.back() != NoIndex);
            REQUIRE(world.BodyCount() == 33);
            settle();
            for (Index index : added) REQUIRE(world.RemoveBody(index));
            floors = 1;
            settle();
            CHECK(world.BodyCount() == 2);
        }
    }
}
