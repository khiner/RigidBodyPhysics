#include "Scenarios.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace ji {
namespace {

rbp::Index AddShape(rbp::World &world, rbp::Shape shape) {
    const rbp::Index result = world.AddShape(shape);
    if (result == rbp::NoIndex) throw std::runtime_error("Ji scenario shape capacity exceeded");
    return result;
}

rbp::Index AddBox(rbp::World &world, rbp::Index shape, rbp::float3 half,
                  rbp::float3 position, float mass, float friction,
                  rbp::float4 orientation = {0, 0, 0, 1},
                  rbp::float3 velocity = {0, 0, 0}) {
    const rbp::float3 inertia{
        mass * (half.y * half.y + half.z * half.z) / 3,
        mass * (half.x * half.x + half.z * half.z) / 3,
        mass * (half.x * half.x + half.y * half.y) / 3};
    const rbp::Index result = world.AddBody({
        .Pose = rbp::At(position, orientation), .Velocity = {.Linear = velocity},
        .Shape = shape, .Mass = rbp::AuthoredMass{mass, inertia}, .Friction = friction});
    if (result == rbp::NoIndex) throw std::runtime_error("Ji scenario body capacity exceeded");
    return result;
}

void Ground(rbp::World &world, float friction) {
    const rbp::Index shape = AddShape(world, {.Normal = {0, 1, 0},
                                              .Offset = 0, .Kind = rbp::ShapePlane});
    if (world.AddBody({.Shape = shape, .Density = 0, .Friction = friction}) == rbp::NoIndex)
        throw std::runtime_error("Ji scenario ground capacity exceeded");
}

void Hinge(rbp::World &world, rbp::Index child, rbp::Index parent, rbp::float3 at,
           uint32_t free_axis) {
    rbp::JointDesc joint;
    joint.BodyA = child;
    joint.BodyB = parent;
    joint.At = at;
    for (uint32_t axis = 0; axis < 3; ++axis)
        if (axis != free_axis) joint.Angular[axis] = rbp::AxisLocked;
    if (world.AddJoint(joint) == rbp::NoIndex)
        throw std::runtime_error("Ji scenario joint capacity exceeded");
}

} // namespace

ScenarioConfig ScenarioFor(std::string_view name, uint32_t sliding_bodies) {
    ScenarioConfig config{};
    config.settings.solve.tolerance = 0; // Table I compares fixed iteration budgets.
    config.settings.contact_error_reduction = 0; // No overlap stabilization; positive look-ahead gaps remain in the contact law.
    config.settings.joint_error_reduction = 0; // Published coefficient unspecified; benchmark correction separately.
    config.settings.max_contact_reach = 0.05;
    if (name == "sliding") {
        if (sliding_bodies != 100 && sliding_bodies != 500 && sliding_bodies != 1000 &&
            sliding_bodies != 1500 && sliding_bodies != 2000)
            throw std::invalid_argument("Sliding bodies must be 100, 500, 1000, 1500 or 2000");
        config.kind = ScenarioKind::Sliding;
        config.name = "sliding";
        config.geometry = "Two-layer box grid sliding across a plane, 0.40m x pitch and 0.46m z pitch; reauthored geometry";
        config.dynamic_bodies = sliding_bodies;
        config.settings.dt = 0.005;
        config.settings.solve.iterations = 120;
        config.settings.solve.penalty_period = 30;
    } else if (name == "vertical" || name == "vertical-shear") {
        const bool shear = name == "vertical-shear";
        config.kind = shear ? ScenarioKind::VerticalShear : ScenarioKind::Vertical;
        config.name = shear ? "vertical-shear" : "vertical";
        config.geometry = shear
            ? "Auxiliary 120 wide-footprint 10g/1kg/100kg stacks, pitched top; not the paper geometry"
            : "120 separated 10g/1kg/100kg three-box stacks with pitched top; reauthored geometry";
        if (shear) {
            config.drive = "Constant world +x force on each top box; auxiliary shear control";
            config.stack_shear_force_N = 942;
        }
        config.dynamic_bodies = 360;
        config.settings.dt = 0.005;
        config.settings.solve.iterations = 200;
        config.settings.solve.penalty_period = 50;
    } else if (name == "oblique") {
        config.kind = ScenarioKind::Oblique;
        config.name = "oblique";
        config.geometry = "Seven five-level oblique stacks of 10kg plates and 50g cube supports; reauthored geometry";
        config.dynamic_bodies = 140;
        config.settings.dt = 0.005;
        config.settings.solve.iterations = 150;
        config.settings.solve.penalty_period = 30;
    } else if (name == "cards") {
        config.kind = ScenarioKind::Cards;
        config.name = "cards";
        config.geometry = "Six connected six-level 66-card houses with leaning pairs, bridges, foundations and small inter-level openings; reauthored geometry";
        config.dynamic_bodies = 396;
        config.card_interlevel_opening_m = 50e-6;
        config.settings.dt = 0.001;
        config.settings.solve.iterations = 300;
        config.settings.solve.penalty_period = 100;
    } else if (name == "articulated") {
        config.kind = ScenarioKind::Articulated;
        config.name = "articulated";
        config.geometry = "Five-finger 34-body/33-hinge hand between two three-level, three-deep cube piles plus 67 floor cubes; reauthored geometry";
        config.drive = "Opposed x-axis finger torques at 30 hinges with palm support, 2s sine period; reauthored, no VIST input";
        config.dynamic_bodies = 281;
        config.joints = 33;
        config.drive_amplitude = 0.05;
        config.wrist_support_N = 40;
        config.settings.dt = 0.004;
        config.settings.solve.iterations = 100;
        config.settings.solve.penalty_period = 20;
    } else if (name == "impact-demo") {
        config.kind = ScenarioKind::ImpactDemo;
        config.name = "impact-demo";
        config.geometry = "Two initially resting boxes and one falling box; authored visual comparison, not a published Ji scene";
        config.dynamic_bodies = 3;
        config.settings.dt = 0.005;
        config.settings.solve.iterations = 120;
        config.settings.solve.penalty_period = 30;
    } else {
        throw std::invalid_argument("Scenario must be sliding, vertical, vertical-shear, oblique, cards, articulated or impact-demo");
    }
    // Twelve slots per moving body cover Table I's 22,660-contact cube peak;
    // the floor exceeds its 2,680-contact card peak. Overflow fails the step.
    config.settings.contact_capacity = std::max(3072u, 12 * config.dynamic_bodies);
    return config;
}

std::vector<Wrench> ScenarioWrenches(const ScenarioConfig &config, const rbp::World &world,
                                    uint64_t step) {
    if (config.kind == ScenarioKind::VerticalShear) {
        std::vector<Wrench> wrenches(world.BodyCount());
        for (uint32_t stack = 0; stack < 120; ++stack)
            wrenches[3 * stack + 3].force[0] = config.stack_shear_force_N;
        return wrenches;
    }
    if (config.kind != ScenarioKind::Articulated ||
        (config.drive_amplitude == 0 && config.wrist_support_N == 0 &&
         (config.wrist_pulse_N == 0 || step >= config.wrist_pulse_steps))) return {};
    std::vector<Wrench> wrenches(world.BodyCount());
    wrenches[world.Joints[0].BodyB].force[1] = config.wrist_support_N +
        (step < config.wrist_pulse_steps ? config.wrist_pulse_N : 0);
    const double time = double(step) * config.settings.dt;
    const double amplitude = config.drive_amplitude * std::sin(std::numbers::pi * time);
    for (uint32_t joint = 0; joint < 30; ++joint) {
        const double direction = joint / 6 < 2 ? 1 : -1;
        const auto &hinge = world.Joints[joint];
        wrenches[hinge.BodyA].torque[0] += direction * amplitude;
        wrenches[hinge.BodyB].torque[0] -= direction * amplitude;
    }
    return wrenches;
}

rbp::WorldLimits LimitsFor(const ScenarioConfig &config) {
    rbp::WorldLimits limits;
    limits.Bodies = config.dynamic_bodies + 1;
    limits.Shapes = 8;
    limits.Joints = std::max(1u, config.joints);
    limits.ShapeVertices = limits.HullFaces = limits.Triangles =
        limits.BvhNodes = limits.CompoundChildren = 1;
    return limits;
}

void PopulateScenario(rbp::World &world, const ScenarioConfig &config) {
    switch (config.kind) {
        case ScenarioKind::ImpactDemo: {
            Ground(world, 0.7f);
            constexpr rbp::float3 half{0.2f, 0.2f, 0.2f};
            const rbp::Index shape = AddShape(world, {.HalfExtents = half, .Kind = rbp::ShapeBox});
            AddBox(world, shape, half, rbp::float3{-0.42f, 0.2f, 0}, 1, 0.7f);
            AddBox(world, shape, half, rbp::float3{0.42f, 0.2f, 0}, 1, 0.7f);
            AddBox(world, shape, half, rbp::float3{-0.22f, 1.5f, 0}, 1, 0.7f,
                   rbp::float4{0, 0, 0, 1}, rbp::float3{0.45f, 0, 0});
            break;
        }
        case ScenarioKind::Sliding: {
            Ground(world, 0.6f);
            constexpr rbp::float3 half{0.2f, 0.2f, 0.2f};
            const rbp::Index shape = AddShape(world, {.HalfExtents = half, .Kind = rbp::ShapeBox});
            const uint32_t width = uint32_t(std::ceil(std::sqrt(double(config.dynamic_bodies) / 2)));
            const uint32_t depth = (config.dynamic_bodies + 2 * width - 1) / (2 * width);
            for (uint32_t i = 0; i < config.dynamic_bodies; ++i) {
                const uint32_t x = i % width, z = (i / width) % depth, layer = i / (width * depth);
                AddBox(world, shape, half,
                       rbp::float3{0.4f * (float(x) - float(width - 1) / 2), 0.2f + 0.4f * float(layer),
                                   0.46f * (float(z) - float(depth - 1) / 2)},
                       1, 0.6f, rbp::float4{0, 0, 0, 1}, rbp::float3{0.4f, 0, 0});
            }
            break;
        }
        case ScenarioKind::Vertical:
        case ScenarioKind::VerticalShear: {
            Ground(world, 0.8f);
            const bool shear = config.kind == ScenarioKind::VerticalShear;
            const rbp::float3 half = shear ? rbp::float3{1, 0.15f, 1}
                                            : rbp::float3{0.15f, 0.15f, 0.15f};
            const rbp::Index shape = AddShape(world, {.HalfExtents = half, .Kind = rbp::ShapeBox});
            for (uint32_t stack = 0; stack < 120; ++stack) {
                const float pitch = shear ? 3.f : 1.5f;
                const float x = pitch * float(stack % 12), z = pitch * float(stack / 12);
                AddBox(world, shape, half, rbp::float3{x, 0.15f, z}, 0.01f, 0.8f);
                AddBox(world, shape, half, rbp::float3{x, 0.45f, z}, 1, 0.8f);
                // Lift the pitched box so its low edge touches without starting interpenetrated.
                const float top_y = shear ? 0.6f + half.y * std::cos(0.025f) +
                                                 half.x * std::sin(0.025f) : 0.7536f;
                AddBox(world, shape, half, rbp::float3{x, top_y, z}, 100, 0.8f,
                       rbp::QuatFromRotationVector({0, 0, 0.025f}));
            }
            break;
        }
        case ScenarioKind::Oblique: {
            Ground(world, 3);
            constexpr rbp::float3 support_half{0.08f, 0.1f, 0.08f};
            constexpr rbp::float3 plate_half{0.45f, 0.05f, 0.25f};
            const rbp::Index support = AddShape(world, {.HalfExtents = support_half, .Kind = rbp::ShapeBox});
            const rbp::Index plate = AddShape(world, {.HalfExtents = plate_half, .Kind = rbp::ShapeBox});
            for (uint32_t stack = 0; stack < 7; ++stack) {
                const float base_x = 2 * float(stack);
                for (uint32_t level = 0; level < 5; ++level) {
                    const float x = base_x + 0.015f * float(level);
                    const float y = 0.3f * float(level);
                    for (float offset : {-0.3f, 0.f, 0.3f})
                        AddBox(world, support, support_half,
                               rbp::float3{x + offset, y + 0.1f, 0}, 0.05f, 3);
                    AddBox(world, plate, plate_half, rbp::float3{x, y + 0.25f, 0},
                           10, 3, rbp::float4{0, 0, 0, 1},
                           level == 4 ? rbp::float3{0.02f, 0, 0} : rbp::float3{0, 0, 0});
                }
            }
            break;
        }
        case ScenarioKind::Cards: {
            Ground(world, 0.7f);
            constexpr rbp::float3 upright_half{0.0125f, 0.3f, 0.12f};
            constexpr rbp::float3 bridge_half{0.3f, 0.0125f, 0.05f};
            // Crossmembers span every pair of feet at the base. Narrow ones left
            // half the upright cards unsupported above the ground-level bridges.
            constexpr rbp::float3 brace_half{0.42f, 0.0125f, 0.3f};
            const rbp::Index upright = AddShape(world, {.HalfExtents = upright_half, .Kind = rbp::ShapeBox});
            const rbp::Index bridge = AddShape(world, {.HalfExtents = bridge_half, .Kind = rbp::ShapeBox});
            const rbp::Index brace = AddShape(world, {.HalfExtents = brace_half, .Kind = rbp::ShapeBox});
            // The paper's card geometry is unavailable. Give intended
            // touching interfaces a small gap while retaining the authored
            // short fall from each bridge onto the uprights below it.
            constexpr float lean = 0.25f, interface_gap = 15e-6f;
            const float interlevel_opening = float(config.card_interlevel_opening_m);
            const float reach = upright_half.y * std::cos(lean) + upright_half.x * std::sin(lean);
            const float apex_half_span = upright_half.y * std::sin(lean) +
                upright_half.x * std::cos(lean) + interface_gap / (2 * std::cos(lean));
            const float bridge_z = bridge_half.z + interface_gap / 2;
            const float brace_y = 2 * bridge_half.y + brace_half.y + interface_gap;
            const float base_y = brace_y + brace_half.y + reach + interface_gap;
            const float level_rise = 2 * reach + 2 * bridge_half.y +
                interlevel_opening + interface_gap;
            for (uint32_t house = 0; house < 6; ++house) {
                const float base_x = 4 * float(house % 3);
                const float base_z = 3 * float(house / 3);
                for (uint32_t i = 0; i < 6; ++i) {
                    const float x = base_x + 0.5f * (float(i) - 2.5f);
                    const float z = base_z + (i % 2 ? bridge_z : -bridge_z);
                    AddBox(world, bridge, bridge_half, rbp::float3{x, 0.0125f, z}, 0.1f, 0.7f);
                }
                for (uint32_t i = 0; i < 3; ++i)
                    AddBox(world, brace, brace_half,
                           rbp::float3{base_x + float(i) - 1, brace_y, base_z}, 0.1f, 0.7f);
                for (uint32_t level = 0; level < 6; ++level) {
                    const uint32_t pairs = 6 - level;
                    const float y = base_y + level_rise * float(level);
                    for (uint32_t i = 0; i < pairs; ++i) {
                        const float x = base_x + 0.5f * (float(i) - 0.5f * float(pairs - 1));
                        AddBox(world, upright, upright_half,
                               rbp::float3{x - apex_half_span, y, base_z}, 0.1f, 0.7f,
                               rbp::QuatFromRotationVector({0, 0, -lean}));
                        AddBox(world, upright, upright_half,
                               rbp::float3{x + apex_half_span, y, base_z}, 0.1f, 0.7f,
                               rbp::QuatFromRotationVector({0, 0, lean}));
                    }
                    if (level == 5) continue;
                    for (uint32_t i = 0; i + 1 < pairs; ++i) {
                        const float x = base_x + 0.5f * (float(i) - 0.5f * float(pairs - 2));
                        const float z = base_z + (pairs == 2 ? 0.f : ((i + level) % 2 ? bridge_z : -bridge_z));
                        AddBox(world, bridge, bridge_half,
                               rbp::float3{x, y + reach + bridge_half.y + interlevel_opening, z},
                               0.1f, 0.7f);
                    }
                }
            }
            break;
        }
        case ScenarioKind::Articulated: {
            Ground(world, 0.8f);
            constexpr rbp::float3 cube_half{0.075f, 0.075f, 0.075f};
            const rbp::Index cube = AddShape(world, {.HalfExtents = cube_half, .Kind = rbp::ShapeBox});
            for (float side : {-1.f, 1.f})
                for (uint32_t depth = 0; depth < 3; ++depth)
                    for (uint32_t row = 0; row < 3; ++row)
                        for (uint32_t column = 0; column < 10; ++column)
                            AddBox(world, cube, cube_half,
                                   rbp::float3{0.17f * (float(column) - 4.5f),
                                               0.075f + 0.15f * float(row),
                                               side * (0.115f + 0.15f * float(depth))},
                                   0.5f, 0.8f);
            for (uint32_t i = 0; i < 67; ++i) {
                const uint32_t row = i / 13;
                AddBox(world, cube, cube_half,
                       rbp::float3{0.17f * (float(i % 13) - 6), 0.075f,
                                   (row % 2 ? -1.f : 1.f) *
                                       (0.66f + 0.17f * float(row / 2))},
                       0.05f, 0.8f);
            }
            constexpr rbp::float3 palm_half{0.28f, 0.1f, 0.16f};
            constexpr rbp::float3 bone_half{0.04f, 0.07f, 0.04f};
            const rbp::Index palm_shape = AddShape(world, {.HalfExtents = palm_half, .Kind = rbp::ShapeBox});
            const rbp::Index bone_shape = AddShape(world, {.HalfExtents = bone_half, .Kind = rbp::ShapeBox});
            const rbp::Index palm = AddBox(world, palm_shape, palm_half, rbp::float3{0, 0.97f, 0}, 1, 0.8f);
            for (uint32_t finger = 0; finger < 5; ++finger) {
                const float x = 0.1f * (float(finger) - 2);
                rbp::Index parent = palm;
                for (uint32_t segment = 0; segment < 6; ++segment) {
                    const float y = 0.795f - 0.145f * float(segment);
                    const rbp::Index child = AddBox(world, bone_shape, bone_half, rbp::float3{x, y, 0},
                                                     0.1f, 0.8f);
                    Hinge(world, child, parent, rbp::float3{x, y + 0.0725f, 0}, 0);
                    parent = child;
                }
            }
            rbp::Index parent = palm;
            for (uint32_t segment = 0; segment < 3; ++segment) {
                const float y = 1.145f + 0.145f * float(segment);
                const rbp::Index child = AddBox(world, bone_shape, bone_half, rbp::float3{0, y, 0},
                                                 0.3f, 0.8f);
                Hinge(world, child, parent, rbp::float3{0, y - 0.0725f, 0}, 1);
                parent = child;
            }
            break;
        }
    }
    if (world.BodyCount() != config.dynamic_bodies + 1 || world.JointCount() != config.joints)
        throw std::runtime_error("Ji scenario population does not match declared dimensions");
}

} // namespace ji
