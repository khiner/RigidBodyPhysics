#pragma once

// The scenes RbpScenes and RbpBench both build. One pile of boxes answers a different question in each
// - RbpScenes reports what the solver did with it, RbpBench times it - so the geometry lives here and
// each tool keeps only its own instrumentation.

#include "World.h"

#include <cmath>
#include <numbers>
#include <vector>

constexpr float Half = 0.5f, Density = 1000, Friction = 0.5f;
constexpr Shape UnitBox{.HalfExtents = {Half, Half, Half}, .Kind = ShapeBox};
constexpr Shape GroundPlane{.Normal = {0, 1, 0}, .Offset = 0, .Kind = ShapePlane};

// A chain: boxes stacked on the plane, which is the shape the contact run was sized against - a box in
// one touches the box below and the box above, and no more.
inline std::vector<Index> BuildStack(World &world, uint32_t boxes) {
    const auto shape = world.AddShape(UnitBox);
    world.AddBody({.Shape = world.AddShape(GroundPlane), .Friction = Friction});
    std::vector<Index> stack;
    for (uint32_t i = 0; i < boxes; ++i)
        stack.push_back(world.AddBody({.Pose = {.Position = {0, Half + 1.02f * float(i), 0}, .Orientation = {0, 0, 0, 1}},
                                       .Shape = shape,
                                       .Density = Density,
                                       .Friction = Friction}));
    return stack;
}

// A pile rather than a chain: each layer one box narrower than the one under it and centred on it, so
// every box of an upper layer sits across four of the lower one rather than squarely on a single box.
// A box in the middle of it names nine partners, which is what the contact budget is sized from.
inline std::vector<Index> BuildRaft(World &world, uint32_t side, uint32_t layers) {
    const auto shape = world.AddShape(UnitBox);
    world.AddBody({.Shape = world.AddShape(GroundPlane), .Friction = Friction});
    std::vector<Index> boxes;
    for (uint32_t layer = 0; layer < layers; ++layer) {
        const uint32_t across = side > layer ? side - layer : 1;
        for (uint32_t x = 0; x < across; ++x)
            for (uint32_t z = 0; z < across; ++z)
                boxes.push_back(world.AddBody({.Pose = {.Position = {float(x) - 0.5f * float(across - 1), Half + 1.02f * float(layer), float(z) - 0.5f * float(across - 1)},
                                                        .Orientation = {0, 0, 0, 1}},
                                               .Shape = shape,
                                               .Density = Density,
                                               .Friction = Friction}));
    }
    return boxes;
}

constexpr float CoinRadius = 0.5f, CoinHalfHeight = 0.15f;

// How far round a coin of `sides` is turned against the one below it, `twist` being in steps of the
// prism's own face - so 0.5 is the worst case, and it is what the stack is measured back against.
inline float CoinTwist(uint32_t sides, float twist, uint32_t coin) {
    return 2 * std::numbers::pi_v<float> / float(sides) * twist * float(coin);
}

// A stack of prisms, each twisted against the one below - the one scene here with round faces, and so
// the only one whose manifolds are wider than the four a pair may keep. Empty when the prism will not
// cook, which is the one way this scene can fail to build.
inline std::vector<Index> BuildCoins(World &world, uint32_t sides, uint32_t coins, float twist) {
    world.AddBody({.Shape = world.AddShape(GroundPlane), .Friction = Friction});
    std::vector<float3> points;
    for (uint32_t i = 0; i < sides; ++i) {
        const float angle = 2 * std::numbers::pi_v<float> * float(i) / float(sides);
        for (const float y : {-CoinHalfHeight, CoinHalfHeight})
            points.push_back(float3{CoinRadius * std::cos(angle), y, CoinRadius * std::sin(angle)});
    }
    const auto shape = world.AddHull(points);
    if (shape == NoIndex) return {};
    std::vector<Index> stack;
    for (uint32_t i = 0; i < coins; ++i)
        stack.push_back(world.AddBody({.Pose = {.Position = {0, CoinHalfHeight + 2 * CoinHalfHeight * 1.02f * float(i), 0},
                                                .Orientation = QuatFromRotationVector(float3{0, CoinTwist(sides, twist, i), 0})},
                                       .Shape = shape,
                                       .Density = Density,
                                       .Friction = Friction}));
    return stack;
}

// A flat floor of `cells` by `cells` quads across twenty metres, wound to face up.
inline Index FloorMesh(World &world, uint32_t cells) {
    std::vector<float3> points;
    std::vector<uint32_t> indices;
    for (uint32_t x = 0; x <= cells; ++x)
        for (uint32_t z = 0; z <= cells; ++z)
            points.push_back(float3{20 * (float(x) / float(cells) - 0.5f), 0, 20 * (float(z) / float(cells) - 0.5f)});
    const auto at = [cells](uint32_t x, uint32_t z) { return x * (cells + 1) + z; };
    for (uint32_t x = 0; x < cells; ++x)
        for (uint32_t z = 0; z < cells; ++z) {
            indices.insert(indices.end(), {at(x, z), at(x, z + 1), at(x + 1, z + 1)});
            indices.insert(indices.end(), {at(x, z), at(x + 1, z + 1), at(x + 1, z)});
        }
    return world.AddMesh(points, indices);
}
