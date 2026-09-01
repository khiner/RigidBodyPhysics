#pragma once

// The scene geometry RbpScenes and RbpBench share.
// Each tool keeps its own instrumentation: RbpScenes reports the solve, RbpBench times it.

#include "Shapes.h"
#include "Solver.h"

#include <numbers>
#include <span>
#include <vector>

using namespace rbp;

constexpr float Density = 1000, Friction = 0.5f;

// The ground plane, at the friction every scene uses.
inline Index AddGround(World &world) { return world.AddBody({.Shape = world.AddShape(GroundPlane), .Friction = Friction}); }
// A body at the density and friction every scene uses.
inline Index Place(World &world, Index shape, float3 at, float4 turn = float4{0, 0, 0, 1}) {
    return world.AddBody({.Pose = At(at, turn), .Shape = shape, .Density = Density, .Friction = Friction});
}

// How many of `bodies` are asleep.
inline uint32_t Asleep(const World &world, std::span<const Index> bodies, const StepSettings &settings) {
    uint32_t count = 0;
    for (const Index body : bodies) count += world.Quiet[body] >= settings.SleepSteps ? 1 : 0;
    return count;
}

// A vertical chain of boxes, each touching only the one below it and the one above it.
inline std::vector<Index> BuildStack(World &world, uint32_t boxes) {
    const auto shape = world.AddShape(UnitBox);
    AddGround(world);
    std::vector<Index> stack;
    for (uint32_t i = 0; i < boxes; ++i) stack.push_back(Place(world, shape, float3{0, Half + 1.02f * float(i), 0}));
    return stack;
}

// A pile rather than a chain: each layer is one box narrower and centred, so an upper box sits across four lower ones.
// A box in the middle has nine contact partners.
inline std::vector<Index> BuildRaft(World &world, uint32_t side, uint32_t layers) {
    const auto shape = world.AddShape(UnitBox);
    AddGround(world);
    std::vector<Index> boxes;
    for (uint32_t layer = 0; layer < layers; ++layer) {
        const uint32_t across = side > layer ? side - layer : 1;
        const float centre = 0.5f * float(across - 1);
        for (uint32_t x = 0; x < across; ++x)
            for (uint32_t z = 0; z < across; ++z)
                boxes.push_back(Place(world, shape, float3{float(x) - centre, Half + 1.02f * float(layer), float(z) - centre}));
    }
    return boxes;
}

constexpr float CoinRadius = 0.5f, CoinHalfHeight = 0.15f;

// How far a coin is turned against the one below it.
// `twist` is in steps of the prism's own face, so 0.5 is the worst case.
inline float CoinTwist(uint32_t sides, float twist, uint32_t coin) {
    return 2 * std::numbers::pi_v<float> / float(sides) * twist * float(coin);
}

// A stack of prisms, each twisted against the one below.
// These are the only faces in the scenery that clip to more than the four points a pair keeps.
// Empty when the prism does not cook.
inline std::vector<Index> BuildCoins(World &world, uint32_t sides, uint32_t coins, float twist) {
    AddGround(world);
    const auto shape = world.AddHull(PrismPoints(sides, CoinRadius, CoinHalfHeight));
    if (shape == NoIndex) return {};
    std::vector<Index> stack;
    for (uint32_t i = 0; i < coins; ++i)
        stack.push_back(Place(world, shape, float3{0, CoinHalfHeight + 2 * CoinHalfHeight * 1.02f * float(i), 0}, QuatFromRotationVector(float3{0, CoinTwist(sides, twist, i), 0})));
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

// A static body with that floor as its shape.
// False when the mesh does not cook.
inline bool AddMeshFloor(World &world, uint32_t cells) {
    const Index floor = FloorMesh(world, cells);
    if (floor == NoIndex) return false;
    world.AddBody({.Shape = floor, .Friction = Friction});
    return true;
}
