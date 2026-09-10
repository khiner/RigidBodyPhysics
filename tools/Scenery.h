#pragma once

#include "BenchGeometry.h"
#include "Shapes.h"
#include "Solver.h"

#include <numbers>
#include <span>
#include <vector>

using namespace rbp;

constexpr float Density = 1000, Friction = 0.5f;

inline Index AddGround(World &world) { return world.AddBody({.Shape = world.AddShape(GroundPlane), .Friction = Friction}); }

inline Index Place(World &world, Index shape, float3 at, float4 turn = float4{0, 0, 0, 1}) {
    return world.AddBody({.Pose = At(at, turn), .Shape = shape, .Density = Density, .Friction = Friction});
}

inline uint32_t Asleep(const World &world, std::span<const Index> bodies, const StepSettings &settings) {
    uint32_t count = 0;
    for (const Index body : bodies) count += world.Quiet[body] >= settings.SleepSteps ? 1 : 0;
    return count;
}

inline std::vector<Index> BuildStack(World &world, uint32_t boxes) {
    const auto shape = world.AddShape(UnitBox);
    AddGround(world);
    std::vector<Index> stack;
    for (uint32_t i = 0; i < boxes; ++i) stack.push_back(Place(world, shape, float3{0, Half + 1.02f * float(i), 0}));
    return stack;
}

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

inline float CoinTwist(uint32_t sides, float twist, uint32_t coin) {
    return 2 * std::numbers::pi_v<float> / float(sides) * twist * float(coin);
}

inline std::vector<Index> BuildCoins(World &world, uint32_t sides, uint32_t coins, float twist) {
    AddGround(world);
    const auto shape = world.AddHull(PrismPoints(sides, CoinRadius, CoinHalfHeight));
    if (shape == NoIndex) return {};
    std::vector<Index> stack;
    for (uint32_t i = 0; i < coins; ++i)
        stack.push_back(Place(world, shape, float3{0, CoinHalfHeight + 2 * CoinHalfHeight * 1.02f * float(i), 0}, QuatFromRotationVector(float3{0, CoinTwist(sides, twist, i), 0})));
    return stack;
}

inline Index FloorMesh(World &world, uint32_t cells) {
    std::vector<float3> points;
    std::vector<uint32_t> indices;
    benchmark::FloorGrid(cells, [&](const auto &point) { points.push_back(float3{point[0], point[1], point[2]}); }, [&](const auto &triangle) { indices.insert(indices.end(), triangle.begin(), triangle.end()); });
    return world.AddMesh(points, indices);
}

inline bool AddMeshFloor(World &world, uint32_t cells) {
    const Index floor = FloorMesh(world, cells);
    if (floor == NoIndex) return false;
    world.AddBody({.Shape = floor, .Friction = Friction});
    return true;
}
