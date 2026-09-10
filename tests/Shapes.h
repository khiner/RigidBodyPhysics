#pragma once

#include "World.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <span>
#include <vector>

using namespace rbp;

constexpr float Half = 0.5f;
constexpr Shape UnitBox{.HalfExtents = {Half, Half, Half}, .Kind = ShapeBox};
constexpr Shape GroundPlane{.Normal = {0, 1, 0}, .Offset = 0, .Kind = ShapePlane};

inline float3 CornerSign(uint32_t corner) {
    return float3{(corner & 1) ? 1.f : -1.f, (corner & 2) ? 1.f : -1.f, (corner & 4) ? 1.f : -1.f};
}

inline std::vector<float3> CubeCorners(float side, float3 at = {0, 0, 0}) {
    const float half = side / 2;
    std::vector<float3> points;
    for (uint32_t corner = 0; corner < 8; ++corner) points.push_back(at + half * CornerSign(corner));
    return points;
}

inline std::vector<float3> PrismPoints(uint32_t sides, float radius, float half_height) {
    std::vector<float3> points;
    for (uint32_t i = 0; i < sides; ++i) {
        const float angle = 2 * std::numbers::pi_v<float> * float(i) / float(sides);
        for (const float y : {-half_height, half_height})
            points.push_back(float3{radius * std::cos(angle), y, radius * std::sin(angle)});
    }
    return points;
}

inline std::vector<float3> SpherePoints(float radius) {
    const float phi = (1 + std::sqrt(5.f)) / 2;
    const std::vector<float3> corners{float3{0, 1, phi}, float3{0, -1, phi}, float3{0, 1, -phi}, float3{0, -1, -phi}, float3{1, phi, 0}, float3{-1, phi, 0}, float3{1, -phi, 0}, float3{-1, -phi, 0}, float3{phi, 0, 1}, float3{-phi, 0, 1}, float3{phi, 0, -1}, float3{-phi, 0, -1}};
    std::vector<float3> points;
    const auto add = [&points, radius](float3 point) {
        const float3 on_sphere = radius * simd::normalize(point);
        for (const float3 already : points)
            if (simd::distance(already, on_sphere) < 1e-4f) return;
        points.push_back(on_sphere);
    };
    for (const float3 corner : corners) add(corner);

    for (uint32_t i = 0; i < corners.size(); ++i)
        for (uint32_t j = i + 1; j < corners.size(); ++j)
            if (simd::distance(corners[i], corners[j]) < 2.01f) add(corners[i] + corners[j]);
    return points;
}

inline std::vector<float3> DenseSpherePoints(uint32_t count, float radius) {
    std::vector<float3> points;
    for (uint32_t i = 0; i < count; ++i) {
        const float angle = 2.39996323f * float(i), height = 1 - 2 * (float(i) + 0.5f) / float(count);
        const float ring = std::sqrt(1 - height * height);
        points.push_back(radius * float3{ring * std::cos(angle), height, ring * std::sin(angle)});
    }
    return points;
}

inline std::vector<float3> WedgePoints() {
    return {float3{-0.4f, -0.15f, -0.3f}, float3{0.5f, -0.15f, -0.3f}, float3{0.5f, -0.15f, 0.35f}, float3{-0.4f, -0.15f, 0.35f}, float3{-0.2f, 0.25f, -0.1f}, float3{0.3f, 0.25f, 0.2f}};
}

// Return a one-metre plate with a 0.1-metre chamfer whose rise tests face classification tolerance.
inline std::vector<float3> ChamferedPlate(float rise) {
    std::vector<float3> points;
    for (const float z : {-Half, Half}) {
        points.push_back(float3{-Half, -0.1f, z});
        points.push_back(float3{Half - 0.1f, -0.1f, z});
        points.push_back(float3{Half, -0.1f + rise, z});
        points.push_back(float3{-Half, 0.1f, z});
        points.push_back(float3{Half, 0.1f, z});
    }
    return points;
}

inline float NearestTo(float3 point, std::span<const float3> among) {
    float nearest = INFINITY;
    for (const float3 other : among) nearest = std::min(nearest, float(simd::distance(point, other)));
    return nearest;
}

constexpr float WheelRadius = 1, WheelHubRadius = 0.35f, WheelHalfWidth = 0.35f;

inline Index WheelMesh(World &world, uint32_t sides, uint32_t paddles) {
    std::vector<float3> points;
    std::vector<uint32_t> indices;

    const auto quad = [&points, &indices](float3 a, float3 b, float3 c, float3 d) {
        const uint32_t at = uint32_t(points.size());
        points.insert(points.end(), {a, b, c, d});
        indices.insert(indices.end(), {at, at + 1, at + 2, at, at + 2, at + 3});
    };
    constexpr float Pi = std::numbers::pi_v<float>;
    const float3 across{0, 0, WheelHalfWidth};
    const auto around = [](float angle, float radius) { return float3{radius * std::cos(angle), radius * std::sin(angle), 0}; };

    for (uint32_t i = 0; i < sides; ++i) {
        const float3 from = around(2 * Pi * float(i) / float(sides), WheelHubRadius);
        const float3 to = around(2 * Pi * float(i + 1) / float(sides), WheelHubRadius);
        quad(from - across, to - across, to + across, from + across);
    }

    for (uint32_t i = 0; i < paddles; ++i) {
        const float angle = 2 * Pi * float(i) / float(paddles);
        const float3 root = around(angle, WheelHubRadius), tip = around(angle, WheelRadius);
        quad(root - across, root + across, tip + across, tip - across);
    }
    return world.AddMesh(points, indices);
}

inline AuthoredMass WheelMass(float mass = 20) {
    const float across = mass * (WheelRadius * WheelRadius / 4 + WheelHalfWidth * WheelHalfWidth / 3);
    return {.Mass = mass, .Inertia = {across, across, mass * WheelRadius * WheelRadius / 2}};
}

inline BodyMass MassOf(const World &world, Index shape, float density) {
    return MassProperties(world.Shapes[shape], density, world.ShapeVertices.All(), world.Shapes.All(), world.CompoundChildren.All());
}

inline uint32_t ActiveContacts(const World &world) {
    uint32_t live = 0;
    for (uint32_t slot = 0; slot < world.Contacts.Capacity; ++slot) live += world.Contacts[slot].Active ? 1 : 0;
    return live;
}
inline uint32_t ActiveContacts(const World &world, Index body) {
    uint32_t live = 0;
    for (uint32_t slot = 0; slot < ContactsPerBody; ++slot) live += world.Contacts[body * ContactsPerBody + slot].Active ? 1 : 0;
    return live;
}

inline void Drive(World &world, Index body, float3 to, float3 velocity) {
    world.Poses[body].Position = to;
    world.Velocities[body] = {.Linear = velocity};
}

inline Index BoxMesh(World &world, float half) {
    std::vector<float3> points;
    std::vector<uint32_t> indices;
    for (uint32_t axis = 0; axis < 3; ++axis) {
        const uint32_t u = (axis + 1) % 3, v = (axis + 2) % 3;
        for (const float side : {-1.f, 1.f}) {
            const auto corner = [&](float a, float b) {
                float3 point{0, 0, 0};
                point[axis] = side * half;
                point[u] = a * half;
                point[v] = b * half;
                return point;
            };
            const uint32_t at = uint32_t(points.size());

            points.insert(points.end(), {corner(-1, -1), corner(side, -side), corner(1, 1), corner(-side, side)});
            indices.insert(indices.end(), {at, at + 1, at + 2, at, at + 2, at + 3});
        }
    }
    return world.AddMesh(points, indices);
}
