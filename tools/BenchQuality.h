#pragma once

#include <algorithm>
#include <array>
#include <cmath>

namespace benchmark {
// State contains position, xyzw quaternion, linear velocity and angular velocity.
// Boxes rest on y = 0 with unit side length by default.
struct Quality {
    bool Finite = true;
    double FloorPenetration = 0, VerticalOverlap = 0, HorizontalDrift = 0, QuaternionError = 0;
    double PreviousY = 0, PreviousHalfHeight = 0;

    void Observe(const std::array<float, 13> &state, std::array<float, 2> origin = {}, int column_row = -1, std::array<float, 3> half = {0.5f, 0.5f, 0.5f}) {
        for (float value : state)
            if (!std::isfinite(value)) {
                Finite = false;
                return;
            }
        const double x = state[3], y = state[4], z = state[5], w = state[6];
        QuaternionError = std::max(QuaternionError, std::abs(std::sqrt(x * x + y * y + z * z + w * w) - 1));
        if (column_row < 0) return;
        const double half_height = half[0] * std::abs(2 * (x * y + w * z)) + half[1] * std::abs(1 - 2 * (x * x + z * z)) + half[2] * std::abs(2 * (y * z - w * x));
        FloorPenetration = std::max(FloorPenetration, half_height - state[1]);
        if (column_row > 0) VerticalOverlap = std::max(VerticalOverlap, PreviousHalfHeight + half_height - (state[1] - PreviousY));
        const double dx = double(state[0]) - origin[0], dz = double(state[2]) - origin[1];
        HorizontalDrift = std::max(HorizontalDrift, std::sqrt(dx * dx + dz * dz));
        PreviousY = state[1];
        PreviousHalfHeight = half_height;
    }
};
} // namespace benchmark
