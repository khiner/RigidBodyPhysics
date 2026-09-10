#pragma once

#include <array>
#include <cstdint>
#include <stdexcept>

namespace benchmark {

template<class Vertex, class Triangle> void FloorGrid(uint32_t cells, Vertex vertex, Triangle triangle) {
    if (cells == 0) throw std::invalid_argument("Floor grid needs at least one cell");
    for (uint32_t x = 0; x <= cells; ++x)
        for (uint32_t z = 0; z <= cells; ++z)
            vertex(std::array<float, 3>{20 * (float(x) / float(cells) - 0.5f), 0, 20 * (float(z) / float(cells) - 0.5f)});
    const auto at = [cells](uint32_t x, uint32_t z) { return x * (cells + 1) + z; };
    for (uint32_t x = 0; x < cells; ++x)
        for (uint32_t z = 0; z < cells; ++z) {
            triangle(std::array<uint32_t, 3>{at(x, z), at(x, z + 1), at(x + 1, z + 1)});
            triangle(std::array<uint32_t, 3>{at(x, z), at(x + 1, z + 1), at(x + 1, z)});
        }
}
} // namespace benchmark
