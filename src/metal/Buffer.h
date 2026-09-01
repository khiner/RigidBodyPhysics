#pragma once

#include <Metal/Metal.hpp>

#include <cstdint>
#include <format>
#include <stdexcept>
#include <span>

namespace rbp::mtl {
// A fixed-capacity shared-storage buffer.
// Under UMA host and device address the same bytes, so there is no upload, staging or readback: the host writes a span and a kernel reads a GPU address.
template<typename T> struct Buffer {
    NS::SharedPtr<MTL::Buffer> Handle;
    uint32_t Capacity{};

    Buffer() = default;
    Buffer(MTL::Device *device, uint32_t capacity)
        : Handle(NS::TransferPtr(device->newBuffer(capacity * sizeof(T), MTL::ResourceStorageModeShared))), Capacity(capacity) {}

    T *Data() const { return static_cast<T *>(Handle->contents()); }
    std::span<T> All() const { return {Data(), Capacity}; }
    // Always bounds checked.
    // These are GPU pages, and a host access past the end of one is not an ordinary segfault.
    // The kernel tears the task down while Metal's submission thread is still inside the driver.
    // The driver's own teardown then dereferences a null and panics the machine.
    // The host-side check turns a bad index into an exception, and the compare is off the hot path.
    T &operator[](uint32_t i) const {
        if (i >= Capacity) throw std::out_of_range(std::format("buffer index {} of {}{}", i, Capacity, i == ~0u ? " (NoIndex, the value a refused add returns)" : ""));
        return Data()[i];
    }
    uint64_t Address() const { return Handle->gpuAddress(); }
};
} // namespace rbp::mtl
