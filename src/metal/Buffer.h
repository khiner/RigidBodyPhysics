#pragma once

#include "AutoreleasePool.h"

#include <Metal/Metal.hpp>

#include <cstdint>
#include <format>
#include <span>
#include <stdexcept>
#include <utility>

namespace rbp::mtl {
// Fixed-capacity shared storage exposes the same bytes to host and device.
template<typename T> struct Buffer {
    NS::SharedPtr<MTL::Buffer> Handle;
    uint32_t Capacity{};

    Buffer() = default;
    Buffer(const Buffer &) = default;
    Buffer(Buffer &&) noexcept = default;
    Buffer &operator=(Buffer other) noexcept {
        const AutoreleasePool pool;
        Handle = std::move(other.Handle);
        Capacity = other.Capacity;
        Mapped = other.Mapped;
        return *this;
    }
    ~Buffer() {
        if (Handle) AutoreleasePool::Release(Handle);
    }
    Buffer(MTL::Device *device, uint32_t capacity) : Capacity(capacity) {
        const AutoreleasePool pool;
        Handle = NS::TransferPtr(device->newBuffer(capacity * sizeof(T), MTL::ResourceStorageModeShared));
        Mapped = static_cast<T *>(Handle->contents());
    }

    T *Data() const { return Handle ? Mapped : nullptr; }
    std::span<T> All() const { return {Data(), Capacity}; }
    // Bounds checks protect GPU allocations from invalid host accesses.
    T &operator[](uint32_t i) const {
        if (i >= Capacity) throw std::out_of_range(std::format("buffer index {} of {}{}", i, Capacity, i == ~0u ? " (NoIndex, the value a refused add returns)" : ""));
        return Data()[i];
    }
    uint64_t Address() const { return Handle->gpuAddress(); }

private:
    // Shared storage keeps this address for the allocation's lifetime.
    T *Mapped{};
};
} // namespace rbp::mtl
