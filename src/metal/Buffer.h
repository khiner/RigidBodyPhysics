#pragma once

#include <Metal/Metal.hpp>

#include <cstdint>
#include <span>

namespace mtl {
// A fixed-capacity shared-storage buffer. Under UMA these are the same bytes on both sides, so there
// is no upload, no staging and no readback: the host writes a span, a kernel reads a GPU address.
template<typename T> struct Buffer {
    NS::SharedPtr<MTL::Buffer> Handle;
    uint32_t Capacity{};

    Buffer() = default;
    Buffer(MTL::Device *device, uint32_t capacity)
        : Handle(NS::TransferPtr(device->newBuffer(capacity * sizeof(T), MTL::ResourceStorageModeShared))), Capacity(capacity) {}

    T *Data() const { return static_cast<T *>(Handle->contents()); }
    std::span<T> All() const { return {Data(), Capacity}; }
    T &operator[](uint32_t i) const { return Data()[i]; }
    uint64_t Address() const { return Handle->gpuAddress(); }
};
} // namespace mtl
