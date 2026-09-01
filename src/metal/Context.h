#pragma once

#include <Metal/Metal.hpp>

#include <string_view>

namespace rbp::mtl {
// An owned, default-constructed metal-cpp object.
template<typename T> NS::SharedPtr<T> Make() { return NS::TransferPtr(T::alloc()->init()); }

// Blocks until every command committed to the queue before this call has been carried into the driver and reported complete by the runtime.
// A process that exits while Metal's submission thread is still inside the driver panics the kernel.
// Every owner of queued work therefore drains before its destructor returns.
// A world and a solver drain after returning their residency sets, and the context before releasing the queue.
void Drain(MTL4::CommandQueue *queue);

// The device, its Metal 4 queue and the runtime shader compiler.
// Kernels are compiled from embedded source rather than from a .metallib, which keeps the build off Xcode's separately installed Metal toolchain.
// The shared struct header is then the same text on host and device.
struct Context {
    NS::SharedPtr<MTL::Device> Device;
    NS::SharedPtr<MTL4::CommandQueue> Queue;
    NS::SharedPtr<MTL4::Compiler> Compiler;

    Context();
    ~Context(); // drains. See Drain.
    Context(Context &&) = delete;
    Context &operator=(Context &&) = delete;

    // Compiles the shared header, then `prefix`, then `source`, and returns the named kernel.
    // `prefix` supplies a #define when one kernel text is compiled more than one way.
    // Throws on a compile failure, carrying the Metal diagnostic.
    NS::SharedPtr<MTL::ComputePipelineState> Pipeline(std::string_view source, const char *name, std::string_view prefix = {}) const;
};
} // namespace rbp::mtl
