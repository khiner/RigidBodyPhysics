#pragma once

#include <Metal/Metal.hpp>

#include <cstdint>

namespace rbp::mtl {
template<typename T> NS::SharedPtr<T> Make() { return NS::TransferPtr(T::alloc()->init()); }

// Block until the runtime reports completion of all commands already submitted to the queue.
// Drain queued work before releasing its resources.
void Drain(MTL4::CommandQueue *queue);

struct Context {
    NS::SharedPtr<MTL::Device> Device;
    NS::SharedPtr<MTL4::CommandQueue> Queue;
    NS::SharedPtr<MTL::Library> Library;
    // Exactly one of these is set: the packaged archive when the build produced one, otherwise a runtime compiler.
    NS::SharedPtr<MTL4::Archive> Archive;
    NS::SharedPtr<MTL4::Compiler> Compiler;

    Context();
    ~Context();
    Context(Context &&) = delete;
    Context &operator=(Context &&) = delete;

    // Load a pipeline from the packaged GPU archive, or compile it from the library when no archive is packaged.
    NS::SharedPtr<MTL::ComputePipelineState> Pipeline(uint32_t index) const;
};
} // namespace rbp::mtl
