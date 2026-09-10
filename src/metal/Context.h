#pragma once

#include <Metal/Metal.hpp>

#include <string_view>

namespace rbp::mtl {
template<typename T> NS::SharedPtr<T> Make() { return NS::TransferPtr(T::alloc()->init()); }

// Block until the runtime reports completion of all commands already submitted to the queue.
// Drain queued work before releasing its resources.
void Drain(MTL4::CommandQueue *queue);

// Runtime compilation uses shared host/device headers and requires no separate Metal compiler installation.
struct Context {
    NS::SharedPtr<MTL::Device> Device;
    NS::SharedPtr<MTL4::CommandQueue> Queue;
    NS::SharedPtr<MTL4::Compiler> Compiler;

    Context();
    ~Context();
    Context(Context &&) = delete;
    Context &operator=(Context &&) = delete;

    // Return the named pipeline compiled from the shared header, prefix and source.
    // Compilation failures throw with the Metal diagnostic.
    NS::SharedPtr<MTL::ComputePipelineState> Pipeline(std::string_view source, const char *name, std::string_view prefix = {}, bool safe_math = false, bool indirect = false) const;
};
} // namespace rbp::mtl
