#pragma once

#include <Metal/Metal.hpp>

#include <string_view>

namespace mtl {
// A default-constructed metal-cpp object, owned. Every descriptor in the engine is made this way.
template<typename T> NS::SharedPtr<T> Make() { return NS::TransferPtr(T::alloc()->init()); }

// The device, its Metal 4 queue and the runtime shader compiler. Kernels are compiled from embedded
// source rather than a .metallib, which keeps the build off Xcode's separately-installed Metal
// toolchain and lets the shared struct header be the literal same text on both sides.
struct Context {
    NS::SharedPtr<MTL::Device> Device;
    NS::SharedPtr<MTL4::CommandQueue> Queue;
    NS::SharedPtr<MTL4::Compiler> Compiler;

    Context();

    // Compiles the shared header, then `prefix`, then `source`, and returns the named kernel. The
    // prefix is where a #define goes when one kernel text needs compiling more than one way. Throws on
    // a compile failure, carrying the Metal diagnostic - there is no useful way to continue without it.
    NS::SharedPtr<MTL::ComputePipelineState> Pipeline(std::string_view source, const char *name, std::string_view prefix = {}) const;
};
} // namespace mtl
