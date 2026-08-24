#include "metal/Context.h"

#include "GpuSource.h"

#include <format>
#include <stdexcept>
#include <string>

namespace {
std::string Describe(NS::Error *error) {
    return error ? error->localizedDescription()->utf8String() : "unknown error";
}
} // namespace

namespace mtl {
Context::Context() {
    Device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
    if (!Device) throw std::runtime_error("No Metal device.");
    NS::Error *error{};
    Compiler = NS::TransferPtr(Device->newCompiler(Make<MTL4::CompilerDescriptor>().get(), &error));
    if (!Compiler) throw std::runtime_error(std::format("Metal compiler: {}", Describe(error)));
    Queue = NS::TransferPtr(Device->newMTL4CommandQueue(Make<MTL4::CommandQueueDescriptor>().get(), &error));
    if (!Queue) throw std::runtime_error(std::format("Metal queue: {}", Describe(error)));
}

NS::SharedPtr<MTL::ComputePipelineState> Context::Pipeline(std::string_view source, const char *name, std::string_view prefix) const {
    const auto text = std::format("{}\n{}\n{}", gpu::SharedSource, prefix, source);
    NS::Error *error{};
    auto library = NS::TransferPtr(Device->newLibrary(NS::String::string(text.c_str(), NS::UTF8StringEncoding), nullptr, &error));
    if (!library) throw std::runtime_error(std::format("Compiling {}: {}", name, Describe(error)));

    auto function = Make<MTL4::LibraryFunctionDescriptor>();
    function->setName(NS::String::string(name, NS::UTF8StringEncoding));
    function->setLibrary(library.get());
    auto descriptor = Make<MTL4::ComputePipelineDescriptor>();
    descriptor->setComputeFunctionDescriptor(function.get());
    auto pipeline = NS::TransferPtr(Compiler->newComputePipelineState(descriptor.get(), nullptr, &error));
    if (!pipeline) throw std::runtime_error(std::format("Pipeline {}: {}", name, Describe(error)));
    return pipeline;
}
} // namespace mtl
