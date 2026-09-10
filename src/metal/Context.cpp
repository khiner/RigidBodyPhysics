#include "metal/Context.h"

#include "GpuSource.h"

#include <cstdlib>
#include <format>
#include <semaphore>
#include <stdexcept>
#include <string>

namespace {
std::string Describe(NS::Error *error) {
    return error ? error->localizedDescription()->utf8String() : "unknown error";
}
} // namespace

namespace rbp::mtl {
Context::Context() {
    Device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
    if (!Device) throw std::runtime_error("No Metal device.");
    NS::Error *error{};
    Compiler = NS::TransferPtr(Device->newCompiler(Make<MTL4::CompilerDescriptor>().get(), &error));
    if (!Compiler) throw std::runtime_error(std::format("Metal compiler: {}", Describe(error)));
    Queue = NS::TransferPtr(Device->newMTL4CommandQueue(Make<MTL4::CommandQueueDescriptor>().get(), &error));
    if (!Queue) throw std::runtime_error(std::format("Metal queue: {}", Describe(error)));
}

Context::~Context() {
    if (Queue) Drain(Queue.get());
}

void Drain(MTL4::CommandQueue *queue) {
    auto *device = queue->device();
    auto allocator = NS::TransferPtr(device->newCommandAllocator());
    auto commands = NS::TransferPtr(device->newCommandBuffer());
    commands->beginCommandBuffer(allocator.get());
    commands->endCommandBuffer();
    // The serial queue reports earlier commands complete before invoking this feedback handler.
    std::binary_semaphore reported{0};
    auto options = Make<MTL4::CommitOptions>();
    options->addFeedbackHandler([&reported](MTL4::CommitFeedback *) { reported.release(); });
    const MTL4::CommandBuffer *list[]{commands.get()};
    queue->commit(list, 1, options.get());
    reported.acquire();
}

NS::SharedPtr<MTL::ComputePipelineState> Context::Pipeline(std::string_view source, const char *name, std::string_view prefix, bool safe_math, bool indirect) const {
    const auto text = std::format("{}\n{}\n{}", gpu::SharedSource, prefix, source);
    NS::Error *error{};
    // RBP_MATH=safe disables fast-math transformations for rounding diagnostics.
    const char *const math = getenv("RBP_MATH");
    auto options = Make<MTL::CompileOptions>();
    if (safe_math || (math != nullptr && std::string_view{math} == "safe")) options->setMathMode(MTL::MathModeSafe);
    auto library = NS::TransferPtr(Device->newLibrary(NS::String::string(text.c_str(), NS::UTF8StringEncoding), options.get(), &error));
    if (!library) throw std::runtime_error(std::format("Compiling {}: {}", name, Describe(error)));

    auto function = Make<MTL4::LibraryFunctionDescriptor>();
    function->setName(NS::String::string(name, NS::UTF8StringEncoding));
    function->setLibrary(library.get());
    auto descriptor = Make<MTL4::ComputePipelineDescriptor>();
    descriptor->setComputeFunctionDescriptor(function.get());
    if (indirect) descriptor->setSupportIndirectCommandBuffers(MTL4::IndirectCommandBufferSupportStateEnabled);
    auto pipeline = NS::TransferPtr(Compiler->newComputePipelineState(descriptor.get(), nullptr, &error));
    if (!pipeline) throw std::runtime_error(std::format("Pipeline {}: {}", name, Describe(error)));
    return pipeline;
}
} // namespace rbp::mtl
