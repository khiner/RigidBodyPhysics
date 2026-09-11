#pragma once

#include "GpuSource.h"
#include "metal/Buffer.h"
#include "metal/Context.h"

#include <algorithm>
#include <format>
#include <initializer_list>
#include <stdexcept>
#include <string_view>
#include <utility>

inline NS::SharedPtr<MTL::ComputePipelineState> CompileProbe(const rbp::mtl::Context &context, std::string_view source, const char *name, std::string_view prefix = {}) {
    const auto text = std::format("{}\n{}\n{}", gpu::SharedSource, prefix, source);
    NS::Error *error{};
    auto options = rbp::mtl::Make<MTL::CompileOptions>();
#ifdef RBP_SAFE_MATH
    options->setMathMode(MTL::MathModeSafe);
#endif
    auto library = NS::TransferPtr(context.Device->newLibrary(NS::String::string(text.c_str(), NS::UTF8StringEncoding), options.get(), &error));
    if (!library) throw std::runtime_error(error->localizedDescription()->utf8String());
    auto function = rbp::mtl::Make<MTL4::LibraryFunctionDescriptor>();
    function->setName(NS::String::string(name, NS::UTF8StringEncoding));
    function->setLibrary(library.get());
    auto descriptor = rbp::mtl::Make<MTL4::ComputePipelineDescriptor>();
    descriptor->setComputeFunctionDescriptor(function.get());
    auto compiler = NS::TransferPtr(context.Device->newCompiler(rbp::mtl::Make<MTL4::CompilerDescriptor>().get(), &error));
    if (!compiler) throw std::runtime_error(error->localizedDescription()->utf8String());
    auto pipeline = NS::TransferPtr(compiler->newComputePipelineState(descriptor.get(), nullptr, &error));
    if (!pipeline) throw std::runtime_error(error->localizedDescription()->utf8String());
    return pipeline;
}

inline void RunGpu(const rbp::mtl::Context &context, MTL::ComputePipelineState *pipeline, std::initializer_list<std::pair<uint32_t, MTL::Buffer *>> bindings, uint32_t count, uint32_t width = 32, bool groups = true) {
    NS::Error *error{};
    auto residency = NS::TransferPtr(context.Device->newResidencySet(rbp::mtl::Make<MTL::ResidencySetDescriptor>().get(), &error));
    uint32_t slots = 0;
    for (const auto [slot, buffer] : bindings) {
        residency->addAllocation(buffer);
        slots = std::max(slots, slot + 1);
    }
    residency->commit();
    residency->requestResidency();
    context.Queue->addResidencySet(residency.get());
    auto descriptor = rbp::mtl::Make<MTL4::ArgumentTableDescriptor>();
    descriptor->setMaxBufferBindCount(slots);
    auto table = NS::TransferPtr(context.Device->newArgumentTable(descriptor.get(), &error));
    for (const auto [slot, buffer] : bindings) table->setAddress(buffer->gpuAddress(), slot);
    auto allocator = NS::TransferPtr(context.Device->newCommandAllocator());
    auto commands = NS::TransferPtr(context.Device->newCommandBuffer());
    commands->beginCommandBuffer(allocator.get());
    auto *encoder = commands->computeCommandEncoder();
    encoder->setArgumentTable(table.get());
    encoder->setComputePipelineState(pipeline);
    if (groups) encoder->dispatchThreadgroups({count, 1, 1}, {width, 1, 1});
    else encoder->dispatchThreads({count, 1, 1}, {width, 1, 1});
    encoder->endEncoding();
    commands->endCommandBuffer();
    auto done = NS::TransferPtr(context.Device->newSharedEvent());
    const MTL4::CommandBuffer *list[]{commands.get()};
    context.Queue->commit(list, 1);
    context.Queue->signalEvent(done.get(), 1);
    while (!done->waitUntilSignaledValue(1, 1000)) {}
    context.Queue->removeResidencySet(residency.get());
    residency->endResidency();
}
