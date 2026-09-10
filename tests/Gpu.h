#pragma once

#include "metal/Buffer.h"
#include "metal/Context.h"

#include <algorithm>
#include <initializer_list>
#include <utility>

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
