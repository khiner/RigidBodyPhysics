// The Metal compiler and clang compile the same struct text, and this checks the two layouts match.
// Every shared struct's size and alignment is read back from a kernel and compared against the host's.
// A layout drift fails here rather than corrupting a solve.

#include "GpuSource.h"
#include "gpu/Shared.h"
#include "metal/Buffer.h"
#include "metal/Context.h"

#include <algorithm>

#include <doctest/doctest.h>

using namespace rbp;

namespace {
struct Layout {
    const char *Name;
    uint32_t Size, Align;
};

// Same order as Layout.metal's REPORT list.
constexpr Layout HostLayouts[]{
    {"Pose", sizeof(Pose), alignof(Pose)},
    {"Velocity", sizeof(Velocity), alignof(Velocity)},
    {"BodyMass", sizeof(BodyMass), alignof(BodyMass)},
    {"Shape", sizeof(Shape), alignof(Shape)},
    {"Filter", sizeof(Filter), alignof(Filter)},
    {"Contact", sizeof(Contact), alignof(Contact)},
    {"Joint", sizeof(Joint), alignof(Joint)},
    {"Adjacency", sizeof(Adjacency), alignof(Adjacency)},
    {"ContactEvent", sizeof(ContactEvent), alignof(ContactEvent)},
    {"StepParams", sizeof(StepParams), alignof(StepParams)},
};
} // namespace

TEST_CASE("shared structs have the same layout on host and device") {
    const mtl::Context context;
    auto pipeline = context.Pipeline(gpu::LayoutSource, "ReportLayout");

    constexpr uint32_t Slots = 2 * std::size(HostLayouts);
    const mtl::Buffer<uint32_t> reported{context.Device.get(), Slots};
    std::ranges::fill(reported.All(), 0u);

    NS::Error *error{};
    auto residency = NS::TransferPtr(context.Device->newResidencySet(mtl::Make<MTL::ResidencySetDescriptor>().get(), &error));
    residency->addAllocation(reported.Handle.get());
    residency->commit();
    residency->requestResidency();
    context.Queue->addResidencySet(residency.get());

    auto table_desc = mtl::Make<MTL4::ArgumentTableDescriptor>();
    table_desc->setMaxBufferBindCount(1);
    auto table = NS::TransferPtr(context.Device->newArgumentTable(table_desc.get(), &error));
    table->setAddress(reported.Address(), 0);

    auto allocator = NS::TransferPtr(context.Device->newCommandAllocator());
    auto command_buffer = NS::TransferPtr(context.Device->newCommandBuffer());
    command_buffer->beginCommandBuffer(allocator.get());
    auto *encoder = command_buffer->computeCommandEncoder();
    encoder->setArgumentTable(table.get());
    encoder->setComputePipelineState(pipeline.get());
    encoder->dispatchThreads({1, 1, 1}, {1, 1, 1});
    encoder->endEncoding();
    command_buffer->endCommandBuffer();

    // Queue signalling publishes GPU writes to the CPU.
    // Kernel-written flags are unreliable for this.
    auto done = NS::TransferPtr(context.Device->newSharedEvent());
    const MTL4::CommandBuffer *list[]{command_buffer.get()};
    context.Queue->commit(list, 1);
    context.Queue->signalEvent(done.get(), 1);
    while (!done->waitUntilSignaledValue(1, 1000)) {}

    for (uint32_t i = 0; i < std::size(HostLayouts); ++i) {
        const auto &host = HostLayouts[i];
        CAPTURE(host.Name);
        CHECK(reported[2 * i] == host.Size);
        CHECK(reported[2 * i + 1] == host.Align);
    }
}
