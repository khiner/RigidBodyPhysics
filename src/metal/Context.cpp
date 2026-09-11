#include "metal/Context.h"

#include "Pipelines.h"

#include <filesystem>
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
    const std::filesystem::path directory{NS::Bundle::mainBundle()->resourcePath()->utf8String()};
    const auto library_path = (directory / "rbp.metallib").string();
    Library = NS::TransferPtr(Device->newLibrary(NS::String::string(library_path.c_str(), NS::UTF8StringEncoding), &error));
    if (!Library) throw std::runtime_error(std::format("Metal library {}: {}", library_path, Describe(error)));
    const auto archive_path = (directory / "rbp.binary.metallib").string();
    Archive = NS::TransferPtr(Device->newArchive(NS::URL::fileURLWithPath(NS::String::string(archive_path.c_str(), NS::UTF8StringEncoding)), &error));
    if (!Archive) throw std::runtime_error(std::format("Metal archive {}: {}", archive_path, Describe(error)));
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

NS::SharedPtr<MTL::ComputePipelineState> Context::Pipeline(uint32_t index) const {
    if (index >= std::size(shaders::Pipelines)) throw std::out_of_range("Invalid solver pipeline");
    const auto &entry = shaders::Pipelines[index];
    auto function = Make<MTL4::LibraryFunctionDescriptor>();
    function->setName(NS::String::string(entry.Name, NS::UTF8StringEncoding));
    function->setLibrary(Library.get());
    auto descriptor = Make<MTL4::ComputePipelineDescriptor>();
    descriptor->setComputeFunctionDescriptor(function.get());
    if (entry.Indirect) descriptor->setSupportIndirectCommandBuffers(MTL4::IndirectCommandBufferSupportStateEnabled);
    NS::Error *error{};
    auto pipeline = NS::TransferPtr(Archive->newComputePipelineState(descriptor.get(), &error));
    if (!pipeline) throw std::runtime_error(std::format("Loading pipeline {}: {}", entry.Name, Describe(error)));
    return pipeline;
}
} // namespace rbp::mtl
