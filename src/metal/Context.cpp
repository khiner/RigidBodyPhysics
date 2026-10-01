#include "metal/Context.h"
#include "metal/AutoreleasePool.h"

#include "Pipelines.h"

#include <filesystem>
#include <format>
#include <semaphore>
#include <stdexcept>
#include <string>
#include <utility>

namespace {
std::string Describe(NS::Error *error) {
    return error ? error->localizedDescription()->utf8String() : "unknown error";
}
} // namespace

namespace rbp::mtl {
Context::Context() {
    const AutoreleasePool pool;
    auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
    if (!device) throw std::runtime_error("No Metal device.");
    NS::Error *error{};
    const std::filesystem::path directory{NS::Bundle::mainBundle()->resourcePath()->utf8String()};
    const auto library_path = (directory / "rbp.metallib").string();
    auto library = NS::TransferPtr(device->newLibrary(NS::String::string(library_path.c_str(), NS::UTF8StringEncoding), &error));
    if (!library) throw std::runtime_error(std::format("Metal library {}: {}", library_path, Describe(error)));
    // The build packages an archive only when the Metal toolchain translates shaders for the local GPU ahead of time.
    NS::SharedPtr<MTL4::Archive> archive;
    NS::SharedPtr<MTL4::Compiler> compiler;
    const auto archive_path = (directory / "rbp.binary.metallib").string();
    if (std::filesystem::exists(archive_path)) {
        archive = NS::TransferPtr(device->newArchive(NS::URL::fileURLWithPath(NS::String::string(archive_path.c_str(), NS::UTF8StringEncoding)), &error));
        if (!archive) throw std::runtime_error(std::format("Metal archive {}: {}", archive_path, Describe(error)));
    } else {
        compiler = NS::TransferPtr(device->newCompiler(Make<MTL4::CompilerDescriptor>().get(), &error));
        if (!compiler) throw std::runtime_error(std::format("Metal compiler: {}", Describe(error)));
    }
    auto queue = NS::TransferPtr(device->newMTL4CommandQueue(Make<MTL4::CommandQueueDescriptor>().get(), &error));
    if (!queue) throw std::runtime_error(std::format("Metal queue: {}", Describe(error)));
    Device = std::move(device);
    Library = std::move(library);
    Archive = std::move(archive);
    Compiler = std::move(compiler);
    Queue = std::move(queue);
}

Context::~Context() {
    if (Queue) Drain(Queue.get());
    AutoreleasePool::Release(Compiler, Archive, Library, Queue, Device);
}

void Drain(MTL4::CommandQueue *queue) {
    const AutoreleasePool pool;
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
    const AutoreleasePool pool;
    if (index >= std::size(shaders::Pipelines)) throw std::out_of_range("Invalid solver pipeline");
    const auto &entry = shaders::Pipelines[index];
    auto function = Make<MTL4::LibraryFunctionDescriptor>();
    function->setName(NS::String::string(entry.Name, NS::UTF8StringEncoding));
    function->setLibrary(Library.get());
    auto descriptor = Make<MTL4::ComputePipelineDescriptor>();
    descriptor->setComputeFunctionDescriptor(function.get());
    if (entry.Indirect) descriptor->setSupportIndirectCommandBuffers(MTL4::IndirectCommandBufferSupportStateEnabled);
    NS::Error *error{};
    auto pipeline = NS::TransferPtr(Archive ? Archive->newComputePipelineState(descriptor.get(), &error) : Compiler->newComputePipelineState(descriptor.get(), nullptr, &error));
    if (!pipeline) throw std::runtime_error(std::format("Pipeline {}: {}", entry.Name, Describe(error)));
    return pipeline;
}
} // namespace rbp::mtl
