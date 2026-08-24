// Measure what a dependent compute dispatch costs through a Metal 4 barrier.
//
//   DispatchChain [iterations] [max_chain] [groups] [work]
//
// A solver substep is a chain of dispatches that each read what the previous one wrote, so the
// marginal cost of one link bounds substeps x colors per tick. Four ways to express the chain are
// compared: barriers inside one encoder, one encoder per pass, one command buffer per pass, and
// no barrier at all as the floor.
//
// The same chain costs several times more on a GPU that has been idle than on one already running,
// so every measurement is preceded by a ramp, and the last sweep pays the chain out on a fixed tick
// the way a solver would, which is the number that actually applies.

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Metal/Metal.hpp>
#include <QuartzCore/CABase.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <print>
#include <ranges>
#include <span>
#include <thread>
#include <vector>

// Avoid allocating an NSAutoreleasePool object for every dispatch.
extern "C" void *objc_autoreleasePoolPush();
extern "C" void objc_autoreleasePoolPop(void *);

namespace {
// Each dispatch reads in place what the previous one wrote, so a run of them is a run of true
// dependencies and needs a barrier between every pair. Work sets how much arithmetic hides behind
// the dependency, and at Work 0 an element ends up counting the dispatches that touched it.
constexpr auto StepSource = R"(
#include <metal_stdlib>
using namespace metal;

struct StepParams { uint Count, Work; };

kernel void Step(device float *data [[buffer(0)]], constant StepParams &p [[buffer(1)]],
                 uint i [[thread_position_in_grid]]) {
    if (i >= p.Count) return;
    float v = data[i];
    for (uint k = 0; k < p.Work; ++k) v = fma(v, 1.0000001f, 1e-7f);
    data[i] = v + 1;
}
)";

struct StepParams { uint32_t Count, Work; };

constexpr uint32_t ThreadsPerGroup = 256;

enum class Mode { Barrier, Encoders, Buffers, NoBarrier };
constexpr Mode Modes[]{Mode::Barrier, Mode::Encoders, Mode::Buffers, Mode::NoBarrier};
constexpr std::string_view ModeNames[]{"encoder barrier", "encoder per pass", "buffer per pass", "no barrier"};
std::string_view Name(Mode mode) { return ModeNames[size_t(mode)]; }

// Seconds. Committed is absolute until the GPU stamps are folded against it.
struct Sample { double Committed, Encode, Commit, Start, Gpu, Readable; };

struct Stats { double Gpu50, Gpu99, Encode50, Start50, Readable50; };

constexpr uint32_t WarmupIterations = 64;
constexpr double RampSeconds = 0.25; // back-to-back submission until the GPU clocks settle
constexpr uint32_t RampChain = 64;
constexpr double SpinSeconds = 200e-6; // tail of a paced gap that is spun rather than slept

double Now() { return CACurrentMediaTime(); }

struct AutoreleasePool {
    void *Token = objc_autoreleasePoolPush();
    ~AutoreleasePool() { objc_autoreleasePoolPop(Token); }
};

void WaitUntil(MTL::SharedEvent *event, uint64_t value) { while (!event->waitUntilSignaledValue(value, 1000)) {} }

template<typename T> NS::SharedPtr<T> Make() { return NS::TransferPtr(T::alloc()->init()); }

double Pct(const std::vector<double> &sorted, double p) { return sorted[size_t(p * double(sorted.size() - 1))] * 1e6; } // microseconds

double Pct(std::span<const Sample> samples, double Sample::*field, double p) {
    auto v = samples | std::views::transform(field) | std::ranges::to<std::vector>();
    std::ranges::sort(v);
    return Pct(v, p);
}

struct Probe {
    NS::SharedPtr<MTL::Device> Device{NS::TransferPtr(MTL::CreateSystemDefaultDevice())};
    NS::SharedPtr<MTL::ComputePipelineState> Pipeline;
    NS::SharedPtr<MTL4::CommandQueue> Queue;
    NS::SharedPtr<MTL4::ArgumentTable> Table;
    NS::SharedPtr<MTL::SharedEvent> Event{NS::TransferPtr(Device->newSharedEvent())};
    NS::SharedPtr<MTL::ResidencySet> Residency;
    NS::SharedPtr<MTL::Buffer> Data, ParamBuffer;
    NS::SharedPtr<MTL4::CommandAllocator> Allocator{NS::TransferPtr(Device->newCommandAllocator())};
    std::vector<NS::SharedPtr<MTL4::CommandBuffer>> Buffers;
    std::vector<const MTL4::CommandBuffer *> List;
    uint64_t Signal{};

    // Buffers-per-pass needs one command buffer per link, the other modes need one.
    Probe(uint32_t max_chain, uint32_t max_elements) {
        NS::Error *error{};
        auto library = NS::TransferPtr(Device->newLibrary(NS::String::string(StepSource, NS::UTF8StringEncoding), nullptr, &error));
        if (!library) std::println(stderr, "Shader: {}", error->localizedDescription()->utf8String()), exit(1);
        // Compiling at runtime avoids needing Xcode's Metal toolchain, which is a separate download.
        auto compiler = NS::TransferPtr(Device->newCompiler(Make<MTL4::CompilerDescriptor>().get(), &error));
        if (!compiler) std::println(stderr, "Compiler: {}", error->localizedDescription()->utf8String()), exit(1);
        auto function = Make<MTL4::LibraryFunctionDescriptor>();
        function->setName(MTLSTR("Step"));
        function->setLibrary(library.get());
        auto pipeline_desc = Make<MTL4::ComputePipelineDescriptor>();
        pipeline_desc->setComputeFunctionDescriptor(function.get());
        Pipeline = NS::TransferPtr(compiler->newComputePipelineState(pipeline_desc.get(), nullptr, &error));
        if (!Pipeline) std::println(stderr, "Pipeline: {}", error->localizedDescription()->utf8String()), exit(1);

        Data = NS::TransferPtr(Device->newBuffer(max_elements * sizeof(float), MTL::ResourceStorageModeShared));
        ParamBuffer = NS::TransferPtr(Device->newBuffer(sizeof(StepParams), MTL::ResourceStorageModeShared));

        // Metal 4 has no implicit residency tracking, so resources go in a set attached to the queue.
        Residency = NS::TransferPtr(Device->newResidencySet(Make<MTL::ResidencySetDescriptor>().get(), &error));
        Residency->addAllocation(Data.get());
        Residency->addAllocation(ParamBuffer.get());
        Residency->commit();
        Residency->requestResidency();

        auto table_desc = Make<MTL4::ArgumentTableDescriptor>();
        table_desc->setMaxBufferBindCount(2);
        Table = NS::TransferPtr(Device->newArgumentTable(table_desc.get(), &error));
        Table->setAddress(Data->gpuAddress(), 0);
        Table->setAddress(ParamBuffer->gpuAddress(), 1);

        Queue = NS::TransferPtr(Device->newMTL4CommandQueue(Make<MTL4::CommandQueueDescriptor>().get(), &error));
        Queue->addResidencySet(Residency.get());

        Buffers.reserve(max_chain);
        while (Buffers.size() < max_chain) Buffers.push_back(NS::TransferPtr(Device->newCommandBuffer()));
        List.reserve(max_chain);
    }

    void SetParams(uint32_t groups, uint32_t work) const {
        *static_cast<StepParams *>(ParamBuffer->contents()) = {.Count = groups * ThreadsPerGroup, .Work = work};
    }

    void Fill(float value, uint32_t elements) const {
        std::ranges::fill(std::span{static_cast<float *>(Data->contents()), elements}, value);
    }

    void Begin(MTL4::ComputeCommandEncoder *encoder) const {
        encoder->setArgumentTable(Table.get());
        encoder->setComputePipelineState(Pipeline.get());
    }

    void Encode(Mode mode, uint32_t chain, uint32_t groups) {
        const MTL::Size grid{groups, 1, 1}, threads{ThreadsPerGroup, 1, 1};
        List.clear();
        Allocator->reset();
        if (mode == Mode::Buffers) {
            for (uint32_t k = 0; k < chain; ++k) {
                auto *command_buffer = Buffers[k].get();
                command_buffer->beginCommandBuffer(Allocator.get());
                auto *encoder = command_buffer->computeCommandEncoder();
                Begin(encoder);
                // Queue-scope, since the producing dispatch is in an earlier command buffer.
                if (k) encoder->barrierAfterQueueStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
                encoder->dispatchThreadgroups(grid, threads);
                encoder->endEncoding();
                command_buffer->endCommandBuffer();
                List.push_back(command_buffer);
            }
            return;
        }
        auto *command_buffer = Buffers[0].get();
        command_buffer->beginCommandBuffer(Allocator.get());
        if (mode == Mode::Encoders) {
            for (uint32_t k = 0; k < chain; ++k) {
                auto *encoder = command_buffer->computeCommandEncoder();
                Begin(encoder);
                if (k) encoder->barrierAfterQueueStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
                encoder->dispatchThreadgroups(grid, threads);
                encoder->endEncoding();
            }
        } else {
            auto *encoder = command_buffer->computeCommandEncoder();
            Begin(encoder);
            for (uint32_t k = 0; k < chain; ++k) {
                if (k && mode == Mode::Barrier)
                    encoder->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
                encoder->dispatchThreadgroups(grid, threads);
            }
            encoder->endEncoding();
        }
        command_buffer->endCommandBuffer();
        List.push_back(command_buffer);
    }

    void Commit(MTL4::CommitOptions *options = nullptr) {
        if (options) Queue->commit(List.data(), List.size(), options);
        else Queue->commit(List.data(), List.size());
        Queue->signalEvent(Event.get(), ++Signal);
    }

    // Submit back to back for long enough that the GPU is at a settled clock before measuring.
    void Ramp() {
        SetParams(1, 0);
        for (const double until = Now() + RampSeconds; Now() < until;) {
            const AutoreleasePool pool;
            Encode(Mode::Barrier, RampChain, 1);
            Commit();
            WaitUntil(Event.get(), Signal);
        }
    }

    // One chain per iteration. At tick_hz 0 they follow each other as fast as the round trip allows,
    // otherwise each starts on a fixed period and the GPU idles for whatever is left of it.
    Stats Measure(Mode mode, uint32_t chain, uint32_t groups, uint32_t work, uint32_t iterations, double tick_hz = 0) {
        SetParams(groups, work);
        Fill(1, groups * ThreadsPerGroup);
        const auto total = iterations + WarmupIterations;
        std::vector<Sample> samples(total);
        std::atomic<uint32_t> feedback{0};
        const double period = tick_hz > 0 ? 1 / tick_hz : 0;
        for (uint32_t i = 0; i < total; ++i) {
            const AutoreleasePool pool;
            auto &sample = samples[i];
            const double started = Now();
            const double deadline = started + period;
            Encode(mode, chain, groups);
            const double encoded = Now();
            // Commit consumes its options, so each iteration needs a fresh object.
            auto options = Make<MTL4::CommitOptions>();
            options->addFeedbackHandler([&sample, &feedback](MTL4::CommitFeedback *fb) {
                sample.Start = fb->GPUStartTime(); // raw stamps, folded against Committed after the run
                sample.Gpu = fb->GPUEndTime() - fb->GPUStartTime();
                feedback.fetch_add(1, std::memory_order_release);
            });
            sample.Encode = encoded - started; // the feedback object above is measurement scaffolding
            sample.Committed = Now();
            Commit(options.get());
            sample.Commit = Now() - sample.Committed;
            WaitUntil(Event.get(), Signal);
            sample.Readable = Now() - sample.Committed;
            // Sleep off the bulk of the gap the way a solver waiting on a frame would, then spin the
            // last stretch, so neither a busy core nor timer slop stands in for the GPU's idle time.
            if (const double remaining = deadline - SpinSeconds - Now(); remaining > 0)
                std::this_thread::sleep_for(std::chrono::duration<double>(remaining));
            while (Now() < deadline) {}
        }
        while (feedback.load(std::memory_order_acquire) < total) {} // GPU stamps arrive on the feedback queue
        for (auto &sample : samples) sample.Start -= sample.Committed;

        const auto steady = std::span<const Sample>(samples).subspan(WarmupIterations);
        return {.Gpu50 = Pct(steady, &Sample::Gpu, .5), .Gpu99 = Pct(steady, &Sample::Gpu, .99),
                .Encode50 = Pct(steady, &Sample::Encode, .5), .Start50 = Pct(steady, &Sample::Start, .5),
                .Readable50 = Pct(steady, &Sample::Readable, .5)};
    }

    // A chain of `chain` increments from zero must land on exactly `chain` if the ordering held.
    float RunChain(Mode mode, uint32_t chain, uint32_t groups) {
        const AutoreleasePool pool;
        SetParams(groups, 0);
        Fill(0, groups * ThreadsPerGroup);
        Encode(mode, chain, groups);
        Commit();
        WaitUntil(Event.get(), Signal);
        const std::span values{static_cast<const float *>(Data->contents()), groups * ThreadsPerGroup};
        return std::ranges::min(values);
    }
};

// Marginal cost of one more link, which is what a substep or a colour actually buys.
double PerLink(double gpu_short, double gpu_long, uint32_t short_chain, uint32_t long_chain) {
    return (gpu_long - gpu_short) / double(long_chain - short_chain);
}
} // namespace

int main(int argc, char **argv) {
    const std::span args{argv, size_t(argc)};
    const uint32_t iterations = args.size() > 1 ? uint32_t(std::atoi(args[1])) : 400;
    const uint32_t max_chain = std::max(2u, args.size() > 2 ? uint32_t(std::atoi(args[2])) : 128);
    const uint32_t groups = args.size() > 3 ? uint32_t(std::atoi(args[3])) : 1;
    const uint32_t work = args.size() > 4 ? uint32_t(std::atoi(args[4])) : 0;

    constexpr uint32_t WorkGroups[]{1, 8, 64, 512, 4096}; // threadgroups in the amortization sweep
    constexpr double TickRates[]{0, 2000, 1000, 500, 240, 120, 60}; // 0 submits as fast as the round trip allows
    const uint32_t tick_iterations = std::min(iterations, 100u); // a 60 Hz sweep pays 16.7 ms per iteration
    const uint32_t max_groups = std::max(groups, std::ranges::max(WorkGroups));
    Probe probe(max_chain, max_groups * ThreadsPerGroup);

    std::vector<uint32_t> chains;
    for (uint32_t chain = 1; chain <= max_chain; chain *= 2) chains.push_back(chain);

    std::println("--- dependent dispatch chains on {}, {} threadgroups x {} threads, work {}, {} iterations ---",
                 probe.Device->name()->utf8String(), groups, ThreadsPerGroup, work, iterations);
    std::println("Submitted back to back onto a ramped GPU, which is the cheapest a link ever gets.");
    std::println("{:<17} {:>6} {:>10} {:>10} {:>10} {:>12} {:>12}", "mode", "chain", "gpu p50", "gpu p99",
                 "per link", "cpu encode", "commit->read");
    std::vector<std::pair<Mode, double>> per_link;
    for (auto mode : Modes) {
        double first = 0, last = 0;
        for (auto chain : chains) {
            probe.Ramp();
            const auto stats = probe.Measure(mode, chain, groups, work, iterations);
            if (chain == chains.front()) first = stats.Gpu50;
            last = stats.Gpu50;
            const auto link = chain == chains.front() ? std::string("-")
                                                     : std::format("{:.2f}u", PerLink(first, stats.Gpu50, chains.front(), chain));
            std::println("{:<17} {:>6} {:>9.1f}u {:>9.1f}u {:>10} {:>11.1f}u {:>11.1f}u", Name(mode), chain,
                         stats.Gpu50, stats.Gpu99, link, stats.Encode50, stats.Readable50);
        }
        per_link.emplace_back(mode, PerLink(first, last, chains.front(), max_chain));
    }

    std::println("\n--- how much work hides a barrier, encoder barrier, chain {} ---", max_chain);
    std::println("{:<10} {:>10} {:>10} {:>10} {:>12}", "groups", "threads", "solo p50", "chain p50", "per link");
    for (auto work_groups : WorkGroups) {
        probe.Ramp();
        const auto solo = probe.Measure(Mode::Barrier, 1, work_groups, work, iterations);
        const auto chained = probe.Measure(Mode::Barrier, max_chain, work_groups, work, iterations);
        std::println("{:<10} {:>10} {:>9.1f}u {:>9.1f}u {:>11.2f}u", work_groups, work_groups * ThreadsPerGroup,
                     solo.Gpu50, chained.Gpu50, PerLink(solo.Gpu50, chained.Gpu50, 1, max_chain));
    }

    std::println("\n--- the same chain paid out on a tick, so the GPU idles in between ---");
    std::println("{:<14} {:>10} {:>10} {:>10} {:>13} {:>13}", "tick", "gpu p50", "gpu p99", "per link",
                 "commit->start", "commit->read");
    double tick_link = 0;
    for (auto tick : TickRates) {
        probe.Ramp();
        const auto solo = probe.Measure(Mode::Barrier, 1, groups, work, tick_iterations, tick);
        const auto chained = probe.Measure(Mode::Barrier, max_chain, groups, work, tick_iterations, tick);
        const auto link = PerLink(solo.Gpu50, chained.Gpu50, 1, max_chain);
        if (tick == 60) tick_link = link;
        const auto label = tick == 0 ? std::string("back to back") : std::format("{:g} Hz", tick);
        std::println("{:<14} {:>9.1f}u {:>9.1f}u {:>9.2f}u {:>12.1f}u {:>12.1f}u", label, chained.Gpu50,
                     chained.Gpu99, link, chained.Start50, chained.Readable50);
    }

    std::println("\n--- ordering check, chain {} of increments from zero ---", max_chain);
    for (auto mode : Modes)
        std::println("{:<17} lowest element {:g}, expected {}", Name(mode), probe.RunChain(mode, max_chain, groups), max_chain);

    std::println("\n--- budget ---");
    for (auto [mode, link] : per_link)
        std::println("{:<17} {:>7.2f} us per link ramped: {:>6.0f} links in a 16.67 ms tick, {:>6.0f} in 8.33 ms, {:>5.0f} in 2.00 ms",
                     Name(mode), link, 16.667e3 / link, 8.333e3 / link, 2e3 / link);
    std::println("{:<17} {:>7.2f} us per link on a 60 Hz tick: {:>6.0f} links in a 16.67 ms tick",
                 Name(Mode::Barrier), tick_link, 16.667e3 / tick_link);
    return 0;
}
