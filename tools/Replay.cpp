#include "Replay.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <optional>
#include <print>
#include <string_view>

namespace {
struct Input {
    rbp::StepSettings Settings;
    uint64_t Refusals{};
    std::vector<rbp::replay::State> States;
};
} // namespace

int main(int argc, char **argv) try {
    using namespace rbp;
    using namespace rbp::replay;
    using Clock = std::chrono::steady_clock;
    uint32_t limit = ~0u, batch = 1;
    std::vector<std::filesystem::path> paths;
    for (int i = 1; i < argc; ++i) {
        if (std::string_view(argv[i]) == "--steps" && i + 1 < argc) limit = uint32_t(std::stoul(argv[++i]));
        else if (std::string_view(argv[i]) == "--batch" && i + 1 < argc) batch = uint32_t(std::stoul(argv[++i]));
        else paths.emplace_back(argv[i]);
    }
    if (paths.empty() || limit == 0 || batch == 0) {
        std::println(stderr, "Usage: RbpReplay [--steps N] [--batch N] fixture.rbp ...");
        return 1;
    }
    const mtl::Context context;
    Solver solver{context};
    bool passed = true;
    for (const auto &path : paths) {
        {
            Reader reader{path};
            World world{context, reader.Info.Limits};
            reader.Load(world);
            Input input;
            if (!reader.Step(input.Settings, input.Refusals, input.States)) throw std::runtime_error("Physics replay has no steps");
            solver.Advance(world, input.Settings, 1, reader.Followers);
        }
        Reader reader{path};
        World world{context, reader.Info.Limits};
        reader.Load(world);
        std::optional<Input> pending;
        uint64_t mismatches = 0, refusals = 0, nonfinite = 0;
        uint32_t steps = 0;
        std::vector<double> times;
        double total = 0;
        std::array<float, 4> max_error{};
        while (steps < limit) {
            std::vector<Input> inputs;
            while (inputs.size() < batch && inputs.size() < limit - steps) {
                Input input;
                if (pending) {
                    input = std::move(*pending);
                    pending.reset();
                } else if (!reader.Step(input.Settings, input.Refusals, input.States)) break;
                if (!inputs.empty() && !SameSettings(inputs.front().Settings, input.Settings)) {
                    pending = std::move(input);
                    break;
                }
                inputs.push_back(std::move(input));
            }
            if (inputs.empty()) break;
            size_t seen = 0;
            double checking = 0;
            const auto start = Clock::now();
            solver.Advance(world, inputs.front().Settings, uint32_t(inputs.size()), reader.Followers, [&](const StepResult &step) {
                const auto check_start = Clock::now();
                const auto &expected = inputs.at(seen++);
                if (step.Poses.size() != expected.States.size()) throw std::runtime_error("Physics replay body count changed");
                const auto refused = step.ContactRefusals + step.SensorRefusals;
                refusals += refused;
                passed &= refused == 0 && expected.Refusals == 0;
                for (Index i = 0; i < step.Poses.size(); ++i) {
                    const auto actual = StateOf(step.Poses[i], step.Velocities[i]);
                    for (size_t c = 0; c < actual.size(); ++c) {
                        nonfinite += !std::isfinite(actual[c]) || !std::isfinite(expected.States[i][c]);
                        mismatches += std::bit_cast<uint32_t>(actual[c]) != std::bit_cast<uint32_t>(expected.States[i][c]);
                        const size_t group = c < 3 ? 0 : c < 7 ? 1 :
                            c < 10                             ? 2 :
                                                                 3;
                        max_error[group] = std::max(max_error[group], std::abs(actual[c] - expected.States[i][c]));
                    }
                }
                world.TakeContactChanges();
                world.TakeSensorChanges();
                checking += std::chrono::duration<double, std::milli>(Clock::now() - check_start).count();
            });
            const double elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start).count() - checking;
            if (seen != inputs.size()) throw std::runtime_error("Physics replay did not complete every substep");
            steps += uint32_t(seen);
            total += elapsed;
            times.push_back(elapsed);
        }
        if (times.empty()) throw std::runtime_error("Physics replay has no measured steps");
        std::ranges::sort(times);
        passed &= mismatches == 0 && nonfinite == 0;
        std::println("{}: {} bodies, {} steps in {} batches, mean {:.3f} ms/substep, p50 {:.3f} ms/batch, p90 {:.3f} ms/batch, batch {}, {} differing floats, {} nonfinite, {} refusals", path.filename().string(), world.BodyCount(), steps, times.size(), total / steps, times[times.size() / 2], times[(times.size() - 1) * 9 / 10], batch, mismatches, nonfinite, refusals);
        std::println("  max component errors: position {} m, quaternion {}, linear velocity {} m/s, angular velocity {} rad/s", max_error[0], max_error[1], max_error[2], max_error[3]);
    }
    return passed ? 0 : 1;
} catch (const std::exception &error) {
    std::println(stderr, "RbpReplay: {}", error.what());
    return 1;
}
