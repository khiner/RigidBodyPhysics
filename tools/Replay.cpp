#include "Replay.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <print>
#include <string_view>

int main(int argc, char **argv) try {
    using namespace rbp;
    using namespace rbp::replay;
    uint32_t limit = ~0u;
    std::vector<std::filesystem::path> paths;
    for (int i = 1; i < argc; ++i) {
        if (std::string_view(argv[i]) == "--steps" && i + 1 < argc) limit = uint32_t(std::stoul(argv[++i]));
        else paths.emplace_back(argv[i]);
    }
    if (paths.empty() || limit == 0) {
        std::println(stderr, "Usage: RbpReplay [--steps N] fixture.rbp ...");
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
            StepSettings settings;
            uint64_t refusals;
            std::vector<State> states;
            if (!reader.Step(settings, refusals, states)) throw std::runtime_error("Physics replay has no steps");
            Follow(world, reader.Followers);
            solver.Step(world, settings);
        }
        Reader reader{path};
        World world{context, reader.Info.Limits};
        reader.Load(world);
        StepSettings settings;
        uint64_t expected_refusals, mismatches = 0, refusals = 0, nonfinite = 0;
        std::vector<State> states;
        std::vector<double> times;
        double total = 0;
        std::array<float, 4> max_error{};
        while (times.size() < limit && reader.Step(settings, expected_refusals, states)) {
            Follow(world, reader.Followers);
            const auto start = std::chrono::steady_clock::now();
            solver.Step(world, settings);
            const auto elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            total += elapsed;
            times.push_back(elapsed);
            const auto refused = Refusals(world);
            refusals += refused;
            passed &= refused == 0 && expected_refusals == 0;
            for (Index i = 0; i < world.BodyCount(); ++i) {
                const auto actual = StateOf(world, i);
                for (size_t c = 0; c < actual.size(); ++c) {
                    nonfinite += !std::isfinite(actual[c]) || !std::isfinite(states[i][c]);
                    mismatches += std::bit_cast<uint32_t>(actual[c]) != std::bit_cast<uint32_t>(states[i][c]);
                    const size_t group = c < 3 ? 0 : c < 7 ? 1 :
                        c < 10                             ? 2 :
                                                             3;
                    max_error[group] = std::max(max_error[group], std::abs(actual[c] - states[i][c]));
                }
            }
            world.TakeContactChanges();
            world.TakeSensorChanges();
        }
        std::ranges::sort(times);
        passed &= mismatches == 0 && nonfinite == 0;
        std::println("{}: {} bodies, {} steps, mean {:.3f} ms, p50 {:.3f} ms, p90 {:.3f} ms, {} differing floats, {} nonfinite, {} refusals", path.filename().string(), world.BodyCount(), times.size(), total / times.size(), times[times.size() / 2], times[(times.size() - 1) * 9 / 10], mismatches, nonfinite, refusals);
        std::println("  max component errors: position {} m, quaternion {}, linear velocity {} m/s, angular velocity {} rad/s", max_error[0], max_error[1], max_error[2], max_error[3]);
    }
    return passed ? 0 : 1;
} catch (const std::exception &error) {
    std::println(stderr, "RbpReplay: {}", error.what());
    return 1;
}
