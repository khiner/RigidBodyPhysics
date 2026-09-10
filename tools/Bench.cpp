

#include "BenchQuality.h"
#include "Replay.h"
#include "Scenery.h"
#include "Solver.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <print>
#include <span>
#include <string>
#include <unistd.h>
#include <vector>

using namespace rbp;

namespace {
uint32_t Env(const char *name, uint32_t fallback) {
    const char *value = getenv(name);
    return value != nullptr ? uint32_t(std::atoi(value)) : fallback;
}

std::string FirstLine(const char *command) {
    std::string line;
    if (FILE *pipe = popen(command, "r")) {
        char buffer[256];
        if (fgets(buffer, sizeof(buffer), pipe)) line = buffer;
        pclose(pipe);
    }
    if (const auto end = line.find_last_not_of(" \n"); end != std::string::npos) line.resize(end + 1);
    return line;
}

void Chain(World &world, uint32_t links, float3 shift = {}) {
    const auto shape = world.AddShape(UnitBox);
    Index previous = world.AddBody({.Pose = At(shift + float3{-Half - 0.01f, 0, 0}), .Density = 0});
    for (uint32_t i = 0; i < links; ++i) {
        const float x = 1.02f * float(i);
        const auto link = Place(world, shape, shift + float3{x, 0, 0});
        world.AddJoint({.BodyA = previous, .BodyB = link, .At = shift + float3{x - Half - 0.01f, 0, 0}});
        previous = link;
    }
}

bool Slab(World &world, float half_width) {
    if (!AddMeshFloor(world, 64)) return false;
    Place(world, world.AddShape({.HalfExtents = {half_width, 0.25f, half_width}, .Kind = ShapeBox}), float3{0, 0.25f, 0});
    return true;
}

void Lattice(World &world, uint32_t across, uint32_t deep, uint32_t high) {
    const auto shape = world.AddShape(UnitBox);
    AddGround(world);
    for (uint32_t x = 0; x < across; ++x)
        for (uint32_t z = 0; z < deep; ++z)
            for (uint32_t y = 0; y < high; ++y)
                Place(world, shape, float3{1.5f * float(x), Half + 1.02f * float(y), 1.5f * float(z)});
}

struct Result {
    std::string Name;
    uint32_t Bodies, Rows, Colors;
    double Min, Median, P90, Max;
    benchmark::Quality Quality;
    uint64_t Refusals;
    uint32_t ColumnHeight;
};

struct BenchSettings {
    bool Sleeping = false;
    uint32_t Capacity = WorldLimits{}.Bodies;
    uint32_t ColumnHeight = 0;
    std::array<float, 3> BoxHalf{0.5f, 0.5f, 0.5f};
};

double Percentile(std::span<const double> sorted, double q) {
    return sorted[size_t(std::llround(q * double(sorted.size() - 1)))];
}
} // namespace

int main(int argc, char **argv) try {
    const std::span args(argv, size_t(argc));
    const auto wanted = [&](const std::string &name) {
        if (args.size() < 2) return true;
        for (size_t i = 1; i < args.size(); ++i)
            if (name == args[i]) return true;
        return false;
    };
    const uint32_t timed = Env("STEPS", 300), warmup = Env("WARMUP", 120);
    const int requested_substeps = getenv("SUBSTEPS") ? std::atoi(getenv("SUBSTEPS")) : 1;
    if (timed == 0 || requested_substeps <= 0) throw std::runtime_error("STEPS and SUBSTEPS must be positive");
    const uint32_t substeps = uint32_t(requested_substeps);

    const std::string power = FirstLine("pmset -g batt");
    const bool on_ac = power.contains("AC Power");
    std::println("RbpBench: {}, {}, {} timed updates after {} warmup, {} substeps/update, ms/substep, sleeping off", FirstLine("sysctl -n machdep.cpu.brand_string"), on_ac ? "AC power" : "ON BATTERY", timed, warmup, substeps);

    const std::string siblings = "pgrep -l Rbp | grep -v '^" + std::to_string(getpid()) + " ' | head -1";
    if (const std::string other = FirstLine(siblings.c_str()); !other.empty())
        std::println("!! another Rbp process is running ({}) - kill it or these timings are noise", other);

    StepSettings settings;
    settings.SleepSteps = ~0u;

    if (getenv("GRAVITY")) settings.Gravity = {0, float(std::atof(getenv("GRAVITY"))), 0};
    settings.Iterations = Env("ITERATIONS", settings.Iterations);
    settings.MaxColors = Env("COLORS", settings.MaxColors);
    if (settings.Iterations != StepSettings{}.Iterations || settings.MaxColors != StepSettings{}.MaxColors)
        std::println("overridden: {} iterations, {} colours cap", settings.Iterations, settings.MaxColors);
    const mtl::Context context;
    std::vector<Result> results;

    const auto bench = [&](const std::string &name, auto build, BenchSettings options = {}) {
        if (!wanted(name)) return;
        Solver solver{context};
        World world{context, {.Bodies = options.Capacity}};
        if (!build(world)) throw std::runtime_error(name + ": scene would not build");
        const auto &overflow = world.Overflow;
        if (overflow.Bodies || overflow.Shapes || overflow.Joints || overflow.ShapeVertices || overflow.HullFaces || overflow.Triangles || overflow.BvhNodes || overflow.CompoundChildren)
            throw std::runtime_error(name + ": scene capacity exceeded");
        benchmark::Quality quality;
        uint64_t refusals = 0;
        std::vector<std::array<float, 2>> origins(world.BodyCount());
        for (Index body = 0; body < world.BodyCount(); ++body) origins[body] = {world.Poses[body].Position.x, world.Poses[body].Position.z};
        const auto observe = [&] {
            for (Index body = 0; body < world.BodyCount(); ++body)
                quality.Observe(replay::StateOf(world, body), origins[body], options.ColumnHeight && body > 0 ? int((body - 1) % options.ColumnHeight) : -1, options.BoxHalf);
        };

        StepSettings scene_settings = settings;
        if (options.Sleeping) scene_settings.SleepSteps = StepSettings{}.SleepSteps;
        const auto advance = [&] {
            const auto result = solver.Advance(world, scene_settings, substeps);
            if (result.Steps != substeps) throw std::runtime_error(name + ": incomplete update");
            refusals += result.ContactRefusals + result.SensorRefusals;
        };
        for (uint32_t step = 0; step < warmup; ++step) {
            advance();
            observe();
        }
        if (options.Sleeping) {
            const auto all_asleep = [&] {
                for (uint32_t body = 0; body < world.BodyCount(); ++body)
                    if (world.Masses[body].InvMass > 0 && world.Quiet[body] < scene_settings.SleepSteps) return false;
                return true;
            };
            uint32_t patience = 3600;
            while (!all_asleep() && patience-- > 0) {
                advance();
                observe();
            }
            if (!all_asleep()) throw std::runtime_error(name + ": never fell asleep, nothing to time");
        }
        std::vector<double> ms(timed);
        for (auto &sample : ms) {
            const auto begin = std::chrono::steady_clock::now();
            advance();
            sample = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count() / substeps;
            observe();
        }
        std::ranges::sort(ms);
        uint32_t colors = 0;

        for (uint32_t body = 0; body < world.BodyCount(); ++body) colors = std::max(colors, ColorOf(world.Colors[body]) + 1);

        if (getenv("HISTOGRAM")) {
            std::vector<uint32_t> spread(colors, 0);
            for (uint32_t body = 0; body < world.BodyCount(); ++body) ++spread[ColorOf(world.Colors[body])];
            std::print("{}: ", name);
            for (uint32_t c = 0; c < colors; ++c) std::print("c{}={} ", c, spread[c]);
            std::println("");
        }
        results.push_back({name, world.BodyCount(), ActiveContacts(world), colors, ms.front(), Percentile(ms, 0.5), Percentile(ms, 0.9), ms.back(), quality, refusals, options.ColumnHeight});
    };

    bench("floor", [](World &world) { return BuildStack(world, 1), true; }, {.ColumnHeight = 1});
    bench("stack2", [](World &world) { return BuildStack(world, 2), true; }, {.ColumnHeight = 2});
    bench("stack4", [](World &world) { return BuildStack(world, 4), true; }, {.ColumnHeight = 4});
    bench("stack8", [](World &world) { return BuildStack(world, 8), true; }, {.ColumnHeight = 8});
    bench("stack16", [](World &world) { return BuildStack(world, 16), true; }, {.ColumnHeight = 16});
    bench("stack20", [](World &world) { return BuildStack(world, 20), true; }, {.ColumnHeight = 20});
    bench("raft", [](World &world) { return BuildRaft(world, 5, 3), true; });
    bench("coins", [](World &world) { return !BuildCoins(world, 16, 10, 0.5f).empty(); });
    bench("coins2", [](World &world) { return !BuildCoins(world, 16, 2, 0.5f).empty(); });
    bench("coins4", [](World &world) { return !BuildCoins(world, 16, 4, 0.5f).empty(); });
    bench("coins8", [](World &world) { return !BuildCoins(world, 16, 8, 0.5f).empty(); });
    bench("chain", [](World &world) { return Chain(world, 10), true; });
    bench("chain2", [](World &world) { return Chain(world, 2), true; });
    bench("chain4", [](World &world) { return Chain(world, 4), true; });
    bench("chain8", [](World &world) { return Chain(world, 8), true; });
    bench("chain64", [](World &world) { return Chain(world, 64), true; });

    bench("slab", [](World &world) { return Slab(world, 2); }, {.ColumnHeight = 1, .BoxHalf = {2, 0.25f, 2}});
    bench("slab2m", [](World &world) { return Slab(world, 1); }, {.ColumnHeight = 1, .BoxHalf = {1, 0.25f, 1}});
    bench("slab1m", [](World &world) { return Slab(world, 0.5f); }, {.ColumnHeight = 1, .BoxHalf = {0.5f, 0.25f, 0.5f}});

    bench("slabs25", [](World &world) {
        if (!AddMeshFloor(world, 64)) return false;
        const auto shape = world.AddShape({.HalfExtents = {0.5f, 0.25f, 0.5f}, .Kind = ShapeBox});
        for (uint32_t x = 0; x < 5; ++x)
            for (uint32_t z = 0; z < 5; ++z)
                Place(world, shape, float3{3.0f * float(x) - 6, 0.25f, 3.0f * float(z) - 6});
        return true;
    },
          {.ColumnHeight = 1, .BoxHalf = {0.5f, 0.25f, 0.5f}});
    bench("lattice125", [](World &world) { return Lattice(world, 5, 5, 5), true; }, {.ColumnHeight = 5});
    bench("lattice294", [](World &world) { return Lattice(world, 7, 7, 6), true; }, {.ColumnHeight = 6});

    bench("resting294", [](World &world) { return Lattice(world, 7, 7, 6), true; }, {.Sleeping = true, .ColumnHeight = 6});
    bench("lattice600", [](World &world) { return Lattice(world, 10, 10, 6), true; }, {.ColumnHeight = 6});
    bench("lattice1176", [](World &world) { return Lattice(world, 14, 14, 6), true; }, {.ColumnHeight = 6});

    bench("mixed6000", [](World &world) { Lattice(world, 25, 40, 6); Chain(world, 64, float3{-100, 200, -100}); return true; }, {.Capacity = 6066});
    bench("mixed600", [](World &world) { Lattice(world, 10, 10, 6); Chain(world, 64, float3{-100, 200, -100}); return true; }, {.Capacity = 666});
    bench("lattice6000", [](World &world) { return Lattice(world, 25, 40, 6), true; }, {.Capacity = 6001, .ColumnHeight = 6});
    if (args.size() > 1) {
        bench("lattice96000", [](World &world) { return Lattice(world, 100, 160, 6), true; }, {.Capacity = 96001, .ColumnHeight = 6});
        bench("lattice192000", [](World &world) { return Lattice(world, 200, 160, 6), true; }, {.Capacity = 192001, .ColumnHeight = 6});
    }
    if (results.empty()) throw std::runtime_error("No matching scenes");

    std::println("{:<12} {:>6} {:>5} {:>6} {:>9} {:>9} {:>9} {:>9}", "scene", "bodies", "rows", "colors", "min", "p50", "p90", "max");
    bool passed = true;
    for (const auto &r : results) {
        std::println("{:<12} {:>6} {:>5} {:>6} {:>9.3f} {:>9.3f} {:>9.3f} {:>9.3f}", r.Name, r.Bodies, r.Rows, r.Colors, r.Min, r.Median, r.P90, r.Max);
        std::println("  finite={} refusals={} quaternion_norm_error={:.8f}", r.Quality.Finite, r.Refusals, r.Quality.QuaternionError);
        if (r.ColumnHeight) std::println("  floor_penetration={:.8f} vertical_overlap={:.8f} horizontal_drift={:.8f} m", r.Quality.FloorPenetration, r.Quality.VerticalOverlap, r.Quality.HorizontalDrift);
        passed &= r.Quality.Finite && r.Refusals == 0;
    }
    return passed ? 0 : 2;
} catch (const std::exception &error) {
    std::println(stderr, "RbpBench: {}", error.what());
    return 1;
}
