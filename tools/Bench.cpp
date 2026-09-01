// Times the solver's step, per scene, in a form two runs can be diffed.
//
//   RbpBench [scene ...]      no names runs every scene, names run just those
//   STEPS=n WARMUP=n          timed steps per scene and the steps discarded before them
//   ITERATIONS=n COLORS=n     override the solver's iteration count and colour cap, to separate
//                             sweep-count cost from per-sweep cost
//
// A COLORS clamp below the colours a scene needs degrades its physics deliberately, so it prices dispatches rather than the solve.
// Sleeping is off, so every timed step does the whole solve rather than measuring the sleep gate.
// Steps are timed one at a time around Solver::Step, which blocks until the GPU signals.
// Reported are min, median, p90 and max in ms over the timed window.
// Compare two runs on the median.
// The min is the machine's noise floor.
//
// A run on battery is noise, and the header reports which power source was in use.
// The first run after a shader change pays kernel compilation past the warmup, and a 0.5 ms scene read 2.6 ms for a whole window.
// That compilation lands in the system shader cache, so run twice and read the second.
// Absolute numbers only compare on the same quiet, plugged-in machine.
//
// The scenes: `floor` is the fixed overhead of a near-empty step and `stack20` a box-box chain at two colours.
// `raft` is a pile under contact-budget pressure and `coins` the hull path with ConvexManifold and reduction live.
// `chain` is ten jointed links swinging and `slab` a wide body over mesh quads, which runs the batched gather.
// The `lattice` series measures the slope of the N^2 narrowphase.

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

// Links ball-jointed to an anchor, released straight out.
// Nothing damps the swing, so the joint rows stay loaded for the whole run.
void Chain(World &world, uint32_t links) {
    const auto shape = world.AddShape(UnitBox);
    Index previous = world.AddBody({.Pose = At(float3{-Half - 0.01f, 0, 0}), .Density = 0});
    for (uint32_t i = 0; i < links; ++i) {
        const float x = 1.02f * float(i);
        const auto link = Place(world, shape, float3{x, 0, 0});
        world.AddJoint({.BodyA = previous, .BodyB = link, .At = {x - Half - 0.01f, 0, 0}});
        previous = link;
    }
}

// A slab on a floor cut into 0.3125 m quads.
// At four metres that is about 330 triangles under one body, so the resumable mesh gather runs its batch loop ten or so times a step.
bool Slab(World &world, float half_width) {
    if (!AddMeshFloor(world, 64)) return false;
    Place(world, world.AddShape({.HalfExtents = {half_width, 0.25f, half_width}, .Kind = ShapeBox}), float3{0, 0.25f, 0});
    return true;
}

// Columns of stacked boxes in a grid, spaced so every contact is vertical.
// The physics is then the same at any body count, so the series reads as a slope.
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

    const std::string power = FirstLine("pmset -g batt");
    const bool on_ac = power.contains("AC Power");
    std::println("RbpBench: {}, {}, {} timed steps after {} warmup, sleeping off", FirstLine("sysctl -n machdep.cpu.brand_string"), on_ac ? "AC power" : "ON BATTERY", timed, warmup);
    if (!on_ac) std::println("!! on battery - these timings are noise, plug in and run again");
    // Another Rbp process with a Metal queue open costs the heavy scenes 2-3x.
    // The light scenes read clean either way, so a spot check misses it.
    const std::string siblings = "pgrep -l Rbp | grep -v '^" + std::to_string(getpid()) + " ' | head -1";
    if (const std::string other = FirstLine(siblings.c_str()); !other.empty())
        std::println("!! another Rbp process is running ({}) - kill it or these timings are noise", other);

    StepSettings settings;
    settings.SleepSteps = ~0u;
    // GRAVITY=0 leaves every body at its spawn pose, so an ablated kernel is priced against a control doing the same work in the same place.
    if (getenv("GRAVITY")) settings.Gravity = {0, float(std::atof(getenv("GRAVITY"))), 0};
    settings.Iterations = Env("ITERATIONS", settings.Iterations);
    settings.MaxColors = Env("COLORS", settings.MaxColors);
    if (settings.Iterations != StepSettings{}.Iterations || settings.MaxColors != StepSettings{}.MaxColors)
        std::println("overridden: {} iterations, {} colours cap", settings.Iterations, settings.MaxColors);
    const mtl::Context context;
    std::vector<Result> results;

    const auto bench = [&](const std::string &name, auto build, bool sleeping = false) {
        if (!wanted(name)) return;
        Solver solver{context};
        World world{context};
        if (!build(world)) return (void)std::println(stderr, "{}: scene would not build", name);
        // A sleeping scene prices the idle step, so timing starts only once every body is asleep rather than averaging two regimes.
        StepSettings scene_settings = settings;
        if (sleeping) scene_settings.SleepSteps = StepSettings{}.SleepSteps;
        for (uint32_t step = 0; step < warmup; ++step) solver.Step(world, scene_settings);
        if (sleeping) {
            const auto all_asleep = [&] {
                for (uint32_t body = 0; body < world.BodyCount(); ++body)
                    if (world.Masses[body].InvMass > 0 && world.Quiet[body] < scene_settings.SleepSteps) return false;
                return true;
            };
            uint32_t patience = 3600;
            while (!all_asleep() && patience-- > 0) solver.Step(world, scene_settings);
            if (!all_asleep()) return (void)std::println(stderr, "{}: never fell asleep, nothing to time", name);
        }
        std::vector<double> ms(timed);
        for (auto &sample : ms) {
            const auto begin = std::chrono::steady_clock::now();
            solver.Step(world, scene_settings);
            sample = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
        }
        std::ranges::sort(ms);
        uint32_t colors = 0;
        // The sweep count a step pays includes the extra colour dispatched to grow into.
        for (uint32_t body = 0; body < world.BodyCount(); ++body) colors = std::max(colors, ColorOf(world.Colors[body]) + 1);
        // The first thing to look at when a scene's colour count reads higher than its graph needs.
        if (getenv("HISTOGRAM")) {
            std::vector<uint32_t> spread(colors, 0);
            for (uint32_t body = 0; body < world.BodyCount(); ++body) ++spread[ColorOf(world.Colors[body])];
            std::print("{}: ", name);
            for (uint32_t c = 0; c < colors; ++c) std::print("c{}={} ", c, spread[c]);
            std::println("");
        }
        results.push_back({name, world.BodyCount(), ActiveContacts(world), colors, ms.front(), Percentile(ms, 0.5), Percentile(ms, 0.9), ms.back()});
    };

    bench("floor", [](World &world) { return BuildStack(world, 1), true; });
    bench("stack20", [](World &world) { return BuildStack(world, 20), true; });
    bench("raft", [](World &world) { return BuildRaft(world, 5, 3), true; });
    bench("coins", [](World &world) { return !BuildCoins(world, 16, 10, 0.5f).empty(); });
    bench("chain", [](World &world) { return Chain(world, 10), true; });
    // Three widths, so the cost per triangle reached and any fixed cost under it read off the slope.
    bench("slab", [](World &world) { return Slab(world, 2); });
    bench("slab2m", [](World &world) { return Slab(world, 1); });
    bench("slab1m", [](World &world) { return Slab(world, 0.5f); });
    // 25 bodies at slab1m's triangle load, so occupancy is the only difference.
    // One wide body is a single latency-bound thread at ~13 us a triangle and 25 bodies pay ~1.2 us.
    // The ratio of this line to slab1m's is the standing check on that amortization.
    bench("slabs25", [](World &world) {
        if (!AddMeshFloor(world, 64)) return false;
        const auto shape = world.AddShape({.HalfExtents = {0.5f, 0.25f, 0.5f}, .Kind = ShapeBox});
        for (uint32_t x = 0; x < 5; ++x)
            for (uint32_t z = 0; z < 5; ++z)
                Place(world, shape, float3{3.0f * float(x) - 6, 0.25f, 3.0f * float(z) - 6});
        return true;
    });
    bench("lattice125", [](World &world) { return Lattice(world, 5, 5, 5), true; });
    bench("lattice294", [](World &world) { return Lattice(world, 7, 7, 6), true; });
    // The same 294 bodies fully asleep.
    // Sleeping skips the solve and still runs the narrowphase, so this is the per-frame cost of a world of resting bodies.
    bench("resting294", [](World &world) { return Lattice(world, 7, 7, 6), true; }, true);
    bench("lattice600", [](World &world) { return Lattice(world, 10, 10, 6), true; });
    bench("lattice1176", [](World &world) { return Lattice(world, 14, 14, 6), true; });

    std::println("{:<12} {:>6} {:>5} {:>6} {:>9} {:>9} {:>9} {:>9}", "scene", "bodies", "rows", "colors", "min", "p50", "p90", "max");
    for (const auto &r : results)
        std::println("{:<12} {:>6} {:>5} {:>6} {:>9.3f} {:>9.3f} {:>9.3f} {:>9.3f}", r.Name, r.Bodies, r.Rows, r.Colors, r.Min, r.Median, r.P90, r.Max);
    return 0;
} catch (const std::exception &error) {
    // See mtl::Buffer: a bad index is reported here and the process exits normally.
    std::println(stderr, "RbpBench: {}", error.what());
    return 1;
}
