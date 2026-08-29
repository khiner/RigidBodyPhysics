// The phase 5 bench harness: what a step costs, per scene, in a form two runs can be diffed in.
//
//   RbpBench [scene ...]      no names runs every scene, names run just those
//   STEPS=n WARMUP=n          timed steps per scene and the steps discarded before them
//   ITERATIONS=n COLORS=n     override the solver's iteration count and colour cap, for separating
//                             sweep-count cost from per-sweep cost - a COLORS clamp below what a
//                             scene needs degrades its physics on purpose, so it prices dispatches
//                             rather than answers
//
// Each scene is a fixed configuration stepped with sleeping off, so every timed step does the whole
// solve - a benchmark that lets a scene fall asleep measures the sleep gate and not the solver. Every
// step is timed on its own around Solver::Step, which blocks until the GPU signals, so a step time is
// well defined. The warmup absorbs the settling transient and the first-use costs, and what is
// reported is the distribution over the timed window - min, median, p90, max - because a median is
// what two CI runs can compare and a min is the noise floor of the machine it ran on.
//
// The header names the machine and whether it is on AC power. **A number taken on battery is noise**,
// which this repo has measured the hard way, so the harness says so rather than trusting whoever ran
// it to check. Absolute numbers are comparable only on the same quiet, plugged-in machine. **And the
// first run after the shaders change is not comparable either** - it pays their compilation into the
// system shader cache, and the cost bleeds past the warmup (a 0.5 ms scene read 2.6 ms for a whole
// timed window). Run twice and read the second.
//
// The scenes cover the code paths phase 4 bought correctness with: `floor` is the fixed overhead of a
// near-empty step, `stack20` the box-box chain at two colours, `raft` the pile whose contact runs are
// under budget pressure, `coins` the hull path with `ConvexManifold` and reduction live, `chain` ten
// jointed links swinging, `slab` a four metre body over third-of-a-metre mesh quads - the batched
// gather at about ten batches a step - and the `lattice` series the same packed-columns scene at
// several body counts, which is where the unconditionally-N-squared narrowphase shows its slope.

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

namespace {
// An environment override, or what the harness would have used without one.
uint32_t Env(const char *name, uint32_t fallback) {
    const char *value = getenv(name);
    return value != nullptr ? uint32_t(std::atoi(value)) : fallback;
}

// First line of a command's output, for the machine header.
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

// Ten links ball-jointed to an anchor, released straight out sideways so nothing damps the swing and
// the joint rows stay loaded for the whole run.
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

// A four metre slab on a floor cut into 0.3125 m quads: about 330 triangles under one body, so the
// resumable mesh gather runs its batch loop ten or so times a step. The configuration a fixed gather
// cap dropped through the floor.
bool Slab(World &world, float half_width) {
    if (!AddMeshFloor(world, 64)) return false;
    Place(world, world.AddShape({.HalfExtents = {half_width, 0.25f, half_width}, .Kind = ShapeBox}), float3{0, 0.25f, 0});
    return true;
}

// Columns of stacked boxes in a grid, spaced so contacts are vertical only: the same physics at any
// scale, which makes it the series to read the narrowphase's N^2 slope from.
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

int main(int argc, char **argv) {
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
    // A hung sibling holding a Metal queue cost the heavier scenes 2-3x while the light ones read
    // clean, so the poison is quiet exactly where a spot check would look. Refuse to share the GPU
    // with our own binaries rather than trust the run to notice.
    const std::string siblings = "pgrep -l Rbp | grep -v '^" + std::to_string(getpid()) + " ' | head -1";
    if (const std::string other = FirstLine(siblings.c_str()); !other.empty())
        std::println("!! another Rbp process is running ({}) - kill it or these timings are noise", other);

    StepSettings settings;
    settings.SleepSteps = ~0u;
    // GRAVITY=0 holds every body at its spawn pose, which is what lets an ablated kernel - one with a
    // stage cut out to attribute its cost - be measured against a control doing the same work in the
    // same place, instead of against a body that fell through the contacts the ablation dropped.
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
        // A sleeping scene prices the idle step - what a world full of resting bodies costs every
        // frame for ever - so it warms up until everything is asleep and refuses to time a step that
        // is not, rather than reporting an average of two different regimes.
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
        // The colouring the scene settled on, which with the extra dispatched to grow into is the
        // sweep count a step pays.
        for (uint32_t body = 0; body < world.BodyCount(); ++body) colors = std::max(colors, ColorOf(world.Colors[body]) + 1);
        // How the bodies spread over the colours, which is the first thing to look at when a scene's
        // colour count reads higher than its graph looks like it needs.
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
    // 25 bodies of slab1m's triangle load at once, so occupancy is the only thing that changed. One
    // wide body pays ~13 us a triangle - a single thread, latency-bound on its spilled clip arrays,
    // with nothing else in flight to hide behind - and 25 bodies pay ~1.2 us. The ratio of this line
    // to slab1m's is the standing check on that amortization.
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
    // The same 294 bodies fully asleep: the idle tax. Sleeping skips the solve but not the
    // narrowphase - CollectContacts still collides every pair of frozen poses every step - so this
    // line is what a world full of resting bodies costs every frame for ever, and the number the
    // sleeping-skips-collision work is measured against.
    bench("resting294", [](World &world) { return Lattice(world, 7, 7, 6), true; }, true);
    bench("lattice600", [](World &world) { return Lattice(world, 10, 10, 6), true; });
    bench("lattice1176", [](World &world) { return Lattice(world, 14, 14, 6), true; });

    std::println("{:<12} {:>6} {:>5} {:>6} {:>9} {:>9} {:>9} {:>9}", "scene", "bodies", "rows", "colors", "min", "p50", "p90", "max");
    for (const auto &r : results)
        std::println("{:<12} {:>6} {:>5} {:>6} {:>9.3f} {:>9.3f} {:>9.3f} {:>9.3f}", r.Name, r.Bodies, r.Rows, r.Colors, r.Min, r.Median, r.P90, r.Max);
    return 0;
}
