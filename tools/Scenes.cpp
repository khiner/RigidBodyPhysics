// Standing instrument scenes: each one runs the solver and reports measurements the tests do not make.
//
//   RbpScenes stack [boxes] [steps] [iterations] [beta] [colors]
//   RbpScenes slide [steps] [iterations] [beta] [colors]
//   RbpScenes shear [hold] [boxes] [fraction] [iterations]
//   RbpScenes slam [speed] [density] [steps] [beta]
//   RbpScenes bounce [restitution] [steps] [iterations] [beta]
//   RbpScenes seams [cells] [steps] [speed]
//   RbpScenes raft [side] [layers] [steps] [iterations]
//   RbpScenes coins [sides] [coins] [steps] [twist] [iterations]
//   RbpScenes bullet [thickness] [graze]
//   RbpScenes order [boxes] [steps] [iterations]
//   RbpScenes motion [steps] [damping]
//   RbpScenes teleport [boxes] [settle] [after]
//   RbpScenes platform [steps] [speed]
//   RbpScenes wheel [steps] [balls] [paddles]
//   RbpScenes perch [steps] [iterations]
//   RbpScenes rocker [steps] [iterations] [degrees] [shove]
//   RbpScenes frame [steps] [iterations] [degrees] [release]
//   RbpScenes slider [steps] [iterations]
//   RbpScenes brake [steps] [iterations] [damping]
//   RbpScenes offset [steps] [offset] [ledge]
//   RbpScenes pinned [steps]
//   RbpScenes compound [steps] [speed]
//   RbpScenes join [steps] [speed]
//
// NOSLEEP=1 stops bodies sleeping and EVERY=1 reports every step rather than every sixtieth.
// Sleeping and coarse sampling together hide a ringing stack: it falls asleep mid-sway, and sleeping then sets its velocity to zero.
//
// What each scene measures.
// The longer notes are at each scene's own function.
//
//   stack     each box's sag below where it analytically belongs, the churn that produced it, and
//             the mean normal penalty against m/h^2 - far below that ratio a stack rings
//   raft      a pile, so a middle box has nine partners: the points demanded of each contact run
//             against the slots it has, and the churn split by the cost of each identity
//   coins     twisted prisms, the only faces in the suite whose manifolds are reduced
//   slide     the net contact force against the motion it produced, and the sink while producing it
//   shear     a settled stack loaded sideways inside the friction cone, swept by how long it stood
//   slam      how far up the penalty ramp a hard impact gets against m/h^2. Contacts are quietest at
//             1 and usable over about [0.3, 3], and a stack held above 10 comes apart
//   bounce    every bounce of a dropped box against the restitution asked for, and the step where
//             the threshold takes over and the box settles
//   seams     a ball rolled across a triangle floor and across a plane at once. The mesh side must
//             show one contact wherever the ball is, a constant height, and the same arrival
//   bullet    a small body fired at a thin box and at a mesh wall across the speed range, which must
//             stop short of both, and one shot grazing the corner for the ghost collision
//   order     one stack built bottom-up and top-down. A pair belongs to the lower-indexed of its
//             bodies and that fixes the reference face, so the divergence is the cost of that choice
//   motion    per-body gravity scale and damping against their closed forms, falling and at rest
//   offset    a shape's pose within the body frame: where the weight is against where the geometry is
//   teleport  the effect of a host pose write on a sleeping world, with World::Wake and without
//   platform  a kinematic body carrying, striking, and waking what rests on it
//   wheel     a dynamic one-sided mesh: a paddle wheel on a hinge, and what a moving mesh rests on
//   perch     a body at rest on a mesh the host gave a mass, against the same mesh as scenery
//   rocker    a crowned plate set down level, which must settle on its apex facet and stay there
//   frame     a hinge on a joint frame that is an axis of neither body, and the unwrapped twist
//   slider    the five linear axis modes, in metres and newtons
//   brake     the damping coefficient beside each stiffness, against its closed form
//   pinned    a mass of zero read as infinite rather than absent
//   compound  a body of several pieces: the same solid split, a table, a sibling seam, a full run
//   join      the same seam between two separate bodies, swept by how far out of true it is

#include "Scenery.h"
#include "Solver.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <format>
#include <map>
#include <optional>
#include <print>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace rbp;

namespace {
uint32_t Arg(std::span<char *> args, size_t at, uint32_t fallback) {
    return args.size() > at ? uint32_t(std::atoi(args[at])) : fallback;
}
float ArgF(std::span<char *> args, size_t at, float fallback) {
    return args.size() > at ? float(std::atof(args[at])) : fallback;
}

// Every active contact, in slot order.
// That order is fixed, so the sums below compare between runs.
void EachActive(const World &world, const auto &each) {
    for (uint32_t slot = 0; slot < world.Contacts.Capacity; ++slot)
        if (const Contact &contact = world.Contacts[slot]; contact.Active) each(contact);
}

// Every contact force, in world space.
// Normal and tangential are kept apart because their ratio against the cone shows whether Coulomb friction is saturated.
struct ContactTotals {
    float3 Normal{0, 0, 0}, Tangential{0, 0, 0};
    float Cone{};
    uint32_t Count{};
};

ContactTotals Totals(const World &world) {
    ContactTotals totals;
    EachActive(world, [&totals](const Contact &contact) {
        ++totals.Count;
        // The same basis the solve applied these forces in, so the recovered force equals the applied one.
        const ContactBasis basis = MakeContactBasis(contact.Normal);
        totals.Normal += contact.Lambda[0] * basis.Axis[0];
        totals.Tangential += contact.Lambda[1] * basis.Axis[1] + contact.Lambda[2] * basis.Axis[2];
        totals.Cone += std::abs(contact.Lambda[0]) * contact.Friction;
    });
    return totals;
}

// Every active contact, keyed the way warm starting keys them.
// A contact that keeps its identity keeps its dual across the step, and a renamed one restarts from zero.
std::set<std::tuple<Index, Index, uint32_t>> ContactKeys(const World &world) {
    std::set<std::tuple<Index, Index, uint32_t>> keys;
    EachActive(world, [&keys](const Contact &contact) { keys.emplace(contact.BodyA, contact.BodyB, contact.Feature); });
    return keys;
}

// Contacts gained and lost since the last Take, which is zero for a stack genuinely at rest.
// Step it every step, not only the reported ones.
struct Churner {
    std::set<std::tuple<Index, Index, uint32_t>> Keys;
    uint32_t Since{};

    explicit Churner(const World &world) : Keys(ContactKeys(world)) {}

    void Step(const World &world) {
        auto now = ContactKeys(world);
        for (const auto &key : now) Since += Keys.contains(key) ? 0 : 1;
        for (const auto &key : Keys) Since += now.contains(key) ? 0 : 1;
        Keys = std::move(now);
    }
    uint32_t Take() { return std::exchange(Since, 0); }
};

void ApplyNoSleep(StepSettings &settings) {
    if (getenv("NOSLEEP")) settings.SleepSteps = ~0u;
}

// Which steps a scene reports on: one in `Period` and the last, or every one under EVERY=1.
struct Reporting {
    uint32_t Period, Steps;
    bool Every = getenv("EVERY") != nullptr;

    bool operator()(uint32_t step) const { return Every || step % Period == Period - 1 || step == Steps - 1; }
};

void Stack(std::span<char *> args) {
    const uint32_t boxes = Arg(args, 2, 3), steps = Arg(args, 3, 240);
    StepSettings settings{.Iterations = Arg(args, 4, 10), .Beta = ArgF(args, 5, StepSettings{}.Beta), .MaxColors = Arg(args, 6, StepSettings{}.MaxColors)};
    ApplyNoSleep(settings);
    const Reporting report{60, steps};
    const mtl::Context context;
    Solver solver{context};
    World world{context};
    const std::vector<Index> stack = BuildStack(world, boxes);

    std::println("stack of {}, {} iterations, beta {:g}, {} colors", boxes, settings.Iterations, settings.Beta, settings.MaxColors);
    Churner churner{world};
    for (uint32_t step = 0; step < steps; ++step) {
        solver.Step(world, settings);
        churner.Step(world);
        if (!report(step)) continue;
        // Mean normal penalty as a fraction of one box's inertial stiffness.
        // Below about one, the stack's rocking mode falls under the rate the integrator damps at.
        double penalty = 0;
        uint32_t active = 0;
        EachActive(world, [&](const Contact &contact) {
            penalty += contact.Penalty[0];
            ++active;
        });
        const float inertial = 1 / (world.Masses[stack[0]].InvMass * settings.DeltaTime * settings.DeltaTime);
        std::print("{:4} churn={:4} asleep={} k/(m/h2)={:6.3f}", step, churner.Take(), Asleep(world, stack, settings), active > 0 ? penalty / active / inertial : 0);
        for (uint32_t i = 0; i < boxes; ++i) {
            // Where the box belongs: resting on the one below, one contact margin into it.
            const float ideal = Half + float(i) * (1 - settings.ContactMargin) - settings.ContactMargin;
            const float height = world.Poses[stack[i]].Position.y;
            std::print("  sag={:7.4f} |v|={:7.5f}", ideal - height, simd::length(world.Velocities[stack[i]].Linear));
        }
        std::println("");
    }
}

// The same churn, split by the cost of each identity.
// A speculative row carries no dual, so gaining or losing its name costs nothing, and renaming a pushing row discards a converged force.
// `engaged` counts only identities that were pushing.
// `renamed` counts pairings where the same geometry reappears under another name, so each one accounts for two of `churn`.
struct PileChurn {
    // The whole identity CollectContacts matches a warm start on: feature, triangle, and pair of leaves.
    // No part of it is a slot index.
    using Name = std::tuple<Index, Index, uint32_t, Index, uint32_t>;
    struct Row {
        float3 At; // the point in world space, used to match two names at one place
        float Push; // the normal dual, negated, so a loaded row reads positive
    };
    struct Counts {
        uint32_t All, Engaged, Renamed, Flicker;
        float Load, Spread; // the pile's total normal force now, and how far it moved over the interval
    };

    // How near two points must be to count as one piece of geometry named twice.
    static constexpr float Near = 1e-3f;

    std::map<Name, Row> Rows;
    uint32_t All{}, Engaged{}, Renamed{};
    float Load{}, Least{INFINITY}, Most{};

    explicit PileChurn(const World &world) : Rows(Read(world)) {}

    static std::map<Name, Row> Read(const World &world) {
        std::map<Name, Row> rows;
        EachActive(world, [&](const Contact &contact) {
            const Pose pose = world.Poses[contact.BodyA];
            rows.emplace(Name{contact.BodyA, contact.BodyB, contact.Feature, contact.SubShape, contact.Children},
                         Row{WorldPoint(pose, contact.AnchorA), -contact.Lambda[0]});
        });
        return rows;
    }

    void Step(const World &world) {
        auto now = Read(world);
        std::vector<std::pair<Name, Row>> gone, came;
        for (const auto &[name, row] : Rows)
            if (!now.contains(name)) gone.emplace_back(name, row);
        for (const auto &[name, row] : now)
            if (!Rows.contains(name)) came.emplace_back(name, row);
        All += uint32_t(gone.size() + came.size());
        // A pushing dual is lost both when its name stops being written and when a fresh name replaces it.
        for (const auto &[name, row] : gone) Engaged += row.Push > 0 ? 1 : 0;
        for (const auto &[name, row] : came) Engaged += row.Push > 0 ? 1 : 0;
        // Greedy from the lowest name outwards, so the pairing is fixed rather than incidental.
        std::vector<bool> paired(came.size(), false);
        for (const auto &[name, row] : gone) {
            for (size_t i = 0; i < came.size(); ++i) {
                const auto &[other, at] = came[i];
                if (paired[i] || std::get<0>(other) != std::get<0>(name) || std::get<1>(other) != std::get<1>(name)) continue;
                if (simd::length(at.At - row.At) > Near) continue;
                paired[i] = true;
                ++Renamed;
                break;
            }
        }
        // A load that moves as names come and go is a force being rebuilt.
        Load = 0;
        for (const auto &[name, row] : now) Load += row.Push;
        Least = std::min(Least, Load);
        Most = std::max(Most, Load);
        Rows = std::move(now);
    }

    Counts Take() {
        const Counts counts{All, Engaged, Renamed, All - 2 * Renamed, Load, Most - Least};
        All = Engaged = Renamed = 0;
        Least = INFINITY;
        Most = 0;
        return counts;
    }
};

// A pile rather than a chain, so a middle body needs far more points than one contact run has slots for.
// The refused points are the measurement, and an underfed run shows as creep.
void Raft(std::span<char *> args) {
    const uint32_t side = Arg(args, 2, 4), layers = Arg(args, 3, 2), steps = Arg(args, 4, 300);
    StepSettings settings{.Iterations = Arg(args, 5, StepSettings{}.Iterations)};
    ApplyNoSleep(settings);
    const Reporting report{60, steps};
    const mtl::Context context;
    Solver solver{context};
    World world{context};
    const std::vector<Index> boxes = BuildRaft(world, side, layers);

    // The points demanded of a body's run.
    // Every partner is collided whether or not the run is full, and each refused point is counted once, so this is exact rather than a lower bound.
    const auto wanted = [&world](Index body) { return ActiveContacts(world, body) + world.ContactRefusals[body]; };

    std::println("a raft {} across and {} layers deep, {} boxes, {} slots a body", side, layers, boxes.size(), ContactsPerBody);
    PileChurn churner{world};
    uint32_t busiest = 0, demand = 0;
    for (uint32_t step = 0; step < steps; ++step) {
        solver.Step(world, settings);
        churner.Step(world);
        for (const auto body : boxes)
            if (wanted(body) > demand) demand = wanted(busiest = body);
        if (!report(step)) continue;
        uint32_t refused = 0;
        float fastest = 0;
        for (const auto body : boxes) {
            refused += world.ContactRefusals[body];
            fastest = std::max(fastest, simd::length(world.Velocities[body].Linear));
        }
        const auto churn = churner.Take();
        std::println("{:4} churn={:4} engaged={:4} renamed={:4} flicker={:4} asleep={:3}/{} refused={:4} |v|max={:8.5f} top y={:7.4f} normal={:9.1f} spread={:8.2f}",
                     step, churn.All, churn.Engaged, churn.Renamed, churn.Flicker, Asleep(world, boxes, settings), boxes.size(), refused, fastest,
                     float(world.Poses[boxes.back()].Position.y), churn.Load, churn.Spread);
    }
    std::println("the busiest run was body {}'s, asked for {} points against {} slots", busiest, demand, ContactsPerBody);
}

// The only scene with faces of more than four sides: box on box clips to exactly four points, so `ReduceManifold` runs nowhere else.
// Two eight-gons crossed at half a step intersect in a sixteen-gon, so the four points kept here are selected rather than incidental.
void Coins(std::span<char *> args) {
    const uint32_t sides = Arg(args, 2, 8), coins = Arg(args, 3, 4), steps = Arg(args, 4, 400);
    const float twist = ArgF(args, 5, 0.5f); // in steps of the prism's own face, so 0.5 is the worst case
    StepSettings settings{.Iterations = Arg(args, 6, StepSettings{}.Iterations)};
    ApplyNoSleep(settings);
    const Reporting report{100, steps};
    const mtl::Context context;
    Solver solver{context};
    World world{context};
    const std::vector<Index> stack = BuildCoins(world, sides, coins, twist);
    if (stack.empty()) return (void)std::println(stderr, "the prism made no hull");

    std::println("{} coins of {} sides, each twisted {:g} of a face against the one below, {} manifold points a pair", coins, sides, twist, ManifoldPoints);
    Churner churner{world};
    for (uint32_t step = 0; step < steps; ++step) {
        solver.Step(world, settings);
        churner.Step(world);
        if (!report(step)) continue;
        uint32_t held = 0;
        float worst_tilt = 0;
        for (uint32_t i = 0; i < coins; ++i) {
            const Index body = stack[i];
            // Turn away from the twist it was set down with, the first symptom of a too-narrow manifold.
            const float4 was = QuatFromRotationVector(float3{0, CoinTwist(sides, twist, i), 0});
            worst_tilt = std::max(worst_tilt, simd::length(RotationVector(QuatMul(world.Poses[body].Orientation, QuatConjugate(was)))));
            held += ActiveContacts(world, body);
        }
        std::print("{:4} churn={:4} asleep={}/{} rows={:3} turned={:8.5f}", step, churner.Take(), Asleep(world, stack, settings), coins, held, worst_tilt);
        for (uint32_t i = 0; i < coins; ++i) {
            const float ideal = CoinHalfHeight + float(i) * (2 * CoinHalfHeight - settings.ContactMargin) - settings.ContactMargin;
            std::print("  sag={:7.4f}", ideal - float(world.Poses[stack[i]].Position.y));
        }
        std::println("");
    }
}

void Slide(std::span<char *> args) {
    const uint32_t steps = Arg(args, 2, 120);
    const Reporting report{60, steps};
    const StepSettings settings{.Iterations = Arg(args, 3, 10), .Beta = ArgF(args, 4, StepSettings{}.Beta), .MaxColors = Arg(args, 5, StepSettings{}.MaxColors)};
    const mtl::Context context;
    Solver solver{context};
    World world{context};
    const auto shape = world.AddShape(UnitBox);
    AddGround(world);
    const auto box = Place(world, shape, float3{0, Half, 0});

    for (uint32_t settle = 0; settle < 30; ++settle) solver.Step(world, settings);
    constexpr float Speed = 2;
    world.Velocities[box].Linear = {Speed, 0, 0};

    const float gravity = std::abs(settings.Gravity.y), mass = 1 / world.Masses[box].InvMass;
    std::println("slide at {} m/s, {} iterations, beta {:g}, {} colors. Coulomb says it decelerates at mu g = {:.3f} m/s^2", Speed, settings.Iterations, settings.Beta, settings.MaxColors, Friction * gravity);
    for (uint32_t step = 0; step < steps; ++step) {
        const float was = world.Velocities[box].Linear.x;
        solver.Step(world, settings);
        if (!report(step)) continue;
        const auto totals = Totals(world);
        const float speed = world.Velocities[box].Linear.x;
        const float implied = -totals.Tangential.x / mass, measured = (speed - was) / settings.DeltaTime;
        std::println("{:4} vx={:7.4f} sink={:7.4f} contacts={} normal={:8.1f} tangential={:8.1f} of cone {:8.1f} | "
                     "force implies {:7.3f} m/s^2, motion shows {:7.3f}",
                     step, speed, Half - float(world.Poses[box].Position.y), totals.Count, simd::length(totals.Normal), simd::length(totals.Tangential), totals.Cone, implied, measured);
    }
}
// The friction rows of one pair in a stack, which either carry a steady shear load or let it through.
// Penalty and dual are reported apart: the dual is the force the row converged on, and the penalty is the stiffness the primal meets new motion with.
// A settled row's penalty decays by Gamma towards PenaltyMin, and the normal row's floor is deliberately not shared with friction.
struct ShearRows {
    uint32_t Count{}, Stuck{};
    float LowPenalty = INFINITY, HighPenalty = 0;
    float3 Tangential{0, 0, 0};
    float Cone{};
};

ShearRows LoadedRows(const World &world, Index upper, Index lower) {
    ShearRows rows;
    EachActive(world, [&](const Contact &contact) {
        if (!((contact.BodyA == upper && contact.BodyB == lower) || (contact.BodyA == lower && contact.BodyB == upper))) return;
        ++rows.Count;
        rows.Stuck += contact.Stick ? 1 : 0;
        for (uint32_t r = 1; r < 3; ++r) {
            rows.LowPenalty = std::min(rows.LowPenalty, contact.Penalty[r]);
            rows.HighPenalty = std::max(rows.HighPenalty, contact.Penalty[r]);
        }
        const ContactBasis basis = MakeContactBasis(contact.Normal);
        rows.Tangential += contact.Lambda[1] * basis.Axis[1] + contact.Lambda[2] * basis.Axis[2];
        rows.Cone += std::abs(contact.Lambda[0]) * contact.Friction;
    });
    return rows;
}

// A settled stack pushed sideways, well inside the friction cone, for long enough to show whether it holds.
// A settled contact's dual absorbs the load, C goes to zero, and Eq. 19's Gamma decay takes the penalty down unopposed.
// The normal row is floored at the pair's inertial stiffness for that reason.
// Friction is deliberately not floored, and by measurement must not be.
// A floor there locks the stick-slip transition early, and a sliding box then stops a third short of v^2/(2 mu g).
// Shear is therefore carried by the dual plus almost no stiffness.
//
// The load is applied as the velocity a constant force adds over one step, at the centre of mass, so it is pure shear with no torque.
// It is half the static cone, so Coulomb friction predicts no motion at all.
// The sweep varies how long the stack stood first.
// A bounded offset that is the same at every wait is the elastic response of a stiff contact, and a displacement that grows with the wait is the shear spring.
void Shear(std::span<char *> args) {
    const uint32_t hold = Arg(args, 2, 600), boxes = std::max(2u, Arg(args, 3, 4)); // two boxes is the least that has a loaded pair
    const float fraction = ArgF(args, 4, 0.5f);
    StepSettings settings{.Iterations = Arg(args, 5, StepSettings{}.Iterations)};
    // Sleep is off here rather than NOSLEEP-gated, since a sleeping stack's penalties stop decaying and the decay running is the premise.
    settings.SleepSteps = ~0u;
    const Reporting report{100, hold};
    const mtl::Context context;
    Solver solver{context};

    const float gravity = std::abs(settings.Gravity.y);
    std::println("a stack of {} settled for N steps, then a steady lateral force on the top box for {}, at {:g} of "
                 "the static cone mu N. Sleep is off throughout, so the penalties go on decaying.\n"
                 "  Coulomb says the box does not move. A bounded offset the same at every N is the contact's "
                 "elastic answer, a creep that grows with N is the friction rows decayed to a soft spring",
                 boxes, hold, fraction);

    struct Outcome {
        uint32_t Settled{};
        float Drift{}, Creep{}, Penalty{};
    };
    std::vector<Outcome> outcomes;
    for (const uint32_t settled : {60u, 600u, 1800u}) {
        World world{context};
        const auto stack = BuildStack(world, boxes);
        const Index top = stack.back(), below = stack[boxes - 2];
        for (uint32_t step = 0; step < settled; ++step) solver.Step(world, settings);

        const float mass = 1 / world.Masses[top].InvMass;
        // The top box's contact carries only its own weight, so this is exactly the cone fraction asked for.
        const float force = fraction * Friction * mass * gravity;
        const float from = float(world.Poses[top].Position.x), floor_from = float(world.Poses[stack.front()].Position.x);
        const ShearRows before = LoadedRows(world, top, below);
        std::println("\nsettled {} steps, then {:.4f} N on {:.3f} kg for {} steps. At the moment the load arrives the "
                     "loaded pair holds {} rows, friction penalty {:.4g}..{:.4g} N/m, {} of them stuck",
                     settled, force, mass, hold, before.Count, before.LowPenalty, before.HighPenalty, before.Stuck);

        float window = 0; // the drift sixty steps from the end, the start of the interval the creep is read over
        for (uint32_t step = 0; step < hold; ++step) {
            world.Velocities[top].Linear.x += force * world.Masses[top].InvMass * settings.DeltaTime;
            solver.Step(world, settings);
            const float drift = float(world.Poses[top].Position.x) - from;
            if (step + 61 == hold) window = drift;
            if (!report(step)) continue;
            const ShearRows rows = LoadedRows(world, top, below);
            std::println("{:5} drift {:10.3e} m  vx {:10.3e} m/s  |  rows {} stuck {}  friction penalty {:9.4g}..{:9.4g} N/m  "
                         "shear dual {:8.4f} N of cone {:8.4f}  |  stack base moved {:10.3e} m",
                         step, drift, float(world.Velocities[top].Linear.x), rows.Count, rows.Stuck, rows.LowPenalty, rows.HighPenalty,
                         simd::length(rows.Tangential), rows.Cone, float(world.Poses[stack.front()].Position.x) - floor_from);
        }
        const float drift = float(world.Poses[top].Position.x) - from;
        const ShearRows after = LoadedRows(world, top, below);
        outcomes.push_back({settled, drift, (drift - window) / (60 * settings.DeltaTime), after.HighPenalty});
    }

    std::println("");
    for (const auto &outcome : outcomes)
        std::println("settled {:5} steps: drift {:10.3e} m, creep over the last second {:10.3e} m/s, friction penalty {:9.4g} N/m",
                     outcome.Settled, outcome.Drift, outcome.Creep, outcome.Penalty);
}

// Per-body gravity scale and damping, in two worlds.
// Free flight measures the integrator and the plane measures the warm start, over the same five factors.
void Motion(std::span<char *> args) {
    const uint32_t steps = Arg(args, 2, 120);
    const float damping = ArgF(args, 3, 1.5f);
    constexpr float Scales[]{0, 0.5f, 1, 2, -1}; // the MotionProperties sample's scales
    constexpr float Speed = 4, Spin = 5; // the thrown box's speed and spin, both well under the angular cap
    const StepSettings settings{};
    const Reporting report{30, steps};
    const float h = settings.DeltaTime, gravity = settings.Gravity.y;
    const mtl::Context context;
    Solver solver{context};

    World falling{context};
    const auto shape = falling.AddShape(UnitBox);
    std::vector<Index> boxes;
    for (const float scale : Scales) // in free space, far enough apart that no pair ever collides
        boxes.push_back(falling.AddBody({.Pose = At(float3{4 * float(boxes.size()), 0, 0}), .Shape = shape, .Density = Density, .GravityScale = scale}));
    // No gravity, so damping is the only force on it.
    // Set well clear of the column above.
    const Index thrown = falling.AddBody({.Pose = At(float3{0, 0, 50}), .Velocity = {.Linear = {Speed, 0, 0}, .Angular = {0, Spin, 0}}, .Shape = shape, .Density = Density, .GravityScale = 0, .LinearDamping = damping, .AngularDamping = damping});

    std::println("free fall at gravity scales 0, 0.5, 1, 2 and -1, beside a box thrown at {:g} m/s and spun at {:g} rad/s, damped {:g} a second", Speed, Spin, damping);
    std::println("  err is how far each body is off scale h^2 g n (n + 1) / 2, and how far the thrown box is off v (1 - c h)^n");
    for (uint32_t step = 0; step < steps; ++step) {
        solver.Step(falling, settings);
        if (!report(step)) continue;
        const float n = float(step + 1);
        const float fell = h * h * gravity * n * (n + 1) / 2, left = std::pow(1 - damping * h, n);
        std::print("{:4}", step);
        for (uint32_t i = 0; i < std::size(Scales); ++i) {
            const float y = falling.Poses[boxes[i]].Position.y;
            std::print("  s={:4.1f} y={:9.4f} err={:9.2e}", Scales[i], y, y - Scales[i] * fell);
        }
        const auto motion = falling.Velocities[thrown];
        std::println("  |  damped v={:7.4f} err={:9.2e} spin={:7.4f} err={:9.2e}", motion.Linear.x, motion.Linear.x - Speed * left, motion.Angular.y, motion.Angular.y - Spin * left);
    }

    World resting{context};
    const auto plate = resting.AddShape(UnitBox);
    AddGround(resting);
    std::vector<Index> standing;
    for (const float scale : Scales) // set down exactly on the plane, so each body starts at rest
        standing.push_back(resting.AddBody({.Pose = At(float3{4 * float(standing.size()), Half, 0}), .Shape = plate, .Density = Density, .Friction = Friction, .GravityScale = scale}));
    for (uint32_t step = 0; step < steps; ++step) solver.Step(resting, settings);

    const float weight = std::abs(gravity) / resting.Masses[standing[2]].InvMass;
    std::println("and the same scales set down on the plane for {} steps. m g is {:.1f} N, so the scale is the multiple of it each body should be held by", steps, weight);
    for (uint32_t i = 0; i < std::size(Scales); ++i) {
        // Both sides of the pair: a manifold belongs to the lower-indexed body, so the plane owns all of these rather than each body's own run.
        float held = 0;
        uint32_t rows = 0;
        EachActive(resting, [&](const Contact &contact) {
            if (contact.BodyA != standing[i] && contact.BodyB != standing[i]) return;
            held += std::abs(contact.Lambda[0]);
            ++rows;
        });
        std::println("  s={:4.1f} y={:8.5f} rows={} normal={:9.1f} N against {:9.1f} N, ratio {:6.3f}", Scales[i], float(resting.Poses[standing[i]].Position.y), rows, held, Scales[i] * weight, Scales[i] != 0 ? held / (Scales[i] * weight) : held);
    }
}

// A shape's pose within the body frame, reported as where the weight is against where the geometry is.
// The body frame is unchanged: the centre of mass, with the inertia diagonal in it.
void Offset(std::span<char *> args) {
    const uint32_t steps = Arg(args, 2, 400);
    const float offset = ArgF(args, 3, 0.3f), ledge = ArgF(args, 4, 0.2f);
    const StepSettings settings{};
    const Reporting report{60, steps};
    const mtl::Context context;
    Solver solver{context};

    // The MotionProperties sample's off-centre body: two boxes on a ledge narrower than they are.
    // Both present the same face, and only the place the weight hangs differs.
    constexpr float Top = 0.5f, Mass = 1000, Centred = Mass / 12 * (1 + 1);
    World balancing{context};
    AddGround(balancing);
    balancing.AddBody({.Pose = At(float3{0, Top / 2, 0}), .Shape = balancing.AddShape({.HalfExtents = {ledge, Top / 2, 4}, .Kind = ShapeBox}), .Density = 0, .Friction = Friction});
    std::vector<Index> boxes;
    for (const float local : {0.f, offset}) {
        const Index shape = balancing.AddShape({.HalfExtents = {Half, Half, Half}, .Kind = ShapeBox, .Local = {{local, 0, 0}, {0, 0, 0, 1}}});
        // The parallel axis carry onto the body's origin, which the host computes.
        // The offset is along x, so the tensor stays diagonal.
        const float carry = Mass * local * local;
        // Placed so the *box* is centred over the ledge in both, two metres apart along it.
        const Index body = balancing.AddBody({.Pose = At(float3{-local, Top + Half, 2 * float(boxes.size()) - 1}), .Shape = shape, .Density = Density, .Mass = AuthoredMass{.Mass = Mass, .Inertia = {Centred, Centred + carry, Centred + carry}}, .Friction = Friction});
        if (body == NoIndex) return (void)std::println("refused a body: {} offsets with no authored mass", balancing.OffsetsWithoutMass);
        boxes.push_back(body);
    }

    // The weeble: a sphere whose centre is offset along the body's y from the centre of mass, laid on its side.
    // It rolls to the one orientation that puts the mass at the bottom.
    // Damped, since nothing here models rolling resistance and an undamped ball rocks for ever.
    constexpr float Radius = 0.5f, Rise = 0.25f, BallMass = 500, Inertia = 2.f / 5 * BallMass * Radius * Radius;
    World rocking{context};
    AddGround(rocking);
    const Index weeble = rocking.AddBody({.Pose = At(float3{0, Radius, 0}, QuatFromRotationVector(float3{0, 0, std::numbers::pi_v<float> / 2})), .Shape = rocking.AddShape({.Radius = Radius, .Kind = ShapeSphere, .Local = {{0, Rise, 0}, {0, 0, 0, 1}}}), .Mass = AuthoredMass{.Mass = BallMass, .Inertia = {Inertia, Inertia, Inertia}}, .Friction = Friction, .LinearDamping = 1.2f, .AngularDamping = 1.2f});
    if (weeble == NoIndex) return (void)std::println("refused the weeble: {} offsets with no authored mass", rocking.OffsetsWithoutMass);

    std::println("a box centred over a ledge {:g} m half-wide beside one whose weight hangs {:g} m off the side of it, and a ball whose mass sits {:g} m below its centre", ledge, offset, Rise);
    std::println("  tilt is how far each box has turned, in radians. The centred one stands; the offset one goes over, which is the whole of what an offset means");
    for (uint32_t step = 0; step < steps; ++step) {
        solver.Step(balancing, settings);
        solver.Step(rocking, settings);
        if (!report(step)) continue;
        std::print("{:4}", step);
        for (const Index body : boxes) {
            const Pose pose = balancing.Poses[body];
            std::print("  y={:7.4f} x={:8.4f} tilt={:6.3f} contacts={}", float(pose.Position.y), float(pose.Position.x), simd::length(RotationVector(pose.Orientation)), ActiveContacts(balancing, body));
        }
        // Where the sphere's own centre sits relative to the centre of mass: straight up once settled.
        const float3 centre = Rotate(rocking.Poses[weeble].Orientation, float3{0, Rise, 0});
        std::println("  |  weeble y={:7.4f} centre above mass={:7.4f} of {:g} |v|={:7.5f}", float(rocking.Poses[weeble].Position.y), float(centre.y), Rise, simd::length(rocking.Velocities[weeble].Linear));
    }

    // A hull given a local pose, against the same points handed in already at that pose.
    // Both need an authored mass, since neither body frame is one the cook would produce.
    const Pose local = At(float3{0.35f, -0.2f, 0.15f}, QuatFromRotationVector(float3{0.4f, 0.9f, -0.25f}));
    const std::vector<float3> wedge = WedgePoints();
    std::vector<float3> moved;
    for (const float3 point : wedge) moved.push_back(WorldPoint(local, point));
    constexpr AuthoredMass Weight{.Mass = 30, .Inertia = {4, 5, 6}};
    World placed{context}, baked{context};
    AddGround(placed);
    AddGround(baked);
    const Index placed_shape = placed.AddHull(wedge, nullptr, local), baked_shape = baked.AddHull(moved, nullptr, IdentityPose);
    if (placed_shape == NoIndex || baked_shape == NoIndex) return (void)std::println("the wedge would not cook");
    // Turned to undo the shape's own rotation, so the wedge lands squarely on its base rather than toppling onto it.
    // Set down just clear of the plane.
    const float4 turn = QuatConjugate(placed.Shapes[placed_shape].Local.Orientation);
    float lowest = INFINITY;
    for (uint32_t i = 0; i < placed.Shapes[placed_shape].VertexCount; ++i) {
        const Pose shape_local = placed.Shapes[placed_shape].Local;
        const float3 corner = WorldPoint(shape_local, placed.ShapeVertices[placed.Shapes[placed_shape].FirstVertex + i]);
        lowest = std::min(lowest, float(Rotate(turn, corner).y));
    }
    const Pose start = At(float3{0.2f, 5e-4f - lowest, -0.1f}, turn);
    const Index one = placed.AddBody({.Pose = start, .Shape = placed_shape, .Mass = Weight, .Friction = Friction});
    const Index other = baked.AddBody({.Pose = start, .Shape = baked_shape, .Mass = Weight, .Friction = Friction});
    if (one == NoIndex || other == NoIndex) return (void)std::println("refused the wedge: {} offsets with no authored mass", placed.OffsetsWithoutMass + baked.OffsetsWithoutMass);

    std::println("and one wedge described twice - placed by its local pose, and handed in already there. One body, so one answer");
    float worst = 0;
    for (uint32_t step = 0; step < steps; ++step) {
        solver.Step(placed, settings);
        solver.Step(baked, settings);
        const Pose mine = placed.Poses[one], theirs = baked.Poses[other];
        const float apart = std::max(float(simd::distance(mine.Position, theirs.Position)), simd::length(RotationVector(QuatMul(mine.Orientation, QuatConjugate(theirs.Orientation)))));
        worst = std::max(worst, apart);
        if (!report(step)) continue;
        std::println("{:4} apart {:9.2e}  |  placed y={:8.5f} contacts={}  baked y={:8.5f} contacts={}", step, apart, float(mine.Position.y), ActiveContacts(placed, one), float(theirs.Position.y), ActiveContacts(baked, other));
    }
    std::println("worst the two descriptions ever got apart: {:.3e} m", worst);
}

// The floor `compound` and `join` both slide a box across: two static boxes meeting at x = 0, or the same floor as one box for the control.
// The slider starts with its leading face just short of the join.
// At 2 m/s mu g stops a box in 0.41 m, and that distance has to be spent astride the join rather than reaching it.
constexpr float FloorHalf = 2.5f, FloorTop = 0.25f, SlideRest = FloorTop + Half, SlideFrom = -Half - 0.05f;
constexpr Shape FloorBox{.HalfExtents = {FloorHalf, FloorTop, FloorHalf}, .Kind = ShapeBox};
constexpr Shape SeamlessFloor{.HalfExtents = {2 * FloorHalf, FloorTop, FloorHalf}, .Kind = ShapeBox};

Index AddSlider(World &world, float speed) {
    return world.AddBody({.Pose = At(float3{SlideFrom, SlideRest, 0}), .Velocity = {.Linear = {speed, 0, 0}}, .Shape = world.AddShape(UnitBox), .Density = Density, .Friction = Friction});
}

// NoIndex after reporting which compound was refused.
Index AddCompound(World &world, std::span<const Index> parts, Pose &frame, const char *what) {
    const Index shape = world.AddCompound(parts, &frame);
    if (shape == NoIndex) std::println("the {} would not compound: {} refused", what, world.RefusedCompounds);
    return shape;
}

// A body made of several pieces, which KHR authors as a node with a motion and several collider descendants.
// Four measurements on the one mechanism: a solid split in two against the same solid described in one.
// Then where a table of five pieces stands and whether the host can put its top back where it authored it.
// Then what a body sliding across the seam of two coplanar siblings does.
// Last, what eight leaves at once cost a run budgeted in manifolds.
void Compound(std::span<char *> args) {
    const uint32_t steps = Arg(args, 2, 240);
    const float speed = ArgF(args, 3, 2);
    const StepSettings settings{};
    const Reporting report{60, steps};
    // A slide at mu g is over in a fifth of a second, so sixty steps would average over nothing.
    const Reporting sliding{10, steps};
    const mtl::Context context;
    Solver solver{context};

    // One box, and the same solid described as two halves.
    // "The same body" means the same mass, principal inertia, rest height, friction, and the same contacts at step one.
    World whole{context}, split{context};
    AddGround(whole);
    AddGround(split);
    const Index one_shape = whole.AddShape(UnitBox);
    std::vector<Index> halves;
    for (const float side : {-1.f, 1.f})
        halves.push_back(split.AddShape({.HalfExtents = {Half / 2, Half, Half}, .Kind = ShapeBox, .Local = At(float3{side * Half / 2, 0, 0})}));
    Pose halved_frame{};
    const Index split_shape = AddCompound(split, halves, halved_frame, "two halves");
    if (split_shape == NoIndex) return;
    const BodyMass solid = MassOf(whole, one_shape, Density);
    const BodyMass built = MassOf(split, split_shape, Density);
    std::println("one box against the same solid described as two halves, both pushed at {:g} m/s", speed);
    std::println("  mass {:10.5f} against {:10.5f} kg", 1 / solid.InvMass, 1 / built.InvMass);
    std::println("  inertia ({:9.5f} {:9.5f} {:9.5f}) against ({:9.5f} {:9.5f} {:9.5f}) kg m^2", 1 / solid.InvInertiaLocal.x, 1 / solid.InvInertiaLocal.y,
                 1 / solid.InvInertiaLocal.z, 1 / built.InvInertiaLocal.x, 1 / built.InvInertiaLocal.y, 1 / built.InvInertiaLocal.z);
    std::println("  the pieces put the body frame at ({:8.5f} {:8.5f} {:8.5f}), turned {:9.2e} rad - the middle of the two, which is where the one box already was",
                 float(halved_frame.Position.x), float(halved_frame.Position.y), float(halved_frame.Position.z), simd::length(RotationVector(halved_frame.Orientation)));
    const Index one_box = whole.AddBody({.Pose = At(float3{0, Half, 0}), .Velocity = {.Linear = {speed, 0, 0}}, .Shape = one_shape, .Density = Density, .Friction = Friction});
    const Index two_halves = split.AddBody({.Pose = At(float3{0, Half, 0}), .Velocity = {.Linear = {speed, 0, 0}}, .Shape = split_shape, .Density = Density, .Friction = Friction});
    if (one_box == NoIndex || two_halves == NoIndex) return (void)std::println("refused a body: {} offsets with no authored mass", whole.OffsetsWithoutMass + split.OffsetsWithoutMass);

    // The speed Coulomb removes per step, the unit deceleration is reported in.
    const float coulomb = Friction * std::abs(settings.Gravity.y) * settings.DeltaTime;
    float previous_whole = speed, previous_split = speed, worst_apart = 0;
    for (uint32_t step = 0; step < steps; ++step) {
        solver.Step(whole, settings);
        solver.Step(split, settings);
        const float now_whole = simd::length(whole.Velocities[one_box].Linear), now_split = simd::length(split.Velocities[two_halves].Linear);
        const float slowed_whole = previous_whole - now_whole, slowed_split = previous_split - now_split;
        previous_whole = now_whole;
        previous_split = now_split;
        const float apart = float(simd::distance(whole.Poses[one_box].Position, split.Poses[two_halves].Position));
        worst_apart = std::max(worst_apart, apart);
        if (step == 0) {
            // Every contact each body has at the end of step one, so the two descriptions can be compared.
            for (const auto &[name, world, body] : {std::tuple{"one box   ", &whole, one_box}, std::tuple{"two halves", &split, two_halves}}) {
                std::print("  {} step 1:", name);
                for (uint32_t slot = 0; slot < ContactsPerBody; ++slot) {
                    const Contact &contact = world->Contacts[body * ContactsPerBody + slot];
                    if (!contact.Active) continue;
                    std::print("  n=({:5.2f} {:5.2f} {:5.2f}) C0={:9.2e} leaf={}", contact.Normal.x, contact.Normal.y, contact.Normal.z, contact.C0.x, OwnChild(contact.Children));
                }
                std::println("");
            }
        }
        if (!report(step)) continue;
        std::println("{:4}  one y={:8.5f} v={:7.4f} a/mu g={:6.3f}  |  halves y={:8.5f} v={:7.4f} a/mu g={:6.3f}  |  apart {:9.2e} m", step,
                     float(whole.Poses[one_box].Position.y), previous_whole, slowed_whole / coulomb, float(split.Poses[two_halves].Position.y),
                     previous_split, slowed_split / coulomb, apart);
    }
    std::println("  worst the two descriptions ever got apart: {:.3e} m", worst_apart);

    // A table: a slab and four legs, each authored where a modeller put it.
    // The compound moves the body frame onto the centre of mass and returns `frame`, which the host uses to put the top back where it was authored.
    constexpr float TopHalf = 0.05f, TopY = 0.75f, LegHalf = 0.05f, LegHigh = (TopY - TopHalf) / 2;
    World table_world{context};
    AddGround(table_world);
    std::vector<Index> table_parts{table_world.AddShape({.HalfExtents = {1, TopHalf, 0.6f}, .Kind = ShapeBox, .Local = At(float3{0, TopY, 0})})};
    for (const float x : {-0.9f, 0.9f})
        for (const float z : {-0.5f, 0.5f})
            table_parts.push_back(table_world.AddShape({.HalfExtents = {LegHalf, LegHigh, LegHalf}, .Kind = ShapeBox, .Local = At(float3{x, LegHigh, z})}));
    Pose table_frame{};
    const Index table_shape = AddCompound(table_world, table_parts, table_frame, "table");
    if (table_shape == NoIndex) return;
    // The returned frame puts the pieces where they were authored.
    // The millimetre clears the plane, so the table lands rather than starting inside it.
    const Pose authored = At(table_frame.Position + float3{0, 1e-3f, 0}, table_frame.Orientation);
    const Index table = table_world.AddBody({.Pose = authored, .Shape = table_shape, .Density = Density, .Friction = Friction});
    if (table == NoIndex) return (void)std::println("refused the table: {} offsets with no authored mass", table_world.OffsetsWithoutMass);
    const float3 top_authored{0, TopY, 0};
    const float3 top_in_body = LocalPoint(table_frame, top_authored);
    std::println("a table of a slab and four legs, its body frame at ({:8.5f} {:8.5f} {:8.5f}) in the frame it was authored in", float(table_frame.Position.x),
                 float(table_frame.Position.y), float(table_frame.Position.z));
    for (uint32_t step = 0; step < steps; ++step) {
        solver.Step(table_world, settings);
        if (!report(step)) continue;
        const Pose pose = table_world.Poses[table];
        const float3 top = WorldPoint(pose, top_in_body);
        std::println("{:4}  y={:8.5f} of {:8.5f} authored  tilt={:9.2e}  contacts={:2} refused={}  |  top at ({:8.5f} {:8.5f} {:8.5f}), authored ({:8.5f} {:8.5f} {:8.5f})", step,
                     float(pose.Position.y), float(table_frame.Position.y), simd::length(RotationVector(QuatMul(pose.Orientation, QuatConjugate(table_frame.Orientation)))),
                     ActiveContacts(table_world, table), table_world.ContactRefusals[table], float(top.x), float(top.y), float(top.z), float(top_authored.x),
                     float(top_authored.y), float(top_authored.z));
    }

    // The seam two coplanar siblings share.
    // A mesh's tessellation seams are marked by the cook's active-edge flags, and a floor of two boxes has the same internal edge with no such mark.
    // The measurement is whether a body sliding across is kicked or braked.
    // Three floors run side by side at the same speed from the same place.
    // They are the two boxes as one compound, the same two welded by World::WeldStatic, and the whole thing as one box for the control.
    // The weld marks the join at the geometry rather than at cook time, and must read as the compound does.
    World jointed{context}, separate{context}, seamless{context};
    std::vector<Index> floor_parts;
    for (const float side : {-1.f, 1.f})
        floor_parts.push_back(jointed.AddShape({.HalfExtents = {FloorHalf, FloorTop, FloorHalf}, .Kind = ShapeBox, .Local = At(float3{side * FloorHalf, 0, 0})}));
    Pose floor_frame{};
    const Index floor_shape = AddCompound(jointed, floor_parts, floor_frame, "floor");
    if (floor_shape == NoIndex) return;
    jointed.AddBody({.Pose = At(floor_frame.Position), .Shape = floor_shape, .Density = 0, .Friction = Friction});
    for (const float side : {-1.f, 1.f})
        separate.AddBody({.Pose = At(float3{side * FloorHalf, 0, 0}), .Shape = separate.AddShape(FloorBox), .Density = 0, .Friction = Friction});
    const uint32_t welded = separate.WeldStatic(); // the two faces the join is made of, one from each box
    seamless.AddBody({.Shape = seamless.AddShape(SeamlessFloor), .Density = 0, .Friction = Friction});
    const Index over_seam = AddSlider(jointed, speed), over_two = AddSlider(separate, speed), over_none = AddSlider(seamless, speed);
    std::println("a box slid at {:g} m/s across a floor of two boxes, against the same floor as one - the kick is how far it left the height the control kept", speed);
    std::print("  the join the two halves share, as the faces each of them found buried (the weld buried {} between the two bodies):", welded);
    for (uint32_t i = 0; i < ChildrenPerCompound; ++i) {
        const Index child = ChildOf(jointed.Shapes[floor_shape], i);
        if (child == NoIndex) break;
        std::print("  child {} mask {:#06b}", i, InternalFaces(jointed.Shapes[child]));
    }
    std::println("");
    float kick = 0, two_kick = 0, worst_step_loss = 0;
    float last_seam = speed, last_two = speed, last_none = speed;
    for (uint32_t step = 0; step < steps; ++step) {
        solver.Step(jointed, settings);
        solver.Step(separate, settings);
        solver.Step(seamless, settings);
        const float seam_y = float(jointed.Poses[over_seam].Position.y), two_y = float(separate.Poses[over_two].Position.y);
        const float none_y = float(seamless.Poses[over_none].Position.y);
        const float seam_v = float(jointed.Velocities[over_seam].Linear.x), two_v = float(separate.Velocities[over_two].Linear.x);
        const float none_v = float(seamless.Velocities[over_none].Linear.x);
        kick = std::max(kick, seam_y - none_y);
        two_kick = std::max(two_kick, two_y - none_y);
        worst_step_loss = std::max(worst_step_loss, (last_seam - seam_v) - (last_none - none_v));
        last_seam = seam_v;
        last_two = two_v;
        last_none = none_v;
        if (!sliding(step)) continue;
        std::println("{:4}  compound x={:8.5f} y={:8.5f} vx={:7.4f} rows={:2}  |  two bodies x={:8.5f} y={:8.5f} vx={:7.4f}  |  one box x={:8.5f} y={:8.5f} vx={:7.4f}", step,
                     float(jointed.Poses[over_seam].Position.x), seam_y, seam_v, ActiveContacts(jointed, over_seam),
                     float(separate.Poses[over_two].Position.x), two_y, two_v, float(seamless.Poses[over_none].Position.x), none_y, none_v);
    }
    std::println("  worst kick {:.3e} m as one compound, {:.3e} m as two bodies, against a {:.1e} m margin scale", kick, two_kick, double(settings.ContactMargin));
    std::println("  worst extra loss in one step {:.3e} m/s against Coulomb's {:.3e}", worst_step_loss, coulomb);
    std::println("  speed left: {:8.5f} over the compound, {:8.5f} over two bodies, {:8.5f} over one box", last_seam, last_two, last_none);

    // And the budget.
    // Eight boxes in one body resting on the plane is one manifold per leaf.
    // That is the most one partner can demand of a compound, and eight of the ten manifolds a run has.
    // It is the busiest single pair in the engine, with two manifolds spare.
    World eight{context};
    AddGround(eight);
    constexpr float Cube = 0.2f;
    std::vector<Index> cubes;
    for (uint32_t i = 0; i < ChildrenPerCompound; ++i)
        cubes.push_back(eight.AddShape({.HalfExtents = {Cube, Cube, Cube}, .Kind = ShapeBox, .Local = At(float3{2.5f * Cube * (float(i) - 3.5f), 0, 0})}));
    Pose row_frame{};
    const Index row_shape = AddCompound(eight, cubes, row_frame, "row of eight");
    if (row_shape == NoIndex) return;
    const Index row = eight.AddBody({.Pose = At(row_frame.Position + float3{0, Cube + 1e-3f, 0}), .Shape = row_shape, .Density = Density, .Friction = Friction});
    if (row == NoIndex) return (void)std::println("refused the row: {} offsets with no authored mass", eight.OffsetsWithoutMass);
    std::println("eight cubes in one body on a plane, which asks the run for one manifold a leaf against a budget of {}", ManifoldsPerBody);
    for (uint32_t step = 0; step < steps; ++step) {
        solver.Step(eight, settings);
        if (!report(step)) continue;
        const Pose pose = eight.Poses[row];
        std::println("{:4}  y={:8.5f} of {:8.5f}  tilt={:9.2e}  contacts={:2} of {}  refused={}", step, float(pose.Position.y), double(Cube),
                     simd::length(RotationVector(pose.Orientation)), ActiveContacts(eight, row), ContactsPerBody, eight.ContactRefusals[row]);
    }
}

// The join two separate bodies share, swept by how far out of true it is.
// A contact rests ContactMargin inside the face carrying it, so a slider arrives at the far box a resting depth below its top.
// The box SAT reads the horizontal face as the least overlap while that depth is the smaller, which makes a wall of the engine's own resting depth.
// The sweep is the step height: coplanar, where the only ledge is that resting depth, then half a margin, a margin, twice it and five times it.
// The four raised columns have a real step on top of that depth.
// The control is the same floor as one box, so speed left and height kept are differences against a slider that met nothing.
// Only the coplanar world welds, so the sweep shows the fix on one column and the untouched defect on the other four.
void Join(std::span<char *> args) {
    const uint32_t steps = Arg(args, 2, 240);
    const float speed = ArgF(args, 3, 2);
    const StepSettings settings{};
    // A slide at mu g is over in a fifth of a second, so sixty steps would average over nothing.
    const Reporting sliding{10, steps};
    const mtl::Context context;
    Solver solver{context};

    const float margin = settings.ContactMargin;
    const float coulomb = Friction * std::abs(settings.Gravity.y) * settings.DeltaTime;

    // The control: one box wide enough for the whole slide, with no join anywhere on it.
    World seamless{context};
    seamless.AddBody({.Shape = seamless.AddShape(SeamlessFloor), .Density = 0, .Friction = Friction});
    const Index alone = AddSlider(seamless, speed);

    // One world per step height, with the far box raised by that step.
    // Each is stepped beside the control, so the two are always the same age.
    const std::vector<float> steps_high{0, 0.5f * margin, margin, 2 * margin, 5 * margin};
    std::vector<World> worlds;
    std::vector<Index> sliders;
    std::vector<uint32_t> welded;
    worlds.reserve(steps_high.size());
    for (const float high : steps_high) {
        World &world = worlds.emplace_back(context);
        for (const float side : {-1.f, 1.f})
            world.AddBody({.Pose = At(float3{side * FloorHalf, side > 0 ? high : 0, 0}), .Shape = world.AddShape(FloorBox), .Density = 0, .Friction = Friction});
        sliders.push_back(AddSlider(world, speed));
        // The weld: the boxes are static, so faces meeting flush lie inside the solid the two make together.
        // Those faces are marked as a compound's cook marks a sibling seam.
        // The weld fires on the coplanar world only, since a face standing proud is not covered.
        welded.push_back(world.WeldStatic());
    }

    std::println("a box slid at {:g} m/s into the join of two separate static boxes, swept by how far the far one stands proud, against a margin of {:.1e} m",
                 speed, double(margin));
    std::print("  faces the static weld buried:");
    for (uint32_t i = 0; i < worlds.size(); ++i) std::print("  {:.2f}xm {}", double(steps_high[i] / margin), welded[i]);
    std::println("");
    std::vector<float> kick(worlds.size(), 0), worst_loss(worlds.size(), 0), last(worlds.size(), speed);
    float last_alone = speed;
    for (uint32_t step = 0; step < steps; ++step) {
        solver.Step(seamless, settings);
        const float alone_y = float(seamless.Poses[alone].Position.y), alone_v = float(seamless.Velocities[alone].Linear.x);
        const float alone_lost = last_alone - alone_v;
        for (uint32_t i = 0; i < worlds.size(); ++i) {
            solver.Step(worlds[i], settings);
            const float y = float(worlds[i].Poses[sliders[i]].Position.y), v = float(worlds[i].Velocities[sliders[i]].Linear.x);
            // Measured against the control's height rather than the authored one, so a step legitimately climbed only reads as a kick above that height.
            kick[i] = std::max(kick[i], y - alone_y - steps_high[i]);
            worst_loss[i] = std::max(worst_loss[i], (last[i] - v) - alone_lost);
            last[i] = v;
        }
        last_alone = alone_v;
        if (!sliding(step)) continue;
        std::print("{:4}  no join x={:8.5f} vx={:7.4f}", step, float(seamless.Poses[alone].Position.x), alone_v);
        for (uint32_t i = 0; i < worlds.size(); ++i)
            std::print("  |  {:.2f}xm x={:8.5f} y={:8.5f} vx={:7.4f}", double(steps_high[i] / margin),
                       float(worlds[i].Poses[sliders[i]].Position.x), float(worlds[i].Poses[sliders[i]].Position.y), last[i]);
        std::println("");
    }
    std::println("  the control travelled {:8.5f} m and kept {:7.4f} m/s", float(seamless.Poses[alone].Position.x) - SlideFrom, last_alone);
    for (uint32_t i = 0; i < worlds.size(); ++i)
        std::println("  step {:5.2f} x margin: travelled {:8.5f} m, kept {:7.4f} m/s, worst kick {:.3e} m, worst extra loss in one step {:.3e} m/s against Coulomb's {:.3e}, resting on {} rows",
                     double(steps_high[i] / margin), float(worlds[i].Poses[sliders[i]].Position.x) - SlideFrom, last[i], kick[i], worst_loss[i], coulomb,
                     ActiveContacts(worlds[i], sliders[i]));
}

void Drop(std::span<char *> args) {
    const float restitution = ArgF(args, 2, 0.5f);
    const uint32_t steps = Arg(args, 3, 600);
    const StepSettings settings{.Iterations = Arg(args, 4, 10), .Beta = ArgF(args, 5, StepSettings{}.Beta)};
    const mtl::Context context;
    Solver solver{context};
    World world{context};
    const auto shape = world.AddShape(UnitBox);
    world.AddBody({.Shape = world.AddShape(GroundPlane), .Friction = Friction, .Restitution = restitution});
    const auto box = world.AddBody({.Pose = At(float3{0, Half + 1, 0}), .Shape = shape, .Density = Density, .Friction = Friction, .Restitution = restitution});

    std::println("drop from 1 m at restitution {:g}, {} iterations, beta {:g}. Below {:g} m/s an impact does not bounce", restitution, settings.Iterations, settings.Beta, settings.BounceSpeedFactor * simd::length(settings.Gravity) * settings.DeltaTime);
    uint32_t bounces = 0;
    for (uint32_t step = 0; step < steps; ++step) {
        const float before = world.Velocities[box].Linear.y;
        solver.Step(world, settings);
        const float after = world.Velocities[box].Linear.y;
        if (before >= 0 || after <= 0) continue; // report only the step a bounce happened on
        const auto totals = Totals(world);
        std::println("{:4} bounce {:2} arrived {:7.4f} left {:7.4f} ratio {:6.4f} | contacts={} normal={:9.1f}", step, ++bounces, -before, after, after / -before, totals.Count, simd::length(totals.Normal));
    }
    std::println("came to rest at {:.5f} m with |v| {:.6f} after {} bounces", float(world.Poses[box].Position.y), simd::length(world.Velocities[box].Linear), bounces);
}

// The largest normal penalty, as a multiple of one box's inertial stiffness.
// That is the unit the penalty floor and the instability band are both expressed in.
float PeakPenaltyRatio(const World &world, float inertial) {
    float peak = 0;
    EachActive(world, [&peak](const Contact &contact) { peak = std::max(peak, contact.Penalty[0]); });
    return peak / inertial;
}

void Slam(std::span<char *> args) {
    const float speed = ArgF(args, 2, 40), density = ArgF(args, 3, 1000);
    const uint32_t steps = Arg(args, 4, 180);
    const StepSettings settings{.Beta = ArgF(args, 5, StepSettings{}.Beta)};
    const mtl::Context context;
    Solver solver{context};
    World world{context};
    const auto shape = world.AddShape(UnitBox);
    AddGround(world);
    // Just clear of the plane, so the first step arrives at the speed asked for rather than at whatever a drop height gives.
    const auto box = world.AddBody({.Pose = At(float3{0, Half + speed * settings.DeltaTime, 0}), .Velocity = {.Linear = {0, -speed, 0}}, .Shape = shape, .Density = density, .Friction = Friction});
    const float mass = 1 / world.Masses[box].InvMass;
    const float inertial = mass / (settings.DeltaTime * settings.DeltaTime);
    // PenaltyMax is absolute, so a light body has orders of magnitude of headroom before it binds and a very heavy one is already inside the band.
    std::println("slam at {:g} m/s, density {:g}, mass {:g}, beta {:g}. m/h^2 = {:.4g}, and PenaltyMax {:g} is {:.4g} of it", speed, density, mass, settings.Beta, inertial, settings.PenaltyMax, settings.PenaltyMax / inertial);

    float peak_ratio = 0, deepest = 0, fastest_out = 0;
    for (uint32_t step = 0; step < steps; ++step) {
        solver.Step(world, settings);
        const float ratio = PeakPenaltyRatio(world, inertial);
        const float height = world.Poses[box].Position.y, out = world.Velocities[box].Linear.y;
        peak_ratio = std::max(peak_ratio, ratio);
        deepest = std::max(deepest, Half - height);
        fastest_out = std::max(fastest_out, out);
        if (step < 12) std::println("{:4} y={:8.5f} vy={:9.4f} sink={:8.5f} k/(m/h2)={:11.4g} contacts={}", step, height, out, Half - height, ratio, ActiveContacts(world));
    }
    std::println("peak k/(m/h2) {:.4g}, deepest {:.5f} m, fastest rebound {:.4f} m/s, "
                 "rest at {:.5f} m with |v| {:.6f}",
                 peak_ratio, deepest, fastest_out, float(world.Poses[box].Position.y), simd::length(world.Velocities[box].Linear));
}
void Seams(std::span<char *> args) {
    const uint32_t cells = Arg(args, 2, 16), steps = Arg(args, 3, 180);
    const float speed = ArgF(args, 4, 2);
    constexpr float Radius = 0.25f;
    const mtl::Context context;
    Solver solver{context};
    World mesh{context}, plane{context};
    if (!AddMeshFloor(mesh, cells)) return (void)std::println(stderr, "the mesh would not cook");
    plane.AddBody({.Shape = plane.AddShape(GroundPlane)});
    // Off the grid lines, so it crosses seams rather than running along one.
    for (World *world : {&mesh, &plane})
        world->AddBody({.Pose = At(float3{-8, Radius, 0.07f}), .Velocity = {.Linear = {speed, 0, 0}}, .Shape = world->AddShape({.HalfExtents = {0, 0, 0}, .Radius = Radius, .Kind = ShapeSphere})});
    std::println("a ball rolling across {} triangles, and the same ball on a plane. {} cells over 20 m, {} m/s", 2 * cells * cells, cells, speed);
    uint32_t crowded = 0, empty = 0;
    float worst_height = 0, worst_behind = 0;
    for (uint32_t step = 0; step < steps; ++step) {
        solver.Step(mesh);
        solver.Step(plane);
        const uint32_t contacts = ActiveContacts(mesh, 1);
        crowded += contacts > 1 ? 1 : 0;
        empty += contacts == 0 ? 1 : 0;
        worst_height = std::max(worst_height, std::abs(float(mesh.Poses[1].Position.y) - Radius));
        worst_behind = std::max(worst_behind, std::abs(float(mesh.Poses[1].Position.x - plane.Poses[1].Position.x)));
        if (step % 30 == 29)
            std::println("{:4} contacts={} x={:8.4f} against {:8.4f} y={:7.5f} vx={:7.4f} against {:7.4f}", step, contacts, float(mesh.Poses[1].Position.x), float(plane.Poses[1].Position.x), float(mesh.Poses[1].Position.y), float(mesh.Velocities[1].Linear.x), float(plane.Velocities[1].Linear.x));
    }
    std::println("steps holding more than one contact: {}, holding none: {}. Worst height error {:.6f} m, "
                 "worst distance behind the plane {:.6f} m, refused {}",
                 crowded, empty, worst_height, worst_behind, mesh.ContactRefusals[1]);
}
// A wall of two triangles wound to face back down the line of fire.
// A body that crosses a mesh cannot be recovered.
Index WallMesh(World &world, float reach) {
    const std::vector<float3> points{float3{0, -reach, -reach}, float3{0, -reach, reach}, float3{0, reach, reach}, float3{0, reach, -reach}};
    const std::vector<uint32_t> indices{0, 1, 2, 0, 2, 3};
    return world.AddMesh(points, indices);
}

// Which of the two boxes presented the reference face, over every box-on-box contact in a world.
// Bit 13 of a feature is the side the separating axis picked, and a flipped pair moves that to the other box.
struct FaceOwners {
    uint32_t Lower{}, Upper{};
};

FaceOwners Owners(const World &world) {
    FaceOwners owners;
    EachActive(world, [&](const Contact &contact) {
        if (contact.BodyA == 0 || contact.BodyB == 0) return; // box on box, not on the plane
        const bool on_b = ((contact.Feature >> 13) & 1) != 0;
        const Index face = on_b ? contact.BodyB : contact.BodyA;
        const Index other = on_b ? contact.BodyA : contact.BodyB;
        if (world.Poses[face].Position.y < world.Poses[other].Position.y) ++owners.Lower;
        else ++owners.Upper;
    });
    return owners;
}

void Order(std::span<char *> args) {
    const uint32_t boxes = Arg(args, 2, 12), steps = Arg(args, 3, 600);
    StepSettings settings{.Iterations = Arg(args, 4, StepSettings{}.Iterations), .MaxColors = Arg(args, 5, StepSettings{}.MaxColors)};
    ApplyNoSleep(settings);
    const Reporting report{60, steps};
    const mtl::Context context;
    Solver solver{context};
    World up{context}, down{context};

    // The plane is added first in both, so only the boxes differ.
    // `at` maps a place in the stack to the body at that place, which makes the two runs comparable.
    std::vector<Index> up_at(boxes), down_at(boxes);
    for (World *world : {&up, &down}) {
        world->AddShape(UnitBox);
        AddGround(*world);
    }
    const auto place = [](World &world, uint32_t i) { return Place(world, 0, float3{0, Half + 1.02f * float(i), 0}); };
    for (uint32_t i = 0; i < boxes; ++i) up_at[i] = place(up, i);
    for (uint32_t i = boxes; i-- > 0;) down_at[i] = place(down, i);

    std::println("a stack of {} built bottom-up and top-down, {} iterations, {} colors. Same geometry, "
                 "same places, opposite pair ownership.\n  One color is Jacobi throughout, which is what "
                 "takes the coloring's own index order out of the comparison",
                 boxes, settings.Iterations, settings.MaxColors);
    float worst = 0, apart = 0;
    uint32_t up_slept = ~0u, down_slept = ~0u;
    for (uint32_t step = 0; step < steps; ++step) {
        solver.Step(up, settings);
        solver.Step(down, settings);
        apart = 0;
        for (uint32_t i = 0; i < boxes; ++i)
            apart = std::max(apart, simd::length(up.Poses[up_at[i]].Position - down.Poses[down_at[i]].Position));
        worst = std::max(worst, apart);

        const uint32_t up_asleep = Asleep(up, up_at, settings), down_asleep = Asleep(down, down_at, settings);
        if (up_slept == ~0u && up_asleep == boxes) up_slept = step;
        if (down_slept == ~0u && down_asleep == boxes) down_slept = step;
        if (!report(step)) continue;

        const auto up_owners = Owners(up), down_owners = Owners(down);
        // The colours each stack settled on, bottom first in both.
        // Colours solve in sequence, so load propagates from the end with colour zero.
        // Priority goes by index, so add order decides which end that is.
        std::string up_colors, down_colors;
        for (uint32_t i = 0; i < boxes; ++i) {
            up_colors += char('0' + ColorOf(up.Colors[up_at[i]]) % 10);
            down_colors += char('0' + ColorOf(down.Colors[down_at[i]]) % 10);
        }
        // Where resting on the one below would put the top box.
        const float ideal = Half + float(boxes - 1) * (1 - settings.ContactMargin) - settings.ContactMargin;
        std::println("{:4} apart {:9.6f}  |  up: top sag {:8.5f} asleep {:2} face on lower/upper {:3}/{:3}"
                     "  |  down: top sag {:8.5f} asleep {:2} face on lower/upper {:3}/{:3}",
                     step, apart, ideal - float(up.Poses[up_at[boxes - 1]].Position.y), up_asleep, up_owners.Lower, up_owners.Upper, ideal - float(down.Poses[down_at[boxes - 1]].Position.y), down_asleep, down_owners.Lower, down_owners.Upper);
        std::println("      colors, bottom of the stack first:  up {}  down {}", up_colors, down_colors);
    }
    const auto slept = [](uint32_t at) { return at == ~0u ? std::string{"never"} : std::to_string(at); };
    std::println("at rest the two are {:.6f} m apart, worst they ever got {:.6f} m. Bottom-up slept at {}, top-down at {}", apart, worst, slept(up_slept), slept(down_slept));
}

// A kinematic body has no inverse mass and the host moves it every step.
// Three mechanisms are measured here.
// A translating slab carries a box, which measures static friction anchored against a body the solve never moves.
// A swung paddle strikes a resting ball, which measures the contact reach and the constraint reading a static body's motion.
// A slab starting under a sleeping stack wakes it, which measures the wake spread, where a static neighbour propagates nothing.
void Platform(std::span<char *> args) {
    const uint32_t steps = Arg(args, 2, 180);
    const float speed = ArgF(args, 3, 2);
    StepSettings settings{};
    ApplyNoSleep(settings); // separates the two mechanisms here: waking and riding
    const float dt = settings.DeltaTime;
    constexpr float BallRadius = 0.25f;
    constexpr Shape Slab{.HalfExtents = {8, 0.25f, 4}, .Kind = ShapeBox};
    const mtl::Context context;

    std::println("a kinematic body: no inverse mass, and the host writes its pose and a velocity to match\n"
                 "  every step. {} steps at {:g} m/s, which is {:.5f} m a step", steps, speed, speed * dt);

    // A box on a slab the host translates, which must arrive where the slab did at the slab's speed.
    // Lag means the friction anchor is slipping, and a box reading zero velocity while it travels means the motion bypassed the velocity state.
    {
        Solver solver{context};
        World world{context};
        const auto slab = world.AddBody({.Pose = At(float3{0, -0.25f, 0}), .Shape = world.AddShape(Slab), .Density = 0, .Friction = Friction});
        const auto box = Place(world, world.AddShape(UnitBox), float3{0, Half, 0});
        for (uint32_t settle = 0; settle < 120; ++settle) solver.Step(world, settings);
        const float held = float(world.Poses[box].Position.x - world.Poses[slab].Position.x);
        std::println("a box riding a translating slab, both mu {:g}:", Friction);
        // The animation track, owned by the host.
        float3 at = world.Poses[slab].Position;
        for (uint32_t step = 0; step < steps; ++step) {
            at.x += speed * dt;
            Drive(world, slab, at, float3{speed, 0, 0});
            solver.Step(world, settings);
            if (step % 30 != 29 && step != steps - 1) continue;
            std::println("  {:4} slab x={:8.4f} box x={:8.4f} slipped {:9.6f} box vx={:8.5f} of {:g} asleep={}",
                         step, float(world.Poses[slab].Position.x), float(world.Poses[box].Position.x),
                         float(world.Poses[box].Position.x - world.Poses[slab].Position.x) - held,
                         float(world.Velocities[box].Linear.x), speed, world.Quiet[box] >= settings.SleepSteps);
        }
    }

    // A paddle swung into a resting ball, in free space, so the ball's motion comes only from the paddle.
    // Struck means it leaves at a speed of its own and keeps it, shoved means it moves only while the paddle is against it, and crossed is unrecoverable.
    std::println("a kinematic paddle swung into a ball at rest, in free space. It must leave with speed of its own:");
    for (const float swing : {1.f, 5.f, 10.f, 20.f}) {
        Solver solver{context};
        World world{context};
        StepSettings free_space{.Gravity = {0, 0, 0}};
        ApplyNoSleep(free_space);
        // A run-up of a second, so every row strikes at about the same step, against a ball that has had time to fall asleep.
        const auto paddle = world.AddBody({.Pose = At(float3{-1 - swing, 0, 0}), .Shape = world.AddShape({.HalfExtents = {0.1f, 1, 1}, .Kind = ShapeBox}), .Density = 0, .Friction = Friction});
        const auto ball = world.AddBody({.Pose = At(float3{0, 0, 0}), .Shape = world.AddShape({.Radius = BallRadius, .Kind = ShapeSphere}), .Density = Density, .Friction = Friction});
        // Driven past where the ball started and then released, so the measurement afterwards is the ball on its own rather than the ball being carried.
        // The ball is left to sleep first.
        for (uint32_t settle = 0; settle < 60; ++settle) solver.Step(world, free_space);
        const bool slept = world.Quiet[ball] >= free_space.SleepSteps;
        float behind = 0, fastest = 0;
        float3 at = world.Poses[paddle].Position;
        for (uint32_t step = 0; step < steps; ++step) {
            if (at.x < 4) {
                at.x += swing * dt;
                Drive(world, paddle, at, float3{swing, 0, 0});
            } else {
                Drive(world, paddle, at, float3(0));
            }
            solver.Step(world, free_space);
            behind = std::min(behind, float(world.Poses[ball].Position.x - world.Poses[paddle].Position.x));
            fastest = std::max(fastest, float(world.Velocities[ball].Linear.x));
        }
        // A ball faster than the paddle face is being driven rather than struck, and a slower one is being shoved.
        std::println("  {:5.0f} m/s ({:7.4f} m a step)  ball ended at x={:8.4f} going {:8.4f} m/s, fastest {:8.4f}, "
                     "worst place relative to the paddle {:8.4f}, asleep when struck {}",
                     swing, swing * dt, float(world.Poses[ball].Position.x), float(world.Velocities[ball].Linear.x), fastest, behind, slept);
    }

    // And the stack from `teleport`, on a slab that starts moving rather than one that teleports.
    // A sleeping body skips its pairs against a static partner, and waking spreads only from moving dynamic bodies.
    // The stack therefore stays asleep when the slab gets under way.
    {
        Solver solver{context};
        World world{context};
        const auto slab = world.AddBody({.Pose = At(float3{0, -0.25f, 0}), .Shape = world.AddShape(Slab), .Density = 0, .Friction = Friction});
        const auto shape = world.AddShape(UnitBox);
        std::vector<Index> stack;
        for (uint32_t i = 0; i < 3; ++i) stack.push_back(Place(world, shape, float3{0, Half + 1.02f * float(i), 0}));
        for (uint32_t settle = 0; settle < 400; ++settle) solver.Step(world, settings);
        std::println("a stack of {} asleep on a slab that then starts moving: {}/{} asleep when it does",
                     stack.size(), Asleep(world, stack, settings), stack.size());
        const float held = float(world.Poses[stack.front()].Position.x - world.Poses[slab].Position.x);
        float3 at = world.Poses[slab].Position;
        for (uint32_t step = 0; step < steps; ++step) {
            at.x += speed * dt;
            Drive(world, slab, at, float3{speed, 0, 0});
            solver.Step(world, settings);
            if (step % 30 != 29 && step != steps - 1) continue;
            // Every level, since friction accelerates each layer at no more than mu g.
            // A slab starting at speed leaves each level sliding v^2 / 2 mu g behind the one under it.
            // At 2 m/s that is 0.41 m a level, which takes the top of the stack over the side.
            // That shear is Coulomb friction, not a kinematics defect.
            std::print("  {:4} slab x={:8.4f} asleep={}/{}", step, float(world.Poses[slab].Position.x), Asleep(world, stack, settings), stack.size());
            for (const auto body : stack)
                std::print("  x={:8.4f} slipped {:9.6f} y={:7.4f}", float(world.Poses[body].Position.x),
                           float(world.Poses[body].Position.x - world.Poses[slab].Position.x) - held, float(world.Poses[body].Position.y));
            std::println("");
        }
    }
}

// A dynamic one-sided mesh: a body whose shape is a triangle mesh, with the mass the host gave it, moving and colliding like any other body.
// A kinematic paddle mesh, one open quad, strikes a resting ball, which runs the mesh path with the mesh moving and the convex side owning the pair.
// A paddle wheel on a hinge is turned by dropped balls, against the same wheel with its paddles off as the control.
// A mesh is dropped onto a plane and onto a box: neither a plane nor a mesh presents a manifold, so those two meet with no contact and the mesh falls through.
void Wheel(std::span<char *> args) {
    const uint32_t steps = Arg(args, 2, 600), balls = Arg(args, 3, 12), paddles = Arg(args, 4, 8);
    StepSettings settings{};
    ApplyNoSleep(settings);
    const float dt = settings.DeltaTime;
    const float BallRadius = ArgF(args, 5, 0.16f);
    const mtl::Context context;

    // Two triangles and no volume anywhere in the striker, so this only works if the mesh path itself carries the motion.
    {
        Solver solver{context};
        World world{context};
        StepSettings free_space{.Gravity = {0, 0, 0}};
        ApplyNoSleep(free_space);
        constexpr float Swing = 5;
        // Wound to face along +x, the direction it is driven.
        // (B-A)x(C-A) is the side it pushes from.
        const std::vector<float3> quad{float3{0, -1, -1}, float3{0, -1, 1}, float3{0, 1, 1}, float3{0, 1, -1}};
        const std::vector<uint32_t> wound{0, 2, 1, 0, 3, 2};
        const Index shape = world.AddMesh(quad, wound);
        if (shape == NoIndex) return (void)std::println(stderr, "the paddle would not cook");
        const auto paddle = world.AddBody({.Pose = At(float3{-2, 0, 0}), .Shape = shape, .Friction = Friction});
        const auto ball = world.AddBody({.Pose = At(float3{0, 0, 0}), .Shape = world.AddShape({.Radius = BallRadius, .Kind = ShapeSphere}), .Density = Density, .Friction = Friction});
        for (uint32_t settle = 0; settle < 60; ++settle) solver.Step(world, free_space);
        float3 at = world.Poses[paddle].Position;
        for (uint32_t step = 0; step < 120; ++step) {
            at.x += Swing * dt;
            Drive(world, paddle, at, float3{Swing, 0, 0});
            solver.Step(world, free_space);
        }
        std::println("a kinematic paddle mesh - one open quad - swung at {:g} m/s into a ball at rest:\n"
                     "  the ball ended at x={:8.4f} going {:8.4f} m/s, and the paddle is at {:8.4f}",
                     Swing, float(world.Poses[ball].Position.x), float(world.Velocities[ball].Linear.x), float(world.Poses[paddle].Position.x));
    }

    // Same balls, same drop, same hinge, and the only difference is whether the mesh has paddles to push on.
    std::println("a paddle wheel as an open mesh on a hinge, {} balls dropped onto it over {} steps.\n"
                 "  the bare hub is the same wheel with its paddles taken off, which is the control:", balls, steps);
    for (const uint32_t vanes : {paddles, 0u}) {
        Solver solver{context};
        World world{context};
        const Index shape = WheelMesh(world, 16, vanes);
        if (shape == NoIndex) return (void)std::println(stderr, "the wheel would not cook");
        // A floor under it, since the contact reach is unclamped by default and a ball falling for ever reaches hundreds of metres a second.
        // Such a ball stays within reach of the wheel from a hundred metres below and makes ghost pairs with it.
        // The sample has a floor anyway.
        AddGround(world);
        // With the brake a water wheel has.
        // KHR's WaterWheel drives its hinge with k = 0, c = 0.15, target 0, which is a damper alone.
        // Per-body angular damping is the same thing on the same body.
        // The wheel then reaches a rate the balls account for rather than winding up without limit.
        const auto wheel = world.AddBody({.Pose = At(float3{0, 0, 0}), .Shape = shape, .Mass = WheelMass(ArgF(args, 6, 20)), .Friction = Friction, .AngularDamping = ArgF(args, 7, 0.6f)});
        // A hinge about z: a joint to a shapeless zero-density body, with two angular axes locked and the third free.
        // The axes are body B's and B is upright, so the free axis is the world's z.
        const auto axle = world.AddBody({});
        world.AddJoint({.BodyA = wheel, .BodyB = axle, .At = float3{0, 0, 0}, .Angular = {AxisLocked, AxisLocked, AxisFree}});

        const Index ball_shape = world.AddShape({.Radius = BallRadius, .Kind = ShapeSphere});
        float turned = 0, fastest = 0;
        uint32_t dropped = 0, spun_at = 0;
        float4 was = world.Poses[wheel].Orientation;
        std::println("  {} paddles:", vanes);
        for (uint32_t step = 0; step < steps; ++step) {
            // One ball every twenty steps, onto the near side where the paddles face up.
            // The far side's paddles present their backs, which stop nothing, so a one-sided wheel turns one way.
            if (dropped < balls && step % 20 == 0) {
                world.AddBody({.Pose = At(float3{0.62f, 2.2f, 0.1f * (float(dropped % 3) - 1)}), .Shape = ball_shape, .Density = Density, .Friction = Friction});
                ++dropped;
            }
            solver.Step(world, settings);
            // Accumulated, since a quaternion difference only ever gives the short way round.
            const float4 now = world.Poses[wheel].Orientation;
            turned += RotationVector(QuatMul(now, QuatConjugate(was))).z;
            was = now;
            const float rate = std::abs(float(world.Velocities[wheel].Angular.z));
            if (rate > fastest) {
                fastest = rate;
                spun_at = step;
            }
            if (step % 100 != 99 && step != steps - 1) continue;
            std::println("    {:4} turned {:9.4f} rad  rate {:8.4f} rad/s  balls {:2}  rows {:3}  off its axle {:.6f} m",
                         step, turned, float(world.Velocities[wheel].Angular.z), dropped, ActiveContacts(world), simd::length(world.Poses[wheel].Position));
        }
        std::println("    -> {:9.4f} rad, {:6.2f} turns, fastest {:8.4f} rad/s at step {}",
                     turned, turned / (2 * std::numbers::pi_v<float>), fastest, spun_at);
    }

    // And what a moving mesh rests on.
    // Neither a plane nor a mesh presents a manifold, so the two meet with no contact and a moving mesh needs a box to land on.
    // That is the same thin box MeshEditor substitutes wherever a plane has to be finite or double-sided.
    {
        Solver solver{context};
        World world{context};
        // ShapeTypes' DynamicMesh: every face reachable from below is a front face, so no result here depends on which side of a surface a body is on.
        constexpr float MeshHalf = 0.4f;
        const Index shape = BoxMesh(world, MeshHalf);
        if (shape == NoIndex) return (void)std::println(stderr, "the mesh cube would not cook");
        AddGround(world);
        world.AddBody({.Pose = At(float3{6, -0.25f, 0}), .Shape = world.AddShape({.HalfExtents = {2, 0.25f, 2}, .Kind = ShapeBox}), .Density = 0, .Friction = Friction});
        const auto on_plane = world.AddBody({.Pose = At(float3{0, 3, 0}), .Shape = shape, .Mass = {{.Mass = 10, .Inertia = {1, 1, 1}}}, .Friction = Friction});
        const auto on_box = world.AddBody({.Pose = At(float3{6, 3, 0}), .Shape = shape, .Mass = {{.Mass = 10, .Inertia = {1, 1, 1}}}, .Friction = Friction});
        for (uint32_t step = 0; step < steps; ++step) solver.Step(world, settings);
        std::println("a {:g} m mesh cube dropped from 3 m onto a plane and onto a static box, both at y = 0:\n"
                     "  over the plane it is at y={:9.4f} - a plane presents no manifold and neither does a mesh, so\n"
                     "    the pair has no owner, there is no contact, and it falls. Give a moving mesh a box to land on\n"
                     "  over the box   it is at y={:9.4f}, against the {:g} its own half extent puts it at",
                     2 * MeshHalf, float(world.Poses[on_plane].Position.y), float(world.Poses[on_box].Position.y), MeshHalf);
    }
}

// The rows between one body and another, keyed as warm starting keys them, ordered by that key, with the normal force each carries.
// A pair at rest prints the same list every step, so a row renamed, moved to another triangle, or swapping its load shows here rather than as a wrong height.
// The separation each row began the step at is printed beside the force, since the two must agree.
std::string PairRows(const World &world, Index owner, Index against) {
    std::vector<std::tuple<Index, uint32_t, float, float>> rows;
    for (uint32_t slot = 0; slot < ContactsPerBody; ++slot) {
        const auto &contact = world.Contacts[owner * ContactsPerBody + slot];
        if (contact.Active && contact.BodyB == against) rows.emplace_back(contact.SubShape, contact.Feature, contact.Lambda[0], contact.C0[0]);
    }
    std::sort(rows.begin(), rows.end());
    std::string listed;
    for (const auto &[sub, feature, normal, separation] : rows)
        listed += std::format(" {}/{:x}: fn={:.1f} C0={:.4f}", sub == NoIndex ? std::string{"-"} : std::to_string(sub), feature, std::abs(normal), separation);
    return listed;
}

// A body at rest on a mesh the host gave a mass, beside the same body on the same mesh left as scenery.
// The mass is the only difference, so any divergence between the two is the cost of a mesh being a body.
// Both keep their height, so the report is the pair of churn, which must reach zero once settled, and the normal force per row, which must be steady.
void Perch(std::span<char *> args) {
    const uint32_t steps = Arg(args, 2, 600);
    StepSettings settings{.Iterations = Arg(args, 3, StepSettings{}.Iterations)};
    ApplyNoSleep(settings);
    const Reporting report{60, steps};
    const mtl::Context context;
    Solver solver{context};
    World moving{context}, scenery{context};

    // A cube as a mesh with a box on it, on a static slab, since a slab is the only thing a moving mesh can rest on.
    // The mesh weighs what the same cube of water would, so a body standing on it cannot push it aside.
    // The box is half the width of the face it stands on.
    constexpr float MeshHalf = 0.4f, TopHalf = 0.2f, SlabHalf = 0.25f;
    // A solid cube of side s about its own centre: m s^2 / 6 on every axis.
    constexpr float MeshMass = 512, MeshInertia = MeshMass * (2 * MeshHalf) * (2 * MeshHalf) / 6;
    constexpr AuthoredMass Weight{.Mass = MeshMass, .Inertia = {MeshInertia, MeshInertia, MeshInertia}};
    Index slab[2], mesh[2], box[2];
    for (uint32_t side = 0; side < 2; ++side) {
        World &world = side == 0 ? moving : scenery;
        const Index shape = BoxMesh(world, MeshHalf);
        if (shape == NoIndex) return (void)std::println(stderr, "the mesh cube would not cook");
        slab[side] = world.AddBody({.Pose = At(float3{0, -SlabHalf, 0}), .Shape = world.AddShape({.HalfExtents = {2, SlabHalf, 2}, .Kind = ShapeBox}), .Density = 0, .Friction = Friction});
        // The mesh is added first for the ownership case.
        // The convex side owns the pair despite its higher index, since a mesh's own thread returns before colliding anything.
        mesh[side] = world.AddBody({.Pose = At(float3{0, MeshHalf, 0}), .Shape = shape, .Mass = side == 0 ? std::optional<AuthoredMass>{Weight} : std::nullopt, .Friction = Friction});
        box[side] = Place(world, world.AddShape({.HalfExtents = {TopHalf, TopHalf, TopHalf}, .Kind = ShapeBox}), float3{0, 2 * MeshHalf + TopHalf + 0.05f, 0});
        if (slab[side] == NoIndex || mesh[side] == NoIndex || box[side] == NoIndex) return (void)std::println(stderr, "the perch would not build");
    }

    std::println("a {:g} m box set down on a {:g} m mesh cube, on a mesh the host gave {:g} kg and on the same\n"
                 "  mesh left as scenery. Both hold their height - the question is whether the rows holding it\n"
                 "  there are the same rows step to step, so churn must reach zero and the forces must hold",
                 2 * TopHalf, 2 * MeshHalf, MeshMass);
    Churner churn[2]{Churner{moving}, Churner{scenery}};
    for (uint32_t step = 0; step < steps; ++step) {
        solver.Step(moving, settings);
        solver.Step(scenery, settings);
        churn[0].Step(moving);
        churn[1].Step(scenery);
        if (!report(step)) continue;
        for (uint32_t side = 0; side < 2; ++side) {
            const World &world = side == 0 ? moving : scenery;
            const Index bodies[]{mesh[side], box[side]};
            // A cube is the same solid on any face, so heights alone cannot separate a settled pile from one that fell over.
            // The tilt shows that the mesh still stands as it was put down.
            const auto tilt = [&world](Index body) { return simd::length(RotationVector(world.Poses[body].Orientation)); };
            std::println("  {:7} {:4} churn={:4} asleep={}/2  box |v|={:8.5f} y={:7.4f} tilt={:8.5f}  mesh |v|={:8.5f} |w|={:8.5f} y={:7.4f} tilt={:8.5f}",
                         side == 0 ? "moving" : "scenery", step, churn[side].Take(), Asleep(world, bodies, settings),
                         simd::length(world.Velocities[box[side]].Linear), float(world.Poses[box[side]].Position.y), tilt(box[side]),
                         simd::length(world.Velocities[mesh[side]].Linear), simd::length(world.Velocities[mesh[side]].Angular),
                         float(world.Poses[mesh[side]].Position.y), tilt(mesh[side]));
            std::println("            box on mesh: {:2} rows{}\n            mesh on slab: {:2} rows{}",
                         ActiveContacts(world, box[side]), PairRows(world, box[side], mesh[side]),
                         ActiveContacts(world, slab[side]), PairRows(world, slab[side], mesh[side]));
        }
    }
}

constexpr float PlateHalf = 0.5f, PlateHalfThickness = 0.05f, RockerBall = 0.25f;
// Odd, so no sample sits on the plate's axis.
// The four innermost samples then share one height and the hull's lowest face is a small horizontal facet, where a body set down exactly level starts.
constexpr uint32_t CrownCells = 5;

// A square plate, flat on top and crowned underneath.
// The underside rises as the square of the radius out to `crown` at the rim, which makes the solid above it convex and every sample a hull corner.
// Sampled rather than analytic, since the facets themselves are the subject.
std::vector<float3> CrownedPlate(uint32_t cells, float crown) {
    std::vector<float3> points;
    for (const float x : {-PlateHalf, PlateHalf})
        for (const float z : {-PlateHalf, PlateHalf}) points.push_back(float3{x, PlateHalfThickness, z});
    for (uint32_t i = 0; i <= cells; ++i)
        for (uint32_t j = 0; j <= cells; ++j) {
            const float x = PlateHalf * (2 * float(i) / float(cells) - 1), z = PlateHalf * (2 * float(j) / float(cells) - 1);
            const float rise = crown * (x * x + z * z) / (PlateHalf * PlateHalf);
            points.push_back(float3{x, rise - PlateHalfThickness - crown, z});
        }
    return points;
}

// How far a body has turned away from upright, in radians.
// The length of its rotation vector will not do, since the cook may spin a square plate about its own normal while diagonalizing and that is not a tilt.
float Tilt(const World &world, Index body) {
    const float3 up = Rotate(world.Poses[body].Orientation, float3{0, 1, 0});
    return std::acos(std::clamp(float(up.y), -1.f, 1.f));
}

// Where the load sits: contact points in world space, averaged by the normal force each carries.
// A body rolling on a curve moves this across the surface, and one at rest keeps it in place.
std::pair<float3, float> ContactCentroid(const World &world, Index body) {
    float3 weighted{0, 0, 0};
    float held = 0;
    for (uint32_t slot = 0; slot < ContactsPerBody; ++slot) {
        const auto &contact = world.Contacts[body * ContactsPerBody + slot];
        if (!contact.Active) continue;
        const bool mine_is_a = contact.BodyA == body;
        const float3 anchor = mine_is_a ? contact.AnchorA : contact.AnchorB;
        const float force = std::abs(contact.Lambda[0]);
        weighted += force * WorldPoint(world.Poses[body], anchor);
        held += force;
    }
    return {held > 0 ? weighted / held : float3{0, 0, 0}, held};
}

// A body's contact set measured over the whole interval rather than sampled at the end of it.
// A rock faster than the reporting period reads as stillness wherever the samples land.
struct Watch {
    float3 Where{0, 0, 0};
    float Wander{}, Low = INFINITY, High = 0, Fastest = 0, Turned = 0;
    bool Started = false;

    void Step(const World &world, Index body) {
        const auto [centre, held] = ContactCentroid(world, body);
        if (held > 0) {
            if (Started) Wander = std::max(Wander, simd::length(centre - Where));
            Where = centre;
            Started = true;
        }
        Low = std::min(Low, held);
        High = std::max(High, held);
        Fastest = std::max(Fastest, simd::length(world.Velocities[body].Linear));
        Turned = std::max(Turned, simd::length(world.Velocities[body].Angular));
    }
    // The interval's figures, then a fresh interval, as with the raft's churn.
    Watch Take() { return std::exchange(*this, Watch{.Where = Where, .Started = Started}); }
};

// A plate whose underside is a shallow convex crown, set down exactly level on the plane, beside a sphere as the control.
// The sphere rests on one point and sleeps, so any divergence between the two is the cost of the crown.
//
// The measurement is which corners a plane's manifold keeps.
// The plate stands on the apex facet's four corners, and the next ring is a couple of millimetres behind them, inside a step's contact reach.
// More corners reach the plane than a manifold has room for, and they are at different depths.
// Keeping the wrong four leaves the plate on one point, with its rocking resisted only by I/h^2.
// That is Sec. 3.4's weak-force case reached through the geometry rather than through the penalty.
void Rocker(std::span<char *> args) {
    const uint32_t steps = Arg(args, 2, 900);
    StepSettings settings{.Iterations = Arg(args, 3, StepSettings{}.Iterations)};
    const float degrees = ArgF(args, 4, 3), shove = ArgF(args, 5, 1);
    const uint32_t shoved = steps / 3;
    ApplyNoSleep(settings);
    const Reporting report{60, steps};
    const mtl::Context context;
    Solver solver{context};
    World world{context};

    // `degrees` is the slope the underside reaches at the rim, and a paraboloid's slope there is twice its rise over its half width.
    const float crown = PlateHalf * std::tan(degrees * std::numbers::pi_v<float> / 180) / 2;
    Pose frame;
    const Index shape = world.AddHull(CrownedPlate(CrownCells, crown), &frame);
    if (shape == NoIndex) return (void)std::println(stderr, "the crowned plate would not cook");
    AddGround(world);
    // Apex exactly on the plane, measured through the frame the cook returned rather than the authored points it moved onto the centre of mass.
    const Index plate = Place(world, shape, float3{0, PlateHalfThickness + crown + float(frame.Position.y), 0});
    const Index ball = Place(world, world.AddShape({.Radius = RockerBall, .Kind = ShapeSphere}), float3{3, RockerBall, 0});
    if (plate == NoIndex || ball == NoIndex) return (void)std::println(stderr, "the rocker would not build");

    std::println("a {:g} m plate crowned {:.1f} mm under {:g} degrees of rim slope, set down level, beside a\n"
                 "  {:g} m sphere on the same plane. A sphere rests on one point and sleeps - the question is\n"
                 "  whether a faceted crown does, so each line is over the whole interval and not its last step",
                 2 * PlateHalf, 1000 * crown, degrees, 2 * RockerBall);
    Churner churner{world};
    Watch plate_watch, ball_watch;
    for (uint32_t step = 0; step < steps; ++step) {
        // A shove once the plate has had a third of the run to settle.
        // A body that cannot be disturbed and come back to rest has stopped being simulated rather than settled.
        // Rolled about x, so the plate rocks on the crown rather than sliding along the plane.
        if (step == shoved) {
            world.Velocities[plate].Angular = {shove, 0, 0};
            world.Wake(plate);
            std::println("  -- shoved at {} with {:g} rad/s about x", shoved, shove);
        }
        solver.Step(world, settings);
        churner.Step(world);
        plate_watch.Step(world, plate);
        ball_watch.Step(world, ball);
        if (!report(step)) continue;
        const Watch p = plate_watch.Take(), b = ball_watch.Take();
        std::println("  {:4} plate rows={:2} |v|={:8.5f} |w|={:8.5f} tilt={:8.5f} wander={:8.5f} fn=[{:8.1f},{:8.1f}] churn={:3} asleep={}",
                     step, ActiveContacts(world, plate), p.Fastest, p.Turned, Tilt(world, plate), p.Wander,
                     p.Low, p.High, churner.Take(), Asleep(world, std::span{&plate, 1}, settings));
        std::println("       ball  rows={:2} |v|={:8.5f} |w|={:8.5f} tilt={:8.5f} wander={:8.5f} fn=[{:8.1f},{:8.1f}]           asleep={}",
                     ActiveContacts(world, ball), b.Fastest, b.Turned, Tilt(world, ball), b.Wander, b.Low, b.High,
                     Asleep(world, std::span{&ball, 1}, settings));
    }
}

// The effect of a host pose write on a sleeping world, with World::Wake and without it.
// Both outcomes are the contract.
// An authored edit changes the world and must wake it, and a cache restore must leave it asleep, or scrubbing a timeline would restart every settled pile.
//
// Two writes, the two MeshEditor makes.
// A static slab under a sleeping stack, moved out from under it: nothing about a slab is integrated, so no velocity appears for a wake to spread along.
// And a sleeping box teleported into another, whose partner is asleep and has no contact against it to be woken through.
void Teleport(std::span<char *> args) {
    const uint32_t boxes = Arg(args, 2, 3), settle = Arg(args, 3, 300), after = Arg(args, 4, 300);
    const StepSettings settings{};
    constexpr float SlabTop = 2, Aside = 5, Apart = 3, Into = 0.8f;
    const mtl::Context context;

    // `wake` is the only difference between the two runs.
    struct Result {
        float StackY{}, StackX{}, PushedX{}, MovedX{};
        uint32_t StackAsleep{}, PushedAsleep{};
    };
    const auto run = [&](bool wake) {
        Solver solver{context};
        World world{context};
        const auto shape = world.AddShape(UnitBox);
        AddGround(world);
        // In the air, since a slab on the ground would leave the stack where it already was.
        const auto slab = world.AddBody({.Pose = At(float3{0, SlabTop - 0.25f, 0}), .Shape = world.AddShape({.HalfExtents = {2, 0.25f, 2}, .Kind = ShapeBox}), .Density = 0, .Friction = Friction});
        std::vector<Index> stack;
        for (uint32_t i = 0; i < boxes; ++i) stack.push_back(Place(world, shape, float3{0, SlabTop + Half + 1.02f * float(i), 0}));
        // Two boxes on the ground, well apart, one about to arrive on top of the other.
        const auto pushed = Place(world, shape, float3{Apart, Half, 0});
        const auto moved = Place(world, shape, float3{-Apart, Half, 0});

        for (uint32_t step = 0; step < settle; ++step) solver.Step(world, settings);
        const uint32_t slept = Asleep(world, stack, settings);

        world.Poses[slab].Position.x += Aside;
        world.Poses[moved].Position = {Apart - 1 + Into, Half, 0};
        if (wake) {
            world.Wake(slab);
            world.Wake(moved);
        }
        for (uint32_t step = 0; step < after; ++step) solver.Step(world, settings);
        std::println("  settled with {}/{} of the stack asleep", slept, boxes);
        return Result{
            .StackY = float(world.Poses[stack.front()].Position.y),
            .StackX = float(world.Poses[stack.front()].Position.x),
            .PushedX = float(world.Poses[pushed].Position.x),
            .MovedX = float(world.Poses[moved].Position.x),
            .StackAsleep = Asleep(world, stack, settings),
            .PushedAsleep = Asleep(world, std::span{&pushed, 1}, settings),
        };
    };

    std::println("a stack of {} asleep on a static slab, and two boxes asleep on the ground {} m apart.\n"
                 "  the host moves the slab {} m sideways and drops the far box {} m into the near one",
                 boxes, 2 * Apart, Aside, Into);
    std::println("with World::Wake:");
    const Result woken = run(true);
    std::println("without it, which is what a cache restore wants:");
    const Result left = run(false);
    // Where the bottom box belongs: on the plane once the slab has gone, or still at the slab's top face when nothing woke it.
    std::println("  bottom of the stack:  woken y={:7.4f} x={:7.4f} asleep {}/{}  |  left alone y={:7.4f} x={:7.4f} asleep {}/{}",
                 woken.StackY, woken.StackX, woken.StackAsleep, boxes, left.StackY, left.StackX, left.StackAsleep, boxes);
    std::println("  the box moved into another: woken it ended at x={:7.4f} having pushed the other to {:7.4f}"
                 "  |  left alone {:7.4f} and {:7.4f}",
                 woken.MovedX, woken.PushedX, left.MovedX, left.PushedX);
    std::println("  the stack falls {:.4f} m when woken and {:.4f} m when not, and the box it landed on moves {:.4f} m against {:.4f} m",
                 SlabTop + Half - woken.StackY, SlabTop + Half - left.StackY, woken.PushedX - Apart, left.PushedX - Apart);
}

void Bullet(std::span<char *> args) {
    const float thickness = ArgF(args, 2, 0.02f);
    constexpr float Size = 0.05f, From = -1, Reach = 2; // the shot's half extent, where it starts, and the wall's half extent
    // Fired level, in free space, since the measurement is whether a body ends up on the far side of something.
    // Gravity would only add a fall, with a slow shot sliding down the wall and off the bottom.
    // Zero gravity costs the reach its gravity term, a millimetre and a half against metres of travel.
    const StepSettings settings{.Gravity = {0, 0, 0}};
    const mtl::Context context; // one for the thirty-three worlds below

    enum Wall : uint32_t { NoWall,
                           BoxWall,
                           MeshWall };
    // One shot: where it rested, how far past the wall's near face it reached, and what its run refused.
    // The two walls fail differently.
    // A body crossing a box comes out of the far side, and one crossing a mesh is gone.
    struct Shot {
        float3 Rest{};
        float Deepest{}, Refused{}, Kept{};
        uint32_t Crossed{};
    };
    const auto fire = [&](float3 velocity, float3 from, Wall against, float clamped = INFINITY) {
        Solver solver{context};
        World world{context};
        if (against != NoWall) {
            const Index wall = against == MeshWall ? WallMesh(world, Reach) : world.AddShape({.HalfExtents = {thickness / 2, Reach, Reach}, .Kind = ShapeBox});
            if (wall == NoIndex) return Shot{};
            world.AddBody({.Shape = wall, .Density = 0});
        }
        const auto shot = world.AddBody({.Pose = At(from), .Velocity = {.Linear = velocity}, .Shape = world.AddShape({.HalfExtents = {Size, Size, Size}, .Kind = ShapeBox}), .Density = Density});
        // Enough to cover the metres either side of the wall, and short enough that a slow shot is dropped soon after it comes to rest.
        const float speed = simd::length(velocity);
        const uint32_t steps = std::min(3000u, uint32_t(3 / (speed * settings.DeltaTime)) + 60);
        StepSettings shot_settings = settings;
        shot_settings.MaxContactReach = clamped;
        // The wall's near face: its own front for a box, the surface itself for a mesh.
        const float face = against == BoxWall ? -thickness / 2 : 0;
        Shot out{.Rest = from};
        for (uint32_t step = 0; step < steps; ++step) {
            solver.Step(world, shot_settings);
            const float3 at = world.Poses[shot].Position;
            out.Rest = at;
            out.Kept = simd::length(world.Velocities[shot].Linear) / speed;
            out.Deepest = std::max(out.Deepest, at.x + Size - face);
            out.Refused += float(world.ContactRefusals[shot]);
            // Out the far side, the one unrecoverable outcome.
            if (against != NoWall && at.x - Size > thickness / 2) ++out.Crossed;
        }
        return out;
    };

    std::println("a {:g} m box fired at a {:g} m wall from {:g} m, box wall and mesh wall. It must never end behind either.", 2 * Size, thickness, -From);
    std::println("  deepest is how far its leading face ever got past the wall's, so a contact margin is where it rests");
    for (const float speed : {5.f, 20.f, 60.f, 120.f, 250.f, 500.f}) {
        const float3 velocity{speed, 0, 0}, from{From, 0, 0};
        const auto box = fire(velocity, from, BoxWall), mesh = fire(velocity, from, MeshWall);
        const auto pinned = fire(velocity, from, MeshWall, ArgF(args, 4, 0.05f));
        std::println("{:6.0f} m/s ({:7.3f} m a step)  box: rest {:8.4f} deepest {:8.5f} refused {:3.0f} crossed {:3}"
                     "  |  mesh: rest {:8.4f} deepest {:8.5f} crossed {:3}  |  clamped: crossed {:3}",
                     speed, speed * settings.DeltaTime, box.Rest.x, box.Deepest, box.Refused, box.Crossed, mesh.Rest.x, mesh.Deepest, mesh.Crossed, pinned.Crossed);
    }

    // And the cost of that reach.
    // This shot starts squarely in front of the wall and rises fast enough to clear the top edge before it arrives.
    // At the pose every step begins from, it is closing on a face it never meets, which is a ghost collision.
    // A shot passing a wall to the side produces nothing, since points are clipped into the face that made them rather than to an infinite plane.
    // The report is the speed kept, since a ghost takes speed rather than position, beside the same shot with the reach clamped.
    // Read the cost of the clamp above before using it.
    const float clearance = ArgF(args, 3, 0.05f);
    const float clamped = ArgF(args, 4, 0.05f);
    std::println("clearing the {:g} m corner by {:g} m on the way past, unclamped and with the reach clamped to {:g} m", Reach, clearance, clamped);
    for (const float speed : {20.f, 120.f, 500.f}) {
        // Rising just fast enough over the metre of approach to pass the corner by the clearance.
        const float3 velocity{speed, speed * (Reach + Size + clearance) / -From, 0}, from{From, 0, 0};
        const auto clear = fire(velocity, from, NoWall);
        std::println("{:6.0f} m/s  box keeps {:6.2f}% of its speed, clamped {:6.2f}%  |  mesh {:6.2f}%, clamped {:6.2f}%"
                     "  |  missing it entirely keeps {:6.2f}%",
                     speed, 100 * fire(velocity, from, BoxWall).Kept, 100 * fire(velocity, from, BoxWall, clamped).Kept, 100 * fire(velocity, from, MeshWall).Kept, 100 * fire(velocity, from, MeshWall, clamped).Kept, 100 * clear.Kept);
    }
}

// A box jointed at its own centre to a fixed point, so the joint is an axle.
// The caller's frame and modes then determine the twist axis entirely.
struct Spun {
    Index Body, Joint; // the joint too, since the angle a hinge has turned through is read from the joint
    float Inertia, Mass;
};

Spun SpinOnAxle(World &world, JointDesc joint, float3 spin) {
    const auto shape = world.AddShape(UnitBox);
    const auto axle = world.AddBody({}); // no shape, so no mass: a fixed point to turn about
    const auto wheel = world.AddBody({.Pose = At(float3{0, 0, 0}), .Velocity = {.Angular = spin}, .Shape = shape, .Density = Density});
    joint.BodyA = wheel;
    joint.BodyB = axle;
    return {wheel, world.AddJoint(joint), 1 / world.Masses[wheel].InvInertiaLocal[2], 1 / world.Masses[wheel].InvMass};
}

// A pendulum on a hinge whose free axis is the joint frame's z, an axis of neither body.
// The frame is turned about x and again about y, and the pivot it hangs from is turned somewhere else entirely.
// Released at `release` from the bottom of its swing, so the closed form is the flat pendulum's energy with the in-plane part of gravity in place of gravity.
//
// It swings twice `release` from the pose the joint was made in, so past 90 degrees it is past the half turn a single rotation vector can represent.
// Swing-twist decomposition covers that: the two locked rows read the swing, which stays small on a working hinge, and the twist is accumulated.
struct FramedHinge {
    Index Arm;
    float3 Axis; // the world axis the hinge is free about, the swing plane's normal
    float Inertia, Peak; // about the pivot, and the speed energy gives at the bottom of the swing
};

FramedHinge HangOnFrame(World &world, float4 frame, float4 pivot_turn, float3 gravity, float distance, float release) {
    const float3 axis = Rotate(frame, float3{0, 0, 1});
    const float3 in_plane = gravity - simd::dot(gravity, axis) * axis;
    const float3 down = simd::normalize(in_plane); // where the arm hangs, with `along` level and square to it
    const float3 along = simd::normalize(simd::cross(axis, gravity));
    const float3 start = distance * (std::cos(release) * down + std::sin(release) * along);
    const auto shape = world.AddShape(UnitBox);
    const auto pivot = world.AddBody({.Pose = At(float3{0, 0, 0}, pivot_turn)}); // no shape, so no mass
    const auto arm = world.AddBody({.Pose = At(start), .Shape = shape, .Density = Density});
    world.AddJoint({.BodyA = arm, .BodyB = pivot, .At = {0, 0, 0}, .Frame = frame, .Angular = {AxisLocked, AxisLocked, AxisFree}});
    const float mass = 1 / world.Masses[arm].InvMass;
    // Parallel axis, about the pivot rather than about the arm's own centre.
    const float inertia = 1 / world.Masses[arm].InvInertiaLocal[2] + mass * distance * distance;
    // All of the height it gives up becomes rotation: 1/2 I w^2 = m g . (bottom - start).
    return {arm, axis, inertia, std::sqrt(2 * mass * simd::dot(gravity, distance * down - start) / inertia)};
}

void JointFrame(std::span<char *> args) {
    const uint32_t steps = Arg(args, 2, 600);
    StepSettings settings{.Iterations = Arg(args, 3, StepSettings{}.Iterations)};
    ApplyNoSleep(settings);
    constexpr float Distance = 1; // a metre of arm
    constexpr float Degrees = std::numbers::pi_v<float> / 180;
    // Zero is the control, where the frame is the world's, the frame a joint made without one uses.
    const float turn = ArgF(args, 4, 30) * Degrees, release = ArgF(args, 5, 60) * Degrees;
    const float4 frame = QuatMul(QuatFromRotationVector(float3{0, turn, 0}), QuatFromRotationVector(float3{turn, 0, 0}));
    const mtl::Context context;
    Solver solver{context};

    // Two pivots turned to arbitrary orientations.
    // The frame is the joint's own, so the two must swing identically, where a hinge using body B's axes instead would diverge.
    const float4 turns[]{float4{0, 0, 0, 1}, QuatFromRotationVector(float3{0.7f, -1.3f, 0.4f})};
    World worlds[]{World{context}, World{context}};
    FramedHinge hinges[]{HangOnFrame(worlds[0], frame, turns[0], settings.Gravity, Distance, release),
                         HangOnFrame(worlds[1], frame, turns[1], settings.Gravity, Distance, release)};

    std::println("a hinge free about ({:.3f} {:.3f} {:.3f}), which is no axis of either body, on two pivots turned differently.",
                 float(hinges[0].Axis.x), float(hinges[0].Axis.y), float(hinges[0].Axis.z));
    std::println("  released {:.0f} degrees off the bottom, energy says it reaches {:.4f} rad/s there. Backwards Euler dissipates,",
                 release / Degrees, hinges[0].Peak);
    std::println("  so what it reaches is about the 93% a flat pendulum reaches at this step - and past 90 the error's own seam is in the way");
    const Reporting report{60, steps};
    float peak = 0, out_of_plane = 0, apart = 0;
    for (uint32_t step = 0; step < steps; ++step) {
        for (uint32_t i = 0; i < 2; ++i) solver.Step(worlds[i], settings);
        for (uint32_t i = 0; i < 2; ++i) {
            const float3 at = worlds[i].Poses[hinges[i].Arm].Position;
            peak = std::max(peak, simd::length(worlds[i].Velocities[hinges[i].Arm].Angular));
            out_of_plane = std::max(out_of_plane, std::abs(simd::dot(at, hinges[i].Axis)));
        }
        apart = std::max(apart, simd::distance(worlds[0].Poses[hinges[0].Arm].Position, worlds[1].Poses[hinges[1].Arm].Position));
        if (!report(step)) continue;
        const float3 at = worlds[0].Poses[hinges[0].Arm].Position;
        std::println("{:4} |w| {:8.4f} of {:7.4f} ({:6.1f}%)  out of plane {:9.6f}  pivots apart {:9.6f}  arm ({:6.3f} {:6.3f} {:6.3f})",
                     step, peak, hinges[0].Peak, 100 * peak / hinges[0].Peak, out_of_plane, apart, float(at.x), float(at.y), float(at.z));
    }
    std::println("peak {:.4f} rad/s is {:.1f}% of the closed form, it left the plane by {:.6f} m, and the two pivots never diverged past {:.6f} m",
                 peak, 100 * peak / hinges[0].Peak, out_of_plane, apart);

    // The other half of a frame: a joint whose two ends do not coincide when it is made, so the error is present before any load.
    // A hard row ignores the error a step began with, and stabilization removes it after velocity has been read.
    // The cost to watch for is a jump in the speed the two ends close at, which the pair is measured on.
    std::println("\ntwo boxes hung side by side off a fixed point, jointed to each other with their ends apart");
    for (const float offset : {0.f, 0.01f}) {
        World world{context};
        const auto shape = world.AddShape(UnitBox);
        const auto anchor = world.AddBody({}); // no shape, so no mass: a fixed point to hang from
        const auto left = world.AddBody({.Pose = At(float3{-1, 0, 0}), .Shape = shape, .Density = Density});
        const auto right = world.AddBody({.Pose = At(float3{1, 0, 0}), .Shape = shape, .Density = Density});
        world.AddJoint({.BodyA = left, .BodyB = anchor, .At = {-1, 1, 0}});
        world.AddJoint({.BodyA = left, .BodyB = right, .At = {0, 0, 0}, .AtB = float3{offset, 0, 0}});
        float worst_closing = 0, gap = 0;
        uint32_t closed = ~0u;
        for (uint32_t step = 0; step < steps; ++step) {
            const float was = gap;
            solver.Step(world, settings);
            const Pose &a = world.Poses[left], &b = world.Poses[right];
            const Joint &joint = world.Joints[1];
            gap = simd::distance(WorldPoint(a, joint.AnchorA), WorldPoint(b, joint.AnchorB));
            if (step > 0) worst_closing = std::max(worst_closing, std::abs(gap - was) / settings.DeltaTime);
            if (closed == ~0u && gap < 1e-4f) closed = step;
        }
        const auto when = closed == ~0u ? std::string{"never"} : std::to_string(closed);
        std::println("  ends {:.3f} m apart at rest: {:9.6f} m apart after {} steps, under 0.1 mm at step {}, fastest they ever closed {:8.5f} m/s",
                     offset, gap, steps, when, worst_closing);
    }

    // The twist itself, on the same tilted axle, revolution after revolution.
    // The joint accumulates the angle turned through rather than reading it off the relative rotation, so this number goes on climbing.
    // Free space, since a box jointed at its own centre has no weight about it.
    const StepSettings spinning{.Gravity = {0, 0, 0}, .Iterations = settings.Iterations};
    constexpr float Turns = 20;
    const float wanted = Turns * 2 * std::numbers::pi_v<float>;
    const float speed = wanted / (float(steps) * spinning.DeltaTime);
    {
        World world{context};
        const auto wheel = SpinOnAxle(world, {.Frame = frame,
                                              .Angular = {AxisLocked, AxisLocked, AxisDriven},
                                              .MotorSpeed = {0, 0, speed},
                                              .MotorMaxTorque = {0, 0, 1e6f}},
                                      float3{0, 0, 0});
        const float3 axle = Rotate(frame, float3{0, 0, 1});
        std::println("\na wheel driven at {:.4f} rad/s about that same axle, {:g} turns in {} steps:", speed, Turns, steps);
        float off_axis = 0;
        for (uint32_t step = 0; step < steps; ++step) {
            solver.Step(world, spinning);
            const float3 rate = world.Velocities[wheel.Body].Angular;
            off_axis = std::max(off_axis, simd::length(rate - simd::dot(rate, axle) * axle));
            if (!report(step)) continue;
            const float turned = world.Joints[wheel.Joint].Twist;
            std::println("  {:4} turned {:9.4f} rad ({:7.3f} turns)  |w| {:8.4f}  off the axle {:9.6f} rad/s",
                         step, turned, turned / (2 * std::numbers::pi_v<float>),
                         simd::length(world.Velocities[wheel.Body].Angular), off_axis);
        }
        std::println("  -> {:.4f} rad of the {:.4f} asked for ({:.2f}%), and it never turned more than {:.6f} rad/s off the axle",
                     world.Joints[wheel.Joint].Twist, wanted, 100 * world.Joints[wheel.Joint].Twist / wanted, off_axis);
    }

    // The second use of an unwrapped angle: a stop past the half turn.
    // A wrapped angle reads 200 degrees as -160 and would drive the row towards the low stop, so where the wheel comes to rest is the measurement.
    {
        constexpr float Stop = 200 * Degrees;
        World world{context};
        const auto wheel = SpinOnAxle(world, {.Frame = frame,
                                              .Angular = {AxisLocked, AxisLocked, AxisLimited},
                                              .LimitLow = {0, 0, -0.5f},
                                              .LimitHigh = {0, 0, Stop}},
                                      Rotate(frame, float3{0, 0, 4}));
        std::println("\na wheel spun at 4 rad/s into a stop at {:.0f} degrees, which is past the seam a rotation vector has:", Stop / Degrees);
        for (uint32_t step = 0; step < steps; ++step) {
            solver.Step(world, spinning);
            if (!report(step)) continue;
            const float turned = world.Joints[wheel.Joint].Twist;
            std::println("  {:4} at {:9.4f} rad ({:7.2f} degrees)  |w| {:8.5f}",
                         step, turned, turned / Degrees, simd::length(world.Velocities[wheel.Body].Angular));
        }
        std::println("  -> rests at {:.2f} degrees, of the {:.0f} its stop is at", world.Joints[wheel.Joint].Twist / Degrees, Stop / Degrees);
    }
}

// The linear axes: the same five modes the angular axes have, in metres and newtons.
// A slider dropped onto its lower stop, a body free in a box with all three axes limited, and a velocity drive against the force bounding it.
// Then a position drive arriving at its offset and returning after being moved off, and KHR's offset lock, min == max at something other than zero.
//
// Watch the position drive after the host moves it.
// A hard positioned row spreads the error a step begins with by alpha and stabilization closes it.
// The box therefore walks back to its target at Force h^2 / m a step and reports no velocity while doing it.
//
// Every case is a box jointed to a fixed point at its own centre, so a row's value is the box's own displacement.
// The closed forms are then its mass and the force on it and nothing else.
struct SliderCase {
    Index Box;
    float Mass;
};

SliderCase MakeSlider(World &world, JointDesc joint, float3 at = {0, 0, 0}) {
    const auto shape = world.AddShape(UnitBox);
    const auto anchor = world.AddBody({}); // no shape, so no mass and no contacts: a fixed point
    const auto box = world.AddBody({.Pose = At(at), .Shape = shape, .Density = Density});
    joint.BodyA = box;
    joint.BodyB = anchor;
    world.AddJoint(joint);
    return {box, 1 / world.Masses[box].InvMass};
}

void Slider(std::span<char *> args) {
    const uint32_t steps = Arg(args, 2, 600);
    StepSettings settings{.Iterations = Arg(args, 3, StepSettings{}.Iterations)};
    ApplyNoSleep(settings);
    const Reporting report{60, steps};
    const mtl::Context context;
    Solver solver{context};

    constexpr float Stop = 0.5f, Reach = 0.25f, Speed = 2, Force = 3000, Target = 0.8f, Offset = 1;

    const auto run = [&](const char *what, JointDesc joint, float3 at, const StepSettings &use, float3 move_to, uint32_t move_at) {
        World world{context};
        const auto slider = MakeSlider(world, joint, at);
        std::println("{}", what);
        for (uint32_t step = 0; step < steps; ++step) {
            if (step == move_at) { // the host putting it somewhere else
                world.Poses[slider.Box].Position = move_to;
                world.Wake(slider.Box);
            }
            solver.Step(world, use);
            if (!report(step)) continue;
            const float3 at_now = world.Poses[slider.Box].Position, v = world.Velocities[slider.Box].Linear;
            std::println("  {:4} at ({:8.5f} {:8.5f} {:8.5f})  v ({:8.5f} {:8.5f} {:8.5f})",
                         step, float(at_now.x), float(at_now.y), float(at_now.z), float(v.x), float(v.y), float(v.z));
        }
        return std::pair{world.Poses[slider.Box].Position, world.Velocities[slider.Box].Linear};
    };

    const auto [dropped, dropped_v] = run("a slider free along y between +/- 0.5, dropped from its zero onto the lower stop:",
                                          {.Angular = {AxisLocked, AxisLocked, AxisLocked},
                                           .Linear = {AxisLocked, AxisLimited, AxisLocked},
                                           .LinearLimitLow = {0, -Stop, 0},
                                           .LinearLimitHigh = {0, Stop, 0}},
                                          float3{0, 0, 0}, settings, float3{}, ~0u);
    std::println("  -> rests at y {:.6f} against the stop at {:g}, {:.6f} m under it, going {:.6f} m/s",
                 float(dropped.y), -Stop, -Stop - float(dropped.y), simd::length(dropped_v));

    const auto boxed = run("free in a box, all three axes limited to +/- 0.25, gravity pointing into one corner:",
                           {.Angular = {AxisLocked, AxisLocked, AxisLocked},
                            .Linear = {AxisLimited, AxisLimited, AxisLimited},
                            .LinearLimitLow = {-Reach, -Reach, -Reach},
                            .LinearLimitHigh = {Reach, Reach, Reach}},
                           float3{0, 0, 0}, {.Gravity = {-5, -8, -3}, .Iterations = settings.Iterations}, float3{}, ~0u)
                          .first;
    std::println("  -> rests at ({:.6f} {:.6f} {:.6f}), against the corner at ({:g} {:g} {:g})",
                 float(boxed.x), float(boxed.y), float(boxed.z), -Reach, -Reach, -Reach);

    const StepSettings free_space{.Gravity = {0, 0, 0}, .Iterations = settings.Iterations};
    const auto driven = run("a velocity drive along x to 2 m/s, held to 3000 N, in free space:",
                            {.Linear = {AxisDriven, AxisLocked, AxisLocked},
                             .LinearMotorSpeed = {Speed, 0, 0},
                             .LinearMotorMaxForce = {Force, 0, 0}},
                            float3{0, 0, 0}, free_space, float3{}, ~0u)
                           .second;
    World sizing{context};
    const float mass = MakeSlider(sizing, {}).Mass;
    std::println("  -> {:.6f} m/s, of the {:g} asked for. F/m is {:.4f} m/s^2, so it took {:.3f} s to arrive",
                 float(driven.x), Speed, Force / mass, Speed * mass / Force);

    const auto placed = run("a position drive to y = 0.8, moved to y = -1.5 halfway through:",
                            {.Angular = {AxisLocked, AxisLocked, AxisLocked},
                             .Linear = {AxisLocked, AxisPositioned, AxisLocked},
                             .LinearMotorTarget = {0, Target, 0},
                             .LinearMotorMaxForce = {0, 3e4f, 0}},
                            float3{0, 0, 0}, settings, float3{0, -1.5f, 0}, steps / 2)
                           .first;
    std::println("  -> back at y {:.6f}, of the {:g} it holds", float(placed.y), Target);

    const auto locked = run("an axis locked at an offset - KHR's min == max == 1.0, made a metre out:",
                            {.At = {Offset, 0, 0},
                             .AtB = float3{0, 0, 0},
                             .Angular = {AxisLocked, AxisLocked, AxisLocked},
                             .Linear = {AxisPositioned, AxisLocked, AxisLocked},
                             .LinearMotorTarget = {Offset, 0, 0},
                             .LinearMotorMaxForce = {INFINITY, 0, 0}},
                            float3{Offset, 0, 0}, settings, float3{}, ~0u)
                           .first;
    std::println("  -> held at x {:.6f}, of the {:g} it was made at", float(locked.x), Offset);
}

// The damping coefficient beside each stiffness, the other half of KHR's k (xT - x) + c (vT - v).
// A viscous force is -c times the row's rate, and backwards Euler takes that rate at the end of the step.
// It therefore enters Eq. 7 as a second force of stiffness c/h on how far the row moved.
// Three consequences follow, each reported against its closed form:
//
//   A wheel left to a k = 0, c brake decays as w0 exp(-c t / I).
//   It is checked against the geometric w0 (1 + c h / I)^-n that backwards Euler integrates.
//   The couple of percent between them at c t / I of two is the integrator's gap rather than the solve's.
//
//   A k = 0, c drive approaches its target as w_T (1 - exp(-c t / I)) rather than arriving inside one step.
//   A stiff bounded motor arrives in one step, and most of the sample drives are authored that way.
//
//   A spring and damper across a linear axis settle without overshoot once c reaches 2 sqrt(k m).
//   The same spring with no damper rings for as long as it is watched.
void Brake(std::span<char *> args) {
    const uint32_t steps = Arg(args, 2, 300);
    StepSettings settings{.Gravity = {0, 0, 0}, .Iterations = Arg(args, 3, StepSettings{}.Iterations)};
    ApplyNoSleep(settings);
    const Reporting report{30, steps};
    const mtl::Context context;
    Solver solver{context};
    constexpr float Spin = 10;
    const float damping = ArgF(args, 4, 166.667f); // one second of time constant on a unit box

    // A hinge free about z but for the brake across it, spun up and let go.
    {
        World world{context};
        const auto wheel = SpinOnAxle(world, {.Angular = {AxisLocked, AxisLocked, AxisDriven},
                                              .MotorMaxTorque = {0, 0, INFINITY},
                                              .AngularStiffness = {INFINITY, INFINITY, 0},
                                              .AngularDamping = {0, 0, damping}},
                                      float3{0, 0, Spin});
        const float per_step = 1 + damping * settings.DeltaTime / wheel.Inertia;
        std::println("a wheel of inertia {:.4f} spun to {:g} rad/s against a k = 0, c = {:g} brake. Time constant I/c is {:.4f} s",
                     wheel.Inertia, Spin, damping, wheel.Inertia / damping);
        std::println("  step   measured    geometric   exponential   measured/geometric");
        for (uint32_t step = 0; step < steps; ++step) {
            solver.Step(world, settings);
            if (!report(step)) continue;
            const float elapsed = float(step + 1) * settings.DeltaTime;
            const float geometric = Spin / std::pow(per_step, float(step + 1));
            const float exponential = Spin * std::exp(-damping * elapsed / wheel.Inertia);
            const float measured = world.Velocities[wheel.Body].Angular.z;
            std::println("  {:4} {:10.6f} {:12.6f} {:13.6f} {:16.5f}", step, measured, geometric, exponential, measured / geometric);
        }
    }

    // The same row with a target it is being brought up to rather than down to.
    {
        constexpr float Target = 5;
        World world{context};
        const auto wheel = SpinOnAxle(world, {.Angular = {AxisLocked, AxisLocked, AxisDriven},
                                              .MotorSpeed = {0, 0, Target},
                                              .MotorMaxTorque = {0, 0, INFINITY},
                                              .AngularStiffness = {INFINITY, INFINITY, 0},
                                              .AngularDamping = {0, 0, damping}},
                                      float3{0, 0, 0});
        std::println("\na k = 0, c = {:g} drive to {:g} rad/s from rest, which approaches rather than arrives:", damping, Target);
        std::println("  step   measured    geometric   exponential");
        const float per_step = 1 + damping * settings.DeltaTime / wheel.Inertia;
        for (uint32_t step = 0; step < steps; ++step) {
            solver.Step(world, settings);
            if (!report(step)) continue;
            const float elapsed = float(step + 1) * settings.DeltaTime;
            const float geometric = Target * (1 - 1 / std::pow(per_step, float(step + 1)));
            const float exponential = Target * (1 - std::exp(-damping * elapsed / wheel.Inertia));
            std::println("  {:4} {:10.6f} {:12.6f} {:13.6f}", step, float(world.Velocities[wheel.Body].Angular.z), geometric, exponential);
        }
    }

    // And a spring with a damper across it, on a length rather than an angle.
    {
        constexpr float Stiffness = 2e5f;
        std::println("\na box hung on a linear spring of {:g} N/m, with and without the damper that predicts no overshoot:", Stiffness);
        for (const bool damped : {false, true}) {
            World world{context};
            const auto shape = world.AddShape(UnitBox);
            const auto anchor = world.AddBody({});
            const auto box = world.AddBody({.Pose = At(float3{0, 0, 0}), .Shape = shape, .Density = Density});
            const float m = 1 / world.Masses[box].InvMass;
            const float critical = 2 * std::sqrt(Stiffness * m);
            world.AddJoint({.BodyA = box, .BodyB = anchor, .At = {0, 0, 0},
                            .LinearStiffness = {INFINITY, Stiffness, INFINITY},
                            .LinearDamping = {0, damped ? critical : 0, 0}});
            const StepSettings falling{.Iterations = settings.Iterations};
            const float rest = -std::abs(falling.Gravity.y) * m / Stiffness;
            float lowest = 0, highest = -1e9f;
            for (uint32_t step = 0; step < steps; ++step) {
                solver.Step(world, falling);
                const float y = world.Poses[box].Position.y;
                lowest = std::min(lowest, y);
                if (step > 30) highest = std::max(highest, y);
            }
            std::println("  c = {:9.1f} ({:12}): rests at {:9.6f} against mg/k of {:9.6f}, dipped {:9.6f} m past that, and came back up to {:9.6f}",
                         damped ? critical : 0, damped ? "critical" : "no damper", float(world.Poses[box].Position.y), rest,
                         std::max(0.f, rest - lowest), highest);
        }
    }
}

// An explicit mass of zero is infinite, not absent.
// KHR physics rigid bodies Sec. 128 makes such a body impossible to translate and leaves its inertia finite, so it turns freely where it stands.
// Its MotionProperties sample authors one: a wheel of mass 0 with an inertia diagonal of (0, 1, 0).
void Pinned(std::span<char *> args) {
    const uint32_t steps = Arg(args, 2, 300);
    const StepSettings settings{};
    const StepSettings free_space{.Gravity = {0, 0, 0}};
    const float dt = settings.DeltaTime;
    const mtl::Context context;
    // A solid cube's inertia about its own centre, m (e^2 + e^2) / 12 with the two extents equal.
    const auto cube_inertia = [](float mass, float side) { return mass * side * side / 6; };

    std::println("a mass of zero is an infinite mass, not an absent one: nothing translates the body and\n"
                 "  its inertia still turns it. {} steps at {:g} Hz", steps, 1 / dt);

    // Weight is m g and there is no m, so the body stays exactly where it was put.
    {
        Solver solver{context};
        World world{context};
        const float inertia = cube_inertia(10, 1);
        const auto pinned = world.AddBody({.Pose = At(float3{0, 3, 0}), .Shape = world.AddShape(UnitBox),
                                           .Mass = {{.Mass = 0, .Inertia = {inertia, inertia, inertia}}}});
        const Pose was = world.Poses[pinned];
        for (uint32_t step = 0; step < steps; ++step) solver.Step(world, settings);
        std::println("  hanging in gravity for {} steps: moved {:.9f} m, and its inverse inertia is {:g}",
                     steps, float(simd::distance(world.Poses[pinned].Position, was.Position)), float(world.Masses[pinned].InvInertiaLocal.x));
        // Spun by hand, which is free flight about the body's own axis, so the rate is unchanged.
        world.Velocities[pinned] = {.Angular = {0, 1.5f, 0}};
        for (uint32_t step = 0; step < 120; ++step) solver.Step(world, settings);
        std::println("  spun at 1.500000 rad/s and left alone for 120: reads {:.6f}, and has moved {:.9f} m",
                     float(world.Velocities[pinned].Angular.y), float(simd::distance(world.Poses[pinned].Position, was.Position)));
    }

    // A ball into its face, off centre, in free space.
    // With restitution and friction both zero the only force is along the contact normal, so the ball's change of momentum measures the whole impulse.
    // The pinned body must then turn at r x J / I.
    // The arm is exactly the ball's centre height, since the contact sits on the ball's surface along the face normal.
    std::println("a ball of 5 kg at 3 m/s into a pinned box, {:g} m above its centre, in free space:", 0.25f);
    for (const float friction : {0.f, 0.5f}) {
        for (const bool turns : {true, false}) {
            Solver solver{context};
            World world{context};
            constexpr float BallMass = 5, BallRadius = 0.25f, Offset = 0.25f, Speed = 3, Inertia = 20;
            const auto pinned = world.AddBody({.Shape = world.AddShape(UnitBox),
                                               .Mass = {{.Mass = 0, .Inertia = turns ? float3{Inertia, Inertia, Inertia} : float3{0, 0, 0}}},
                                               .Friction = friction});
            const auto ball = world.AddBody({.Pose = At(float3{-Half - BallRadius - 0.3f, Offset, 0}),
                                             .Velocity = {.Linear = {Speed, 0, 0}},
                                             .Shape = world.AddShape({.Radius = BallRadius, .Kind = ShapeSphere}),
                                             .Mass = {{.Mass = BallMass, .Inertia = {1, 1, 1}}}, .Friction = friction});
            const Pose was = world.Poses[pinned];
            for (uint32_t step = 0; step < 24; ++step) solver.Step(world, free_space);
            const float impulse = BallMass * (Speed - float(world.Velocities[ball].Linear.x));
            std::println("  mu {:3.1f} inertia {:5.1f}: J {:8.4f} N s, spin {:9.6f} rad/s against r x J / I of {:9.6f}, moved {:.9f} m",
                         friction, turns ? Inertia : 0, impulse, float(world.Velocities[pinned].Angular.z),
                         turns ? -Offset * impulse / Inertia : 0, float(simd::distance(world.Poses[pinned].Position, was.Position)));
        }
    }

    // On no joint: an infinite mass keeps the axle in place, and an inertia of (0, 1, 0) leaves one axis free with the other two infinite.
    {
        Solver solver{context};
        World world{context};
        const Index shape = WheelMesh(world, 16, 8);
        if (shape == NoIndex) return (void)std::println(stderr, "the wheel would not cook");
        AddGround(world); // so a ball that misses stops rather than falling for ever at ghost speeds
        const auto wheel = world.AddBody({.Shape = shape, .Mass = {{.Mass = 0, .Inertia = {0, 0, 20 * WheelRadius * WheelRadius / 2}}},
                                          .Friction = Friction, .AngularDamping = 0.6f});
        const Index ball = world.AddShape({.Radius = 0.16f, .Kind = ShapeSphere});
        float turned = 0, wandered = 0, tilted = 0;
        float4 was = world.Poses[wheel].Orientation;
        std::println("a pinned paddle wheel on no joint at all, balls dropped on its near side:");
        for (uint32_t step = 0; step < 600; ++step) {
            if (step % 20 == 0 && step < 240) world.AddBody({.Pose = At(float3{0.62f, 2.2f, 0}), .Shape = ball, .Density = Density, .Friction = Friction});
            solver.Step(world, settings);
            // Accumulated, since a quaternion difference only ever gives the short way round.
            const float4 now = world.Poses[wheel].Orientation;
            turned += RotationVector(QuatMul(now, QuatConjugate(was))).z;
            was = now;
            wandered = std::max(wandered, float(simd::length(world.Poses[wheel].Position)));
            const float3 tilt = RotationVector(now);
            tilted = std::max(tilted, std::max(std::abs(float(tilt.x)), std::abs(float(tilt.y))));
            if (step % 120 != 119) continue;
            std::println("  {:4} turned {:9.4f} rad  rate {:8.4f} rad/s  off its place {:.9f} m  off its axis {:.9f} rad",
                         step, turned, float(world.Velocities[wheel].Angular.z), wandered, tilted);
        }
        std::println("    ->{:9.4f} rad, {:6.2f} turns, never further than {:.9f} m from where it started",
                     turned, turned / (2 * std::numbers::pi_v<float>), wandered);
    }

    // And what a body of infinite mass does for a body leaning on it.
    // The pair's reduced mass is the box's alone, since summing inverse masses drops the pinned side out rather than dividing by zero.
    // The normal row's penalty floor is then that box's own m/h^2 and stays finite.
    {
        Solver solver{context};
        World world{context};
        constexpr float BoxMass = 1000;
        const auto slab = world.AddBody({.Pose = At(float3{0, -0.25f, 0}), .Shape = world.AddShape({.HalfExtents = {3, 0.25f, 3}, .Kind = ShapeBox}),
                                         .Mass = {{.Mass = 0, .Inertia = {200, 200, 200}}}, .Friction = Friction});
        const float inertia = cube_inertia(BoxMass, 1);
        const auto box = world.AddBody({.Pose = At(float3{0, Half + 0.2f, 0}), .Shape = world.AddShape(UnitBox),
                                        .Mass = {{.Mass = BoxMass, .Inertia = {inertia, inertia, inertia}}}, .Friction = Friction});
        const Pose was = world.Poses[slab];
        for (uint32_t step = 0; step < steps; ++step) solver.Step(world, settings);
        float penalty = 0;
        uint32_t rows = 0;
        EachActive(world, [&](const Contact &contact) {
            penalty += contact.Penalty[0];
            ++rows;
        });
        const float inertial = BoxMass / (dt * dt);
        std::println("a box of {:g} kg resting on a pinned slab: rests at {:.6f}, slab sank {:.9f} m and turned {:.9f} rad/s,\n"
                     "  {} rows at a mean normal penalty of {:.4g}, which is {:.4f} of the box's own m/h^2 - the reduced mass of the pair",
                     BoxMass, float(world.Poses[box].Position.y), float(simd::distance(world.Poses[slab].Position, was.Position)),
                     float(simd::length(world.Velocities[slab].Angular)), rows, rows > 0 ? penalty / float(rows) : 0,
                     rows > 0 ? penalty / float(rows) / inertial : 0);
    }

    // Sleeping, the same as for any other body the solve moves.
    {
        Solver solver{context};
        World world{context};
        const float inertia = cube_inertia(40, 1);
        const auto pinned = world.AddBody({.Velocity = {.Angular = {0, 0, 2}}, .Shape = world.AddShape(UnitBox),
                                           .Mass = {{.Mass = 0, .Inertia = {inertia, inertia, inertia}}}, .Friction = 0, .AngularDamping = 4});
        for (uint32_t step = 0; step < steps; ++step) solver.Step(world, free_space);
        std::println("a pinned box spun at 2 rad/s into a damper: after {} steps it reads {:.6f} rad/s, asleep={}",
                     steps, float(world.Velocities[pinned].Angular.z), world.Quiet[pinned] >= free_space.SleepSteps);
        world.AddBody({.Pose = At(float3{-1.2f, 0.25f, 0}), .Velocity = {.Linear = {4, 0, 0}},
                       .Shape = world.AddShape({.Radius = 0.25f, .Kind = ShapeSphere}),
                       .Mass = {{.Mass = 30, .Inertia = {1, 1, 1}}}, .Friction = 0});
        for (uint32_t step = 0; step < 30; ++step) solver.Step(world, free_space);
        std::println("  and a ball into its face 30 steps later: {:.6f} rad/s, asleep={}",
                     float(world.Velocities[pinned].Angular.z), world.Quiet[pinned] >= free_space.SleepSteps);
    }
}
} // namespace

int main(int argc, char **argv) try {
    const std::span args{argv, size_t(argc)};
    const std::string_view scene = args.size() > 1 ? args[1] : "";
    // The usage line is spelled out rather than built from this list, since it names the scenes in reading order rather than dispatch order.
    const std::pair<std::string_view, void (*)(std::span<char *>)> scenes[]{
        {"stack", Stack}, {"slam", Slam}, {"slide", Slide}, {"shear", Shear}, {"bounce", Drop}, {"seams", Seams},
        {"raft", Raft}, {"bullet", Bullet}, {"order", Order}, {"coins", Coins}, {"motion", Motion}, {"offset", Offset},
        {"teleport", Teleport}, {"platform", Platform}, {"wheel", Wheel}, {"perch", Perch}, {"rocker", Rocker},
        {"frame", JointFrame}, {"slider", Slider}, {"brake", Brake}, {"pinned", Pinned}, {"compound", Compound}, {"join", Join}};
    for (const auto &[name, run] : scenes)
        if (name == scene) {
            run(args);
            return 0;
        }
    return std::println(stderr, "usage: RbpScenes <stack|raft|coins|slide|shear|slam|bounce|seams|bullet|order|motion|offset|teleport|platform|wheel|perch|frame|slider|brake|pinned|compound|join|rocker> [args...]"), 1;
} catch (const std::exception &error) {
    // See mtl::Buffer: a bad index is reported here and the process exits normally.
    std::println(stderr, "RbpScenes: {}", error.what());
    return 1;
}
