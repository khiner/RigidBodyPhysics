// Scenes that report what the solver is doing, for the questions tests cannot answer on their own.
//
//   RbpScenes stack [boxes] [steps] [iterations] [beta] [colors]
//   RbpScenes slide [steps] [iterations] [beta] [colors]
//   RbpScenes slam [speed] [density] [steps] [beta]
//   RbpScenes bounce [restitution] [steps] [iterations] [beta]
//   RbpScenes seams [cells] [steps] [speed]
//   RbpScenes raft [side] [layers] [steps] [iterations]
//   RbpScenes coins [sides] [coins] [steps] [twist] [iterations]
//   RbpScenes bullet [thickness] [graze]
//   RbpScenes order [boxes] [steps] [iterations]
//
// `stack` reports how far each box has sunk below where it analytically belongs - the standing error
// the penalty ramp leaves - and the contact churn that produced it: identities gained or lost since
// last step, each one a warm-start dual thrown away. A settled stack must show none. It also reports
// the mean normal penalty against the inertial stiffness m/h^2, which is the ratio that says whether
// the contacts are stiff enough for the integrator to damp what stands on them - far below it a stack
// rings at about a hertz and implicit Euler cannot touch it.
//
// `raft` builds a pile instead of a chain and reports what each body's contact run was asked to hold
// against what it has room for. A box in a stack touches two things and fits any budget, and one in the
// middle of a raft touches nine, and the points it cannot keep are the load-bearing ones. Refusals are
// exact, so the demand it prints is the number the run has to be sized against.
//
// NOSLEEP=1 stops bodies sleeping, and EVERY=1 reports every step rather than every sixtieth. Both
// exist because between them sleeping and coarse sampling hid that ringing for a whole session: a
// five-box stack falls asleep mid-sway, and its velocity then reads zero because sleeping sets it so.
//
// `seams` rolls a ball across a floor made of triangles and across a plane at once. A flat floor cut
// into triangles has edges running all over it that are not edges of anything, and the whole difficulty
// of a mesh is not letting a body find them: it must show one contact rather than two wherever the ball
// is, a height that never steps, and a ball that arrives where the one on the plane did. Every way of
// getting it wrong shows up as the ball falling behind, a second contact doubling the friction it feels.
//
// `bullet` is what speculative contacts have to survive. A small body is fired at a thin box and at a
// mesh wall of the same thickness across the speed range and must never end up behind either -
// penetration is a bounded error the next step takes back, crossing is unrecoverable, and a mesh has no
// back face to push anything out of. It also fires one shot past the corner of the wall, close enough
// that the reach crosses it but the geometry never does, which is the ghost collision speculation
// trades away for the crossings it prevents and the number the reach clamp is set against.
//
// `order` builds one stack twice, bottom-up and top-down. The two worlds hold the same geometry in the
// same places, so the engine has to answer them the same way - which it does not, because every pair
// belongs to the lower-indexed of its two bodies and that ownership decides which shape presents the
// reference face. Add order is arbitrary in any real scene, so the divergence is the engine giving two
// answers to one question, and its size says what flipping a pair costs.
//
// `slide` reports the net contact force against the motion it produced, and how far the box is into
// the plane while producing it.
//
// `slam` drives a box into the plane hard enough to make the penalty ramp work, and reports how far up
// the ramp it gets against the inertial stiffness m/h^2. Contacts are quietest at 1, usable over about
// [0.3, 3], and a stack held above 10 comes apart - and PenaltyMax is an absolute where that danger is
// a ratio, so what a light body reaches under a hard impact is the question.
//
// `bounce` reports every bounce a dropped box makes: what it arrived at, what it left at, and their
// ratio against the restitution asked for. The tests check a single bounce, so what this is for is the
// series - whether the ratio holds as the box slows, and where the threshold takes over and lets it
// settle rather than buzzing forever.

#include "Scenery.h"
#include "Solver.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <print>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {
uint32_t Arg(std::span<char *> args, size_t at, uint32_t fallback) {
    return args.size() > at ? uint32_t(std::atoi(args[at])) : fallback;
}
float ArgF(std::span<char *> args, size_t at, float fallback) {
    return args.size() > at ? float(std::atof(args[at])) : fallback;
}

// Every contact force in the world, resolved back into world space and summed. Normal and tangential
// are reported apart because it is their ratio against the friction cone that says whether Coulomb
// friction is saturated.
struct ContactTotals {
    float3 Normal{0, 0, 0}, Tangential{0, 0, 0};
    float Cone{};
    uint32_t Count{};
};

ContactTotals Totals(const World &world) {
    ContactTotals totals;
    for (uint32_t slot = 0; slot < world.Contacts.Capacity; ++slot) {
        const auto &contact = world.Contacts[slot];
        if (!contact.Active) continue;
        ++totals.Count;
        const float3 normal = contact.Normal;
        float3 first = std::abs(normal.x) > std::abs(normal.z) ? float3{-normal.y, normal.x, 0} : float3{0, -normal.z, normal.y};
        first = simd::normalize(first);
        const float3 second = simd::cross(normal, first);
        totals.Normal += contact.Lambda[0] * normal;
        totals.Tangential += contact.Lambda[1] * first + contact.Lambda[2] * second;
        totals.Cone += std::abs(contact.Lambda[0]) * contact.Friction;
    }
    return totals;
}

// The set of contacts the world is holding, named the way warm starting names them. A contact that
// keeps its identity keeps its dual across the step, and one that does not starts again from nothing -
// a disturbance the solver did not ask for and cannot see.
std::set<std::tuple<Index, Index, uint32_t>> ContactKeys(const World &world) {
    std::set<std::tuple<Index, Index, uint32_t>> keys;
    for (uint32_t slot = 0; slot < world.Contacts.Capacity; ++slot) {
        const auto &contact = world.Contacts[slot];
        if (contact.Active) keys.emplace(contact.BodyA, contact.BodyB, contact.Feature);
    }
    return keys;
}

// How many contacts have been gained and lost since it was last asked, which is zero for a stack that
// is genuinely at rest. Kept across a scene's whole loop, since every step has to be counted whether or
// not that step is one being reported on.
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

void Stack(std::span<char *> args) {
    const uint32_t boxes = Arg(args, 2, 3), steps = Arg(args, 3, 240);
    StepSettings settings{.Iterations = Arg(args, 4, 10), .Beta = ArgF(args, 5, StepSettings{}.Beta), .MaxColors = Arg(args, 6, StepSettings{}.MaxColors)};
    if (getenv("NOSLEEP")) settings.SleepSteps = ~0u;
    const bool every = getenv("EVERY") != nullptr;
    const mtl::Context context;
    Solver solver{context};
    World world{context};
    const std::vector<Index> stack = BuildStack(world, boxes);

    std::println("stack of {}, {} iterations, beta {:g}, {} colors", boxes, settings.Iterations, settings.Beta, settings.MaxColors);
    Churner churner{world};
    for (uint32_t step = 0; step < steps; ++step) {
        solver.Step(world, settings);
        churner.Step(world);
        if (!every && step % 60 != 59 && step != steps - 1) continue;
        uint32_t asleep = 0;
        for (uint32_t b = 1; b < world.BodyCount(); ++b) asleep += world.Quiet[b] >= settings.SleepSteps ? 1 : 0;
        // The mean normal penalty the contacts are holding, as a fraction of the inertial stiffness of
        // one box. Below about one the contacts are softer than what stands on them, and the stack's
        // rocking mode drops under the step rate where the integrator stops damping it.
        double penalty = 0;
        uint32_t active = 0;
        for (uint32_t slot = 0; slot < world.Contacts.Capacity; ++slot) {
            if (!world.Contacts[slot].Active) continue;
            penalty += world.Contacts[slot].Penalty[0];
            ++active;
        }
        const float inertial = 1 / (world.Masses[stack[0]].InvMass * settings.DeltaTime * settings.DeltaTime);
        std::print("{:4} churn={:4} asleep={} k/(m/h2)={:6.3f}", step, churner.Take(), asleep, active > 0 ? penalty / active / inertial : 0);
        for (uint32_t i = 0; i < boxes; ++i) {
            // Where it belongs: resting on the one below, one contact margin into it.
            const float ideal = Half + float(i) * (1 - settings.ContactMargin) - settings.ContactMargin;
            const float height = world.Poses[stack[i]].Position.y;
            std::print("  sag={:7.4f} |v|={:7.5f}", ideal - height, simd::length(world.Velocities[stack[i]].Linear));
        }
        std::println("");
    }
}

// A pile rather than a chain, so a body in the middle names nine partners rather than two and asks for
// far more points than a stack does. What it does with the ones it cannot have is the question the
// contact budget answers, and creep is what an underfed run looks like from outside.
void Raft(std::span<char *> args) {
    const uint32_t side = Arg(args, 2, 4), layers = Arg(args, 3, 2), steps = Arg(args, 4, 300);
    StepSettings settings{.Iterations = Arg(args, 5, StepSettings{}.Iterations)};
    if (getenv("NOSLEEP")) settings.SleepSteps = ~0u;
    const bool every = getenv("EVERY") != nullptr;
    const mtl::Context context;
    Solver solver{context};
    World world{context};
    const std::vector<Index> boxes = BuildRaft(world, side, layers);

    // What a body's own run was asked to hold this step. Every partner is collided whether or not the
    // run is full and each point that finds no place is counted once, so this is exact.
    const auto wanted = [&world](Index body) {
        uint32_t active = 0;
        for (uint32_t slot = 0; slot < ContactsPerBody; ++slot) active += world.Contacts[body * ContactsPerBody + slot].Active ? 1 : 0;
        return active + world.ContactRefusals[body];
    };

    std::println("a raft {} across and {} layers deep, {} boxes, {} slots a body", side, layers, boxes.size(), ContactsPerBody);
    Churner churner{world};
    uint32_t busiest = 0, demand = 0;
    for (uint32_t step = 0; step < steps; ++step) {
        solver.Step(world, settings);
        churner.Step(world);
        for (const auto body : boxes)
            if (wanted(body) > demand) demand = wanted(busiest = body);
        if (!every && step % 60 != 59 && step != steps - 1) continue;
        uint32_t asleep = 0, refused = 0;
        float fastest = 0;
        for (const auto body : boxes) {
            asleep += world.Quiet[body] >= settings.SleepSteps ? 1 : 0;
            refused += world.ContactRefusals[body];
            fastest = std::max(fastest, simd::length(world.Velocities[body].Linear));
        }
        std::println("{:4} churn={:4} asleep={:3}/{} refused={:4} |v|max={:8.5f} top y={:7.4f}",
                     step, churner.Take(), asleep, boxes.size(), refused, fastest, float(world.Poses[boxes.back()].Position.y));
    }
    std::println("the busiest run was body {}'s, asked for {} points against {} slots", busiest, demand, ContactsPerBody);
}

// Round faces, which no other scene has: a box face meeting a box face clips to exactly four points, so
// `ReduceManifold` never runs anywhere else in the suite. A stack of many-sided prisms each twisted
// against the one below produces more points than a manifold may keep - two eight-gons crossed at half
// a step intersect in a sixteen-gon - so the four that survive are chosen rather than found.
void Coins(std::span<char *> args) {
    const uint32_t sides = Arg(args, 2, 8), coins = Arg(args, 3, 4), steps = Arg(args, 4, 400);
    const float twist = ArgF(args, 5, 0.5f); // in steps of the prism's own, so 0.5 is the worst case
    StepSettings settings{.Iterations = Arg(args, 6, StepSettings{}.Iterations)};
    if (getenv("NOSLEEP")) settings.SleepSteps = ~0u;
    const bool every = getenv("EVERY") != nullptr;
    const mtl::Context context;
    Solver solver{context};
    World world{context};
    const std::vector<Index> stack = BuildCoins(world, sides, coins, twist);
    if (stack.empty()) return (void)std::println(stderr, "the prism made no hull");

    std::println("{} coins of {} sides, each twisted {:g} of a face against the one below, {} manifold points a pair",
                 coins, sides, twist, ManifoldPoints);
    Churner churner{world};
    for (uint32_t step = 0; step < steps; ++step) {
        solver.Step(world, settings);
        churner.Step(world);
        if (!every && step % 100 != 99 && step != steps - 1) continue;
        uint32_t asleep = 0, held = 0;
        float worst_tilt = 0;
        for (uint32_t i = 0; i < coins; ++i) {
            const Index body = stack[i];
            asleep += world.Quiet[body] >= settings.SleepSteps ? 1 : 0;
            // How far it has turned out of the twist it was set down with, which is what a manifold too
            // narrow to carry a round face shows as first.
            const float4 was = QuatFromRotationVector(float3{0, CoinTwist(sides, twist, i), 0});
            worst_tilt = std::max(worst_tilt, simd::length(RotationVector(QuatMul(world.Poses[body].Orientation, QuatConjugate(was)))));
            for (uint32_t slot = 0; slot < ContactsPerBody; ++slot) held += world.Contacts[body * ContactsPerBody + slot].Active ? 1 : 0;
        }
        std::print("{:4} churn={:4} asleep={}/{} rows={:3} turned={:8.5f}", step, churner.Take(), asleep, coins, held, worst_tilt);
        for (uint32_t i = 0; i < coins; ++i) {
            const float ideal = CoinHalfHeight + float(i) * (2 * CoinHalfHeight - settings.ContactMargin) - settings.ContactMargin;
            std::print("  sag={:7.4f}", ideal - float(world.Poses[stack[i]].Position.y));
        }
        std::println("");
    }
}

void Slide(std::span<char *> args) {
    const uint32_t steps = Arg(args, 2, 120);
    const StepSettings settings{.Iterations = Arg(args, 3, 10), .Beta = ArgF(args, 4, StepSettings{}.Beta), .MaxColors = Arg(args, 5, StepSettings{}.MaxColors)};
    const mtl::Context context;
    Solver solver{context};
    World world{context};
    const auto shape = world.AddShape(UnitBox);
    world.AddBody({.Shape = world.AddShape(GroundPlane), .Friction = Friction});
    const auto box = world.AddBody(
        {.Pose = {.Position = {0, Half, 0}, .Orientation = {0, 0, 0, 1}}, .Shape = shape, .Density = Density, .Friction = Friction}
    );

    for (uint32_t settle = 0; settle < 30; ++settle) solver.Step(world, settings);
    constexpr float Speed = 2;
    world.Velocities[box].Linear = {Speed, 0, 0};

    const float gravity = std::abs(settings.Gravity.y), mass = 1 / world.Masses[box].InvMass;
    std::println("slide at {} m/s, {} iterations, beta {:g}, {} colors. Coulomb says it decelerates at mu g = {:.3f} m/s^2", Speed, settings.Iterations, settings.Beta, settings.MaxColors, Friction * gravity);
    for (uint32_t step = 0; step < steps; ++step) {
        const float was = world.Velocities[box].Linear.x;
        solver.Step(world, settings);
        if (step % 60 != 59 && step != steps - 1) continue;
        const auto totals = Totals(world);
        const float speed = world.Velocities[box].Linear.x;
        // What the reported force should have done to the body over one step, against what it did.
        const float implied = -totals.Tangential.x / mass, measured = (speed - was) / settings.DeltaTime;
        std::println("{:4} vx={:7.4f} sink={:7.4f} contacts={} normal={:8.1f} tangential={:8.1f} of cone {:8.1f} | "
                     "force implies {:7.3f} m/s^2, motion shows {:7.3f}",
                     step, speed, Half - float(world.Poses[box].Position.y), totals.Count, simd::length(totals.Normal), simd::length(totals.Tangential), totals.Cone, implied, measured);
    }
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
    const auto box = world.AddBody({.Pose = {.Position = {0, Half + 1, 0}, .Orientation = {0, 0, 0, 1}},
                                    .Shape = shape,
                                    .Density = Density,
                                    .Friction = Friction,
                                    .Restitution = restitution});

    std::println("drop from 1 m at restitution {:g}, {} iterations, beta {:g}. Below {:g} m/s an impact does not bounce",
                 restitution, settings.Iterations, settings.Beta,
                 settings.BounceSpeedFactor * simd::length(settings.Gravity) * settings.DeltaTime);
    uint32_t bounces = 0;
    for (uint32_t step = 0; step < steps; ++step) {
        const float before = world.Velocities[box].Linear.y;
        solver.Step(world, settings);
        const float after = world.Velocities[box].Linear.y;
        if (before >= 0 || after <= 0) continue; // not the step a bounce happened on
        const auto totals = Totals(world);
        std::println("{:4} bounce {:2} arrived {:7.4f} left {:7.4f} ratio {:6.4f} | contacts={} normal={:9.1f}",
                     step, ++bounces, -before, after, after / -before, totals.Count, simd::length(totals.Normal));
    }
    std::println("came to rest at {:.5f} m with |v| {:.6f} after {} bounces",
                 float(world.Poses[box].Position.y), simd::length(world.Velocities[box].Linear), bounces);
}

// The largest normal penalty any live contact is holding, as a multiple of one box's inertial
// stiffness. This is the number the penalty floor is expressed in and the one the instability band is
// measured in, so it is what a ramp has to be watched in too.
float PeakPenaltyRatio(const World &world, float inertial) {
    float peak = 0;
    for (uint32_t slot = 0; slot < world.Contacts.Capacity; ++slot)
        if (world.Contacts[slot].Active) peak = std::max(peak, world.Contacts[slot].Penalty[0]);
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
    world.AddBody({.Shape = world.AddShape(GroundPlane), .Friction = Friction});
    // Started just clear of the plane so the first step is the one that arrives, at whatever speed is
    // asked for rather than whatever a drop height happens to give.
    const auto box = world.AddBody({.Pose = {.Position = {0, Half + speed * settings.DeltaTime, 0}, .Orientation = {0, 0, 0, 1}}, .Velocity = {.Linear = {0, -speed, 0}}, .Shape = shape, .Density = density, .Friction = Friction});
    const float mass = 1 / world.Masses[box].InvMass;
    const float inertial = mass / (settings.DeltaTime * settings.DeltaTime);
    // What the ceiling means for a body of this mass. It is absolute, so a light body has orders of
    // room above the band before it binds and a very heavy one is already inside it.
    std::println("slam at {:g} m/s, density {:g}, mass {:g}, beta {:g}. m/h^2 = {:.4g}, and PenaltyMax {:g} is {:.4g} of it", speed, density, mass, settings.Beta, inertial, settings.PenaltyMax, settings.PenaltyMax / inertial);

    float peak_ratio = 0, deepest = 0, fastest_out = 0;
    for (uint32_t step = 0; step < steps; ++step) {
        solver.Step(world, settings);
        const float ratio = PeakPenaltyRatio(world, inertial);
        const float height = world.Poses[box].Position.y, out = world.Velocities[box].Linear.y;
        peak_ratio = std::max(peak_ratio, ratio);
        deepest = std::max(deepest, Half - height);
        fastest_out = std::max(fastest_out, out);
        if (step < 12) std::println("{:4} y={:8.5f} vy={:9.4f} sink={:8.5f} k/(m/h2)={:11.4g} contacts={}", step, height, out, Half - height, ratio, Totals(world).Count);
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
    if (FloorMesh(mesh, cells) == NoIndex) return (void)std::println(stderr, "the mesh would not cook");
    mesh.AddBody({.Shape = 0});
    plane.AddBody({.Shape = plane.AddShape(GroundPlane)});
    // Off the grid lines, so it crosses seams rather than running along one.
    for (World *world : {&mesh, &plane})
        world->AddBody({.Pose = {.Position = {-8, Radius, 0.07f}, .Orientation = {0, 0, 0, 1}},
                        .Velocity = {.Linear = {speed, 0, 0}},
                        .Shape = world->AddShape({.HalfExtents = {0, 0, 0}, .Radius = Radius, .Kind = ShapeSphere})});
    std::println("a ball rolling across {} triangles, and the same ball on a plane. {} cells over 20 m, {} m/s",
                 2 * cells * cells, cells, speed);
    uint32_t crowded = 0, empty = 0;
    float worst_height = 0, worst_behind = 0;
    for (uint32_t step = 0; step < steps; ++step) {
        solver.Step(mesh);
        solver.Step(plane);
        uint32_t contacts = 0;
        for (uint32_t slot = ContactsPerBody; slot < 2 * ContactsPerBody; ++slot) contacts += mesh.Contacts[slot].Active ? 1 : 0;
        crowded += contacts > 1 ? 1 : 0;
        empty += contacts == 0 ? 1 : 0;
        worst_height = std::max(worst_height, std::abs(float(mesh.Poses[1].Position.y) - Radius));
        worst_behind = std::max(worst_behind, std::abs(float(mesh.Poses[1].Position.x - plane.Poses[1].Position.x)));
        if (step % 30 == 29)
            std::println("{:4} contacts={} x={:8.4f} against {:8.4f} y={:7.5f} vx={:7.4f} against {:7.4f}", step, contacts,
                         float(mesh.Poses[1].Position.x), float(plane.Poses[1].Position.x), float(mesh.Poses[1].Position.y),
                         float(mesh.Velocities[1].Linear.x), float(plane.Velocities[1].Linear.x));
    }
    std::println("steps holding more than one contact: {}, holding none: {}. Worst height error {:.6f} m, "
                 "worst distance behind the plane {:.6f} m, refused {}",
                 crowded, empty, worst_height, worst_behind, mesh.ContactRefusals[1]);
}
// A wall of two triangles standing in the y-z plane, wound so its face looks back down the line of
// fire. The one shape a body can cross and never be recovered from.
Index WallMesh(World &world, float reach) {
    const std::vector<float3> points{float3{0, -reach, -reach}, float3{0, -reach, reach}, float3{0, reach, reach},
                                     float3{0, reach, -reach}};
    const std::vector<uint32_t> indices{0, 1, 2, 0, 2, 3};
    return world.AddMesh(points, indices);
}

// Which of the two boxes presented the reference face, over every box-on-box contact a world holds.
// Bit 13 of a feature is the side the separating axis chose, but which physical box that landed on is
// the interesting thing, since that is what a flipped pair changes.
struct FaceOwners {
    uint32_t Lower{}, Upper{};
};

FaceOwners Owners(const World &world) {
    FaceOwners owners;
    for (uint32_t slot = 0; slot < world.Contacts.Capacity; ++slot) {
        const auto &contact = world.Contacts[slot];
        if (!contact.Active || contact.BodyA == 0 || contact.BodyB == 0) continue; // box on box, not on the plane
        const bool on_b = ((contact.Feature >> 13) & 1) != 0;
        const Index face = on_b ? contact.BodyB : contact.BodyA;
        const Index other = on_b ? contact.BodyA : contact.BodyB;
        if (world.Poses[face].Position.y < world.Poses[other].Position.y) ++owners.Lower;
        else ++owners.Upper;
    }
    return owners;
}

void Order(std::span<char *> args) {
    const uint32_t boxes = Arg(args, 2, 12), steps = Arg(args, 3, 600);
    StepSettings settings{.Iterations = Arg(args, 4, StepSettings{}.Iterations),
                          .MaxColors = Arg(args, 5, StepSettings{}.MaxColors)};
    if (getenv("NOSLEEP")) settings.SleepSteps = ~0u;
    const bool every = getenv("EVERY") != nullptr;
    const mtl::Context context;
    Solver solver{context};
    World up{context}, down{context};

    // The same stack in both, and the plane first in both so only the boxes differ. `at` maps a place
    // in the stack to the body holding it, which is what makes the two comparable at all.
    std::vector<Index> up_at(boxes), down_at(boxes);
    for (World *world : {&up, &down}) {
        world->AddShape(UnitBox);
        world->AddBody({.Shape = world->AddShape(GroundPlane), .Friction = Friction});
    }
    const auto place = [&](World &world, uint32_t i) {
        return world.AddBody({.Pose = {.Position = {0, Half + 1.02f * float(i), 0}, .Orientation = {0, 0, 0, 1}},
                              .Shape = 0,
                              .Density = Density,
                              .Friction = Friction});
    };
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
        // How far apart the two answers have drifted, over every box in the stack.
        apart = 0;
        for (uint32_t i = 0; i < boxes; ++i)
            apart = std::max(apart, simd::length(up.Poses[up_at[i]].Position - down.Poses[down_at[i]].Position));
        worst = std::max(worst, apart);

        uint32_t up_asleep = 0, down_asleep = 0;
        for (uint32_t i = 0; i < boxes; ++i) {
            up_asleep += up.Quiet[up_at[i]] >= settings.SleepSteps ? 1 : 0;
            down_asleep += down.Quiet[down_at[i]] >= settings.SleepSteps ? 1 : 0;
        }
        if (up_slept == ~0u && up_asleep == boxes) up_slept = step;
        if (down_slept == ~0u && down_asleep == boxes) down_slept = step;
        if (!every && step % 60 != 59 && step != steps - 1) continue;

        const auto up_owners = Owners(up), down_owners = Owners(down);
        // The colours each stack settled on, bottom first in both. Colours solve in sequence, so which
        // end of the chain holds colour zero is the end load propagates from - and priority goes by
        // index, so that end is decided by the order the boxes were added.
        std::string up_colors, down_colors;
        for (uint32_t i = 0; i < boxes; ++i) {
            up_colors += char('0' + ColorOf(up.Colors[up_at[i]]) % 10);
            down_colors += char('0' + ColorOf(down.Colors[down_at[i]]) % 10);
        }
        // Where each stack's top box has got to, against where resting on the one below would put it.
        const float ideal = Half + float(boxes - 1) * (1 - settings.ContactMargin) - settings.ContactMargin;
        std::println("{:4} apart {:9.6f}  |  up: top sag {:8.5f} asleep {:2} face on lower/upper {:3}/{:3}"
                     "  |  down: top sag {:8.5f} asleep {:2} face on lower/upper {:3}/{:3}",
                     step, apart, ideal - float(up.Poses[up_at[boxes - 1]].Position.y), up_asleep,
                     up_owners.Lower, up_owners.Upper, ideal - float(down.Poses[down_at[boxes - 1]].Position.y),
                     down_asleep, down_owners.Lower, down_owners.Upper);
        std::println("      colors, bottom of the stack first:  up {}  down {}", up_colors, down_colors);
    }
    const auto slept = [](uint32_t at) { return at == ~0u ? std::string{"never"} : std::to_string(at); };
    std::println("at rest the two are {:.6f} m apart, worst they ever got {:.6f} m. Bottom-up slept at {}, top-down at {}",
                 apart, worst, slept(up_slept), slept(down_slept));
}

void Bullet(std::span<char *> args) {
    const float thickness = ArgF(args, 2, 0.02f);
    constexpr float Size = 0.05f, From = -1, Reach = 2; // the shot's half extent, where it starts, the wall's
    // Fired level, in free space. The question here is whether a body ends up on the wrong side of
    // something, and gravity only adds a fall - watch a slow shot long enough and it slides down the
    // face of the wall and off the bottom, which is not what is being measured. It costs the reach its
    // gravity term, which over a step is a millimetre and a half against the metres of travel below.
    const StepSettings settings{.Gravity = {0, 0, 0}};
    const mtl::Context context; // one for the thirty-three worlds below, which is what it is there for

    enum Wall : uint32_t { NoWall,
                           BoxWall,
                           MeshWall };
    // One shot: where it came to rest, how far past the wall's near face it ever reached, and what its
    // run refused on the way. The two walls are not the same failure - a body that crosses a box comes
    // back out of the far side, and one that crosses a mesh is gone, since a mesh has no back to push
    // anything out of.
    struct Shot {
        float3 Rest{};
        float Deepest{}, Refused{}, Kept{};
        uint32_t Crossed{};
    };
    const auto fire = [&](float3 velocity, float3 from, Wall against, float clamped = INFINITY) {
        Solver solver{context};
        World world{context};
        if (against != NoWall) {
            const Index wall = against == MeshWall
                ? WallMesh(world, Reach)
                : world.AddShape({.HalfExtents = {thickness / 2, Reach, Reach}, .Kind = ShapeBox});
            if (wall == NoIndex) return Shot{};
            world.AddBody({.Shape = wall, .Density = 0});
        }
        const auto shot = world.AddBody({.Pose = {.Position = from, .Orientation = {0, 0, 0, 1}},
                                         .Velocity = {.Linear = velocity},
                                         .Shape = world.AddShape({.HalfExtents = {Size, Size, Size}, .Kind = ShapeBox}),
                                         .Density = Density});
        // Long enough to cover the metres either side of the wall at this speed, and never so long that
        // a slow shot is still being watched a minute after it stopped.
        const float speed = simd::length(velocity);
        const uint32_t steps = std::min(3000u, uint32_t(3 / (speed * settings.DeltaTime)) + 60);
        StepSettings shot_settings = settings;
        shot_settings.MaxContactReach = clamped;
        // Where the near face of the wall is: its own front for a box, and the surface itself for a
        // mesh, which has no thickness at all.
        const float face = against == BoxWall ? -thickness / 2 : 0;
        Shot out{.Rest = from};
        for (uint32_t step = 0; step < steps; ++step) {
            solver.Step(world, shot_settings);
            const float3 at = world.Poses[shot].Position;
            out.Rest = at;
            out.Kept = simd::length(world.Velocities[shot].Linear) / speed;
            out.Deepest = std::max(out.Deepest, at.x + Size - face);
            out.Refused += float(world.ContactRefusals[shot]);
            // Out the far side, which is the one outcome nothing can walk back.
            if (against != NoWall && at.x - Size > thickness / 2) ++out.Crossed;
        }
        return out;
    };

    std::println("a {:g} m box fired at a {:g} m wall from {:g} m, box wall and mesh wall. It must never end behind either.",
                 2 * Size, thickness, -From);
    std::println("  deepest is how far its leading face ever got past the wall's, so a contact margin is where it rests");
    for (const float speed : {5.f, 20.f, 60.f, 120.f, 250.f, 500.f}) {
        const float3 velocity{speed, 0, 0}, from{From, 0, 0};
        const auto box = fire(velocity, from, BoxWall), mesh = fire(velocity, from, MeshWall);
        const auto pinned = fire(velocity, from, MeshWall, ArgF(args, 4, 0.05f));
        std::println("{:6.0f} m/s ({:7.3f} m a step)  box: rest {:8.4f} deepest {:8.5f} refused {:3.0f} crossed {:3}"
                     "  |  mesh: rest {:8.4f} deepest {:8.5f} crossed {:3}  |  clamped: crossed {:3}",
                     speed, speed * settings.DeltaTime, box.Rest.x, box.Deepest, box.Refused, box.Crossed,
                     mesh.Rest.x, mesh.Deepest, mesh.Crossed, pinned.Crossed);
    }

    // And what it costs. This shot starts squarely in front of the wall and rises fast enough to clear
    // the top edge before it arrives, so at the pose every step begins from it is closing on a face it
    // never meets, which is the whole of what a ghost collision is. A shot merely passing a wall by is
    // not the case worth reporting: points are clipped into the face that made them rather than to an
    // infinite plane, so a body beside a wall is given no point to be pushed by.
    //
    // It reports the speed the shot keeps, since a ghost takes speed rather than position, and the same
    // shot with the reach clamped - which is the dial. Note what the clamp costs above before using it.
    const float clearance = ArgF(args, 3, 0.05f);
    const float clamped = ArgF(args, 4, 0.05f);
    std::println("clearing the {:g} m corner by {:g} m on the way past, unclamped and with the reach clamped to {:g} m",
                 Reach, clearance, clamped);
    for (const float speed : {20.f, 120.f, 500.f}) {
        // Rising just fast enough over the metre of approach to pass the corner by the clearance.
        const float3 velocity{speed, speed * (Reach + Size + clearance) / -From, 0}, from{From, 0, 0};
        const auto clear = fire(velocity, from, NoWall);
        std::println("{:6.0f} m/s  box keeps {:6.2f}% of its speed, clamped {:6.2f}%  |  mesh {:6.2f}%, clamped {:6.2f}%"
                     "  |  missing it entirely keeps {:6.2f}%",
                     speed, 100 * fire(velocity, from, BoxWall).Kept, 100 * fire(velocity, from, BoxWall, clamped).Kept,
                     100 * fire(velocity, from, MeshWall).Kept, 100 * fire(velocity, from, MeshWall, clamped).Kept,
                     100 * clear.Kept);
    }
}
} // namespace

int main(int argc, char **argv) {
    const std::span args{argv, size_t(argc)};
    const std::string_view scene = args.size() > 1 ? args[1] : "";
    if (scene == "stack") Stack(args);
    else if (scene == "slam") Slam(args);
    else if (scene == "slide") Slide(args);
    else if (scene == "bounce") Drop(args);
    else if (scene == "seams") Seams(args);
    else if (scene == "raft") Raft(args);
    else if (scene == "bullet") Bullet(args);
    else if (scene == "order") Order(args);
    else if (scene == "coins") Coins(args);
    else return std::println(stderr, "usage: RbpScenes <stack|raft|coins|slide|slam|bounce|seams|bullet|order> [args...]"), 1;
    return 0;
}
