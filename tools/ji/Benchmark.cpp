#include "Scenarios.h"
#include "metal/Context.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

uint32_t ParseCount(const char *text) {
    uint32_t value{};
    const char *end = text + std::char_traits<char>::length(text);
    const auto [at, error] = std::from_chars(text, end, value);
    if (error != std::errc{} || at != end) throw std::invalid_argument("Expected an unsigned integer");
    return value;
}

double ParseNonnegativeFinite(const char *text) {
    size_t used = 0;
    const double value = std::stod(text, &used);
    if (used != std::char_traits<char>::length(text) || !std::isfinite(value) || value < 0)
        throw std::invalid_argument("Expected a nonnegative finite number");
    return value;
}

double Percentile(std::vector<double> values, double fraction) {
    std::sort(values.begin(), values.end());
    const double at = fraction * double(values.size() - 1);
    const size_t below = size_t(at), above = std::min(below + 1, values.size() - 1);
    return values[below] + (at - double(below)) * (values[above] - values[below]);
}

void Array(std::ostream &out, const auto &values) {
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) out << ',';
        out << values[i];
    }
    out << ']';
}

std::array<uint32_t, 11> ContactKeyFields(const ji::ContactKey &key) {
    // Same field order as the GPU's persistent JiWarmContact::key.
    return {key.a.Slot, key.a.Spawn, key.b.Slot, key.b.Spawn,
            key.shape_a, key.shape_b, key.child_a, key.child_b,
            key.subshape_a, key.subshape_b, key.feature};
}

std::string CapturePath(const std::string &path, uint32_t count, uint32_t step) {
    if (count == 1) return path;
    const std::filesystem::path prefix{path};
    return (prefix.parent_path() / (prefix.stem().string() + ".step" +
            std::to_string(step) + prefix.extension().string())).string();
}

struct ContactEvents {
    // These thresholds classify diagnostics, never physics admission. Force
    // units resolve card support without the former fixed 0.01 N s cutoff.
    static constexpr double ForceN = 2e-6, SpeedMps = 1e-4, ConeRelative = 1e-3, Opposition = 0.99;
    using Feature = std::array<uint32_t, 11>;
    using Pair = std::array<uint32_t, 4>;
    struct Sample {
        uint32_t step;
        bool previous_available{};
        bool loaded_history_available{};
        std::array<uint32_t, 15> counts{};
        std::array<double, 11> values{};
    };
    explicit ContactEvents(bool complete_history) : full_history_available(complete_history) {}
    bool full_history_available;
    std::vector<Sample> samples;
    std::map<Feature, uint32_t> prior_features;
    std::map<Pair, double> prior_pairs;
    std::set<Pair> ever_loaded;

    static void WriteCounts(std::ostream &out, const Sample &sample) {
        // Birth/death and mode changes need a previous sample; pair birth/return
        // additionally needs the entire loaded history since factory start.
        for (size_t i = 0; i < sample.counts.size(); ++i) {
            if (i) out << ',';
            if ((!sample.previous_available && (i == 1 || i == 2 || i >= 9)) ||
                (!sample.loaded_history_available && (i == 9 || i == 11 || i == 12)))
                out << "null";
            else out << sample.counts[i];
        }
    }

    void Add(const ji::SceneResult &result, uint32_t step, double dt) {
        const auto &system = result.solve.system;
        std::map<Feature, uint32_t> features;
        std::map<Pair, double> pairs;
        std::vector<ji::Vec> contact_wrench(system.bodies.size()), all_wrench(system.bodies.size());
        Sample sample{step};
        sample.previous_available = !samples.empty();
        sample.loaded_history_available = sample.previous_available && full_history_available;
        for (size_t index = 0; index < system.constraints.size(); ++index) {
            const auto &row = system.constraints[index];
            ji::Vec relative{};
            for (const auto &side : row.side) {
                if (side.body == ji::NoBody) continue;
                for (uint32_t axis = 0; axis < 6; ++axis)
                    for (uint32_t at = 0; at < row.rows; ++at) {
                        const double impulse = row.impulse[at] * side.j[6 * at + axis];
                        all_wrench[side.body][axis] += impulse;
                        if (row.kind == ji::Kind::Contact) contact_wrench[side.body][axis] += impulse;
                        relative[at] += side.j[6 * at + axis] * system.bodies[side.body].v[axis];
                    }
            }
            if (row.kind == ji::Kind::Soft) {
                for (uint32_t at = 0; at < row.rows; ++at) {
                    sample.values[8] += row.impulse[at] * relative[at];
                    sample.values[9] += row.impulse[at] * row.velocity_bias[at];
                }
            }
            if (row.kind != ji::Kind::Contact) continue;
            if (index >= result.contacts.size()) throw std::runtime_error("Contact event/source order mismatch");
            const auto key = ContactKeyFields(result.contacts[index].key);
            Pair pair{key[0], key[1], key[2], key[3]};
            if (std::array{pair[2], pair[3]} < std::array{pair[0], pair[1]})
                pair = {pair[2], pair[3], pair[0], pair[1]};
            pairs[pair] += std::max(0.0, row.impulse[0]);
            const double travel = std::hypot(relative[1] + row.error[1], relative[2] + row.error[2]);
            const double force = std::hypot(row.impulse[1], row.impulse[2]);
            const double dot = row.impulse[1] * (relative[1] + row.error[1]) +
                               row.impulse[2] * (relative[2] + row.error[2]);
            uint32_t mode = 0;
            if (row.impulse[0] > ForceN) {
                mode = travel / dt <= SpeedMps ? 1 : 3;
                if (mode == 3 && force > 0 && -dot >= Opposition * force * travel &&
                    std::abs(row.friction * row.impulse[0] - force) <= ForceN +
                         ConeRelative * row.friction * row.impulse[0]) mode = 2;
            }
            if (!features.emplace(key, mode).second) throw std::runtime_error("Duplicate contact event identity");
            sample.counts[3] += mode != 0;
            sample.counts[4] += mode == 1; sample.counts[5] += mode == 2; sample.counts[6] += mode == 3;
            sample.values[0] += dt * std::max(0.0, row.impulse[0]);
            sample.values[6] += row.impulse[0] * relative[0];
            sample.values[7] += row.impulse[1] * relative[1] + row.impulse[2] * relative[2];
            sample.values[10] = std::max(sample.values[10], travel / dt);
        }
        sample.counts[0] = uint32_t(features.size()); sample.counts[7] = uint32_t(pairs.size());
        if (sample.previous_available) {
            for (const auto &[key, mode] : features) {
                const auto previous = prior_features.find(key);
                sample.counts[1] += previous == prior_features.end();
                if (previous != prior_features.end()) {
                    sample.counts[13] += previous->second == 1 && mode == 2;
                    sample.counts[14] += previous->second == 2 && mode == 1;
                }
            }
            for (const auto &[key, mode] : prior_features) sample.counts[2] += !features.contains(key);
            for (const auto &[key, load] : prior_pairs)
                sample.counts[10] += load > ForceN && (!pairs.contains(key) || pairs.at(key) <= ForceN);
        }
        for (const auto &[key, load] : pairs) {
            sample.values[1] = std::max(sample.values[1], dt * load);
            if (load <= ForceN) continue;
            ++sample.counts[8];
            if (sample.loaded_history_available &&
                (!prior_pairs.contains(key) || prior_pairs.at(key) <= ForceN)) {
                const bool returned = ever_loaded.contains(key);
                sample.counts[9] += !returned; sample.counts[11] += returned;
                sample.counts[12] += returned && !prior_pairs.contains(key);
            }
            ever_loaded.insert(key);
        }
        for (size_t body = 0; body < system.bodies.size(); ++body)
            for (uint32_t group = 0; group < 2; ++group) {
                const auto &wrench = group ? all_wrench[body] : contact_wrench[body];
                sample.values[2 + 2 * group] = std::max(sample.values[2 + 2 * group],
                    std::hypot(wrench[0], wrench[1], wrench[2]));
                sample.values[3 + 2 * group] = std::max(sample.values[3 + 2 * group],
                    std::hypot(wrench[3], wrench[4], wrench[5]));
            }
        samples.push_back(sample); prior_features = std::move(features); prior_pairs = std::move(pairs);
    }

    void Write(std::ostringstream &out) const {
        out << ",\"events\":{\"scope\":\"CPU diagnostics; feature churn is not impact or recontact. Loaded return is force engagement; pair absence is a separate geometric-contact-set diagnostic. Thresholds classify events only.\""
            << ",\"normal_force_threshold_N\":" << ForceN << ",\"stick_speed_threshold_m_s\":" << SpeedMps
            << ",\"cone_absolute_tolerance_N\":" << ForceN << ",\"cone_relative_tolerance\":" << ConeRelative
            << ",\"slip_opposition_minimum\":" << Opposition
            << ",\"counts_order\":[\"features\",\"feature_births\",\"feature_deaths\",\"loaded_features\",\"stick_features\",\"slip_features\",\"loaded_moving_unclassified\",\"body_pairs\",\"loaded_body_pairs\",\"loaded_pair_births\",\"loaded_pair_releases\",\"loaded_pair_returns\",\"loaded_returns_after_pair_absence\",\"stick_to_slip\",\"slip_to_stick\"]"
            << ",\"physical_order\":[\"sum_normal_impulse_N_s\",\"strongest_pair_normal_impulse_N_s\",\"max_contact_body_linear_impulse_N_s\",\"max_contact_body_angular_impulse_N_m_s\",\"max_allrow_body_linear_impulse_N_s\",\"max_allrow_body_angular_impulse_N_m_s\",\"dynamic_normal_contact_work_J\",\"dynamic_tangential_contact_work_J\",\"soft_joint_dynamic_work_J\",\"soft_joint_prescribed_work_J\",\"max_tangent_speed_m_s\"],\"steps\":[";
        for (size_t i = 0; i < samples.size(); ++i) {
            const auto &sample = samples[i];
            out << (i ? "," : "") << "{\"scene_step\":" << sample.step << ",\"counts\":[";
            WriteCounts(out,sample);
            out << "],\"transition_counts_available\":" << sample.previous_available
                << ",\"loaded_history_counts_available\":" << sample.loaded_history_available
                << ",\"physical\":";
            Array(out, sample.values);
            out << '}';
        }
        out << "]}";
    }
};

void WriteFrozenEquations(const std::string &path, const ji::System &initial,
                          const ji::System &solved, const std::string &scenario,
                          uint32_t step, double dt, ji::Vec3 gravity,
                          const std::vector<std::array<double, 4>> &orientations,
                          const std::vector<std::array<double, 3>> &positions,
                          const std::vector<uint32_t> &world_slots,
                          const std::vector<ji::SceneContact> &contacts) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot open frozen equation output: " + path);
    ji::ValidateSystem(initial);
    ji::ValidateSystem(solved);
    if (initial.bodies.size() != solved.bodies.size() ||
        initial.constraints.size() != solved.constraints.size() ||
        initial.bodies.size() != orientations.size() ||
        initial.bodies.size() != positions.size() ||
        initial.bodies.size() != world_slots.size() ||
        contacts.size() > initial.constraints.size())
        throw std::runtime_error("Frozen equation body/constraint counts changed during solve");
    for (size_t i = 0; i < initial.constraints.size(); ++i)
        if ((i < contacts.size()) != (initial.constraints[i].kind == ji::Kind::Contact))
            throw std::runtime_error("Frozen contact keys do not align with contact constraints");
    out << std::setprecision(17) << "{\"schema\":1"
        << ",\"scene\":\"" << scenario
        << "\",\"step\":" << step << ",\"dt_s\":" << dt
        << ",\"solver\":\"subadmm\"";
    const auto numbers = [&](const auto &values, uint32_t count) {
        Array(out, std::span{values}.first(count));
    };
    out << ",\"gravity\":"; numbers(gravity, 3);
    out << ",\"bodies\":[";
    for (size_t i = 0; i < initial.bodies.size(); ++i) {
        if (i) out << ',';
        out << "{\"world_slot\":" << world_slots[i] << ",\"a\":";
        numbers(initial.bodies[i].a, 6);
        out << ",\"b\":"; numbers(initial.bodies[i].b, 6);
        out << ",\"v\":"; numbers(solved.bodies[i].v, 6);
        out << ",\"orientation_xyzw\":"; numbers(orientations[i], 4);
        out << ",\"position_m\":"; numbers(positions[i], 3);
        out << '}';
    }
    out << "],\"constraints\":[";
    for (size_t i = 0; i < initial.constraints.size(); ++i) {
        if (i) out << ',';
        const auto &before = initial.constraints[i];
        if (before.kind != solved.constraints[i].kind ||
            before.rows != solved.constraints[i].rows)
            throw std::runtime_error("Frozen equation constraint changed during solve");
        out << "{\"kind\":\"" <<
            (before.kind == ji::Kind::Contact ? "contact" :
             before.kind == ji::Kind::Hard ? "hard" : "soft")
            << "\",\"rows\":" << before.rows << ",\"friction\":" << before.friction
            << ",\"stiffness\":" << before.stiffness
            << ",\"damping\":" << before.damping;
        if (i < contacts.size()) {
            const auto key = ContactKeyFields(contacts[i].key);
            out << ",\"contact_key\":"; numbers(key, key.size());
        }
        out << ",\"error\":";
        numbers(before.error, before.rows);
        out << ",\"velocity_bias\":"; numbers(before.velocity_bias, before.rows);
        out << ",\"impulse\":"; numbers(solved.constraints[i].impulse, before.rows);
        out << ",\"sides\":[";
        for (uint32_t side = 0; side < 2; ++side) {
            if (side) out << ',';
            out << "{\"body\":" << before.side[side].body << ",\"j\":";
            numbers(before.side[side].j, 6 * before.rows);
            out << '}';
        }
        out << "]}";
    }
    out << "]}\n";
    if (!out) throw std::runtime_error("Could not write frozen equation output: " + path);
}

struct SpeedExtrema {
    double linear = 0, angular = 0;
    uint32_t linear_body = 0, angular_body = 0;
};

double MaxJointAnchorError(const rbp::World &world) {
    double error = 0;
    for (uint32_t index = 0; index < world.JointCount(); ++index) {
        const auto &joint = world.Joints[index];
        if (!joint.Active) continue;
        const rbp::float3 a = rbp::WorldPoint(world.Poses[joint.BodyA], joint.AnchorA);
        const rbp::float3 b = rbp::WorldPoint(world.Poses[joint.BodyB], joint.AnchorB);
        const rbp::float3 delta = a - b;
        error = std::max(error,
            std::hypot(double(delta.x), double(delta.y), double(delta.z)));
    }
    return error;
}

struct MechanicalEnergy {
    double kinetic = 0, gravitational = 0;
    double Total() const { return kinetic + gravitational; }
};

MechanicalEnergy MeasureEnergy(const rbp::World &world, ji::Vec3 gravity) {
    MechanicalEnergy energy;
    for (uint32_t body = 0; body < world.BodyCount(); ++body) {
        if (!world.Alive(body) || !(world.Masses[body].InvMass > 0)) continue;
        const auto &mass = world.Masses[body];
        const auto &pose = world.Poses[body];
        const auto &velocity = world.Velocities[body];
        const double m = 1 / double(mass.InvMass);
        const rbp::float3 linear = velocity.Linear;
        const rbp::float3 angular = rbp::Rotate(rbp::QuatConjugate(pose.Orientation), velocity.Angular);
        energy.kinetic += 0.5 * m * (double(linear.x) * linear.x +
            double(linear.y) * linear.y + double(linear.z) * linear.z);
        energy.kinetic += 0.5 * (double(angular.x) * angular.x / mass.InvInertiaLocal.x +
            double(angular.y) * angular.y / mass.InvInertiaLocal.y +
            double(angular.z) * angular.z / mass.InvInertiaLocal.z);
        energy.gravitational -= m * (gravity[0] * pose.Position.x +
            gravity[1] * pose.Position.y + gravity[2] * pose.Position.z);
    }
    return energy;
}

struct ContactWork {
    double normal = 0, tangent = 0, positive_normal = 0, positive_tangent = 0;
    double positive_normal_separated = 0, positive_normal_overlapping = 0;
    double largest_separated_gap = 0;
    uint32_t separated_count = 0;
};

ContactWork MeasureContactWork(const ji::System &system,
                               const std::vector<ji::SceneContact> &reports) {
    ContactWork work;
    for (size_t index = 0; index < system.constraints.size(); ++index) {
        const ji::Constraint &contact = system.constraints[index];
        if (contact.kind != ji::Kind::Contact) continue;
        if (index >= reports.size())
            throw std::runtime_error("Ji contact-work audit lacks a geometry report");
        double relative[3]{};
        for (const ji::Side &side : contact.side) {
            if (side.body == ji::NoBody) continue;
            for (uint32_t row = 0; row < 3; ++row)
                for (uint32_t axis = 0; axis < 6; ++axis)
                    relative[row] += side.j[6 * row + axis] * system.bodies[side.body].v[axis];
        }
        const double normal = contact.impulse[0] * relative[0];
        const double tangent = contact.impulse[1] * relative[1] +
                               contact.impulse[2] * relative[2];
        work.normal += normal;
        work.tangent += tangent;
        work.positive_normal += std::max(0.0, normal);
        work.positive_tangent += std::max(0.0, tangent);
        if (reports[index].gap > 0) {
            ++work.separated_count;
            work.largest_separated_gap = std::max(work.largest_separated_gap, reports[index].gap);
            work.positive_normal_separated += std::max(0.0, normal);
        } else work.positive_normal_overlapping += std::max(0.0, normal);
    }
    return work;
}

SpeedExtrema MeasureSpeed(const rbp::World &world) {
    SpeedExtrema result;
    for (uint32_t body = 0; body < world.BodyCount(); ++body) {
        if (!world.Alive(body) || !(world.Masses[body].InvMass > 0)) continue;
        const auto &velocity = world.Velocities[body];
        const double linear = std::hypot(double(velocity.Linear.x), double(velocity.Linear.y),
                                         double(velocity.Linear.z));
        const double angular = std::hypot(double(velocity.Angular.x), double(velocity.Angular.y),
                                          double(velocity.Angular.z));
        if (linear > result.linear) result.linear = linear, result.linear_body = body;
        if (angular > result.angular) result.angular = angular, result.angular_body = body;
    }
    return result;
}

} // namespace

int main(int argc, char **argv) {
    try {
        std::string name = "sliding";
        uint32_t bodies = 100, warmup = 1, steps = 3;
        bool advance_only = false, gpu_timing = false, cpu_timing = false;
        bool energy_trace = false, contact_events = false, final_state = false;
        std::optional<std::string> equation_output, trajectory_output;
        std::vector<uint32_t> equation_steps;
        std::map<std::string, double> options;
        const std::set<std::string> counts{"--iterations", "--penalty-period", "--contact-capacity", "--wrist-pulse-steps"};
        const std::set<std::string> numbers{"--dt-ms", "--contact-error-reduction", "--joint-error-reduction",
            "--penalty-offset", "--penalty-alpha", "--drive-amplitude", "--wrist-support",
            "--wrist-pulse-N", "--stack-shear-force", "--card-interlevel-opening-mm"};
        for (int i = 1; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "--help") {
                std::cout << "JiBenchmark --case <sliding|vertical|vertical-shear|oblique|cards|articulated|impact-demo>\n"
                    "  --bodies <sliding count> --warmup <count> --steps <count>\n"
                    "  --advance-only --gpu-timing --cpu-timing --energy --contact-events --final-state\n"
                    "  --iterations <count> --penalty-period <count> --contact-capacity <count>\n"
                    "  --dt-ms <ms> --penalty-offset <c> --penalty-alpha <alpha>\n"
                    "  --contact-error-reduction <value> --joint-error-reduction <value>\n"
                    "  --drive-amplitude <N m> --wrist-support <N>\n"
                    "  --wrist-pulse-N <N> --wrist-pulse-steps <count> (articulated; absolute steps [0,count))\n"
                    "  --stack-shear-force <N> (vertical-shear) --card-interlevel-opening-mm <mm> (cards)\n"
                    "  --frozen-equations-output <json> --equations-step <absolute step> (repeatable)\n"
                    "  --trajectory-output <jsonl> (initial and completed box poses)\n";
                return 0;
            }
            if (option == "--advance-only") { advance_only = true; continue; }
            if (option == "--gpu-timing") { gpu_timing = true; continue; }
            if (option == "--cpu-timing") { cpu_timing = true; continue; }
            if (option == "--energy") { energy_trace = true; continue; }
            if (option == "--contact-events") { contact_events = true; continue; }
            if (option == "--final-state") { final_state = true; continue; }
            if (++i >= argc) throw std::invalid_argument("Option requires a value: " + option);
            if (option == "--case") name = argv[i];
            else if (option == "--bodies") bodies = ParseCount(argv[i]);
            else if (option == "--warmup") warmup = ParseCount(argv[i]);
            else if (option == "--steps") steps = ParseCount(argv[i]);
            else if (option == "--equations-step") equation_steps.push_back(ParseCount(argv[i]));
            else if (option == "--frozen-equations-output") equation_output = argv[i];
            else if (option == "--trajectory-output") trajectory_output = argv[i];
            else if (counts.contains(option)) options[option] = ParseCount(argv[i]);
            else if (numbers.contains(option)) options[option] = ParseNonnegativeFinite(argv[i]);
            else throw std::invalid_argument("Unknown benchmark option: " + option);
        }
        if (!steps || steps > 10000 || warmup > 1000)
            throw std::invalid_argument("Require 1-10000 measured steps and at most 1000 warmup steps");
        if (options.contains("--wrist-pulse-N") != options.contains("--wrist-pulse-steps"))
            throw std::invalid_argument("Supply wrist pulse force and duration together");
        auto config = ji::ScenarioFor(name, bodies);
        const auto set = [&](const char *option, auto &value, double scale = 1) {
            if (options.contains(option)) value = options.at(option) * scale;
        };
        set("--iterations", config.settings.solve.iterations);
        set("--penalty-period", config.settings.solve.penalty_period);
        set("--contact-capacity", config.settings.contact_capacity);
        set("--dt-ms", config.settings.dt, 0.001);
        set("--contact-error-reduction", config.settings.contact_error_reduction);
        set("--joint-error-reduction", config.settings.joint_error_reduction);
        set("--penalty-offset", config.settings.penalty_offset);
        set("--penalty-alpha", config.settings.solve.alpha);
        for (const char *option : {"--drive-amplitude", "--wrist-support", "--wrist-pulse-N", "--wrist-pulse-steps"})
            if (options.contains(option) && config.kind != ji::ScenarioKind::Articulated)
                throw std::invalid_argument(std::string(option) + " requires articulated");
        if (options.contains("--stack-shear-force") && config.kind != ji::ScenarioKind::VerticalShear)
            throw std::invalid_argument("Stack shear force requires vertical-shear");
        if (options.contains("--card-interlevel-opening-mm") && config.kind != ji::ScenarioKind::Cards)
            throw std::invalid_argument("Card opening requires cards");
        set("--drive-amplitude", config.drive_amplitude);
        set("--wrist-support", config.wrist_support_N);
        set("--wrist-pulse-N", config.wrist_pulse_N);
        set("--wrist-pulse-steps", config.wrist_pulse_steps);
        set("--stack-shear-force", config.stack_shear_force_N);
        set("--card-interlevel-opening-mm", config.card_interlevel_opening_m, 0.001);
        for (const auto &[option, value] : options)
            if (!std::isfinite(float(value)) ||
                (value == 0 && ((counts.contains(option) && option != "--penalty-period") || option == "--dt-ms" || option == "--penalty-offset" ||
                 option == "--wrist-pulse-N" || option == "--card-interlevel-opening-mm")))
                throw std::invalid_argument("Positive representable value required for " + option);
        if (!(float(config.settings.dt) > 0) ||
            (config.kind == ji::ScenarioKind::Cards && !(float(config.card_interlevel_opening_m) > 0)) ||
            (options.contains("--wrist-pulse-N") && !(float(config.wrist_pulse_N) > 0)) ||
            !std::isfinite(float(config.wrist_support_N + config.wrist_pulse_N)) || config.settings.solve.alpha < 1)
            throw std::invalid_argument("Invalid timestep, card opening, wrist force or penalty alpha");
        if (bool(equation_output) != !equation_steps.empty())
            throw std::invalid_argument("Supply equation output and selected steps together");
        std::sort(equation_steps.begin(), equation_steps.end());
        equation_steps.erase(std::unique(equation_steps.begin(), equation_steps.end()), equation_steps.end());
        for (uint32_t step : equation_steps)
            if (step < warmup || step >= warmup + steps)
                throw std::invalid_argument("Equation capture must be in the measured window");
        if (advance_only && (energy_trace || contact_events || final_state || equation_output || trajectory_output))
            throw std::invalid_argument("Audit outputs require Scene::Step");
        config.settings.profile_gpu = gpu_timing;
        const rbp::mtl::Context context;
        rbp::World world{context, ji::LimitsFor(config)};
        ji::PopulateScenario(world, config);
        ji::Scene scene{context};
        for (uint32_t step = 0; step < warmup; ++step) {
            const auto wrenches = ji::ScenarioWrenches(config, world, step);
            if (advance_only) scene.Advance(world, config.settings, wrenches);
            else scene.Step(world, config.settings, wrenches);
        }
        std::map<std::string, std::vector<double>> series;
        const auto add = [&](const std::string &key, double value) {
            if (!std::isfinite(value)) throw std::runtime_error("Nonfinite benchmark metric: " + key);
            series[key].push_back(value);
        };
        ContactEvents events{warmup == 0};
        std::vector<rbp::float3> initial_positions;
        std::vector<bool> joint_body(world.BodyCount(), false);
        for (uint32_t body = 0; body < world.BodyCount(); ++body) initial_positions.push_back(world.Poses[body].Position);
        for (uint32_t index = 0; index < world.JointCount(); ++index) {
            const auto &joint = world.Joints[index];
            if (joint.Active) joint_body[joint.BodyA] = joint_body[joint.BodyB] = true;
        }
        const auto initial_energy = MeasureEnergy(world, config.settings.gravity);
        const double initial_anchor_error = MaxJointAnchorError(world);
        double previous_energy = initial_energy.Total(), max_energy_increase = 0;
        uint64_t peak_device_bytes = 0;
        std::ostringstream captures;
        bool first_capture = true;
        std::ofstream trajectory;
        if (trajectory_output) {
            trajectory.open(*trajectory_output);
            if (!trajectory) throw std::runtime_error("Cannot open trajectory output");
            trajectory << std::setprecision(9) << "{\"schema\":1,\"scene\":" << std::quoted(name)
                << ",\"solver\":\"subadmm\",\"dt_s\":" << config.settings.dt << ",\"boxes\":[";
            bool first = true;
            for (uint32_t body = 0; body < world.BodyCount(); ++body) {
                if (!world.Alive(body) || !(world.Masses[body].InvMass > 0)) continue;
                const auto &shape = world.Shapes[world.BodyShapes[body]];
                if (shape.Kind != rbp::ShapeBox) throw std::runtime_error("Trajectory requires box bodies");
                trajectory << (first ? "" : ",") << '[' << body << ',' << shape.HalfExtents.x << ','
                    << shape.HalfExtents.y << ',' << shape.HalfExtents.z << ']';
                first = false;
            }
            trajectory << "]}\n";
        }
        const auto frame = [&](int64_t step, const ji::SceneResult *result) {
            if (!trajectory_output) return;
            trajectory << "{\"completed_step\":" << step << ",\"contacts\":";
            if (result) trajectory << result->contacts.size(); else trajectory << "null";
            trajectory << ",\"contact_law\":";
            if (result) trajectory << result->audit.contact_law; else trajectory << "null";
            trajectory << ",\"poses\":[";
            bool first = true;
            for (uint32_t body = 0; body < world.BodyCount(); ++body) {
                if (!world.Alive(body) || !(world.Masses[body].InvMass > 0)) continue;
                const auto &p = world.Poses[body];
                trajectory << (first ? "" : ",") << '[' << body << ',' << p.Position.x << ',' << p.Position.y
                    << ',' << p.Position.z << ',' << p.Orientation.x << ',' << p.Orientation.y << ','
                    << p.Orientation.z << ',' << p.Orientation.w << ']';
                first = false;
            }
            trajectory << "]}\n" << std::flush;
            if (!trajectory) throw std::runtime_error("Cannot write trajectory output");
        };
        frame(int64_t(warmup) - 1, nullptr);
        for (uint32_t step = warmup; step < warmup + steps; ++step) {
            const bool capture = std::binary_search(equation_steps.begin(), equation_steps.end(), step);
            config.settings.capture_initial_system = capture;
            std::vector<std::array<double, 4>> orientations;
            std::vector<std::array<double, 3>> positions;
            std::vector<uint32_t> slots;
            if (capture) for (uint32_t body = 0; body < world.BodyCount(); ++body) {
                if (!world.Alive(body) || !(world.Masses[body].InvMass > 0)) continue;
                const auto &p = world.Poses[body];
                orientations.push_back({p.Orientation.x, p.Orientation.y, p.Orientation.z, p.Orientation.w});
                positions.push_back({p.Position.x, p.Position.y, p.Position.z});
                slots.push_back(body);
            }
            const auto begin = std::chrono::steady_clock::now();
            ji::SceneAdvance advanced;
            ji::SceneResult result;
            if (advance_only) advanced = scene.Advance(world, config.settings, ji::ScenarioWrenches(config, world, step));
            else result = scene.Step(world, config.settings, ji::ScenarioWrenches(config, world, step));
            add("wall_ms", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count());
            const auto residual = advance_only ? advanced.residual : result.solve.residual;
            add("contacts", advance_only ? advanced.geometry_features - advanced.static_only_features : result.contacts.size());
            add("warm_started", advance_only ? advanced.warm_started : result.warm_started);
            add("iterations", advance_only ? advanced.iterations : result.solve.iterations);
            add("primal_residual", residual.primal); add("dual_residual", residual.dual);
            add("dynamics_residual", residual.dynamics); add("contact_law_residual", residual.contact_law);
            const auto stages = advance_only ? advanced.gpu_stages : result.gpu_stages;
            if (gpu_timing && !stages) throw std::runtime_error("Requested GPU timestamps unavailable");
            if (stages) {
                add("gpu_collision_ms", stages->collision_ms); add("gpu_construction_ms", stages->construction_ms);
                add("gpu_preparation_ms", stages->preparation_ms); add("gpu_solve_ms", stages->solve_ms);
                add("gpu_finalize_ms", stages->finalize_ms);
                add("gpu_body_update_sample_ms", stages->body_update_sample_ms);
                add("gpu_constraint_update_sample_ms", stages->constraint_update_sample_ms);
            }
            if (cpu_timing) {
                const auto &cpu = advance_only ? advanced.cpu_stages : result.cpu_stages;
                add("cpu_setup_ms", cpu.setup_ms); add("cpu_encode_ms", cpu.encode_ms);
                add("cpu_commit_ms", cpu.commit_ms); add("cpu_wait_ms", cpu.wait_ms);
                add("cpu_geometry_ms", cpu.geometry_ms); add("cpu_decode_ms", cpu.decode_ms);
                add("cpu_audit_ms", cpu.audit_ms); add("cpu_report_ms", cpu.report_ms);
            }
            peak_device_bytes = std::max(peak_device_bytes, uint64_t(context.Device->currentAllocatedSize()));
            if (advance_only) continue;
            frame(step, &result);
            if (contact_events) events.Add(result, step, config.settings.dt);
            const auto speed = MeasureSpeed(world);
            add("max_linear_speed", speed.linear); add("max_angular_speed", speed.angular);
            add("fastest_linear_body", speed.linear_body); add("fastest_angular_body", speed.angular_body);
            double penetration = 0, min_beta = std::numeric_limits<double>::infinity(), max_beta = 0;
            uint32_t joint_contacts = 0;
            for (const auto &contact : result.contacts) {
                penetration = std::max(penetration, -contact.gap);
                joint_contacts += joint_body[contact.key.a.Slot] || joint_body[contact.key.b.Slot];
            }
            for (const auto &body : result.solve.system.bodies) {
                min_beta = std::min(min_beta, body.beta); max_beta = std::max(max_beta, body.beta);
            }
            add("max_penetration", penetration); add("joint_body_contacts", joint_contacts);
            add("min_beta", min_beta); add("max_beta", max_beta);
            if (config.joints) {
                double displacement = 0, object_speed = 0;
                uint32_t moved = 0;
                for (uint32_t body = 0; body < world.BodyCount(); ++body) {
                    if (joint_body[body] || !(world.Masses[body].InvMass > 0)) continue;
                    const auto delta = world.Poses[body].Position - initial_positions[body];
                    const double distance = std::hypot(double(delta.x), double(delta.y), double(delta.z));
                    const auto v = world.Velocities[body].Linear;
                    displacement = std::max(displacement, distance);
                    object_speed = std::max(object_speed, std::hypot(double(v.x), double(v.y), double(v.z)));
                    moved += distance > 0.001;
                }
                add("free_object_max_displacement_m", displacement); add("free_object_max_speed_mps", object_speed);
                add("free_objects_moved_1mm", moved); add("joint_max_anchor_error_m", MaxJointAnchorError(world));
            }
            const auto &audit = result.audit;
            add("audit_dynamics", audit.dynamics); add("audit_dynamics_scale", audit.dynamics_scale);
            add("audit_dynamics_body", audit.dynamics_body); add("audit_dynamics_axis", audit.dynamics_axis);
            add("audit_hard", audit.hard); add("audit_soft", audit.soft); add("audit_normal", audit.normal);
            add("audit_normal_complementarity", audit.normal_complementarity); add("audit_cone", audit.cone);
            add("audit_contact_law", audit.contact_law); add("audit_positive_friction_work", audit.positive_friction_work);
            if (energy_trace) {
                const auto energy = MeasureEnergy(world, config.settings.gravity);
                const auto work = MeasureContactWork(result.solve.system, result.contacts);
                max_energy_increase = std::max(max_energy_increase, energy.Total() - previous_energy);
                previous_energy = energy.Total();
                add("kinetic_energy_j", energy.kinetic); add("gravitational_energy_j", energy.gravitational);
                add("mechanical_energy_j", energy.Total()); add("contact_normal_work_proxy_j", work.normal);
                add("contact_tangent_work_proxy_j", work.tangent);
                add("positive_normal_contact_work_proxy_j", work.positive_normal);
                add("positive_tangent_contact_work_proxy_j", work.positive_tangent);
                add("positive_normal_separated_work_proxy_j", work.positive_normal_separated);
                add("positive_normal_overlapping_work_proxy_j", work.positive_normal_overlapping);
                add("separated_contact_count", work.separated_count);
                add("largest_separated_contact_gap_m", work.largest_separated_gap);
            }
            if (capture) {
                if (!result.initial_system) throw std::runtime_error("Initial equation capture unavailable");
                const auto path = CapturePath(*equation_output, uint32_t(equation_steps.size()), step);
                WriteFrozenEquations(path, *result.initial_system, result.solve.system, name, step,
                    config.settings.dt, config.settings.gravity, orientations, positions, slots, result.contacts);
                captures << (first_capture ? "" : ",") << "{\"step\":" << step << ",\"path\":" << std::quoted(path) << '}';
                first_capture = false;
            }
        }
        std::ostringstream out;
        out << std::setprecision(9) << std::boolalpha
            << "{\"schema\":1,\"record\":\"ji_benchmark\",\"solver\":\"subadmm\",\"source\":\"reauthored\""
            << ",\"scenario\":" << std::quoted(config.name) << ",\"geometry\":" << std::quoted(config.geometry)
            << ",\"drive\":" << std::quoted(config.drive) << ",\"drive_amplitude_Nm\":" << config.drive_amplitude
            << ",\"wrist_support_N\":" << config.wrist_support_N << ",\"wrist_pulse_N\":" << config.wrist_pulse_N
            << ",\"wrist_pulse_steps\":" << config.wrist_pulse_steps << ",\"stack_shear_force_N\":" << config.stack_shear_force_N
            << ",\"card_interlevel_opening_mm\":" << config.card_interlevel_opening_m * 1000
            << ",\"device\":" << std::quoted(context.Device->name()->utf8String())
            << ",\"dynamic_bodies\":" << config.dynamic_bodies << ",\"joints\":" << config.joints
            << ",\"dt_ms\":" << config.settings.dt * 1000 << ",\"configured_iterations\":" << config.settings.solve.iterations
            << ",\"penalty_period\":" << config.settings.solve.penalty_period << ",\"penalty_mass\":" << config.settings.penalty_mass
            << ",\"penalty_offset\":" << config.settings.penalty_offset << ",\"penalty_alpha\":" << config.settings.solve.alpha
            << ",\"contact_error_reduction\":" << config.settings.contact_error_reduction
            << ",\"joint_error_reduction\":" << config.settings.joint_error_reduction
            << ",\"collision_margin\":" << config.settings.collision_margin << ",\"max_contact_reach\":" << config.settings.max_contact_reach
            << ",\"contact_capacity\":" << config.settings.contact_capacity
            << ",\"warmup_steps\":" << warmup << ",\"requested_steps\":" << steps << ",\"measured_steps\":" << steps
            << ",\"trajectory_complete\":true,\"mechanics_audited\":" << !advance_only
            << ",\"gpu_timing_enabled\":" << gpu_timing << ",\"cpu_timing_enabled\":" << cpu_timing
            << ",\"timing_scope\":" << std::quoted(advance_only ? "Synchronous Scene::Advance without CPU contact decode/audit" : "Synchronous Scene::Step including CPU contact decode/audit")
            << ",\"wall_ms_p50\":" << Percentile(series.at("wall_ms"), 0.5)
            << ",\"wall_ms_p95\":" << Percentile(series.at("wall_ms"), 0.95)
            << ",\"wall_ms_worst\":" << *std::max_element(series.at("wall_ms").begin(), series.at("wall_ms").end())
            << ",\"peak_device_allocated_bytes\":" << peak_device_bytes;
        if (config.joints) out << ",\"initial_joint_anchor_error_m\":" << initial_anchor_error;
        if (energy_trace) out << ",\"initial_mechanical_energy_j\":" << initial_energy.Total()
            << ",\"net_mechanical_energy_change_j\":" << previous_energy - initial_energy.Total()
            << ",\"max_step_mechanical_energy_increase_j\":" << max_energy_increase;
        for (const auto &[key, values] : series) { out << ',' << std::quoted(key) << ':'; Array(out, values); }
        out << ",\"equations_captures\":[" << captures.str() << ']';
        if (contact_events) events.Write(out);
        if (final_state) {
            out << ",\"dynamic_body_slots\":[";
            bool first = true;
            for (uint32_t body = 0; body < world.BodyCount(); ++body) {
                if (!world.Alive(body) || !(world.Masses[body].InvMass > 0)) continue;
                out << (first ? "" : ",") << body; first = false;
            }
            out << "],\"final_state\":[";
            for (uint32_t body = 0; body < world.BodyCount(); ++body) {
                const auto &p = world.Poses[body]; const auto &v = world.Velocities[body];
                out << (body ? "," : "") << '[' << p.Position.x << ',' << p.Position.y << ',' << p.Position.z
                    << ',' << p.Orientation.x << ',' << p.Orientation.y << ',' << p.Orientation.z << ',' << p.Orientation.w
                    << ',' << v.Linear.x << ',' << v.Linear.y << ',' << v.Linear.z
                    << ',' << v.Angular.x << ',' << v.Angular.y << ',' << v.Angular.z << ']';
            }
            out << ']';
        }
        std::cout << out.str() << "}\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "Ji benchmark: " << error.what() << '\n';
        return 1;
    }
}
