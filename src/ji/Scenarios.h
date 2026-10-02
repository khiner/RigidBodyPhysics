#pragma once

#include "Scene.h"

#include <string_view>
#include <vector>

namespace ji {

enum class ScenarioKind { Sliding, Vertical, VerticalShear, Oblique, Cards, Articulated, ImpactDemo };

struct ScenarioConfig {
    ScenarioKind kind;
    const char *name, *geometry, *drive = "none";
    uint32_t dynamic_bodies, joints;
    double drive_amplitude = 0;
    double wrist_support_N = 0;
    double wrist_pulse_N = 0;
    uint32_t wrist_pulse_steps = 0;
    double stack_shear_force_N = 0;
    double card_interlevel_opening_m = 0;
    SceneSettings settings;
};

ScenarioConfig ScenarioFor(std::string_view name, uint32_t sliding_bodies = 100);
rbp::WorldLimits LimitsFor(const ScenarioConfig &);
void PopulateScenario(rbp::World &, const ScenarioConfig &);
std::vector<Wrench> ScenarioWrenches(const ScenarioConfig &, const rbp::World &, uint64_t step);

} // namespace ji
