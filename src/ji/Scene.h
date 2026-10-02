#pragma once

#include "Audit.h"
#include "Metal.h"
#include "Solver.h"

#include <array>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ji {

using Vec3 = std::array<double, 3>;
struct Wrench { Vec3 force{}, torque{}; }; // world frame; torque about COM

struct ContactKey {
    rbp::BodyId a, b;
    rbp::Index shape_a, shape_b;
    uint32_t child_a, child_b, subshape_a, subshape_b, feature;
    bool operator<(const ContactKey &) const;
};

struct SceneSettings {
    double dt = 1.0 / 120;
    Vec3 gravity{0, -9.81, 0};
    double collision_margin = 0, max_contact_reach = 1;
    double contact_error_reduction = 1, joint_error_reduction = 1; // local stabilization choices
    uint32_t contact_capacity = 4096;
    bool profile_gpu = false; // precise timestamps perturb dispatch timing
    bool capture_initial_system = false; // one-step convergence diagnostic
    double penalty_mass = 16, penalty_offset = 1e5;
    Settings solve{.iterations = 100, .penalty_period = 20, .tolerance = 1e-4};
};

struct SceneContact {
    ContactKey key;
    Vec3 point_a{}, point_b{}, normal{};
    Vec3 impulse_on_a{}, force_on_a{}; // world frame, force in newtons
    double gap = 0, friction = 0;
};

struct SceneResult {
    uint64_t step = 0;
    uint32_t warm_started = 0;
    uint32_t geometry_features = 0, static_only_features = 0;
    std::vector<SceneContact> contacts;
    GpuResult solve;
    std::optional<System> initial_system;
    MechanicalAudit audit;
    struct GpuStages {
        double collision_ms{}, construction_ms{}, preparation_ms{}, solve_ms{}, finalize_ms{};
        double body_update_sample_ms{}, constraint_update_sample_ms{};
    };
    std::optional<GpuStages> gpu_stages;
    struct CpuStages {
        double setup_ms{}, encode_ms{}, commit_ms{}, wait_ms{};
        double geometry_ms{}, decode_ms{}, audit_ms{}, report_ms{};
    } cpu_stages;
};

struct SceneAdvance {
    uint64_t step = 0;
    uint32_t warm_started = 0, geometry_features = 0, static_only_features = 0;
    Residual residual;
    uint32_t iterations = 0;
    std::optional<SceneResult::GpuStages> gpu_stages;
    SceneResult::CpuStages cpu_stages;
};

// Uses RBP geometry and ownership, but never calls the AVBD Step/Advance.
class Scene {
public:
    explicit Scene(const rbp::mtl::Context &);
    ~Scene();
    SceneResult Step(rbp::World &, const SceneSettings & = {}, std::span<const Wrench> external = {});
    SceneAdvance Advance(rbp::World &, const SceneSettings & = {}, std::span<const Wrench> external = {});

private:
    SceneResult Run(rbp::World &, const SceneSettings &, std::span<const Wrench>, bool diagnostics);
    struct Workspace;
    const rbp::mtl::Context &Context;
    rbp::Solver Collision;
    std::unique_ptr<Workspace> Work;
    uint64_t Steps = 0;
};

} // namespace ji
