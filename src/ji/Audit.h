#pragma once

#include "Oracle.h"

namespace ji {

// Independent final-state checks of Eq. (1) and contact/constraint laws.
// Each field retains its own units; callers choose scale-aware tolerances.
struct MechanicalAudit {
    double dynamics = 0;
    uint32_t dynamics_body = 0, dynamics_axis = 0;
    double dynamics_signed = 0, dynamics_scale = 0;
    double hard = 0;
    double soft = 0;
    double normal = 0;
    double normal_complementarity = 0;
    double cone = 0;
    double contact_law = 0; // strict normal-first/tangent-disk natural-map residual in solver multiplier units
    double positive_friction_work = 0;
    double dual_impulse = 0;
};

MechanicalAudit AuditFinal(const System &);

// Equation residuals for one frozen ADMM pass before any penalty refresh.
// `before` is the input state; `after` and `reported` are the completed pass.
struct IterationAudit {
    double body = 0, x = 0, z = 0, impulse = 0, dual_update = 0;
    double primal_residual = 0, dual_residual = 0, dynamics_residual = 0;
};
IterationAudit AuditIteration(const System &before, const System &after, Residual reported);

} // namespace ji
