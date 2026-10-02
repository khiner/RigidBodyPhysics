#include "Audit.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ji {

MechanicalAudit AuditFinal(const System &system) {
    MechanicalAudit out;
    std::vector<Vec> impulse(system.bodies.size());
    for (const Constraint &constraint : system.constraints) {
        Vec relative{};
        for (const Side &side : constraint.side) {
            if (side.body == NoBody) continue;
            for (uint32_t row = 0; row < constraint.rows; ++row) {
                for (uint32_t k = 0; k < 6; ++k) {
                    relative[row] += side.j[6 * row + k] * system.bodies[side.body].v[k];
                    impulse[side.body][k] += side.j[6 * row + k] * constraint.impulse[row];
                }
                out.dual_impulse = std::max(out.dual_impulse, std::abs(side.u[row] + constraint.impulse[row]));
            }
        }
        if (constraint.kind == Kind::Hard) {
            for (uint32_t row = 0; row < constraint.rows; ++row)
                out.hard = std::max(out.hard, std::abs(relative[row] + constraint.error[row]));
        } else if (constraint.kind == Kind::Soft) {
            for (uint32_t row = 0; row < constraint.rows; ++row)
                out.soft = std::max(out.soft, std::abs(constraint.impulse[row] +
                    constraint.stiffness * constraint.error[row] +
                    constraint.damping * (relative[row] + constraint.velocity_bias[row])));
        } else {
            const double normal_velocity = relative[0] + constraint.error[0];
            const double normal_impulse = constraint.impulse[0];
            const double tangent_velocity_1 = relative[1] + constraint.error[1];
            const double tangent_velocity_2 = relative[2] + constraint.error[2];
            out.contact_law = std::max(out.contact_law,
                ContactLawResidual(system, constraint, relative));
            const double tangent_impulse = std::hypot(constraint.impulse[1], constraint.impulse[2]);
            out.normal = std::max({out.normal, std::max(0.0, -normal_velocity), std::max(0.0, -normal_impulse)});
            out.normal_complementarity = std::max(out.normal_complementarity, std::abs(normal_impulse * normal_velocity));
            out.cone = std::max(out.cone, std::max(0.0, tangent_impulse - constraint.friction * normal_impulse));
            out.positive_friction_work = std::max(out.positive_friction_work,
                std::max(0.0, constraint.impulse[1] * tangent_velocity_1 + constraint.impulse[2] * tangent_velocity_2));
        }
    }
    for (uint32_t body = 0; body < system.bodies.size(); ++body)
        for (uint32_t k = 0; k < 6; ++k) {
            const double inertial = system.bodies[body].a[k] * system.bodies[body].v[k] -
                                    system.bodies[body].b[k];
            const double residual = inertial - impulse[body][k];
            if (std::abs(residual) > out.dynamics) {
                out.dynamics = std::abs(residual);
                out.dynamics_body = body;
                out.dynamics_axis = k;
                out.dynamics_signed = residual;
                out.dynamics_scale = std::max(std::abs(inertial), std::abs(impulse[body][k]));
            }
        }
    return out;
}

IterationAudit AuditIteration(const System &before, const System &after, Residual reported) {
    if (before.bodies.size() != after.bodies.size() || before.constraints.size() != after.constraints.size())
        throw std::invalid_argument("Iteration audit needs matching frozen systems");
    IterationAudit out;
    std::vector<Vec> body_equation(before.bodies.size()), dual(before.bodies.size()), force(before.bodies.size());
    Residual measured;
    for (uint32_t i = 0; i < before.bodies.size(); ++i)
        for (uint32_t k = 0; k < 6; ++k)
            body_equation[i][k] = before.bodies[i].a[k] * after.bodies[i].v[k] - before.bodies[i].b[k];
    for (uint32_t c = 0; c < before.constraints.size(); ++c) {
        const Constraint &old = before.constraints[c], &now = after.constraints[c];
        if (old.kind != now.kind || old.rows != now.rows)
            throw std::invalid_argument("Iteration audit constraint topology changed");
        double inverse_beta[2]{}, weighted_y[6]{};
        for (uint32_t s = 0; s < 2; ++s) {
            const Side &old_side = old.side[s], &side = now.side[s];
            if (old_side.body != side.body) throw std::invalid_argument("Iteration audit endpoint changed");
            if (side.body == NoBody) continue;
            const uint32_t id = side.body;
            const double beta = before.bodies[id].beta;
            inverse_beta[s] = 1 / beta;
            for (uint32_t r = 0; r < old.rows; ++r) {
                double mapped = 0;
                for (uint32_t k = 0; k < 6; ++k) {
                    const double j = old_side.j[6 * r + k];
                    mapped += j * after.bodies[id].v[k];
                    dual[id][k] += beta * j * (old_side.z[r] - side.z[r]);
                    force[id][k] += j * now.impulse[r];
                }
                for (uint32_t k = 0; k < 6; ++k)
                    body_equation[id][k] += old_side.j[6 * r + k] *
                        (beta * (mapped - old_side.z[r]) + old_side.u[r]);
                out.x = std::max(out.x, std::abs(side.x[r] - mapped));
                const double y = beta * side.x[r] + old_side.u[r];
                weighted_y[r] += y / beta;
                out.z = std::max(out.z, std::abs(beta * side.z[r] - y - now.impulse[r]));
                out.dual_update = std::max(out.dual_update,
                    std::abs(side.u[r] - old_side.u[r] - beta * (side.x[r] - side.z[r])));
                measured.primal = std::max(measured.primal, std::abs(side.x[r] - side.z[r]));
            }
        }
        const double inverse_sum = inverse_beta[0] + inverse_beta[1];
        for (uint32_t r = 0; r < old.rows; ++r) {
            double expected = 0;
            if (old.kind == Kind::Soft)
                expected = -(old.stiffness * old.error[r] +
                             old.damping * (weighted_y[r] + old.velocity_bias[r])) /
                           (1 + old.damping * inverse_sum);
            else expected = -(old.error[r] + weighted_y[r]) / inverse_sum;
            if (old.kind == Kind::Contact) {
                if (r == 0) expected = std::max(0.0, expected);
                else continue;
            }
            out.impulse = std::max(out.impulse, std::abs(now.impulse[r] - expected));
        }
        if (old.kind == Kind::Contact) {
            double trial[2]{};
            for (uint32_t r = 1; r < 3; ++r) trial[r - 1] = -(old.error[r] + weighted_y[r]) / inverse_sum;
            const double length = std::hypot(trial[0], trial[1]);
            const double radius = old.friction * now.impulse[0];
            if (length > radius) {
                trial[0] *= radius / length;
                trial[1] *= radius / length;
            }
            for (uint32_t r = 1; r < 3; ++r)
                out.impulse = std::max(out.impulse, std::abs(now.impulse[r] - trial[r - 1]));
        }
    }
    for (uint32_t i = 0; i < before.bodies.size(); ++i)
        for (uint32_t k = 0; k < 6; ++k) {
            out.body = std::max(out.body, std::abs(body_equation[i][k]));
            measured.dual = std::max(measured.dual, std::abs(dual[i][k]));
            measured.dynamics = std::max(measured.dynamics,
                std::abs(after.bodies[i].a[k] * after.bodies[i].v[k] - after.bodies[i].b[k] - force[i][k]));
        }
    out.primal_residual = std::abs(measured.primal - reported.primal);
    out.dual_residual = std::abs(measured.dual - reported.dual);
    out.dynamics_residual = std::abs(measured.dynamics - reported.dynamics);
    return out;
}

} // namespace ji
