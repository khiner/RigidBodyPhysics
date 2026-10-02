#include "Oracle.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ji {
namespace {

double Entry(const Mat &m, uint32_t row, uint32_t col) { return m[6 * row + col]; }

Vec Multiply(const Side &side, const Vec &v, uint32_t rows) {
    Vec out{};
    for (uint32_t r = 0; r < rows; ++r)
        for (uint32_t k = 0; k < 6; ++k) out[r] += Entry(side.j, r, k) * v[k];
    return out;
}

void AddTranspose(Vec &out, const Side &side, const Vec &x, uint32_t rows, double scale = 1) {
    for (uint32_t k = 0; k < 6; ++k)
        for (uint32_t r = 0; r < rows; ++r) out[k] += scale * Entry(side.j, r, k) * x[r];
}

Vec CholeskySolve(Mat a, Vec b) {
    Mat l{};
    for (uint32_t i = 0; i < 6; ++i) {
        for (uint32_t j = 0; j <= i; ++j) {
            double value = a[6 * i + j];
            for (uint32_t k = 0; k < j; ++k) value -= l[6 * i + k] * l[6 * j + k];
            if (i == j) {
                if (!(value > 0) || !std::isfinite(value)) throw std::runtime_error("SubADMM body matrix is not SPD");
                l[6 * i + j] = std::sqrt(value);
            } else l[6 * i + j] = value / l[6 * j + j];
        }
    }
    for (uint32_t i = 0; i < 6; ++i) {
        for (uint32_t k = 0; k < i; ++k) b[i] -= l[6 * i + k] * b[k];
        b[i] /= l[6 * i + i];
    }
    for (int i = 5; i >= 0; --i) {
        for (int k = i + 1; k < 6; ++k) b[i] -= l[6 * k + i] * b[k];
        b[i] /= l[6 * i + i];
    }
    return b;
}

bool Finite(const Vec &v) { return std::all_of(v.begin(), v.end(), [](double x) { return std::isfinite(x); }); }
bool Finite(const Mat &m) { return std::all_of(m.begin(), m.end(), [](double x) { return std::isfinite(x); }); }

} // namespace

void ValidateSystem(const System &system, bool require_penalties) {
    for (const Body &body : system.bodies) {
        for (double a : body.a)
            if (!(a > 0) || !std::isfinite(a)) throw std::invalid_argument("SubADMM body A must be positive diagonal");
        if (!Finite(body.b) || !Finite(body.v)) throw std::invalid_argument("SubADMM body data must be finite");
        if (require_penalties && (!(body.beta > 0) || !std::isfinite(body.beta)))
            throw std::invalid_argument("SubADMM beta must be positive");
    }
    for (const Constraint &constraint : system.constraints) {
        if (constraint.kind != Kind::Soft && constraint.kind != Kind::Hard && constraint.kind != Kind::Contact)
            throw std::invalid_argument("Invalid SubADMM constraint kind");
        if (constraint.rows != (constraint.kind == Kind::Contact ? 3u : 6u))
            throw std::invalid_argument("SubADMM constraint rows must be three for contact or six otherwise");
        if (constraint.side[0].body == NoBody && constraint.side[1].body == NoBody)
            throw std::invalid_argument("SubADMM constraint needs a moving endpoint");
        if (!Finite(constraint.error) || !Finite(constraint.impulse) || !Finite(constraint.velocity_bias))
            throw std::invalid_argument("SubADMM constraint data must be finite");
        if (!(constraint.stiffness >= 0) || !std::isfinite(constraint.stiffness) ||
            !(constraint.damping >= 0) || !std::isfinite(constraint.damping))
            throw std::invalid_argument("SubADMM stiffness and damping must be nonnegative finite");
        if (constraint.kind == Kind::Contact && (!(constraint.friction >= 0) || !std::isfinite(constraint.friction)))
            throw std::invalid_argument("SubADMM friction must be nonnegative");
        for (const Side &side : constraint.side)
            if (side.body != NoBody) {
                if (side.body >= system.bodies.size()) throw std::invalid_argument("SubADMM endpoint body index out of range");
                if (!Finite(side.j) || !Finite(side.x) || !Finite(side.z) || !Finite(side.u))
                    throw std::invalid_argument("SubADMM side data must be finite");
            }
    }
}

void InitializePenalties(System &system, double M, double c) {
    if (!(M >= 0) || !(c > 0)) throw std::invalid_argument("Invalid SubADMM penalty settings");
    ValidateSystem(system, false);
    std::vector<double> trace(system.bodies.size());
    for (const Constraint &constraint : system.constraints)
        for (const Side &side : constraint.side)
            if (side.body != NoBody)
                for (uint32_t r = 0; r < constraint.rows; ++r)
                    for (uint32_t k = 0; k < 6; ++k) trace[side.body] += std::pow(Entry(side.j, r, k), 2);
    for (uint32_t i = 0; i < system.bodies.size(); ++i) {
        double a_trace = 0;
        for (double a : system.bodies[i].a) a_trace += a;
        system.bodies[i].beta = trace[i] > 0 ? M * a_trace / trace[i] + c : c;
    }
    ValidateSystem(system);
}

void UpdatePenalties(System &system, Residual residual, double alpha) {
    if (!(alpha >= 1) || !std::isfinite(alpha)) throw std::invalid_argument("Invalid SubADMM penalty clamp");
    if (!(residual.primal >= 0) || !std::isfinite(residual.primal) ||
        !(residual.dual >= 0) || !std::isfinite(residual.dual))
        throw std::invalid_argument("Invalid SubADMM penalty residual");
    double ratio = 1;
    if (residual.dual == 0) ratio = residual.primal > 0 ? alpha : 1;
    else ratio = std::clamp(residual.primal / residual.dual, 1 / alpha, alpha);
    for (Body &body : system.bodies) body.beta *= ratio;
}

std::array<double, 3> StrictContactProjection(std::array<double, 3> trial, double mu) {
    if (!(mu >= 0) || !std::isfinite(mu)) throw std::invalid_argument("Invalid SubADMM friction");
    const double normal = std::max(0.0, trial[0]);
    const double radius = mu * normal;
    const double tangent = std::hypot(trial[1], trial[2]);
    if (tangent > radius) {
        const double scale = radius / tangent;
        trial[1] *= scale;
        trial[2] *= scale;
    }
    trial[0] = normal;
    return trial;
}

double ContactLawResidual(const System &system, const Constraint &constraint, const Vec &relative) {
    if (constraint.kind != Kind::Contact) return 0;
    double normal_mobility = 0;
    for (const Side &side : constraint.side) {
        if (side.body == NoBody) continue;
        for (uint32_t k = 0; k < 6; ++k)
            normal_mobility += side.j[k] * side.j[k] / system.bodies[side.body].a[k];
    }
    const double rho = 1 / std::max(normal_mobility, 1e-12);
    const auto projected = StrictContactProjection({
        constraint.impulse[0] - rho * (relative[0] + constraint.error[0]),
        constraint.impulse[1] - rho * (relative[1] + constraint.error[1]),
        constraint.impulse[2] - rho * (relative[2] + constraint.error[2])}, constraint.friction);
    double residual = 0;
    for (uint32_t row = 0; row < 3; ++row)
        residual = std::max(residual, std::abs(constraint.impulse[row] - projected[row]));
    return residual;
}

Residual Iterate(System &system) {
    ValidateSystem(system);
    std::vector<Vec> old_z_a, old_z_b;
    old_z_a.reserve(system.constraints.size());
    old_z_b.reserve(system.constraints.size());
    for (const Constraint &constraint : system.constraints) {
        old_z_a.push_back(constraint.side[0].z);
        old_z_b.push_back(constraint.side[1].z);
    }

    // Eq. (6): each body can be solved independently from the previous z/u.
    for (uint32_t id = 0; id < system.bodies.size(); ++id) {
        Body &body = system.bodies[id];
        Mat matrix{};
        for (uint32_t k = 0; k < 6; ++k) matrix[6 * k + k] = body.a[k];
        Vec rhs = body.b;
        for (const Constraint &constraint : system.constraints)
            for (const Side &side : constraint.side) {
                if (side.body != id) continue;
                Vec scaled{};
                for (uint32_t r = 0; r < constraint.rows; ++r) scaled[r] = body.beta * side.z[r] - side.u[r];
                AddTranspose(rhs, side, scaled, constraint.rows);
                for (uint32_t r = 0; r < constraint.rows; ++r)
                    for (uint32_t j = 0; j < 6; ++j)
                        for (uint32_t k = 0; k < 6; ++k)
                            matrix[6 * j + k] += body.beta * Entry(side.j, r, j) * Entry(side.j, r, k);
            }
        body.v = CholeskySolve(matrix, rhs);
    }

    // Eqs. (7)-(10): constraints are independent given the new velocities.
    for (Constraint &constraint : system.constraints) {
        Vec y[2]{};
        double inverse_beta[2]{};
        for (uint32_t s = 0; s < 2; ++s) {
            Side &side = constraint.side[s];
            if (side.body == NoBody) continue;
            const Body &body = system.bodies[side.body];
            inverse_beta[s] = 1 / body.beta;
            side.x = Multiply(side, body.v, constraint.rows);
            for (uint32_t r = 0; r < constraint.rows; ++r) y[s][r] = body.beta * side.x[r] + side.u[r];
        }
        const double inverse_sum = inverse_beta[0] + inverse_beta[1];
        Vec trial{};
        for (uint32_t r = 0; r < constraint.rows; ++r)
            trial[r] = -(constraint.error[r] + inverse_beta[0] * y[0][r] + inverse_beta[1] * y[1][r]) / inverse_sum;
        if (constraint.kind == Kind::Soft) {
            const double denominator = 1 + constraint.damping * inverse_sum;
            for (uint32_t r = 0; r < constraint.rows; ++r)
                constraint.impulse[r] = -(constraint.stiffness * constraint.error[r] +
                    constraint.damping * (inverse_beta[0] * y[0][r] + inverse_beta[1] * y[1][r] +
                                          constraint.velocity_bias[r])) / denominator;
        } else if (constraint.kind == Kind::Hard) constraint.impulse = trial;
        else {
            const auto projected = StrictContactProjection({trial[0], trial[1], trial[2]}, constraint.friction);
            for (uint32_t r = 0; r < 3; ++r) constraint.impulse[r] = projected[r];
        }
        for (Side &side : constraint.side) {
            if (side.body == NoBody) continue;
            const double beta = system.bodies[side.body].beta;
            for (uint32_t r = 0; r < constraint.rows; ++r) {
                side.z[r] = side.x[r] + (side.u[r] + constraint.impulse[r]) / beta;
                side.u[r] += beta * (side.x[r] - side.z[r]);
            }
        }
    }

    Residual residual;
    std::vector<Vec> dual(system.bodies.size()), forces(system.bodies.size());
    for (uint32_t n = 0; n < system.constraints.size(); ++n) {
        const Constraint &constraint = system.constraints[n];
        Vec relative{};
        for (uint32_t s = 0; s < 2; ++s) {
            const Side &side = constraint.side[s];
            if (side.body == NoBody) continue;
            Vec dz{};
            const Vec &old = s == 0 ? old_z_a[n] : old_z_b[n];
            for (uint32_t r = 0; r < constraint.rows; ++r) {
                relative[r] += side.x[r];
                residual.primal = std::max(residual.primal, std::abs(side.x[r] - side.z[r]));
                dz[r] = old[r] - side.z[r];
            }
            AddTranspose(dual[side.body], side, dz, constraint.rows, system.bodies[side.body].beta);
            AddTranspose(forces[side.body], side, constraint.impulse, constraint.rows);
        }
        residual.contact_law = std::max(residual.contact_law,
            ContactLawResidual(system, constraint, relative));
    }
    for (uint32_t i = 0; i < system.bodies.size(); ++i)
        for (uint32_t k = 0; k < 6; ++k) {
            residual.dual = std::max(residual.dual, std::abs(dual[i][k]));
            residual.dynamics = std::max(residual.dynamics,
                std::abs(system.bodies[i].a[k] * system.bodies[i].v[k] - system.bodies[i].b[k] - forces[i][k]));
        }
    return residual;
}

Result Solve(System &system, Settings settings) {
    if (!(settings.alpha >= 1) || !std::isfinite(settings.alpha) ||
        !(settings.tolerance >= 0) || !std::isfinite(settings.tolerance))
        throw std::invalid_argument("Invalid SubADMM solve settings");
    ValidateSystem(system);
    Result result;
    for (uint32_t i = 0; i < settings.iterations; ++i) {
        result.residual = Iterate(system);
        result.iterations = i + 1;
        result.converged = settings.tolerance > 0 && result.residual.primal <= settings.tolerance &&
            result.residual.dual <= settings.tolerance && result.residual.dynamics <= settings.tolerance &&
            result.residual.contact_law <= settings.tolerance;
        if (result.converged) break;
        if (i + 1 < settings.iterations && settings.penalty_period &&
            (i + 1) % settings.penalty_period == 0)
            UpdatePenalties(system, result.residual, settings.alpha);
    }
    return result;
}

} // namespace ji
