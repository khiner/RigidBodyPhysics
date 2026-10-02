#!/usr/bin/env python3
"""Independent coupled contact-law solve for one frozen frictional stack step.

Requires NumPy and SciPy. This solves the captured rigid velocity equations, not
Ji's ADMM iterates or a moving scene. Each disconnected contact component is
small enough for a dense double-precision Levenberg-Marquardt solve.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import least_squares, minimize

from NormalLcpReference import contact_groups


FIXTURE = Path(__file__).resolve().parents[2] / "tests/ji/fixtures/vertical_step7_frictional.json.gz"


def assemble(bodies, contacts, ids):
    body_ids = sorted({side["body"] for i in ids for side in contacts[i]["sides"]
                       if side["body"] < len(bodies)})
    local = {body: k for k, body in enumerate(body_ids)}
    a = np.asarray([bodies[i]["a"] for i in body_ids]).reshape(-1)
    b = np.asarray([bodies[i]["b"] for i in body_ids]).reshape(-1)
    j = np.zeros((3 * len(ids), len(a)))
    error = np.zeros(3 * len(ids))
    friction = np.empty(len(ids))
    for k, i in enumerate(ids):
        contact = contacts[i]
        error[3 * k:3 * k + 3] = contact["error"]
        friction[k] = contact["friction"]
        for side in contact["sides"]:
            body = side["body"]
            if body < len(bodies):
                start = 6 * local[body]
                j[3 * k:3 * k + 3, start:start + 6] += np.asarray(side["j"]).reshape(3, 6)
    if not np.all(np.isfinite(a)) or not np.all(a > 0):
        raise ValueError("All moving-body diagonal inertias must be positive and finite")
    # Explicit contraction avoids spurious Accelerate BLAS floating-point flags
    # seen for ordinary matmul on this input; both contractions were cross-checked.
    w = np.einsum("ik,jk,k->ij", j, j, 1 / a, optimize=False)
    q = np.einsum("ik,k->i", j, b / a, optimize=False) + error
    if not np.all(np.isfinite(w)) or not np.all(np.isfinite(q)):
        raise ValueError("Nonfinite contact system")
    return body_ids, a, b, j, w, q, friction


def project(trial, friction):
    contact = trial.reshape(-1, 3).copy()
    contact[:, 0] = np.maximum(contact[:, 0], 0)
    radius = friction * contact[:, 0]
    length = np.linalg.norm(contact[:, 1:], axis=1)
    contact[:, 1:] *= np.minimum(1, radius / np.maximum(length, 1e-100))[:, None]
    return contact.reshape(-1)


def projection_derivative(trial, friction):
    derivative = np.zeros((len(friction), 3, 3))
    for k, (normal, t1, t2) in enumerate(trial.reshape(-1, 3)):
        if normal <= 0:
            continue
        block = derivative[k]
        block[0, 0] = 1
        if friction[k] == 0:
            continue  # The zero-radius tangent disk is the constant zero map.
        tangent = np.array([t1, t2])
        length = np.linalg.norm(tangent)
        radius = friction[k] * normal
        if length <= radius:
            block[1, 1] = block[2, 2] = 1
        elif length > 0:
            direction = tangent / length
            block[1:, 0] = friction[k] * direction
            block[1:, 1:] = (
                radius / length * (np.eye(2) - np.outer(direction, direction)))
    return derivative


def solve(data, initial):
    bodies, contacts = data["bodies"], data["contacts"]
    a = np.asarray([body["a"] for body in bodies])
    b = np.asarray([body["b"] for body in bodies])
    velocity = b / a
    impulses = np.zeros((len(contacts), 3))
    saved = np.asarray([contact["impulse"] for contact in contacts])
    groups = contact_groups(contacts, len(bodies))
    evaluations, component_residuals, maximum_law_residual = [], [], 0.0
    fallback_components, homotopy_seed_components, limit_hit_components = 0, 0, 0
    normal_seed_components = 0
    for ids in groups:
        body_ids, diagonal, rhs, j, w, q, friction = assemble(bodies, contacts, ids)
        rho = np.repeat(1 / np.maximum(np.diag(w)[::3], 1e-12), 3)

        def law(z, matrix=w):
            return z - project(z - rho * (q + np.einsum("ij,j->i", matrix, z,
                                                       optimize=False)), friction)

        def derivative(z, matrix=w):
            trial = z - rho * (q + np.einsum("ij,j->i", matrix, z,
                                            optimize=False))
            dp = projection_derivative(trial, friction)
            step = np.eye(len(z)) - rho[:, None] * matrix
            return np.eye(len(z)) - np.einsum(
                "nik,nkj->nij", dp, step.reshape(-1, 3, len(z)),
                optimize=False).reshape(len(z), len(z))

        start = saved[ids].reshape(-1) if initial == "ji" else np.zeros(3 * len(ids))
        result = least_squares(law, start, jac=derivative, method="lm", x_scale="jac",
                               max_nfev=500, ftol=1e-13, xtol=1e-13, gtol=1e-13)
        residual = float(np.abs(law(result.x)).max())
        evaluations_used = result.nfev
        improved = False
        if initial == "ji" and residual > 2e-6:
            # Solve the convex normal-only problem for an active-set seed, then
            # solve and audit the original full-friction law without changing it.
            normal_matrix, normal_offset = w[::3, ::3], q[::3]

            def normal_objective(z):
                mapped = np.einsum("ij,j->i", normal_matrix, z, optimize=False)
                return (float(0.5 * np.einsum("i,i->", z, mapped, optimize=False) +
                              np.einsum("i,i->", normal_offset, z, optimize=False)),
                        normal_offset + mapped)

            normal = minimize(normal_objective, np.maximum(start[::3], 0), jac=True,
                              method="L-BFGS-B", bounds=[(0, None)] * len(ids),
                              options={"maxiter": 10000, "ftol": 0, "gtol": 1e-12,
                                       "maxls": 50})
            seed = np.zeros_like(start)
            seed[::3] = normal.x
            candidate = least_squares(law, seed, jac=derivative,
                                      method="lm", x_scale="jac", max_nfev=500,
                                      ftol=1e-13, xtol=1e-13, gtol=1e-13)
            evaluations_used += candidate.nfev
            candidate_residual = float(np.abs(law(candidate.x)).max())
            if candidate_residual < residual:
                result, residual = candidate, candidate_residual
                normal_seed_components += 1
                improved = True
        if initial == "ji" and residual > 1e-5:
            # A near-null contact direction can trap the rigid fixed-point solve.
            # Use a tiny diagonal perturbation only to find another initial
            # branch, then solve and audit the original, unregularized equations.
            seed_matrix = w + (2e-11 * np.linalg.norm(w, ord=2)) * np.eye(len(start))
            seed = least_squares(lambda z: law(z, seed_matrix), start,
                                 jac=lambda z: derivative(z, seed_matrix),
                                 method="lm", x_scale="jac", max_nfev=500,
                                 ftol=1e-13, xtol=1e-13, gtol=1e-13)
            candidate = least_squares(law, seed.x, jac=derivative,
                                      method="lm", x_scale="jac", max_nfev=500,
                                      ftol=1e-13, xtol=1e-13, gtol=1e-13)
            evaluations_used += seed.nfev + candidate.nfev
            candidate_residual = float(np.abs(law(candidate.x)).max())
            if candidate_residual < residual:
                result, residual = candidate, candidate_residual
                homotopy_seed_components += 1
                improved = True
            # A few other changing-geometry islands land on a nonsmooth LM
            # stationary point; retain the bounded branch search for those.
            for trial in (result.x, 0.5 * start, 1.5 * start, 0.25 * start):
                alternative = least_squares(law, trial, jac=derivative,
                                            method="trf", tr_solver="lsmr", x_scale="jac",
                                            max_nfev=500,
                                            ftol=1e-13, xtol=1e-13, gtol=1e-13)
                evaluations_used += alternative.nfev
                alternative_residual = float(np.abs(law(alternative.x)).max())
                if alternative_residual < residual:
                    result, residual = alternative, alternative_residual
                    improved = True
                if residual <= 2e-6:
                    break
        fallback_components += improved
        if (not np.all(np.isfinite(result.x)) or
                (not result.success and residual > 1e-4)):
            raise RuntimeError(f"Coupled contact component {len(evaluations)} with {len(ids)} "
                               f"contacts failed after {result.nfev} evaluations; "
                               f"max law residual={residual}: {result.message}")
        limit_hit_components += not result.success
        z = result.x
        maximum_law_residual = max(maximum_law_residual, residual)
        evaluations.append(evaluations_used)
        component_residuals.append({
            "component": len(evaluations) - 1,
            "contacts": len(ids),
            "world_body_slots": [bodies[body].get("world_slot", body) for body in body_ids],
            "max_contact_law_residual_impulse_units": residual,
        })
        impulses[ids] = z.reshape(-1, 3)
        velocity[body_ids] = ((rhs + np.einsum("ij,i->j", j, z,
                                              optimize=False)) / diagonal).reshape(-1, 6)
    return velocity, impulses, {
        "components": len(groups),
        "largest_component_contacts": max(map(len, groups)),
        "max_contact_law_residual_impulse_units": maximum_law_residual,
        "worst_contact_law_components": sorted(
            component_residuals,
            key=lambda item: item["max_contact_law_residual_impulse_units"],
            reverse=True)[:3],
        "fallback_components": fallback_components,
        "normal_seed_components": normal_seed_components,
        "homotopy_seed_components": homotopy_seed_components,
        "limit_hit_components": limit_hit_components,
        "median_function_evaluations_per_component": float(np.median(evaluations)),
        "max_function_evaluations_per_component": max(evaluations),
    }


def audit(data, velocity, impulses):
    bodies, contacts = data["bodies"], data["contacts"]
    a = np.asarray([body["a"] for body in bodies])
    b = np.asarray([body["b"] for body in bodies])
    force = np.zeros_like(velocity)
    relative = np.zeros_like(impulses)
    friction = np.asarray([contact["friction"] for contact in contacts])
    for i, contact in enumerate(contacts):
        for side in contact["sides"]:
            body = side["body"]
            if body < len(bodies):
                j = np.asarray(side["j"]).reshape(3, 6)
                relative[i] += j @ velocity[body]
                force[body] += impulses[i] @ j
    # The contact law uses Jv + gap bias; physical contact work uses Jv alone.
    contact_velocity = relative + np.asarray([contact["error"] for contact in contacts])
    normal_work = impulses[:, 0] * relative[:, 0]
    tangential_work = np.sum(impulses[:, 1:] * relative[:, 1:], axis=1)
    complementarity = impulses[:, 0] * contact_velocity[:, 0]
    cone_excess = np.maximum(np.linalg.norm(impulses[:, 1:], axis=1) -
                             friction * impulses[:, 0], 0)
    return {
        "max_negative_normal_impulse": float(np.maximum(-impulses[:, 0], 0).max()),
        "positive_normal_work_J": float(np.maximum(normal_work, 0).sum()),
        "max_normal_complementarity_J": float(np.abs(complementarity).max()),
        "max_negative_normal_velocity_scaled": float(np.maximum(-contact_velocity[:, 0], 0).max()),
        "positive_tangential_work_J": float(np.maximum(tangential_work, 0).sum()),
        "max_cone_excess_impulse_units": float(cone_excess.max()),
        "max_dynamics_residual_impulse_units": float(np.abs(a * velocity - b - force).max()),
    }


def rotate(orientation, local):
    twice_cross = 2 * np.cross(orientation[:, :3], local)
    return (local + orientation[:, 3, None] * twice_cross +
            np.cross(orientation[:, :3], twice_cross))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", nargs="?", type=Path, default=FIXTURE)
    args = parser.parse_args()
    fixture = args.fixture
    with (gzip.open if fixture.suffix == ".gz" else open)(fixture, "rt") as source:
        data = json.load(source)
    if "constraints" in data:
        if any(c["kind"] != "contact" or c["rows"] != 3 for c in data["constraints"]):
            raise ValueError("This reference solves contact-only frozen systems")
        data["contacts"] = data["constraints"]
    original_velocity = np.asarray([body["v"] for body in data["bodies"]])
    original_impulses = np.asarray([contact["impulse"] for contact in data["contacts"]])
    warm_velocity, warm_impulses, statistics = solve(data, "ji")
    cold_error = None
    try:
        cold_velocity, cold_impulses, cold_statistics = solve(data, "zero")
    except RuntimeError as error:
        cold_error = str(error)
    warm_audit = audit(data, warm_velocity, warm_impulses)
    cold_audit = audit(data, cold_velocity, cold_impulses) if cold_error is None else None
    def passes(state, stats):
        return (stats["max_contact_law_residual_impulse_units"] <= 2e-6 and
                state["max_negative_normal_impulse"] <= 1e-6 and
                state["max_normal_complementarity_J"] <= 1e-8 and
                state["max_negative_normal_velocity_scaled"] <= 1e-8 and
                state["positive_tangential_work_J"] <= 1e-5 and
                state["max_cone_excess_impulse_units"] <= 1e-6 and
                state["max_dynamics_residual_impulse_units"] <= 1e-8)
    passed = {"ji_start": passes(warm_audit, statistics),
              "zero_start": cold_error is None and passes(cold_audit, cold_statistics)}
    a = np.asarray([body["a"] for body in data["bodies"]])
    kinetic_change = float(np.sum(0.5 * a * (warm_velocity**2 - original_velocity**2)))
    energy_comparison = {"kinetic_energy_change_J": kinetic_change}
    if "dt_s" in data and all("orientation_xyzw" in body for body in data["bodies"]):
        orientation = np.asarray([body["orientation_xyzw"] for body in data["bodies"]])
        if not np.allclose(np.linalg.norm(orientation, axis=1), 1, atol=1e-5):
            raise ValueError("Frozen body orientations are not unit quaternions")
        local_change = warm_velocity[:, :3] - original_velocity[:, :3]
        world_change = rotate(orientation, local_change)
        gravity = np.asarray(data["gravity"])
        gravitational_change = -float(np.sum(a[:, 0] * data["dt_s"] *
                                             np.sum(world_change * gravity, axis=1)))
        energy_comparison.update({
            "ji_solver_kinetic_energy_J": float(np.sum(0.5 * a * original_velocity**2)),
            "gravitational_energy_change_J": gravitational_change,
            "one_step_mechanical_energy_change_relative_to_ji_J": kinetic_change + gravitational_change,
        })
        if all("position_m" in body for body in data["bodies"]):
            prior_position = np.asarray([body["position_m"] for body in data["bodies"]])
            ji_position = prior_position + data["dt_s"] * rotate(
                orientation, original_velocity[:, :3])
            energy_comparison["ji_solver_gravitational_energy_J"] = -float(np.sum(
                a[:, 0] * np.sum(ji_position * gravity, axis=1)))
    report = {
        "source": (f"{data['scene']} frozen step {data['step']}; full frictional contact"
                   if "scene" in data else
                   "reauthored vertical stack, frozen before step 8; full frictional contact"),
        "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
        "numpy": np.__version__, "scipy": scipy.__version__,
        "bodies": len(data["bodies"]), "contacts": len(data["contacts"]),
        "ji_candidate": audit(data, original_velocity, original_impulses),
        "reference_ji_start": {**statistics, **warm_audit},
        "reference_zero_start": {**cold_statistics, **cold_audit} if cold_error is None else None,
        "zero_start_failure": cold_error,
        "mechanics_gate": passed,
        "start_sensitivity": {
            "max_linear_velocity_m_s": float(np.abs(warm_velocity[:, :3] -
                                                    cold_velocity[:, :3]).max()),
            "max_angular_velocity_rad_s": float(np.abs(warm_velocity[:, 3:] -
                                                         cold_velocity[:, 3:]).max()),
            "max_contact_impulse_difference": float(np.abs(warm_impulses -
                                                          cold_impulses).max()),
            "max_body_wrench_impulse_difference": float(np.abs(np.asarray(
                [body["a"] for body in data["bodies"]]) *
                (warm_velocity - cold_velocity)).max()),
        } if cold_error is None else None,
        "ji_vs_reference": {
            "max_linear_velocity_m_s": float(np.abs(warm_velocity[:, :3] -
                                                    original_velocity[:, :3]).max()),
            "max_angular_velocity_rad_s": float(np.abs(warm_velocity[:, 3:] -
                                                         original_velocity[:, 3:]).max()),
            **energy_comparison,
        },
    }
    print(json.dumps(report, indent=2))
    if not all(passed.values()):
        raise SystemExit("Frozen coupled reference did not meet every mechanics gate")


if __name__ == "__main__":
    main()
