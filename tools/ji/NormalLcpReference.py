#!/usr/bin/env python3
"""Solve the frozen vertical stack's frictionless normal-contact subproblem.

Requires NumPy and SciPy. The bundled fixture has zero contact error and no
joints; each contact group is a small convex, bound-constrained least-squares
problem. This is an independent mechanics control, not a frictional solver.
"""

import gzip
import json
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear


FIXTURES = Path(__file__).resolve().parents[2] / "tests/ji/fixtures"


def read(name):
    with gzip.open(FIXTURES / name, "rt") as source:
        return json.load(source)


def contact_groups(contacts, body_count):
    parent = list(range(len(contacts)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first_contact = {}
    for i, contact in enumerate(contacts):
        for side in contact["sides"]:
            body = side["body"]
            if body >= body_count:
                continue
            if body in first_contact:
                parent[root(i)] = root(first_contact[body])
            else:
                first_contact[body] = i

    groups = {}
    for i in range(len(contacts)):
        groups.setdefault(root(i), []).append(i)
    return list(groups.values())


def audit(data, velocity, multiplier):
    contacts = data["contacts"]
    a = np.asarray([body["a"] for body in data["bodies"]])
    b = np.asarray([body["b"] for body in data["bodies"]])
    force = np.zeros_like(velocity)
    normal = np.empty(len(contacts))
    for i, contact in enumerate(contacts):
        normal[i] = contact["error"]
        for side in contact["sides"]:
            body = side["body"]
            if body >= len(a):
                continue
            row = np.asarray(side["j"])
            normal[i] += row @ velocity[body]
            force[body] += multiplier[i] * row
    work = multiplier * normal
    return {
        "max_negative_normal_velocity_scaled": float(np.maximum(-normal, 0).max()),
        "max_normal_complementarity_J": float(np.abs(work).max()),
        "positive_normal_work_J": float(np.maximum(work, 0).sum()),
        "max_momentum_residual": float(np.abs(a * velocity - b - force).max()),
    }


def solve(data):
    bodies = data["bodies"]
    contacts = data["contacts"]
    if any(contact["error"] != 0 for contact in contacts):
        raise ValueError("The least-squares reference requires zero normal error")
    a = np.asarray([body["a"] for body in bodies])
    b = np.asarray([body["b"] for body in bodies])
    if not np.all(a > 0):
        raise ValueError("Every moving body must have positive diagonal inertia")
    velocity = b / a
    multiplier = np.zeros(len(contacts))
    groups = contact_groups(contacts, len(bodies))

    # A*v=b+J^T*lambda, 0<=lambda perpendicular to J*v>=0.
    # With B=A^-1/2 J^T and y=A^-1/2 b, this is the KKT system of
    # min_{lambda>=0} ||B*lambda+y||^2 / 2, separately per contact group.
    for ids in groups:
        body_ids = sorted({side["body"] for i in ids for side in contacts[i]["sides"]
                           if side["body"] < len(bodies)})
        local = {body: j for j, body in enumerate(body_ids)}
        matrix = np.zeros((6 * len(body_ids), len(ids)))
        for column, i in enumerate(ids):
            for side in contacts[i]["sides"]:
                body = side["body"]
                if body < len(bodies):
                    start = 6 * local[body]
                    matrix[start:start + 6, column] += np.asarray(side["j"]) / np.sqrt(a[body])
        rhs = (b[body_ids] / np.sqrt(a[body_ids])).reshape(-1)
        result = lsq_linear(matrix, -rhs, bounds=(0, np.inf), method="bvls",
                            tol=1e-13, max_iter=1000)
        if not result.success:
            raise RuntimeError(f"Normal-contact reference failed: {result.message}")
        multiplier[ids] = result.x
        delta = (matrix @ result.x).reshape(len(body_ids), 6) / np.sqrt(a[body_ids])
        velocity[body_ids] += delta

    return groups, velocity, multiplier


def main():
    data = read("vertical_step7_normal.json.gz")
    groups, velocity, multiplier = solve(data)
    report = {
        "source": "vertical step 7 frozen state; frictionless normal-only control",
        "bodies": len(data["bodies"]),
        "contacts": len(data["contacts"]),
        "components": len(groups),
        "largest_component_contacts": max(map(len, groups)),
        "reference": audit(data, velocity, multiplier),
    }
    if (report["reference"]["max_negative_normal_velocity_scaled"] > 1e-8 or
            report["reference"]["max_normal_complementarity_J"] > 1e-8 or
            report["reference"]["max_momentum_residual"] > 1e-8):
        raise RuntimeError("Independent normal-contact solution failed its mechanics audit")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
