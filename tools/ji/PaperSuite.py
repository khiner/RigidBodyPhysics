#!/usr/bin/env python3
"""Audit reauthored Ji scenes and measure synchronous SubADMM step latency."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import shlex
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
CASES = [("sliding", count, 100) for count in (100, 500, 1000, 1500, 2000)] + [
    ("vertical", None, 100), ("oblique", None, 100),
    ("cards", None, 200), ("articulated", None, 600)]
# Ji 2025 Table I; authors' RTX 4080 solver means are context, not our latency target.
PAPER_SOLVE_MS = (3.82, 4.15, 4.85, 5.26, 5.51, 5.81, 4.52, 9.12, 5.00)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_files():
    return sorted(path for directory in ("src", "cmake", "tools/ji")
                  for path in (ROOT / directory).rglob("*")
                  if path.is_file() and "__pycache__" not in path.parts) + [ROOT / "CMakeLists.txt"]


def source_pin(files):
    value = hashlib.sha256()
    for path in files:
        value.update(str(path.relative_to(ROOT)).encode())
        value.update(path.read_bytes())
    return value.hexdigest()


def freshness(benchmark, files):
    build = next((p for p in benchmark.parents if (p / "CMakeCache.txt").is_file()), None)
    depfiles = [] if build is None else [p for p in build.rglob("*.o.d")
        if any(part in ("JiBenchmark.dir", "JiSceneRuntime.dir", "rbp.dir") for part in p.parts)]
    host = set()
    for path in depfiles:
        host.update(Path(name).resolve() for name in shlex.split(
            path.read_text().replace("\\\n", " ").split(":", 1)[1]) if Path(name).is_absolute())
    host = [p for p in host if p.is_relative_to(ROOT) and p.is_file()] or [
        p for p in files if p.suffix in (".cpp", ".h")]
    shader = [p for p in files if p.is_relative_to(ROOT / "src/gpu") or
              p.name in ("Shaders.py", "FullStep.py", "Shaders.cmake")]
    artifacts = [benchmark, benchmark.parent / "rbp.metallib"]
    archive = benchmark.parent / "rbp.binary.metallib"
    if archive.exists():
        artifacts.append(archive)
    for artifact in artifacts:
        if not artifact.is_file():
            raise ValueError(f"Build missing artifact: {artifact}")
        producer = artifact
        if build and artifact != benchmark:
            generated = build / "gen" / artifact.name
            if generated.is_file():
                if digest(generated) != digest(artifact):
                    raise ValueError(f"Repackage {artifact.name}: bytes differ from CMake product")
                producer = generated
        newer = [p for p in (host if artifact == benchmark else shader)
                 if p.stat().st_mtime > producer.stat().st_mtime]
        if newer:
            raise ValueError(f"Rebuild {artifact.name}: predates {newer[0].relative_to(ROOT)}")
    return artifacts, {"compiler_dependency_files": len(depfiles), "host_inputs": len(host),
                       "shader_inputs": len(shader)}


def decode(text):
    return json.loads(text, parse_constant=lambda value: (_ for _ in ()).throw(
        ValueError(f"Nonfinite JSON value: {value}")))


def captures(audit):
    for capture in audit["equations_captures"]:
        path = Path(capture["path"])
        data = decode(path.read_text())
        physical = {key: data[key] for key in ("dt_s", "gravity")}
        physical["bodies"] = [{key: body[key] for key in
            ("world_slot", "a", "b", "orientation_xyzw", "position_m")} for body in data["bodies"]]
        physical["constraints"] = [{key: row[key] for key in
            ("kind", "rows", "friction", "stiffness", "damping", "error", "velocity_bias", "sides", "contact_key")
            if key in row} for row in data["constraints"]]
        capture.update(file_sha256=digest(path), physical_input_sha256=hashlib.sha256(
            json.dumps(physical, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest())
    return audit["equations_captures"]


def validate(run, steps):
    if run["measured_steps"] != steps or not run["trajectory_complete"]:
        raise ValueError("Trajectory incomplete")
    for key in ("contacts", "warm_started", "iterations", "wall_ms", "primal_residual",
                "dual_residual", "dynamics_residual", "contact_law_residual"):
        if len(run[key]) != steps or not all(math.isfinite(v) for v in run[key]):
            raise ValueError(f"Invalid per-step series: {key}")


def summary(audit, timed, profiled):
    result = {key: audit[key] for key in ("scenario", "dynamic_bodies", "joints", "dt_ms",
              "configured_iterations", "warmup_steps", "measured_steps", "peak_device_allocated_bytes")}
    result["maximum"] = {key: max(values) for key, values in audit.items()
                         if isinstance(values, list) and values and isinstance(values[0], (int, float))
                         and key not in ("dynamic_body_slots",)}
    result["energy"] = {key: audit[key] for key in ("initial_mechanical_energy_j",
        "net_mechanical_energy_change_j", "max_step_mechanical_energy_increase_j")}
    result["events"] = audit.get("events")
    result["equations_captures"] = captures(audit)
    result["dynamics_relative_to_local_scale"] = max(
        (value / max(scale, 1e-30) for value, scale in
         zip(audit["audit_dynamics"], audit["audit_dynamics_scale"])), default=0)
    if timed:
        result["complete_step_ms"] = {key: statistics.median(run[key] for run in timed)
            for key in ("wall_ms_p50", "wall_ms_p95")}
        result["complete_step_ms"]["wall_ms_worst"] = max(run["wall_ms_worst"] for run in timed)
        result["complete_step_p95_over_dt"] = result["complete_step_ms"]["wall_ms_p95"] / audit["dt_ms"]
    if profiled:
        result["gpu_solver_ms_median"] = statistics.median(
            statistics.median(run["gpu_solve_ms"]) for run in profiled)
    result["observed_workload_differences"] = [{key: sum(a != b for a, b in zip(audit[key], run[key]))
        for key in ("contacts", "warm_started", "iterations")} for run in timed + profiled]
    result["timing_workload_matches_audit"] = all(
        not any(row.values()) for row in result["observed_workload_differences"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=ROOT / "build/cmake/src/ji/JiBenchmark")
    parser.add_argument("--case", choices=sorted({case[0] for case in CASES}))
    parser.add_argument("--bodies", type=int, choices=(100, 500, 1000, 1500, 2000))
    parser.add_argument("--steps", type=int, help="override each measured window")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--audit-only", action="store_true", help="omit timing repetitions")
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--case-seconds", type=float, default=180)
    parser.add_argument("--seconds", type=float, default=1800)
    parser.add_argument("--capture-dir", type=Path)
    parser.add_argument("--capture-step", type=int, action="append", default=[])
    parser.add_argument("--hand-pulse", action="store_true",
                        help="known regression: articulated, factory start, 60 N extra wrist force for 5 steps; default 120 steps/100 iterations")
    parser.add_argument("--raw", action="store_true", help="include complete audited arrays")
    args = parser.parse_args()
    if args.hand_pulse and (args.case or args.bodies):
        parser.error("--hand-pulse selects its own articulated scenario")
    if args.bodies and args.case != "sliding":
        parser.error("--bodies requires --case sliding")
    if not 0 <= args.warmup <= 1000 or args.timing_repeats < 1 or any(
            not math.isfinite(v) or v <= 0 for v in (args.case_seconds, args.seconds)):
        parser.error("Invalid warmup, repeats or wall cap")
    if args.steps is not None and not 1 <= args.steps <= 10000 or args.iterations is not None and args.iterations < 1:
        parser.error("Steps and iterations must be positive; steps <=10000")
    if bool(args.capture_dir) != bool(args.capture_step):
        parser.error("Supply --capture-dir and --capture-step together")
    selected = [("articulated", None, 120)] if args.hand_pulse else [case for case in CASES
        if (args.case is None or case[0] == args.case) and (args.bodies is None or case[1] == args.bodies)]
    warmup = 0 if args.hand_pulse else args.warmup
    if any(step < warmup or step >= warmup + (args.steps or declared)
           for _, _, declared in selected for step in args.capture_step):
        parser.error("Capture steps must be within every selected measured window")
    benchmark = args.benchmark.resolve()
    if args.capture_dir:
        args.capture_dir = args.capture_dir.resolve()
        args.capture_dir.mkdir(parents=True, exist_ok=True)
    files = source_files()
    try:
        artifacts, build_freshness = freshness(benchmark, files)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    pin = source_pin(files)
    runtime = {str(p): digest(p) for p in artifacts}
    deadline = time.monotonic() + args.seconds
    rows = []
    for name, bodies, declared in selected:
        steps = args.steps or declared
        print(f"Ji suite: {name}/{bodies or ''}, {steps} steps", file=sys.stderr, flush=True)
        def invoke(advance=False, profile=False):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Aggregate wall cap reached")
            command = [str(benchmark), "--case", name, "--warmup", str(warmup), "--steps", str(steps)]
            if bodies:
                command += ["--bodies", str(bodies)]
            if args.iterations or args.hand_pulse:
                command += ["--iterations", str(args.iterations or 100)]
            if args.hand_pulse:
                command += ["--wrist-pulse-N", "60", "--wrist-pulse-steps", "5"]
            if advance:
                command += ["--advance-only"]
            else:
                command += ["--energy", "--contact-events"]
                if args.capture_dir:
                    command += ["--frozen-equations-output", str(args.capture_dir / f"{name}{bodies or ''}.json")]
                    for step in args.capture_step:
                        command += ["--equations-step", str(step)]
            if profile:
                command += ["--gpu-timing"]
            run = decode(subprocess.run(command, check=True, capture_output=True, text=True,
                timeout=min(args.case_seconds, remaining)).stdout)
            validate(run, steps)
            return run
        try:
            audit = invoke()
            timed = [] if args.audit_only else [invoke(True) for _ in range(args.timing_repeats)]
            profiled = [] if args.audit_only else [invoke(True, True) for _ in range(args.timing_repeats)]
            row = summary(audit, timed, profiled)
            row["completed"] = True
            row["paper_solver_mean_ms"] = None if args.hand_pulse else PAPER_SOLVE_MS[CASES.index((name, bodies, declared))]
            if args.raw:
                row["audit"] = audit
            rows.append(row)
        except (OSError, ValueError, KeyError, subprocess.SubprocessError, TimeoutError) as error:
            diagnostic = getattr(error, "stderr", None) or str(error)
            if isinstance(diagnostic, bytes):
                diagnostic = diagnostic.decode(errors="replace")
            rows.append({"scenario": name, "bodies": bodies, "completed": False, "error": diagnostic[:4096]})
    unchanged = pin == source_pin(source_files()) and all(digest(p) == runtime[str(p)] for p in artifacts)
    complete = unchanged and all(row["completed"] for row in rows)
    print(json.dumps({"schema": 1, "completed": complete, "solver": "subadmm",
        "completion_scope": "All selected finite trajectories completed with unchanged source/runtime pins; no universal physical or real-time certification",
        "audit_scope": "Residuals retain separate units; inspect momentum scale, corrected motion, penetration, work, energy and event coverage. A contact-law residual above 2e-6 N alone does not fail a real-time scene.",
        "timing_scope": "Uninstrumented synchronous Advance wall latency; separate timestamped Advance solver measurements. Short windows do not certify sustained real-time performance.",
        "hand_pulse_regression": args.hand_pulse, "argv": sys.orig_argv,
        "captured_utc": datetime.now(timezone.utc).isoformat(), "macos_version": platform.mac_ver()[0],
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "dirty_worktree": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
        "source_sha256": pin, "runtime_sha256": runtime, "pins_unchanged": unchanged,
        "build_freshness": build_freshness, "cases": rows}, indent=2, allow_nan=False))
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
