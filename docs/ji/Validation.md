# Ji solver validation

## Tests

```sh
ctest --test-dir build/cmake --output-on-failure
```

[JiTests](../../tests/ji/SolverTest.cpp) covers analytical dynamics, contact/friction, joints, cache lifecycle, stepping and numerical safeguards. `RbpTests` covers shared world and collision behavior.

## Scene audits and timing

```sh
python3 tools/ji/PaperSuite.py --audit-only > build/ji-audit.json
python3 tools/ji/PaperSuite.py --case sliding --bodies 100 --steps 120 > build/ji-timing.json
```

The suite uses the Python standard library and defaults to `build/cmake/src/ji/JiBenchmark`; `--benchmark` selects another build. It checks build freshness and pins sources, executable and shaders. The default audit exercises all nine reauthored paper scales. Timing uses separate uninstrumented complete-step and timestamped GPU-solver runs.

Completion means the trajectory remained finite. Assess physical behavior using motion, penetration, momentum balance and work; residual units are described in [PhysicsModel.md](PhysicsModel.md). Driven scenes include external work, and short timing windows do not establish sustained realtime performance.

## Known driven-hand failure

```sh
python3 tools/ji/PaperSuite.py --hand-pulse --audit-only --raw > build/ji-hand-pulse.json
```

This applies an additional 60 N wrist force for five steps, then releases it, with the factory drive still active. The 120-step run at 100 iterations exposes unstable motion and excessive penetration. It remains an open solver limitation.

## Optional frozen-contact references

```sh
python3 tools/ji/PaperSuite.py --case vertical --warmup 0 --steps 8 --audit-only --capture-dir build/ji-captures --capture-step 7 > build/ji-frozen-audit.json
python3 -m venv build/ji-venv
build/ji-venv/bin/python -m pip install -r tools/ji/requirements.txt
build/ji-venv/bin/python tools/ji/NormalLcpReference.py
build/ji-venv/bin/python tools/ji/FrictionalReference.py build/ji-captures/vertical.json
```

The normal reference is a frictionless control. The frictional reference independently solves contact-only captures from candidate and zero starts, then audits the roots. It can fail its convergence gates; a frozen solution does not validate a trajectory, and contact reactions need not be unique.
