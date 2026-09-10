# Validation

Build the library, tools and tests, then run the complete suite:

```sh
cmake -S . -B build/cmake
cmake --build build/cmake -j4
ctest --test-dir build/cmake --output-on-failure
```

The unit suite is organized around contracts, with shared tables for shape, mass, timestep and scheduling variants.
Keep independent analytic, finite-difference, brute-force and conservation checks.
Compare full states and reports for execution equivalence, alongside independent physical checks.

| Contract | Oracle |
| --- | --- |
| Numerical kernels and host/device layout | Independent solutions, finite differences and compiler layouts |
| Cooking and collision geometry | Solid integrals, geometric support and brute-force pair membership |
| Dynamics and joints | Discrete motion, momentum, energy, friction and force bounds |
| World lifecycle | Allocation rollback, stable identity, ownership and topology |
| Contact and sensor reporting | Momentum transfer, patch geometry and complete overlap lifetimes |
| Batched execution | Every intermediate state, report and refusal against serial execution |

Add a row to an existing matrix when a defect falls under the same contract.
Add a separate test when it needs a different independent oracle or lifecycle sequence.
Keep performance measurements in benchmarks and exploratory scenes in `RbpScenes`.
Batch runs that inspect only the final state.
Check individual substeps when testing temporal reports or changing inputs.

Run a focused contract group during iteration, then the complete suite before accepting the change:

```sh
build/cmake/tests/RbpTests --test-case='advance:*'
build/cmake/tests/RbpTests --test-case='joints:*'
```

## Benchmarks

`RbpBench` runs all default scenes when no names are supplied.
The two largest scenes require explicit names:

```sh
SUBSTEPS=10 WARMUP=120 STEPS=120 build/cmake/RbpBench
SUBSTEPS=10 WARMUP=120 STEPS=120 build/cmake/RbpBench lattice96000 lattice192000
```

Timings cover blocking `Solver::Advance` calls, divided by the number of substeps.
They include encoding, submission, GPU execution, completion waiting and result processing.
Contact and sensor reporting are disabled in these benchmarks.
Sleeping is disabled except in the explicitly sleeping scene.
State checks run outside the timing interval.
Nonfinite states or refused contacts fail the run.
The column scenes report floor penetration, horizontal drift and adjacent-body vertical projection overlap.
Projection overlap measures separation along the vertical axis alone.

Use the same power source, scene, settings and output requirements when comparing builds.
Run benchmarks serially on a quiet machine and compare repeated medians in both execution orders.
Use blocking advance timings to measure end-to-end performance.

An optional Jolt comparison tool lives in `tools/jolt` and builds against a separately supplied Jolt checkout:

```sh
cmake -S tools/jolt -B build/jolt -DCMAKE_BUILD_TYPE=Release -DJOLT_SOURCE_DIR=/path/to/JoltPhysics
cmake --build build/jolt -j4
```

Match collision-step size, substep count, sleeping and reporting settings before comparing solvers.
Compare physical accuracy independently of iteration counts.

## Replays and diagnostic scenes

`RbpReplay` checks every stored intermediate state against a supplied RBP capture:

```sh
build/cmake/RbpReplay --batch 10 fixture.rbp
build/cmake/RbpReplay --steps 120 fixture.rbp
```

Replay differences indicate trajectory changes.
Use the numerical and physical tests to assess intentional trajectory changes.

`RbpScenes` reports physical measurements for individual scenarios, including stacks, sliding, impacts, joints and moving meshes.
Run it without arguments to list the scene names.

## Compiler check

The RBP target uses `-O1` to avoid a Homebrew Clang vector-growth miscompile.
The standalone reproducer allows this workaround to be checked when updating the compiler:

```sh
./run VectorMiscompile
OPT=-O2 ./run VectorMiscompile
```

Both optimization levels should retain the reserved vector capacity on an unaffected compiler.
