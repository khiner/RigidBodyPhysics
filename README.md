# RigidBodyPhysics
Rigid body physics and collision detection running on Apple Silicon

## Probes

Standalone probes, run as `./run <probe> [args...]` with Homebrew clang and metal-cpp.
Everything builds at `-O1` because Homebrew clang 22.1.8 miscompiles an idiomatic vector-growth pattern at `-O2` and above.

- [`probes/DispatchChain.cpp`](probes/DispatchChain.cpp) — how a chain of dependent compute dispatches must be expressed in Metal 4, and what it costs.
  `./run DispatchChain [iterations] [max_chain] [groups] [work]` sweeps chain length across the barrier expressions, then dispatch size, then submission rate, and ends with the ordering check.
- [`probes/VectorMiscompile.cpp`](probes/VectorMiscompile.cpp) — re-checks the miscompile.
  Re-run both lines when the toolchain updates:

```
./run VectorMiscompile          # passes
OPT=-O2 ./run VectorMiscompile  # reports the doubling, and throws outright at larger counts
```
