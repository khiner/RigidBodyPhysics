# Ji 2025 Metal SubADMM

This is an independent Apple Silicon implementation of Ji et al., *GPU-Accelerated Subsystem-Based ADMM for Large-Scale Interactive Simulation*. One SubADMM solver owns the scene, collision, contact-cache and integration pipeline. The repository's AVBD solver is separate. A CPU equation oracle and optional frozen-contact references provide independent checks.

The nine procedural Table I scale points are sliding cubes (100, 500, 1,000, 1,500 and 2,000 bodies, 5 ms step), vertical stacks (360 bodies, 5 ms), oblique stacks (140 bodies, 5 ms), card houses (396 bodies, 1 ms), and articulated hand (281 bodies, 33 joints, 4 ms). The baseline uses the paper's `M=16`, `c=10^5` and `alpha=4`, with each scene's published iteration and penalty-update settings. The authors' CUDA implementation, exact assets and VIST hand animation were unavailable in the recorded source search; exact author-output or GPU-time reproduction is not claimed.

Fixed iteration results can violate contact mechanics or produce unstable motion. A driven hand pulse remains a known regression target. No all-scene real-time or modal-contact audio qualification is claimed. [PhysicsModel.md](PhysicsModel.md) identifies paper-versus-local modeling choices; [Validation.md](Validation.md) gives reproducible audits and limitations; [Literature.md](Literature.md) records provenance.

## Build and run

On Apple Silicon with the [repository prerequisites](../../README.md), from the repository root:

```sh
git submodule update --init --recursive
cmake -S . -B build/cmake
cmake --build build/cmake -j4
ctest --test-dir build/cmake --output-on-failure
build/cmake/src/ji/JiBenchmark --case sliding --bodies 100 --warmup 0 --steps 1
python3 tools/ji/PaperSuite.py --case sliding --bodies 100 --steps 10 --audit-only
```

`JiBenchmark --help` lists timing, factory force controls, source-equation captures, final states and pose traces. `Step` reads back and independently audits the discrete equations; `Advance` runs synchronous physics without CPU contact decode and audit. Complete-step wall latency, timestamped GPU solve time and simulated timestep are distinct quantities. Reports and captures belong under ignored `build/`.
