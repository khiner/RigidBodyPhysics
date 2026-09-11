# RigidBodyPhysics

Rigid body physics and collision detection for Apple Silicon, using an AVBD solver with native Metal 4 execution.

The library supports primitive, convex, triangle-mesh and compound colliders, joints with limits and drives, sleeping, contact reports and sensors.
Collision detection is discrete and runs at every physics substep.

Build on macOS 26 or later with CMake, Python 3, Homebrew LLVM and Apple’s Metal Toolchain:

```sh
xcodebuild -downloadComponent MetalToolchain
cmake -S . -B build/cmake
cmake --build build/cmake -j4
ctest --test-dir build/cmake --output-on-failure
```

Create a `rbp::mtl::Context`, then construct `rbp::World` and `rbp::Solver` with that context.
The context must outlive its worlds and solvers.
Add shapes and bodies to the world before advancing it.

`Solver::Step` advances one fixed step.
`Solver::Advance` batches equal substeps and optionally delivers each completed step's states and reports through an observer.
Both calls block until their GPU work and reporting complete.
Finish an advance before editing the world.
Use `World::ResetDynamics` after restoring poses and velocities to restart a simulation while retaining bodies, geometry, and GPU buffers.
Copy observer data that must remain valid after the callback returns.

For embedding, add this directory with CMake's `add_subdirectory` and link the `rbp` target.
Call `rbp_copy_shaders(app_target)` for each executable that uses RBP.
This packages the compiled Metal library and GPU archive beside the executable or in its app bundle’s Resources directory.
Solver shaders compile during the build for the local Mac’s GPU and load directly from the archive at runtime.
Set `RBP_SAFE_MATH=ON` for builds that disable fast shader math during numerical diagnosis.
Set `RBP_TOOLS=OFF` to omit standalone tools and tests.
Set `RBP_METAL_CPP_IMPL=OFF` if the host already supplies metal-cpp's implementation translation unit.

## Jolt comparison

RBP provides the following [Jolt rigid-body functionality](https://jrouwe.github.io/JoltPhysics/) through its own API.

| Jolt API area | RBP coverage |
| --- | --- |
| `PhysicsSystem::Update` | `Solver::Step` and `Advance` run collision detection, solving and reporting for each substep. |
| `BodyInterface` | `World` creates and removes bodies, replaces shapes, exposes poses and velocities, and supports sleeping and waking. |
| Body motion and mass | Static, dynamic and kinematic motion, authored mass and inertia, gravity scale and damping. |
| Collision shapes | Boxes, bounded planes, spheres, capsules, cylinders, convex hulls, triangle meshes and compounds. |
| Constraints | `JointDesc` supports fixed, point, hinge and slider configurations, with linear and angular limits, drives, springs and damping. |
| Collision filtering and materials | Body and collider masks, joint collision exclusions, friction, restitution and material combine policies. |
| `ContactListener` | `TakeContactChanges` reports added, persisted and removed contacts, with solved forces, impulses and contact geometry. |
| Sensors | `Overlaps` and `TakeSensorChanges` expose current overlaps and enter/exit events. |

### Performance

Recorded Apple M5 Max timings average the median substep time from two runs, with 120 warmup updates and 120 measured updates.
Each update advances ten 1/60-second substeps with sleeping and contact reporting disabled.
RBP timings include CPU encoding, GPU submission, execution and completion waiting.

| Workload | RBP, ms/substep | Jolt, ms/substep |
| --- | ---: | ---: |
| One box on a plane | 0.077 | 0.030 |
| 25 boxes on a triangle-mesh floor | 0.129 | 0.054 |
| 6,000 boxes in separate six-box columns | 0.583 | 1.134 |
| 192,000 boxes in separate six-box columns | 14.973 | 33.754 |

RBP uses ten AVBD iterations and host optimization `-O1`.
Jolt revision `187da15` uses Release `-O3`, ten workers, ten velocity iterations, two position iterations and enabled contact caching.
Both solvers use the same scene inputs and timestep.
These timings do not imply equal physical accuracy.
See [Validation.md](Validation.md#benchmarks) for benchmark commands and physical validation criteria.

See [Architecture.md](Architecture.md) for data and execution contracts.
See [Validation.md](Validation.md) for tests and benchmarks and [NOTICE.md](NOTICE.md) for attribution.
