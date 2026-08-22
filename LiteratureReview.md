# Literature Review

**RigidBodyPhysics** — a Metal GPU realtime rigid body physics and collision detection library for Apple Silicon, matching the Jolt Physics API surface used by [MeshEditor](../MeshEditor), working toward integrating modal vibrations directly into the contact solver per [Zheng & James 2011, "Toward High-Quality Modal Contact Sound"](https://www.cs.cornell.edu/projects/Sound/mc/ModalContactSound2011.pdf).

Compiled 2026-08-22. All primary URLs were fetched or HTTP-verified at compile time unless noted otherwise. Sections: [Synthesis](#synthesis-design-implications) · [1. Jolt (the compatibility target)](#1-jolt-physics--the-compatibility-target) · [2. GPU collision detection](#2-gpu-collision-detection-broad-phase-and-narrow-phase) · [3. Constraint solvers and GPU parallelization](#3-constraint-solvers-and-gpu-parallelization) · [4. Existing GPU physics engines](#4-survey-of-existing-gpu-physics-engine-implementations) · [5. Metal 4 and Apple Silicon](#5-the-metal-api-metal-4-and-apple-silicon-gpu-architecture) · [6. Modal contact sound](#6-modal-contact-sound-and-physically-based-rigid-body-sound-synthesis) · [7. Open questions and early measurements](#7-open-questions-and-early-measurements)

---

## Synthesis: design implications

The detailed surveys below support a small set of decisive conclusions for this project.

### The niche is genuinely open

PhysX 5's GPU rigid pipeline is CUDA-only. Jolt has publicly ruled out GPU rigid bodies for its core pipeline ([discussion #501](https://github.com/jrouwe/JoltPhysics/discussions/501)) — its v5.6 compute abstraction (DX12/Vulkan/Metal) exists but serves only strand/hair physics. Havok, Unity Physics, and Unreal Chaos are CPU-only for rigid bodies. The Warp/Newton/MJWarp robotics stack is CUDA-bound. The only Apple-GPU-native rigid engines are months-old research codebases (avbd-metal, MetalAVBD), the robotics-batch-oriented Genesis (which does prove a full rigid pipeline runs on Metal via its Quadrants compiler), and Dimforge Nexus (soft-TGS with graph coloring, entirely on GPU via wgpu — the nearest competitor in spirit, and brand new). A Metal-native, single-big-world, Jolt-semantics GPU engine does not exist.

Equally open on the audio side: everything since Zheng & James 2011 accelerated the *precompute* (KleinPAT, NeuralSound), the *rendering* (WaveBlender), or the *synthesis* (Bonneel), leaving the coupled contact-dynamics pass — which ran 10³–10⁴× slower than realtime on 2009 CPUs — untouched. No published system has made vibration-aware contact solving realtime. That pass's costs (many small dense modal blocks, independent per-group solves, BVH refits, banked IIR synthesis) are all GPU-shaped.

### Solver: substepped soft-constraint impulses, colored for the GPU

The strongest evidence line (Catto's Solver2D shootout, Small Steps, PhysX TGS in production, Nexus shipping soft-TGS on GPU) says the baseline should be a **substepped soft-constraint sequential-impulse solver (TGS Soft) in maximal coordinates**:

- It reproduces Jolt/Box2D behavior — Jolt is PGS + position solver with warm starting, and Rouwe describes Jolt as "Box2D in 3D".
- Substepping is the great equalizer: one Gauss-Seidel iteration per substep beats many iterations per big step, so per-iteration convergence deficits of parallel-friendly schemes matter less as substep rate rises. This also moves the solver rate *toward* the rates modal coupling needs.
- GPU parallelization: constraint-graph coloring (Vivace-style dynamic coloring, or supernodal block coloring) for Gauss-Seidel-quality sweeps, with mass-splitting Jacobi (Tonge 2012) reserved for high-degree hub bodies (the ground plane touching everything). Islands are kept for sleeping and per-island solver selection, not as the primary parallelism axis (Box2D v3's conclusion).
- Soft constraints (Catto 2011) are the natural modal-coupling hook: frequency/damping-parameterized constraint rows are literally what a modal oscillator in contact is.

**AVBD** (Augmented Vertex Block Descent, SIGGRAPH 2025) is the tracked alternative — position-level, unconditionally stable, natively parallel, with three independent GPU implementations already public (avbd-metal, MetalAVBD, webphysics). Its per-body block solve generalizes cleanly to (6+m)-DOF modal-augmented bodies. Reasons it is not the baseline: behavioral parity with Jolt is the phase-1 goal and TGS matches Jolt semantics much more closely, and AVBD's friction fidelity and 3D stack behavior are younger. Revisit at the modal-coupling phase — the three candidate hosts for modal DOFs, in increasing ambition, are (a) soft-constraint rows at modal frequencies inside TGS, (b) (6+m)-DOF body blocks in a VBD/AVBD-style local solver, (c) staggered/ADMM splitting with modal subsystems as local solves (the Zheng-James/Kamino lineage).

A critical constraint from the audio literature that ordinary engines never face: **impulse quality is the product**. Solver-order noise, contact-set cycling, and non-unique multipliers are inaudible in motion and catastrophic in audio. Warm starting, persistent contact identity, contact generation on intersection (not approach velocity), and temporally coherent impulse distribution must be designed in from the start, not retrofitted.

### Collision pipeline: the convergent skeleton

Every serious GPU engine lands on the same shape, and this project should too:

```
radix sort + prefix scan  (build these first — Metal has no CUB)
        ↓
broadphase: one-axis parallel SAP over AABBs (temporally coherent re-sort),
            or hybrid grid-for-smalls + SAP-for-larges
        ↓
pair compaction → indirect dispatch
        ↓
narrowphase: per-shape-pair-type specialized kernels
  - signed-volume GJK (Montanari 2017) + bounded EPA, support mappings
  - SAT for boxes/small hulls (uniform control flow)
  - hull vertex cap sized to the 32-wide SIMD group (PhysX caps at 64)
  - quantized 4-wide BVH midphase for trimeshes, baked active edges
        ↓
one-shot manifolds (reference/incident face clipping, ≤4 points, feature IDs)
persistent contact cache keyed by (body pair, subshape pair) for warm starts
        ↓
speculative contacts as the default anti-tunneling mechanism (Jolt semantics),
LinearCast-style TOI as a sparse second pass over fast pairs, later
```

Full per-frame rebuild of broadphase structures is standard and cheap at rigid-body scales. BVH *refit* (not rebuild) is exactly right for modally deformed geometry later — BD-Tree (James & Pai 2004) updates bounds from modal amplitudes alone. Metal ray-tracing hardware (M3+) is worth a prototype as a trimesh midphase, but the opaque build and ray-centric API likely rule it out for the AABB-pair broadphase.

### Metal architecture

- **Target Metal 4 outright** (macOS 26, M1+; dev machine M5 Max). Encode each physics frame — the whole substep chain — as one `MTL4CommandBuffer` with one unified compute encoder, pass barriers (`barrierAfterEncoderStages:`) only on true dependencies. Commands are concurrent by default, so independent islands/stages overlap for free.
- **Bindless from day one**: buffers from a few `MTLHeap`s, one `MTLResidencySet`, one argument table pointing at a world argument buffer of GPU addresses.
- **Design around what Metal lacks**: no float atomics, `memory_order_relaxed` only, no device-wide barrier inside a dispatch, no device-side enqueue. This mandates gather-based accumulation (each body sums its own constraint list — also the deterministic option), colored dispatches, CPU-encoded fixed iteration counts, and indirect dispatch for variable-cardinality stages.
- **Exploit what Apple GPUs are unusually good at**: huge register file + best-in-class SIMD shuffles (SIMD-group-cooperative narrowphase, 8-lanes-per-body 6×6 block solves as in avbd-metal), Dynamic Caching's tolerance for divergent kernels (M3+), unified memory for zero-copy contact/query readback and near-free CPU fallbacks, and M5 Neural Accelerators (FP16 tensors via MPP) as a candidate for dense modal matrix work.
- **Determinism is achievable and worth it** (replays, regression tests, reproducible audio): Apple-Silicon-only scope means one platform. Requirements: fixed orderings everywhere (sort after every atomic-append stage), gather-based float accumulation, fixed kernel binaries. Fast-math is fine within a binary.
- **UMA changes the readback story, not the pipeline shape**: bulk state never returns to the CPU (rendering consumes solver output directly), the CPU sends batched commands and consumes a compact event stream. This is the PhysX Direct-GPU-API model minus its stale-mirror problem.

### API surface

The Jolt-shaped API survives GPU residency — the mismatch is execution semantics, not capability. Immediate reads, synchronous queries, and callbacks-during-solve become deferred/batched operations and device-written event buffers. The parity target is MeshEditor's *conventional* usage (inventoried in §1.9): `PhysicsSystem` + `BodyInterface` lifecycle, the full shape zoo it constructs, two constraint types (SixDOF with per-axis motors/limits/springs, Hinge with velocity motor), layer/group filtering, contact listeners with `ContactSettings` modification, and fixed-step `Update(dt, collisionSteps)`. Notably *not* needed for parity: collision queries (raycasts/shapecasts — MeshEditor uses its own BVH), CCD motion qualities, character/vehicle/soft-body.

MeshEditor's audio path currently scrapes per-point solver impulses through a fork-local Jolt patch (`GetAppliedContactImpulses`). That is explicitly a hack to be *replaced*, not ported: in the target formulation, contact excitation of modal DOFs is native solver state, and the audio-facing API will be redesigned around it (guided by Zheng & James, not by the current wrapper). MeshEditor's surface/sustained-contact path (SurfaceAudio, sustained-contact collection, slip/sweep velocities, and the tests around them) is default-gated-off and a dead end kept as notes — ignore it entirely. What the impact path does tell us is *which physical quantities* matter downstream: per-point normal impulses, contact direction, manifold area, approach speed.

### The modal end-state, briefly

The rate hierarchy to respect (from §6): graphics-rate rigid dynamics → contact-coupled stepping at `Δt ≈ 1/(6.5·f_h)` with adaptive modal bandwidth (f_h ≈ 5 kHz ⇒ ~32.5k steps/s) organized as asynchronous contact groups → exact two-term IIR recurrence synthesis to 20 kHz (unconditionally stable at any sample rate — only the *coupled* pass has a stiffness-limited timestep) → Hertz-timescale impulse shaping → FFAT-map radiation. The GPU-natural reformulation of Zheng-James's priority-queue asynchrony is a synchronous power-of-two timestep ladder per contact group. Eigenmode compression (Langlois 2014) or procedural MLS mode evaluation addresses the `U`-matrix memory problem — modal Jacobian columns are rows of `U` at contact points. WaveBlender (open-source CUDA FDTD, a tractable Metal port) is the offline golden renderer, with two-level validation: dynamics vs. high-accuracy dynamics, audio vs. FDTD audio.

---

## 1. Jolt Physics — the compatibility target

Jolt Physics ([jrouwe/JoltPhysics](https://github.com/jrouwe/JoltPhysics), MIT) is a multicore-friendly rigid/soft body physics and collision detection library in C++17, used in Horizon Forbidden West and Death Stranding 2, and the default 3D physics engine in Godot 4.6. It was architected around two goals presented at GDC 2022: concurrent query+modification of the physics world (lock-free broadphase) and lock-free simulation island building. Primary references: the [Architecture doc](https://github.com/jrouwe/JoltPhysics/blob/master/Docs/Architecture.md) ([Doxygen render](https://jrouwe.github.io/JoltPhysics/index.html)), [release notes](https://github.com/jrouwe/JoltPhysics/blob/master/Docs/ReleaseNotes.md), and the GDC 2022 talk ["Architecting Jolt Physics for 'Horizon Forbidden West'"](https://gdcvault.com/play/1027560/Architecting-Jolt-Physics-for-Horizon) ([annotated slides](https://jrouwe.nl/architectingjolt/ArchitectingJoltPhysics_Rouwe_Jorrit_Notes.pdf)).

### 1.1 Conventions an API-compatible engine must reproduce

- **Units**: SI — meters, kilograms, seconds, radians. Recommended dynamic object sizes 0.1–10 m, velocities 0–500 m/s. Float precision holds to ~5 km from origin (`JPH_DOUBLE_PRECISION` builds exist but MeshEditor builds single-precision).
- **Coordinate system**: right-handed, Y-up by convention. Default gravity `Vec3(0, -9.81f, 0)`. Column-major 4×4 matrices, quaternions for rotation.
- **Center of mass**: every `Shape` recenters around its COM at construction. Body transforms internally are COM transforms, with paired accessors (`GetPosition` vs `GetCenterOfMassPosition`). Most shape-space functions operate in COM space. One of the most compatibility-sensitive behaviors.
- **Mass properties**: computed from shape volume × `ConvexShapeSettings::mDensity` (default 1000 kg/m³). `mOverrideMassProperties` ∈ {CalculateMassAndInertia (default), CalculateInertia, MassAndInertiaProvided (required for dynamic mesh-like shapes)}.
- **Combine rules** (overridable): friction = `sqrt(f1*f2)`, restitution = `max(r1, r2)`. Restitution applies only above `mMinVelocityForRestitution` = 1 m/s.
- **Key solver defaults** ([PhysicsSettings.h](https://github.com/jrouwe/JoltPhysics/blob/master/Jolt/Physics/PhysicsSettings.h)): `mBaumgarte` = 0.2, `mSpeculativeContactDistance` = 0.02 m, `mPenetrationSlop` = 0.02 m, `mNumVelocitySteps` = 10, `mNumPositionSteps` = 2, `mMaxPenetrationDistance` = 0.2 m, `mTimeBeforeSleep` = 0.5 s, `mPointVelocitySleepThreshold` = 0.03 m/s, warm start / body-pair cache / manifold reduction / large-island splitter all on. `cDefaultConvexRadius` = 0.05 m.
- **Damping**: exponential, `dv/dt = -c·v` (defaults 0.05 linear and angular).
- **Triangle winding**: CCW, single-sided for simulation (MeshEditor forces `CollideWithBackFaces` — see §1.9).

### 1.2 Broadphase

QuadTree AABB trees — one per **BroadPhaseLayer** (`BroadPhaseQuadTree`). Each node holds SoA bounds for 4 children (atomic min/max floats, atomic child indices, `Changed` flag) so a query tests 4 child AABBs per SIMD op. The lock-free protocol (GDC notes, slides 19–30):

- **Moving a body**: the new AABB is merged in via atomic min/max, widening bounds up the parent chain. In-flight queries never miss hits because bounds only widen. The degraded tree is rebuilt each `Update` and atomically swapped; the old tree survives one extra frame for long-running background queries.
- **Batched add**: `AddBodiesPrepare` builds a sub-tree on a background thread; `AddBodiesFinalize` links it in O(1) at the root via compare-exchange. One-at-a-time adds degenerate the tree — batch adds are the intended path.
- **Layer mapping**: 16/32-bit `ObjectLayer` per body; three user interfaces configure filtering: `BroadPhaseLayerInterface` (ObjectLayer → BroadPhaseLayer, many-to-one), `ObjectVsBroadPhaseLayerFilter`, `ObjectLayerPairFilter`. Typical setup: NON_MOVING and MOVING broadphase layers so the static tree rebuilds rarely.
- `OptimizeBroadPhase()` forces a rebuild (call after bulk-loading a level).

### 1.3 Narrowphase and shapes

- **GJK/EPA**: all convex-convex collision uses GJK (`Jolt/Geometry/GJKClosestPoint.h`) with EPA fallback for penetration. Every convex shape exposes a **support function**; shapes are shrunk by their **convex radius** and re-inflated, giving rounded shapes so GJK usually terminates in the cheap non-penetrating case.
- **Shape hierarchy**: `Shape` (refcounted, immutable, shareable) created from `ShapeSettings::Create()` → `ShapeResult`. Categories: Convex (Sphere, Box, Capsule, TaperedCapsule, Cylinder, TaperedCylinder, ConvexHull, Triangle, Plane, Empty), Compound (StaticCompoundShape with internal BV tree, MutableCompoundShape), Decorated (Scaled, RotatedTranslated, OffsetCenterOfMass), non-convex (Mesh, HeightField). `SubShapeID` is a 32-bit bit-path through the hierarchy.
- **MeshShape internal format**: triangles sanitized and packed into a quad-AABB-tree blob — `NodeCodecQuadTreeHalfFloat` (4 children/node, half-float bounds) + `TriangleCodecIndexed8BitPackSOA4Flags` (vertices quantized to 64 bits relative to node bounds, blocks of 4 triangles with 8-bit local indices + 8-bit flags in 16 bytes). v5.3 scaled to 110M triangles.
- **HeightFieldShape**: N×N samples in blocks with per-block min/max quantization at 1–16 bits/sample, hierarchical grid culling, hole punching.
- **Active edge detection**: Mesh/HeightField bake per-edge "active" flags (threshold default cos 5°, concave edges always inactive); contacts with inactive edges get their normal replaced by the face normal to prevent ghost collisions. Runtime alternative: `mEnhancedInternalEdgeRemoval`.
- **Contact pipeline**: manifolds computed with speculative contact distance (contacts within 0.02 m created as speculative, solved velocity-only unless touching — the default anti-tunneling mechanism); manifold reduction merges coplanar sub-shape manifolds; body-pair cache reuses last step's manifold + warm-start lambdas when relative pose barely changed.

### 1.4 Solver

Sequential impulses with warm starting per Catto's GDC 2009 formulation — Rouwe: "You can see Jolt Physics as Box2D in 3D". Symplectic Euler. Per collision step: apply gravity → solve velocity constraints (10 iterations, contacts + constraints interleaved by island) → integrate positions → **position solver** (2 iterations, split-impulse-style direct position correction resolving penetration beyond slop without adding kinetic energy). Per-body and per-constraint iteration-count overrides since v5.0. v5.6 replaced per-contact-point friction with averaged-contact-point friction (2 linear + 1 angular row per manifold — 15% faster, 40% less memory).

- **Islands**: lock-free union-find with limited path compression over active bodies, links built during parallel narrowphase via compare-exchange; a single-threaded O(N) pass assigns island IDs. A **LargeIslandSplitter** re-splits huge islands into parallelizable batches. Bodies deliberately store no contact/constraint lists.
- **Sleeping**: island-wide — all bodies below `mPointVelocitySleepThreshold` (measured on 3 points) for `mTimeBeforeSleep` → island sleeps; touching a sleeping body wakes it.
- **Motion quality / CCD**: `Discrete` (default, speculative contacts catch most tunneling) vs `LinearCast` (post-integration cast, rewind to first contact with "time stealing"). MeshEditor never uses LinearCast.
- **Sub-stepping**: v4.0 removed integration sub-steps — `Update(dt, collisionSteps, tempAllocator, jobSystem)` where each collision step is a full simulate at `dt/steps`.
- **Determinism**: bit-exact given identical binary + API call order + update cadence. `CROSS_PLATFORM_DETERMINISTIC` (~8% cost) extends across compilers/OSes/architectures. Parallel union-find requires a constraint re-sort to stay deterministic. Callback *invocation order* is not deterministic (multi-threaded), but contact resolution order is made deterministic internally.

### 1.5 Constraints and motors

Each a `TwoBodyConstraint` from a `ConstraintSettings` subclass (Body1/Body2 spaces via `EConstraintSpace::LocalToBodyCOM/WorldSpace`): Fixed, Point, Hinge, Slider, Distance, Cone, SwingTwist, SixDOF (per-axis Free/Limited/Fixed, per-axis motors, pyramid swing limits since v5.0), Path, Gear, RackAndPinion, Pulley, plus vehicles. **Motors** (`MotorSettings.h`): states Off / Velocity / Position / PositionAndVelocity (v5.6); position motors driven by `SpringSettings` — modes `FrequencyAndDamping` | `StiffnessAndDamping` | `MassNormalizedStiffnessAndDamping` (v5.6, added for glTF `KHR_physics_rigid_bodies` interop); force/torque limits; frequency ≤ 0 gives hard limits.

### 1.6 Multithreading model and API shape

- **JobSystem** abstraction: `Update` decomposes into up to 2048 dependency-linked jobs; reference `JobSystemThreadPool`, `JobSystemSingleThreaded` for debugging. `TempAllocatorImpl` is a pre-reserved LIFO scratch allocator.
- **No double-buffered body state**: one copy, protected by a fixed array of sharded body mutexes (≥32 drops per-thread wait to ~10 µs at 13 workers). The broadphase tree is effectively double-buffered for one frame after rebuild.
- **BodyInterface** is the thread-safe façade (with NoLock variants); `BodyLockRead/Write` RAII-lock a `BodyID` for multi-property access. Queries snapshot a `TransformedShape` under the lock, then run narrowphase outside it. `BodyID` = 23-bit index + 8-bit sequence number.
- **Lifecycle**: `CreateBody(BodyCreationSettings)` → `AddBody(id, EActivation)` → `RemoveBody` → `DestroyBody`; batch variants. `PhysicsSystem::Init(maxBodies, numBodyMutexes, maxBodyPairs, maxContactConstraints, bpInterface, objVsBpFilter, objPairFilter)`.
- **Queries**: `GetBroadPhaseQuery()` (AABB-level) and `GetNarrowPhaseQuery()` (CastRay, CollidePoint, CollideShape, CastShape) with the `CollisionCollector` pattern and a filter chain (BroadPhaseLayerFilter → ObjectLayerFilter → BodyFilter → ShapeFilter). `SimShapeFilter` (v5.3) filters sub-shape pairs during simulation.
- **Contact listeners**: `OnContactValidate` / `OnContactAdded` / `OnContactPersisted` / `OnContactRemoved`, called from simulation job threads (thread-safe required, order nondeterministic; Added/Persisted fire before the solver so `ContactSettings` can be modified: combined friction/restitution, sensor flag, relative surface velocity, per-contact inv-mass scale). `EstimateCollisionResponse` predicts impulses from a manifold.

### 1.7 v5.6 GPU compute layer

The big one for this project: **`JPH::Compute`** ([Jolt/Compute/](https://github.com/jrouwe/JoltPhysics/tree/master/Jolt/Compute), [PR #1847](https://github.com/jrouwe/JoltPhysics/pull/1847)) — `ComputeSystem`/`ComputeBuffer`/`ComputeQueue`/`ComputeShader` interfaces with **DX12, Vulkan, and Metal backends plus a CPU fallback** (`Jolt/Compute/MTL/*.mm` is real Objective-C++). Shaders authored in HLSL, cross-compiled via dxc + spirv-cross to MSL; the same HLSL compiles *as C++* for the CPU backend. First and only client: GPU strand-based hair (Cosserat rods, ~15 dispatches). **No GPU rigid-body solver exists or is planned** — in [discussion #501](https://github.com/jrouwe/JoltPhysics/discussions/501) Rouwe frames GPU work as optional add-ons ("lots of X"), never a replacement for the CPU rigid pipeline, and rules out CUDA. The CPU-executable-shader discipline is the pattern to keep (authoring MSL directly rather than HLSL→spirv-cross).

### 1.8 Test suite (porting targets)

Framework: **doctest** (vendored) wrapped by `UnitTestFramework.h`; `PhysicsTestContext` is the key harness — creates a `PhysicsSystem` with standard layers, steps deterministically, provides closed-form position/velocity predictions; `Logging*Listener` classes record callback sequences for exact-order assertions. Directory: [UnitTests/](https://github.com/jrouwe/JoltPhysics/tree/master/UnitTests).

- **Physics/**: `PhysicsTests.cpp` (free fall, forces/torques, restitution/friction ramps, kinematics, damping — the primary conformance suite), `PhysicsDeterminismTests`, `ContactListenerTests` (callback ordering + ContactSettings modification), `SensorTests`, `BroadPhaseTests`, `CastShapeTests`/`CollideShapeTests`/`CollidePointTests`/`RayShapeTests`, `ActiveEdgesTests`, `HeightFieldShapeTests`, `ShapeTests` (mass properties), constraint suites (`HingeConstraintTests`, `SixDOFConstraintTests`, … — limits/motors/springs vs analytic solutions), `MotionQualityLinearCastTests`, `EstimateCollisionResponseTest`, `CollisionGroupTests`.
- **Geometry/**: `GJKTests`, `EPATests`, `ConvexHullBuilderTest`, `ClosestPointTests` — directly portable for validating a reimplemented narrowphase.
- **Compute/ComputeTests.cpp** — the v5.6 GPU compute interface test, a useful reference for testing a Metal backend.
- **Samples** (`Samples/Tests/General/…`): the stability scenes (BoxStack/Pyramid/Wall, HighSpeed, Funnel, IslandTest, ActiveEdges, ManifoldReduction) used as behavioral benchmarks; a separate `PerformanceTest` app validates determinism via state hashing.

### 1.9 The API surface MeshEditor actually uses

From a full scan of MeshEditor (all Jolt usage quarantined in `src/physics/PhysicsSystem.cpp`, ~1900 lines). This scopes phase-1 parity; the audio-readback machinery is context only (see Synthesis).

- **Setup**: `PhysicsSystem::Init(65536, 0, 65536, 65536, …)` with 2 object layers (NonMoving/Moving), 2 broadphase layers, `TempAllocatorImpl{64 MiB}`, `JobSystemThreadPool{hw_concurrency-1}`. `SetContactListener`, `SetSimShapeFilter`, `SetSimCollideBodyVsBody` (forcing `CollideWithBackFaces`), `SetGravity`, `OptimizeBroadPhase`, `GetPhysicsSettings`/`SetPhysicsSettings`.
- **Settings touched**: only `mNumVelocitySteps` (default 10, UI 2–50), `mPenetrationSlop` and `mSpeculativeContactDistance` (both auto-scaled to 2% of min collider dimension). Stepping: fixed `Update(dt·timeScale, substepsPerFrame)` with substeps default 10, no accumulator, frame-baked pose playback.
- **Bodies**: `BodyInterface::{CreateBody, AddBody, RemoveBodies, DestroyBodies, SetShape, Get/SetMotionType, SetGravityFactor, SetFriction, SetRestitution, ActivateBody, IsActive, SetPositionAndRotation(WhenChanged), GetPosition, GetRotation}`; `BodyLockWrite`; `BodyCreationSettings` fields incl. `mAllowedDOFs`, `mMassPropertiesOverride`, `mCollisionGroup`, damping, velocities. Readback is poses only — velocities are never read.
- **Shapes**: Box, Sphere, Capsule, TaperedCapsule, Cylinder, TaperedCylinder, ConvexHull, Mesh, Plane (static-only, thin-box fallback otherwise), Empty, RotatedTranslated, Scaled, StaticCompound, OffsetCenterOfMass. Sub-shape user data for compound → entity resolution.
- **Constraints**: SixDOF (WorldSpace, pyramid swing, per-axis free/limited/fixed, limit springs, per-axis position/velocity motors) and Hinge (detected hinge-like joints, since SixDOF swing-twist is ill-conditioned at ±π; velocity motor, torque limit). `SpringSettings` with `StiffnessAndDamping` / `MassNormalizedStiffnessAndDamping` (KHR Force vs Acceleration drive modes).
- **Filtering**: custom `GroupFilter` subclass implementing KHR collision-filter bitmask semantics + joint-pair disabling; `OnContactValidate` adds sub-shape-level filtering; `SimShapeFilter` rejects Mesh-vs-Mesh.
- **Listeners**: all four `ContactListener` overrides; reads `ContactManifold::{mWorldSpaceNormal, mRelativeContactPointsOn1, mSubShapeID1/2}`, writes `ContactSettings::{mCombinedFriction, mCombinedRestitution}` (KHR combine-mode priority).
- **Never used**: raycasts/shapecasts/`NarrowPhaseQuery`/`BroadPhaseQuery` (MeshEditor picks via its own BVH), `EMotionQuality`/CCD, soft bodies, characters, vehicles, `StateRecorder`, activation listeners.
- **glTF integration**: scenes via `KHR_implicit_shapes` + `KHR_physics_rigid_bodies` + `KHR_audio_rigid_bodies` (fastgltf fork); the [glTF_Physics](https://github.com/eoineoineoin/glTF_Physics) sample scenes drive validation.
- **Workarounds worth designing away**: dynamic-trimesh→convex-hull promotion, mesh mass-properties placeholder, back-face forcing, batched body removal ordered after joint removal (Jolt asserts otherwise), no angular spring limits on SixDOF (warns, treats as hard).

### Sources

- Repo: https://github.com/jrouwe/JoltPhysics · Architecture: https://github.com/jrouwe/JoltPhysics/blob/master/Docs/Architecture.md · Doxygen: https://jrouwe.github.io/JoltPhysics/index.html
- Release notes: https://github.com/jrouwe/JoltPhysics/blob/master/Docs/ReleaseNotes.md · API changes: https://github.com/jrouwe/JoltPhysics/blob/master/Docs/APIChanges.md
- GDC 2022 talk: https://gdcvault.com/play/1027560/Architecting-Jolt-Physics-for-Horizon · notes: https://jrouwe.nl/architectingjolt/ArchitectingJoltPhysics_Rouwe_Jorrit_Notes.pdf · Guerrilla writeup: https://www.guerrilla-games.com/read/architecting-jolt-physics-for-horizon-forbidden-west
- PhysicsSettings: https://github.com/jrouwe/JoltPhysics/blob/master/Jolt/Physics/PhysicsSettings.h
- Mesh codecs: https://github.com/jrouwe/JoltPhysics/blob/master/Jolt/AABBTree/TriangleCodec/TriangleCodecIndexed8BitPackSOA4Flags.h · https://github.com/jrouwe/JoltPhysics/blob/master/Jolt/AABBTree/NodeCodec/NodeCodecQuadTreeHalfFloat.h
- Constraints: https://github.com/jrouwe/JoltPhysics/tree/master/Jolt/Physics/Constraints · ContactListener: https://github.com/jrouwe/JoltPhysics/blob/master/Jolt/Physics/Collision/ContactListener.h
- Compute layer: https://github.com/jrouwe/JoltPhysics/tree/master/Jolt/Compute · PR: https://github.com/jrouwe/JoltPhysics/pull/1847 · GPU direction: https://github.com/jrouwe/JoltPhysics/discussions/501
- UnitTests: https://github.com/jrouwe/JoltPhysics/tree/master/UnitTests · Samples: https://github.com/jrouwe/JoltPhysics/blob/master/Docs/Samples.md · PerformanceTest: https://github.com/jrouwe/JoltPhysics/blob/master/Docs/PerformanceTest.md

---

## 2. GPU collision detection: broad-phase and narrow-phase

### 2.1 Broad-phase

#### Sweep-and-prune on GPU

- **Liu, Harada, Lee & Kim, 2010 — "Real-time Collision Culling of a Million Bodies on Graphics Processing Units" (gSaP), SIGGRAPH Asia / TOG 29(6).** The canonical GPU sweep-and-prune: massively parallel SAP that mitigates dense endpoint clustering by choosing the sweep axis via PCA and adding spatial subdivision to cut false positives, reaching ~1M moving bodies at interactive rates. *Metal/UMA:* maps directly to radix sort + segmented sweep kernels; the PCA axis selection is a cheap reduction. https://graphics.ewha.ac.kr/gSaP/gSaP.pdf
- **Le Grand, 2007 — "Broad-Phase Collision Detection with CUDA," GPU Gems 3 ch. 32.** Earliest detailed GPU broad-phase writeup: parallel uniform spatial subdivision with the home-cell/phantom-cell duplicate-avoidance scheme and cell-ID radix sort. Also the clearest explanation of why classic *incremental* SAP (per-swap serial dependencies) does not map cleanly to GPUs. https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-32-broad-phase-collision-detection-cuda

#### Uniform grids and spatial hashing

- **Green, 2007–2012 — "Particle Simulation using CUDA" (NVIDIA whitepaper).** The reference sorting-based uniform grid: hash to cell ID, radix-sort, find cell start/end — no atomics, no fixed per-cell capacity. Degrades when object sizes vary widely. https://developer.download.nvidia.com/compute/cuda/2_2/sdk/website/projects/particles/doc/particles.pdf
- **Teschner et al., 2003 — "Optimized Spatial Hashing for Collision Detection of Deformable Objects," VMV 2003.** The classic `(x·73856093 ⊕ y·19349663 ⊕ z·83492791) mod n` hash for infinite implicit grids; cell size optimal ≈ primitive size. The answer for unbounded worlds. https://matthias-research.github.io/pages/publications/tetraederCollision.pdf
- **Fan et al., 2011 — "A Hierarchical Grid Based Framework for Fast Collision Detection," CGF 30(7).** Two-level GPU grid for non-uniform size distributions — the standard fix for the uniform grid's mixed-size weakness. https://onlinelibrary.wiley.com/doi/10.1111/j.1467-8659.2011.02019.x

#### Linear BVH construction family

- **Lauterbach et al., 2009 — "Fast BVH Construction on GPUs" (LBVH), Eurographics.** Morton codes + sort + hierarchy emission — turned BVH build into a sort problem, making full rebuild-per-frame viable. https://luebke.us/publications/eg09.pdf
- **Karras, 2012 — "Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees," HPG.** In-place parallel binary radix tree over sorted Morton codes: every internal node emitted independently in one kernel, AABBs filled by bottom-up atomic refit. The de-facto standard fast GPU builder; the likely core of a Metal broad-phase BVH. https://research.nvidia.com/publication/2012-06_maximizing-parallelism-construction-bvhs-octrees-and-k-d-trees · Tutorial series "Thinking Parallel" I–III (Part II is the direct blueprint for a `findPairs` kernel): https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/
- **Apetrei, 2014 — "Fast and Simple Agglomerative LBVH Construction," CGVC.** Fuses Karras's two passes into a single bottom-up kernel (parents found on the fly via XOR-of-adjacent-Morton-codes) — simpler and faster with identical topology. Arguably the best-value builder for a Metal broad-phase: one radix sort + one kernel. https://diglib.eg.org/items/3aca7692-f2be-4b5d-a7f0-b7a865be6e5b · revision: https://arxiv.org/abs/2402.00665
- **HLBVH (Pantaleoni & Luebke 2010), TRBVH (Karras & Aila 2013), PLOC (Meister & Bittner 2018), PLOC++ (Benthin et al. 2022), H-PLOC (Benthin et al. 2024).** The quality ladder for on-device builds — mostly overkill for a body-AABB broad-phase (small trees, latency dominates), relevant if large triangle BVHs must be built on-device. PLOC++'s subgroup-heavy kernels are a good case study in porting wave ops to Metal SIMD-group functions. https://research.nvidia.com/publication/2010-06_hlbvh-hierarchical-lbvh-construction-real-time-ray-tracing · https://research.nvidia.com/publication/2013-07_fast-parallel-construction-high-quality-bounding-volume-hierarchies · https://meistdan.github.io/publications/ploc/paper.pdf · https://cdrdv2-public.intel.com/737298/ploc-for-bounding-volume.pdf · https://gpuopen.com/download/HPLOC.pdf

#### Refit vs. rebuild for dynamic scenes

- **Kopta et al., 2012 — "Fast, Effective BVH Updates for Animated Scenes," I3D.** Refitting preserves topology so quality degrades as objects move; interleaved tree rotations arrest the degradation at near-refit cost. https://hwrt.cs.utah.edu/papers/hwrt_rotations.pdf
- **Practical rule** across the engine literature: rebuild the broad-phase from scratch every frame on GPU — construction is a tiny fraction of frame time. Refit-only is exactly right for a midphase over *deforming* meshes (the modal case), and unnecessary for rigid meshes (store the BVH in local space, transform the query). The production counterexample is PhysX's incremental GPU SAP, which optimizes for mostly-sleeping scenes.

#### Traversal and pair management

- **Aila & Laine, 2009 — "Understanding the Efficiency of Ray Traversal on GPUs," HPG.** The foundational GPU tree-traversal analysis: persistent threads with global work queues, speculative traversal, lane convergence. Caveat for Apple GPUs: no documented cross-threadgroup forward-progress guarantee — prefer oversubscribed dispatch with atomic work counters over blocking persistent-thread queues. https://research.nvidia.com/sites/default/files/publications/aila2009hpg_paper.pdf
- **Chitalu et al., 2018 — "Bulk-synchronous parallel simultaneous BVH traversal for collision detection on GPUs," I3D.** Level-synchronous BVTT front expansion, cacheable across frames — the safe traversal pattern on Apple GPUs. https://www.researchgate.net/publication/324904327
- **Pair-management practice** (Karras, Le Grand, Coumans): emit `(i, j)` only when `i < j`; output via two-pass count → prefix-sum → write (deterministic) or atomic append (faster, needs headroom + post-sort for determinism). On UMA the pair buffer is host-visible with no blit, but winning designs still avoid CPU round-trips via `dispatchThreadgroups(indirectBuffer:)`.

#### Engine practice — broad-phase

- **PhysX 5 `PxBroadPhaseType::eGPU`**: "a GPU implementation of incremental sweep and prune, additionally using ABP-style initial pair generation" — production practice is *incremental* GPU SAP with persistent sorted endpoint lists, performing well with many sleeping and many moving shapes. https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/_api_build/structPxBroadPhaseType.html
- **Bullet 3 OpenCL** shipped three interchangeable broad-phases (`b3GpuSapBroadphase` — one-axis parallel SAP with per-frame GPU radix re-sort, `b3GpuGridBroadphase`, `b3GpuParallelLinearBvh`), plus a hybrid pairing the grid for small objects with SAP for large ones. The hybrid is directly reusable. https://github.com/bulletphysics/bullet3/tree/master/src/Bullet3OpenCL/BroadphaseCollision
- **MuJoCo Warp**: `nxn` (filtered brute force — competitive for small worlds) and `sap` with segmented sort. For moderate object counts, filtered N² can beat clever structures on GPU — benchmark before building anything fancy. https://github.com/google-deepmind/mujoco_warp
- **Genesis**: warm-started one-axis SAP — endpoints insertion-sorted each frame ("almost sorted from the previous frame ⇒ insertion sort is almost linear"), hibernation-aware. Validates temporal-coherence SAP on GPU, on Metal. https://genesis-world.readthedocs.io/en/v0.3.14/user_guide/advanced_topics/collision_contacts_forces.html

### 2.2 Narrow-phase

#### Convex queries: GJK, EPA, MPR, SAT

- **Gilbert, Johnson & Keerthi, 1988** — the original support-mapping simplex-descent distance algorithm. Support mappings are ideal for GPU (no per-shape geometry code); the variable iteration count and simplex case analysis is the canonical divergence hazard. https://ieeexplore.ieee.org/document/2083
- **Cameron, 1997 — enhanced GJK.** Hill-climbing warm starts make tracked queries expected-constant-time — maps directly to a persistent per-pair GPU cache. https://www.cs.ox.ac.uk/stephen.cameron/distances/
- **Montanari, Petrinic & Barbieri, 2017 — "Improving the GJK Algorithm…", TOG 36(3).** Signed-volume distance subalgorithm: machine-precision robust on degenerate simplices, removes the backup-procedure branch entirely (one fewer divergence path), 15–30% faster in contact cases. **The formulation to implement.** https://ora.ox.ac.uk/objects/uuid:69c743d9-73de-4aff-8e6f-b4dd7c010907 · reference impl: https://github.com/MattiaMontanari/openGJK
- **Muratori, 2006 — "Implementing GJK."** The geometric simplex-case reformulation most shipping engines implement. https://caseymuratori.com/blog_0003
- **Montaut et al., 2022 — "Collision Detection Accelerated," RSS.** GJK as Frank-Wolfe + Nesterov acceleration, up to ~2× fewer iterations — tighter worst-case bounds for fixed-iteration GPU kernels. https://arxiv.org/abs/2205.09663
- **van den Bergen, 2001/2003 — EPA.** The dynamically growing polytope (face heap, variable memory) is the worst-behaved convex kernel on GPU — engines run bounded-iteration EPA in fixed scratch (MuJoCo Warp, Genesis). https://graphics.stanford.edu/courses/cs468-01-fall/Papers/van-den-bergen.pdf
- **Snethen, 2008 — XenoCollide / MPR, Game Programming Gems 7.** Portal refinement on the Minkowski difference: fixed structure (a portal is always 3 vertices, no growing polytope) — arguably the most GPU-friendly convex algorithm, and Genesis's default. No separation distance for non-touching pairs. https://github.com/erwincoumans/xenocollide · libccd (CPU reference/oracle): https://github.com/danfis/libccd
- **Gregorius, 2013 — "The Separating Axis Test between Convex Polyhedra," GDC.** SAT with Gauss-map edge pruning and feature identification. Fixed, predictable loop bounds — more FLOPs but *uniform* control flow, which is why branchless SAT keeps reappearing on SIMT hardware (MJX chose it over GJK for exactly this reason, though only viable for small hulls). https://media.gdcvault.com/gdc2013/slides/822403Gregorius_Dirk_TheSeparatingAxisTest.pdf

#### Contact manifold generation

- **Gregorius, 2015 — "Robust Contact Creation for Physics Simulations," GDC.** The standard one-shot manifold recipe: reference/incident face selection, Sutherland-Hodgman clipping, reduction to ≤4 points maximizing area, edge-edge contacts, feature IDs for warm starting. The recipe for GPU one-shot manifolds — one thread or SIMD group per pair, fixed-size clip buffers. https://media.steampowered.com/apps/valve/2015/DirkGregorius_Contacts.pdf
- **Catto — Box2D publications.** "Contact Manifolds" (GDC 2007): incremental vs one-shot manifolds, feature IDs; "Computing Distance using GJK" (GDC 2010): the most accessible robust-GJK treatment. Manifold point IDs + accumulated impulses are per-pair persistent state — on UMA, CPU tooling can inspect and patch them with zero copies, invaluable when debugging a young GPU engine. https://box2d.org/publications/
- **Persistent contact manifolds (Bullet PCM / PhysX PCM).** Accumulate one new GJK/MPR point per frame into a 4-point cache (PhysX: up to 6, distance-based, `GuPersistentContactManifold.h`) — *the only form of contact generation implemented on PhysX's GPU*. Tiny fixed per-pair state, one convex query per frame; requires a stable persistent pair table. https://nvidia-omniverse.github.io/PhysX/physx/5.5.0/docs/AdvancedCollisionDetection.html
- **Contact patches (PhysX contact stream).** Solver-facing contacts organized as patches sharing one normal and material state, with bounded reduction — patch-major layout means one friction-basis computation per patch and coalesced solver access. Kier Storey, "Game Physics on the GPU with PhysX 3.4," GDC 2017: https://www.gdcvault.com/play/1024345/Game-Physics-on-the-GPU
- **Perturbation-based manifold sampling.** Genesis (and MuJoCo's convex path) build multi-point manifolds by re-running the convex query under small pose perturbations around the contact normal (up to 5 contacts). Embarrassingly parallel and branch-uniform at N× narrow-phase cost — a reasonable trade on ALU-rich Apple GPUs.
- **Jolt's manifold pipeline** (face-vs-face clipping, cross-subshape manifold reduction, baked active edges) defines the quality bar for stable stacking and mesh sliding — active-edge flags baked at cook time are cheap to evaluate in a GPU trimesh kernel. See §1.3.

#### Divergence and batching

- **Lauterbach, Mo & Manocha, 2010 — gProximity, CGF 29(2).** GPU BVH collision/distance queries with lightweight work queues for irregular traversal — the canonical divergence answer, with the same Apple forward-progress caveat. http://gamma.cs.unc.edu/GPUCOL/
- **Pair-type batching (industry practice).** PhysX buckets candidate pairs by shape-type pair, one specialized kernel per bucket; MuJoCo Warp builds a specialized kernel per convex pair type; Genesis advertises divergence-minimized narrowphase. Directly expressible as Metal function constants / one PSO per shape-pair type, with CPU fallback nearly free on UMA.
- **MJX's branchless SAT (cautionary tale).** XLA executes both branch sides, so MJX replaced GJK/EPA with branchless SAT — but only for <32-vert hulls, and MJX ships no broad-phase at all. Metal's real (merely slow) intra-SIMD branching means the extreme position is not automatically optimal — measure. https://mujoco.readthedocs.io/en/stable/mjx.html

#### Trimesh and SDF collision

- **PhysX 5 GPU narrow phase**: PCM-only, GPU-cooked mesh data, convex hulls capped at ≤64 verts/64 polys (sized so a hull fits a warp's registers) — adopt an analogous cap sized to Apple's 32-wide SIMD groups. Convex-trimesh uses the BVH34 midphase (4-wide, refittable). 4-wide trees map better to SIMD-group traversal than binary BVHs (Jolt's quad-tree makes the same choice). https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/docs/GPURigidBodies.html
- **Macklin et al., 2020 — "Local Optimization for Robust Signed Distance Field Collision," PACMCGIT/I3D.** Per-element fixed-iteration local optimization against the SDF isosurface recovers sharp-feature and edge-edge contacts that vertex sampling misses; the algorithmic core of PhysX 5's SDF collision. The best starting point for a Metal SDF narrow phase — texture-stored SDFs get free trilinear filtering from Apple's sampler hardware. https://mmacklin.com/sdfcontact.pdf
- **MuJoCo SDF plugin**: analytic user SDFs + gradient-descent deepest-point search, GPU port in MJWarp — analytic SDFs cost no memory but need per-shape shader functions (Metal visible function tables or PSO specialization). https://github.com/google-deepmind/mujoco/tree/main/plugin/sdf
- **Trimesh-trimesh in practice**: game engines avoid dynamic mesh-mesh (convex decomposition instead); when supported it is SDF-vs-mesh. Plan on convex decomposition + hull-hull as the main dynamic path, BVH midphase for static trimesh, SDF for complex dynamic shapes.

#### Speculative contacts and CCD

- **Firth, 2011 — "Speculative Contacts: a continuous collision engine approach."** Generate contacts *before* touching; the solver removes exactly the closing velocity that would cross the gap. No sweeps, no TOI ordering, fully data-parallel — the natural GPU CCD and the right first implementation (it is also Jolt's default mechanism). https://wildbunny.co.uk/blog/2011/03/25/speculative-contacts-an-continuous-collision-engine-approach-part-1/ (mirror: http://vodacek.zvb.cz/archiv/286.html)
- **Mirtich, 1996 (conservative advancement); Tang, Kim & Manocha 2009 (C2A); Catto, "Continuous Collision," GDC 2013 (bilateral advancement).** True-TOI machinery — iterative and data-dependent, a poor batch-GPU citizen. Schedule as a sparse second pass over the few fast pairs, as PhysX does. https://box2d.org/files/ErinCatto_ContinuousCollision_GDC2013.pdf · https://graphics.ewha.ac.kr/projects/details/C2A/C2A.pdf

### 2.3 Cross-cutting takeaways

1. **Convergent skeleton** (see Synthesis diagram): sort-based broadphase → pair compaction → specialized narrowphase → fixed-capacity contact buffers with reduction.
2. **Radix sort is the prerequisite.** Grid, SAP, and LBVH all assume a fast GPU radix sort; Metal has no CUB equivalent — build threadgroup radix sort + prefix scan first.
3. **UMA changes the readback story, not the pipeline shape.**
4. **Respect Apple's execution model**: bounded level-synchronous passes over persistent-thread blocking queues.
5. **Fight divergence with specialization and fixed bounds**: signed-volume GJK, bounded EPA or MPR, clipping- or perturbation-based one-shot manifolds, precomputed lookups (Genesis support fields, Jolt baked active edges).
6. **Speculative contacts first, sweeps second.**
7. **Persist per-pair state in shared buffers** — warm-start simplices, feature IDs, accumulated impulses are tiny fixed records, CPU-inspectable for free.

### Sources

- gSaP: https://graphics.ewha.ac.kr/gSaP/gSaP.pdf · Le Grand GPU Gems 3: https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-32-broad-phase-collision-detection-cuda · Green particles: https://developer.download.nvidia.com/compute/cuda/2_2/sdk/website/projects/particles/doc/particles.pdf · Teschner hashing: https://matthias-research.github.io/pages/publications/tetraederCollision.pdf · Fan hierarchical grid: https://onlinelibrary.wiley.com/doi/10.1111/j.1467-8659.2011.02019.x
- LBVH: https://luebke.us/publications/eg09.pdf · Karras 2012: https://research.nvidia.com/publication/2012-06_maximizing-parallelism-construction-bvhs-octrees-and-k-d-trees · Thinking Parallel I–III: https://developer.nvidia.com/blog/thinking-parallel-part-i-collision-detection-gpu/ · Apetrei: https://diglib.eg.org/items/3aca7692-f2be-4b5d-a7f0-b7a865be6e5b · HLBVH: https://research.nvidia.com/publication/2010-06_hlbvh-hierarchical-lbvh-construction-real-time-ray-tracing · Garanzha work queues: https://www.highperformancegraphics.org/previous/www_2011/media/Papers/HPG2011_Papers_Garanzha.pdf · TRBVH: https://research.nvidia.com/publication/2013-07_fast-parallel-construction-high-quality-bounding-volume-hierarchies · PLOC: https://meistdan.github.io/publications/ploc/paper.pdf · PLOC++: https://cdrdv2-public.intel.com/737298/ploc-for-bounding-volume.pdf · H-PLOC: https://gpuopen.com/download/HPLOC.pdf · Kopta rotations: https://hwrt.cs.utah.edu/papers/hwrt_rotations.pdf
- Aila & Laine: https://research.nvidia.com/sites/default/files/publications/aila2009hpg_paper.pdf · Chitalu BSP traversal: https://www.researchgate.net/publication/324904327 · gProximity: http://gamma.cs.unc.edu/GPUCOL/ · gDist: https://arxiv.org/abs/2411.11244 · Collision-Streams: https://gamma.cs.unc.edu/CSTREAMS/i3d.pdf
- GJK 1988: https://ieeexplore.ieee.org/document/2083 · Cameron: https://www.cs.ox.ac.uk/stephen.cameron/distances/ · Montanari: https://ora.ox.ac.uk/objects/uuid:69c743d9-73de-4aff-8e6f-b4dd7c010907 · openGJK: https://github.com/MattiaMontanari/openGJK · Muratori: https://caseymuratori.com/blog_0003 · Montaut: https://arxiv.org/abs/2205.09663 · EPA: https://graphics.stanford.edu/courses/cs468-01-fall/Papers/van-den-bergen.pdf · XenoCollide: https://github.com/erwincoumans/xenocollide · libccd: https://github.com/danfis/libccd · Gregorius SAT: https://media.gdcvault.com/gdc2013/slides/822403Gregorius_Dirk_TheSeparatingAxisTest.pdf
- Gregorius contacts: https://media.steampowered.com/apps/valve/2015/DirkGregorius_Contacts.pdf · Box2D publications: https://box2d.org/publications/ · Catto GJK: https://box2d.org/files/ErinCatto_GJK_GDC2010.pdf · Sutherland-Hodgman: https://dl.acm.org/doi/10.1145/360767.360802 · PhysX PCM/CCD: https://nvidia-omniverse.github.io/PhysX/physx/5.5.0/docs/AdvancedCollisionDetection.html · PhysX 3.4 PCM source: https://github.com/NVIDIAGameWorks/PhysX-3.4/blob/master/PhysX_3.4/Source/GeomUtils/src/pcm/GuPersistentContactManifold.h · Storey GDC 2017: https://www.gdcvault.com/play/1024345/Game-Physics-on-the-GPU
- SDF collision: https://mmacklin.com/sdfcontact.pdf · MuJoCo SDF: https://github.com/google-deepmind/mujoco/tree/main/plugin/sdf · PhysX GPU rigid bodies: https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/docs/GPURigidBodies.html · PhysX collision: https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/docs/RigidBodyCollision.html · PxBroadPhaseType: https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/_api_build/structPxBroadPhaseType.html
- Speculative contacts: https://wildbunny.co.uk/blog/2011/03/25/speculative-contacts-an-continuous-collision-engine-approach-part-1/ · Mirtich thesis: https://dl.acm.org/doi/10.5555/924581 · C2A: https://graphics.ewha.ac.kr/projects/details/C2A/C2A.pdf · Catto continuous collision: https://box2d.org/files/ErinCatto_ContinuousCollision_GDC2013.pdf
- Bullet3 OpenCL: https://github.com/bulletphysics/bullet3/tree/master/src/Bullet3OpenCL/BroadphaseCollision · https://github.com/bulletphysics/bullet3/tree/master/src/Bullet3OpenCL/NarrowphaseCollision · Coumans course notes: https://www.multithreadingandvfx.org/course_notes/GPU_rigidbody_using_OpenCL.pdf
- MJWarp: https://github.com/google-deepmind/mujoco_warp · MJX: https://mujoco.readthedocs.io/en/stable/mjx.html · Genesis collision docs: https://genesis-world.readthedocs.io/en/v0.3.14/user_guide/advanced_topics/collision_contacts_forces.html · Flex: https://mmacklin.com/uppfrta_preprint.pdf · ToruNiina/lbvh: https://github.com/ToruNiina/lbvh · Apple Metal RT: https://developer.apple.com/documentation/metal/accelerating-ray-tracing-using-metal · Ten Minute Physics: https://matthias-research.github.io/pages/tenMinutePhysics/index.html

---

## 3. Constraint solvers and GPU parallelization

Recurring Metal-specific concerns throughout: synchronization only at threadgroup scope or kernel boundaries, 32-bit integer atomics as the portable path (no float atomics in MSL), SIMD-group reductions and `simdgroup_matrix` for small dense linear algebra, unified memory making CPU↔GPU handoff nearly free, and low dispatch overhead via Metal 4 enabling many small per-color/per-substep dispatches.

### 3.1 Sequential impulses / PGS (the Box2D/Jolt lineage)

- **Catto, "Iterative Dynamics with Temporal Coherence" (GDC 2005).** Sequential impulses ≡ Projected Gauss-Seidel on the velocity-level LCP, plus *warm starting* — the algorithmic core of Box2D, Jolt, Rapier, Bullet. Warm starting requires persistent contact IDs, which on GPU means a stable contact-matching pass before solver setup. https://box2d.org/files/ErinCatto_IterativeDynamicsSlides_GDC2005.pdf
- **Catto, "Modeling and Solving Constraints" (GDC 2009), "Understanding Constraints" (GDC 2014).** Jacobian construction, effective mass, Baumgarte bias, NGS position correction (what Box2D/Jolt use to remove overlap without injecting energy — a GPU port needs it or substepped relaxation to match Jolt stacking). https://box2d.org/files/ErinCatto_ModelingAndSolvingConstraints_GDC2009.pdf · https://box2d.org/files/ErinCatto_UnderstandingConstraints_GDC2014.pdf
- **Baumgarte, 1972.** The original drift stabilization — cheap, energy-adding, timestep-sensitive; the parameterization the API must expose Jolt-compatibly. https://doi.org/10.1016/0045-7825(72)90018-7
- **Catto, "Soft Constraints" (GDC 2011).** Constraints as implicit spring-dampers parameterized by frequency and damping ratio, with exact per-iteration coefficients — as cheap as Baumgarte, frame-rate independent, non-energizing. The mathematical ingredient of TGS Soft, and the natural modal-coupling hook: a modal oscillator in contact is literally a frequency/damping-parameterized constraint. https://box2d.org/files/ErinCatto_SoftConstraints_GDC2011.pdf
- **Catto, Solver2D (2024) — Box2D v3.** Head-to-head PGS, PGS+NGS, block, PGS Soft, TGS variants, XPBD under equalized cost. Findings: substepping (TGS Soft = one GS iteration per substep + soft constraints + a bias-free relax pass) beats everything on stacks, mass ratios, joint chains; contacts in local anchors let substeps skip re-detection. The strongest evidence for the baseline architecture. https://box2d.org/posts/2024/02/solver2d/ · https://github.com/erincatto/solver2d
- **Block solvers.** Solving a manifold's rows (or a joint's 3–6 rows) as one small coupled solve removes intra-manifold fighting and raises arithmetic intensity per thread — maps to SIMD-group cooperative solves, and generalizes to (6+m)×(6+m) blocks when modal DOFs join a body's contact block.
- **PhysX TGS.** Production substepping-inside-the-solver: each "iteration" advances an internal sub-timestep with re-integrated positions. Identical semantics on CPU and GPU in a shipping engine. https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/RigidBodyDynamics.html

### 3.2 Position-based dynamics lineage

- **Müller et al., 2007 — PBD.** Position-level nonlinear GS projection; unconditionally stable, stiffness iteration-dependent. Root of the XPBD/small-steps branch. https://matthias-research.github.io/pages/publications/posBasedDyn.pdf
- **Macklin et al., 2016 — XPBD.** Compliance-based Lagrange multipliers give timestep-independent stiffness and physical force readout via λ/h² — force magnitudes for free, relevant to modal excitation. Solver2D found rigid XPBD ≈ TGS Soft minus the relax pass. https://mmacklin.com/xpbd.pdf
- **Macklin et al., 2019 — "Small Steps in Physics Simulation," SCA.** n substeps × 1 iteration beats 1 step × n iterations — smaller h reduces both integration error and conditioning. The "great equalizer" for parallel-friendly iterations, and it moves solver rates toward what modal coupling needs. https://mmacklin.com/smallsteps.pdf
- **Müller et al., 2020 — "Detailed Rigid Body Simulation with XPBD," SCA.** A complete rigid engine on XPBD + substeps: quaternion-based projection, current-direction contact normals, joints, soft-rigid coupling. The cleanest published recipe for a from-scratch substepped rigid solver; same work-shape as TGS, so both can share one GPU backend. https://matthias-research.github.io/pages/publications/PBDBodies.pdf

### 3.3 Coordinates, articulations, and contact models

- **Featherstone 2008** (reduced coordinates, O(n) tree dynamics — parallelizes across articulations, not within) vs **Baraff 1996** (maximal coordinates with O(n) acyclic sparse factorization). Maximal coordinates win here: simpler GPU data layout, uniform 6-DOF bodies, easy modal-DOF augmentation. https://link.springer.com/book/10.1007/978-1-4899-7560-7 · https://dl.acm.org/doi/10.1145/237170.237226
- **Todorov — MuJoCo (IROS 2012, ICRA 2014).** Convex, smooth, strictly-feasible contact optimization (regularized friction cone), solved by PGS/CG/Newton. Convexity buys guaranteed solvability and *smooth force trajectories* — directly desirable for modal excitation (no impulsive chatter). Trade: slightly soft contact, "action at a distance", diverging from Jolt hard contact. https://mujoco.readthedocs.io/en/stable/computation/index.html
- **Stewart & Trinkle 1996, Anitescu & Potra 1997, Stewart SIAM Review 2000.** The rigorous time-stepping LCP formulations iterative solvers approximate — the vocabulary of staggered projections. https://link.springer.com/article/10.1023/A:1008292328909 · https://doi.org/10.1137/S0036144599360110
- **Kaufman, Sueda, James, Pai, 2008 — "Staggered Projections," SIGGRAPH Asia.** Splits nonconvex Coulomb friction into two alternating convex QPs (normal projection, maximal-dissipation friction projection) with warm starts. **The solver Zheng & James built modal contact on** — the published precedent that a projection-structured contact solver hosts thousands of modal DOFs stably. https://www.cs.ubc.ca/labs/sensorimotor/projects/sp_sigasia08/
- **Zheng & James 2011** — see §6.3. The transfer to solver design: solver-order noise and non-smooth per-iteration impulses that are visually invisible are *audible*, pushing toward smooth/soft contact models and deterministic reduction orders on GPU.
- **Andrews, Erleben, Ferguson — "Contact and Friction Simulation for Computer Graphics" (SIGGRAPH 2021/22 course).** The best single survey stitching all of the above together. https://siggraphcontact.github.io/

### 3.4 Optimization-based and primal-dual solvers

- **Chen, Liu, Yang, Yuksel, 2024 — "Vertex Block Descent," SIGGRAPH.** Block coordinate descent on the implicit-Euler energy: per-block local Newton solves, graph-colored parallel execution, unconditional stability, monotone energy descent. A modal-augmented body is just a (6+m)-DOF block — VBD's per-block local solve is a very natural modal host. Weakness: stiff/hard constraints (fixed by AVBD). https://graphics.cs.utah.edu/research/projects/vbd/vbd-siggraph2024.pdf
- **Giles, Diaz, Yuksel, 2025 — "Augmented Vertex Block Descent," SIGGRAPH.** Augmented-Lagrangian layer over VBD: hard constraints via per-constraint dual updates + adaptive penalty warm-started across frames; demonstrated on rigid stacking, joints, friction at Real-Time Live! rates. The strongest new challenger for a GPU-first engine. Watch: Coulomb fidelity, behavioral match to Jolt, 3D validation depth. https://graphics.cs.utah.edu/research/projects/avbd/
- **Macklin et al., 2020 — "Primal/Dual Descent Methods for Dynamics," SCA.** Theory bridge for choosing relaxation/preconditioning between Jacobi and GS. https://mmacklin.com/primaldual.pdf
- **Overby et al., 2017 — "ADMM ⊇ Projective Dynamics," TVCG.** ADMM as the formal umbrella for "parallel local projections + coupling step" — the skeleton Kamino uses for multibody, and a candidate structure for coupling modal subsystems through shared contact duals. https://www.cse.iitd.ac.in/~narain/admm-pd/
- **IPC / Rigid IPC / ABD** (Li 2020, Ferguson 2021, Lan 2022, StiffGIPC 2024). The robustness ceiling (intersection-free, globally convergent), not realtime. ABD's "small fixed DOF block per body + barrier contact" (12 ≈ 6+m) is architecturally adjacent to modal-augmented bodies. https://ipc-sim.github.io/ · https://ipc-sim.github.io/rigid-ipc/ · https://github.com/Autodesk/affine-body-dynamics
- **Chen, Ly, Wojtan, 2024 — primal-dual non-smooth friction.** The current friction accuracy reference (interior-point, not realtime-GPU-shaped). https://visualcomputing.ist.ac.at/publications/2024/PDNSF/

### 3.5 GPU parallelization strategies

- **Fratarcangeli, Tibaldo, Pellacini, 2016 — "Vivace," SIGGRAPH Asia.** Randomized greedy graph coloring *on the GPU per frame*, then GS one color per dispatch. For rigid contacts the graph changes every frame — cheap dynamic coloring is mandatory. High-degree hubs (ground plane) inflate color counts → hybrid schemes (color the tractable part, Jacobi/mass-split the hubs). https://mfratarcangeli.github.io/publication/sigasia2016/
- **Ton-That, Kry, Andrews, 2022 — supernodal block coloring.** Cluster strongly-coupled constraints into supernodes, color at block granularity — fewer colors, bigger per-thread block solves, better convergence. Transplants directly to per-body-pair manifold blocks. http://profs.etsmtl.ca/sandrews/publication/xpbd_mig2022/
- **Tonge, Benevolenski, Voroshilov, 2012 — "Mass Splitting for Jitter-Free Parallel Rigid Body Simulation," SIGGRAPH.** The canonical parallel-Jacobi fix: divide each body's mass among its n contacts (solve with mass/n, apply with full mass) — provably convergent, jitter-free 5000-body GPU piles. The default answer for Jacobi-over-coloring choices and hub bodies. Slower convergence than GS — mitigated by substepping. https://dl.acm.org/doi/10.1145/2185520.2185601
- **Macklin et al., 2014 — FleX "Unified Particle Physics."** Averaged Jacobi: atomic scatter of deltas, divide by count, SOR relax. Metal note: atomic float-add is the crux — prefer gather (per-body loop over its constraints), which is also the deterministic option. https://mmacklin.com/uppfrta_preprint.pdf
- **Islands on GPU.** ECL-CC (Jaiganesh & Burtscher, HPDC 2018): lock-free GPU connected components — the right primitive for on-GPU island building (https://userweb.cs.txstate.edu/~mb92/papers/hpdc18.pdf). Jolt's lock-free union-find + deterministic re-sort (GDC notes). Catto's "Simulation Islands" (2023): per-frame island building is a serial bottleneck; Box2D v3 keeps *persistent* islands with deferred splitting, **for sleeping only**, and gets solver parallelism from coloring instead (https://box2d.org/posts/2023/10/simulation-islands/). With UMA, a hybrid (GPU union-find for big worlds, CPU incremental islands for sleep bookkeeping) costs no transfers. Islands remain the granularity for per-island solver selection — e.g. promoting an audio-active island to the modal-coupled solver.
- **PhysX 5 GPU solver.** Full GPU pipeline with identical CPU/GPU solver semantics; constraints organized into independent partitions (batching ≈ coloring) with averaging stages for high-degree bodies; only D6 joints natively GPU (other joint types round-trip to CPU — the trap to avoid: design all joints as data-driven rows). Pre-allocate everything. https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/GPURigidBodies.html
- **Bullet 3 OpenCL solver.** Two-level batching: local batching within a compute unit via shared-memory body reservation with retry, global batching by spatial cells, in-kernel barrier sync over batches to avoid ~200 dispatches per step. The most detailed public engineering writeup — and a cautionary tale on divergence/maintenance cost. https://www.multithreadingandvfx.org/course_notes/GPU_rigidbody_using_OpenCL.pdf
- **MuJoCo Warp / Genesis Newton solvers.** Per-world/per-island Newton on the convex objective with GPU tile ops — fixed-size blocked Cholesky per island, batched. Proven pattern for small dense per-island direct solves via cooperative SIMD-group tiles (`simdgroup_matrix`), attractive for audio-quality smooth forces when islands are modest. https://mujoco.readthedocs.io/en/latest/mjwarp/ · https://github.com/Genesis-Embodied-AI/genesis-world
- **Jacobi vs GS on stacks, substepping as equalizer.** Consistent picture (Tonge, Vivace, Solver2D, Small Steps, PhysX TGS docs): per-iteration convergence deficits matter less as substep count rises — each substep's problem is nearly warm-started by the last. Spend the GPU's advantage on substep rate; validate on tall towers, 1000:1 mass ratios, long joint chains.
- **Determinism.** Catto, "Determinism" (2024): fixed constraint ordering and reduction order regardless of scheduling; cross-platform additionally requires taming FMA/libm (https://box2d.org/posts/2024/08/determinism/). On GPU: gather-based accumulation, fixed-order SIMD-group tree reductions, stable sorts for pair/constraint IDs. Apple-Silicon-only is effectively one platform — design run-to-run determinism in from day one (replays, and *reproducible audio*).
- **Sleeping/waking on GPU.** Practice, not literature: islands are the sleep unit; per-body activity flags reduce to per-island votes via SIMD-group reduction; broadphase hits against sleeping islands re-activate. With UMA the CPU can own sleep bookkeeping at zero copy cost.

### 3.6 2024–2026 GPU solver papers

- **Kamino** (Chierichetti, Grandia, Bächer et al., Disney Research + NVIDIA, 2026): proximal-ADMM in maximal coordinates, handles kinematic loops, redundant constraints, extreme mass ratios; 4096 heterogeneous worlds. First production-adjacent evidence that an ADMM splitting is competitive for rigid multibody on GPU — a plausible future host for modal subsystems coupled through shared duals. https://disneyresearch.github.io/kamino/ · https://arxiv.org/abs/2603.16536
- **JGS2** (SIGGRAPH 2025): near-second-order converging Jacobi/GS hybrid for GPU elastodynamics — state of the art on the exact Jacobi-vs-GS tension. https://dl.acm.org/doi/10.1145/3731183
- **Zhou et al. 2025** — Gauss-Seidel ordering fixes for simultaneous collisions (Newton's-cradle propagation bias) — the same order sensitivity that pollutes contact-driven audio. https://dl.acm.org/doi/10.1145/3728291
- **Barrier-Augmented Lagrangian GPU contact** (SIGGRAPH Asia 2024) — further evidence the AL + parallel-local-solve pattern is the emerging GPU consensus. https://arxiv.org/abs/2407.00046

### Sources

- Catto GDC/blog: https://box2d.org/files/ErinCatto_IterativeDynamicsSlides_GDC2005.pdf · https://box2d.org/files/ErinCatto_ModelingAndSolvingConstraints_GDC2009.pdf · https://box2d.org/files/ErinCatto_SoftConstraints_GDC2011.pdf · https://box2d.org/files/ErinCatto_UnderstandingConstraints_GDC2014.pdf · https://box2d.org/posts/2024/02/solver2d/ · https://github.com/erincatto/solver2d · https://box2d.org/posts/2023/10/simulation-islands/ · https://box2d.org/posts/2024/08/determinism/
- Baumgarte: https://doi.org/10.1016/0045-7825(72)90018-7 · PBD: https://matthias-research.github.io/pages/publications/posBasedDyn.pdf · XPBD: https://mmacklin.com/xpbd.pdf · Small Steps: https://mmacklin.com/smallsteps.pdf · XPBD rigid bodies: https://matthias-research.github.io/pages/publications/PBDBodies.pdf
- Featherstone: https://link.springer.com/book/10.1007/978-1-4899-7560-7 · Baraff 1996: https://dl.acm.org/doi/10.1145/237170.237226 · Coumans MLCP GDC 2014: https://archive.org/stream/GDC2014Coumans/GDC2014-Coumans_djvu.txt · MuJoCo computation: https://mujoco.readthedocs.io/en/stable/computation/index.html · Anitescu-Potra: https://link.springer.com/article/10.1023/A:1008292328909 · Stewart: https://doi.org/10.1137/S0036144599360110 · Staggered Projections: https://www.cs.ubc.ca/labs/sensorimotor/projects/sp_sigasia08/ · SIGGRAPH contact course: https://siggraphcontact.github.io/
- VBD: https://graphics.cs.utah.edu/research/projects/vbd/vbd-siggraph2024.pdf · AVBD: https://graphics.cs.utah.edu/research/projects/avbd/ · Primal/Dual: https://mmacklin.com/primaldual.pdf · PDNSF: https://visualcomputing.ist.ac.at/publications/2024/PDNSF/ · ADMM-PD: https://www.cse.iitd.ac.in/~narain/admm-pd/ · IPC: https://ipc-sim.github.io/ · Rigid IPC: https://ipc-sim.github.io/rigid-ipc/ · ABD: https://github.com/Autodesk/affine-body-dynamics · StiffGIPC: https://arxiv.org/abs/2411.06224
- Vivace: https://mfratarcangeli.github.io/publication/sigasia2016/ · supernodal XPBD: http://profs.etsmtl.ca/sandrews/publication/xpbd_mig2022/ · Mass Splitting: https://dl.acm.org/doi/10.1145/2185520.2185601 · FleX: https://mmacklin.com/uppfrta_preprint.pdf · ECL-CC: https://userweb.cs.txstate.edu/~mb92/papers/hpdc18.pdf · Jolt GDC notes: https://jrouwe.nl/architectingjolt/ArchitectingJoltPhysics_Rouwe_Jorrit_Notes.pdf
- PhysX GPU: https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/GPURigidBodies.html · https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/RigidBodyDynamics.html · Bullet OpenCL notes: https://www.multithreadingandvfx.org/course_notes/GPU_rigidbody_using_OpenCL.pdf · MJWarp: https://mujoco.readthedocs.io/en/latest/mjwarp/ · Genesis: https://github.com/Genesis-Embodied-AI/genesis-world · Brax: https://arxiv.org/abs/2106.13281 · Isaac Gym: https://arxiv.org/abs/2108.10470
- Kamino: https://disneyresearch.github.io/kamino/ · https://arxiv.org/abs/2603.16536 · JGS2: https://dl.acm.org/doi/10.1145/3731183 · Zhou et al.: https://dl.acm.org/doi/10.1145/3728291 · Barrier-AL: https://arxiv.org/abs/2407.00046

---

## 4. Survey of existing GPU physics engine implementations

*(Repos, releases, and docs verified live as of 2026-08-22.)*

### 4.1 NVIDIA PhysX 5 GPU rigid bodies

[Repo](https://github.com/NVIDIA-Omniverse/PhysX) — BSD-3-Clause, active (PhysX 5.9.0, 2026-07). The **full CUDA GPU pipeline source is public**: `gpubroadphase`, `gpunarrowphase`, `gpusolver`, `gpuarticulation`, `gpusimulationcontroller`.

- **On GPU**: broadphase (incremental SAP, opt-in), PCM contact gen, body management, PGS/TGS solve + integration. **On CPU**: scene queries, CCD, triggers, callbacks, vehicles, cooking. CUDA-only — the gap a Metal engine fills.
- **CPU fallbacks**: hulls >64 verts/64 polys, meshes cooked without `buildGPUData`, contact modification, custom geometry. Only D6 joints natively GPU — every other joint type does constraint prep on CPU each frame.
- **Direct GPU API** (`eENABLE_DIRECT_GPU_API`): app reads/writes poses/velocities/forces directly in GPU buffers via per-object GPU indices and batched accessors. Trade-off: stale CPU mirrors, long broken-feature list. The model for "sim state lives on GPU, indices not pointers, batched accessors" — and Apple UMA dissolves its worst downside, so a Jolt-like API over shared `MTLBuffer`s can offer both views coherently. https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/docs/DirectGPUAPI.html
- **Memory**: fixed-capacity pools (`PxGpuDynamicsMemoryConfig`), overflow drops contacts with warnings.
- **Verdict**: the most valuable open production reference — study `PxgConstraintPartition` (constraint batching), the fallback architecture as a cautionary cost (define a GPU-complete shape/joint envelope up front), and fixed-capacity pool discipline.

### 4.2 Bullet 3 OpenCL

[Repo](https://github.com/bulletphysics/bullet3) (zlib), `src/Bullet3OpenCL` frozen since ~2014-15. Design doc: Coumans, ["GPU rigid body simulation using OpenCL"](https://www.multithreadingandvfx.org/course_notes/GPU_rigidbody_using_OpenCL.pdf) — "runs 100% on the GPU", 110k bodies at 15–30 fps on a Radeon 7970 in 2013.

- **Modules**: BroadphaseCollision (SAP/grid/LBVH, sort-and-diff pair derivation), NarrowphaseCollision (SAT-with-compaction pipeline: `findSeparatingAxis` → stream compaction → `clipHullHull` → contact reduction; GJK/EPA; quantized stackless BVH midphase), RigidBody (`b3GpuRigidBodyPipeline` — read this first; two-level batched PGS; mass-splitting Jacobi as the "convergence was not as good" alternative), **ParallelPrimitives** (radix sort, prefix scan, bound search, `b3LauncherCL` with kernel I/O serialization for offline unit-testing of individual kernels — a genuinely great idea).
- **Documented lessons**: indices not pointers; engine-owned contiguous allocation (Bullet 2's user-allocated objects made GPU transfer impossible — "one of many reasons for a full rewrite"); a CPU reference implementation of every kernel; sort work by shape-type pair; determinism requires sorting atomically-appended arrays before solving.
- **Why it stalled**: never reached CPU-Bullet feature parity (joints, sleeping, callbacks, queries), OpenCL decayed, Coumans pivoted. Its failure modes are the requirements list — joints, sleeping/islands, determinism, and API feature parity from day one, or you get a demo, not an engine.
- **Verdict**: the closest permissively-licensed blueprint for "whole pipeline on GPU from scratch"; zlib means kernels can be ported outright. Borrow ParallelPrimitives, the sort-and-diff SAP, the SAT-compaction narrowphase, two-level batching, and the CPU-mirror + kernel-I/O-serialization test discipline.

### 4.3 Genesis

[genesis-world](https://github.com/Genesis-Embodied-AI/genesis-world) — Apache-2.0, 29.8k stars, weekly releases. From-scratch MuJoCo-inspired generalized-coordinate solver in Python kernels compiled by **Quadrants** (their Taichi fork, [repo](https://github.com/Genesis-Embodied-AI/quadrants)) to **CUDA, ROCm, Apple Metal, Vulkan, x86, ARM64**.

- **Solver**: MuJoCo-style soft constraints with selectable CG or Newton; contact islands are central (union-find build, per-island grouping/solves, hibernation); dense per-env CRB + Cholesky with a shared-memory GPU factorization, plus a sparse skyline-Cholesky path with fill-reducing DOF permutation. 2026 releases emphasize fp32 convergence and CPU/GPU determinism.
- **Collision**: hibernation-aware one-axis SAP; GJK/EPA (+differentiable GJK), MPR, support-field maps, specialized box-box, multi-contact patches.
- **Metal status — genuinely runs on Apple Silicon GPU**: `gs.metal` first-class, Apple Metal CI, continuous Metal-specific fixes (native float atomics PR #3190, autodiff on Metal #3119, zero-copy race #2758) — real support, youngest backend. The Metal PR trail is a ready-made catalog of Metal pitfalls.
- **Batch-RL vs single-world**: batch axis is the primary parallel axis; a single big world depends on the secondary axes (per-constraint, per-pair, per-island) — features Genesis added in 2026 but not its center of gravity.
- **Verdict**: the working proof that a MuJoCo-class rigid solver runs on Apple Silicon GPU via Metal today. Study the GPU Newton solver, islanding, batch layout, and the Quadrants Metal runtime.

### 4.4 NVIDIA Newton + MuJoCo Warp

[Newton](https://github.com/newton-physics/newton) — Apache-2.0, Linux Foundation (Disney Research + Google DeepMind + NVIDIA), v1.5.0 (2026-08). All compute in NVIDIA Warp; pluggable solvers behind one interface: SolverMuJoCo (wraps [mujoco_warp](https://github.com/google-deepmind/mujoco_warp) — GPU tile ops, blocked Cholesky on compacted per-world spaces), SolverXPBD (maximal-coordinate rigid), SolverVBD, SolverFeatherstone, SolverKamino. Clean `ModelBuilder → Model`, per-step `State`/`Control`/`Contacts` separation — the right API bones. Collision: NxN/SAP/precomputed-pairs broadphase, GJK+MPR, BVH or SDF meshes, solver-agnostic contact descriptors. **macOS = CPU only** (Warp has no Metal backend, none planned). Port concepts, not code.

### 4.5 Brax

[Repo](https://github.com/google/brax) — JAX, four pipelines (MJX/Generalized/Positional/Spring), physics effectively wound down (only `brax/training` maintained). **jax-metal is practically dead** (last release 2024, broken against modern JAX). Lasting lessons: the fixed-shape, allocation-free, branch-light formulation of an entire physics step (exactly what a Metal command-buffer pipeline wants); do not build on JAX for Apple GPU.

### 4.6 Dimforge Nexus — "Rapier on the GPU"

[Repo](https://github.com/dimforge/nexus) — Rust, MIT/Apache-2.0, created 2025-09, active. Author: Sébastien Crozet (Rapier). Rigid-body module released Q2 2026 ([technical report](https://dimforge.com/blog/2026/07/01/dimforge-Q2-technical-report/)).

- **Whole pipeline on GPU** — broadphase, narrowphase, constraint solve, integration — as compute shaders written in Rust via rust-gpu → SPIR-V → wgpu (**Metal on macOS**). They switched from WGSL to rust-gpu (~2× faster than their WGSL version) for code reuse and CPU debugging of GPU code.
- **Solver** (from source, `src_rbd/dynamics/solver.rs`): *"Constraint-based physics solver running entirely on the GPU, using graph coloring to solve constraints in parallel without data races. Uses the Soft-TGS algorithm (as in Rapier)."* Per substep: warmstart → biased GS sweeps → linearized position integration → RHS refresh → unbiased stabilization sweeps.
- **Maturity**: "still under heavy development... missing many features."
- **Verdict**: the single most relevant open project — it *is* the target architecture (full GPU pipeline, Rapier/Jolt-style semantics, soft-TGS + coloring + warmstarting, on Metal via wgpu). Study `solver.rs` and `coloring.rs` for color-bucketed dispatch structure and biased/unbiased sweep ordering. Permissively licensed.

### 4.7 LuckyIYI/avbd-metal — Swift + Metal AVBD

[Repo](https://github.com/LuckyIYI/avbd-metal) — created 2026-06, very active, **no license (study only)**. Unified AVBD solver for 3D rigid bodies + tet soft bodies + cloth, doubling as an MJCF/RL robotics platform.

- **GPU pipeline (all Metal)**: contact-aware dynamic graph coloring for rigid stacks (precomputed palettes for soft scenes); **lane-split SIMD primal — 8 threads per body with `simd_shuffle_xor` reductions** (one simdgroup slice factorizes/solves each body's 6×6 LDLᵀ block — a genuinely Metal-native trick); counting-sort spatial hashing with deterministic indexing; CAS open-addressing persistence maps for warm-started persistent contacts; runtime shader concatenation.
- **Verdict**: the most directly instructive codebase — same platform, modern solver, disciplined. Divergence from a Jolt-like engine: position-level AVBD rather than velocity-impulse TGS. Related: [MetalAVBD](https://github.com/tatsuya-ogawa/MetalAVBD) (MIT, Swift + Metal 4, early-stage), [avbd-demo2d](https://github.com/savant117/avbd-demo2d) (MIT reference), and a [PhysX discussion on adopting AVBD](https://github.com/NVIDIA-Omniverse/PhysX/discussions/423).

### 4.8 LuisaComputeSimulator — ABD/IPC on LuisaCompute

[Repo](https://github.com/ChengzhuUwU/LuisaComputeSimulator) — Apache-2.0 (LICENSE file governs over README's MIT claim). Implicit Newton + PCG, 12-DoF ABD bodies, IPC barrier contacts, CCD; C++ DSL JIT-compiled to CUDA/Metal/DX12/Vulkan/CPU via [LuisaCompute](https://github.com/LuisaGroup/LuisaCompute). Reported: 88K verts / 3M+ pairs → ~3 FPS on RTX 3090, **~2 FPS on M2 Max** — notable evidence M2 Max lands within ~1.5× of a 3090 on this workload, and the argument against adopting barrier/Newton solvers wholesale for realtime. Worth reading for ABD stiff-body treatment and CCD/barrier robustness.

### 4.9 Kamino

Disney Research + NVIDIA, arXiv 2026 ([paper](https://arxiv.org/abs/2603.16536), [project](https://disneyresearch.github.io/kamino/)); code public inside Newton (Apache-2.0, BETA). Proximal-ADMM in maximal coordinates, native closed kinematic loops, redundant constraints, extreme mass ratios; batched (4096 envs). Warp/CUDA — not portable code, but the state of the art on GPU constraint formulation for ill-conditioning, and the ADMM architecture to study for future modal-subsystem coupling.

### 4.10 Others worth a paragraph

- **NVIDIA Warp**: Python-to-kernel JIT, CPU + CUDA only, `warp.sim` removed in v1.10 (→ Newton). Its Tile API (tile Cholesky) is a design reference for any Metal codegen layer. https://github.com/NVIDIA/warp
- **Taichi**: maintenance mode (one release in the past year) — why Genesis forked it. Read Quadrants' Metal runtime instead. https://github.com/taichi-dev/taichi
- **Isaac Lab / Isaac Sim**: production = PhysX 5 GPU TGS; Newton integration experimental. Defines the RL crowd's perf bar and fixed-shape-tensor API expectations. https://isaac-sim.github.io/IsaacLab/main/source/experimental-features/newton-physics-integration/index.html
- **NVIDIA FleX**: unified-particle PBD; rigids as shape-matched particle clusters — mushy stacking, weak joints, dormant. A cautionary tale: unified-particle rigids are not a path to a Jolt-like engine. https://github.com/NVIDIAGameWorks/FleX
- **Havok**: CPU-only rigid bodies in 2026; Havok FX (2005, segregated "effects physics" on shaders) was cancelled — the classic gameplay-vs-effects split failure pattern a gameplay-grade full-GPU engine transcends.
- **Unity Physics (DOTS)**: CPU-only, *stateless* (no persistent caches — everything rebuilt per frame for determinism/rollback). The statelessness experiment matters: per-frame rebuild is what GPUs prefer. No GPU rigid path roadmapped.
- **Unreal Chaos**: CPU-only rigid bodies as of UE 5.8; GPU compute serves Niagara/hair/cloth. Its async physics-thread/game-thread interpolation is the same producer/consumer decoupling a GPU engine needs against rendering. Together with Unity: both dominant engines are CPU-only for rigid bodies — market evidence the niche is open.
- **jure/webphysics**: "WebGPU physics engine based on the AVBD solver," TypeScript + WGSL, **MIT**, 417 stars. Full GPU pipeline: LBVH broadphase → warm-started manifolds → per-body constraint lists → greedy coloring → colored primal iterations → dual updates; ~85% of a native CUDA implementation's reported performance. WGSL maps 1:1 onto MSL — the best open MIT-licensed reference for a *complete* GPU AVBD pipeline including LBVH. https://github.com/jure/webphysics
- **ecto/phyz**: Rust differentiable multi-physics; honest data point that naive wgpu batching loses to CPU threading below ~128-world batches. https://github.com/ecto/phyz

### 4.11 Cross-cutting takeaways

1. **The niche is genuinely open** (see Synthesis).
2. **Two solver camps ship on GPU in 2026**: velocity-impulse soft-TGS with coloring/warm starts (PhysX GPU, Nexus) — the "Jolt semantics on GPU" path — and AVBD/VBD (avbd-metal, MetalAVBD, webphysics, Newton SolverVBD). MuJoCo-class Newton solvers are proven but robotics-shaped; barrier/IPC is robust but ~2 FPS-class on M2 Max.
3. **Every serious project converges on the same pipeline skeleton** (see §2.3), glued by radix sort + prefix scan — build those first.
4. **Data layout dogma is unanimous**: indices not pointers, engine-owned contiguous SoA, flattened shape blobs, fixed-capacity pools with overflow stats, determinism via sorting after every atomic-append stage.
5. **Metal-specific leverage seen in the wild**: simdgroup lane-split per-body block solves, CAS contact-persistence maps, deterministic counting-sort hashing, UMA dissolving the stale-mirror problem, Genesis's Metal PR trail as a hazards catalog.
6. **Debuggability discipline**: Bullet 3, Jolt, and Nexus independently converged on CPU-executable mirrors of every GPU kernel (plus Bullet's kernel I/O serialization for tests). Non-negotiable.
7. **Licensing map**: portable outright — Bullet3 OpenCL (zlib), Nexus (MIT/Apache), webphysics/MetalAVBD/avbd-demo2d (MIT), PhysX 5 (BSD-3), Genesis/Quadrants/Newton/MJWarp/LuisaComputeSimulator (Apache-2.0), Jolt (MIT). avbd-metal: no license, study only.

### Sources

- PhysX: https://github.com/NVIDIA-Omniverse/PhysX · https://nvidia-omniverse.github.io/PhysX/physx/5.8.0/docs/GPURigidBodies.html · https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/docs/DirectGPUAPI.html · AVBD discussion: https://github.com/NVIDIA-Omniverse/PhysX/discussions/423
- Bullet3: https://github.com/bulletphysics/bullet3 · https://github.com/bulletphysics/bullet3/tree/master/src/Bullet3OpenCL · https://www.multithreadingandvfx.org/course_notes/GPU_rigidbody_using_OpenCL.pdf
- Genesis: https://github.com/Genesis-Embodied-AI/genesis-world · https://github.com/Genesis-Embodied-AI/quadrants · https://genesis-world.readthedocs.io/en/latest/user_guide/overview/installation.html
- Newton: https://github.com/newton-physics/newton · https://newton-physics.github.io/newton/latest/ · MJWarp: https://github.com/google-deepmind/mujoco_warp · https://mujoco.readthedocs.io/en/latest/mjwarp/
- Brax: https://github.com/google/brax · jax-metal: https://developer.apple.com/metal/jax/ · https://github.com/jax-ml/jax/issues/34109
- Nexus: https://github.com/dimforge/nexus · https://nexus.dimforge.com/ · https://dimforge.com/blog/2026/07/01/dimforge-Q2-technical-report/ · https://github.com/dimforge/nexus/tree/main/src_rbd/dynamics
- avbd-metal: https://github.com/LuckyIYI/avbd-metal · AVBD: https://graphics.cs.utah.edu/research/projects/avbd/ · https://github.com/savant117/avbd-demo2d · https://github.com/tatsuya-ogawa/MetalAVBD
- LuisaComputeSimulator: https://github.com/ChengzhuUwU/LuisaComputeSimulator · https://github.com/LuisaGroup/LuisaCompute
- Kamino: https://arxiv.org/abs/2603.16536 · https://disneyresearch.github.io/kamino/ · https://github.com/newton-physics/newton/tree/main/newton/_src/solvers/kamino
- Jolt: https://github.com/jrouwe/JoltPhysics · https://github.com/jrouwe/JoltPhysics/discussions/501 · Godot 4.6: https://docs.godotengine.org/en/4.6/tutorials/physics/using_jolt_physics.html
- Warp: https://github.com/NVIDIA/warp · Taichi: https://github.com/taichi-dev/taichi · Isaac Lab: https://isaac-sim.github.io/IsaacLab/main/source/experimental-features/newton-physics-integration/index.html · FleX: https://github.com/NVIDIAGameWorks/FleX · Unity Physics: https://docs.unity3d.com/Packages/com.unity.physics@1.0/manual/ecs-packages.html · Chaos: https://www.unrealengine.com/en-US/tech-blog/chaos-scene-queries-and-rigid-body-engine-in-ue5
- webphysics: https://github.com/jure/webphysics · phyz: https://github.com/ecto/phyz · Ten Minute Physics: https://matthias-research.github.io/pages/tenMinutePhysics/ · PositionBasedDynamics: https://github.com/InteractiveComputerGraphics/PositionBasedDynamics

---

## 5. The Metal API (Metal 4) and Apple Silicon GPU architecture

### 5.1 Metal 4: the new command model

Metal 4 (WWDC 2025, [Discover Metal 4](https://developer.apple.com/videos/play/wwdc2025/205/)) is a ground-up redesign of command submission, exclusive to Apple silicon (M1+/A14+), shipping with macOS 26 / Xcode 26.

- **New object model**: `MTL4CommandQueue`/`MTL4CommandBuffer`/`MTL4CommandAllocator`. Command buffers created from the *device*, long-lived and reusable, drawing encoding memory from app-owned allocators (one per encoding thread / frame-in-flight) — fully parallel encoding, zero dynamic allocation, batch submission.
- **Unified compute encoder**: `MTL4ComputeCommandEncoder` merges compute + blit + acceleration-structure encoders. **Commands within an encoder run concurrently by default** — no implicit serialization between successive dispatches.
- **Explicit barriers replace hazard tracking**: all resources untracked; *pass barriers* within an encoder (`barrierAfterEncoderStages:beforeEncoderStages:visibilityOptions:` — e.g. Dispatch→Dispatch) and *queue barriers* across encoders, with minimal stage pairs to avoid over-sync ([Explore Metal 4 games](https://developer.apple.com/videos/play/wwdc2025/254/)).
- **Physics substepping fit**: a substep chain (integrate → broadphase → narrowphase → N solver colors × M substeps) encodes into *one* command buffer with cheap dispatch→dispatch barriers only on true dependencies, independent work concurrent. Caveats: `MTL4CommandBuffer` does **not** retain resources (lifetime is on the app), and an over-broad barrier after every dispatch reproduces serial behavior. No published microbenchmark of barrier cost vs Metal 3 fences — measure on target hardware.
- **Binding**: `MTL4ArgumentTable` holds raw GPU addresses, shareable across encoders, mutable between dispatches. Natural pattern: fully bindless — one table slot pointing at a world argument buffer. `MTLResidencySet` (attach to queue, populate once) is the only residency mechanism; heap suballocation keeps residency cardinality low.
- **MTLTensor / Shader ML**: first-class tensor resource; Metal Performance Primitives embed tensor ops inside custom kernels (registers/threadgroup memory, no round-trips). Cooperative tensors distribute storage across SIMD-group registers. Relevant to dense modal matrices — see §5.4.
- **Coexistence**: Metal 4 sits alongside Metal ≤3 (same `MTLDevice`, resources, heaps); `MTLSharedEvent` synchronizes across `MTLCommandQueue` and `MTL4CommandQueue` — a physics library can ship a Metal 4 fast path over a shared resource layer.

### 5.2 Compute fundamentals

- **Execution**: 32-wide SIMD-groups; full SIMD-group toolkit in MSL (`simd_shuffle*`, `simd_ballot`, `simd_sum/min/max`, `simd_prefix_exclusive_sum`). Measured shuffle throughput is exceptional (256 B/cycle/core, full reductions ~14.5 cycles) — SIMD-group-cooperative narrowphase and reductions are cheap ([metal-benchmarks](https://github.com/philipturner/metal-benchmarks)).
- **Threadgroups**: max 1024 threads, **32 KB threadgroup memory**; ~208 KB register file per core vs only 8 KB L1D — the tuning advice *inverts* the CUDA habit: keep working sets in registers, exchange via shuffles, treat threadgroup memory as an exchange tier, not a data tier.
- **Atomics**: 32-bit int atomics (device + threadgroup), **`memory_order_relaxed` only** — ordering via `threadgroup_barrier(mem_flags)`, not acquire/release. **No float atomics in MSL** (emulate via CAS, or design around with gather) — this matters for impulse scatter: prefer gather-based or colored solvers. 64-bit: ulong min/max from Apple8 (M2), fuller set from Apple9 (M3) — the 64-bit min/max "pack (key, payload)" trick works for closest-contact reductions.
- **Occupancy/divergence**: 4 schedulers/core over 128 ALUs, saturation ~24 SIMD-groups/core. From M3 (Apple9), **Dynamic Caching** allocates registers/threadgroup/tile memory from one on-chip pool on demand — worst-case register usage no longer caps occupancy, making divergent "rare expensive branch" kernels (EPA fallbacks) far less punishing ([Tech Talk 111375](https://developer.apple.com/videos/play/tech-talks/111375/)).
- **Function pointers**: visible function tables work in compute but block cross-call optimization — for a small shape-type matrix, a switch usually beats function tables.
- **GPU-driven execution**: indirect dispatch (`dispatchThreadgroups(indirectBuffer:)`) is the backbone of variable-cardinality stages. ICBs let kernels encode dispatches on GPU. **No device-side enqueue** (no CUDA dynamic parallelism) and no device-wide barrier within a dispatch — iterative solvers choose between CPU-encoded fixed iteration counts (the standard choice, cheap in Metal 4), indirect dispatch with convergence-written counts, or threadgroup-scoped mega-kernels. ICB semantics under the MTL4 encoders were not fully confirmable — verify when implementing.

### 5.3 CPU-GPU synchronization, unified memory, latency

- **UMA**: `MTLStorageModeShared` (default) is genuine zero-copy, cache-coherent at the fabric level. "Zero-copy" still requires *temporal* correctness — per-frame/per-substep ring buffers, never mutate what the GPU may read. `MTLStorageModePrivate` still worthwhile for GPU-only data.
- **Events**: `MTLSharedEvent.waitUntilSignaledValue` has markedly lower latency than `waitUntilCompleted`.
- **Latency numbers**: best public data point is the [Anukari GPU-audio devlog](https://anukari.com/blog/devlog/huge-macos-performance-improvements): **<50 µs total scheduling+wait per round trip** with shared-event-triggered launches + double-buffered encoding. Uncommitted-path command-buffer latency is commonly hundreds of µs. Dependent dispatch→dispatch cost inside a committed buffer is order single-digit µs but unpublished — **measure on M5 Max**, it bounds max substep rate. Same devlog documents the trap for bursty low-duty-cycle compute (exactly the physics-tick profile): macOS downclocks the GPU between short workloads, causing latency spikes.
- **Practical architecture**: encode the whole frame's substep chain in one command buffer, one shared-event signal at frame end, keep the GPU fed to avoid clock-ramp latency.

### 5.4 Apple GPU microarchitecture, M1→M5

| Generation | Compute-relevant additions |
|---|---|
| M1 (Apple7, G13) | Baseline AGX: 32-wide SIMD, 128 ALU/core, TBDR ([dougallj ISA docs](https://dougallj.github.io/applegpu/docs.html), [Asahi AGX docs](https://asahilinux.org/docs/hw/soc/agx/)) |
| M2 (Apple8) | bfloat in MSL, ulong min/max atomics |
| M3 / A17 Pro (Apple9) | **Dynamic Caching**, hardware ray tracing, mesh shading, full 64-bit atomics |
| M4 | ~2× RT vs M3; M4 Max: 40-core GPU, 546 GB/s, up to 128 GB |
| M5 (2025) / M5 Pro/Max (2026) | Per-core **Neural Accelerators**: FP16/INT8 matmul via Metal 4 tensor APIs / MPP only (not raw MSL), ≈1024 FP16 FMA/cycle/core (~4× prior), best tiles ≥32×32; FP32 matmul stays on general SIMD pipes. M5 Max: 32–40-core GPU, 460–614 GB/s, up to 128 GB ([Apple M5](https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/), [M5 Pro/Max](https://www.apple.com/newsroom/2026/03/apple-debuts-m5-pro-and-m5-max-to-supercharge-the-most-demanding-pro-workflows/), [A19/M5 NA benchmark](https://tzakharko.github.io/apple-neural-accelerators-benchmark/)) |

Additional facts ([metal-benchmarks](https://github.com/philipturner/metal-benchmarks)): FP16 issues at 1 cycle vs FP32 at 2 (throughput, not just bandwidth); large SLC (8–48 MB) amplifies bandwidth for scattered access; `simdgroup_matrix` on M1–M4 lowers to regular pipes (~2× FP32 FFMA via register-pressure relief, no dedicated hardware). For dense modal work: the M5 FP16 tensor path is the fast path but FP16 range limits it to well-scaled modal bases; FP32 dense algebra runs at ordinary ALU rate.

**Ray tracing hardware (M3+)**: AS builds encode in the unified Metal 4 compute encoder; `intersector<>` callable from compute kernels. Academic work casts collision queries as ray queries ([arXiv 2409.09918](https://arxiv.org/pdf/2409.09918)) — credible for CCD casts and point-vs-mesh queries; AS refit cost for moving bodies vs a custom BVH is unbenchmarked on Apple hardware — prototype both.

### 5.5 Precision and determinism

- **No fp64** on Apple GPUs ([metal-float64](https://github.com/philipturner/metal-float64) emulates at severe cost). Compensated arithmetic if a stage ever needs it.
- **Fast math is the default** (`-ffast-math`, `-ffp-contract=fast`); modern control is `MTLMathMode` (safe/relaxed/fast) per compile, `metal::precise::` per call.
- **Determinism**: run-to-run on one machine — the realistic goal — requires fixed kernel binaries, fixed dispatch/reduction orders (no atomic-order float accumulation), fixed thread-to-data mapping. Fast-math is deterministic per-binary. Cross-device bit-exactness is not promised anywhere in the MSL spec. Denormal handling is implementation-defined territory — clamp explicitly rather than relying on denormals for tiny impulse accumulation.
- **fp16**: attractive (issue rate, bandwidth, registers, M5 NA) but ~3 decimal digits — viable for normals, relative contact offsets, modal excitation; not positions or accumulated velocities.

### 5.6 Tooling

- **Profilers**: Xcode Metal debugger (Metal 4 adds barrier visualization), Instruments Metal System Trace, GPU performance counters/limiters, `MTLCounterSampleBuffer` for in-stream GPU timestamps around dispatches. WWDC26 added Metal command-line tools for scripted profiling. https://developer.apple.com/metal/tools/
- **metal-cpp**: official C++ bindings, updated for Metal 4; watch manual refcounting (`NS::TransferPtr`) — doubly important since MTL4 command buffers don't retain resources. https://developer.apple.com/metal/cpp/
- **Shared struct layouts**: one header for C++ and MSL via `<simd/simd.h>` types guarded by `#ifdef __METAL_VERSION__`; `float3` is 16-byte sized/aligned — use `packed_float3` for tight arrays or design SoA buffers of scalars/float4 to sidestep the issue.

### Sources

**Apple**: Discover Metal 4: https://developer.apple.com/videos/play/wwdc2025/205/ · Explore Metal 4 games: https://developer.apple.com/videos/play/wwdc2025/254/ · ML + graphics: https://developer.apple.com/videos/play/wwdc2025/262/ · Metal tensors (WWDC26): https://developer.apple.com/videos/play/wwdc2026/330/ · Perf tools (WWDC26): https://developer.apple.com/videos/play/wwdc2026/388/ · WWDC26 Metal guide: https://developer.apple.com/wwdc26/guides/metal/ · M3/A17 GPU tech talk: https://developer.apple.com/videos/play/tech-talks/111375/ · Shader best practices: https://developer.apple.com/videos/play/tech-talks/111373/ · Function pointers: https://developer.apple.com/videos/play/wwdc2020/10013/ · MSL spec: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf · Feature Set Tables: https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf · MTL4ArgumentTable: https://developer.apple.com/documentation/metal/mtl4argumenttable · MTL4ComputeCommandEncoder: https://developer.apple.com/documentation/metal/mtl4computecommandencoder · Storage modes: https://developer.apple.com/documentation/metal/mtlstoragemode · MTLSharedEvent: https://developer.apple.com/documentation/metal/mtlsharedevent · CPU-GPU sync: https://developer.apple.com/documentation/Metal/synchronizing-cpu-and-gpu-work · Indirect command encoding: https://developer.apple.com/documentation/metal/indirect-command-encoding · MTLMathMode: https://developer.apple.com/documentation/metal/mtlmathmode/relaxed · Tools: https://developer.apple.com/metal/tools/ · metal-cpp: https://developer.apple.com/metal/cpp/ · M5: https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/ · M5 Pro/Max: https://www.apple.com/newsroom/2026/03/apple-debuts-m5-pro-and-m5-max-to-supercharge-the-most-demanding-pro-workflows/ · MBP M5 specs: https://support.apple.com/en-us/126319
**Third-party**: metal-benchmarks: https://github.com/philipturner/metal-benchmarks · metal-experiment-1: https://github.com/philipturner/metal-experiment-1 · metal-float64: https://github.com/philipturner/metal-float64 · dougallj G13 ISA: https://dougallj.github.io/applegpu/docs.html · Asahi AGX: https://asahilinux.org/docs/hw/soc/agx/ · A19/M5 Neural Accelerators: https://tzakharko.github.io/apple-neural-accelerators-benchmark/ · Rigel (M4 Max tensor path): https://arxiv.org/pdf/2606.12765 · Metal by Example, Metal 4: https://metalbyexample.com/metal-4/ · Metal 4 Essentials: https://blakecrosley.com/blog/metal-4-essentials · Anukari devlog: https://anukari.com/blog/devlog/huge-macos-performance-improvements · Apple GPU GEMM: https://percisely.xyz/gemm · Dynamic Caching analysis: https://forum.beyond3d.com/threads/apple-dynamic-caching-on-m3-gpu.63419/ · RT-hardware collision detection: https://arxiv.org/pdf/2409.09918 · MSL/C++ types: https://whackylabs.com/metal/cpp/2019/01/19/metal-types/
**Forums/issues** (specific points): float atomics: https://developer.apple.com/forums/thread/69703 · threadExecutionWidth: https://developer.apple.com/forums/thread/709208 · fast-math: https://developer.apple.com/forums/thread/131733 · struct alignment: https://developer.apple.com/forums/thread/98020 · storage on Apple silicon: https://developer.apple.com/forums/thread/710878 · compute overhead: https://developer.apple.com/forums/thread/767853 · gpuweb atomics: https://github.com/gpuweb/gpuweb/issues/1360 · no device enqueue: https://github.com/gpuweb/gpuweb/issues/31 · fast-math divergence: https://github.com/gpuweb/gpuweb/issues/2076

**Flagged uncertainties**: exact per-dispatch/barrier overhead on Metal 4; ICB semantics under MTL4 encoders; `simdgroup_async_copy` status on Apple9+; write-combined CPU mapping behavior; bf16 Neural Accelerator support; per-generation denormal handling.

---

## 6. Modal contact sound and physically-based rigid-body sound synthesis

### 6.1 Foundations

- **van den Doel & Pai (ICAD 1996, Presence 1998).** The analytic modal model for point-contact sound: damped sinusoid banks with location-dependent gains, implemented as two-pole IIR resonators. *GPU:* one thread/lane per mode, gain lookup ≈ texture fetch. https://www.cs.ubc.ca/~kvdoel/publications/
- **van den Doel, Kry & Pai — FoleyAutomatic (SIGGRAPH 2001).** The canonical real-time pipeline: modal banks excited by contact-force models for impact, rolling, sliding — roughness "contact texture" played at sliding speed, filtered by patch size. Demonstrated that audio-rate excitation detail is what makes rolling/scraping plausible. Its slow-dynamics/fast-excitation split is exactly the two-rate structure a GPU engine needs. https://www.cs.ubc.ca/~kvdoel/publications/foleyautomatic.pdf
- **O'Brien, Cook & Essl (SIGGRAPH 2001).** Audio-rate nonlinear FEM + Rayleigh-integral radiation — fully general, intractably expensive for stiff materials. The motivating negative result for modal reduction. http://graphics.berkeley.edu/papers/Obrien-SSF-2001-08/Obrien-SSF-2001-08.pdf
- **O'Brien, Shen & Gatchalian (SCA 2002).** Made numerical modal analysis practical: tet FEM → generalized eigenproblem → mass-normalized modal oscillators driven by rigid-solver contact forces. The direct template for Zheng-James's sound model — whose critique is that rigid-solver impulses are too dirty and too decoupled. http://graphics.berkeley.edu/papers/Obrien-SSR-2002-07/Obrien-SSR-2002-07.pdf
- **Bonneel et al. (SIGGRAPH 2008).** Frequency-domain synthesis over ~46 ms frames with perceptual mode culling — the answer when the synthesis pass itself becomes the bottleneck. https://www-sop.inria.fr/reves/Basilic/2008/BDTVJ08/FastModalSounds.pdf
- **Ren, Yeh & Lin (TOG 2013).** Fits modal materials — including *frequency-dependent* damping beyond two-parameter Rayleigh — from a single recorded impact. The "sound like this recording" recipe. https://dl.acm.org/doi/10.1145/2421636.2421637

### 6.2 Acoustic transfer

- **James, Barbič & Pai — PAT (SIGGRAPH 2006).** Per-mode precomputed Helmholtz radiation compressed as multipole equivalent sources — radiation (why mechanical amplitude ≠ loudness) as an output-sensitive per-mode lookup. Zheng-James 2011 renders with exactly this. http://graphics.cs.cmu.edu/projects/pat/
- **Chadwick, An & James — Harmonic Shells (SIGGRAPH Asia 2009).** Reduced nonlinear thin-shell forces (cymbals, sheet metal) + **FFAT maps** — per-mode |pressure|×distance fields on a surrounding grid: literally textures, per-mode cube-map lookups at the listener direction. The transfer representation KleinPAT/NeuralSound/WaveBlender standardize on. https://www.cs.cornell.edu/projects/HarmonicShells/
- **Langlois, An, Jin & James — Eigenmode Compression (SIGGRAPH 2014).** The dense eigenmode matrix `U` (tens–thousands of MB per object) compressed ~100× via nonlinear-optimized MLS with per-mode perceptual error tolerances. **Directly load-bearing for this project's memory budget**: modal Jacobian columns are rows of `U` at contact points — MLS-compressed modes mean evaluating `U(x)` procedurally at hit points, trading ALU for bandwidth (very shader-friendly). https://www.cs.cornell.edu/projects/Sound/modec/
- **Wang & James — KleinPAT (SIGGRAPH 2019).** Kills per-mode Helmholtz solves: optimal mode conflation into "chords", one time-domain GPU FDTD vector wavesolve per chord, deconflation via specialized QR — ~4000× faster transfer precompute (minutes per object). The modern offline transfer stage to pair with the Metal runtime. https://graphics.stanford.edu/projects/kleinpat/

### 6.3 Zheng & James 2011 — "Toward High-Quality Modal Contact Sound" (detailed)

https://www.cs.cornell.edu/projects/Sound/mc/ · PDF: https://www.cs.cornell.edu/projects/Sound/mc/ModalContactSound2011.pdf

**Thesis**: rigid contact solving + post-hoc modal excitation is fundamentally deficient. Rigid solvers produce (a) non-unique, point-like, temporally incoherent impulses (static indeterminacy + iterative-solver noise → humming/buzzing/crackle), and (b) no vibrational feedback into contact — missing micro-collisions, chattering, vibrational energy exchange (bunny-on-table making dishes clatter), stick-slip, and contact-dependent damping (a mug rings differently rim-down vs handle-touching).

**Generalized coordinates and contact Jacobian.** Per body: `q` = 6 rigid DOFs + modal amplitudes; block mass matrix = rigid inertia ⊕ identity (mass-normalized modes). World contact-point position composes rigid motion + linear modal displacement `u = U q_modal`. **The contact Jacobian gains modal columns** (`U_kᵀ n_k` alongside the rigid `[I, r×]` block): impulses simultaneously change rigid momentum and excite/damp modes, and modal velocity at the contact feeds separation/sliding speed. Friction via Stewart-Trinkle polyhedral cones. Velocity-level Euler-Lagrange stepping.

**Modified Staggered Projections for low-noise impulses.** Kaufman 2008's alternating contact-QP / friction-QP pair, warm-started. Two noise pathologies: multipliers of the contact QP are non-unique when `N` is rank-deficient (nearly always with many contacts) — impulses fluctuate wildly while motion stays correct; and the friction Hessian goes singular as contact count grows. Fixes: **contact generation on interpenetration regardless of approach velocity** (velocity-triggered contact sets cycle audibly at rest), **contact filtering/reduction** via warm-started active sets (~10× solver speedup), and friction-solve culling for resting groups.

**Impulse redistribution (sound pass only).** The solver's minimal active-set impulses are spatially concentrated and sound noisy. Exploiting the same non-uniqueness, impulses are redistributed within the null spaces of `N` and `D` — a small per-object QP finds the distribution closest to the previous step's, subject to nonnegativity and the cone bound. Total impulse and simulated motion are *exactly unchanged*, but excitation becomes spatially smoother and temporally coherent (2–10% overhead, visibly cleaner spectrograms).

**Asynchronous contact groups with adaptive modal bandwidth.** Stability ties the timestep to the highest simulated modal frequency, so: simulate only currently-audible low modes, asynchronously per contact group (connected components of the contact graph). Mirtich-timewarp-style private timestamps and rollback via state interpolation. Collision: OBB trees for rigid, **BD-Tree** (James & Pai 2004) refits under modal deformation. Per group: **`Δt = min(1/(6.5 f_h), Δt_max)`**. Mode adaptation: estimated impulse changes projected through `Uᵀ` activate modes above an excitement threshold; modes below an energy threshold deactivate, down to fully rigid.

**Frequency budget.** Simulated (coupled) bandwidth capped at **f_h = 5 kHz** → min timestep ≈ 3.07×10⁻⁵ s (~32.5k steps/s). Two-pass pipeline: pass 1 records filtered/redistributed impulses; pass 2 integrates each object's full modal bank to **20 kHz** open-loop from those impulses. Tiny bodies whose lowest modes exceed 20 kHz fall back to recorded clicks. Radiation: PAT multipoles + HRTF.

**Contact-dependent modal damping.** Each sound-pass contact contributes a viscous damper proportional to estimated contact force, giving a time-varying diagonal modal damping term `γ G_mm(t)` on top of Rayleigh `αM + βK` — cubically interpolated into time-varying IIR resonator coefficients. Reproduces the mug-orientation ring and the tuning fork dying on the ground. Material (α, β, γ): ceramic (6, 1e-7, 3e-2), polystyrene (30, 8e-7, 4e-4), steel (5, 3e-8, 3e-1), MDF (35, 5e-6, 9e-3), wood (60, 2e-6, 5e-4).

**Performance (8-core Xeon X5570).** Dynamics pass hours per simulated second — roughly **10³–10⁴× slower than realtime** (table+dishes: 4.8 h / 2.2 s ≈ 7900×; marble tracks ≈ 990×), dominated by serial sparse QP solves; sound pass ~10× realtime. Mode adaptivity gave 2.6–6.8×, contact filtering ~10×. Stated limitations: fragile QP solvers, redistribution not dissipation-optimal, hand-tuned thresholds, no squeak theory, triangle-faceting clicks on rolling, single-body transfer only.

*GPU notes:* per-contact-group solves are independent (group-per-workgroup parallelism); modal Jacobian columns are dense small blocks (matrix-free or compressed `U_k` evaluation); the QP pair can be replaced by GPU-friendly proximal/projected solvers **if** their impulse noise is controlled — the paper's central warning is that iterative-solver impulse chatter is *the* audio artifact, so warm-starting, contact filtering, generation-on-intersection, and impulse smoothing are not niceties but the product. The priority queue is the least GPU-friendly component; a bucketed synchronous multirate ladder (power-of-two timesteps per group) is the natural Metal adaptation.

### 6.4 The follow-up line

- **Zheng & James — Rigid-Body Fracture Sound (SIGGRAPH 2010).** Time-varying modal models from precomputed **ellipsoidal sound proxies** (soundbank keyed on shape/material) instead of eigensolving every shard — the standard answer to runtime-generated geometry. https://www.cs.cornell.edu/projects/FractureSound/
- **Chadwick, Zheng & James — Precomputed Acceleration Noise (SIGGRAPH 2012).** Modal ringing isn't the whole impact: small/stiff objects emit the "thump" of abrupt rigid acceleration. Converts solver impulses into continuous force profiles via **Hertz contact theory** (half-sine pulse, duration τ ~ 10⁻⁵–10⁻⁴ s), then plays precomputed pulse responses per rigid DOF. The Hertz timescale estimate is also exactly what a GPU solver needs to spread constraint impulses into audio-band profiles. SCA 2012 companion adds ellipsoid soundbanks. https://dl.acm.org/doi/10.1145/2185520.2185599
- **Langlois & James — Inverse-Foley Animation (SIGGRAPH 2014)** — contact events as a first-class queryable representation. https://www.cs.cornell.edu/projects/Sound/ifa/
- **Fluids/shells branches** (Harmonic Fluids 2009, Fire 2011, Complex Acoustic Bubbles 2016, Coupled Bubbles 2023 [open source: FluidSound], Crumpling 2016, Wave Turbulence shells 2018, Metallophone inverse design 2015) — lineage completeness; the low/high-frequency split strategies recur. https://www.cs.cornell.edu/projects/HarmonicFluids/ · https://github.com/kangruix/FluidSound

### 6.5 Wave-based "golden renderers"

- **Wang, Qu, Langlois & James (SIGGRAPH 2018).** First unified offline wavesolver for animation sound: sharp-interface FDTD with moving/vibrating embedded boundaries — captures near-field scattering and multi-object diffraction that per-object transfer misses (the mug-cavity spoon case Zheng-James flagged as unsolved). CPU-cluster expensive. https://www.antequ.net/assets/projects/wavesolver/wavesolver2018optimized.pdf
- **Xue, Wang, Langlois & James — WaveBlender (SIGGRAPH Asia 2024).** Deliberately simple GPU FDTD on uniform grids: per-timestep voxelized occupancy with **β-blending** between successive discretizations and approximate velocity boundary conditions — minimal modification of vanilla FDTD, robust to moving/deforming interfaces, ~1000× faster than the 2018 solver. **Modal sources enter as boundary velocities** sampled from modal amplitudes — the wavesolver renders whatever the contact solver did. Known limitations: first-order boundaries, staircasing, one-way coupling, prescribed sources. **Open source (CUDA)** — a tractable Metal stencil port, and this project's offline golden renderer: record rigid+modal trajectories, render reference audio, validate the fast FFAT runtime path against it. https://graphics.stanford.edu/papers/waveblender/ · https://github.com/kangruix/WaveBlender

### 6.6 Neural and differentiable work, 2020–2026

- **Deep-Modal (ACMMM 2020)** — network maps voxelized shape + contact to mode data, for runtime-created objects. https://hellojxt.github.io/DeepModal/
- **NeuralSound (SIGGRAPH 2022)** — learned eigen-subspace *refined by LOBPCG against the true FEM matrices* (accuracy grounded, not purely amortized) + learned per-mode FFAT maps; full modal+transfer for a novel object in <1 s. The LOBPCG-corrected-network pattern is directly reusable for near-instant on-device eigensolves. https://hellojxt.github.io/NeuralSound/
- **DiffSound (SIGGRAPH 2024)** — end-to-end differentiable modal pipeline (implicit shape → differentiable high-order FEM → differentiable synthesizer): material/geometry/impact-position inference from audio. Also the best citation that high-order FEM audibly matters (linear tets run sharp). https://hellojxt.github.io/DiffSound/
- **NeuroSonic (CAVW 2026)** — modal + transfer for a new object in <0.02 s. https://onlinelibrary.wiley.com/doi/10.1002/cav.70162
- **Datasets**: RealImpact (150k measured impacts, the ground truth for validating modal+transfer pipelines), ObjectFolder. https://arxiv.org/abs/2306.09944 · https://arxiv.org/abs/2109.07991

**Assessment**: the neural line accelerates per-object precompute and inverse problems; none of it addresses vibration-aware *contact solving*. Through the available 2025–26 literature, no published system has made the coupled modal-contact dynamics pass realtime. The field routed around it. **A GPU-resident coupled solver is genuinely open territory.**

### 6.7 Supporting numerics

- **Modal analysis pipeline** (O'Brien 2002; SIGGRAPH 2016 course + [ModalSound code](https://github.com/cxzheng/ModalSound)): tetrahedralize → assemble `K`, lumped `M` → generalized eigenproblem `Kφ = ω²Mφ` up to ~20 kHz (shift-invert Lanczos, or LOBPCG — GPU-friendly) → mass-normalize (`ΦᵀMΦ = I`) → discard rigid + inaudible modes. Rayleigh damping `ξ_i = (α/ω_i + βω_i)/2`; measured materials deviate → per-mode fits (Ren 2013) or Zheng-James's contact-dependent term.
- **Exact integration of damped modal oscillators.** Each mode's free response integrates *exactly* at audio rate via the two-term recurrence `q[n+1] = 2e^{-ξωh} cos(ω_d h) q[n] − e^{-2ξωh} q[n−1] + b f[n]` — unconditionally stable at any sample rate regardless of ω. This is why the *synthesis* pass never has a stiffness-limited timestep; only the *contact-coupled* pass does. Time-varying damping handled by per-block coefficient re-derivation. *GPU:* thousands of independent 2-tap recurrences, one thread per mode, coefficients in registers.
- **Kaufman et al. 2008 — Staggered Projections** (see §3.3). The sound-quality constraint on any GPU replacement: whatever replaces the QPs must produce *temporally coherent, spatially distributed* multipliers, not merely correct velocities — keep the 2011 filtering/redistribution machinery (or a dissipation-optimal variant) around any iterative solver. https://www.cs.ubc.ca/labs/sensorimotor/projects/sp_sigasia08/
- **Multirate / asynchronous integration.** Lineage: Mirtich Timewarp (SIGGRAPH 2000, per-object virtual time + rollback), Harmon et al. Asynchronous Contact Mechanics (SIGGRAPH 2009, provably safe, slow), Kim & James online model reduction (2009). Zheng-James's group-level asynchrony is the pragmatic midpoint. *GPU reformulation:* hierarchical synchronous multirate — quantize group timesteps to a power-of-two ladder under `Δt ≤ 1/(6.5 f_h)`, advance all groups per ladder level in parallel, restrict rollback to group-merge events; record impulse events with sub-step timestamps for the audio pass. https://dl.acm.org/doi/10.1145/344779.344866 · https://dl.acm.org/doi/10.1145/1531326.1531393
- **Hertz timescales and micro-collisions.** Sphere contact duration `τ ≈ 2.87 (m²/(R E*² v))^{1/5}` — tens of µs: below physics rates, squarely audible, so instantaneous impulses must be spread into finite-duration profiles before exciting modal banks (else all-frequency clicks). Micro-collisions and chattering are *emergent* — they only appear when modal displacement/velocity participates in contact kinematics, and cannot be layered on afterward. They are the concrete audible payoff justifying modal DOFs in the solver.
- **James & Pai — BD-Tree (SIGGRAPH 2004).** Output-sensitive bound updates for reduced-coordinate deformation: refit sphere/box hierarchies from modal amplitudes alone. *GPU:* per-node refit from r amplitudes is a tiny kernel over the BVH — the reason modal deformation need not force full-mesh rebuilds. https://dl.acm.org/doi/10.1145/1015706.1015772

### Sources

- Zheng & James 2011: https://www.cs.cornell.edu/projects/Sound/mc/ · PDF: https://www.cs.cornell.edu/projects/Sound/mc/ModalContactSound2011.pdf
- FoleyAutomatic: https://www.cs.ubc.ca/~kvdoel/publications/foleyautomatic.pdf · O'Brien 2001: http://graphics.berkeley.edu/papers/Obrien-SSF-2001-08/Obrien-SSF-2001-08.pdf · O'Brien 2002: http://graphics.berkeley.edu/papers/Obrien-SSR-2002-07/Obrien-SSR-2002-07.pdf · Scanning behavior: https://dl.acm.org/doi/10.1145/383259.383296 · Bonneel 2008: https://www-sop.inria.fr/reves/Basilic/2008/BDTVJ08/FastModalSounds.pdf · Ren 2013: https://dl.acm.org/doi/10.1145/2421636.2421637
- PAT: http://graphics.cs.cmu.edu/projects/pat/ · Harmonic Shells: https://www.cs.cornell.edu/projects/HarmonicShells/ · Eigenmode Compression: https://www.cs.cornell.edu/projects/Sound/modec/ · Interactive transfer: https://www.cs.columbia.edu/cg/transfer/ · KleinPAT: https://graphics.stanford.edu/projects/kleinpat/
- Fracture Sound: https://www.cs.cornell.edu/projects/FractureSound/ · Acceleration Noise: https://dl.acm.org/doi/10.1145/2185520.2185599 · slides: https://graphics.stanford.edu/courses/sound/sig16/PBSound2016_AccelerationNoise_slides.pdf · Harmonic Fluids: https://www.cs.cornell.edu/projects/HarmonicFluids/ · Fire: https://www.cs.cornell.edu/projects/Sound/fire/ · Inverse-Foley: https://www.cs.cornell.edu/projects/Sound/ifa/ · Bubbles 2016: https://www.cs.cornell.edu/projects/Sound/bubbles/ · Crumpling: https://www.cs.columbia.edu/cg/crumpling/ · Wave turbulence: https://www.cs.columbia.edu/cg/waveturb/ · Metallophone: https://dl.acm.org/doi/10.1145/2816795.2818108 · FluidSound: https://github.com/kangruix/FluidSound
- Wavesolver 2018: https://www.antequ.net/assets/projects/wavesolver/wavesolver2018optimized.pdf · WaveBlender: https://graphics.stanford.edu/papers/waveblender/ · code: https://github.com/kangruix/WaveBlender
- Deep-Modal: https://hellojxt.github.io/DeepModal/ · NeuralSound: https://hellojxt.github.io/NeuralSound/ · DiffSound: https://hellojxt.github.io/DiffSound/ · Differentiable modal resonators: https://arxiv.org/abs/2210.15306 · NeuroSonic: https://onlinelibrary.wiley.com/doi/10.1002/cav.70162 · RealImpact: https://arxiv.org/abs/2306.09944 · ObjectFolder: https://arxiv.org/abs/2109.07991
- Staggered Projections: https://www.cs.ubc.ca/labs/sensorimotor/projects/sp_sigasia08/ · Timewarp: https://dl.acm.org/doi/10.1145/344779.344866 · Async Contact Mechanics: https://dl.acm.org/doi/10.1145/1531326.1531393 · Skipping Steps: https://dl.acm.org/doi/10.1145/1618452.1618469 · BD-Tree: https://dl.acm.org/doi/10.1145/1015706.1015772
- SIGGRAPH 2016 sound course: https://graphics.stanford.edu/courses/sound/ · ModalSound code: https://github.com/cxzheng/ModalSound · James publications: https://graphics.stanford.edu/~djames/publications/ · Zheng publications: http://www.cs.columbia.edu/~cxz/publications.htm

---

## 7. Open questions and early measurements

Things the literature cannot answer and the first prototypes should:

1. **Dependent-dispatch cost through a Metal 4 pass barrier** (µs, M5 Max) — bounds the maximum substep/color rate and therefore the whole solver architecture. No public number exists.
2. **Event-signaled frame round-trip latency under GPU clock-ramping** — the Anukari devlog documents downclocking between bursty short workloads (exactly the physics-tick profile); measure with and without keep-alive strategies.
3. **Filtered N² vs SAP vs LBVH broadphase crossover** on Apple Silicon at MeshEditor-scale body counts (10²–10⁴) — MuJoCo Warp's experience says brute force can win at moderate counts.
4. **Metal RT acceleration structures as a trimesh midphase** — AS refit cost for moving bodies vs a custom quantized 4-wide BVH is unbenchmarked on Apple hardware.
5. **Colored Gauss-Seidel vs mass-split Jacobi convergence** at equal wall-clock on the standard failure cases (tall towers, 1000:1 mass ratios, joint chains) at high substep counts.
6. **ICB semantics under MTL4 encoders**, `simdgroup_async_copy` availability on Apple9+, denormal behavior — verify against current docs when implementing.
7. **Impulse noise of candidate GPU solvers measured as audio** — spectrograms of contact-force traces from TGS-colored vs Jacobi vs AVBD on a resting/sliding/chattering scene, judged by the Zheng-James criteria (temporal coherence, spatial distribution), long before any modal coupling exists. This is the cheapest way to know whether the eventual audio goal constrains the solver choice now.
8. **The Zheng-James feasibility sweep** (from the project's framing conversation): two modal bodies + plane collision + friction + coupled modal/rigid impulses entirely in Metal, sweeping the coupled modal cutoff 1 → 2 → 5 → 10 → 20 kHz and measuring achievable contact count — the fastest way to learn whether the full idea has legs.
