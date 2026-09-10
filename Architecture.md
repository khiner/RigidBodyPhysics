# Architecture

## World data

A world owns fixed-capacity Metal shared buffers for bodies, shapes, joints, contacts and solver scratch.
Indices remain stable until retirement.
Pool exhaustion is reported through refusal counts.

The coordinate system is right-handed with Y up and SI units.
Body poses are centered on the center of mass, and inertia is diagonal in the body frame.
Collider-local transforms are composed with body transforms when measuring contacts.
Movable meshes and offset colliders require explicit mass properties about the body frame.
Zero inverse mass or inverse inertia locks the corresponding translation or rotation.

CPU and GPU code share the layouts in `src/gpu/Shared.h`.
Topology edits rebuild sorted joint incidence and collision-exclusion lists.
World edits must finish before an advance begins.

## Collision detection

Each substep builds body bounds and queries a GPU broad phase.
Small worlds use candidate masks.
Larger worlds use a Morton-sorted binary hierarchy with ordered body traversal.
Later substeps can refit the same hierarchy while retaining conservative bounds.

Convex distance and penetration use GJK and EPA, with face clipping and manifold reduction for contact construction.
Mesh traversal uses cooked bounds, triangle adjacency and active-edge information.
Welding merges duplicate contacts while preserving distinct features and material and filter boundaries.
Collision detection samples each substep discretely.

Mesh queries use a bounded GPU arena and shared SIMD lanes.
Queries that exceed arena capacity are recomputed directly.
Cached solid queries require identical geometry, topology, poses and query inputs.
Sensor queries use separate scratch storage and produce overlap reports.

A pair has one contact owner.
Body, child and feature information identify contacts for warm starting and reporting.
Contacts beyond the fixed per-body capacity are counted as refusals.
Filters and joint exclusions apply before narrow-phase evaluation.

## Solver

AVBD alternates colored primal body updates with contact and joint dual updates.
Each body gathers its incident constraints.
Neighbor reads within a conflicting color use the iteration snapshot.
Contact measurements are taken before warm starting moves bodies to their initial guesses.

The normal points from body B toward A.
Penetration and the normal dual are negative.
Contact rows include normal response and Coulomb friction.
Joint rows support independently configured linear and angular limits, drives, springs and damping.
The body solve assembles a compensated 6-by-6 matrix and gradient and uses LDL factorization.
Compensated arithmetic preserves inertial terms beside stiff constraints.
Poses and velocities use single precision.

SIMD lanes share matrix assembly and factorization while retaining constraint accumulation order.
Small connected components solve within a workgroup.
Larger components use the global colored schedule.
Only solved endpoints connect components.
Exact fixed-point checks stop redundant coloring and island iterations.

Velocity finalization precedes positional stabilization to exclude penetration correction from velocity.
Restitution gathers impulses from shared velocity snapshots.
Sleeping and waking propagate through contacts and joints, including moving kinematic bodies.

## Metal execution

An advance encodes dependent dispatches into one Metal 4 compute encoder with explicit device-visibility barriers.
Queue-signalled events establish completion before the CPU reads shared outputs.
The output-buffer budget limits the number of substeps per submission.
Every substep retains collision detection and a complete solve.
GPU-generated color budgets and fallback command streams avoid host reads between substeps and iterations.

Eligible small worlds with primitive bodies and zero joints use one complete-step kernel.
Other worlds use the same numerical routines through the ordinary dispatch schedule.
`cmake/FullStep.py` adapts the shared routines' bindings, scratch storage and synchronization scope.

## Outputs

`Solver::Step` and `Solver::Advance` block through completion and CPU reporting.
Requested snapshots retain intermediate poses, velocities, contact reports, sensor pairs and refusal counts.
Reporting consumes completed substeps in order, even when the live GPU state has advanced further.
Sensor followers update before each substep in dependency order.

Observer spans are immutable and valid only during the callback.
Observers may consume event queues and require an unchanged world and exclusive use of the solver.
If an observer throws, the solver drains reporting for the completed submission before propagating the exception.

See [Validation.md](Validation.md) for independent checks and benchmark usage, and [NOTICE.md](NOTICE.md) for algorithm and source attribution.
