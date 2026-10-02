# Ji physics model

The solver implements [Ji et al. 2025](https://doi.org/10.1109/ICRA55743.2025.11128665), using the normal-first, tangential-disk contact projection from [Lee et al., Eq. (26)](https://arxiv.org/abs/2201.09212). The conventions and local modeling choices below govern scene stepping and physical interpretation.

## Coordinates and units

Solver velocities use body-frame linear and angular coordinates about the center of mass. Contact points, axes, external wrenches and reported contact forces/impulses use world coordinates; external torque is about the center of mass.

Each constraint Jacobian `J` and prescribed-endpoint velocity offset includes `dt`. This is the implementation's interpretation of the paper's time-scaled Jacobian. The multiplier `lambda` has force/torque units: physical impulse is `dt * lambda`, and the updated ADMM dual is `u = -lambda`. The contact cache stores physical impulse.

## Dynamics and constraints

The body equation is `A*v = b + sum(Jᵀ*lambda)`, with `A = diag(m,m,m,Ixx,Iyy,Izz)`. The right-hand side contains previous momentum and time-integrated external, gravity and body-frame inertial terms.

Contact gap is positive in separation. Normal error is `max(gap, 0) + contact_error_reduction*min(gap, 0)`: separated features retain their speculative offset, while overlap correction is configurable. This collision policy and the separate hard-joint error-reduction coefficient are local choices.

Soft joints store `error = dt*position_error`, `velocity_bias = dt*prescribed_velocity`, `Ktile = k/dt`, and `Dtile = d/dt`. Their law is:

```text
lambda + Ktile*error + Dtile*(J*v + velocity_bias) = 0
```

Thus physical force/torque is `-k*position_error - d*relative_velocity`. Prescribed motion belongs in the damping term; zero stiffness or damping remains valid without a compliance floor.

## Integration and solver settings

Scene stepping uses a local first-order body-frame pose update and adds no restitution. The reauthored paper scenarios disable overlap and hard-joint error correction while preserving positive speculative gaps. Integration and stabilization choices are outside the frozen-equation comparison. Switching to midpoint integration would also require changing the momentum equation and velocity reconstruction.

Iteration count, penalty-update period and stopping tolerance are explicit inputs. A zero tolerance runs the fixed iteration budget; a positive tolerance requires primal, dual, dynamics and contact-law residuals to pass. A small residual describes the discrete solve, not trajectory quality.

## Reading diagnostics

Dynamics residuals mix linear momentum (N s) and angular momentum (N m s) components. Inspect the worst body/axis and its local impulse scale alongside the scalar maximum.

Contact-law residuals measure the normal-first/tangential-disk fixed point, scaled by inverse normal mobility. Contact-law and cone residuals have multiplier units (N). The optional frictional reference's historical `impulse_units` field names also refer to these multipliers. Assess physical behavior using motion, penetration, momentum balance and work.

## Code map

[JiScene.metal](../../src/gpu/JiScene.metal) constructs scene equations and integrates poses. [Ji.metal](../../src/gpu/Ji.metal) implements SubADMM; [Oracle.cpp](../../src/ji/Oracle.cpp) provides the CPU iteration and [Audit.cpp](../../src/ji/Audit.cpp) independently checks equations. See [Validation.md](Validation.md) for tests and reproducible workflows.
