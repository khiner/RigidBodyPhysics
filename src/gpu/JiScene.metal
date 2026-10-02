// Shared.h and JiData.h are prepended by the shader generator.

kernel void JiBuildBodies(
    device const Pose *poses [[buffer(0)]],
    device const Velocity *velocities [[buffer(1)]],
    device const BodyMass *masses [[buffer(2)]],
    device const uint *body_slots [[buffer(3)]],
    device const JiWrenchData *wrenches [[buffer(4)]],
    constant JiSceneParams &p [[buffer(5)]],
    device JiBodyData *bodies [[buffer(6)]],
    uint id [[thread_position_in_grid]]) {
    if (id >= p.moving_bodies) return;
    const uint slot = body_slots[id];
    const Pose pose = poses[slot];
    const Velocity velocity = velocities[slot];
    const BodyMass mass = masses[slot];
    const JiWrenchData wrench = wrenches[slot];
    const float m = 1 / mass.InvMass;
    const float3 inertia = 1 / mass.InvInertiaLocal;
    const float4 inverse = QuatConjugate(pose.Orientation);
    const float3 linear = Rotate(inverse, velocity.Linear);
    const float3 angular = Rotate(inverse, velocity.Angular);
    const float3 gravity = float3(p.gravity[0], p.gravity[1], p.gravity[2]);
    const float3 force = Rotate(inverse, float3(wrench.force[0], wrench.force[1], wrench.force[2]) +
                         m * mass.GravityScale * gravity);
    const float3 torque = Rotate(inverse, float3(wrench.torque[0], wrench.torque[1], wrench.torque[2]));
    const float3 momentum = inertia * angular;
    const float3 linear_rhs = force - m * cross(angular, linear);
    const float3 angular_rhs = torque - cross(angular, momentum);
    JiBodyData body = {};
    body.beta = 1;
    for (uint k = 0; k < 3; ++k) {
        body.a[k] = m;
        body.a[k + 3] = inertia[k];
        body.b[k] = m * linear[k] + p.dt * linear_rhs[k];
        body.b[k + 3] = inertia[k] * angular[k] + p.dt * angular_rhs[k];
        body.v[k] = linear[k];
        body.v[k + 3] = angular[k];
    }
    bodies[id] = body;
}

// Local first-order body-frame update; midpoint integration also requires a different body equation.
kernel void JiIntegrateBodies(
    device Pose *poses [[buffer(0)]],
    device Velocity *velocities [[buffer(1)]],
    device const uint *body_slots [[buffer(2)]],
    device const JiBodyData *bodies [[buffer(3)]],
    constant JiSceneParams &p [[buffer(4)]],
    device const JiStatus &status [[buffer(5)]],
    constant JiParams &solve [[buffer(6)]],
    uint id [[thread_position_in_grid]]) {
    if (id >= p.moving_bodies || !status.iterations ||
        (solve.tolerance > 0 && status.active) ||
        !isfinite(status.primal) || !isfinite(status.dual) || !isfinite(status.dynamics) ||
        !isfinite(status.contact_law)) return;
    const uint slot = body_slots[id];
    Pose pose = poses[slot];
    const float3 linear = float3(bodies[id].v[0], bodies[id].v[1], bodies[id].v[2]);
    const float3 angular = float3(bodies[id].v[3], bodies[id].v[4], bodies[id].v[5]);
    pose.Position += p.dt * Rotate(pose.Orientation, linear);
    pose.Orientation = normalize(QuatMul(pose.Orientation, QuatFromRotationVector(p.dt * angular)));
    poses[slot] = pose;
    velocities[slot] = {Rotate(pose.Orientation, linear), Rotate(pose.Orientation, angular)};
}

static void JiContactRow(thread JiSideData &side, uint row, Pose pose, float3 point,
                         float3 axis, float sign) {
    const float4 inverse = QuatConjugate(pose.Orientation);
    const float3 direction = Rotate(inverse, axis);
    const float3 arm = Rotate(inverse, point - pose.Position);
    const float3 angular = cross(arm, direction);
    for (uint k = 0; k < 3; ++k) {
        side.j[6 * row + k] = sign * direction[k];
        side.j[6 * row + k + 3] = sign * angular[k];
    }
}

static void JiAngularRow(thread JiSideData &side, uint row, Pose pose, float3 axis, float sign) {
    const float3 direction = Rotate(QuatConjugate(pose.Orientation), axis);
    for (uint k = 0; k < 3; ++k) side.j[6 * row + k + 3] = sign * direction[k];
}

static JiConstraintData JiJointRows(device const Joint &joint, device const Pose *poses,
                                    device const Velocity *velocities,
                                    device const uint *body_map, float3 anchor_a, float3 anchor_b,
                                    float4 frame_a, float3 rotation, uint mask, uint kind,
                                    float stiffness, float damping, constant JiSceneParams &p) {
    JiConstraintData result = {};
    result.kind = kind;
    result.rows = 6;
    result.stiffness = stiffness / p.dt;
    result.damping = damping / p.dt;
    result.side[0].body = body_map[joint.BodyA];
    result.side[1].body = body_map[joint.BodyB];
    const float3 translation = anchor_a - anchor_b;
    for (uint row = 0; row < 6; ++row) {
        if (!(mask & (1u << row))) continue;
        float3 unit = float3(0);
        unit[row % 3] = 1;
        const float3 axis = Rotate(frame_a, unit);
        const float error = dot(row < 3 ? translation : rotation, axis);
        result.error[row] = error * (kind == 1 ? p.joint_error_reduction : p.dt);
        float prescribed = 0;
        if (result.side[0].body == ~0u) {
            const Velocity velocity = velocities[joint.BodyA];
            const float3 point = velocity.Linear + cross(velocity.Angular,
                anchor_a - poses[joint.BodyA].Position);
            prescribed += dot(row < 3 ? point : velocity.Angular, axis);
        }
        if (result.side[1].body == ~0u) {
            const Velocity velocity = velocities[joint.BodyB];
            const float3 point = velocity.Linear + cross(velocity.Angular,
                anchor_b - poses[joint.BodyB].Position);
            prescribed -= dot(row < 3 ? point : velocity.Angular, axis);
        }
        if (kind == 0) result.velocity_bias[row] = p.dt * prescribed;
        else result.error[row] += p.dt * prescribed;
        if (row < 3) {
            if (result.side[0].body != ~0u)
                JiContactRow(result.side[0], row, poses[joint.BodyA], anchor_a, p.dt * axis, 1);
            if (result.side[1].body != ~0u)
                JiContactRow(result.side[1], row, poses[joint.BodyB], anchor_b, p.dt * axis, -1);
        } else {
            if (result.side[0].body != ~0u)
                JiAngularRow(result.side[0], row, poses[joint.BodyA], p.dt * axis, 1);
            if (result.side[1].body != ~0u)
                JiAngularRow(result.side[1], row, poses[joint.BodyB], p.dt * axis, -1);
        }
    }
    return result;
}

kernel void JiBuildJoints(
    device const Joint *joints [[buffer(0)]],
    device const Pose *poses [[buffer(1)]],
    device const uint *body_map [[buffer(2)]],
    device JiConstraintTile *constraints [[buffer(3)]],
    device atomic_uint *counts [[buffer(4)]], // contacts, static-only, invalid, joints
    constant JiSceneParams &p [[buffer(5)]],
    device JiJointLink *links [[buffer(6)]],
    device const Velocity *velocities [[buffer(7)]],
    uint id [[thread_position_in_grid]]) {
    if (id >= p.joint_count || !joints[id].Active) return;
    device const Joint &joint = joints[id];
    if (joint.BodyA >= p.body_count || joint.BodyB >= p.body_count) {
        atomic_fetch_add_explicit(counts + 2, 1u, memory_order_relaxed);
        return;
    }
    if (body_map[joint.BodyA] == ~0u && body_map[joint.BodyB] == ~0u) return;
    uint hard = 0, soft_rows[6], soft_count = 0;
    for (uint row = 0; row < 6; ++row) {
        const uint axis = row % 3;
        const uint mode = AxisMode(row < 3 ? joint.LinearModes : joint.AngularModes, axis);
        const float stiffness = row < 3 ? joint.LinearStiffness[axis] : joint.AngularStiffness[axis];
        const float damping = row < 3 ? joint.LinearDamping[axis] : joint.AngularDamping[axis];
        const float motor = row < 3 ? joint.LinearMotorMaxForce[axis] : joint.MotorMaxTorque[axis];
        if (joint.Drives[row].Enabled || motor != 0 || (mode != AxisFree && mode != AxisLocked) ||
            (mode == AxisLocked && !isinf(stiffness) &&
             (!(stiffness >= 0) || !(damping >= 0) || !isfinite(damping)))) {
            atomic_fetch_add_explicit(counts + 2, 1u, memory_order_relaxed);
            return;
        }
        if (mode == AxisFree) continue;
        if (isinf(stiffness)) hard |= 1u << row;
        else soft_rows[soft_count++] = row;
    }
    const uint added = soft_count + uint(hard != 0);
    if (!added) return;
    const uint base = atomic_fetch_add_explicit(counts + 3, added, memory_order_relaxed);
    const uint start = atomic_load_explicit(counts, memory_order_relaxed) + base;
    if (start + added > p.constraint_capacity) {
        atomic_fetch_add_explicit(counts + 2, 1u, memory_order_relaxed);
        return;
    }
    const Pose pose_a = poses[joint.BodyA], pose_b = poses[joint.BodyB];
    const float3 anchor_a = WorldPoint(pose_a, joint.AnchorA);
    const float3 anchor_b = WorldPoint(pose_b, joint.AnchorB);
    const float4 frame_a = normalize(QuatMul(pose_a.Orientation, joint.FrameA));
    const float4 frame_b = normalize(QuatMul(pose_b.Orientation, joint.FrameB));
    const float3 rotation = RotationVector(normalize(QuatMul(frame_a, QuatConjugate(frame_b))));
    for (uint i = 0; i < soft_count; ++i) {
        const uint row = soft_rows[i], axis = row % 3, mask = 1u << row;
        const float stiffness = row < 3 ? joint.LinearStiffness[axis] : joint.AngularStiffness[axis];
        const float damping = row < 3 ? joint.LinearDamping[axis] : joint.AngularDamping[axis];
        const JiConstraintData constraint = JiJointRows(joint, poses, velocities, body_map, anchor_a, anchor_b,
            frame_a, rotation, mask, 0, stiffness, damping, p);
        JiStoreConstraint(constraints, start + i, constraint);
        links[base + i] = {id, mask};
    }
    if (hard) {
        const JiConstraintData constraint = JiJointRows(joint, poses, velocities, body_map, anchor_a, anchor_b,
            frame_a, rotation, hard, 1, 0, 0, p);
        JiStoreConstraint(constraints, start + soft_count, constraint);
        links[base + soft_count] = {id, hard};
    }
}

kernel void JiBuildContacts(
    device const GeometryContact *geometry [[buffer(0)]],
    device const atomic_uint *geometry_count [[buffer(1)]],
    device const Pose *poses [[buffer(2)]],
    device const Velocity *velocities [[buffer(3)]],
    device const uint *body_map [[buffer(4)]],
    device JiConstraintTile *constraints [[buffer(5)]],
    device atomic_uint *counts [[buffer(6)]], // accepted, static-only, invalid
    constant JiSceneParams &p [[buffer(7)]],
    device JiContactLink *links [[buffer(8)]],
    device const uint *body_components [[buffer(9)]],
    device atomic_uint *component_contacts [[buffer(10)]],
    uint id [[thread_position_in_grid]]) {
    if (id >= p.max_features || id >= atomic_load_explicit(geometry_count, memory_order_relaxed)) return;
    const GeometryContact feature = geometry[id];
    if (feature.BodyA >= p.body_count || feature.BodyB >= p.body_count) {
        atomic_fetch_add_explicit(counts + 2, 1u, memory_order_relaxed);
        return;
    }
    const uint original_a = body_map[feature.BodyA], original_b = body_map[feature.BodyB];
    if (original_a == ~0u && original_b == ~0u) {
        atomic_fetch_add_explicit(counts + 1, 1u, memory_order_relaxed);
        return;
    }
    const bool flipped = original_a == ~0u;
    const uint a = flipped ? feature.BodyB : feature.BodyA;
    const uint b = flipped ? feature.BodyA : feature.BodyB;
    const float3 point_a = flipped ? feature.PointB : feature.PointA;
    const float3 point_b = flipped ? feature.PointA : feature.PointB;
    const float3 normal = normalize(flipped ? -feature.Normal : feature.Normal);
    const float3 seed = abs(normal.x) < 0.8f ? float3(1, 0, 0) : float3(0, 1, 0);
    const float3 tangent_1 = normalize(cross(normal, seed));
    const float3 axes[3] = {normal, tangent_1, cross(normal, tangent_1)};

    JiConstraintData contact = {};
    contact.kind = 2; // Kind::Contact
    contact.rows = 3;
    contact.friction = feature.Friction;
    contact.side[0].body = body_map[a];
    contact.side[1].body = body_map[b];
    // Separated look-ahead features retain their gap; overlap correction is optional.
    contact.error[0] = max(feature.Gap, 0.0f) +
                       p.contact_error_reduction * min(feature.Gap, 0.0f);
    for (uint row = 0; row < 3; ++row) {
        JiContactRow(contact.side[0], row, poses[a], point_a, p.dt * axes[row], 1);
        if (contact.side[1].body != ~0u)
            JiContactRow(contact.side[1], row, poses[b], point_b, p.dt * axes[row], -1);
        else {
            const Velocity v = velocities[b];
            const float3 point_velocity = v.Linear + cross(v.Angular, point_b - poses[b].Position);
            contact.error[row] -= p.dt * dot(axes[row], point_velocity);
        }
    }
    const uint slot = atomic_fetch_add_explicit(counts, 1u, memory_order_relaxed);
    JiStoreConstraint(constraints, slot, contact);
    links[slot] = {id, uint(flipped)};
    if (original_a != ~0u)
        atomic_store_explicit(component_contacts + body_components[original_a], 1u, memory_order_relaxed);
    if (original_b != ~0u)
        atomic_store_explicit(component_contacts + body_components[original_b], 1u, memory_order_relaxed);
}

kernel void JiSetConstraintCount(
    device uint *scene_counts [[buffer(6)]], // contacts, static-only, invalid, joints
    device JiParams &solve [[buffer(1)]],
    device JiStatus &status [[buffer(7)]],
    constant JiSceneParams &scene [[buffer(8)]],
    device const atomic_uint *geometry_count [[buffer(9)]],
    device const BroadPhaseNode *nodes [[buffer(10)]],
    uint id [[thread_position_in_grid]]) {
    if (id) return;
    const uint count = scene_counts[0] + scene_counts[3];
    const uint raw = atomic_load_explicit(geometry_count, memory_order_relaxed);
    const BroadPhaseNode root = nodes[scene.broad_phase_root];
    const bool invalid = root.Ready != (scene.body_count <= RadixSimdWidth ? 0u : 2u) || root.Errors ||
        raw > scene.max_features ||
        scene_counts[0] + scene_counts[1] + scene_counts[2] != raw ||
        scene_counts[2] || count > scene.constraint_capacity;
    solve.constraints = invalid ? 0 : count;
    if (invalid) {
        ++scene_counts[2];
        status.active = 0;
    }
}

kernel void JiResetIncidence(
    device atomic_uint *counts [[buffer(2)]],
    constant JiParams &p [[buffer(1)]],
    uint id [[thread_position_in_grid]]) {
    if (id < p.bodies) atomic_store_explicit(counts + id, 0u, memory_order_relaxed);
}

kernel void JiCountIncidence(
    device JiConstraintTile *constraints [[buffer(0)]],
    constant JiParams &p [[buffer(1)]],
    device atomic_uint *counts [[buffer(2)]],
    device const JiBodyData *bodies [[buffer(11)]],
    device const uint *body_components [[buffer(12)]],
    device const atomic_uint *component_contacts [[buffer(13)]],
    uint id [[thread_position_in_grid]]) {
    if (id >= p.constraints) return;
    const uint lane = id % JI_CONSTRAINT_LANES;
    device JiConstraintTile &tile = constraints[id / JI_CONSTRAINT_LANES];
    for (uint s = 0; s < 2; ++s) {
        device JiSideTile &side = tile.side[s];
        const uint body = side.body[lane];
        if (body != ~0u && body < p.bodies) {
            atomic_fetch_add_explicit(counts + body, 1u, memory_order_relaxed);
            if (tile.kind[lane] != 2 &&
                !atomic_load_explicit(component_contacts + body_components[body], memory_order_relaxed)) {
                for (uint row = 0; row < tile.rows[lane]; ++row) {
                    float x = 0;
                    for (uint axis = 0; axis < 6; ++axis)
                        x = fma(JiTileJ(side, row, axis, lane),
                                bodies[body].b[axis] / bodies[body].a[axis], x);
                    side.x[row][lane] = side.z[row][lane] = x;
                    side.x_lo[row][lane] = side.z_lo[row][lane] = 0;
                }
            }
        }
    }
}

kernel void JiPrefixIncidence(
    device const atomic_uint *counts [[buffer(2)]],
    constant JiParams &p [[buffer(1)]],
    device uint *offsets [[buffer(3)]],
    device atomic_uint *cursor [[buffer(4)]],
    uint id [[thread_position_in_grid]]) {
    if (id) return;
    uint sum = 0;
    for (uint body = 0; body < p.bodies; ++body) {
        const uint count = atomic_load_explicit(counts + body, memory_order_relaxed);
        offsets[body] = sum;
        atomic_store_explicit(cursor + body, sum, memory_order_relaxed);
        sum += count;
    }
    offsets[p.bodies] = sum;
}

kernel void JiFillIncidence(
    device const JiConstraintTile *constraints [[buffer(0)]],
    constant JiParams &p [[buffer(1)]],
    device atomic_uint *cursor [[buffer(4)]],
    device uint *incidence [[buffer(5)]],
    uint id [[thread_position_in_grid]]) {
    if (id >= p.constraints) return;
    const uint lane = id % JI_CONSTRAINT_LANES;
    device const JiConstraintTile &tile = constraints[id / JI_CONSTRAINT_LANES];
    for (uint s = 0; s < 2; ++s) {
        const uint body = tile.side[s].body[lane];
        if (body != ~0u && body < p.bodies) {
            const uint at = atomic_fetch_add_explicit(cursor + body, 1u, memory_order_relaxed);
            incidence[at] = 2 * id + s;
        }
    }
}

static bool JiIncidenceLess(uint a, uint b, uint contact_count,
                           device const GeometryContact *geometry,
                           device const JiContactLink *contacts,
                           device const JiJointLink *joints) {
    const uint ca = a / 2, cb = b / 2;
    const bool contact_a = ca < contact_count, contact_b = cb < contact_count;
    if (contact_a != contact_b) return contact_a;
    if (contact_a) {
        const GeometryContact x = geometry[contacts[ca].geometry_index];
        const GeometryContact y = geometry[contacts[cb].geometry_index];
        // Spawn and root-shape values are constant for each body within a step.
        const uint key_a[7] = {x.BodyA, x.BodyB, uint(x.Children), uint(x.Children >> 32),
                               x.SubShapeA, x.SubShapeB, x.Feature};
        const uint key_b[7] = {y.BodyA, y.BodyB, uint(y.Children), uint(y.Children >> 32),
                               y.SubShapeA, y.SubShapeB, y.Feature};
        for (uint k = 0; k < 7; ++k)
            if (key_a[k] != key_b[k]) return key_a[k] < key_b[k];
    } else {
        const JiJointLink x = joints[ca - contact_count], y = joints[cb - contact_count];
        if (x.joint_index != y.joint_index) return x.joint_index < y.joint_index;
        if (x.row_mask != y.row_mask) return x.row_mask < y.row_mask;
    }
    return (a & 1u) < (b & 1u);
}

static void JiSiftIncidence(device uint *entries, uint root, uint count,
                            uint contact_count, device const GeometryContact *geometry,
                            device const JiContactLink *contacts,
                            device const JiJointLink *joints) {
    while (2 * root + 1 < count) {
        uint child = 2 * root + 1;
        if (child + 1 < count && JiIncidenceLess(entries[child], entries[child + 1],
                                                contact_count, geometry, contacts, joints)) ++child;
        if (!JiIncidenceLess(entries[root], entries[child],
                             contact_count, geometry, contacts, joints)) break;
        const uint swap = entries[root];
        entries[root] = entries[child];
        entries[child] = swap;
        root = child;
    }
}

kernel void JiSortIncidence(
    constant JiParams &p [[buffer(1)]],
    device const uint *offsets [[buffer(3)]],
    device uint *incidence [[buffer(5)]],
    device const uint *scene_counts [[buffer(6)]],
    device uint *contribution_slots [[buffer(14)]],
    device const GeometryContact *geometry [[buffer(15)]],
    device const JiContactLink *contacts [[buffer(16)]],
    device const JiJointLink *joints [[buffer(17)]],
    uint body [[thread_position_in_grid]]) {
    if (body >= p.bodies) return;
    const uint begin = offsets[body], count = offsets[body + 1] - begin;
    device uint *entries = incidence + begin;
    const uint contact_count = scene_counts[0];
    for (uint root = count / 2; root; --root)
        JiSiftIncidence(entries, root - 1, count, contact_count, geometry, contacts, joints);
    for (uint end = count; end > 1; --end) {
        const uint swap = entries[0];
        entries[0] = entries[end - 1];
        entries[end - 1] = swap;
        JiSiftIncidence(entries, 0, end - 1, contact_count, geometry, contacts, joints);
    }
    // Body reductions and each constraint side now share the same stable order.
    for (uint n = 0; n < count; ++n) contribution_slots[entries[n]] = begin + n;
}

// Shared contact identity and matching helpers are prepended by the shader generator.

kernel void JiClearWarmIndex(
    device atomic_uint *claims [[buffer(8)]],
    device const uint *previous_count [[buffer(7)]],
    device atomic_uint *warm_count [[buffer(11)]],
    device atomic_uint *heads [[buffer(12)]],
    constant JiSceneParams &p [[buffer(18)]],
    uint id [[thread_position_in_grid]]) {
    if (id <= p.warm_bucket_mask)
        atomic_store_explicit(heads + id, ~0u, memory_order_relaxed);
    if (id < previous_count[0]) atomic_store_explicit(claims + id, 0u, memory_order_relaxed);
    if (id == 0) atomic_store_explicit(warm_count, 0u, memory_order_relaxed);
}

kernel void JiBuildWarmIndex(
    device const JiWarmContact *previous [[buffer(6)]],
    device const uint *previous_count [[buffer(7)]],
    device atomic_uint *heads [[buffer(12)]],
    device uint *next [[buffer(17)]],
    constant JiSceneParams &p [[buffer(18)]],
    uint id [[thread_position_in_grid]]) {
    if (id >= previous_count[0]) return;
    const JiWarmContact entry = previous[id];
    const uint bucket = JiWarmBucket(entry, p.warm_bucket_mask);
    next[id] = atomic_exchange_explicit(heads + bucket, id, memory_order_relaxed);
}

kernel void JiSeedContacts(
    device const GeometryContact *geometry [[buffer(0)]],
    device const JiContactLink *links [[buffer(1)]],
    device const uint *scene_counts [[buffer(2)]],
    device const Pose *poses [[buffer(3)]],
    device const uint *shapes [[buffer(4)]],
    device const uint *spawns [[buffer(5)]],
    device const JiWarmContact *previous [[buffer(6)]],
    device const uint *previous_count [[buffer(7)]],
    device atomic_uint *claims [[buffer(8)]],
    device JiConstraintTile *constraints [[buffer(9)]],
    device const JiBodyData *bodies [[buffer(10)]],
    device atomic_uint *warm_count [[buffer(11)]],
    device const atomic_uint *heads [[buffer(12)]],
    device const uint *next [[buffer(17)]],
    constant JiSceneParams &p [[buffer(18)]],
    uint id [[thread_position_in_grid]]) {
    if (id >= scene_counts[0] || !previous_count[0]) return;
    const JiContactLink link = links[id];
    const GeometryContact raw = geometry[link.geometry_index];
    const uint lane = id % JI_CONSTRAINT_LANES;
    device JiConstraintTile &contact = constraints[id / JI_CONSTRAINT_LANES];
    const uint matched = JiMatchWarm(raw,poses,spawns,shapes,previous,previous_count,
        claims,heads,next,p.warm_bucket_mask,JiMovingContactMask(link,contact,lane));
    if (matched == ~0u) return;
    device const JiWarmContact &old = previous[matched];
    const float3 old_normal = float3(old.normal[0], old.normal[1], old.normal[2]);
    const float3 axis = cross(old_normal, raw.Normal);
    const float cosine = dot(old_normal, raw.Normal);
    const float3 old_impulse = float3(old.impulse_on_a[0], old.impulse_on_a[1], old.impulse_on_a[2]);
    const float3 transported = old_impulse + cross(axis, old_impulse) +
        cross(axis, cross(axis, old_impulse)) / (1 + cosine);
    const float3 normal = normalize(link.flipped ? -raw.Normal : raw.Normal);
    const float3 seed = abs(normal.x) < 0.8f ? float3(1, 0, 0) : float3(0, 1, 0);
    const float3 tangent_1 = normalize(cross(normal, seed));
    const float3 axes[3] = {normal, tangent_1, cross(normal, tangent_1)};
    const float3 impulse = link.flipped ? -transported : transported;
    for (uint row = 0; row < 3; ++row) contact.impulse[row][lane] = dot(impulse, axes[row]) / p.dt;
    contact.impulse[0][lane] = max(0.0f, contact.impulse[0][lane]);
    const float radius = contact.friction[lane] * contact.impulse[0][lane];
    const float tangent = length(float2(contact.impulse[1][lane], contact.impulse[2][lane]));
    if (tangent > radius) {
        const float scale = radius / tangent;
        contact.impulse[1][lane] *= scale;
        contact.impulse[2][lane] *= scale;
    }
    for (uint side = 0; side < 2; ++side) {
        device JiSideTile &current = contact.side[side];
        const uint body = current.body[lane];
        if (body == ~0u) continue;
        for (uint row = 0; row < 3; ++row) {
            float x = 0;
            for (uint k = 0; k < 6; ++k) x += JiTileJ(current, row, k, lane) * bodies[body].v[k];
            current.x[row][lane] = current.z[row][lane] = x;
            current.u[row][lane] = -contact.impulse[row][lane];
        }
    }
    atomic_fetch_add_explicit(warm_count, 1u, memory_order_relaxed);
}

kernel void JiCacheContacts(
    device const GeometryContact *geometry [[buffer(0)]],
    device const JiContactLink *links [[buffer(1)]],
    device atomic_uint *scene_counts [[buffer(2)]],
    device const Pose *poses [[buffer(3)]],
    device const uint *shapes [[buffer(4)]],
    device const uint *spawns [[buffer(5)]],
    device const JiConstraintTile *constraints [[buffer(9)]],
    device JiWarmContact *next [[buffer(13)]],
    device const JiStatus &status [[buffer(14)]],
    constant JiParams &solve [[buffer(15)]],
    constant JiSceneParams &p [[buffer(18)]],
    uint id [[thread_position_in_grid]]) {
    if (id >= atomic_load_explicit(scene_counts, memory_order_relaxed) ||
        atomic_load_explicit(scene_counts + 2, memory_order_relaxed) || !status.iterations ||
        (solve.tolerance > 0 && status.active) ||
        !isfinite(status.primal) || !isfinite(status.dual) || !isfinite(status.dynamics) ||
        !isfinite(status.contact_law)) return;
    const JiContactLink link = links[id];
    const GeometryContact raw = geometry[link.geometry_index];
    JiWarmContact entry = {};
    const uint lane = id % JI_CONSTRAINT_LANES;
    device const JiConstraintTile &tile = constraints[id / JI_CONSTRAINT_LANES];
    JiCacheIdentity(entry, raw, link, tile, lane, poses, spawns, shapes);
    const float3 normal = normalize(link.flipped ? -raw.Normal : raw.Normal);
    const float3 seed = abs(normal.x) < 0.8f ? float3(1, 0, 0) : float3(0, 1, 0);
    const float3 tangent_1 = normalize(cross(normal, seed));
    const float3 axes[3] = {normal, tangent_1, cross(normal, tangent_1)};
    float3 impulse = float3(0);
    for (uint row = 0; row < 3; ++row) impulse += tile.impulse[row][lane] * axes[row];
    const float3 world_impulse = p.dt * (link.flipped ? -impulse : impulse);
    if (!all(isfinite(world_impulse))) {
        atomic_fetch_add_explicit(scene_counts + 2, 1u, memory_order_relaxed);
        return;
    }
    JiStore3(entry.impulse_on_a, world_impulse);
    next[id] = entry;
}

kernel void JiPublishCacheCount(
    device const uint *scene_counts [[buffer(2)]],
    device uint *next_count [[buffer(16)]],
    device JiStatus &status [[buffer(14)]],
    constant JiParams &solve [[buffer(15)]],
    uint id [[thread_position_in_grid]]) {
    if (id) return;
    const bool valid = !scene_counts[2] && status.iterations && (solve.tolerance == 0 || !status.active) &&
        isfinite(status.primal) && isfinite(status.dual) && isfinite(status.dynamics) &&
        isfinite(status.contact_law);
    next_count[0] = valid ? scene_counts[0] : 0;
    if (!valid) status.iterations = 0;
}

// Inspect solved state in parallel, then gate all world writes in JiFinalizeStep.
kernel void JiValidateState(
    device const JiBodyData *bodies [[buffer(0)]],
    device const JiConstraintTile *constraints [[buffer(1)]],
    constant JiParams &solve [[buffer(2)]],
    device const JiStatus &status [[buffer(3)]],
    device atomic_uint *scene_counts [[buffer(4)]],
    device const float *body_inverse [[buffer(5)]],
    device const JiContactLink *contact_links [[buffer(6)]],
    device const uint *geometry_count [[buffer(7)]],
    uint id [[thread_position_in_grid]]) {
    if (!status.iterations) return;
    if (id < solve.bodies) {
        device const JiBodyData &b = bodies[id];
        bool valid = b.beta > 0 && isfinite(b.beta);
        for (uint k = 0; k < 6; ++k)
            valid &= b.a[k] > 0 && isfinite(b.a[k]) && isfinite(b.b[k]) &&
                     isfinite(b.v[k]) && isfinite(b.v_lo[k]);
        if (!valid) {
            atomic_fetch_add_explicit(scene_counts + 2, 1u, memory_order_relaxed);
            atomic_fetch_add_explicit(scene_counts + 4, 1u, memory_order_relaxed);
            atomic_fetch_min_explicit(scene_counts + 6, id, memory_order_relaxed);
        }
        bool valid_inverse = true;
        for (uint k = 0; k < 42; ++k)
            valid_inverse &= isfinite(body_inverse[k * solve.bodies + id]);
        for (uint row = 0; row < 6; ++row)
            valid_inverse &= body_inverse[(row * (row + 1) / 2 + row) * solve.bodies + id] > 0;
        if (!valid_inverse) {
            atomic_fetch_add_explicit(scene_counts + 2, 1u, memory_order_relaxed);
            atomic_fetch_add_explicit(scene_counts + 8, 1u, memory_order_relaxed);
            atomic_fetch_min_explicit(scene_counts + 9, id, memory_order_relaxed);
        }
    }
    if (id < solve.constraints) {
        const uint lane = id % JI_CONSTRAINT_LANES;
        device const JiConstraintTile &c = constraints[id / JI_CONSTRAINT_LANES];
        bool valid = c.kind[lane] <= 2 && c.rows[lane] > 0 && c.rows[lane] <= 6 &&
            isfinite(c.stiffness[lane]) && isfinite(c.damping[lane]) && isfinite(c.friction[lane]);
        for (uint row = 0; row < 6; ++row)
            valid &= isfinite(c.error[row][lane]) && isfinite(c.impulse[row][lane]) &&
                     isfinite(c.velocity_bias[row][lane]);
        for (uint side = 0; side < 2; ++side) {
            device const JiSideTile &s = c.side[side];
            valid &= s.body[lane] == ~0u || s.body[lane] < solve.bodies;
            for (uint row = 0; row < 6; ++row)
                valid &= isfinite(s.x[row][lane]) && isfinite(s.z[row][lane]) &&
                         isfinite(s.u[row][lane]) && isfinite(s.old_z[row][lane]) &&
                         isfinite(s.x_lo[row][lane]) && isfinite(s.z_lo[row][lane]) &&
                         isfinite(s.u_lo[row][lane]) && isfinite(s.old_z_lo[row][lane]);
            for (uint k = 0; k < 36; ++k) valid &= isfinite(s.j[k][lane]);
        }
        if (!valid) {
            atomic_fetch_add_explicit(scene_counts + 2, 1u, memory_order_relaxed);
            atomic_fetch_add_explicit(scene_counts + 5, 1u, memory_order_relaxed);
            atomic_fetch_min_explicit(scene_counts + 7, id, memory_order_relaxed);
        }
    }
    if (id < atomic_load_explicit(scene_counts, memory_order_relaxed) &&
        (contact_links[id].geometry_index >= geometry_count[0] || contact_links[id].flipped > 1)) {
        atomic_fetch_add_explicit(scene_counts + 2, 1u, memory_order_relaxed);
        atomic_fetch_add_explicit(scene_counts + 5, 1u, memory_order_relaxed);
        atomic_fetch_min_explicit(scene_counts + 7, id, memory_order_relaxed);
    }
}

kernel void JiFinalizeStep(
    constant JiParams &solve [[buffer(2)]],
    device JiStatus &status [[buffer(3)]],
    device const uint *scene_counts [[buffer(4)]],
    uint id [[thread_position_in_grid]]) {
    if (id) return;
    const bool valid = !scene_counts[2] && status.iterations &&
        isfinite(status.primal) && isfinite(status.dual) && isfinite(status.dynamics) &&
        isfinite(status.contact_law) &&
        (solve.tolerance == 0 || !status.active);
    if (!valid)
        status.iterations = 0;
}
