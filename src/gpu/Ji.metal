// The shader generator prepends Shared.h and JiData.h.
GPU_CONSTANT uint JiNoBody = ~0u;

static uint JiPackedIndex(uint row, uint col) { return row * (row + 1) / 2 + col; }

// Requires safe_math in Pipelines.json: reassociation can discard the compensating low terms.
struct JiWide { float hi, lo; };
static JiWide JiW(float value) { return {value, 0}; }
static JiWide JiNormalize(float hi, float lo) {
    const float sum = hi + lo;
    return {sum, lo - (sum - hi)};
}
static JiWide JiAdd(JiWide a, JiWide b) {
    const float sum = a.hi + b.hi;
    const float v = sum - a.hi;
    const float error = (a.hi - (sum - v)) + (b.hi - v) + a.lo + b.lo;
    return JiNormalize(sum, error);
}
static JiWide JiNeg(JiWide a) { return {-a.hi, -a.lo}; }
static JiWide JiSub(JiWide a, JiWide b) { return JiAdd(a, JiNeg(b)); }
static JiWide JiMul(JiWide a, JiWide b) {
    const float product = a.hi * b.hi;
    float error = fma(a.hi, b.hi, -product);
    error = fma(a.hi, b.lo, error);
    error = fma(a.lo, b.hi, error);
    error += a.lo * b.lo;
    return JiNormalize(product, error);
}
static JiWide JiDiv(JiWide a, JiWide b) {
    const float first = a.hi / b.hi;
    const JiWide remainder = JiSub(a, JiMul(b, JiW(first)));
    return JiAdd(JiW(first), JiW((remainder.hi + remainder.lo) / b.hi));
}
static JiWide JiSqrt(JiWide a) {
    const float first = sqrt(a.hi);
    const JiWide remainder = JiSub(a, JiMul(JiW(first), JiW(first)));
    return JiAdd(JiW(first), JiW((remainder.hi + remainder.lo) / (2 * first)));
}

static void JiSolveFactor(thread JiWide *rhs, thread const JiWide *lower,
                          thread const JiWide *inverse_diagonal) {
    for (uint i = 0; i < 6; ++i) {
        for (uint k = 0; k < i; ++k)
            rhs[i] = JiSub(rhs[i], JiMul(lower[JiPackedIndex(i, k)], rhs[k]));
        rhs[i] = JiMul(rhs[i], inverse_diagonal[i]);
    }
    for (int i = 5; i >= 0; --i) {
        for (int k = i + 1; k < 6; ++k)
            rhs[i] = JiSub(rhs[i], JiMul(lower[JiPackedIndex(k, i)], rhs[k]));
        rhs[i] = JiMul(rhs[i], inverse_diagonal[i]);
    }
}

kernel void JiPrepareBodyMatrix(device const JiConstraintTile *constraints [[buffer(1)]],
                                constant JiParams &p [[buffer(4)]],
                                device const uint *offsets [[buffer(6)]],
                                device const uint *incidence [[buffer(7)]],
                                device float *body_matrix [[buffer(9)]],
                                uint id [[thread_position_in_grid]]) {
    if (id >= p.bodies) return;
    JiWide matrix[21];
    for (uint k = 0; k < 21; ++k) matrix[k] = JiW(0);
    for (uint n = offsets[id]; n < offsets[id + 1]; ++n) {
        const uint entry = incidence[n], c = entry / 2, side = entry % 2;
        const uint lane = c % JI_CONSTRAINT_LANES;
        device const JiConstraintTile &tile = constraints[c / JI_CONSTRAINT_LANES];
        for (uint r = 0; r < tile.rows[lane]; ++r)
            for (uint j = 0; j < 6; ++j)
                for (uint k = 0; k <= j; ++k)
                    matrix[JiPackedIndex(j, k)] = JiAdd(matrix[JiPackedIndex(j, k)],
                        JiMul(JiW(JiTileJ(tile.side[side], r, j, lane)),
                              JiW(JiTileJ(tile.side[side], r, k, lane))));
    }
    for (uint k = 0; k < 21; ++k) {
        body_matrix[k * p.bodies + id] = matrix[k].hi;
        body_matrix[(21 + k) * p.bodies + id] = matrix[k].lo;
    }
}

kernel void JiInitializePenalty(device JiBodyData *bodies [[buffer(0)]],
                                constant JiParams &p [[buffer(4)]],
                                device const float *body_matrix [[buffer(9)]],
                                uint id [[thread_position_in_grid]]) {
    if (id >= p.bodies) return;
    float trace_a = 0, trace_c = 0;
    for (uint k = 0; k < 6; ++k) {
        trace_a += bodies[id].a[k];
        trace_c += body_matrix[JiPackedIndex(k, k) * p.bodies + id] +
                   body_matrix[(21 + JiPackedIndex(k, k)) * p.bodies + id];
    }
    // The additive offset initializes beta; it is not a floor on later penalty updates.
    bodies[id].beta = trace_c > 0 ? p.penalty_mass * trace_a / trace_c + p.penalty_offset : p.penalty_offset;
}

kernel void JiFactorBody(device JiBodyData *bodies [[buffer(0)]],
                         constant JiParams &p [[buffer(4)]],
                         device const JiStatus &status [[buffer(5)]],
                         device const float *body_matrix [[buffer(9)]],
                         device float *body_inverse [[buffer(10)]],
                         uint id [[thread_position_in_grid]]) {
    if (id >= p.bodies || !status.active) return;
    const JiWide inverse_beta = JiDiv(JiW(1), JiW(bodies[id].beta));
    bodies[id].inverse_beta_hi = inverse_beta.hi;
    bodies[id].inverse_beta_lo = inverse_beta.lo;
    JiWide lower[21];
    for (uint row = 0; row < 6; ++row)
        for (uint col = 0; col <= row; ++col) {
            const uint at = JiPackedIndex(row, col);
            JiWide value = JiMul(JiW(bodies[id].beta),
                {body_matrix[at * p.bodies + id], body_matrix[(21 + at) * p.bodies + id]});
            if (row == col) value = JiAdd(value, JiW(bodies[id].a[row]));
            for (uint k = 0; k < col; ++k)
                value = JiSub(value, JiMul(lower[JiPackedIndex(row, k)],
                                               lower[JiPackedIndex(col, k)]));
            lower[at] = row == col ? JiSqrt(value) : JiDiv(value, lower[JiPackedIndex(col, col)]);
        }
    JiWide inverse_diagonal[6];
    for (uint row = 0; row < 6; ++row) {
        inverse_diagonal[row] = JiDiv(JiW(1), lower[JiPackedIndex(row, row)]);
    }
    for (uint col = 0; col < 6; ++col) {
        JiWide solved[6];
        for (uint row = 0; row < 6; ++row) solved[row] = JiW(row == col ? 1 : 0);
        JiSolveFactor(solved, lower, inverse_diagonal);
        for (uint row = col; row < 6; ++row) {
            const uint at = JiPackedIndex(row, col);
            body_inverse[at * p.bodies + id] = solved[row].hi;
            body_inverse[(21 + at) * p.bodies + id] = solved[row].lo;
        }
    }
}

static void JiAssembleConstraint(device const JiBodyData *bodies,
                                 device const JiConstraintTile &tile, uint id, uint lane,
                                 device JiContributionData *contributions,
                                 device const uint *contribution_slots) {
    for (uint s = 0; s < 2; ++s) {
        device const JiSideTile &side = tile.side[s];
        const uint body = side.body[lane];
        if (body == JiNoBody) continue;
        device JiContributionData &out = contributions[contribution_slots[2 * id + s]];
        JiWide rhs[6];
        for (uint k = 0; k < 6; ++k) rhs[k] = JiW(0);
        const float beta = bodies[body].beta;
        for (uint r = 0; r < tile.rows[lane]; ++r) {
            const JiWide scaled = JiSub(JiMul(JiW(beta), {side.z[r][lane], side.z_lo[r][lane]}),
                                        {side.u[r][lane], side.u_lo[r][lane]});
            for (uint j = 0; j < 6; ++j) {
                const float jvalue = JiTileJ(side, r, j, lane);
                rhs[j] = JiAdd(rhs[j], JiMul(JiW(jvalue), scaled));
            }
        }
        for (uint k = 0; k < 6; ++k) {
            out.rhs_hi[k] = rhs[k].hi;
            out.rhs_lo[k] = rhs[k].lo;
        }
    }
}

kernel void JiConstraintAssemble(device const JiBodyData *bodies [[buffer(0)]],
                                 device JiConstraintTile *constraints [[buffer(1)]],
                                 constant JiParams &p [[buffer(4)]],
                                 device const JiStatus &status [[buffer(5)]],
                                 device JiContributionData *contributions [[buffer(8)]],
                                 device const uint *contribution_slots [[buffer(12)]],
                                 uint id [[thread_position_in_grid]]) {
    if (id >= p.constraints || !status.active) return;
    const uint lane = id % JI_CONSTRAINT_LANES;
    device JiConstraintTile &tile = constraints[id / JI_CONSTRAINT_LANES];
    JiWide inv_sum = JiW(0);
    for (uint s = 0; s < 2; ++s) {
        const uint body = tile.side[s].body[lane];
        if (body != JiNoBody)
            inv_sum = JiAdd(inv_sum, {bodies[body].inverse_beta_hi,
                                      bodies[body].inverse_beta_lo});
    }
    const JiWide denominator = tile.kind[lane] == 0
        ? JiAdd(JiW(1), JiMul(JiW(tile.damping[lane]), inv_sum)) : inv_sum;
    const JiWide inverse = JiDiv(JiW(1), denominator);
    tile.inverse_denominator_hi[lane] = inverse.hi;
    tile.inverse_denominator_lo[lane] = inverse.lo;
    JiAssembleConstraint(bodies, tile, id, lane, contributions, contribution_slots);
}

kernel void JiBodyUpdate(device JiBodyData *bodies [[buffer(0)]],
                         constant JiParams &p [[buffer(4)]], device JiStatus &status [[buffer(5)]],
                         device const uint *offsets [[buffer(6)]],
                         device const JiContributionData *contributions [[buffer(8)]],
                         device const float *body_inverse [[buffer(10)]],
                         uint id [[thread_position_in_grid]]) {
    if (id >= p.bodies || !status.active) return;
    if (id == 0) ++status.iterations;
    JiWide rhs[6], inverse[21], solved[6];
    for (uint k = 0; k < 6; ++k) rhs[k] = JiW(bodies[id].b[k]);
    for (uint at = 0; at < 21; ++at)
        inverse[at] = {body_inverse[at * p.bodies + id],
                       body_inverse[(21 + at) * p.bodies + id]};
    for (uint n = offsets[id]; n < offsets[id + 1]; ++n) {
        device const JiContributionData &contribution = contributions[n];
        for (uint j = 0; j < 6; ++j)
            rhs[j] = JiAdd(rhs[j], {contribution.rhs_hi[j], contribution.rhs_lo[j]});
    }
    for (uint row = 0; row < 6; ++row) {
        solved[row] = JiW(0);
        for (uint col = 0; col < 6; ++col) {
            const uint at = row >= col ? JiPackedIndex(row, col) : JiPackedIndex(col, row);
            solved[row] = JiAdd(solved[row], JiMul(inverse[at], rhs[col]));
        }
    }
    for (uint i = 0; i < 6; ++i) {
        bodies[id].v[i] = solved[i].hi;
        bodies[id].v_lo[i] = solved[i].lo;
    }
}

kernel void JiConstraintUpdate(device const JiBodyData *bodies [[buffer(0)]],
                               device JiConstraintTile *constraints [[buffer(1)]],
                               device JiResidualData *residuals [[buffer(3)]],
                               constant JiParams &p [[buffer(4)]], device const JiStatus &status [[buffer(5)]],
                               device JiContributionData *contributions [[buffer(8)]],
                               device const uint *contribution_slots [[buffer(12)]],
                               uint id [[thread_position_in_grid]]) {
    if (id >= p.constraints || !status.active) return;
    const uint lane = id % JI_CONSTRAINT_LANES;
    device JiConstraintTile &tile = constraints[id / JI_CONSTRAINT_LANES];
    const uint rows = tile.rows[lane];
    const bool capture = p.tolerance > 0 || status.iterations == p.max_iterations ||
        (p.penalty_period && status.iterations % p.penalty_period == 0);
    JiWide y[2][6], current_x[2][6], inv_beta[2] = {JiW(0), JiW(0)};
    for (uint s = 0; s < 2; ++s) {
        device JiSideTile &side = tile.side[s];
        for (uint r = 0; r < rows; ++r) {
            if (capture) {
                side.old_z[r][lane] = side.z[r][lane];
                side.old_z_lo[r][lane] = side.z_lo[r][lane];
            }
            y[s][r] = JiW(0);
            current_x[s][r] = JiW(0);
        }
        const uint body = side.body[lane];
        if (body == JiNoBody) continue;
        const float beta = bodies[body].beta;
        inv_beta[s] = {bodies[body].inverse_beta_hi, bodies[body].inverse_beta_lo};
        for (uint r = 0; r < rows; ++r) {
            JiWide x = JiW(0);
            for (uint k = 0; k < 6; ++k)
                x = JiAdd(x, JiMul(JiW(JiTileJ(side, r, k, lane)),
                                   {bodies[body].v[k], bodies[body].v_lo[k]}));
            current_x[s][r] = x;
            if (capture) {
                side.x[r][lane] = x.hi;
                side.x_lo[r][lane] = x.lo;
            }
            y[s][r] = JiAdd(JiMul(JiW(beta), x), {side.u[r][lane], side.u_lo[r][lane]});
        }
    }
    const JiWide inverse_denominator = {tile.inverse_denominator_hi[lane],
                                        tile.inverse_denominator_lo[lane]};
    JiWide impulse[6];
    for (uint r = 0; r < rows; ++r) {
        const JiWide weighted_y = JiAdd(JiMul(inv_beta[0], y[0][r]),
                                        JiMul(inv_beta[1], y[1][r]));
        if (tile.kind[lane] == 0) {
            const JiWide numerator = JiAdd(JiMul(JiW(tile.stiffness[lane]), JiW(tile.error[r][lane])),
                JiMul(JiW(tile.damping[lane]), JiAdd(weighted_y, JiW(tile.velocity_bias[r][lane]))));
            impulse[r] = JiNeg(JiMul(numerator, inverse_denominator));
        } else {
            impulse[r] = JiNeg(JiMul(JiAdd(JiW(tile.error[r][lane]), weighted_y),
                                       inverse_denominator));
        }
        if (capture && tile.kind[lane] != 2)
            tile.impulse[r][lane] = impulse[r].hi + impulse[r].lo;
    }
    if (tile.kind[lane] == 2) {
        float projected[3] = {max(0.0f, impulse[0].hi + impulse[0].lo),
                              impulse[1].hi + impulse[1].lo, impulse[2].hi + impulse[2].lo};
        const float radius = tile.friction[lane] * projected[0];
        const float tangent = length(float2(projected[1], projected[2]));
        if (tangent > radius) {
            const float scale = radius / tangent;
            projected[1] *= scale;
            projected[2] *= scale;
        }
        for (uint r = 0; r < rows; ++r) {
            impulse[r] = JiW(projected[r]);
            if (capture) tile.impulse[r][lane] = projected[r];
        }
    }
    float contact_law = 0;
    if (capture && tile.kind[lane] == 2) {
        JiWide normal_mobility = JiW(0);
        for (uint s = 0; s < 2; ++s) {
            device const JiSideTile &side = tile.side[s];
            const uint body = side.body[lane];
            if (body == JiNoBody) continue;
            for (uint k = 0; k < 6; ++k) {
                const JiWide j = JiW(JiTileJ(side, 0, k, lane));
                normal_mobility = JiAdd(normal_mobility,
                    JiDiv(JiMul(j, j), JiW(bodies[body].a[k])));
            }
        }
        const JiWide rho = JiDiv(JiW(1),
            normal_mobility.hi + normal_mobility.lo >= 1e-12f ? normal_mobility : JiW(1e-12f));
        JiWide projected[3];
        for (uint r = 0; r < 3; ++r) {
            const JiWide relative = JiAdd(JiAdd(current_x[0][r], current_x[1][r]),
                                          JiW(tile.error[r][lane]));
            projected[r] = JiSub(impulse[r], JiMul(rho, relative));
        }
        if (projected[0].hi + projected[0].lo < 0) projected[0] = JiW(0);
        const JiWide radius = JiMul(JiW(tile.friction[lane]), projected[0]);
        const JiWide tangent_squared = JiAdd(JiMul(projected[1], projected[1]),
                                               JiMul(projected[2], projected[2]));
        const JiWide radius_squared = JiMul(radius, radius);
        if (tangent_squared.hi + tangent_squared.lo > radius_squared.hi + radius_squared.lo) {
            const JiWide scale = JiDiv(radius, JiSqrt(tangent_squared));
            projected[1] = JiMul(projected[1], scale);
            projected[2] = JiMul(projected[2], scale);
        }
        for (uint r = 0; r < 3; ++r) {
            const JiWide difference = JiSub(impulse[r], projected[r]);
            contact_law = max(contact_law, abs(difference.hi + difference.lo));
        }
    }
    float primal = 0;
    for (uint s = 0; s < 2; ++s) {
        device JiSideTile &side = tile.side[s];
        const uint body = side.body[lane];
        if (body == JiNoBody) continue;
        for (uint r = 0; r < rows; ++r) {
            const JiWide x = current_x[s][r];
            const JiWide u = {side.u[r][lane], side.u_lo[r][lane]};
            const JiWide z = JiAdd(x, JiMul(JiAdd(u, impulse[r]), inv_beta[s]));
            // z = x + (u + impulse) / beta, so the dual update is exactly -impulse.
            const JiWide next_u = JiNeg(impulse[r]);
            side.z[r][lane] = z.hi;
            side.z_lo[r][lane] = z.lo;
            side.u[r][lane] = next_u.hi;
            side.u_lo[r][lane] = next_u.lo;
            if (capture) {
                const JiWide difference = JiSub(x, z);
                primal = max(primal, abs(difference.hi + difference.lo));
            }
        }
    }
    if (capture) residuals[id] = {primal, 0, 0, contact_law};
    JiAssembleConstraint(bodies, tile, id, lane, contributions, contribution_slots);
}

kernel void JiBodyResidual(device const JiBodyData *bodies [[buffer(0)]],
                           device const JiConstraintTile *constraints [[buffer(1)]],
                           device JiResidualData *residuals [[buffer(2)]],
                           constant JiParams &p [[buffer(4)]], device const JiStatus &status [[buffer(5)]],
                           device const uint *offsets [[buffer(6)]],
                           device const uint *incidence [[buffer(7)]],
                           uint id [[thread_position_in_grid]]) {
    if (id >= p.bodies || !status.active) return;
    JiWide dual[6], force[6];
    for (uint k = 0; k < 6; ++k) dual[k] = force[k] = JiW(0);
    for (uint n = offsets[id]; n < offsets[id + 1]; ++n) {
        const uint entry = incidence[n], c = entry / 2, s = entry % 2;
        const uint lane = c % JI_CONSTRAINT_LANES;
        device const JiConstraintTile &tile = constraints[c / JI_CONSTRAINT_LANES];
        device const JiSideTile &side = tile.side[s];
        for (uint r = 0; r < tile.rows[lane]; ++r)
            for (uint k = 0; k < 6; ++k) {
                const JiWide j = JiW(JiTileJ(side, r, k, lane));
                const JiWide change = JiSub({side.old_z[r][lane], side.old_z_lo[r][lane]},
                                            {side.z[r][lane], side.z_lo[r][lane]});
                dual[k] = JiAdd(dual[k], JiMul(JiMul(JiW(bodies[id].beta), j), change));
                force[k] = JiAdd(force[k], JiMul(j, JiW(tile.impulse[r][lane])));
            }
    }
    float max_dual = 0, max_dynamics = 0;
    for (uint k = 0; k < 6; ++k) {
        max_dual = max(max_dual, abs(dual[k].hi + dual[k].lo));
        const JiWide balance = JiSub(JiSub(
            JiMul(JiW(bodies[id].a[k]), {bodies[id].v[k], bodies[id].v_lo[k]}),
            JiW(bodies[id].b[k])), force[k]);
        max_dynamics = max(max_dynamics, abs(balance.hi + balance.lo));
    }
    residuals[id] = {0, max_dual, max_dynamics, 0};
}

// Each group reduces a contiguous block; the final check scans only group maxima.
kernel void JiReduceResiduals(device const JiResidualData *body_residuals [[buffer(2)]],
                              device const JiResidualData *constraint_residuals [[buffer(3)]],
                              constant JiParams &p [[buffer(4)]],
                              device const JiStatus &status [[buffer(5)]],
                              device JiResidualData *partial [[buffer(11)]],
                              uint id [[thread_position_in_grid]],
                              uint lane [[thread_index_in_threadgroup]],
                              uint group [[threadgroup_position_in_grid]]) {
    threadgroup JiResidualData shared[64];
    JiResidualData value = {0, 0, 0, 0};
    if (status.active) {
        if (id < p.constraints) {
            value.primal = constraint_residuals[id].primal;
            value.contact_law = constraint_residuals[id].contact_law;
        }
        if (id < p.bodies) {
            value.dual = body_residuals[id].dual;
            value.dynamics = body_residuals[id].dynamics;
        }
    }
    shared[lane] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 32; stride; stride >>= 1) {
        if (lane < stride) {
            shared[lane].primal = max(shared[lane].primal, shared[lane + stride].primal);
            shared[lane].dual = max(shared[lane].dual, shared[lane + stride].dual);
            shared[lane].dynamics = max(shared[lane].dynamics, shared[lane + stride].dynamics);
            shared[lane].contact_law = max(shared[lane].contact_law,
                                           shared[lane + stride].contact_law);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) partial[group] = shared[0];
}

kernel void JiCheckConvergence(device const JiResidualData *partial [[buffer(11)]],
                               constant JiParams &p [[buffer(4)]], device JiStatus &status [[buffer(5)]],
                               uint id [[thread_position_in_grid]]) {
    if (id || !status.active) return;
    float primal = 0, dual = 0, dynamics = 0, contact_law = 0;
    const uint groups = (max(p.constraints, p.bodies) + 63) / 64;
    for (uint i = 0; i < groups; ++i) {
        primal = max(primal, partial[i].primal);
        dual = max(dual, partial[i].dual);
        dynamics = max(dynamics, partial[i].dynamics);
        contact_law = max(contact_law, partial[i].contact_law);
    }
    status.primal = primal;
    status.dual = dual;
    status.dynamics = dynamics;
    status.contact_law = contact_law;
    if (p.tolerance > 0 && primal <= p.tolerance && dual <= p.tolerance &&
        dynamics <= p.tolerance && contact_law <= p.tolerance)
        status.active = 0;
}

kernel void JiPenaltyUpdate(device JiBodyData *bodies [[buffer(0)]],
                            constant JiParams &p [[buffer(4)]], device const JiStatus &status [[buffer(5)]],
                            uint id [[thread_position_in_grid]]) {
    if (id >= p.bodies || !status.active) return;
    const float primal = status.primal, dual = status.dual;
    const float ratio = dual == 0 ? (primal > 0 ? p.penalty_alpha : 1.0f) :
        clamp(primal / dual, 1.0f / p.penalty_alpha, p.penalty_alpha);
    bodies[id].beta *= ratio;
}

// Diagnostic capture of the initialized system before its first ADMM iteration.
kernel void JiSnapshotInitial(device const JiBodyData *bodies [[buffer(0)]],
                              device const JiConstraintTile *constraints [[buffer(1)]],
                              device JiBodyData *saved_bodies [[buffer(2)]],
                              device JiConstraintTile *saved_constraints [[buffer(3)]],
                              constant JiParams &p [[buffer(4)]],
                              uint id [[thread_position_in_grid]]) {
    const uint body_words = p.bodies * (sizeof(JiBodyData) / sizeof(uint));
    const uint tile_words = ((p.constraints + JI_CONSTRAINT_LANES - 1) / JI_CONSTRAINT_LANES) *
                            (sizeof(JiConstraintTile) / sizeof(uint));
    if (id < body_words) {
        reinterpret_cast<device uint *>(saved_bodies)[id] =
            reinterpret_cast<device const uint *>(bodies)[id];
    } else if (id - body_words < tile_words) {
        const uint word = id - body_words;
        reinterpret_cast<device uint *>(saved_constraints)[word] =
            reinterpret_cast<device const uint *>(constraints)[word];
    }
}
