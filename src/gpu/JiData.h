#ifndef JI_GPU_DATA_H
#define JI_GPU_DATA_H

#ifdef __METAL_VERSION__
#define JI_UINT uint
#else
#include <cstdint>
#define JI_UINT uint32_t
#endif

// Frozen-constraint SubADMM research data. Scalar arrays have the same layout
// in C++ and Metal. JiConstraintData is the Metal scene construction record;
// persistent GPU constraints are stored directly in 32-lane component tiles.
#define JI_CONSTRAINT_LANES 32
struct JiBodyData {
    float a[6], b[6], v[6], v_lo[6], beta, inverse_beta_hi, inverse_beta_lo;
};

struct JiSideData {
    JI_UINT body;
    float j[36], x[6], z[6], u[6], old_z[6];
    float x_lo[6], z_lo[6], u_lo[6], old_z_lo[6];
};

struct JiConstraintData {
    JI_UINT kind, rows;
    JiSideData side[2];
    float error[6], impulse[6], velocity_bias[6];
    float stiffness, damping, friction;
    float inverse_denominator_hi, inverse_denominator_lo;
};

struct JiSideTile {
    JI_UINT body[JI_CONSTRAINT_LANES];
    float j[36][JI_CONSTRAINT_LANES];
    float x[6][JI_CONSTRAINT_LANES], z[6][JI_CONSTRAINT_LANES];
    float u[6][JI_CONSTRAINT_LANES], old_z[6][JI_CONSTRAINT_LANES];
    float x_lo[6][JI_CONSTRAINT_LANES], z_lo[6][JI_CONSTRAINT_LANES];
    float u_lo[6][JI_CONSTRAINT_LANES], old_z_lo[6][JI_CONSTRAINT_LANES];
};

struct JiConstraintTile {
    JI_UINT kind[JI_CONSTRAINT_LANES], rows[JI_CONSTRAINT_LANES];
    JiSideTile side[2];
    float error[6][JI_CONSTRAINT_LANES], impulse[6][JI_CONSTRAINT_LANES];
    float velocity_bias[6][JI_CONSTRAINT_LANES]; // Soft: dt-scaled prescribed endpoint velocity.
    float stiffness[JI_CONSTRAINT_LANES], damping[JI_CONSTRAINT_LANES];
    float friction[JI_CONSTRAINT_LANES];
    float inverse_denominator_hi[JI_CONSTRAINT_LANES];
    float inverse_denominator_lo[JI_CONSTRAINT_LANES];
};

#ifdef __METAL_VERSION__
static float JiTileJ(device const JiSideTile &side, uint row, uint col, uint lane) {
    return side.j[6 * row + col][lane];
}

static void JiStoreConstraint(device JiConstraintTile *tiles, uint id,
                              thread const JiConstraintData &source) {
    const uint lane = id % JI_CONSTRAINT_LANES;
    device JiConstraintTile &tile = tiles[id / JI_CONSTRAINT_LANES];
    tile.kind[lane] = source.kind;
    tile.rows[lane] = source.rows;
    tile.stiffness[lane] = source.stiffness;
    tile.damping[lane] = source.damping;
    tile.friction[lane] = source.friction;
    tile.inverse_denominator_hi[lane] = source.inverse_denominator_hi;
    tile.inverse_denominator_lo[lane] = source.inverse_denominator_lo;
    for (uint row = 0; row < 6; ++row) {
        tile.error[row][lane] = source.error[row];
        tile.impulse[row][lane] = source.impulse[row];
        tile.velocity_bias[row][lane] = source.velocity_bias[row];
    }
    for (uint s = 0; s < 2; ++s) {
        thread const JiSideData &src = source.side[s];
        device JiSideTile &dst = tile.side[s];
        dst.body[lane] = src.body;
        for (uint k = 0; k < 36; ++k) dst.j[k][lane] = src.j[k];
        for (uint k = 0; k < 6; ++k) {
            dst.x[k][lane] = src.x[k]; dst.z[k][lane] = src.z[k];
            dst.u[k][lane] = src.u[k]; dst.old_z[k][lane] = src.old_z[k];
            dst.x_lo[k][lane] = src.x_lo[k]; dst.z_lo[k][lane] = src.z_lo[k];
            dst.u_lo[k][lane] = src.u_lo[k]; dst.old_z_lo[k][lane] = src.old_z_lo[k];
        }
    }
}
#endif

// One moving constraint side contributes the changing J^T*(beta*z-u) term.
// Entries are stored in body incidence order; each side's slot is looked up
// through the inverse incidence map built with the current constraint rows.
// Fixed J^T*J is summed once per step into 21 component-major body planes.
struct JiContributionData {
    float rhs_hi[6], rhs_lo[6];
};

struct JiResidualData {
    float primal, dual, dynamics, contact_law;
};

struct JiParams {
    JI_UINT bodies, constraints, penalty_period, max_iterations;
    float penalty_mass, penalty_offset, penalty_alpha, tolerance;
};

struct JiStatus {
    JI_UINT active, iterations;
    float primal, dual, dynamics, contact_law;
};

struct JiSceneParams {
    JI_UINT max_features, body_count, moving_bodies, joint_count, constraint_capacity, broad_phase_root;
    JI_UINT warm_bucket_mask;
    float dt, contact_error_reduction, joint_error_reduction;
    float gravity[3];
};

struct JiWrenchData {
    float force[3], torque[3]; // world frame, torque about COM
};

struct JiContactLink {
    JI_UINT geometry_index, flipped;
};

struct JiJointLink {
    JI_UINT joint_index, row_mask;
};

// Persisted per raw feature, independent of the collector's append order.
struct JiWarmContact {
    JI_UINT key[11]; // slots/generations, root shapes, children, subshapes, feature
    float local_a[3], local_b[3], normal[3], impulse_on_a[3];
    JI_UINT moving_mask; // Dynamic-side identity in typed key A/B order
};

#undef JI_UINT

#endif
