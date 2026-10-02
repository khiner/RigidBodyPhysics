#include "Metal.h"

#include "gpu/JiData.h"
#include "Pipelines.h"
#include "metal/Buffer.h"
#include "metal/Context.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <semaphore>
#include <stdexcept>
#include <string>
#include <vector>

namespace ji {
namespace {

static_assert(sizeof(JiBodyData) == 108);
static_assert(sizeof(JiSideData) == 340);
static_assert(sizeof(JiConstraintData) == 780);
static_assert(sizeof(JiConstraintTile) == JI_CONSTRAINT_LANES * sizeof(JiConstraintData));
static_assert(sizeof(JiContributionData) == 48);
static_assert(sizeof(JiResidualData) == 16);
static_assert(sizeof(JiParams) == 32);
static_assert(sizeof(JiStatus) == 24);
static_assert(sizeof(JiSceneParams) == 52);
static_assert(sizeof(JiWrenchData) == 24);
static_assert(sizeof(JiContactLink) == 8);
static_assert(sizeof(JiJointLink) == 8);
static_assert(sizeof(JiWarmContact) == 96);

template<size_t N> void ToFloat(float (&target)[N], const std::array<double, N> &source) {
    for (size_t i = 0; i < N; ++i) target[i] = float(source[i]);
}
template<size_t N> void ToSplit(float (&hi)[N], float (&lo)[N],
                                const std::array<double, N> &source) {
    for (size_t i = 0; i < N; ++i) {
        hi[i] = float(source[i]);
        lo[i] = float(source[i] - double(hi[i]));
    }
}
void CheckFloat(double value) {
    if (!std::isfinite(float(value))) throw std::invalid_argument("Ji Metal input exceeds finite FP32 range");
}
template<size_t N> void CheckFloat(const std::array<double, N> &values) {
    for (double value : values) CheckFloat(value);
}
template<size_t N> void CheckFinite(const float (&values)[N]) {
    for (float value : values)
        if (!std::isfinite(value)) throw std::runtime_error("Ji Metal produced nonfinite state");
}

} // namespace

MetalPasses::MetalPasses(const rbp::mtl::Context &context) {
    constexpr rbp::shaders::Pass passes[]{
        rbp::shaders::JiPrepareBodyMatrixPass,
        rbp::shaders::JiInitializePenaltyPass, rbp::shaders::JiFactorPass,
        rbp::shaders::JiAssemblePass,
        rbp::shaders::JiBodyPass, rbp::shaders::JiConstraintPass,
        rbp::shaders::JiBodyResidualPass, rbp::shaders::JiReduceResidualPass,
        rbp::shaders::JiCheckPass,
        rbp::shaders::JiPenaltyPass,
        rbp::shaders::JiSnapshotInitialPass};
    for (uint32_t i = 0; i < Pipelines.size(); ++i)
        Pipelines[i] = context.Pipeline(rbp::shaders::PipelineIndices[0][passes[i]][0]);
    if (Pipelines[7]->maxTotalThreadsPerThreadgroup() < 64)
        throw std::runtime_error("Ji residual reduction requires 64 threads per group");
}

void MetalPasses::Encode(MTL4::ComputeCommandEncoder *encoder, MTL4::ArgumentTable *table, Settings settings,
                         uint32_t bodies, uint32_t constraint_capacity, bool initialize_penalties,
                         MTL4::CounterHeap *profile_heap, MTL4::ArgumentTable *snapshot_table) const {
    encoder->setArgumentTable(table);
    auto dispatch = [&](uint32_t pass, uint32_t count) {
        if (!count) return;
        auto *pipeline = Pipelines[pass].get();
        encoder->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
        encoder->setComputePipelineState(pipeline);
        const uint32_t width = std::min(64u, uint32_t(pipeline->maxTotalThreadsPerThreadgroup()));
        encoder->dispatchThreadgroups({(count + width - 1) / width, 1, 1}, {width, 1, 1});
    };
    dispatch(0, bodies);
    if (initialize_penalties) dispatch(1, bodies);
    dispatch(2, bodies);
    dispatch(3, constraint_capacity);
    if (snapshot_table) {
        const uint64_t words = uint64_t(bodies) * sizeof(JiBodyData) / sizeof(uint32_t) +
            uint64_t((constraint_capacity + JI_CONSTRAINT_LANES - 1) / JI_CONSTRAINT_LANES) *
                sizeof(JiConstraintTile) / sizeof(uint32_t);
        if (words > std::numeric_limits<uint32_t>::max())
            throw std::length_error("Ji initial snapshot exceeds dispatch range");
        encoder->setArgumentTable(snapshot_table);
        dispatch(10, uint32_t(words));
        encoder->setArgumentTable(table);
    }
    const uint32_t samples = profile_heap ? std::min(5u, settings.iterations) : 0;
    uint32_t sampled = 0;
    for (uint32_t i = 0; i < settings.iterations; ++i) {
        const bool capture = sampled < samples && i ==
            (samples == 1 ? 0 : sampled * (settings.iterations - 1) / (samples - 1));
        const auto stamp = [&](uint32_t phase) {
            if (capture)
                encoder->writeTimestamp(MTL4::TimestampGranularityPrecise,
                                        profile_heap, 6 + 3 * sampled + phase);
        };
        stamp(0);
        dispatch(4, bodies);
        stamp(1);
        dispatch(5, constraint_capacity);
        stamp(2);
        if (capture) ++sampled;
        const bool penalty_update = i + 1 < settings.iterations && settings.penalty_period &&
            (i + 1) % settings.penalty_period == 0;
        if (settings.tolerance > 0 || penalty_update || i + 1 == settings.iterations) {
            dispatch(6, bodies);
            dispatch(7, std::max(bodies, constraint_capacity));
            dispatch(8, 1);
        }
        if (penalty_update) {
            dispatch(9, bodies);
            dispatch(2, bodies);
            dispatch(3, constraint_capacity);
        }
    }
}

GpuResult DecodeGpu(System system, std::span<const JiBodyData> bodies,
                    std::span<const JiConstraintTile> tiles, uint32_t constraint_count,
                    JiStatus status) {
    if (!std::isfinite(status.primal) || !std::isfinite(status.dual) ||
        !std::isfinite(status.dynamics) || !std::isfinite(status.contact_law))
        throw std::runtime_error("Ji Metal produced nonfinite residual");
    // Frozen runs preserve their original double coefficients for independent audits.
    const bool full = system.bodies.empty() && system.constraints.empty();
    if (full) {
        system.bodies.resize(bodies.size());
        system.constraints.resize(constraint_count);
    } else if (system.bodies.size() != bodies.size() || system.constraints.size() != constraint_count) {
        throw std::invalid_argument("Ji Metal result layout does not match its input");
    }
    if (tiles.size() < (constraint_count + JI_CONSTRAINT_LANES - 1) / JI_CONSTRAINT_LANES)
        throw std::invalid_argument("Ji Metal result has too few constraint tiles");
    for (uint32_t i = 0; i < bodies.size(); ++i) {
        const auto &src = bodies[i];
        CheckFinite(src.v);
        CheckFinite(src.v_lo);
        if (!(src.beta > 0) || !std::isfinite(src.beta))
            throw std::runtime_error("Ji Metal produced invalid penalty");
        Body &dst = system.bodies[i];
        if (full) {
            CheckFinite(src.a);
            CheckFinite(src.b);
            std::copy_n(src.a, 6, dst.a.begin());
            std::copy_n(src.b, 6, dst.b.begin());
        }
        for (uint32_t k = 0; k < 6; ++k) dst.v[k] = double(src.v[k]) + double(src.v_lo[k]);
        dst.beta = src.beta;
    }
    for (uint32_t i = 0; i < constraint_count; ++i) {
        const auto &src = tiles[i / JI_CONSTRAINT_LANES];
        const uint32_t lane = i % JI_CONSTRAINT_LANES;
        const auto read = [lane](const auto &values) {
            const double value = values[lane];
            if (!std::isfinite(value)) throw std::runtime_error("Ji Metal produced nonfinite state");
            return value;
        };
        Constraint &dst = system.constraints[i];
        if (full) {
            if (src.kind[lane] > uint32_t(Kind::Contact) || !src.rows[lane] || src.rows[lane] > 6)
                throw std::runtime_error("Ji Metal produced invalid constraint kind or row count");
            dst.kind = Kind(src.kind[lane]);
            dst.rows = src.rows[lane];
            dst.stiffness = read(src.stiffness);
            dst.damping = read(src.damping);
            dst.friction = read(src.friction);
        }
        for (uint32_t k = 0; k < 6; ++k) {
            dst.impulse[k] = read(src.impulse[k]);
            if (full) {
                dst.error[k] = read(src.error[k]);
                dst.velocity_bias[k] = read(src.velocity_bias[k]);
            }
        }
        for (uint32_t side = 0; side < 2; ++side) {
            const auto &from = src.side[side];
            auto &to = dst.side[side];
            if (full) {
                to.body = from.body[lane];
                for (uint32_t k = 0; k < 36; ++k) to.j[k] = read(from.j[k]);
            }
            for (uint32_t k = 0; k < 6; ++k) {
                to.x[k] = read(from.x[k]) + read(from.x_lo[k]);
                to.z[k] = read(from.z[k]) + read(from.z_lo[k]);
                to.u[k] = read(from.u[k]) + read(from.u_lo[k]);
            }
        }
    }
    return {std::move(system), {status.primal, status.dual, status.dynamics, status.contact_law},
            status.iterations, !status.active};
}

GpuResult RunGpu(const rbp::mtl::Context &context, System system, Settings settings, bool initialize_penalties) {
    if (system.constraints.size() > std::numeric_limits<uint32_t>::max() / 2)
        throw std::length_error("Ji Metal has too many constraint sides");
    const uint32_t bodies = uint32_t(system.bodies.size()), constraints = uint32_t(system.constraints.size());
    if (!bodies || !settings.iterations) throw std::invalid_argument("Ji Metal run needs bodies and iterations");
    if (!(settings.alpha >= 1) || !std::isfinite(settings.alpha) ||
        !(settings.tolerance >= 0) || !std::isfinite(settings.tolerance))
        throw std::invalid_argument("Invalid Ji Metal solve settings");
    ValidateSystem(system, !initialize_penalties);
    CheckFloat(settings.alpha);
    CheckFloat(settings.tolerance);
    for (const Body &body : system.bodies) {
        CheckFloat(body.a);
        CheckFloat(body.b);
        CheckFloat(body.v);
        if (!initialize_penalties) CheckFloat(body.beta);
        for (double a : body.a)
            if (!(float(a) > 0)) throw std::invalid_argument("Ji Metal body inertia underflows FP32");
        if (!initialize_penalties && !(float(body.beta) > 0))
            throw std::invalid_argument("Ji Metal penalty underflows FP32");
    }
    for (const Constraint &constraint : system.constraints) {
        CheckFloat(constraint.error);
        CheckFloat(constraint.velocity_bias);
        CheckFloat(constraint.impulse);
        CheckFloat(constraint.stiffness);
        CheckFloat(constraint.damping);
        CheckFloat(constraint.friction);
        for (const Side &side : constraint.side) {
            CheckFloat(side.j);
            CheckFloat(side.x);
            CheckFloat(side.z);
            CheckFloat(side.u);
        }
    }
    rbp::mtl::Buffer<JiBodyData> body_buffer{context.Device.get(), bodies};
    const uint32_t tile_count = std::max(1u, (constraints + JI_CONSTRAINT_LANES - 1) / JI_CONSTRAINT_LANES);
    rbp::mtl::Buffer<JiConstraintTile> constraint_buffer{context.Device.get(), tile_count};
    rbp::mtl::Buffer<JiResidualData> body_residuals{context.Device.get(), bodies};
    rbp::mtl::Buffer<JiResidualData> constraint_residuals{context.Device.get(), std::max(1u, constraints)};
    rbp::mtl::Buffer<JiResidualData> partial_residuals{
        context.Device.get(), (std::max(bodies, constraints) + 63) / 64};
    rbp::mtl::Buffer<uint32_t> offsets{context.Device.get(), bodies + 1};
    rbp::mtl::Buffer<uint32_t> incidence{context.Device.get(), std::max(1u, 2 * constraints)};
    rbp::mtl::Buffer<uint32_t> contribution_slots{context.Device.get(), std::max(1u, 2 * constraints)};
    rbp::mtl::Buffer<JiContributionData> contributions{context.Device.get(), std::max(1u, 2 * constraints)};
    rbp::mtl::Buffer<float> body_matrix{context.Device.get(), 42 * bodies};
    rbp::mtl::Buffer<float> body_inverse{context.Device.get(), 42 * bodies};
    rbp::mtl::Buffer<JiParams> parameters{context.Device.get(), 1};
    rbp::mtl::Buffer<JiStatus> status{context.Device.get(), 1};
    parameters[0] = {bodies, constraints, settings.penalty_period, settings.iterations,
                     16, 1e5f, float(settings.alpha), float(settings.tolerance)};
    status[0] = {1, 0, 0, 0, 0};
    for (uint32_t i = 0; i < bodies; ++i) {
        JiBodyData &dst = body_buffer[i];
        ToFloat(dst.a, system.bodies[i].a);
        ToFloat(dst.b, system.bodies[i].b);
        ToSplit(dst.v, dst.v_lo, system.bodies[i].v);
        dst.beta = float(system.bodies[i].beta);
    }
    std::fill(constraint_buffer.All().begin(), constraint_buffer.All().end(), JiConstraintTile{});
    for (uint32_t i = 0; i < constraints; ++i) {
        const auto &src = system.constraints[i];
        auto &dst = constraint_buffer[i / JI_CONSTRAINT_LANES];
        const uint32_t lane = i % JI_CONSTRAINT_LANES;
        dst.kind[lane] = uint32_t(src.kind);
        dst.rows[lane] = src.rows;
        dst.stiffness[lane] = float(src.stiffness);
        dst.damping[lane] = float(src.damping);
        dst.friction[lane] = float(src.friction);
        for (uint32_t k = 0; k < 6; ++k) {
            dst.error[k][lane] = float(src.error[k]);
            dst.velocity_bias[k][lane] = float(src.velocity_bias[k]);
            dst.impulse[k][lane] = float(src.impulse[k]);
        }
        for (uint32_t side = 0; side < 2; ++side) {
            const auto &from = src.side[side];
            auto &to = dst.side[side];
            to.body[lane] = from.body;
            for (uint32_t k = 0; k < 36; ++k) to.j[k][lane] = float(from.j[k]);
            for (uint32_t k = 0; k < 6; ++k) {
                to.x[k][lane] = float(from.x[k]);
                to.z[k][lane] = float(from.z[k]);
                to.u[k][lane] = float(from.u[k]);
                to.x_lo[k][lane] = float(from.x[k] - double(to.x[k][lane]));
                to.z_lo[k][lane] = float(from.z[k] - double(to.z[k][lane]));
                to.u_lo[k][lane] = float(from.u[k] - double(to.u[k][lane]));
            }
        }
    }
    std::vector<uint32_t> counts(bodies);
    for (uint32_t c = 0; c < constraints; ++c)
        for (uint32_t s = 0; s < 2; ++s)
            if (system.constraints[c].side[s].body != NoBody)
                ++counts[system.constraints[c].side[s].body];
    offsets[0] = 0;
    for (uint32_t i = 0; i < bodies; ++i) offsets[i + 1] = offsets[i] + counts[i];
    std::vector<uint32_t> cursor(bodies);
    for (uint32_t i = 0; i < bodies; ++i) cursor[i] = offsets[i];
    for (uint32_t c = 0; c < constraints; ++c)
        for (uint32_t s = 0; s < 2; ++s) {
            const uint32_t id = system.constraints[c].side[s].body;
            if (id != NoBody) {
                const uint32_t at = cursor[id]++;
                incidence[at] = 2 * c + s;
                contribution_slots[2 * c + s] = at;
            }
        }

    NS::Error *error{};
    auto descriptor = rbp::mtl::Make<MTL4::ArgumentTableDescriptor>();
    descriptor->setMaxBufferBindCount(13);
    auto table = NS::TransferPtr(context.Device->newArgumentTable(descriptor.get(), &error));
    if (!table) throw std::runtime_error("Cannot create Ji Metal argument table");
    auto residency = NS::TransferPtr(context.Device->newResidencySet(rbp::mtl::Make<MTL::ResidencySetDescriptor>().get(), &error));
    if (!residency) throw std::runtime_error("Cannot create Ji Metal residency set");
    const MTL::Buffer *bindings[]{body_buffer.Handle.get(), constraint_buffer.Handle.get(),
        body_residuals.Handle.get(), constraint_residuals.Handle.get(), parameters.Handle.get(), status.Handle.get(),
        offsets.Handle.get(), incidence.Handle.get(), contributions.Handle.get(), body_matrix.Handle.get(),
        body_inverse.Handle.get(), partial_residuals.Handle.get(), contribution_slots.Handle.get()};
    for (uint32_t slot = 0; slot < std::size(bindings); ++slot) {
        table->setAddress(bindings[slot]->gpuAddress(), slot);
        residency->addAllocation(bindings[slot]);
    }
    residency->commit();
    residency->requestResidency();
    context.Queue->addResidencySet(residency.get());

    auto allocator = NS::TransferPtr(context.Device->newCommandAllocator());
    auto commands = NS::TransferPtr(context.Device->newCommandBuffer());
    MetalPasses passes{context};
    commands->beginCommandBuffer(allocator.get());
    auto *encoder = commands->computeCommandEncoder();
    passes.Encode(encoder, table.get(), settings, bodies, constraints, initialize_penalties);
    encoder->endEncoding();
    commands->endCommandBuffer();
    std::binary_semaphore done{0};
    std::string gpu_error;
    auto options = rbp::mtl::Make<MTL4::CommitOptions>();
    options->addFeedbackHandler([&](MTL4::CommitFeedback *feedback) {
        if (NS::Error *error = feedback->error()) gpu_error = error->localizedDescription()->utf8String();
        done.release();
    });
    const MTL4::CommandBuffer *list[]{commands.get()};
    context.Queue->commit(list, 1, options.get());
    done.acquire();
    context.Queue->removeResidencySet(residency.get());
    if (!gpu_error.empty()) throw std::runtime_error("Ji Metal command failed: " + gpu_error);
    if (status[0].iterations != settings.iterations && status[0].active)
        throw std::runtime_error("Ji Metal run ended without completion or convergence");
    return DecodeGpu(std::move(system), body_buffer.All().first(bodies),
                     constraint_buffer.All().first(tile_count), constraints, status[0]);
}

} // namespace ji
