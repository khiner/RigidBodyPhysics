#include "Gpu.h"
#include "GpuSource.h"

#include <string>

#include <doctest/doctest.h>

using namespace rbp;

TEST_CASE("joint angular rows match finite differences away from alignment") {
    const mtl::Context context;
    constexpr uint32_t Cases = 4 * 4 * 3 * 3;
    const std::string source = std::string(gpu::SolveSource) + R"(
static float3 MeasuredAngles(float4 q, uint twist) {
    return AngularError(q, twist, twist < 3 ? TwistAngle(q, UnitAxis(twist), 0) : 0);
}
kernel void CheckDerivatives(device float *out [[buffer(0)]], uint index [[thread_position_in_grid]]) {
    const float3 turns[]{float3(0), float3(0.0001f,-0.0002f,0.0003f), float3(0.4f,0.7f,-0.2f), float3(2.3f,0.5f,-0.3f)};
    const uint direction = index % 3, row = index / 3 % 3, twist = index / 9 % 4;
    const float4 q = QuatFromRotationVector(turns[index / 36]);
    Joint joint{};
    for (uint i=0;i<3;++i) joint.AngularModes |= uint(i == twist ? AxisFree : AxisLocked) << (3*i);
    const JointMeasure measured{float4(0,0,0,1), float3(0), MeasuredAngles(q,twist), float3(0), q};
    const AxisSetup setup = JointRowAt(joint, measured, 3 + row);
    const float epsilon = 0.001f;
    const float3 delta = epsilon * UnitAxis(direction);
    const float plus = MeasuredAngles(QuatMul(QuatFromRotationVector(delta),q),twist)[row];
    const float minus = MeasuredAngles(QuatMul(QuatFromRotationVector(-delta),q),twist)[row];
    out[2*index] = setup.Axis[direction];
    out[2*index+1] = (plus-minus)/(2*epsilon);
}
)";
    auto pipeline = context.Pipeline(source, "CheckDerivatives");
    const mtl::Buffer<float> reported{context.Device.get(), 2 * Cases};
    RunGpu(context, pipeline.get(), {{0, reported.Handle.get()}}, Cases, 32, false);
    for (uint32_t i = 0; i < Cases; ++i) {
        CAPTURE(i);
        CHECK(reported[2 * i] == doctest::Approx(reported[2 * i + 1]).epsilon(0.001));
    }
}
