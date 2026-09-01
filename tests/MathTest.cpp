// The quaternion helpers in the shared header, checked against rotations worked out by hand.
// Y-up right-handed, so a positive rotation about +Y takes +X to -Z.

#include "gpu/Shared.h"

#include <cmath>

#include <doctest/doctest.h>

using namespace rbp;

namespace {
constexpr float4 Identity{0, 0, 0, 1};

float4 AboutY(float radians) { return {0, std::sin(radians / 2), 0, std::cos(radians / 2)}; }

void CheckNear(float3 actual, float3 expected) {
    for (uint32_t axis = 0; axis < 3; ++axis) CHECK(actual[axis] == doctest::Approx(expected[axis]).epsilon(1e-6));
}
} // namespace

TEST_CASE("the identity quaternion rotates nothing") {
    CheckNear(Rotate(Identity, float3{1, 2, 3}), float3{1, 2, 3});
}

TEST_CASE("a quarter turn about Y takes +X to -Z") {
    CheckNear(Rotate(AboutY(M_PI_2), float3{1, 0, 0}), float3{0, 0, -1});
    CheckNear(Rotate(AboutY(M_PI_2), float3{0, 1, 0}), float3{0, 1, 0}); // the axis of rotation is fixed
}

TEST_CASE("multiplying quaternions composes their rotations") {
    const auto quarter = AboutY(M_PI_2);
    CheckNear(Rotate(QuatMul(quarter, quarter), float3{1, 0, 0}), float3{-1, 0, 0});
    CheckNear(Rotate(QuatMul(quarter, Identity), float3{1, 0, 0}), Rotate(quarter, float3{1, 0, 0}));
}

TEST_CASE("rotation preserves length and the angle between vectors") {
    const auto q = AboutY(0.7f);
    const float3 a{1, 2, 3}, b{-2, 0.5f, 1};
    CHECK(simd::length(Rotate(q, a)) == doctest::Approx(simd::length(a)));
    CHECK(dot(Rotate(q, a), Rotate(q, b)) == doctest::Approx(dot(a, b)));
}
