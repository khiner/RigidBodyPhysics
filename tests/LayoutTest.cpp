
#include "Gpu.h"
#include "GpuSource.h"
#include "gpu/Shared.h"

#include <algorithm>

#include <doctest/doctest.h>

using namespace rbp;

namespace {
struct Layout {
    const char *Name;
    uint32_t Size, Align;
};

// Same order as Layout.metal's REPORT list.
constexpr Layout HostLayouts[]{
    {"Pose", sizeof(Pose), alignof(Pose)},
    {"Velocity", sizeof(Velocity), alignof(Velocity)},
    {"Displacement", sizeof(Displacement), alignof(Displacement)},
    {"BodyIterate", sizeof(BodyIterate), alignof(BodyIterate)},
    {"BodyMass", sizeof(BodyMass), alignof(BodyMass)},
    {"BodyBounds", sizeof(BodyBounds), alignof(BodyBounds)},
    {"BroadPhaseNode", sizeof(BroadPhaseNode), alignof(BroadPhaseNode)},
    {"MortonKey", sizeof(MortonKey), alignof(MortonKey)},
    {"Shape", sizeof(Shape), alignof(Shape)},
    {"Triangle", sizeof(Triangle), alignof(Triangle)},
    {"Material", sizeof(Material), alignof(Material)},
    {"JointDrive", sizeof(JointDrive), alignof(JointDrive)},
    {"Filter", sizeof(Filter), alignof(Filter)},
    {"CollisionMask", sizeof(CollisionMask), alignof(CollisionMask)},
    {"Contact", sizeof(Contact), alignof(Contact)},
    {"Joint", sizeof(Joint), alignof(Joint)},
    {"Adjacency", sizeof(Adjacency), alignof(Adjacency)},
    {"ContactEvent", sizeof(ContactEvent), alignof(ContactEvent)},
    {"StepParams", sizeof(StepParams), alignof(StepParams)},
    {"SensorFollower", sizeof(SensorFollower), alignof(SensorFollower)},
    {"ContactReport", sizeof(ContactReport), alignof(ContactReport)},
    {"SensorPair", sizeof(SensorPair), alignof(SensorPair)},
    {"StepCounts", sizeof(StepCounts), alignof(StepCounts)},
    {"StepCompletion", sizeof(StepCompletion), alignof(StepCompletion)},
    {"StepOutputFlags", sizeof(StepOutputFlags), alignof(StepOutputFlags)},
};
} // namespace

TEST_CASE("shared structs have the same layout on host and device") {
    const mtl::Context context;
    auto pipeline = context.Pipeline(gpu::LayoutSource, "ReportLayout");

    constexpr uint32_t Slots = 2 * std::size(HostLayouts);
    const mtl::Buffer<uint32_t> reported{context.Device.get(), Slots};
    std::ranges::fill(reported.All(), 0u);

    RunGpu(context, pipeline.get(), {{0, reported.Handle.get()}}, 1, 1);

    for (uint32_t i = 0; i < std::size(HostLayouts); ++i) {
        const auto &host = HostLayouts[i];
        CAPTURE(host.Name);
        CHECK(reported[2 * i] == host.Size);
        CHECK(reported[2 * i + 1] == host.Align);
    }
}
