
#include <Jolt/Jolt.h>

#include <Jolt/Core/Factory.h>
#include <Jolt/Core/JobSystemSingleThreaded.h>
#include <Jolt/Core/JobSystemThreadPool.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Physics/Body/Body.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/MeshShape.h>
#include <Jolt/Physics/Collision/Shape/PlaneShape.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/RegisterTypes.h>

#include "../BenchGeometry.h"
#include "../BenchQuality.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string_view>
#include <vector>
using namespace JPH;
struct Layers final : BroadPhaseLayerInterface {
    uint GetNumBroadPhaseLayers() const override { return 2; }
    BroadPhaseLayer GetBroadPhaseLayer(ObjectLayer v) const override { return BroadPhaseLayer(v); }
};
struct PairFilter final : ObjectLayerPairFilter {
    bool ShouldCollide(ObjectLayer a, ObjectLayer b) const override { return a || b; }
};
struct BroadFilter final : ObjectVsBroadPhaseLayerFilter {
    bool ShouldCollide(ObjectLayer a, BroadPhaseLayer b) const override { return a || b.GetValue(); }
};
void TraceLog(const char *format, ...) {
    va_list args;
    va_start(args, format);
    std::vfprintf(stderr, format, args);
    va_end(args);
    std::fputc('\n', stderr);
}
int main(int argc, char **argv) {
    RegisterDefaultAllocator();
    Trace = TraceLog;
    Factory::sInstance = new Factory;
    RegisterTypes();
    const int workers = argc > 1 ? std::atoi(argv[1]) : 0;
    const int velocity_steps = argc > 2 ? std::atoi(argv[2]) : 10;
    const int position_steps = argc > 3 ? std::atoi(argv[3]) : 2;
    const bool pair_cache = !std::getenv("PAIR_CACHE") || std::atoi(std::getenv("PAIR_CACHE")) != 0;
    const int warmup = std::getenv("WARMUP") ? std::atoi(std::getenv("WARMUP")) : 120;
    const int timed = std::getenv("STEPS") ? std::atoi(std::getenv("STEPS")) : 120;
    const int substeps = std::getenv("SUBSTEPS") ? std::atoi(std::getenv("SUBSTEPS")) : 1;
    if (workers < 0 || velocity_steps <= 0 || position_steps <= 0 || warmup < 0 || timed <= 0 || substeps <= 0) return 1;
    std::unique_ptr<JobSystem> jobs;
    if (workers) jobs = std::make_unique<JobSystemThreadPool>(cMaxPhysicsJobs, cMaxPhysicsBarriers, workers);
    else jobs = std::make_unique<JobSystemSingleThreaded>(cMaxPhysicsJobs);
    TempAllocatorImpl alloc(512 * 1024 * 1024);
    Layers layers;
    PairFilter pair;
    BroadFilter broad;
    struct Scene {
        const char *name;
        int across, deep, high;
        float half_width = 0;
    };
    std::printf("Jolt, workers=%d, dt=1/60, velocity iterations=%d, position iterations=%d, sleep off, damping zero, slop=0.0005, pair_cache=%d, warmup=%d updates, timed=%d updates, substeps/update=%d, ms/substep\n", workers, velocity_steps, position_steps, int(pair_cache), warmup, timed, substeps);
    bool ran = false;
    for (auto scene : {Scene{"floor", 1, 1, 1}, Scene{"stack20", 1, 1, 20}, Scene{"lattice600", 10, 10, 6}, Scene{"lattice6000", 25, 40, 6}, Scene{"lattice96000", 100, 160, 6}, Scene{"lattice192000", 200, 160, 6}, Scene{"slab", 1, 1, 1, 2}, Scene{"slab2m", 1, 1, 1, 1}, Scene{"slab1m", 1, 1, 1, 0.5f}, Scene{"slabs25", 5, 5, 1, 0.5f}}) {
        const bool mesh = scene.half_width > 0;
        const std::array<float, 3> half = mesh ? std::array<float, 3>{scene.half_width, 0.25f, scene.half_width} : std::array<float, 3>{0.5f, 0.5f, 0.5f};
        const uint32_t bodies = uint32_t(scene.across * scene.deep * scene.high) + 1;
        bool selected = argc <= 4 && bodies <= 6001;
        for (int i = 4; i < argc; ++i) selected |= std::string_view(argv[i]) == scene.name;
        if (!selected) continue;
        ran = true;
        PhysicsSystem world;
        world.Init(std::max(10000u, bodies), 0, std::max(100000u, bodies * 4), std::max(100000u, bodies * 2), layers, broad, pair);
        PhysicsSettings ps;
        ps.mUseBodyPairContactCache = pair_cache;
        ps.mNumVelocitySteps = velocity_steps;
        ps.mNumPositionSteps = position_steps;
        ps.mPenetrationSlop = 0.0005f;
        world.SetPhysicsSettings(ps);
        world.SetGravity(Vec3(0, -9.81f, 0));
        auto &bi = world.GetBodyInterface();
        RefConst<Shape> ground = new PlaneShape(Plane(Vec3::sAxisY(), 0));
        if (mesh) {
            VertexList vertices;
            IndexedTriangleList triangles;
            benchmark::FloorGrid(64, [&](const auto &point) { vertices.emplace_back(point[0], point[1], point[2]); }, [&](const auto &triangle) { triangles.emplace_back(triangle[0], triangle[1], triangle[2]); });
            const auto cooked = MeshShapeSettings(std::move(vertices), std::move(triangles)).Create();
            if (cooked.HasError()) return 2;
            ground = cooked.Get();
        }
        BodyCreationSettings gs(ground, RVec3::sZero(), Quat::sIdentity(), EMotionType::Static, 0);
        gs.mFriction = 0.5f;
        if (bi.CreateAndAddBody(gs, EActivation::DontActivate).IsInvalid()) return 2;
        RefConst<Shape> box = new BoxShape(Vec3(half[0], half[1], half[2]), 0);
        std::vector<BodyID> ids;
        std::vector<std::array<float, 2>> origins;
        for (int x = 0; x < scene.across; ++x)
            for (int z = 0; z < scene.deep; ++z)
                for (int y = 0; y < scene.high; ++y) {
                    const float px = mesh ? (scene.across == 1 ? 0.f : 3.f * x - 6) : 1.5f * x;
                    const float pz = mesh ? (scene.deep == 1 ? 0.f : 3.f * z - 6) : 1.5f * z;
                    BodyCreationSettings bs(box, RVec3(px, half[1] + 1.02f * y, pz), Quat::sIdentity(), EMotionType::Dynamic, 1);
                    bs.mFriction = 0.5f;
                    bs.mRestitution = 0;
                    bs.mAllowSleeping = false;
                    bs.mLinearDamping = 0;
                    bs.mAngularDamping = 0;
                    bs.mOverrideMassProperties = EOverrideMassProperties::CalculateInertia;
                    bs.mMassPropertiesOverride.mMass = 1000 * 8 * half[0] * half[1] * half[2];
                    const auto body = bi.CreateBody(bs);
                    if (!body) return 2;
                    ids.push_back(body->GetID());
                    origins.push_back({px, pz});
                }

        auto added = ids;
        const auto add_state = bi.AddBodiesPrepare(added.data(), int(added.size()));
        bi.AddBodiesFinalize(added.data(), int(added.size()), add_state, EActivation::Activate);
        world.OptimizeBroadPhase();
        std::vector<double> times;
        benchmark::Quality quality;
        for (int step = 0; step < warmup + timed; ++step) {
            auto begin = std::chrono::steady_clock::now();
            auto result = world.Update(substeps * (1.f / 60), substeps, &alloc, jobs.get());
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count() / substeps;
            if (result != EPhysicsUpdateError::None) return 2;
            if (step >= warmup) times.push_back(ms);
            for (size_t i = 0; i < ids.size(); ++i) {
                const auto id = ids[i];
                const auto p = bi.GetPosition(id);
                const auto q = bi.GetRotation(id);
                const auto v = bi.GetLinearVelocity(id), w = bi.GetAngularVelocity(id);
                quality.Observe({float(p.GetX()), float(p.GetY()), float(p.GetZ()), q.GetX(), q.GetY(), q.GetZ(), q.GetW(), v.GetX(), v.GetY(), v.GetZ(), w.GetX(), w.GetY(), w.GetZ()}, origins[i], int(i % scene.high), half);
            }
        }
        std::sort(times.begin(), times.end());
        std::printf("%s bodies=%zu p50=%.6f p90=%.6f finite=%d quaternion_norm_error=%.8f floor_penetration=%.8f vertical_overlap=%.8f horizontal_drift=%.8f m\n", scene.name, ids.size() + 1, times[size_t(std::llround(0.5 * (timed - 1)))], times[size_t(std::llround(0.9 * (timed - 1)))], quality.Finite, quality.QuaternionError, quality.FloorPenetration, quality.VerticalOverlap, quality.HorizontalDrift);
        if (!quality.Finite) return 3;
    }
    UnregisterTypes();
    delete Factory::sInstance;
    Factory::sInstance = nullptr;
    return ran ? 0 : 1;
}
