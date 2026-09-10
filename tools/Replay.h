#pragma once

#include "Solver.h"

#include <array>
#include <filesystem>
#include <fstream>
#include <tuple>
#include <type_traits>

// Captures use native layouts and require fresh worlds with fixed topology.
// Recapture after input layout or StepSettings changes.
namespace rbp::replay {
inline bool SameSettings(const StepSettings &a, const StepSettings &b) {
    const auto fields = [](const StepSettings &s) {
        // Copy SIMD components because const component references may refer to temporaries.
        return std::tuple(s.Gravity.x, s.Gravity.y, s.Gravity.z, s.DeltaTime, s.Iterations, s.Beta, s.ContactBeta, s.Gamma, s.PenaltyMin, s.PenaltyMax, s.ContactMargin, s.MaxContactReach, s.MaxAngularSpeed, s.BounceSpeedFactor, s.SleepSpeed, s.SleepSteps, s.SleepDrift, s.MaxColors, s.ColoringPasses);
    };
    return fields(a) == fields(b);
}
using State = std::array<float, 13>;
inline State StateOf(Pose p, Velocity v) {
    return {p.Position.x, p.Position.y, p.Position.z, p.Orientation.x, p.Orientation.y, p.Orientation.z, p.Orientation.w, v.Linear.x, v.Linear.y, v.Linear.z, v.Angular.x, v.Angular.y, v.Angular.z};
}
inline State StateOf(const World &w, Index i) { return StateOf(w.Poses[i], w.Velocities[i]); }
inline uint64_t Refusals(const World &w) {
    uint64_t result = 0;
    for (Index i = 0; i < w.BodyCount(); ++i) {
        result += w.ContactRefusals[i];
        if (w.SensorRefusals.Handle) result += w.SensorRefusals[i];
    }
    return result;
}
template<class T> void Write(std::ostream &out, std::span<T> values) {
    static_assert(std::is_trivially_copyable_v<T>);
    out.write(reinterpret_cast<const char *>(values.data()), std::streamsize(values.size_bytes()));
    if (!out) throw std::runtime_error("Cannot write physics replay");
}
template<class T> void Write(std::ostream &out, const T &value) { Write(out, std::span{&value, 1}); }
template<class T> void Read(std::istream &in, std::span<T> values) {
    static_assert(std::is_trivially_copyable_v<T>);
    in.read(reinterpret_cast<char *>(values.data()), std::streamsize(values.size_bytes()));
    if (!in) throw std::runtime_error("Truncated physics replay");
}
template<class T> void Read(std::istream &in, T &value) { Read(in, std::span{&value, 1}); }
inline constexpr auto Layout = std::to_array<uint32_t>({
    0x52504252, 1, sizeof(Pose), sizeof(Velocity), sizeof(BodyMass), sizeof(Shape), sizeof(HullFace),
    sizeof(Triangle), sizeof(BvhNode), sizeof(Material), sizeof(Filter), sizeof(Joint), sizeof(StepSettings), sizeof(SensorFollower)
});
struct Header {
    WorldLimits Limits;
    uint32_t Bodies, Shapes, Joints, Followers, TrackContacts, TrackSensors;
};
struct Writer {
    std::ofstream Out;
    uint32_t Bodies, Shapes, Joints;
    Writer(const std::filesystem::path &path, const World &w, std::span<const SensorFollower> followers)
        : Out(path, std::ios::binary), Bodies(w.BodyCount()), Shapes(w.ShapeCount()), Joints(w.JointCount()) {
        for (Index i = 0; i < Bodies; ++i)
            if (!w.Alive(i) || w.IdOf(i).Spawn != 1 || w.Quiet[i] || w.Colors[i])
                throw std::runtime_error("Physics capture requires a fresh world");
        for (Index i = 0; i < Bodies * ContactsPerBody; ++i)
            if (w.Contacts[i].Active) throw std::runtime_error("Physics capture requires a fresh world");
        const Header header{{w.Poses.Capacity, w.Shapes.Capacity, w.Joints.Capacity, w.ShapeVertices.Capacity, w.HullFaces.Capacity, w.Triangles.Capacity, w.BvhNodes.Capacity, w.CompoundChildren.Capacity}, Bodies, Shapes, Joints, uint32_t(followers.size()), w.TrackContacts, w.TrackSensors};
        Write(Out, Layout);
        Write(Out, header);
        Write(Out, followers);
        Write(Out, w.Shapes.All().first(Shapes));
        Write(Out, w.ShapeVertices.All());
        Write(Out, w.HullFaces.All());
        Write(Out, w.Triangles.All());
        Write(Out, w.BvhNodes.All());
        Write(Out, w.CompoundChildren.All());
        Write(Out, w.Poses.All().first(Bodies));
        Write(Out, w.Velocities.All().first(Bodies));
        Write(Out, w.PreviousVelocities.All().first(Bodies));
        Write(Out, w.RestPoses.All().first(Bodies));
        Write(Out, w.Masses.All().first(Bodies));
        Write(Out, w.BodyShapes.All().first(Bodies));
        Write(Out, w.Materials.All().first(Bodies));
        Write(Out, w.Filters.All().first(Bodies));
        Write(Out, w.Joints.All().first(Joints));
        Write(Out, w.Jointed.All());
    }
    void Step(const World &w, const StepSettings &settings, const StepResult &step) {
        if (w.BodyCount() != Bodies || w.ShapeCount() != Shapes || w.JointCount() != Joints)
            throw std::runtime_error("Physics capture topology changed");
        Write(Out, settings);
        if (step.Poses.size() != Bodies || step.Velocities.size() != Bodies) throw std::runtime_error("Physics capture body count changed");
        Write(Out, step.ContactRefusals + step.SensorRefusals);
        for (Index i = 0; i < Bodies; ++i) Write(Out, StateOf(step.Poses[i], step.Velocities[i]));
        Out.flush();
        if (!Out) throw std::runtime_error("Cannot flush physics replay");
    }
};
struct Reader {
    std::ifstream In;
    Header Info{};
    std::vector<SensorFollower> Followers;
    explicit Reader(const std::filesystem::path &path) : In(path, std::ios::binary) {
        auto layout = Layout;
        Read(In, layout);
        if (layout != Layout) throw std::runtime_error("Physics replay layout changed; recapture fixture");
        Read(In, Info);
        if (Info.Bodies > Info.Limits.Bodies || Info.Shapes > Info.Limits.Shapes || Info.Joints > Info.Limits.Joints || Info.Followers > Info.Bodies)
            throw std::runtime_error("Invalid physics replay counts");
        Followers.resize(Info.Followers);
        Read(In, std::span{Followers});
        for (const auto &f : Followers)
            if (f.Sensor >= Info.Bodies || f.Owner >= Info.Bodies) throw std::runtime_error("Invalid sensor follower");
    }
    void Load(World &w) {
        for (Index i = 0; i < Info.Shapes; ++i) {
            Shape shape{};
            Read(In, shape);
            if (w.AddShape(shape) != i) throw std::runtime_error("Cannot restore replay shape");
        }
        Read(In, w.ShapeVertices.All());
        Read(In, w.HullFaces.All());
        Read(In, w.Triangles.All());
        Read(In, w.BvhNodes.All());
        Read(In, w.CompoundChildren.All());
        BodyDesc body_desc;
        body_desc.Density = 0;
        for (Index i = 0; i < Info.Bodies; ++i)
            if (w.AddBody(body_desc) != i) throw std::runtime_error("Cannot restore replay body");
        Read(In, w.Poses.All().first(Info.Bodies));
        Read(In, w.Velocities.All().first(Info.Bodies));
        Read(In, w.PreviousVelocities.All().first(Info.Bodies));
        Read(In, w.RestPoses.All().first(Info.Bodies));
        Read(In, w.Masses.All().first(Info.Bodies));
        Read(In, w.BodyShapes.All().first(Info.Bodies));
        Read(In, w.Materials.All().first(Info.Bodies));
        Read(In, w.Filters.All().first(Info.Bodies));
        for (Index i = 0; i < Info.Joints; ++i) {
            Joint joint{};
            Read(In, joint);
            JointDesc joint_desc;
            joint_desc.BodyA = joint.BodyA;
            joint_desc.BodyB = joint.BodyB;
            if (!joint.Active || w.AddJoint(joint_desc) != i)
                throw std::runtime_error("Cannot restore replay joint");
            w.Joints[i] = joint;
        }
        Read(In, w.Jointed.All());
        w.TrackContacts = Info.TrackContacts;
        w.TrackSensors = Info.TrackSensors;
    }
    bool Step(StepSettings &settings, uint64_t &refusals, std::vector<State> &states) {
        if (In.peek() == std::char_traits<char>::eof()) return false;
        Read(In, settings);
        Read(In, refusals);
        states.resize(Info.Bodies);
        Read(In, std::span{states});
        return true;
    }
};
} // namespace rbp::replay
