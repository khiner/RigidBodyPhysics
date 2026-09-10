#ifndef RBP_FULL_STEP_DATA_H
#define RBP_FULL_STEP_DATA_H
#ifndef __METAL_VERSION__
#include "gpu/Shared.h"
#endif
#ifdef __METAL_VERSION__
#define GPU_ADDRESS(Type) device Type *
#else
#define GPU_ADDRESS(Type) uint64_t
#endif
namespace rbp {
struct FullStepData {
    GPU_ADDRESS(Pose)
    poses;
    GPU_ADDRESS(Pose)
    initial;
    GPU_ADDRESS(Pose)
    inertial;
    GPU_ADDRESS(Velocity)
    velocities;
    GPU_ADDRESS(BodyMass)
    masses;
    GPU_ADDRESS(Contact)
    contacts;
    GPU_ADDRESS(Index)
    body_shapes;
    GPU_ADDRESS(Shape)
    shapes;
    GPU_ADDRESS(Velocity)
    previous;
    GPU_ADDRESS(Displacement)
    displacements;
    GPU_ADDRESS(BodyIterate)
    iterates;
    GPU_ADDRESS(uint)
    colors;
    GPU_ADDRESS(uint)
    next;
    GPU_ADDRESS(BodyBounds)
    bounds;
    GPU_ADDRESS(BroadPhaseNode)
    nodes;
    GPU_ADDRESS(Material)
    materials;
    GPU_ADDRESS(Index)
    compound_children;
    GPU_ADDRESS(Joint)
    joints;
    GPU_ADDRESS(Adjacency)
    incoming;
    GPU_ADDRESS(uint)
    incoming_slots;
    GPU_ADDRESS(Filter)
    filters;
    GPU_ADDRESS(Index)
    jointed_to;
    GPU_ADDRESS(uint)
    joint_incidence;
    GPU_ADDRESS(uint)
    quiet;
    GPU_ADDRESS(Pose)
    rest;
    GPU_ADDRESS(uint)
    next_quiet;
    GPU_ADDRESS(ContactEvent)
    contact_events;
    GPU_ADDRESS(uint)
    contact_event_counts;
    GPU_ADDRESS(uint)
    contact_refusals;
    GPU_ADDRESS(float3)
    hull_vertices;
    GPU_ADDRESS(Triangle)
    mesh_triangles;
    GPU_ADDRESS(BvhNode)
    bvh_nodes;
    GPU_ADDRESS(HullFace)
    hull_faces;
    GPU_ADDRESS(Contact)
    sensors;
    GPU_ADDRESS(uint)
    sensor_refusals;
    GPU_ADDRESS(uint)
    budgets;
    GPU_ADDRESS(uint)
    islands;
    GPU_ADDRESS(StepParams)
    params;
    GPU_ADDRESS(ContactReport)
    out_contacts;
    GPU_ADDRESS(Pose)
    out_initial;
    GPU_ADDRESS(Pose)
    out_poses;
    GPU_ADDRESS(Velocity)
    out_velocities;
    GPU_ADDRESS(SensorPair)
    out_sensors;
    GPU_ADDRESS(StepCounts)
    out_counts;
    GPU_ADDRESS(StepCompletion)
    completion;
    GPU_ADDRESS(ContactEvent)
    out_removed;
    ulong snapshot;
    ulong gpu_colors;
    ulong output_poses;
    ulong output_sensors;
    ulong restitution;
};
static_assert(sizeof(FullStepData) == 408);
} // namespace rbp
#undef GPU_ADDRESS
#endif
