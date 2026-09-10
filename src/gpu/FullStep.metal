kernel void FullStep(constant FullStepData &s [[buffer(0)]], uint tid [[thread_index_in_threadgroup]], uint wave [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint3 group_size [[threads_per_threadgroup]]) {
    StepParams p = s.params[0];
    const uint threads = group_size.x, waves = threads / SolveLanes;
    threadgroup uint max_colors;
    if (s.gpu_colors) {
        if (wave == 0) {
            const uint mine = lane < p.BodyCount && Moves(s.masses[lane]) ? ColorOf(s.colors[lane]) + 1 : 1;
            const uint count = min(p.MaxColors, simd_max(mine) + 1);
            if (lane == 0) max_colors = count;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        p.MaxColors = max_colors;
    }
    const StepOutputFlags output{uint(s.output_poses), uint(s.output_sensors)};
    threadgroup uint changed_waves[SmallSolveWaves];
    threadgroup uint body_ids[MaxSupportedColors], offsets[MaxSupportedColors + 1];
    threadgroup GeometryManifold geometries[SmallSolveWaves];
    threadgroup ContactHistory history[SmallSolveWaves];
    threadgroup bool measure_cached[SmallSolveWaves];
    solid::BuildBodyBounds(s.poses, s.body_shapes, s.shapes, s.bounds, s.initial, s.inertial, s.velocities, s.masses, s.incoming, s.quiet, s.compound_children, s.hull_vertices, s.bvh_nodes, p, tid);
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (tid < p.BodyCount) {
        broad::InitTreeLeaf(s.nodes, s.bounds[tid], p.BodyCount, tid);
        s.nodes[tid].Candidates = broad::CandidateMask(s.bounds, p.BodyCount, tid);
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    for (uint body = wave; body < p.BodyCount; body += waves) {
        solid::CollectContacts(s.islands, s.contacts, s.poses, s.masses, s.body_shapes, s.shapes, s.materials, s.compound_children, s.velocities, s.filters, s.jointed_to, s.contact_events, s.contact_event_counts, s.contact_refusals, s.hull_vertices, s.mesh_triangles, s.bvh_nodes, s.hull_faces, s.quiet, s.nodes, s.incoming, p, body, lane, &geometries[wave], history[wave], measure_cached[wave]);
        simdgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (wave == 0) solid::PrepareSmallWorld(s.poses, s.initial, s.velocities, s.masses, s.contacts, p, s.previous, s.displacements, s.iterates, s.colors, s.next, s.budgets, s.joints, s.incoming, s.incoming_slots, s.joint_incidence, s.quiet, s.islands, tid);
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (wave == 0) {
        const bool solved = lane < p.BodyCount && solid::Solved(s.masses[lane], s.quiet[lane], p);
        const uint mine = solved ? ColorOf(s.colors[lane]) % p.MaxColors : NoIndex;
        solid::PackBodyColors(lane, mine, p.MaxColors, body_ids, offsets, lane);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint iteration = 0; iteration < s.budgets[0]; ++iteration) {
        bool changed = false;
        for (uint color = 0; color < p.MaxColors; ++color) {
            const uint first = offsets[color];
            for (uint at = wave; at < offsets[color + 1] - first; at += waves) {
                const uint body = body_ids[first + at];
                changed |= solid::SolveBody(s.poses, s.iterates, s.initial, s.inertial, s.masses, s.displacements, s.contacts, s.colors, color, s.joints, s.incoming, s.incoming_slots, s.joint_incidence, s.quiet, p, body, lane);
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        }
        solid::PublishBody(s.poses, s.iterates, s.displacements, s.masses, s.quiet, p, tid);
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        const Adjacency last = s.incoming[p.BodyCount - 1];
        for (uint id = tid; id < last.Start + last.Count; id += threads)
            changed |= solid::UpdateContactDual(s.contacts, s.displacements, s.initial, s.masses, s.quiet, s.incoming, s.incoming_slots, p, id);
        const bool wave_changed = simd_any(changed);
        if (lane == 0) changed_waves[wave] = wave_changed;
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        bool any_changed = false;
        for (uint at = 0; at < waves; ++at) any_changed |= changed_waves[at] != 0;
        if (!any_changed) break;
    }
    solid::Finalize(s.displacements, s.velocities, s.masses, s.quiet, p, tid);
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (s.restitution) {
        for (uint pass = 0; pass < RestitutionPasses; ++pass) {
            for (uint slot = tid; slot < p.BodyCount * ContactsPerBody; slot += threads) {
                solid::Restitution(s.contacts, s.poses, s.velocities, s.masses, s.incoming, s.incoming_slots, p, slot);
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
            solid::ApplyRestitution(s.velocities, s.contacts, s.poses, s.masses, s.incoming, s.incoming_slots, s.quiet, p, tid);
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        }
    }
    for (uint color = 0; color < p.MaxColors; ++color) {
        const uint first = offsets[color];
        for (uint at = wave; at < offsets[color + 1] - first; at += waves) {
            const uint body = body_ids[first + at];
            stabilize::SolveBody(s.poses, s.iterates, s.initial, s.inertial, s.masses, s.displacements, s.contacts, s.colors, color, s.joints, s.incoming, s.incoming_slots, s.joint_incidence, s.quiet, p, body, lane);
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }
    solid::FinishPoses(s.poses, s.velocities, s.displacements, s.iterates, s.masses, s.quiet, s.next_quiet, s.rest, p, tid);
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    solid::FinishWaking(s.next_quiet, s.quiet, s.contacts, s.joints, s.incoming, s.incoming_slots, s.masses, s.velocities, s.joint_incidence, p, tid);
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (s.output_sensors) {
        sensor::BuildBodyBounds(s.poses, s.body_shapes, s.shapes, s.bounds, s.velocities, s.compound_children, s.hull_vertices, s.bvh_nodes, p, tid);
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        if (tid < p.BodyCount) {
            broad::InitTreeLeaf(s.nodes, s.bounds[tid], p.BodyCount, tid);
            s.nodes[tid].Candidates = broad::CandidateMask(s.bounds, p.BodyCount, tid);
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        for (uint body = wave; body < p.BodyCount; body += waves) {
            sensor::CollectContacts(s.islands, s.sensors, s.poses, s.masses, s.body_shapes, s.shapes, s.materials, s.compound_children, s.velocities, s.filters, s.jointed_to, s.contact_events, s.contact_event_counts, s.sensor_refusals, s.hull_vertices, s.mesh_triangles, s.bvh_nodes, s.hull_faces, s.quiet, s.nodes, s.incoming, p, body, lane, &geometries[wave], history[wave], measure_cached[wave]);
            simdgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }
    if (s.snapshot) {
        solid::CaptureStep(s.poses, s.initial, s.velocities, s.contacts, s.sensors, p, s.out_contacts, s.contact_events, s.contact_event_counts, s.out_initial, s.nodes, s.sensor_refusals, s.out_poses, s.out_velocities, s.out_sensors, s.out_counts, s.completion, output, s.out_removed, s.contact_refusals, tid);
    }
}
