// Same fixed-topology contract and edge threshold as Mesh.cpp::RefitMesh.
// Each phase completes before the next dispatch; publication is one final
// error-gated pass. Input vertices are private caller-owned trial geometry.
static void RefitFailure(device GeometryQueryStatus *status) {
    atomic_fetch_or_explicit((device atomic_uint *)&status->Errors, 4u, memory_order_relaxed);
}
kernel void MeshRefitValidate(
    device GeometryQueryStatus *status [[buffer(1)]],
    device const MeshRefitTriangle *map [[buffer(2)]], device const uint *vertex_map [[buffer(6)]],
    device const MeshRefitNode *node_map [[buffer(8)]], device float3 *normals [[buffer(12)]],
    constant MeshRefitCounts &n [[buffer(13)]],
    device const float3 *vertices [[buffer(27)]], device const Triangle *triangles [[buffer(28)]],
    device const BvhNode *nodes [[buffer(29)]], uint id [[thread_position_in_grid]]) {
    if (id < n.Vertices && !all(isfinite(vertices[vertex_map[id]]))) RefitFailure(status);
    if (id < n.Nodes) {
        const MeshRefitNode at = node_map[id]; const BvhNode current = nodes[at.Index];
        if (current.First != at.First || current.Count != at.Count) RefitFailure(status);
    }
    if (id >= n.Triangles) return;
    const MeshRefitTriangle at = map[id]; const Triangle t = triangles[at.Index];
    normals[id] = float3(0);
    if (t.A != at.Original.A || t.B != at.Original.B || t.C != at.Original.C || t.OwnedEdges != at.Original.OwnedEdges) {
        RefitFailure(status); return;
    }
    const float3 turn = cross(vertices[t.B] - vertices[t.A], vertices[t.C] - vertices[t.A]);
    const float area = length(turn);
    if (!(area > 0) || !isfinite(area)) { RefitFailure(status); return; }
    normals[id] = turn / area;
    if (!all(isfinite(normals[id]))) RefitFailure(status);
}
kernel void MeshRefitLeaves(
    device GeometryQueryStatus *status [[buffer(1)]], device const MeshRefitTriangle *map [[buffer(2)]],
    device const uint *far_vertices [[buffer(4)]], device const MeshRefitNode *node_map [[buffer(8)]],
    device uint2 *flags [[buffer(10)]], device BodyBounds *bounds [[buffer(11)]],
    device const float3 *normals [[buffer(12)]], constant MeshRefitCounts &n [[buffer(13)]],
    device const float3 *vertices [[buffer(27)]], device const Triangle *triangles [[buffer(28)]],
    uint id [[thread_position_in_grid]]) {
    if (atomic_load_explicit((device atomic_uint *)&status->Errors, memory_order_relaxed)) return;
    if (id < n.Triangles) {
        const MeshRefitTriangle at = map[id]; const Triangle t = at.Original;
        const uint corner[3]{t.A,t.B,t.C}; const float3 normal = normals[id]; uint front = 0, back = 0;
        for (uint edge = 0; edge < 3; ++edge) {
            bool active = at.SharedCount[edge] != 2, back_active = active;
            for (uint k = 0; k < at.EdgeCount[edge]; ++k) {
                const float3 offset = vertices[far_vertices[at.EdgeOffset[edge]+k]] - vertices[corner[edge]];
                const float span = length(offset), projection = dot(normal, offset);
                if (!isfinite(span) || !isfinite(projection)) { RefitFailure(status); return; }
                if (span > 0 && projection/span < -MeshActiveEdgeSine) active = true;
                if (span > 0 && projection/span > MeshActiveEdgeSine) back_active = true;
            }
            if (active) front |= 1u << edge;
            if (back_active) back |= 1u << edge;
        }
        flags[id] = uint2(front, back);
    }
    if (id < n.Nodes) {
        const MeshRefitNode at = node_map[id]; if (!at.Count) return;
        BodyBounds result{float3(INFINITY),float3(-INFINITY)};
        for (uint k = 0; k < at.Count; ++k) {
            const Triangle t = triangles[at.FirstTriangle+at.First+k];
            const uint corners[3]{t.A,t.B,t.C};
            for (uint j=0;j<3;++j) { result.Low=min(result.Low,vertices[corners[j]]); result.High=max(result.High,vertices[corners[j]]); }
        }
        bounds[id] = result;
    }
}
kernel void MeshRefitLevel(
    device const GeometryQueryStatus *status [[buffer(1)]], device const MeshRefitNode *map [[buffer(8)]],
    device BodyBounds *bounds [[buffer(11)]], constant MeshRefitCounts &n [[buffer(13)]],
    constant uint &depth [[buffer(17)]], uint id [[thread_position_in_grid]]) {
    if (status->Errors || id >= n.Nodes) return;
    const MeshRefitNode at = map[id]; if (at.Count || at.Depth != depth) return;
    const BodyBounds a = bounds[at.Left], b = bounds[at.Right];
    bounds[id] = {min(a.Low,b.Low),max(a.High,b.High)};
}
kernel void MeshRefitCommit(
    device const GeometryQueryStatus *status [[buffer(1)]], device const MeshRefitTriangle *map [[buffer(2)]],
    device const MeshRefitNode *node_map [[buffer(8)]], device const uint2 *flags [[buffer(10)]],
    device const BodyBounds *bounds [[buffer(11)]], constant MeshRefitCounts &n [[buffer(13)]],
    device Triangle *triangles [[buffer(28)]], device BvhNode *nodes [[buffer(29)]], uint id [[thread_position_in_grid]]) {
    if (status->Errors) return;
    if (id < n.Triangles) { device Triangle &t=triangles[map[id].Index];t.ActiveEdges=flags[id].x;t.BackActiveEdges=flags[id].y; }
    if (id < n.Nodes) { device BvhNode &node=nodes[node_map[id].Index];node.Low=bounds[id].Low;node.High=bounds[id].High; }
}
