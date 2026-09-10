static bool FiniteBounds(BodyBounds bounds) {
    return all((as_type<uint3>(bounds.Low) & 0x7f800000u) != 0x7f800000u) &&
        all((as_type<uint3>(bounds.High) & 0x7f800000u) != 0x7f800000u);
}

static BodyBounds ReduceBounds(BodyBounds value, threadgroup float3 *lows, threadgroup float3 *highs, uint lane, uint width) {
    lows[lane] = value.Low;
    highs[lane] = value.High;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = width / 2; stride > 0; stride /= 2) {
        if (lane < stride) {
            lows[lane] = min(lows[lane], lows[lane + stride]);
            highs[lane] = max(highs[lane], highs[lane + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return {lows[0], highs[0]};
}

kernel void ReduceBodyBounds(
    device const BodyBounds *bounds [[buffer(14)]], device BodyBounds *reductions [[buffer(13)]],
    constant StepParams &p [[buffer(7)]], uint body [[thread_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]]
) {
    threadgroup float3 lows[RadixBlockSize], highs[RadixBlockSize];
    BodyBounds value{float3(INFINITY), float3(-INFINITY)};
    if (body < p.BodyCount && FiniteBounds(bounds[body])) value = bounds[body];
    const BodyBounds result = ReduceBounds(value, lows, highs, lane, RadixBlockSize);
    if (lane == 0) reductions[group] = result;
}

kernel void ReduceSceneBounds(
    device BodyBounds *reductions [[buffer(13)]], constant StepParams &p [[buffer(7)]], uint lane [[thread_index_in_threadgroup]]
) {
    const uint blocks = RadixBlocks(p.BodyCount);
    float3 low = INFINITY, high = -INFINITY;
    for (uint i = lane; i < blocks; i += RadixBlockSize) {
        low = min(low, reductions[i].Low);
        high = max(high, reductions[i].High);
    }
    threadgroup float3 lows[RadixBlockSize], highs[RadixBlockSize];
    const BodyBounds result = ReduceBounds({low, high}, lows, highs, lane, RadixBlockSize);
    if (lane == 0) reductions[blocks] = result;
}

static uint MortonSpread(uint value) {
    value = (value | (value << 16)) & 0x030000ffu;
    value = (value | (value << 8)) & 0x0300f00fu;
    value = (value | (value << 4)) & 0x030c30c3u;
    return (value | (value << 2)) & 0x09249249u;
}

static MortonKey MakeKey(BodyBounds box, BodyBounds scene, uint body) {
    uint code = ~0u;
    if (FiniteBounds(box)) {
        const float3 center = 0.5f * box.Low + 0.5f * box.High;
        const float3 extent = max(scene.High - scene.Low, float3(1e-20f));
        const uint3 cell = uint3(clamp((center - scene.Low) / extent, 0.f, 1.f) * 1023.f);
        code = MortonSpread(cell.x) | (MortonSpread(cell.y) << 1) | (MortonSpread(cell.z) << 2);
    }
    return {code, body};
}

static void InitTreeLeaf(device BroadPhaseNode *nodes, BodyBounds box, uint bodies, uint body) {
    nodes[body] = {box, NoIndex, NoIndex, NoIndex, body, body, 0, 0, 0};
    nodes[bodies + body] = {{float3(INFINITY), float3(-INFINITY)}, NoIndex, NoIndex, NoIndex, NoIndex, 0, 0, 0, 0};
}

kernel void MakeMortonKeys(
    device const BodyBounds *bounds [[buffer(14)]], device const BodyBounds *reductions [[buffer(13)]],
    device MortonKey *keys [[buffer(11)]], device BroadPhaseNode *nodes [[buffer(10)]],
    constant StepParams &p [[buffer(7)]], uint body [[thread_position_in_grid]]
) {
    if (body >= p.BodyCount) return;
    keys[body] = MakeKey(bounds[body], reductions[RadixBlocks(p.BodyCount)], body);
    InitTreeLeaf(nodes, bounds[body], p.BodyCount, body);
}

// Stable block ranks use SIMD ballots for equal digits, then counts from preceding SIMD groups.
kernel void RadixHistogram(
    device const MortonKey *keys [[buffer(11)]], device uint *scratch [[buffer(9)]],
    constant uint &digit_index [[buffer(25)]], constant StepParams &p [[buffer(7)]],
    uint body [[thread_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]
) {
    threadgroup uint histogram[RadixWaves * RadixBins];
    for (uint i = tid; i < RadixWaves * RadixBins; i += RadixBlockSize) histogram[i] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool valid = body < p.BodyCount;
    const uint digit = valid ? ((keys[body].Code >> (8 * digit_index)) & 255u) : 0;
    uint match = uint(ulong(simd_ballot(valid)));
    for (uint bit = 0; bit < 8; ++bit) {
        const uint ones = uint(ulong(simd_ballot((digit & (1u << bit)) != 0)));
        match &= (digit & (1u << bit)) ? ones : ~ones;
    }
    const uint earlier = popcount(match & ((1u << lane) - 1));
    if (valid && earlier == 0) histogram[simd * RadixBins + digit] = popcount(match);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (valid) {
        uint rank = earlier;
        for (uint previous = 0; previous < simd; ++previous) rank += histogram[previous * RadixBins + digit];
        scratch[body] = rank;
    }
    uint total = 0;
    for (uint wave = 0; wave < RadixWaves; ++wave) total += histogram[wave * RadixBins + tid];
    scratch[p.BodyCount + group * RadixBins + tid] = total;
}

kernel void RadixOffsets(
    device uint *scratch [[buffer(9)]], constant StepParams &p [[buffer(7)]],
    uint digit [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]
) {
    const uint blocks = RadixBlocks(p.BodyCount);
    device uint *histogram = scratch + p.BodyCount;
    uint total = 0;
    for (uint block = 0; block < blocks; ++block) total += histogram[block * RadixBins + digit];
    uint prefix = simd_prefix_exclusive_sum(total);
    const uint wave_total = simd_sum(total);
    threadgroup uint wave_totals[RadixWaves];
    if (lane == 0) wave_totals[simd] = wave_total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint wave = 0; wave < simd; ++wave) prefix += wave_totals[wave];
    for (uint block = 0; block < blocks; ++block) {
        const uint at = block * RadixBins + digit, count = histogram[at];
        histogram[at] = prefix;
        prefix += count;
    }
}

kernel void RadixScatter(
    device const MortonKey *input [[buffer(11)]], device MortonKey *output [[buffer(13)]],
    device const uint *scratch [[buffer(9)]], constant uint &digit_index [[buffer(25)]],
    constant StepParams &p [[buffer(7)]], uint body [[thread_position_in_grid]]
) {
    if (body >= p.BodyCount) return;
    const MortonKey key = input[body];
    const uint digit = (key.Code >> (8 * digit_index)) & 255u;
    output[scratch[p.BodyCount + (body / RadixBlockSize) * RadixBins + digit] + scratch[body]] = key;
}

// Sorted positions break ties between equal Morton codes, yielding a unique 64-bit radix key.
static int Prefix(device const MortonKey *keys, int count, int a, int b) {
    if (b < 0 || b >= count) return -1;
    const uint difference = keys[a].Code ^ keys[b].Code;
    return difference ? int(clz(difference)) : 32 + int(clz(uint(a) ^ uint(b)));
}

static void BuildTreeNode(device const MortonKey *keys, device BroadPhaseNode *nodes, uint bodies, uint index) {
    // Karras (2012): compute each internal node's range and split independently.
    const int count = int(bodies), i = int(index);
    if (i >= count - 1) return;
    const int direction = Prefix(keys, count, i, i + 1) > Prefix(keys, count, i, i - 1) ? 1 : -1;
    const int minimum = Prefix(keys, count, i, i - direction);
    int maximum = 2;
    while (Prefix(keys, count, i, i + maximum * direction) > minimum) maximum *= 2;
    int length = 0;
    for (int step = maximum / 2; step > 0; step /= 2)
        if (Prefix(keys, count, i, i + (length + step) * direction) > minimum) length += step;
    const int end = i + length * direction, first = min(i, end), last = max(i, end);
    const int common = Prefix(keys, count, first, last);
    int split = first, step = last - first;
    do {
        step = (step + 1) / 2;
        const int next = split + step;
        if (next < last && Prefix(keys, count, first, next) > common) split = next;
    } while (step > 1);
    const uint left = split == first ? keys[split].Body : bodies + uint(split);
    const uint right = split + 1 == last ? keys[split + 1].Body : bodies + uint(split + 1);
    const uint node = bodies + index;
    nodes[node].Left = left;
    nodes[node].Right = right;
    nodes[left].Parent = node;
    nodes[right].Parent = node;
}

kernel void BuildRadixTree(
    device const MortonKey *keys [[buffer(11)]], device BroadPhaseNode *nodes [[buffer(13)]],
    constant StepParams &p [[buffer(7)]], uint index [[thread_position_in_grid]]
) { BuildTreeNode(keys, nodes, p.BodyCount, index); }

static uint CandidateMask(device const BodyBounds *bounds, uint bodies, uint body) {
    uint mask = 0;
    for (uint other = 0; other < bodies; ++other)
        if (other != body && BoundsOverlap(bounds[body], bounds[other])) mask |= 1u << other;
    return mask;
}

kernel void RefreshRadixLeaves(
    device const BodyBounds *bounds [[buffer(14)]], device BroadPhaseNode *nodes [[buffer(13)]],
    constant StepParams &p [[buffer(7)]], uint body [[thread_position_in_grid]]
) {
    if (body >= p.BodyCount) return;
    nodes[body].Bounds = bounds[body];
    if (p.BodyCount <= RadixSimdWidth) nodes[body].Candidates = CandidateMask(bounds, p.BodyCount, body);
    nodes[body].Ready = 0;
    nodes[p.BodyCount + body].Ready = 0;
}

static void RefitTreeLeaf(device BroadPhaseNode *nodes, uint bodies, uint body) {
    if (body >= bodies) return;
    uint parent = nodes[body].Parent;
    // A binary radix tree over 64-bit keys has at most 64 ancestors.
    for (uint depth = 0; parent != NoIndex && depth < 64; ++depth) {
        // Release child writes before incrementing the arrival count.
        // The second arrival acquires the first child's writes before combining bounds (MSL 3.2).
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst);
        if (atomic_fetch_add_explicit((device atomic_uint *)&nodes[parent].Ready, 1u, memory_order_relaxed) == 0) return;
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst);
        uint left = nodes[parent].Left, right = nodes[parent].Right;
        const BroadPhaseNode a = nodes[left], b = nodes[right];
        nodes[parent].Bounds = {min(a.Bounds.Low, b.Bounds.Low), max(a.Bounds.High, b.Bounds.High)};
        nodes[parent].MinBody = min(a.MinBody, b.MinBody);
        nodes[parent].MaxBody = max(a.MaxBody, b.MaxBody);
        if (a.MinBody > b.MinBody) {
            nodes[parent].Left = right;
            nodes[parent].Right = left;
        }
        parent = nodes[parent].Parent;
    }
    if (parent != NoIndex) atomic_fetch_or_explicit((device atomic_uint *)&nodes[BroadPhaseRoot(bodies)].Errors, 1u, memory_order_relaxed);
}

kernel void RefitRadixTree(
    device BroadPhaseNode *nodes [[buffer(13)]], constant StepParams &p [[buffer(7)]], uint body [[thread_position_in_grid]]
) { RefitTreeLeaf(nodes, p.BodyCount, body); }

kernel void BuildSmallBroadPhase(
    device const BodyBounds *bounds [[buffer(14)]], device MortonKey *keys [[buffer(11)]],
    device BroadPhaseNode *nodes [[buffer(13)]], constant StepParams &p [[buffer(7)]],
    uint tid [[thread_index_in_threadgroup]], uint3 group_size [[threads_per_threadgroup]]
) {
    // A world fitting one SIMD group stores its ordered candidate set in one word.
    if (p.BodyCount <= RadixSimdWidth) {
        if (tid < p.BodyCount) {
            InitTreeLeaf(nodes, bounds[tid], p.BodyCount, tid);
            nodes[tid].Candidates = CandidateMask(bounds, p.BodyCount, tid);
        }
        return;
    }
    const uint width = group_size.x;
    threadgroup float3 lows[RadixBlockSize], highs[RadixBlockSize];
    const bool valid = tid < p.BodyCount;
    BodyBounds box{float3(INFINITY), float3(-INFINITY)};
    if (valid && FiniteBounds(bounds[tid])) box = bounds[tid];
    const BodyBounds scene = ReduceBounds(box, lows, highs, tid, width);
    MortonKey key = valid ? MakeKey(bounds[tid], scene, tid) : MortonKey{~0u, NoIndex};
    threadgroup MortonKey ordered[RadixBlockSize];
    for (uint span = 2; span <= width; span *= 2) {
        for (uint stride = span / 2; stride > 0; stride /= 2) {
            ordered[tid] = key;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const MortonKey other = ordered[tid ^ stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const bool less = key.Code < other.Code || (key.Code == other.Code && key.Body < other.Body);
            const bool want_less = ((tid & span) == 0) == ((tid & stride) == 0);
            if (less != want_less) key = other;
        }
    }
    if (valid) {
        keys[tid] = key;
        InitTreeLeaf(nodes, bounds[tid], p.BodyCount, tid);
    }
    threadgroup_barrier(mem_flags::mem_device);
    BuildTreeNode(keys, nodes, p.BodyCount, tid);
    threadgroup_barrier(mem_flags::mem_device);
    RefitTreeLeaf(nodes, p.BodyCount, tid);
}
