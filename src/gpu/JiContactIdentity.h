#ifndef JI_CONTACT_IDENTITY_H
#define JI_CONTACT_IDENTITY_H

static void JiStore3(thread float *out, float3 v) {
    out[0] = v.x; out[1] = v.y; out[2] = v.z;
}

static void JiContactKey(thread JiWarmContact &entry, GeometryContact raw,
                         device const uint *spawns, device const uint *shapes) {
    entry.key[0] = raw.BodyA;
    entry.key[1] = spawns[raw.BodyA];
    entry.key[2] = raw.BodyB;
    entry.key[3] = spawns[raw.BodyB];
    entry.key[4] = shapes[raw.BodyA];
    entry.key[5] = shapes[raw.BodyB];
    entry.key[6] = uint(raw.Children);
    entry.key[7] = uint(raw.Children >> 32);
    entry.key[8] = raw.SubShapeA;
    entry.key[9] = raw.SubShapeB;
    entry.key[10] = raw.Feature;
}

static uint JiWarmBucket(thread const JiWarmContact &entry, uint mask) {
    uint hash = 2166136261u;
    for (uint k = 0; k < 11; ++k) hash = (hash ^ entry.key[k]) * 16777619u;
    hash ^= hash >> 16;
    return hash & mask;
}

static uint JiMovingContactMask(JiContactLink link, device const JiConstraintTile &tile, uint lane) {
    uint mask = link.flipped ? 2u : 1u;
    if (tile.side[1].body[lane] != ~0u) mask |= link.flipped ? 1u : 2u;
    return mask;
}

static void JiCacheIdentity(thread JiWarmContact &entry, GeometryContact raw,
    JiContactLink link, device const JiConstraintTile &tile, uint lane,
    device const Pose *poses, device const uint *spawns, device const uint *shapes) {
    JiContactKey(entry, raw, spawns, shapes);
    JiStore3(entry.local_a, LocalPoint(poses[raw.BodyA], raw.PointA));
    JiStore3(entry.local_b, LocalPoint(poses[raw.BodyB], raw.PointB));
    JiStore3(entry.normal, raw.Normal);
    entry.moving_mask = JiMovingContactMask(link, tile, lane);
}

static uint JiMatchWarm(GeometryContact raw, device const Pose *poses,
    device const uint *spawns, device const uint *shapes,
    device const JiWarmContact *previous, device const uint *previous_count,
    device atomic_uint *claims, device const atomic_uint *heads,
    device const uint *next, uint bucket_mask, uint required_mask) {
    JiWarmContact target = {};
    JiContactKey(target, raw, spawns, shapes);
    uint matched = ~0u;
    const uint bucket = JiWarmBucket(target, bucket_mask);
    for (uint attempt = 0; attempt <= previous_count[0]; ++attempt) {
        float nearest = INFINITY;
        uint best = ~0u;
        for (uint candidate = atomic_load_explicit(heads + bucket, memory_order_relaxed);
             candidate != ~0u; candidate = next[candidate]) {
            if (atomic_load_explicit(claims + candidate, memory_order_relaxed)) continue;
            device const JiWarmContact &old = previous[candidate];
            bool same = true;
            for (uint k = 0; k < 11; ++k) same &= old.key[k] == target.key[k];
            const float3 old_normal = float3(old.normal[0], old.normal[1], old.normal[2]);
            if (!same || dot(old_normal, raw.Normal) < 0.8f ||
                old.moving_mask != required_mask) continue;
            const float3 old_a = float3(old.local_a[0], old.local_a[1], old.local_a[2]);
            const float3 old_b = float3(old.local_b[0], old.local_b[1], old.local_b[2]);
            const float3 a = WorldPoint(poses[raw.BodyA], old_a) - raw.PointA;
            const float3 b = WorldPoint(poses[raw.BodyB], old_b) - raw.PointB;
            const float score = dot(a, a) + dot(b, b);
            if (score < nearest) { nearest = score; best = candidate; }
        }
        if (best == ~0u) break;
        uint expected = 0;
        if (atomic_compare_exchange_weak_explicit(claims + best, &expected, 1u,
                                                  memory_order_relaxed, memory_order_relaxed)) {
            matched = best;
            break;
        }
    }
    return matched;
}

#endif
