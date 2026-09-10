// Reports the size and alignment the Metal compiler gives every shared struct, for a host test to compare against clang's from the same text.

kernel void ReportLayout(device uint *out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i != 0) return;
    uint k = 0;
#define REPORT(T)         \
    out[k++] = sizeof(T); \
    out[k++] = alignof(T);
    REPORT(Pose)
    REPORT(Velocity)
    REPORT(Displacement)
    REPORT(BodyIterate)
    REPORT(BodyMass)
    REPORT(BodyBounds)
    REPORT(BroadPhaseNode)
    REPORT(MortonKey)
    REPORT(Shape)
    REPORT(Triangle)
    REPORT(Material)
    REPORT(JointDrive)
    REPORT(Filter)
    REPORT(CollisionMask)
    REPORT(Contact)
    REPORT(Joint)
    REPORT(Adjacency)
    REPORT(ContactEvent)
    REPORT(StepParams)
    REPORT(SensorFollower)
    REPORT(ContactReport)
    REPORT(SensorPair)
    REPORT(StepCounts)
    REPORT(StepCompletion)
    REPORT(StepOutputFlags)
#undef REPORT
}

kernel void ReportBroadPhasePairs(
    device BroadPhaseNode *nodes [[buffer(13)]], device uint *pairs [[buffer(0)]],
    constant uint &bodies [[buffer(7)]], uint body [[thread_position_in_grid]]
) {
    if (body >= bodies) return;
    for (uint other = NextBodyCandidate(nodes, bodies, body, 0); other != NoIndex; other = NextBodyCandidate(nodes, bodies, body, other + 1))
        pairs[body * bodies + other] = 1;
}
