// Reports the size and alignment the Metal compiler gives every shared struct, for a host test to compare against clang's from the same text.

kernel void ReportLayout(device uint *out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i != 0) return;
    uint k = 0;
#define REPORT(T) out[k++] = sizeof(T); out[k++] = alignof(T);
    REPORT(Pose)
    REPORT(Velocity)
    REPORT(BodyMass)
    REPORT(Shape)
    REPORT(Filter)
    REPORT(Contact)
    REPORT(Joint)
    REPORT(Adjacency)
    REPORT(ContactEvent)
    REPORT(StepParams)
#undef REPORT
}
