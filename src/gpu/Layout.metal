// Reports what the Metal compiler made of every shared struct, so a host test can compare it against
// what clang made of the same text. This is the check that keeps Shared.h honest.

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
