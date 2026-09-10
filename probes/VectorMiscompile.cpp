// Check that filling a reserved member vector retains its capacity.
// This reproduces the compiler issue requiring the RBP target's -O1 setting.

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Metal/Metal.hpp>

#include <print>
#include <vector>

namespace {
struct Holder {
    NS::SharedPtr<MTL::Device> Device{NS::TransferPtr(MTL::CreateSystemDefaultDevice())};
    std::vector<NS::SharedPtr<MTL4::CommandBuffer>> Buffers;

    explicit Holder(size_t count) {
        Buffers.reserve(count);
        while (Buffers.size() < count) Buffers.push_back(NS::TransferPtr(Device->newCommandBuffer()));
    }
};
} // namespace

int main() {
    constexpr size_t Count = 16;
    const Holder holder(Count);
    const auto capacity = holder.Buffers.capacity();
    const bool ok = capacity == Count;
    std::println("{} {} buffers reserved, capacity {} after filling", ok ? "pass:" : "MISCOMPILED:", Count, capacity);
    return ok ? 0 : 1;
}
