// Toolchain check: a member vector of NS::SharedPtr must not grow past its reserved capacity.
//
// Homebrew clang 22.1.8 at -O2, -O3 and -Os reallocates on every push_back into a member
// std::vector<NS::SharedPtr<T>> that was already reserved, doubling the capacity each time until it
// computes a nonsense length and throws std::bad_alloc. It takes all of: an anonymous namespace, a
// member (not local) vector, NS::SharedPtr elements, push_back after reserve, and metal-cpp
// included. A named namespace, raw pointer elements, -O1, or Apple clang each compile correctly,
// and the same shape with std::shared_ptr and no metal-cpp is fine at every level.
//
// This is why ./run builds at -O1. OPT=-O2 ./run VectorMiscompile fails today, so a future toolchain
// can be re-checked by running it. Longer fills throw std::bad_alloc outright.

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
    constexpr size_t Count = 16; // enough to show the doubling, few enough that it reports instead of throwing
    const Holder holder(Count);
    const auto capacity = holder.Buffers.capacity();
    const bool ok = capacity == Count;
    std::println("{} {} buffers reserved, capacity {} after filling", ok ? "pass:" : "MISCOMPILED:", Count, capacity);
    return ok ? 0 : 1;
}
