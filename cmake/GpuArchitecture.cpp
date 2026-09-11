#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Metal/Metal.hpp>

#include <cstdio>

int main() {
    const auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    const auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
    if (!device) return 1;
    return std::puts(device->architecture()->name()->utf8String()) < 0;
}
