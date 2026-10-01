#pragma once

#include <Foundation/NSAutoreleasePool.hpp>

namespace rbp::mtl {
struct AutoreleasePool {
    AutoreleasePool() : Pool(NS::AutoreleasePool::alloc()->init()) {}
    ~AutoreleasePool() { Pool->release(); }
    AutoreleasePool(const AutoreleasePool &) = delete;
    AutoreleasePool &operator=(const AutoreleasePool &) = delete;

    template<typename... T> static void Release(T &...owners) {
        const AutoreleasePool pool;
        ((owners = T{}), ...);
    }

private:
    NS::AutoreleasePool *Pool;
};
} // namespace rbp::mtl
