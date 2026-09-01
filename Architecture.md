# Architecture

The measured Metal 4 facts the engine is built on, established by the probes in [`probes/`](probes/) — see the README for how to run them.

## Command encoding

A step is one `MTL4CommandBuffer` holding a single compute encoder, with a barrier between dispatches that depend on each other.
[`probes/DispatchChain.cpp`](probes/DispatchChain.cpp) runs a 128-link chain of dependent in-place increments: with barriers the readback is exactly 128, without them 36 to 75.
Overlapping dispatches lose writes and Metal reports no error.

Of the three ways to express the same chain, the cost order is stable across runs: barriers inside a single encoder are cheapest and stay flat with chain length, one encoder per pass (`barrierAfterQueueStages`) is worse, and one command buffer per pass is worst and degrades fastest.

Barrier overhead is small next to any real dispatch.
Growing one dispatch from 256 threads to a million takes the per-link cost from mostly barrier to almost entirely execution, so the solve is structured for convergence rather than to minimize passes.

GPU writes are published to the host with queue-signalled events.
Kernel-written flags are not reliable for this.
