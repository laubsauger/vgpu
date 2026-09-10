---
"vgpu": minor
---

Add `evictBindGroups()` to `Draw`, `Effect` and `Compute`. Bind groups live in one gpu-wide cache keyed per drawable and are evicted on their own only when a bound resource is destroyed or recreated, so a drawable that is discarded, or whose resources are rotated out without being destroyed, pins its `GPUBindGroup`s and everything they reference until `gpu.dispose()`. `evictBindGroups()` drops just those cached entries: bindings keep their values and ownership, pipelines and layouts are untouched, and the next draw or dispatch rebuilds what it needs. Compute pipelines now key their cache slot as `compute:<id>` so a compute and a draw with the same numeric id no longer share entries.
