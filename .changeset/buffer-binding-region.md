---
"vgpu": patch
---

`set()` now accepts a bare `{ buffer: GPUBuffer, offset?, size? }` region for buffer bindings. Previously the shape carried none of vgpu's resource markers, so it was classified as a JS value and rejected with "needs a compatible resource, not JS". A region is user-owned, its bind-group identity carries the byte range (two regions of one buffer are two bind groups, not one cache entry with the second's offset silently ignored), and the compute storage-aliasing preflight compares ranges instead of buffers: reading `[0, n)` and writing `[n, 2n)` of one packed buffer passes, overlapping ranges with a writable side still throw `VGPU-R1-STORAGE-ALIASING`.
