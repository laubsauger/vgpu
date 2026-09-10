# Bare `{ buffer, offset, size }` regions as buffer-binding resources

Branch: `loom/buffer-binding-resource` (on `main`; see "Where this should land").

## Problem

`set()` classifies a buffer-binding value as either a JS value (lib-owned, uploaded into a
vgpu-created buffer) or a resource (user-owned, bound as-is). The classifier,
`hasAnyResourceShape` in `set-resources.ts`, looks for vgpu's own markers: `gpu`, `bindGroup`,
`createView`, `resourceIdentity`. A bare WebGPU `GPUBufferBinding` — `{ buffer: GPUBuffer,
offset, size }`, the shape `GPUBindGroupEntry.resource` takes — carries none of them, so it is
routed down the JS-value path and rejected:

```
VGPUError: Binding 'src' needs a compatible resource, not JS.
```

`normalizeBufferResource` already has an `isGPUBufferBinding` branch for exactly this shape; it
is unreachable because the ownership decision happens first. And that branch had a second
problem: it keyed the bind-group identity on the buffer alone (`syntheticIdentity(value.buffer)`),
so two regions of one buffer collapsed onto one cache entry — binding `[0, n)` then `[n, 2n)`
returned the first bind group with the second's offset silently ignored.

Our case (Loom): point-cloud attributes live packed in one storage arena, each attribute a
sub-range bound as its own storage binding. Every kernel binds several regions of the same
buffer, some read, some write.

## Minimal repro (0.4.1)

```ts
const arena = gpu.device.createBuffer({ size: 64, usage: ["storage", "copy_dst"] }).gpu;
sim.set({ src: { buffer: arena, offset: 0, size: 32 }, dst: other });
// -> VGPU-RING1-UNSUPPORTED "Binding 'src' needs a compatible resource, not JS."
```

## Fix

Three hunks, all in the buffer path:

- `hasAnyResourceShape` also recognises `isGPUBufferBinding(record)`, so a region is user-owned.
- `normalizeBufferRegion` gives a region the identity
  `external:<buffer id>@<offset>+<size>` (offset defaulting to 0, size to the remainder of the
  buffer) and binds a normalised `{ buffer, offset, size }`. Distinct ranges of one buffer are
  distinct bind-group identities; the same range is a cache hit.
- `ComputePipeline.#preflightAliasing` compares byte ranges instead of identities: it buckets
  storage bindings by `GPUBuffer` and throws `VGPU-R1-STORAGE-ALIASING` only when two ranges
  overlap and at least one is writable. Whole-buffer bindings (`Buffer`, `StorageBuffer`,
  raw `GPUBuffer`) are the range `[0, size)`, so every existing case — same buffer twice, ping-pong
  halves, read+read — behaves exactly as before, with the same error text.

Docs (`draw.docs.md` Notes, `compute.docs.md` throws + Notes) and one `patch` changeset. Bundle
budgets: `./node` and `triangle-low-level` bumped to the next 512 B multiple (the tooling
warning was inside the growth threshold; the experience budget is a hard gate).

## Tests

- `tests/ring1-mock.test.ts` — "a bare GPUBufferBinding region is a user resource, and two
  regions of one buffer are two bind groups": binds `[0,256)`, then `[256,512)`, then `[0,256)`
  again of one raw `GPUBuffer` across three frames and asserts `createBindGroup === 2`. On
  `main` it fails at the first `set()` (the "not JS" error); with the classifier hunk alone it
  fails with `expected 1 to be 2` (the offset is dropped on the cache hit), so both hunks are
  load-bearing.
- `tests/compute/aliasing.test.ts` — three new cases: disjoint regions of one buffer dispatch;
  overlapping regions with a writable side throw the exact fix-it text (including a region
  without `size`, which runs to the end of the buffer); overlapping read-only regions pass.
  The existing five aliasing tests are untouched and pass.
- `pnpm -w typecheck`, full `packages/vgpu-api` suite (647 passed, 54 env-gated primitive
  snapshots skipped as on `main`), `pnpm build`, `pnpm bundle-check` green.

## How Loom tests it

`src/runtime/backend/vgpu/point-packed-attributes.gpu.test.ts` (Dawn), "renders, and every
attribute reads back EXACTLY out of its own region": eight point attributes packed into one
storage buffer at offsets `[0, 256, 512, ..., 1792]`, each bound to a compute kernel as its own
region of that buffer, the kernel dispatched through the real compiler + backend, and each
attribute read back and compared exactly (`position`, `velocity`, `life` values per slot). The
`point-*` Dawn suites exercise the same binding shape. Loom currently sidesteps the classifier
by attaching a `resourceIdentity` string to the region it hands `set()` (`resources.ts`
`bufferRegion()`), which is why the identity hunk is the one it cannot live without; with this
branch the region needs no marker at all.

## Upstream-worthy?

Yes. `GPUBufferBinding` is the WebGPU resource shape, the code already tries to accept it, and
the identity bug is a silent wrong-bind-group, not a rejection. The range-based preflight is the
part that could be argued either way; it is strictly more permissive than before only for
non-overlapping ranges, which WebGPU itself allows.

## Where this should land

CONTRIBUTING targets `canary`. On `canary`, `set-resources.ts` has grown (destroyed-binding
wrapper, storage-texture normalisation, `resourceLabel`) but `hasAnyResourceShape` and the
`isGPUBufferBinding` line are unchanged, and `compute.ts` still has the identity-keyed
`#preflightAliasing`; the cherry-pick needs only context adjustment (see the audit's
canary-drift notes).
