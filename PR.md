# `evictBindGroups()` on Draw, Effect and Compute

Branch: `loom/clear-draw` (on `main`; see "Where this should land").

## Problem

Bind groups live in one gpu-wide cache (`bind-cache.ts`) keyed
`${drawId}:${group}:${identities}`. The cache evicts on its own in exactly two situations: a
bound resource is destroyed (`subscribeEviction` / `onDestroy`) or a target is recreated
(`onTexturesRecreated`). `clearDraw(drawId)` exists on the cache interface and nothing calls it.

So a drawable that is discarded — or whose bound resources are rotated out without being
destroyed, which is what any pool or ring of buffers does — keeps its `GPUBindGroup`s, and
through them every buffer and texture they reference, until `gpu.dispose()`. A long-lived app
that creates and drops drawables (an editor rebuilding its graph, a pool of effects) leaks GPU
memory it has no handle on. The GC cannot help: the cache holds the entries by string key.

A second, smaller bug sits next to it: `ComputePipeline` keys its cache slot with its own
`nextComputeId` counter while draws use `nextDrawId`, so a compute and a draw can share a
numeric id and, with a `clearDraw`, each other's entries.

Our case (Loom): a node graph compiles to a pass plan; every recompile replaces drawables and
rotates packed attribute buffers. Without eviction, the resident bind groups grow with every
edit for the life of the tab.

## Minimal repro (0.4.1)

```ts
const fx = effect(gpu, SHADER);
fx.set({ camera: bufferA });
frame(gpu, (f) => f.pass({ target }, (p) => p.draw(fx)));
fx.set({ camera: bufferB });          // bufferA is no longer bound anywhere
frame(gpu, (f) => f.pass({ target }, (p) => p.draw(fx)));
// The bind group referencing bufferA is still in the cache. There is no API to drop it short of
// gpu.dispose() or bufferA.destroy() — and a raw GPUBuffer never subscribed in the first place.
```

## Fix

- `Draw.evictBindGroups()`, `Effect.evictBindGroups()` (forwards to its draw) and
  `Compute.evictBindGroups()`: call `cache.clearDraw(id)`. Bindings keep their values and
  ownership (no `VGPU-R1-BINDING-NEVER-SET` afterwards), pipelines, shader modules and layouts
  are untouched; the next draw/dispatch runs `getOrCreate` again. Over-eager calls cost a
  rebuild, never a wrong bind group.
- Compute keys its slot as `` `compute:${id}` `` (`SetCoreOptions.drawId` widened to
  `number | string`, which the cache already accepted), so the eviction of one kind cannot
  reach the other. This is byte-identical to what `canary` already does for the key.
- Docs: signature blocks and a Notes bullet in `draw.docs.md`, `effect.docs.md`,
  `compute.docs.md`; regenerated docs manifest; one `minor` changeset. `triangle-low-level`
  experience budget bumped to the next 512 B multiple (+14 B over the hard gate).

## Tests

`packages/vgpu-api/tests/evict-bind-groups.test.ts` (mock adapter, counts
`createBindGroup` / `createRenderPipeline*` / `createBindGroupLayout` via
`getMockGPUDeviceInstrumentation`):

- "evicting a compute's bind groups leaves a draw with the same numeric id alone" — placed
  first in the file so the first `effect()` and first `compute()` both take id 1 from their
  separate counters; evicting the compute must not evict the draw. Fails with
  `expected 3 to be 2` when the `compute:` prefix is removed (verified).
- "evictBindGroups drops only the bind groups" — rebuild count goes 1 → 2 after eviction with
  no re-`set()`, pipeline and layout counts unchanged, repeated/idle evictions harmless.
- "evictBindGroups on one draw does not touch another draw's entries for the same resources".

All three fail on `main` with `TypeError: ... evictBindGroups is not a function`. Upstream had
no test of the eviction lifecycle; `clearDraw` itself was untested. Full `packages/vgpu-api`
suite, `pnpm -w typecheck`, `pnpm build`, `pnpm bundle-check` green.

## How Loom tests it

`src/runtime/backend/vgpu/bind-group-eviction.test.ts` (mock, control vs treatment): a live
blit re-pointed at every tile of a camera cycle keeps a bounded bind-group count across
identical cycles, and an effect discarded while every resource it bound survives takes its
bind groups with it — both asserting the sampled `createBindGroup` growth series is flat where
the un-evicted control grows. `bind-group-eviction.gpu.test.ts` (Dawn): a shader edit that
keeps every bound resource still renders the edited pixels after eviction, both directions
(`[51,102,153,255]` → swizzled `[153,102,51,255]` → back), so eviction never serves a stale or
wrong bind group.

## Upstream-worthy?

Yes. The cache already has the operation and the interface documents it; what is missing is
the reachable call. The alternative — evicting automatically on `set()` when an identity
changes — would break the R2 cache-hit guarantee that alternating between two resources reuses
both bind groups (`ring1-mock.test.ts` "R2 cache hits when alternating"), so an explicit call
at the lifecycle boundary the caller owns is the shape that keeps both.

## Where this should land

CONTRIBUTING targets `canary`. `canary` already carries the `compute:${id}` key and the
`drawId: number | string` widening (identical lines, so those hunks merge as no-ops); the new
method and its docs apply on top. `draw.ts` on canary hoists `bindGroups()` before encoder
creation and adds `watchDrawResources`, neither of which touches the `set()` region this
branch edits.
