# `preserveSamples`: multi-pass rendering into one MSAA target

Branch: `loom/msaa-discard-store` (on `main`; see "Where this should land").

## Problem

`target(gpu, { msaa: true })` sets `storeOp: "discard"` on the multisample color attachment and
`depthStoreOp` / `stencilStoreOp: "discard"` on multisampled depth (`target-utils.ts`), and
`Frame.pass` refuses `clear: false` into a 4x target with `VGPU-PASS-PRESERVE-MSAA`.

For a single-pass renderer that is the right default: after the resolve the samples are dead and
`discard` is free bandwidth on tilers. It rules out one legitimate WebGPU shape, though: drawing
anti-aliased geometry into the same MSAA target across several passes. The multisampled texture
persists on the target between passes, so loading it is legal; vgpu just never stores it.

Our case (Loom, a node compositor): `render -> cut -> blur -> glow -> clip`, where a later pass
preserves into the render target and a mid-chain effect samples the resolved color. Anti-aliasing
has to happen before the post chain (aliased edges into a bloom get amplified), so the AA target is
the one that gets drawn into twice.

## Minimal repro (0.4.1)

```ts
const scene = target(gpu, { size: [256, 256], depth: true, msaa: true });
frame(gpu, (f) => {
  f.pass({ target: scene }, (p) => p.draw(geometryA));
  f.pass({ target: scene, clear: false }, (p) => p.draw(geometryB)); // throws VGPU-PASS-PRESERVE-MSAA
});
```

Removing the guard alone is not enough: with `discard`, the second pass loads garbage color and
depth-tests against a depth buffer that no longer exists.

## Fix

Opt-in, per target, default unchanged:

- `TargetTextureOptions.preserveSamples?: boolean` (and `Target.preserveSamples?: boolean`, read
  by `Frame.pass`). `OffscreenTarget` passes it to `colorAttachment` / `depthAttachment`, which
  then use `storeOp: "store"` for the multisample color, depth and stencil aspects. Every pass
  still resolves into `.color(s)` (that part was already the upstream behaviour), so a mid-chain
  reader sees the current accumulation.
- `Frame.pass` accepts `clear: false` on an MSAA target whose `preserveSamples` is true.
- `preserveSamples` without `msaa` throws `VGPU-TARGET-PRESERVE-SAMPLES` (the dead-option rule
  the rest of `target()` / `Frame.pass` follows).
- Docs (`target.docs.md`, `frame.docs.md`) and the `VGPU-PASS-PRESERVE-MSAA` fix-it now point at
  the option. One changeset (`minor`: new option).
- Bundle budgets: `./node` re-baselined by `pnpm bundle-check --update`; the `triangle-low-level`
  experience budget bumped by hand to the next 512 B multiple (28721 B measured, was 28672),
  because `--update` does not rewrite experience budgets.

Not changed, deliberately: `depthReadOnly` on an MSAA target still throws
`VGPU-PASS-DEPTH-READONLY-MSAA`. With stored samples the retained depth is real, so that guard
could also be relaxed under `preserveSamples`; we have no consumer for it and left it for a
follow-up rather than widen this change.

## Tests

- `packages/vgpu-api/tests/msaa-mock.test.ts`:
  - "preserveSamples stores the multisample attachments across passes and still resolves every
    pass" — two passes into one MSAA depth-stencil target, the second `clear: false`; asserts
    `storeOp` / `depthStoreOp` / `stencilStoreOp: "store"` on both descriptors, `resolveTarget`
    on both, `loadOp: "clear"` then `"load"` (color, depth, stencil), and that a target without
    the option still reports `preserveSamples === false`. Fails on `main` with
    `VGPU-PASS-PRESERVE-MSAA` at the second pass.
  - "preserveSamples without msaa is a dead option and throws VGPU-TARGET-PRESERVE-SAMPLES".
- The existing `discard`-asserting tests (`msaa-mock`, `blend-load` "rejects clear false with
  MSAA targets", `depth-readonly` "rejected on an MSAA target") are untouched and still pass:
  the default did not move.
- `pnpm -w typecheck`, `pnpm build`, `pnpm bundle-check` green.

## How Loom tests it

`src/nodes/definitions/scene-antialias.gpu.test.ts` (Dawn/Metal), "a hard diagonal edge gains
intermediate coverage in BOTH aa modes, and has none without": one hard-edged diagonal beam
rendered through the real compiler + backend three times — antialias off, 2x SSAA, 4x MSAA
(the MSAA target is drawn into across the chain's preserve passes) — and read back. Off is the
control: 0 columns with a value between dark and bright. SSAA and MSAA must each show more than
20 blended columns and an interior pixel above 200, so neither a broken resolve (all dark / all
bright) nor a no-op flag (identical images) passes. It runs against the patched `dist` today;
on this branch the backend would set `preserveSamples: true` on that target and drop the patch
hunk.

## Upstream-worthy?

Yes, as an opt-in. The `discard` default is a sound performance choice for the common case and
this keeps it. The alternative — flipping the default to `store` for every MSAA target, which is
what our local patch does — costs every single-pass consumer store bandwidth for nothing and
contradicts three existing tests and two doc paragraphs. A per-target flag is the smallest shape
that makes the multi-pass case expressible without changing anyone else's frame.

## Where this should land

CONTRIBUTING says package/API work targets `canary`, not `main`. This branch is on `main` as
requested; it cherry-picks onto `canary` cleanly (`target-utils.ts` moved only in
`validateTargetOptions`'s size check, which is adjacent, not overlapping).
