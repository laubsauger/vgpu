# `TimerFrameExtent`: the frame's own GPU interval beside the per-span durations

Branch: `loom/timer-frame-extent` (on `main`; see "Where this should land").

## Problem

`timer.onResults` reports one millisecond duration per span. The spans bracket their own render
passes, and that is all the timer exposes: the raw begin/end timestamps are decoded in
`InternalTimer.#dispatch` and dropped.

A frame time cannot be recovered from the spans. Two reasons:

1. The gaps between passes (copies, compute the app did not span, queue scheduling) are not in
   any span, so the sum understates the frame.
2. On GPUs that sample timestamps at stage boundaries — Apple GPUs under Metal (Dawn or Chrome)
   are the case we measured — every pass's begin timestamp lands near the start of the command
   buffer while the ends are sequential. The spans nest, and their sum overstates the frame by
   ~(N+1)/2 for N passes. Measured in Loom on Dawn/Metal (Apple silicon), 24 render passes in
   one command buffer, each sampling the previous one's output: every begin within 0.1 ms of
   the command buffer start (0.000, 0.014, 0.026, 0.080 ...), ends sequential (1.45, 2.89,
   4.32 ... 8.5 ms), span sum 11.7x the frame. An empty marker pass appended to the frame does
   not fix it either: it ends early (1.5 ms into a 9 ms frame) because the GPU overlaps
   independent passes. Only `max(end) - min(begin)` over the raw pairs is the frame.

So the one number a frame profiler wants ("how long did this frame take on the GPU") is not
derivable from the API, while the timestamps that answer it are already in the readback.

Our case (Loom): the telemetry panel shows a per-frame GPU figure and a per-pass breakdown, and
the per-frame figure disagreed with the frame's wall time by ~10x on Apple hardware — the
measurement above is what explained it.

## Minimal repro (0.4.1)

```ts
gpuTimer.onResults((spans) => {
  const frameMs = Object.values(spans).reduce((a, b) => a + b, 0); // wrong on both counts above
});
```

## Fix

- `TimerFrameExtent { extentMs: number; frame: Frame }` and
  `TimerResultsListener = (spans, extent) => void`. `#dispatch` computes `extentMs` as the
  earliest span begin to the latest span end from the raw `BigUint64` timestamps, in the same
  pass that decodes the durations; pairs clamped to `0` (negative deltas) are left out of the
  extent as they are meaningless for it. `frame` is the `Frame` that submitted — the identity
  `frameSubmitted(frame)` already receives — so callers that track several frames can tie a
  result back to what they encoded.
- `onResults` listeners get the frozen extent as a second argument. One-argument listeners are
  unaffected.
- Types exported from `vgpu`, `vgpu/mock`, `vgpu/node` (through `init.ts`, as `Timer` is).
- `timer.docs.md`: signature, parameter row, example, and a Notes bullet that says why the sum
  is wrong. Regenerated docs manifest; one `minor` changeset. No budget change needed.

## Tests

`packages/vgpu-api/tests/timer.test.ts` — "onResults hands the frame's extent beside the spans:
earliest begin to latest end, tied to the submitted Frame". Uses the mock's deterministic
timestamps (query `i` → `i*i` ms): spans `(0,1)` = 1 ms and `(2,3)` = 5 ms, extent = query 3 −
query 0 = 9 ms, while the sum is 6 — the test's own numbers show the two are not the same
quantity. Asserts `extent.frame` is the `Frame` handed to the frame callback, the extent is
frozen, a single-span frame's extent equals its span, and consecutive frames name their own
`Frame`. Fails on `main` with `Cannot read properties of undefined (reading 'extentMs')`.
The other 27 timer tests are untouched and pass; full `packages/vgpu-api` suite,
`pnpm -w typecheck`, `pnpm build`, `pnpm bundle-check` green.

## How Loom tests it

`src/runtime/telemetry/frame-extent.gpu.test.ts` (Dawn/Metal): "fits inside the frame's own
interval, and the pass sum does not". Renders warm-up plus measured frames through the real
compiler + backend with one span per pass, asserts one result per submitted frame naming its
submit in order, a positive extent per measured frame, and — on an Apple adapter — that the
mean pass sum exceeds the mean extent by more than 2x (the nesting), while on a serial adapter
the sum stays within the extent plus slack. The telemetry bucket the UI reads reports
`basis: "frame"`, `availability: "measured"` and the extent, not the sum.

## Upstream-worthy?

Yes. It is additive, costs two comparisons per span in a loop that already runs, and exposes
data the timer already holds. The alternative of exposing the raw timestamps would also work
but leaks the query-index allocation into the public API; an extent is the shape every profiler
overlay actually wants. The Apple nesting behaviour is worth documenting upstream regardless of
whether this lands, since it makes the obvious use of the current API wrong on the most common
developer machine.

## Where this should land

CONTRIBUTING targets `canary`. `timer.ts` is byte-identical between `main` and `canary`, so
this cherry-picks clean.
