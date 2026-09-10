---
"vgpu": minor
---

`timer.onResults` listeners receive a second argument, a frozen `TimerFrameExtent`: `extentMs` is the earliest span begin to the latest span end of that frame's submit, and `frame` is the `Frame` the spans were attached to. Per-span durations cannot be summed into a frame time — the gaps between passes are missing, and on GPUs that sample timestamps at stage boundaries (Apple GPUs under Metal) spans nest and the sum overstates the frame by ~(N+1)/2 — and the raw timestamps that resolve it were only ever seen inside the timer. `TimerFrameExtent` and `TimerResultsListener` are exported from `vgpu`, `vgpu/mock` and `vgpu/node`. Existing one-argument listeners are unaffected.
