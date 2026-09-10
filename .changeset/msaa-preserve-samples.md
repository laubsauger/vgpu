---
"vgpu": minor
---

Add `TargetOptions.preserveSamples` for MSAA targets that are drawn into across several passes. By default an MSAA target discards its multisample color and depth attachments after each pass's resolve (`storeOp: "discard"`), which is why `Frame.pass({ clear: false })` on an MSAA target throws `VGPU-PASS-PRESERVE-MSAA`: there is nothing to load. With `target(gpu, { size, msaa: true, preserveSamples: true })` the attachments are stored (`storeOp: "store"` for color, depth and stencil), every pass still resolves into `.color(s)`, and `clear: false` passes load the previous pass's samples and depth-test against its depth. `preserveSamples` without `msaa` is a dead option and throws `VGPU-TARGET-PRESERVE-SAMPLES`. Targets that do not opt in are unchanged.
