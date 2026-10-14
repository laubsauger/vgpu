import { prepareShader } from "@vgpu/wgsl/prepare";
import { expect, test, vi } from "vitest";
import { draw, frame, init, target } from "../src/mock.ts";

const SHADER = prepareShader(`
@vertex fn vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
  var positions = array<vec2f, 3>(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
  return vec4f(positions[index], 0.5, 1);
}
@fragment fn fragment() -> @location(0) vec4f { return vec4f(1); }
`);

test("preserved MSAA stores color, depth and stencil while resolving and loading later passes", async () => {
  const gpu = await init();
  const passes: GPURenderPassDescriptor[] = [];
  const original = gpu.gpu.createCommandEncoder.bind(gpu.gpu);
  vi.spyOn(gpu.gpu, "createCommandEncoder").mockImplementation(options => {
    const encoder = original(options);
    const begin = encoder.beginRenderPass.bind(encoder);
    vi.spyOn(encoder, "beginRenderPass").mockImplementation(descriptor => {
      passes.push(descriptor);
      return begin(descriptor);
    });
    return encoder;
  });
  try {
    for (const msaa of [true, 4] as const) {
      const output = target(gpu, { size: [4, 4], depth: "depth24plus-stencil8", msaa, preserveSamples: true });
      expect(output.preserveSamples).toBe(true);
      frame(gpu, current => {
        current.pass(output, () => undefined);
        current.pass({ target: output, clear: false }, () => undefined);
      });
    }
    expect(passes).toHaveLength(4);
    for (const [index, descriptor] of passes.entries()) {
      const loadOp = index % 2 === 0 ? "clear" : "load";
      expect(descriptor.colorAttachments[0]).toMatchObject({ loadOp, storeOp: "store" });
      expect(descriptor.colorAttachments[0]?.resolveTarget).toBeDefined();
      expect(descriptor.depthStencilAttachment).toMatchObject({ depthLoadOp: loadOp, depthStoreOp: "store", stencilLoadOp: loadOp, stencilStoreOp: "store" });
      if (loadOp === "load") {
        expect(descriptor.colorAttachments[0]?.clearValue).toBeUndefined();
        expect(descriptor.depthStencilAttachment?.depthClearValue).toBeUndefined();
        expect(descriptor.depthStencilAttachment?.stencilClearValue).toBeUndefined();
      }
    }
  } finally { gpu.dispose(); vi.restoreAllMocks(); }
});

test("preserved MRT resolves and stores every multisample color", async () => {
  const gpu = await init();
  try {
    const output = target(gpu, { size: [4, 4], colors: [{ format: "rgba8unorm" }, { format: "rgba8unorm" }], msaa: true, preserveSamples: true });
    const descriptor = output.renderPassDescriptor({ preserve: true });
    expect(descriptor.colorAttachments).toHaveLength(2);
    for (const attachment of descriptor.colorAttachments) {
      expect(attachment).toMatchObject({ loadOp: "load", storeOp: "store" });
      expect(attachment?.resolveTarget).toBeDefined();
    }
  } finally { gpu.dispose(); }
});

test("ordinary MSAA retains discard and preservation/read-only refusal", async () => {
  const gpu = await init();
  try {
    for (const preserveSamples of [undefined, false]) {
      const output = target(gpu, { size: [4, 4], depth: "depth24plus-stencil8", msaa: true, preserveSamples });
      expect(output.preserveSamples).toBe(false);
      const descriptor = output.renderPassDescriptor();
      expect(descriptor.colorAttachments[0]?.storeOp).toBe("discard");
      expect(descriptor.depthStencilAttachment).toMatchObject({ depthStoreOp: "discard", stencilStoreOp: "discard" });
      expect(() => frame(gpu, current => current.pass({ target: output, clear: false }, () => undefined)))
        .toThrowError(expect.objectContaining({ code: "VGPU-PASS-PRESERVE-MSAA" }));
      expect(() => frame(gpu, current => current.pass({ target: output, depthReadOnly: true }, () => undefined)))
        .toThrowError(expect.objectContaining({ code: "VGPU-PASS-DEPTH-READONLY-MSAA" }));
    }
  } finally { gpu.dispose(); }
});

test("preserveSamples rejects malformed and dead options before allocating textures", async () => {
  const gpu = await init();
  const textures = vi.spyOn(gpu.gpu, "createTexture");
  try {
    for (const options of [
      { preserveSamples: true }, { msaa: false, preserveSamples: true },
      { msaa: true, preserveSamples: 1 }, { msaa: true, preserveSamples: "true" }, { msaa: true, preserveSamples: null },
    ]) {
      expect(() => target(gpu, { size: [4, 4], ...options } as never))
        .toThrowError(expect.objectContaining({ code: "VGPU-TARGET-PRESERVE-SAMPLES" }));
    }
    expect(textures).not.toHaveBeenCalled();
    expect(() => target(gpu, { size: [4, 4], preserveSamples: false })).not.toThrow();
  } finally { gpu.dispose(); vi.restoreAllMocks(); }
});

test("preservation options are snapshotted and survive resize with new attachments", async () => {
  const gpu = await init();
  try {
    const options = { size: [4, 4] as const, depth: true, msaa: true, preserveSamples: true };
    const output = target(gpu, options);
    const previous = [output.color, output.depth!];
    options.preserveSamples = false;
    output.resize([8, 8]);
    expect(output.preserveSamples).toBe(true);
    expect(output.size).toEqual([8, 8]);
    expect(output.color).not.toBe(previous[0]);
    expect(output.depth).not.toBe(previous[1]);
    for (const texture of previous) expect(() => texture.view).toThrow(/destroyed/);
    const descriptor = output.renderPassDescriptor({ preserve: true });
    expect(descriptor.colorAttachments[0]).toMatchObject({ loadOp: "load", storeOp: "store" });
    expect(descriptor.depthStencilAttachment).toMatchObject({ depthLoadOp: "load", depthStoreOp: "store" });
  } finally { gpu.dispose(); }
});

test("preserved MSAA read-only depth omits ops and accepts only non-writing draws", async () => {
  const gpu = await init();
  try {
    const output = target(gpu, { size: [4, 4], depth: "depth24plus-stencil8", msaa: true, preserveSamples: true });
    const reader = draw(gpu, { shader: SHADER, depth: { write: false }, vertices: 3 });
    const writer = draw(gpu, { shader: SHADER, vertices: 3 });
    const descriptor = output.renderPassDescriptor({ preserve: true, depthReadOnly: true });
    const depth = descriptor.depthStencilAttachment!;
    expect(depth).toMatchObject({ depthReadOnly: true, stencilReadOnly: true });
    for (const property of ["depthLoadOp", "depthStoreOp", "depthClearValue", "stencilLoadOp", "stencilStoreOp", "stencilClearValue"]) {
      expect(Object.hasOwn(depth, property)).toBe(false);
    }
    expect(() => frame(gpu, current => current.pass({ target: output, clear: false, depthReadOnly: true }, pass => pass.draw(reader)))).not.toThrow();
    expect(() => frame(gpu, current => current.pass({ target: output, clear: false, depthReadOnly: true }, pass => pass.draw(writer))))
      .toThrowError(expect.objectContaining({ code: "VGPU-PASS-DEPTH-READONLY" }));
    expect(() => frame(gpu, current => current.pass({ target: output, depthReadOnly: true, clearDepth: 0 }, () => undefined)))
      .toThrowError(expect.objectContaining({ code: "VGPU-PASS-DEPTH-READONLY" }));
  } finally { gpu.dispose(); }
});
