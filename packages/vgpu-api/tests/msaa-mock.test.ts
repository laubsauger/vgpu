import { prepareShader } from "@vgpu/wgsl/prepare";
import { expect, test, vi } from "vitest";
import { getMockGPUDeviceInstrumentation } from "@vgpu/core";
import type { Texture } from "@vgpu/core";
import { init, effect, frame, target } from "../src/mock.ts";

const SOLID = `
@fragment fn fs_main() -> @location(0) vec4f { return vec4f(1.0); }
`;

const MRT = `
struct Out { @location(0) a: vec4f, @location(1) b: vec4f }
@fragment fn fs_main() -> Out {
  var out: Out;
  out.a = vec4f(1.0, 0.0, 0.0, 1.0);
  out.b = vec4f(0.0, 1.0, 0.0, 1.0);
  return out;
}
`;

test("MSAA render pass descriptors resolve color and discard transient attachments while non-MSAA stores", async () => {
  const gpu = await init();
  try {
    const msaa = target(gpu, { size: [4, 4], depth: true, msaa: true });
    const plain = target(gpu, { size: [4, 4], depth: true });

    const msaaDesc = msaa.renderPassDescriptor();
    const plainDesc = plain.renderPassDescriptor();
    const msaaColor = msaaDesc.colorAttachments[0];
    const plainColor = plainDesc.colorAttachments[0];

    expect(msaaColor?.resolveTarget).toBeDefined();
    expect(msaaColor?.storeOp).toBe("discard");
    expect(msaaDesc.depthStencilAttachment?.depthStoreOp).toBe("discard");
    expect(plainColor?.resolveTarget).toBeUndefined();
    expect(plainColor?.storeOp).toBe("store");
    expect(plainDesc.depthStencilAttachment?.depthStoreOp).toBe("store");
  } finally {
    gpu.dispose();
  }
});

test("MSAA targets compile pipelines with sample count 4", async () => {
  const gpu = await init();
  try {
    const colorTarget = target(gpu, { size: [4, 4], depth: true, msaa: true });
    const draw = effect(gpu, prepareShader(SOLID), { label: "msaa-solid" });

    frame(gpu, (currentFrame) => currentFrame.pass({ target: colorTarget }, (pass) => pass.draw(draw)));

    const mock = getMockGPUDeviceInstrumentation(gpu.device.gpu);
    expect(mock.createRenderPipelineDescriptors.at(-1)?.multisample?.count).toBe(4);
  } finally {
    gpu.dispose();
  }
});

test("invalid runtime msaa values throw VGPU-TARGET-MSAA-INVALID", async () => {
  const gpu = await init();
  try {
    for (const msaa of [2, 8]) {
      expectThrown(() => target(gpu, { size: [4, 4], msaa } as never), { code: "VGPU-TARGET-MSAA-INVALID" });
    }
  } finally {
    gpu.dispose();
  }
});

test("MSAA target with blend keeps resolve descriptor and blend pipeline state", async () => {
  const gpu = await init();
  const renderPasses = spyRenderPassDescriptors(gpu.device.gpu);
  try {
    const colorTarget = target(gpu, { size: [4, 4], format: "rgba8unorm", msaa: true });
    const draw = effect(gpu, prepareShader(SOLID), { label: "msaa-blend", blend: "alpha" });

    frame(gpu, (currentFrame) => currentFrame.pass({ target: colorTarget }, (pass) => pass.draw(draw)));

    expect(renderPasses[0]?.colorAttachments[0]?.resolveTarget).toBeDefined();
    expect(renderPasses[0]?.colorAttachments[0]?.storeOp).toBe("discard");
    const mock = getMockGPUDeviceInstrumentation(gpu.device.gpu);
    const descriptor = mock.createRenderPipelineDescriptors.at(-1);
    expect(descriptor?.multisample?.count).toBe(4);
    expect(descriptor?.fragment?.targets[0]?.blend).toBeDefined();
  } finally {
    gpu.dispose();
    vi.restoreAllMocks();
  }
});

test("MRT MSAA targets resolve every color and compile all color states with sample count 4", async () => {
  const gpu = await init();
  const renderPasses = spyRenderPassDescriptors(gpu.device.gpu);
  try {
    const colorTarget = target(gpu, {
      size: [4, 4],
      colors: [{ format: "rgba8unorm" }, { format: "rgba8unorm" }],
      msaa: true,
    });
    const draw = effect(gpu, prepareShader(MRT), { label: "mrt-msaa" });

    frame(gpu, (currentFrame) => currentFrame.pass({ target: colorTarget }, (pass) => pass.draw(draw)));

    expect(renderPasses[0]?.colorAttachments).toHaveLength(2);
    for (const attachment of renderPasses[0]?.colorAttachments ?? []) {
      expect(attachment?.resolveTarget).toBeDefined();
      expect(attachment?.storeOp).toBe("discard");
    }
    const descriptor = getMockGPUDeviceInstrumentation(gpu.device.gpu).createRenderPipelineDescriptors.at(-1);
    expect(descriptor?.multisample?.count).toBe(4);
    expect(descriptor?.fragment?.targets).toHaveLength(2);
  } finally {
    gpu.dispose();
    vi.restoreAllMocks();
  }
});

test("omitted viewFormats stays absent from native texture descriptors", async () => {
  const gpu = await init();
  const textures = vi.spyOn(gpu.device.gpu, "createTexture");
  try {
    target(gpu, { size: [4, 4] });
    target(gpu, { size: [4, 4], colors: [{ format: "rgba8unorm" }, { format: "rgba16float" }], msaa: true, depth: true });
    expect(textures.mock.calls).toHaveLength(6);
    for (const [descriptor] of textures.mock.calls) {
      expect(Object.hasOwn(descriptor, "viewFormats")).toBe(false);
    }
  } finally {
    gpu.dispose();
    vi.restoreAllMocks();
  }
});

test("resolved color viewFormats are snapshotted and retained on resize", async () => {
  const gpu = await init();
  const textures = vi.spyOn(gpu.device.gpu, "createTexture");
  try {
    const viewFormats: GPUTextureFormat[] = ["rgba8unorm-srgb"];
    const source = target(gpu, { size: [4, 4], format: "rgba8unorm", viewFormats });
    const old = source.color;
    viewFormats[0] = "rgba16float";
    expect(source.color.viewFormats).toEqual(["rgba8unorm-srgb"]);
    expect(Object.isFrozen(source.color.viewFormats)).toBe(true);

    source.resize([8, 8]);
    expect(source.color).not.toBe(old);
    expect(source.color.viewFormats).toEqual(["rgba8unorm-srgb"]);
    expect(source.color.size).toEqual([8, 8]);
    expect(() => old.view).toThrow(/destroyed/);
    expect(textures.mock.calls.map(([descriptor]) => descriptor.viewFormats)).toEqual([
      ["rgba8unorm-srgb"], ["rgba8unorm-srgb"],
    ]);
  } finally {
    gpu.dispose();
    vi.restoreAllMocks();
  }
});

test("shared MRT viewFormats apply to resolved colors without reaching MSAA or depth attachments", async () => {
  const gpu = await init();
  const textures = vi.spyOn(gpu.device.gpu, "createTexture");
  try {
    const source = target(gpu, {
      size: [4, 4], colors: [{ format: "rgba8unorm" }, { format: "rgba8unorm" }],
      viewFormats: ["rgba8unorm-srgb"], msaa: true, depth: "depth24plus-stencil8",
    });
    expect(source.colors.map(color => color.viewFormats)).toEqual([["rgba8unorm-srgb"], ["rgba8unorm-srgb"]]);
    expect(source.depth!.viewFormats).toEqual([]);
    const descriptors = textures.mock.calls.map(([descriptor]) => descriptor);
    expect(descriptors).toHaveLength(5);
    for (const resolved of descriptors.slice(0, 2)) {
      expect(resolved.sampleCount).toBe(1);
      expect(resolved.viewFormats).toEqual(["rgba8unorm-srgb"]);
    }
    for (const attachment of descriptors.slice(2)) {
      expect(attachment.sampleCount).toBe(4);
      expect(Object.hasOwn(attachment, "viewFormats")).toBe(false);
    }
    const pass = source.renderPassDescriptor();
    for (const color of pass.colorAttachments) expect(color?.resolveTarget).toBeDefined();
    expect(pass.depthStencilAttachment?.stencilStoreOp).toBe("discard");
  } finally {
    gpu.dispose();
    vi.restoreAllMocks();
  }
});

test("invalid target viewFormats report the core diagnostic before native allocation", async () => {
  const gpu = await init();
  const textures = vi.spyOn(gpu.device.gpu, "createTexture");
  try {
    for (const viewFormats of [["rgba16float"], "rgba8unorm-srgb", null, [null]]) {
      expectThrown(() => target(gpu, { size: [4, 4], format: "rgba8unorm", viewFormats } as never), {
        code: "VGPU-TEXTURE-VIEW-FORMAT",
      });
    }
    expect(textures).not.toHaveBeenCalled();
  } finally {
    gpu.dispose();
    vi.restoreAllMocks();
  }
});

test("incompatible shared MRT viewFormats reject and destroy partially allocated colors", async () => {
  const gpu = await init();
  const allocated: Texture[] = [];
  const original = gpu.device.createTexture.bind(gpu.device);
  vi.spyOn(gpu.device, "createTexture").mockImplementation(options => {
    const texture = original(options);
    allocated.push(texture);
    return texture;
  });
  try {
    expectThrown(() => target(gpu, {
      size: [4, 4], colors: [{ format: "rgba8unorm" }, { format: "rgba16float" }],
      viewFormats: ["rgba8unorm-srgb"], msaa: true, depth: true,
    }), { code: "VGPU-TEXTURE-VIEW-FORMAT" });
    expect(allocated).toHaveLength(1);
    expect(() => allocated[0]!.view).toThrow(/destroyed/);
  } finally {
    gpu.dispose();
    vi.restoreAllMocks();
  }
});

function expectThrown(fn: () => unknown, shape: Record<string, unknown>): void {
  try {
    fn();
    throw new Error("Expected function to throw");
  } catch (error) {
    expect(error).toMatchObject(shape);
  }
}

function spyRenderPassDescriptors(device: GPUDevice): GPURenderPassDescriptor[] {
  const descriptors: GPURenderPassDescriptor[] = [];
  const originalCreateCommandEncoder = device.createCommandEncoder.bind(device);
  vi.spyOn(device, "createCommandEncoder").mockImplementation((descriptor?: GPUCommandEncoderDescriptor) => {
    const encoder = originalCreateCommandEncoder(descriptor);
    const originalBeginRenderPass = encoder.beginRenderPass.bind(encoder);
    return {
      ...encoder,
      beginRenderPass(renderPassDescriptor: GPURenderPassDescriptor): GPURenderPassEncoder {
        descriptors.push(renderPassDescriptor);
        return originalBeginRenderPass(renderPassDescriptor);
      },
    } as GPUCommandEncoder;
  });
  return descriptors;
}
