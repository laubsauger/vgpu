import { getMockGPUDeviceInstrumentation } from "@vgpu/core";
import { prepareShader } from "@vgpu/wgsl/prepare";
import { expect, test } from "vitest";
import { effect, frame, init, target } from "../src/mock.ts";
import { MAX_OWNER_ENTRIES } from "../src/bind-cache.ts";

const SHADER = prepareShader(`
@group(0) @binding(0) var<uniform> value: vec4f;
@fragment fn main() -> @location(0) vec4f { return value; }
`);

test("a reusable consumer retains historical bind groups across A to B to A rebinding", async () => {
  const gpu = await init();
  const a = gpu.gpu.createBuffer({ size: 16, usage: 0x40 | 0x08 });
  const b = gpu.gpu.createBuffer({ size: 16, usage: 0x40 | 0x08 });
  try {
    const out = target(gpu, { size: [4, 4] });
    const consumer = effect(gpu, SHADER, { set: { value: a } });
    const mock = getMockGPUDeviceInstrumentation(gpu.gpu);
    const render = () => frame(gpu, current => current.pass(out, consumer));
    const before = mock.calls.createBindGroup;

    await render().done;
    expect(mock.calls.createBindGroup).toBe(before + 1);
    consumer.set({ value: b });
    await render().done;
    expect(mock.calls.createBindGroup).toBe(before + 2);
    consumer.set({ value: a });
    await render().done;
    expect(mock.calls.createBindGroup).toBe(before + 2);

    // Neither source was destroyed. A rebind alone retained the old A group for reuse.
    consumer.set({ value: b });
    await render().done;
    expect(mock.calls.createBindGroup).toBe(before + 2);
    consumer.dispose();
    expect(() => consumer.set({ value: a })).toThrow(/disposed/i);
    expect(() => render()).toThrow(/disposed/i);
  } finally {
    a.destroy();
    b.destroy();
    gpu.dispose();
  }
});

test("live-consumer history has the current owner bound rather than unlimited retention", async () => {
  const gpu = await init();
  const resources = Array.from({ length: MAX_OWNER_ENTRIES + 1 }, () => gpu.gpu.createBuffer({ size: 16, usage: 0x40 | 0x08 }));
  try {
    const out = target(gpu, { size: [4, 4] });
    const consumer = effect(gpu, SHADER, { set: { value: resources[0]! } });
    const mock = getMockGPUDeviceInstrumentation(gpu.gpu);
    const before = mock.calls.createBindGroup;
    const render = async (index: number) => {
      consumer.set({ value: resources[index]! });
      await frame(gpu, current => current.pass(out, consumer)).done;
    };

    for (let index = 0; index < MAX_OWNER_ENTRIES; index++) await render(index);
    expect(mock.calls.createBindGroup).toBe(before + MAX_OWNER_ENTRIES);
    await render(0); // Refresh A; resource1 becomes the oldest cached group.
    expect(mock.calls.createBindGroup).toBe(before + MAX_OWNER_ENTRIES);
    await render(MAX_OWNER_ENTRIES);
    expect(mock.calls.createBindGroup).toBe(before + MAX_OWNER_ENTRIES + 1);
    await render(2); // An entry that remains within the bound is still reusable.
    expect(mock.calls.createBindGroup).toBe(before + MAX_OWNER_ENTRIES + 1);
    await render(1); // The oldest group was evicted even though its buffer is still live.
    expect(mock.calls.createBindGroup).toBe(before + MAX_OWNER_ENTRIES + 2);
    consumer.dispose();
  } finally {
    for (const resource of resources) resource.destroy();
    gpu.dispose();
  }
});
