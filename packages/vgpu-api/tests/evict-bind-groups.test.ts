import { expect, test } from "vitest";
import { getMockGPUDeviceInstrumentation } from "@vgpu/core";
import { compute } from "../src/compute.ts";
import { effect } from "../src/effect.ts";
import { frame } from "../src/frame.ts";
import { init } from "../src/mock.ts";
import { storage } from "../src/storage.ts";
import { target } from "../src/target-offscreen.ts";

const CAMERA_SHADER = `
struct Camera { value: f32 }
@group(0) @binding(0) var<uniform> camera: Camera;
@fragment fn main(@location(0) uv: vec2f) -> @location(0) vec4f { return vec4f(camera.value, uv, 1.0); }
`;

const COPY_SHADER = `
@group(0) @binding(0) var<storage, read> src: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4f>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3u) { dst[id.x] = src[id.x]; }
`;

// First in the file on purpose: the draw and the compute below are each the first of their kind in this
// module, so both take id 1 from their separate counters. They share one gpu-wide bind-group cache, and
// this is the collision the `compute:` key prefix exists for.
test("evicting a compute's bind groups leaves a draw with the same numeric id alone", async () => {
  const gpu = await init();
  const drawable = effect(gpu, CAMERA_SHADER, { label: "first-draw" });
  const kernel = compute(gpu, COPY_SHADER, { label: "first-compute" });
  const colorTarget = target(gpu, { size: [4, 4] });
  const camera = gpu.device.createBuffer({ size: 16, usage: ["uniform", "copy_dst"] });
  const mock = getMockGPUDeviceInstrumentation(gpu.device.gpu);

  drawable.set({ camera });
  kernel.set({ src: storage(gpu, 16, "read"), dst: storage(gpu, 16) });
  frame(gpu, (f) => f.pass({ target: colorTarget }, (p) => p.draw(drawable)));
  kernel.dispatch(1);
  expect(mock.calls.createBindGroup).toBe(2);

  kernel.evictBindGroups();
  frame(gpu, (f) => f.pass({ target: colorTarget }, (p) => p.draw(drawable)));
  expect(mock.calls.createBindGroup).toBe(2); // the draw's entry survived the compute's eviction
  kernel.dispatch(1);
  expect(mock.calls.createBindGroup).toBe(3); // the compute's did not
  gpu.dispose();
});

test("evictBindGroups drops only the bind groups: the next draw rebuilds them and nothing else is recreated", async () => {
  const gpu = await init();
  const drawable = effect(gpu, CAMERA_SHADER, { label: "cameraPass" });
  const colorTarget = target(gpu, { size: [4, 4] });
  const camera = gpu.device.createBuffer({ size: 16, usage: ["uniform", "copy_dst"] });
  const mock = getMockGPUDeviceInstrumentation(gpu.device.gpu);

  drawable.set({ camera });
  frame(gpu, (f) => f.pass({ target: colorTarget }, (p) => p.draw(drawable)));
  frame(gpu, (f) => f.pass({ target: colorTarget }, (p) => p.draw(drawable)));
  expect(mock.calls.createBindGroup).toBe(1);
  const pipelines = mock.calls.createRenderPipeline + mock.calls.createRenderPipelineAsync;
  const layouts = mock.calls.createBindGroupLayout;

  drawable.evictBindGroups();
  // Bindings are untouched: no VGPU-R1-BINDING-NEVER-SET, no re-set needed.
  frame(gpu, (f) => f.pass({ target: colorTarget }, (p) => p.draw(drawable)));
  expect(mock.calls.createBindGroup).toBe(2);
  expect(mock.calls.createRenderPipeline + mock.calls.createRenderPipelineAsync).toBe(pipelines);
  expect(mock.calls.createBindGroupLayout).toBe(layouts);

  // Idempotent and harmless before anything was drawn or after a previous eviction.
  drawable.evictBindGroups();
  drawable.evictBindGroups();
  frame(gpu, (f) => f.pass({ target: colorTarget }, (p) => p.draw(drawable)));
  expect(mock.calls.createBindGroup).toBe(3);
  gpu.dispose();
});

test("evictBindGroups on one draw does not touch another draw's entries for the same resources", async () => {
  const gpu = await init();
  const a = effect(gpu, CAMERA_SHADER, { label: "a" });
  const b = effect(gpu, CAMERA_SHADER, { label: "b" });
  const colorTarget = target(gpu, { size: [4, 4] });
  const camera = gpu.device.createBuffer({ size: 16, usage: ["uniform", "copy_dst"] });
  const mock = getMockGPUDeviceInstrumentation(gpu.device.gpu);

  a.set({ camera });
  b.set({ camera });
  frame(gpu, (f) => f.pass({ target: colorTarget }, (p) => { p.draw(a); p.draw(b); }));
  expect(mock.calls.createBindGroup).toBe(2);

  a.evictBindGroups();
  frame(gpu, (f) => f.pass({ target: colorTarget }, (p) => { p.draw(a); p.draw(b); }));
  expect(mock.calls.createBindGroup).toBe(3);
  gpu.dispose();
});
