import { prepareShader } from "@vgpu/wgsl/prepare";
import { expect, test, vi } from "vitest";
import { compute, createMockAdapter, frame, init, target, timer } from "../src/mock.ts";

// Proposal reproduction: computePass currently accepts only label. A runtime timer key is
// silently ignored; these assertions document the gap, not an implemented compute timer API.
test("a runtime compute timer key produces no timestamp writes or span result between timed render passes", async () => {
  const gpu = await init({ adapter: createMockAdapter({ features: ["timestamp-query"] }), requiredFeatures: ["timestamp-query"] });
  const clock = timer(gpu);
  const computeDescriptors: GPUComputePassDescriptor[] = [];
  const renderDescriptors: GPURenderPassDescriptor[] = [];
  const results: Readonly<Record<string, number>>[] = [];
  clock.onResults(spans => results.push(spans));
  const original = gpu.gpu.createCommandEncoder.bind(gpu.gpu);
  vi.spyOn(gpu.gpu, "createCommandEncoder").mockImplementation(options => {
    const encoder = original(options);
    const beginCompute = encoder.beginComputePass.bind(encoder);
    const beginRender = encoder.beginRenderPass.bind(encoder);
    vi.spyOn(encoder, "beginComputePass").mockImplementation(descriptor => {
      computeDescriptors.push(descriptor!);
      return beginCompute(descriptor);
    });
    vi.spyOn(encoder, "beginRenderPass").mockImplementation(descriptor => {
      renderDescriptors.push(descriptor);
      return beginRender(descriptor);
    });
    return encoder;
  });
  try {
    const output = target(gpu, { size: [2, 2] });
    const kernel = compute(gpu, prepareShader("@compute @workgroup_size(1) fn main() {}"));
    // Extra keys are valid JavaScript input; timer is deliberately not part of the typed API.
    const runtimeOptions = { label: "compute-work", timer: clock.span("compute") };
    frame(gpu, current => {
      current.pass({ target: output, timer: clock.span("before") }, () => undefined);
      current.computePass(runtimeOptions, pass => pass.dispatch(kernel, 1));
      current.pass({ target: output, timer: clock.span("after") }, () => undefined);
    });
    await gpu.settled();
    expect(computeDescriptors).toEqual([{ label: "compute-work" }]);
    expect(renderDescriptors.map(descriptor => [descriptor.timestampWrites?.beginningOfPassWriteIndex, descriptor.timestampWrites?.endOfPassWriteIndex])).toEqual([[0, 1], [2, 3]]);
    expect(results).toEqual([{ before: 1, after: 5 }]);
    expect(Object.hasOwn(results[0]!, "compute")).toBe(false);
  } finally { clock.dispose(); gpu.dispose(); vi.restoreAllMocks(); }
});

test("computePass also ignores a malformed runtime timer instead of validating it", async () => {
  const gpu = await init();
  try {
    const runtimeOptions = { label: "invalid-timer", timer: { name: "not-a-real-span" } };
    expect(() => frame(gpu, current => current.computePass(runtimeOptions, () => undefined))).not.toThrow();
    await gpu.settled();
  } finally { gpu.dispose(); }
});
