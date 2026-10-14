import { prepareShader } from "@vgpu/wgsl/prepare";
import { expect, test, vi } from "vitest";
import { compute, createMockAdapter, frame, init, target, timer } from "../src/mock.ts";

/** A timestamp-capable mock gpu whose frame encoders record every pass descriptor they open. */
async function recordingGpu() {
  const gpu = await init({ adapter: createMockAdapter({ features: ["timestamp-query"] }), requiredFeatures: ["timestamp-query"] });
  const computeDescriptors: GPUComputePassDescriptor[] = [];
  const renderDescriptors: GPURenderPassDescriptor[] = [];
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
  return { gpu, computeDescriptors, renderDescriptors };
}

const KERNEL = "@compute @workgroup_size(1) fn main() {}";

test("a compute pass's timer is a span of its frame, between the frame's timed render passes", async () => {
  const { gpu, computeDescriptors, renderDescriptors } = await recordingGpu();
  const clock = timer(gpu);
  const results: Readonly<Record<string, number>>[] = [];
  clock.onResults(spans => results.push(spans));
  try {
    const output = target(gpu, { size: [2, 2] });
    const kernel = compute(gpu, prepareShader(KERNEL));
    frame(gpu, current => {
      current.pass({ target: output, timer: clock.span("before") }, () => undefined);
      current.computePass({ label: "compute-work", timer: clock.span("compute") }, pass => pass.dispatch(kernel, 1));
      current.pass({ target: output, timer: clock.span("after") }, () => undefined);
    });
    await gpu.settled();
    // The compute pass takes the next pair of the frame's one query set, in encode order.
    expect(computeDescriptors.map(descriptor => [descriptor.label, descriptor.timestampWrites?.beginningOfPassWriteIndex, descriptor.timestampWrites?.endOfPassWriteIndex]))
      .toEqual([["compute-work", 2, 3]]);
    expect(renderDescriptors.map(descriptor => [descriptor.timestampWrites?.beginningOfPassWriteIndex, descriptor.timestampWrites?.endOfPassWriteIndex])).toEqual([[0, 1], [4, 5]]);
    expect(Object.keys(results[0] ?? {}).sort()).toEqual(["after", "before", "compute"]);
  } finally { clock.dispose(); gpu.dispose(); vi.restoreAllMocks(); }
});

test("a compute pass without a timer opens with its label only", async () => {
  const { gpu, computeDescriptors } = await recordingGpu();
  try {
    const kernel = compute(gpu, prepareShader(KERNEL));
    frame(gpu, current => current.computePass({ label: "untimed" }, pass => pass.dispatch(kernel, 1)));
    await gpu.settled();
    expect(computeDescriptors).toEqual([{ label: "untimed" }]);
  } finally { gpu.dispose(); vi.restoreAllMocks(); }
});

test("a malformed compute timer is refused by name, as a render pass's is", async () => {
  const gpu = await init();
  try {
    const runtimeOptions = { label: "invalid-timer", timer: { name: "not-a-real-span" } } as never;
    expect(() => frame(gpu, current => current.computePass(runtimeOptions, () => undefined)))
      .toThrow(/FrameComputePassOptions\.timer received/);
  } finally { gpu.dispose(); }
});

test("a compute pass that fails rolls its frame's timer back: no duration for work that never ran", async () => {
  const { gpu } = await recordingGpu();
  const clock = timer(gpu);
  const results: Readonly<Record<string, number>>[] = [];
  clock.onResults(spans => results.push(spans));
  try {
    const kernel = compute(gpu, prepareShader(KERNEL));
    const current = frame(gpu);
    current.computePass({ timer: clock.span("ran") }, pass => pass.dispatch(kernel, 1));
    expect(() => current.computePass({ timer: clock.span("failed") }, () => { throw new Error("kernel body failed"); }))
      .toThrow("kernel body failed");
    current.submit();
    await gpu.settled();
    expect(results).toEqual([]);
    // The timer is whole again on the next frame.
    frame(gpu, next => next.computePass({ timer: clock.span("ran") }, pass => pass.dispatch(kernel, 1)));
    await gpu.settled();
    expect(results.map(spans => Object.keys(spans))).toEqual([["ran"]]);
  } finally { clock.dispose(); gpu.dispose(); vi.restoreAllMocks(); }
});
