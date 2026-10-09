import { prepareShader } from "@vgpu/wgsl/prepare";
import { describe, expect, test } from "vitest";
import { compute, init } from "../../src/node.ts";

const COPY = `
@group(0) @binding(0) var<storage, read_write> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@compute @workgroup_size(1) fn main() { dst[0] = src[0] + 7u; }
`;

describe.skipIf(process.env.VGPU_NATIVE_COMPUTE_TEST !== "1")("disjoint buffer range proposal on Dawn", () => {
  test.each([
    { name: "disjoint writable ranges", destinationOffset: 512, readOnlySource: false, legal: true },
    { name: "overlapping writable ranges", destinationOffset: 0, readOnlySource: false, legal: false },
    { name: "disjoint ranges with mixed storage access", destinationOffset: 512, readOnlySource: true, legal: false },
  ])("native WebGPU validates $name", async ({ destinationOffset, readOnlySource, legal }) => {
    const gpu = await init();
    const device = gpu.gpu;
    const arena = device.createBuffer({ size: 1024, usage: 0x80 | 0x08 | 0x04 });
    const readback = device.createBuffer({ size: 4, usage: 0x01 | 0x08 });
    try {
      device.queue.writeBuffer(arena, 0, new Uint32Array([35]));
      device.pushErrorScope("validation");
      const code = readOnlySource ? COPY.replace("@binding(0) var<storage, read_write>", "@binding(0) var<storage, read>") : COPY;
      const module = device.createShaderModule({ code });
      const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
      const bindings = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: arena, offset: 0, size: 16 } },
          { binding: 1, resource: { buffer: arena, offset: destinationOffset, size: 16 } },
        ],
      });
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindings);
      pass.dispatchWorkgroups(1);
      pass.end();
      encoder.copyBufferToBuffer(arena, destinationOffset, readback, 0, 4);
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      const validation = await device.popErrorScope();
      if (!legal) {
        expect(validation).not.toBeNull();
        expect(validation!.message).toMatch(readOnlySource ? /synchronization scope/i : /alias|overlap/i);
      } else {
        expect(validation).toBeNull();
        await readback.mapAsync(0x01);
        expect(new Uint32Array(readback.getMappedRange())[0]).toBe(42);
        readback.unmap();
      }
    } finally {
      arena.destroy();
      readback.destroy();
      gpu.dispose();
    }
  });

  test("vgpu rejects the same legal disjoint bindings before encoding", async () => {
    const gpu = await init();
    const arena = gpu.gpu.createBuffer({ size: 1024, usage: 0x80 | 0x08 });
    try {
      const pipeline = compute(gpu, prepareShader(COPY), {
        set: { src: { buffer: arena, offset: 0, size: 16 }, dst: { buffer: arena, offset: 512, size: 16 } },
      });
      await pipeline.compile();
      expect(() => pipeline.dispatch(1)).toThrow(/alias/i);
    } finally {
      arena.destroy();
      gpu.dispose();
    }
  });
});
