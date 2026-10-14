import { prepareShader } from "@vgpu/wgsl/prepare";
import { describe, expect, test } from "vitest";
import { compute, frame, init, storage, timer } from "../../src/node.ts";

const native = process.env.VGPU_DOCKER_TEST === "1" || process.env.VGPU_NATIVE_COMPUTE_TEST === "1";

describe.skipIf(!native)("compute pass timer on a native device", () => {
  test("a timed compute pass validates and reports its span", async () => {
    const gpu = await init({ requiredFeatures: ["timestamp-query"] });
    const clock = timer(gpu);
    const results: Readonly<Record<string, number>>[] = [];
    clock.onResults(spans => results.push(spans));
    const errors: unknown[] = [];
    gpu.onError(error => { errors.push(error); });
    try {
      const values = storage(gpu, 4096 * 4, "read-write");
      const kernel = compute(gpu, prepareShader(`@group(0) @binding(0) var<storage, read_write> values: array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3u) { if (id.x < arrayLength(&values)) { values[id.x] = values[id.x] + 1.0; } }`), { set: { values } });
      frame(gpu, current => current.computePass({ label: "timed", timer: clock.span("kernel") }, pass => pass.dispatch(kernel, 64)));
      await gpu.settled();
      const deadline = performance.now() + 5_000;
      while (results.length === 0 && performance.now() < deadline) await new Promise(resolve => setTimeout(resolve, 5));
      expect(errors).toEqual([]);
      expect(results).toHaveLength(1);
      expect(Object.keys(results[0]!)).toEqual(["kernel"]);
      expect(results[0]!.kernel).toBeGreaterThanOrEqual(0);
    } finally { clock.dispose(); gpu.dispose(); }
  });
});
