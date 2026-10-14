import { prepareShader } from "@vgpu/wgsl/prepare";
import { describe, expect, test } from "vitest";
import { compute, effect, init, target } from "../../src/node.ts";

// Proposal reproduction: these assertions document current behavior on a native device, not an
// implemented change. vgpu creates a shader module (and its layouts) inside an error scope of its
// own and hands the error to the first pipeline that names the object. So a caller's own scope
// around the build sees nothing, and a second consumer of the same cached object is told nothing.

const native = process.env.VGPU_DOCKER_TEST === "1" || process.env.VGPU_NATIVE_COMPUTE_TEST === "1";

/** Everything vgpu reported, as `where`. */
async function reportsOf(build: (gpu: Awaited<ReturnType<typeof init>>) => void) {
  const gpu = await init();
  const reported: string[] = [];
  gpu.onError(error => { reported.push(String((error as { where?: unknown }).where)); });
  try {
    gpu.gpu.pushErrorScope("validation");
    build(gpu);
    const caller = await gpu.gpu.popErrorScope();
    await gpu.settled();
    await gpu.settled();
    return { caller, reported };
  } finally { gpu.dispose(); }
}

describe.skipIf(!native)("object errors and the caller's scope (proposal)", () => {
  test("two effects built from one invalid source: the caller's scope sees nothing, and only the first is told", async () => {
    const broken = prepareShader("@fragment fn fs() -> @location(0) vec4f { let x: f32 = 1u; return vec4f(x); }", "broken.wgsl");
    const { caller, reported } = await reportsOf(gpu => {
      const output = target(gpu, { size: [2, 2] });
      effect(gpu, broken, { label: "first" }).compileSync(output);
      effect(gpu, broken, { label: "second" }).compileSync(output);
    });
    expect(caller).toBeNull();
    expect(reported).toEqual(["first.compileSync"]);
  });

  test("two kernels built from one invalid source: the same, for compute", async () => {
    const broken = prepareShader("@compute @workgroup_size(1) fn main() { let y: f32 = 2u; }", "kernel.wgsl");
    const { caller, reported } = await reportsOf(gpu => {
      compute(gpu, broken, { label: "k1" }).compileSync();
      compute(gpu, broken, { label: "k2" }).compileSync();
    });
    expect(caller).toBeNull();
    expect(reported).toEqual(["k1.compileSync"]);
  });
});
