import { describe, expect, test } from "vitest";
import { prepareShader } from "@vgpu/wgsl/prepare";
import { effect, frame, init, target } from "../../src/node.ts";

const native = process.env.VGPU_DOCKER_TEST === "1" || process.env.VGPU_NATIVE_COMPUTE_TEST === "1";
const COPY = prepareShader(`@group(0) @binding(0) var src: texture_2d<f32>;
@fragment fn fs(@location(0) uv: vec2f) -> @location(0) vec4f { return textureLoad(src, vec2i(uv * vec2f(textureDimensions(src))), 0); }`);

describe.skipIf(!native)("compatible target views on the native device", () => {
  test("a declared linear view preserves stored sRGB bytes across target resize", async () => {
    const gpu = await init();
    try {
      const source = target(gpu, { size: [4, 4], format: "rgba8unorm-srgb", viewFormats: ["rgba8unorm"] });
      const output = target(gpu, { size: [4, 4], format: "rgba8unorm" });
      const copy = effect(gpu, COPY);
      for (const size of [[4, 4], [8, 4]] as const) {
        source.resize(size); output.resize(size);
        frame(gpu, f => f.pass({ target: source, clear: [0.5, 0.5, 0.5, 1] }, () => undefined));
        await gpu.settled();
        const stored = await source.color.read({ mipLevel: 0, region: "all" });
        copy.set({ src: source.color });
        frame(gpu, f => f.pass(output, copy)); await gpu.settled();
        const decoded = await output.color.read({ mipLevel: 0, region: "all" });
        copy.set({ src: source.color.createView({ format: "rgba8unorm" }) });
        frame(gpu, f => f.pass(output, copy)); await gpu.settled();
        const linear = await output.color.read({ mipLevel: 0, region: "all" });
        expect(linear).toEqual(stored);
        expect(decoded[0]).toBeLessThan(stored[0]!);
        expect(linear[3]).toBe(255);
      }
    } finally { await gpu.dispose(); }
  });
});
