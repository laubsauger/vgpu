import { prepareShader } from "@vgpu/wgsl/prepare";
import { describe, expect, test } from "vitest";
import { draw, frame, init, target } from "../../src/node.ts";

const native = process.env.VGPU_DOCKER_TEST === "1" || process.env.VGPU_NATIVE_COMPUTE_TEST === "1";

function shader(depth: number, red: boolean) {
  return prepareShader(`
    @vertex fn vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
      var positions = array<vec2f, 3>(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
      return vec4f(positions[index], ${depth}, 1);
    }
    @fragment fn fragment(@builtin(position) position: vec4f) -> @location(0) vec4f {
      ${red ? "if (position.x >= 4.0) { discard; }" : ""}
      return ${red ? "vec4f(1, 0, 0, 1)" : "vec4f(0, 1, 0, 1)"};
    }
  `);
}

describe.skipIf(!native)("preserved MSAA on Dawn", () => {
  test.each([false, true])("two passes retain earlier color and occluding depth with depthReadOnly=%s", async depthReadOnly => {
    const gpu = await init();
    const errors: unknown[] = [];
    gpu.onError(error => { errors.push(error); });
    try {
      const output = target(gpu, { size: [8, 4], format: "rgba8unorm", depth: true, msaa: true, preserveSamples: true });
      const front = draw(gpu, { shader: shader(0.2, true), vertices: 3 });
      const behind = draw(gpu, { shader: shader(0.8, false), vertices: 3, depth: { write: !depthReadOnly, compare: "less" } });
      await frame(gpu, current => {
        current.pass(output, pass => pass.draw(front));
        current.pass({ target: output, clear: false, depthReadOnly }, pass => pass.draw(behind));
      }).done;
      const pixels = await output.color.read({ mipLevel: 0, region: "all" });
      for (let y = 0; y < 4; y++) {
        for (let x = 0; x < 8; x++) {
          const offset = (y * 8 + x) * 4;
          expect([...pixels.subarray(offset, offset + 4)], `pixel ${x},${y}`).toEqual(x < 4 ? [255, 0, 0, 255] : [0, 255, 0, 255]);
        }
      }
      await gpu.settled();
      expect(errors).toEqual([]);
    } finally { gpu.dispose(); }
  });
});
