import { expect, test, vi } from "vitest";
import { createMockAdapter, frame, init, target, timer } from "../src/mock.ts";

async function controlledTimer() {
  const gpu = await init({ adapter: createMockAdapter({ features: ["timestamp-query"] }), requiredFeatures: ["timestamp-query"] });
  const releases: (() => void)[] = [];
  const original = gpu.gpu.createBuffer.bind(gpu.gpu);
  vi.spyOn(gpu.gpu, "createBuffer").mockImplementation(descriptor => {
    const buffer = original(descriptor);
    if (descriptor.label?.includes("staging")) {
      vi.spyOn(buffer, "mapAsync").mockImplementation(() => new Promise<void>(resolve => releases.push(resolve)));
    }
    return buffer;
  });
  const clock = timer(gpu);
  const output = target(gpu, { size: [2, 2] });
  const results: Readonly<Record<string, number>>[] = [];
  const errors: unknown[] = [];
  clock.onResults(spans => results.push(spans));
  gpu.onError(error => { errors.push(error); });
  const submit = vi.spyOn(gpu.gpu.queue, "submit");
  const render = (name: string) => frame(gpu, current => current.pass({ target: output, timer: clock.span(name) }, () => undefined));
  const close = async () => {
    for (const release of releases) release();
    await gpu.settled();
    clock.dispose();
    gpu.dispose();
    vi.restoreAllMocks();
  };
  return { releases, results, errors, submit, render, close };
}

// Desired follow-up: expose lost measurements and evaluate bounded staging growth.
// These reproduce current behavior; they do not claim adaptive buffering already exists.
test("three pending maps exhaust the timer staging ring and the fourth submitted frame loses telemetry without a result or error", async () => {
  const controlled = await controlledTimer();
  try {
    for (const name of ["first", "second", "third", "dropped"]) controlled.render(name);
    expect(controlled.submit).toHaveBeenCalledTimes(4);
    expect(controlled.releases).toHaveLength(3);
    expect(controlled.results).toEqual([]);
    expect(controlled.errors).toEqual([]);
    for (const release of controlled.releases) release();
    await vi.waitFor(() => expect(controlled.results).toHaveLength(3));
    expect(controlled.results.map(result => Object.keys(result))).toEqual([["first"], ["second"], ["third"]]);
    expect(controlled.errors).toEqual([]);
  } finally { await controlled.close(); }
});

test("a free out-of-order staging slot does not prevent drops while the next rotating slot remains busy", async () => {
  const controlled = await controlledTimer();
  try {
    for (const name of ["first", "second", "third"]) controlled.render(name);
    controlled.releases[1]!();
    await vi.waitFor(() => expect(controlled.results).toEqual([{ second: 1 }]));
    controlled.render("dropped-despite-free-slot");
    expect(controlled.submit).toHaveBeenCalledTimes(4);
    expect(controlled.releases).toHaveLength(3);
    expect(controlled.results).toEqual([{ second: 1 }]);
    // Rotation is stuck at slot zero even though slot one completed and is free.
    controlled.releases[0]!();
    await new Promise(resolve => setTimeout(resolve, 0));
    controlled.render("resumed");
    expect(controlled.releases).toHaveLength(4);
    controlled.releases[3]!();
    await vi.waitFor(() => expect(controlled.results).toEqual([{ second: 1 }, { resumed: 1 }]));
    // Older completions are intentionally discarded; they do not overwrite fresher telemetry.
    controlled.releases[0]!();
    controlled.releases[2]!();
    expect(controlled.errors).toEqual([]);
  } finally { await controlled.close(); }
});
