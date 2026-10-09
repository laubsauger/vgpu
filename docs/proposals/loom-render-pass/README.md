# Pipeline encoding measurements

The original seven-trial results remain in `native.json` and `mock.json`. The follow-up
uses 15 paired process trials per workload, alternating reference/candidate order,
at 50, 250 and 2,000 draws per pass. Multi-draw frames contain three passes. A separate
single-draw control has one pass per frame. Every sample is retained.

## Native results

The percent column is the median of the 15 paired percentage changes. The time column
is the median paired CPU encoding time saved for the complete three-pass frame. These
are different statistics; neither is calculated by dividing independent column medians.

| Draws/pass | Workload | Paired CPU change | Exploratory 95% interval | CPU saved/frame |
| ---: | --- | ---: | --- | ---: |
| 50 | Different drawables, same pipeline | -6.8% | -7.6% to -6.3% | +0.041 ms |
| 50 | Same drawable, same pipeline | -8.7% | -10.0% to -7.6% | +0.044 ms |
| 50 | Alternating pipelines | +0.8% | -0.7% to +1.3% | -0.004 ms |
| 250 | Different drawables, same pipeline | -9.7% | -10.2% to -8.8% | +0.235 ms |
| 250 | Same drawable, same pipeline | -10.6% | -11.2% to -10.2% | +0.217 ms |
| 250 | Alternating pipelines | +0.7% | -3.3% to +3.2% | -0.017 ms |
| 2000 | Different drawables, same pipeline | -6.9% | -10.2% to -4.7% | +1.743 ms |
| 2000 | Same drawable, same pipeline | -11.0% | -11.8% to -9.2% | +1.841 ms |
| 2000 | Alternating pipelines | +0.7% | -0.1% to +3.1% | -0.124 ms |

The same-pipeline reductions persist across all tested counts. Alternating-pipeline
intervals include zero; this does not establish zero overhead. The single-draw native
control is inconclusive (paired median −1.6%, interval −7.5% to +9.1%).

Mock results isolate JavaScript bookkeeping from native driver-call savings. Some costs
increase: repeated-draw medians are +4.8%, +0.2% and +1.9% at 50/250/2,000 draws;
alternating-pipeline medians are +4.7%, +3.3% and +1.2%. Several intervals exclude zero.
These increases should not be dismissed as machine contention. They support a small
JavaScript tradeoff when native setters offer little or no savings.

## Repetition, outliers and uncertainty

Each process warms up before measuring. A multi-draw trial averages 60 timed frames;
a single-draw trial averages 1,000. Headline comparisons use process-trial medians,
not one run or an average across all frames from every process.

`statistics.json` reports means, medians, MAD, quartiles, per-trial paired percentage
changes and Tukey 1.5-IQR outlier flags. All flagged samples remain in the results.
A separate sensitivity field removes a pair when either member is flagged; it is not
used to select the headline. The analysis uses 20,000 fixed-seed paired bootstrap
resamples. These exploratory intervals are coarse at 15 pairs, are not adjusted for
multiple comparisons, and cannot establish platform-wide guarantees. Outlier checks
apply to process-trial averages; individual timed-frame samples were not retained.

## Measurement overhead

The harness reads `performance.now()` twice around the complete frame. There is no
per-draw timing probe. Each trial also calibrates 20,000 empty intervals after encoding.
Their average is usually about 35–43 ns per frame (single-draw trials about 35–47 ns).
That is approximately 0.23–0.29 ns/draw at 150 draws/frame, or 0.006–0.007 ns/draw at
6,000 draws/frame. It cannot explain a several-nanosecond per-draw JavaScript increase.

Raw trials include the empty-interval distribution summary, estimated observer cost
per draw and an estimated clock-adjusted value. The headline remains unadjusted: the
calibration cannot remove JIT, GC, scheduling, tracing or GPU timestamp overhead.
The paired reference/candidate measurements use the same boundary probes, so this
small fixed cost largely cancels. CPU timing excludes initialization, queue completion
and pixel readback. These are not GPU-throughput or end-to-end frame-rate results.

## Practical scope

Exact repeated-drawable rendering is a favorable stress case. The more representative
opportunity is consecutive different objects sharing a native pipeline: compatible
meshes/materials, sprites or UI quads, and batched shadow draws. Instancing already
reduces many objects to one draw. Fullscreen effects in separate passes have no repeated
pipeline binding to remove. Pipeline switching also removes this optimization’s benefit.

Loom groups compatible geometry and shadow draws, so it can use the different-drawable
case. Its ordinary one-draw post-processing passes do not benefit. Upstream #489 gives
one concrete many-draw workload; it does not establish how frequent this pattern is
across vGPU consumers. No representative application population or frame-rate change
was measured here.

## Reproduction and correctness

`calibrated/` contains all eight data files, with harness/built-module hashes and clock
calibrations. Run `scripts/bench-render-pass-state.mjs` against separately built pristine
canary and this proposal, using `--runs 15 --count 50` (then 250/2,000), and
`--mode unique,repeated,alternating`. Repeat for `mock.js`; run `--mode single` separately.
Analyze with `node scripts/analyze-render-benchmarks.mjs <data files...>`. Avoid concurrent
benchmarks or builds.

Every native output matches its exact expected color and both entries’ pixel hashes.
Mock measurements produce no images. Runtime source is unchanged from the tested
prototype: seven encoder-state tests, three native correctness tests and hard client
bundle budgets pass. The follow-up modifies only scripts, evidence and this document.
