# v0.11 Optional Docker ODAS (experimental)

The SDK has **no Docker or ODAS installation dependency**. `oda-odas:dev` is a user-provided Linux ARM64 image: set `--image` to the actual local tag. The runner never touches the existing running Onda Spatial v3 ODAS container. It creates a fresh `docker run --rm` per recording, disables networking, restricts permissions/capabilities, limits memory/CPU/PIDs, and mounts only a temporary workspace. Existing ODAS services and sockets remain untouched.

## Usage

```sh
python benchmarks/odas_docker_v011.py --limit 3
python benchmarks/odas_docker_v011.py --limit 3 --enable-odas --image onda-odas:dev
```

Inspect `benchmark_results/v011_odas/summary.json`, `results.csv`, and all `odas_###_frames.json`. Failures are reported as errors and never receive synthetic ODAS detections. The Docker image must contain `/opt/odas/build-linux/bin/odaslive` and accept the generated configuration. **The image is not available in this development environment, so live ODAS execution is unverified.** If the image needs writable `/tmp` or other storage, the container's `--read-only` option may require adding a bounded `--tmpfs /tmp`; the wrapper deliberately does not silently relax sandbox constraints.

## Scientific limitations

ODAS outputs are captured frame-by-frame, and the runner preserves the raw timeStamp counter scaled by the configured raw hop. ODAS has internal buffering and latency; these timestamps are **not yet calibrated** to acoustic capture instants. The CLI's final-frame quick view is **not temporally equivalent** to a full-window SRP estimate and must not be used for performance rankings. The correct follow-on is to use common overlapping evaluation windows with measured ODAS latency, timestamp alignment, and source-count confidence criteria.

Generated ODAS configurations are explicit about geometry and 1-based `mapping` ordering. Before treating any number as authoritative, verify exact microphone coordinates, audio sample representation, configured filters, ODAS executable/configuration compatibility, ODAS frame timing, and JSON output format. `max_sources=4` is an output limit, not a guarantee of resolving four sources. Two-dimensional azimuth comparison drops ODAS elevation and should be constrained to an appropriate horizontal-plane benchmark.

For moving microphones, direct ODAS local hypotheses are **not world-frame**. A calibrated, timestamped pose transform is needed at each ODAS observation's actual acoustic acquisition time. This release is restricted to static-array cases.
