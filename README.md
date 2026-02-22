# FE-Kin

A kinematic library intended to provide kinematics (position, velocity, and acceleration) with
a reasonably high level of performance for "tree-like" systems. There are always trade-offs
for different implementations, this particular experiment is aimed at being able to compute
all velocities and accelerations. It expresses the rigid-body transforms using "twist" math,
that is a rewriting of Lie algebra solutions into 3D vector and matrix operations.

# Learnings thus far

Somewhat surprising is that 4x4 matrices actually seem to operate faster than scalar and
vector operations. This is likely due to what is easily optimized through naive implementation
of various operations. I think most surprising is that the SIMD operations don't naturally
kick-in for quaternion and vector formed operations. See the "benchmarks" branch to see
some of these tests.

# Running the benchmark in `main.rs`

`src/main.rs` is the benchmark harness. For fair and consistent runs, keep the environment
stable and measure only the released binary.

## Recommended process (Linux)

1. Use a fixed toolchain and clean release build.
   ```bash
   rustc --version
   cargo clean
   CARGO_INCREMENTAL=0 cargo build --release
   ```

2. Pin execution to one CPU core to reduce scheduler noise.
   ```bash
   taskset --cpu-list 2 ./target/release/fekin
   ```

3. Run many samples and compare medians, not single runs.
   ```bash
   scripts/run_benchmark.sh --warmup 5 --runs 30 --core 2
   ```

   The script runs warmups, measures each run with a pinned CPU core, and prints:
   mean, stddev, median, min, max, and total time.

   Optional flags:
   ```bash
   scripts/run_benchmark.sh --no-build --per-run
   scripts/run_benchmark.sh --binary ./target/release/fekin --output /tmp/fekin-bench.out
   ```

## Optional system-level controls (Linux)

These reduce frequency scaling variance, but require `sudo`.

```bash
sudo cpupower frequency-set -g performance
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
```

Restore defaults after benchmarking:

```bash
sudo cpupower frequency-set -g schedutil
echo 0 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
```

## Notes for fair comparisons

- Compare binaries built with the same profile (`--release`).
- Avoid background load (browser tabs, updates, other CPU-heavy tasks).
- Re-run with the same tree parameters in `src/main.rs`.
- When comparing branches, rebuild both the same way and run both with the same pinned-core command.
