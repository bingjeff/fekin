# C++ Benchmarks (Eigen)

This folder contains C++ benchmark binaries that mirror the Rust transform benchmarks.

Eigen is vendored in `third_party/eigen`.

## Build

From the repository root:

```bash
make -C cpp
```

Or manually:

```bash
g++ -O3 -march=native -std=c++20 -I third_party/eigen \
  cpp/eigen_transform_chain_bench.cpp \
  -o cpp/eigen_transform_chain_bench
```

## Run

```bash
./cpp/eigen_transform_chain_bench
```
