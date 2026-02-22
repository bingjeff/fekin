#!/usr/bin/env bash
set -euo pipefail

RUNS=30
WARMUP=5
CORE=2
OUTPUT_FILE="/tmp/fekin-bench.out"
BUILD=1
BINARY="./target/release/fekin"
PRINT_PER_RUN=0

usage() {
  cat <<'USAGE'
Usage: scripts/run_benchmark.sh [options]

Options:
  -r, --runs N          Number of measured runs (default: 30)
  -w, --warmup N        Number of warmup runs (default: 5)
  -c, --core ID         CPU core to pin with taskset (default: 2)
  -o, --output PATH     Where benchmark program stdout is redirected
                        (default: /tmp/fekin-bench.out)
  -b, --binary PATH     Benchmark binary path (default: ./target/release/fekin)
      --no-build        Skip release build step
      --per-run         Print each measured run time
  -h, --help            Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -r|--runs)
      RUNS="$2"
      shift 2
      ;;
    -w|--warmup)
      WARMUP="$2"
      shift 2
      ;;
    -c|--core)
      CORE="$2"
      shift 2
      ;;
    -o|--output)
      OUTPUT_FILE="$2"
      shift 2
      ;;
    -b|--binary)
      BINARY="$2"
      shift 2
      ;;
    --no-build)
      BUILD=0
      shift
      ;;
    --per-run)
      PRINT_PER_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if ! [[ "$RUNS" =~ ^[0-9]+$ ]] || ! [[ "$WARMUP" =~ ^[0-9]+$ ]]; then
  echo "Error: --runs and --warmup must be non-negative integers." >&2
  exit 2
fi

if [[ "$RUNS" -eq 0 ]]; then
  echo "Error: --runs must be at least 1." >&2
  exit 2
fi

if ! command -v taskset >/dev/null 2>&1; then
  echo "Error: taskset not found. Install util-linux." >&2
  exit 2
fi

if [[ "$BUILD" -eq 1 ]]; then
  echo "Building release binary..."
  CARGO_INCREMENTAL=0 cargo build --release
fi

if [[ ! -x "$BINARY" ]]; then
  echo "Error: benchmark binary not found or not executable: $BINARY" >&2
  exit 2
fi

times_file=$(mktemp)
trap 'rm -f "$times_file"' EXIT

run_once() {
  taskset --cpu-list "$CORE" "$BINARY" >"$OUTPUT_FILE"
}

echo "Warmup runs: $WARMUP"
for ((i = 1; i <= WARMUP; i++)); do
  run_once
done

echo "Measured runs: $RUNS"
for ((i = 1; i <= RUNS; i++)); do
  start_ns=$(date +%s%N)
  run_once
  end_ns=$(date +%s%N)
  elapsed_ms=$(awk -v ns="$((end_ns - start_ns))" 'BEGIN { printf "%.6f", ns / 1000000.0 }')
  echo "$elapsed_ms" >>"$times_file"

  if [[ "$PRINT_PER_RUN" -eq 1 ]]; then
    printf "run %02d: %s ms\n" "$i" "$elapsed_ms"
  fi
done

read -r mean stddev median min max total < <(
  sort -n "$times_file" | awk '
    {
      a[NR] = $1
      sum += $1
      sumsq += ($1 * $1)
    }
    END {
      n = NR
      if (n == 0) {
        exit 1
      }
      mean = sum / n
      variance = (sumsq / n) - (mean * mean)
      if (variance < 0) {
        variance = 0
      }
      stddev = sqrt(variance)
      min = a[1]
      max = a[n]
      if (n % 2 == 1) {
        median = a[(n + 1) / 2]
      } else {
        median = (a[n / 2] + a[(n / 2) + 1]) / 2
      }
      printf "%.6f %.6f %.6f %.6f %.6f %.6f\n", mean, stddev, median, min, max, sum
    }
  '
)

printf "\nSummary (milliseconds)\n"
printf "  runs:    %d\n" "$RUNS"
printf "  warmup:  %d\n" "$WARMUP"
printf "  core:    %s\n" "$CORE"
printf "  mean:    %s ms\n" "$mean"
printf "  stddev:  %s ms\n" "$stddev"
printf "  median:  %s ms\n" "$median"
printf "  min:     %s ms\n" "$min"
printf "  max:     %s ms\n" "$max"
printf "  total:   %s ms\n" "$total"
printf "  output:  %s\n" "$OUTPUT_FILE"
