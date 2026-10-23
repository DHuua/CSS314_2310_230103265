
import os
import platform
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ARRAY_SIZES = [
    1, 1_000, 10_000, 100_000, 1_000_000, 10_000_000
]
REPEATS = 7
A = 2.5
B = 1.5


def serial_operation(x, y):
    np.multiply(x, A, out=y)
    np.add(y, B, out=y)


def parallel_operation(x, y, executor, ranges):
    def process_chunk(start, end):
        np.multiply(x[start:end], A, out=y[start:end])
        np.add(y[start:end], B, out=y[start:end])

    futures = [
        executor.submit(process_chunk, start, end)
        for start, end in ranges
    ]

    for future in futures:
        future.result()


def median_time_ms(operation, repeats=REPEATS):
    # Warm-up before collecting measurements.
    operation()

    measurements = []
    for _ in range(repeats):
        start = time.perf_counter()
        operation()
        measurements.append((time.perf_counter() - start) * 1000)

    return statistics.median(measurements)


def main():
    workers = os.cpu_count() or 1

    print(f"Python: {platform.python_version()}")
    print(f"OS: {platform.platform()}")
    print(f"Logical CPU cores: {workers}")
    print(f"Workers: {workers}")
    print(f"Operation: y[i] = {A} * x[i] + {B}")
    print(f"Measurement: median of {REPEATS} runs")
    print()

    results = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for n in ARRAY_SIZES:
            x = np.linspace(0.0, 1.0, n, dtype=np.float64)
            y_serial = np.empty_like(x)
            y_parallel = np.empty_like(x)

            active_workers = min(workers, n)
            chunk_size = (n + active_workers - 1) // active_workers
            ranges = [
                (start, min(start + chunk_size, n))
                for start in range(0, n, chunk_size)
            ]

            serial_ms = median_time_ms(
                lambda: serial_operation(x, y_serial)
            )

            parallel_ms = median_time_ms(
                lambda: parallel_operation(
                    x, y_parallel, executor, ranges
                )
            )

            if not np.allclose(y_serial, y_parallel, rtol=0, atol=0):
                raise RuntimeError(f"Results differ for N={n}")

            speedup = serial_ms / parallel_ms
            faster = parallel_ms < serial_ms

            results.append({
                "n": n,
                "serial": serial_ms,
                "parallel": parallel_ms,
                "speedup": speedup,
                "faster": faster,
            })

    print(
        f"{'N':>12} | {'Serial (ms)':>12} | "
        f"{'Parallel (ms)':>14} | {'Speedup':>8} | Result"
    )
    print("-" * 75)

    for row in results:
        result = "FASTER" if row["faster"] else "SLOWER/EQUAL"
        print(
            f"{row['n']:12,d} | {row['serial']:12.4f} | "
            f"{row['parallel']:14.4f} | "
            f"{row['speedup']:8.3f} | {result}"
        )

    print()
    print("=== Crossover analysis ===")

    faster_sizes = [r["n"] for r in results if r["faster"]]
    slower_sizes = [r["n"] for r in results if not r["faster"]]

    if not faster_sizes:
        print("No tested size was faster with parallel execution.")
        print("Crossover was not observed in the tested range.")
    elif not slower_sizes:
        print("Parallel execution was faster at every tested size.")
        print("Crossover was not observed in the tested range.")
    else:
        first_faster = min(faster_sizes)
        slower_before = [
            r["n"] for r in results
            if r["n"] < first_faster and not r["faster"]
        ]

        if slower_before:
            last_slower = max(slower_before)
            print(
                f"Observed crossover bracket: "
                f"{last_slower:,} < N <= {first_faster:,}"
            )
            print(
                "At the lower tested size, parallel was slower; "
                "at the upper tested size, parallel was faster."
            )
        else:
            print(
                f"First faster tested size: N={first_faster:,}, "
                "but no slower tested size precedes it."
            )

        transitions = []
        for previous, current in zip(results, results[1:]):
            if not previous["faster"] and current["faster"]:
                transitions.append((previous["n"], current["n"]))

        if len(transitions) > 1:
            print("Additional slower-to-faster transitions:")
            for low, high in transitions:
                print(f"  {low:,} < N <= {high:,}")

    print()
    print(
        "Note: crossover is bracketed by tested sizes, not "
        "measured at every possible N."
    )


if __name__ == "__main__":
    main()
