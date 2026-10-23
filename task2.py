
import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

STUDENT_ID = "230103265"
ARRAY_SIZE = 20_000_000
P = 0.95


def get_seed(student_id):
    if student_id.isdigit():
        return int(student_id)
    return sum(ord(char) for char in student_id)


def sum_chunk(data):
    return np.sum(data, dtype=np.float64)


def parallel_sum(data, workers):
    chunk_size = (len(data) + workers - 1) // workers
    chunks = [
        data[start:start + chunk_size]
        for start in range(0, len(data), chunk_size)
    ]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        partial_sums = executor.map(sum_chunk, chunks)
        return sum(partial_sums)


def amdahl_bound(workers):
    return 1 / ((1 - P) + P / workers)


def main():
    max_cores = os.cpu_count() or 1

    print(f"Student ID: {STUDENT_ID}")
    print(f"Operating System: {platform.platform()}")
    print(f"CPU Model: {platform.processor() or 'Not available'}")
    print(f"Logical CPU Cores: {max_cores}")
    print(f"Array size: {ARRAY_SIZE}")
    print(f"Parallel fraction (P): {P:.2f}")
    print()

    rng = np.random.default_rng(get_seed(STUDENT_ID))
    data = rng.random(ARRAY_SIZE, dtype=np.float64)

    worker_counts = [1, 2, 4, max_cores, max_cores * 2]
    results = []

    for workers in worker_counts:
        start = time.perf_counter()
        total = parallel_sum(data, workers)
        elapsed_ms = (time.perf_counter() - start) * 1000

        results.append({
            "workers": workers,
            "time_ms": elapsed_ms,
            "sum": total,
        })

    baseline_time = results[0]["time_ms"]

    print(f"{'Cores':>8} | {'Wall Time (ms)':>15} | "
          f"{'Speedup':>10} | {'Amdahl Upper Bound':>19}")
    print("-" * 65)

    for result in results:
        workers = result["workers"]
        speedup = baseline_time / result["time_ms"]
        bound = amdahl_bound(workers)

        print(f"{workers:8d} | {result['time_ms']:15.3f} | "
              f"{speedup:10.3f} | {bound:19.3f}")

    print()
    print(f"Reference sum (1 worker): {results[0]['sum']:.10f}")

    for result in results[1:]:
        if not np.isclose(
            result["sum"], results[0]["sum"],
            rtol=1e-12, atol=1e-8
        ):
            print(
                f"WARNING: Sum differs for "
                f"{result['workers']} workers."
            )


if __name__ == "__main__":
    main()
