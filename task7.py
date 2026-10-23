import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from threadpoolctl import threadpool_limits

N = 30_000_000
DTYPE = np.float64
MAX_CORES = os.cpu_count() or 1
CORE_COUNTS = sorted(set([1, 2, MAX_CORES]))
REPEATS = 3


def make_chunks(n, workers):
    bounds = np.linspace(0, n, workers + 1, dtype=int)
    return [(int(bounds[i]), int(bounds[i + 1])) for i in range(workers)]


def run_kernel(kernel, a, b, workers):
    chunks = make_chunks(len(a), workers)

    def task(start, end):
        x = a[start:end]
        if kernel == "A":
            np.copyto(b[start:end], x)
        else:
            np.sin(x, out=b[start:end])
            b[start:end] *= np.cos(x)
            b[start:end] += np.tan(x)

    start = time.perf_counter()
    if workers == 1:
        task(*chunks[0])
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(lambda c: task(*c), chunks))
    return time.perf_counter() - start


def median_runtime(kernel, a, b, workers):
    run_kernel(kernel, a, b, workers)  # warm-up
    samples = [
        run_kernel(kernel, a, b, workers)
        for _ in range(REPEATS)
    ]
    return float(np.median(samples))


def main():
    print(f"Python: {platform.python_version()}")
    print(f"OS: {platform.platform()}")
    print(f"NumPy: {np.__version__}")
    print(f"Logical CPU threads detected: {MAX_CORES}")
    print(f"Array elements: {N:,}")
    print(f"Array type: {DTYPE}")
    print(f"Data size per array: {N * np.dtype(DTYPE).itemsize / 1e6:.1f} MB")
    print(f"Runs per configuration: {REPEATS} (median)")
    print("Work is split into chunks processed by ThreadPoolExecutor.")
    print()

    rng = np.random.default_rng(314159)
    a = rng.uniform(-0.5, 0.5, size=N).astype(DTYPE)
    b = np.empty_like(a)

    results = {"A": {}, "B": {}}
    labels = {
        "A": "Kernel A — Memory-Bound: B[i] = A[i]",
        "B": "Kernel B — Compute-Bound: sin(A[i])*cos(A[i]) + tan(A[i])",
    }

    for kernel in ("A", "B"):
        print(f"=== {labels[kernel]} ===")
        print(f"{'Workers':>8} | {'Time (s)':>10} | {'Speedup':>9} | {'Efficiency':>11}")
        print("-" * 48)

        for workers in CORE_COUNTS:
            elapsed = median_runtime(kernel, a, b, workers)
            results[kernel][workers] = elapsed
            speedup = results[kernel][1] / elapsed
            efficiency = speedup / workers
            print(f"{workers:8d} | {elapsed:10.4f} | {speedup:9.3f}x | {efficiency:10.1%}")
        print()

    # Correctness checks are run independently of the timing loop.
    run_kernel("A", a, b, 1)
    copy_ok = np.array_equal(a, b)

    sample = a[:1000]
    expected = np.sin(sample) * np.cos(sample) + np.tan(sample)
    run_kernel("B", a, b, 1)
    compute_ok = np.allclose(b[:1000], expected, rtol=1e-12, atol=1e-12)

    print("=== CORRECTNESS CHECKS ===")
    print(f"Copy output matches input: {copy_ok}")
    print(f"Compute output matches formula: {compute_ok}")
    print()

    print("=== SCALING INTERPRETATION ===")
    a2, am = results["A"].get(2), results["A"].get(MAX_CORES)
    b2, bm = results["B"].get(2), results["B"].get(MAX_CORES)

    if a2 and am:
        speedup_2_to_max = a2 / am
        print(f"Kernel A speedup from 2 to {MAX_CORES} workers: {speedup_2_to_max:.3f}x")
        if speedup_2_to_max < 1.25:
            print("Kernel A: scaling largely stalled after 2 workers; this is consistent with memory-bandwidth saturation.")
        else:
            print("Kernel A: gained noticeable speedup beyond 2 workers; RAM bandwidth saturation is not clearly shown.")

    if b2 and bm:
        speedup_2_to_max = b2 / bm
        print(f"Kernel B speedup from 2 to {MAX_CORES} workers: {speedup_2_to_max:.3f}x")
        if speedup_2_to_max < 1.25:
            print("Kernel B: scaling also stalled; possible CPU, threading, or math-library bottleneck.")
        else:
            print("Kernel B: continued scaling beyond 2 workers, consistent with a compute-heavy workload.")

    print()
    print("Caution: timing alone suggests, but does not directly prove, RAM channel saturation.")
    print("If timings vary greatly, close other heavy applications and rerun.")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
