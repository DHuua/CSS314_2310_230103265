import os
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from numba import njit

STUDENT_ID = 230103265     
N_TASKS = 10_000
BUF_SIZE = 1024
ROUNDS = 500
REPEATS = 2
TOLERANCE = 1.05           

CORES = os.cpu_count()


@njit(nogil=True)
def hash_task(buf, rounds):
    h = np.uint64(14695981039346656037)
    prime = np.uint64(1099511628211)
    for _ in range(rounds):
        for i in range(buf.size):
            h ^= np.uint64(buf[i])
            h *= prime
    return h


def run_batch(pool_size, bufs):
    t = time.perf_counter()
    with ThreadPoolExecutor(max_workers=pool_size) as ex:
        results = list(ex.map(lambda b: hash_task(b, ROUNDS), bufs))
    return (time.perf_counter() - t) * 1000.0, results


if __name__ == "__main__":
    rng = np.random.default_rng(STUDENT_ID)
    bufs = [rng.integers(0, 256, BUF_SIZE, dtype=np.uint8) for _ in range(N_TASKS)]

    hash_task(bufs[0], 1)                       

    raw = [1, max(1, CORES // 2), CORES, CORES * 2, CORES * 4, CORES * 16, 256]
    sizes = list(dict.fromkeys(raw))            

    times, ref = {}, None
    for s in sizes:
        best = float("inf")
        for _ in range(REPEATS):
            ms, res = run_batch(s, bufs)
            best = min(best, ms)
        if ref is None:
            ref = res
        assert res == ref, "results differ between pool sizes"
        times[s] = best

    base = times[sizes[0]]
    best_size = min(times, key=times.get)
    best_time = times[best_size]

    degrade = next((s for s in sizes if s > best_size and times[s] > best_time * TOLERANCE), None)

    print(f"Student ID seed: {STUDENT_ID} | Logical cores: {CORES} | Tasks: {N_TASKS:,} "
          f"(1 KB x {ROUNDS} rounds) | best of {REPEATS} runs\n")
    print(f"{'Pool Size':>9} | {'Execution Time (ms)':>19} | {'Speedup vs 1 thread':>19} | Status")
    print("-" * 80)
    for s in sizes:
        if s == best_size:
            status = "OPTIMAL"
        elif s < best_size:
            status = "scaling"
        elif s == degrade:
            status = "<-- DEGRADATION STARTS (context-switch overhead)"
        elif degrade is not None and s > degrade:
            status = "oversubscribed"
        else:
            status = "flat (within 5% of best)"
        print(f"{s:>9} | {times[s]:>19.1f} | {base / times[s]:>18.2f}x | {status}")

    print()
    if degrade is not None:
        print(f"Best pool size: {best_size}. Execution time first worsens at pool size {degrade} "
              f"({times[degrade]:.1f} ms vs {best_time:.1f} ms).")
    else:
        print(f"Best pool size: {best_size}. No pool size beyond it was more than 5% slower.")