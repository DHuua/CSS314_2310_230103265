import os
import sys
import time
import numpy as np
from numba import njit, prange, set_num_threads, get_num_threads

STUDENT_ID = 230103265     
N = 2_000_000
REPEATS = 5
THREADS = os.cpu_count()
set_num_threads(THREADS)


@njit
def serial_scan(a, out):
    out[0] = a[0]
    for i in range(1, a.size):
        out[i] = out[i - 1] + a[i]


@njit(parallel=True)
def parallel_scan(a, out, nchunks):
    n = a.size
    step = (n + nchunks - 1) // nchunks

    sums = np.zeros(nchunks, np.int64)
    for c in prange(nchunks):
        s = 0
        for i in range(c * step, min((c + 1) * step, n)):
            s += a[i]
        sums[c] = s

    offs = np.zeros(nchunks, np.int64)
    for c in range(1, nchunks):
        offs[c] = offs[c - 1] + sums[c - 1]

    for c in prange(nchunks):
        run = offs[c]
        for i in range(c * step, min((c + 1) * step, n)):
            run += a[i]
            out[i] = run


def best_ms(fn, *args):
    best = float("inf")
    for _ in range(REPEATS):
        t = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t)
    return best * 1000.0


if __name__ == "__main__":
    rng = np.random.default_rng(STUDENT_ID)
    a = rng.integers(1, 11, size=N, dtype=np.int64)      

    ser = np.zeros(N, np.int64)
    par = np.zeros(N, np.int64)

    tiny = a[:1024].copy()
    serial_scan(tiny, np.zeros(1024, np.int64))
    parallel_scan(tiny, np.zeros(1024, np.int64), THREADS)

    t_ser = best_ms(serial_scan, a, ser)
    t_par = best_ms(parallel_scan, a, par, THREADS)

    last_ok = par[-1] == ser[-1]
    full_ok = np.array_equal(par, ser)
    status = "PASS" if (last_ok and full_ok) else "FAIL"

    print(f"Seed (Student ID): {STUDENT_ID} | N = {N:,} | Threads: {get_num_threads()}")
    print(f"Serial last = {ser[-1]} | Parallel last = {par[-1]} | Full array equal: {full_ok}")
    print(f"Serial Time: {t_ser:.3f} ms | Parallel Time: {t_par:.3f} ms | "
          f"Speedup: {t_ser / t_par:.2f}x | Verification: {status}")

    try:
        assert par[-1] == ser[-1], "parallel_output[-1] != serial_output[-1]"
        assert full_ok, "parallel output differs from serial output"
    except AssertionError as e:
        print(f"ASSERTION FAILED: {e}")
        sys.exit(1)