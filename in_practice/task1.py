import os
import time
import numpy as np
from numba import njit, prange, set_num_threads, get_num_threads

try:
    import psutil
    PHYSICAL = psutil.cpu_count(logical=False)
except ImportError:
    PHYSICAL = None

STUDENT_ID = 230103265    
N = 8000
REPEATS = 3

LOGICAL = os.cpu_count()
THREADS = PHYSICAL or LOGICAL
set_num_threads(THREADS)


@njit(parallel=True)
def sum_row_major(A, out, nchunks):
    """Mode A: each worker owns a band of rows, walks columns linearly."""
    n, m = A.shape
    step = (n + nchunks - 1) // nchunks
    for c in prange(nchunks):
        s = 0.0
        for i in range(c * step, min((c + 1) * step, n)):
            for j in range(m):
                s += A[i, j]
        out[c] = s


@njit(parallel=True)
def sum_col_major(A, out, nchunks):
    """Mode B: each worker owns a band of columns, walks down rows (stride = 64 KB)."""
    n, m = A.shape
    step = (m + nchunks - 1) // nchunks
    for c in prange(nchunks):
        s = 0.0
        for j in range(c * step, min((c + 1) * step, m)):
            for i in range(n):
                s += A[i, j]
        out[c] = s


def bench(fn, A):
    out = np.zeros(THREADS)
    best = float("inf")
    for _ in range(REPEATS):
        t = time.perf_counter()
        fn(A, out, THREADS)
        best = min(best, time.perf_counter() - t)
    return best * 1000.0, out.sum()


if __name__ == "__main__":
    rng = np.random.default_rng(STUDENT_ID)
    A = rng.random((N, N))                      
    tiny = np.ones((64, 64))
    sum_row_major(tiny, np.zeros(THREADS), THREADS)
    sum_col_major(tiny, np.zeros(THREADS), THREADS)

    t_a, s_a = bench(sum_row_major, A)
    t_b, s_b = bench(sum_col_major, A)

    print(f"Student ID: {STUDENT_ID}")
    print(f"Logical cores: {LOGICAL} | Physical cores: {PHYSICAL} | Threads used: {get_num_threads()}")
    print(f"Matrix: {N}x{N} float64 ({A.nbytes / 1e6:.0f} MB), best of {REPEATS} runs")
    print(f"Mode A (row-major)    : {t_a:10.2f} ms  sum={s_a:.4f}")
    print(f"Mode B (column-major) : {t_b:10.2f} ms  sum={s_b:.4f}")
    print(f"Sums match: {np.isclose(s_a, s_b, rtol=1e-9)}")
    print(f"Cache penalty factor (B/A): {t_b / t_a:.2f}x")