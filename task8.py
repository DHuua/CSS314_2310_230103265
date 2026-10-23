import sys
import time
import math
import numpy as np
from numba import cuda, njit, prange, get_num_threads

BLOCK_SIZES = [32, 64, 128, 256, 512, 1024]
N_VEC = 1 << 24          
REPEATS = 20


def make_dot_kernel(bs):
    """Kernel factory: shared memory size must be a compile-time constant."""
    @cuda.jit
    def dot_kernel(a, b, partial):
        sh = cuda.shared.array(bs, dtype=np.float64)
        tid = cuda.threadIdx.x
        i = cuda.grid(1)
        stride = cuda.gridDim.x * cuda.blockDim.x
        acc = 0.0
        while i < a.size:                 
            acc += a[i] * b[i]
            i += stride
        sh[tid] = acc
        cuda.syncthreads()
        s = bs // 2                       
        while s > 0:
            if tid < s:
                sh[tid] += sh[tid + s]
            cuda.syncthreads()
            s //= 2
        if tid == 0:
            partial[cuda.blockIdx.x] = sh[0]
    return dot_kernel


def run_gpu():
    dev = cuda.get_current_device()
    print(f"GPU: {dev.name.decode() if isinstance(dev.name, bytes) else dev.name}")
    print(f"N = {N_VEC:,} float32 elements, {REPEATS} timed runs (median), kernel-only time\n")

    rng = np.random.default_rng(0)
    a = rng.random(N_VEC, dtype=np.float32)
    b = rng.random(N_VEC, dtype=np.float32)
    ref = float(np.dot(a.astype(np.float64), b.astype(np.float64)))
    d_a, d_b = cuda.to_device(a), cuda.to_device(b)

    print(f"{'block':>6} {'grid':>8} {'latency_ms':>11} {'min_ms':>9} {'result_ok':>10}")
    for bs in BLOCK_SIZES:
        grid = min(math.ceil(N_VEC / bs), 65535)
        kernel = make_dot_kernel(bs)
        d_part = cuda.device_array(grid, dtype=np.float64)
        kernel[grid, bs](d_a, d_b, d_part)        
        cuda.synchronize()

        times = []
        for _ in range(REPEATS):
            s, e = cuda.event(), cuda.event()
            s.record()
            kernel[grid, bs](d_a, d_b, d_part)
            e.record()
            e.synchronize()
            times.append(cuda.event_elapsed_time(s, e))   # ms

        result = float(d_part.copy_to_host().sum())
        ok = abs(result - ref) / abs(ref) < 1e-5
        print(f"{bs:>6} {grid:>8} {np.median(times):>11.4f} {min(times):>9.4f} {str(ok):>10}")


@njit(parallel=True, fastmath=True)
def matmul_naive(A, B, C):
    n = A.shape[0]
    for i in prange(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += A[i, k] * B[k, j]     
            C[i, j] = s


@njit(parallel=True, fastmath=True)
def matmul_tiled(A, B, C, T):
    n = A.shape[0]
    nb = (n + T - 1) // T
    for bi in prange(nb):                 
        i0 = bi * T
        i1 = min(i0 + T, n)
        for j0 in range(0, n, T):
            j1 = min(j0 + T, n)
            for k0 in range(0, n, T):
                k1 = min(k0 + T, n)
                for i in range(i0, i1):
                    for k in range(k0, k1):
                        a = A[i, k]
                        for j in range(j0, j1):
                            C[i, j] += a * B[k, j]


def timed(fn, repeats=3):
    best = float("inf")
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best * 1000.0  # ms


def run_cpu():
    n = 1024
    print(f"CPU mode: {n}x{n} float32 matmul, threads = {get_num_threads()}\n")
    rng = np.random.default_rng(0)
    A = rng.random((n, n), dtype=np.float32)
    B = rng.random((n, n), dtype=np.float32)
    ref = A @ B

    C = np.zeros((n, n), dtype=np.float32)
    matmul_naive(A, B, C)                  
    matmul_tiled(A, B, np.zeros_like(C), 16)

    results = []
    C = np.zeros_like(ref)
    t = timed(lambda: matmul_naive(A, B, C))
    results.append(("non-tiled", t, np.allclose(C, ref, rtol=1e-3)))

    for T in (16, 64):
        C = np.zeros_like(ref)
        def f(T=T):
            C.fill(0.0)
            matmul_tiled(A, B, C, T)
        t = timed(f)
        results.append((f"tiled {T}x{T}", t, np.allclose(C, ref, rtol=1e-3)))

    base = results[0][1]
    print(f"{'variant':<12} {'runtime_ms':>11} {'speedup':>8} {'correct':>8}")
    for name, t, ok in results:
        print(f"{name:<12} {t:>11.1f} {base / t:>7.2f}x {str(ok):>8}")
    print("\nNote: runtime differences reflect cache behaviour. The non-tiled loop strides")
    print("through B column-wise (a new cache line per element), tiles keep A/B/C blocks")
    print("resident in L1/L2. For hardware counters run: perf stat -e cache-misses python task8.py --cpu")


if __name__ == "__main__":
    if "--cpu" not in sys.argv and cuda.is_available():
        run_gpu()
    else:
        run_cpu()