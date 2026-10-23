import sys
import time
import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np

STUDENT_ID = 230103265      
REPEATS = 5


def run_gpu():
    from numba import cuda

    @cuda.jit
    def vmul(a, b, c):
        i = cuda.grid(1)
        if i < c.size:
            c[i] = a[i] * b[i]

    sizes = [100_000, 1_000_000, 20_000_000]
    name = cuda.get_current_device().name
    print(f"GPU: {name.decode() if isinstance(name, bytes) else name}")
    print(f"Student ID seed: {STUDENT_ID} | float32 | best of {REPEATS} runs\n")
    print(f"{'N':>12} {'H2D_ms':>9} {'Kernel_ms':>10} {'D2H_ms':>9} {'Total_ms':>9} {'PCIe_%':>7}")

    rng = np.random.default_rng(STUDENT_ID)
    # warm-up (JIT compile + context init)
    w = np.ones(1024, np.float32)
    d = cuda.to_device(w)
    vmul[4, 256](d, d, d)
    cuda.synchronize()

    for n in sizes:
        a = rng.random(n, dtype=np.float32)
        b = rng.random(n, dtype=np.float32)
        best = None
        for _ in range(REPEATS):
            cuda.synchronize()
            t0 = time.perf_counter()
            d_a = cuda.to_device(a)
            d_b = cuda.to_device(b)
            cuda.synchronize()
            t1 = time.perf_counter()
            d_c = cuda.device_array(n, dtype=np.float32)
            vmul[(n + 255) // 256, 256](d_a, d_b, d_c)
            cuda.synchronize()
            t2 = time.perf_counter()
            c = d_c.copy_to_host()
            t3 = time.perf_counter()
            run = ((t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3)
            if best is None or sum(run) < sum(best):
                best = run
        h2d, ker, d2h = best
        total = h2d + ker + d2h
        print(f"{n:>12,} {h2d:>9.3f} {ker:>10.3f} {d2h:>9.3f} {total:>9.3f} {100 * (h2d + d2h) / total:>6.1f}%")
        assert np.allclose(c[:10], a[:10] * b[:10])


def child(conn):
    """Persistent worker: receives either a pickled array or a shared-memory name."""
    while True:
        msg = conn.recv()
        if msg is None:
            break
        kind = msg[0]
        if kind == "pickle":
            arr = msg[1]
            val = float(arr[0] + arr[-1])
        else:
            _, name, n = msg
            shm = shared_memory.SharedMemory(name=name)
            arr = np.ndarray((n,), dtype=np.float64, buffer=shm.buf)
            val = float(arr[0] + arr[-1])
            del arr
            shm.close()
        conn.send(val)


def run_cpu():
    sizes = [100_000, 1_000_000, 10_000_000]
    parent, kid = mp.Pipe()
    p = mp.Process(target=child, args=(kid,), daemon=True)
    p.start()

    print("CPU mode: inter-process transfer of a float64 array")
    print(f"Student ID seed: {STUDENT_ID} | best of {REPEATS} runs | time = send + child receives + ack\n")
    print(f"{'N':>12} {'MB':>8} {'A_pipe_pickle_ms':>17} {'B_shared_mem_ms':>16} {'A/B':>7}")

    rng = np.random.default_rng(STUDENT_ID)
    for n in sizes:
        a = rng.random(n)                      
        expect = float(a[0] + a[-1])

        best_a = float("inf")
        for _ in range(REPEATS):
            t = time.perf_counter()
            parent.send(("pickle", a))      
            r = parent.recv()
            best_a = min(best_a, time.perf_counter() - t)
            assert abs(r - expect) < 1e-12

        best_b = float("inf")
        for _ in range(REPEATS):
            t = time.perf_counter()
            shm = shared_memory.SharedMemory(create=True, size=a.nbytes)
            view = np.ndarray((n,), dtype=np.float64, buffer=shm.buf)
            view[:] = a                        
            parent.send(("shm", shm.name, n))   
            r = parent.recv()
            best_b = min(best_b, time.perf_counter() - t)
            assert abs(r - expect) < 1e-12
            del view
            shm.close()
            shm.unlink()

        a_ms, b_ms = best_a * 1e3, best_b * 1e3
        print(f"{n:>12,} {a.nbytes / 1e6:>8.1f} {a_ms:>17.3f} {b_ms:>16.3f} {a_ms / b_ms:>6.2f}x")

    parent.send(None)
    p.join(timeout=5)


if __name__ == "__main__":
    use_gpu = False
    if "--cpu" not in sys.argv:
        try:
            from numba import cuda
            use_gpu = cuda.is_available()
        except Exception:
            use_gpu = False
    run_gpu() if use_gpu else run_cpu()