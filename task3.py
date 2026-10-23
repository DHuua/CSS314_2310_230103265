
import os
import platform
import time
import ctypes
from concurrent.futures import ThreadPoolExecutor

ITERATIONS = 50_000_000
PHYSICAL_CORES = 8 


def get_physical_cores():
    if PHYSICAL_CORES is not None:
        return PHYSICAL_CORES

    import subprocess

    try:
        result = subprocess.check_output(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_Processor | "
                "Measure-Object -Property NumberOfCores -Sum).Sum"
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return int(result)
    except Exception:
        return os.cpu_count() or 1


def run_shared(counts, index):
    for _ in range(ITERATIONS):
        counts[index] += 1


def run_local(counts, index):
    local_count = 0
    for _ in range(ITERATIONS):
        local_count += 1
    counts[index] = local_count


def benchmark(workers, local_mode):
    counts = (ctypes.c_longlong * workers)()

    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        if local_mode:
            executor.map(
                lambda i: run_local(counts, i),
                range(workers),
            )
        else:
            executor.map(
                lambda i: run_shared(counts, i),
                range(workers),
            )

    elapsed_ms = (time.perf_counter() - start) * 1000

    expected = workers * ITERATIONS
    actual = sum(counts)

    if actual != expected:
        raise RuntimeError(
            f"Incorrect result: expected {expected}, got {actual}"
        )

    print(f"Total count: {actual}")
    print(f"Wall time: {elapsed_ms:.3f} ms")
    return elapsed_ms


def main():
    workers = get_physical_cores()

    print(f"Python version: {platform.python_version()}")
    print(f"Operating System: {platform.platform()}")
    print(f"Logical CPU cores: {os.cpu_count()}")
    print(f"Physical worker count: {workers}")
    print(f"Iterations per worker: {ITERATIONS}")
    print()

    print("=== Unpadded Shared ===")
    shared_time = benchmark(workers, local_mode=False)

    print()
    print("=== Local Counter ===")
    local_time = benchmark(workers, local_mode=True)

    print()
    print(
        f"Execution ratio (Shared / Local): "
        f"{shared_time / local_time:.3f}x"
    )


if __name__ == "__main__":
    main()
