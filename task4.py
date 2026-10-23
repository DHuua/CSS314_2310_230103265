
import os
import platform
import time
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

ITEM_COUNT = 1000
WORKERS = 4


def simulate_work(i):
    time.sleep(0.0001 * (i % 20))


def run_static():
    """Assign one fixed block of 250 items to each worker."""
    completion_times = [0.0] * WORKERS
    start = time.perf_counter()

    def worker(worker_id):
        begin = worker_id * (ITEM_COUNT // WORKERS)
        end = begin + (ITEM_COUNT // WORKERS)

        for i in range(begin, end):
            simulate_work(i)

        completion_times[worker_id] = time.perf_counter()

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = [
            executor.submit(worker, worker_id)
            for worker_id in range(WORKERS)
        ]
        for future in futures:
            future.result()

    wall_time = time.perf_counter() - start
    completion_delta = max(completion_times) - min(completion_times)

    return wall_time, completion_delta


def run_dynamic(chunk_size):
    """Workers repeatedly take the next available chunk from a queue."""
    tasks = queue.Queue()

    for start_index in range(0, ITEM_COUNT, chunk_size):
        chunk = list(
            range(start_index, min(start_index + chunk_size, ITEM_COUNT))
        )
        tasks.put(chunk)

    completion_times = [0.0] * WORKERS

    start = time.perf_counter()

    def worker(worker_id):
        while True:
            try:
                chunk = tasks.get_nowait()
            except queue.Empty:
                break

            for i in chunk:
                simulate_work(i)

        completion_times[worker_id] = time.perf_counter()

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = [
            executor.submit(worker, worker_id)
            for worker_id in range(WORKERS)
        ]
        for future in futures:
            future.result()

    wall_time = time.perf_counter() - start
    completion_delta = max(completion_times) - min(completion_times)

    return wall_time, completion_delta


def print_result(strategy, wall_time, completion_delta):
    print(
        f"{strategy:<24} | "
        f"{wall_time * 1000:12.3f} ms | "
        f"{completion_delta * 1000:18.3f} ms"
    )


def main():
    print(f"Python version: {platform.python_version()}")
    print(f"Operating System: {platform.platform()}")
    print(f"Logical CPU cores: {os.cpu_count()}")
    print(f"Items: {ITEM_COUNT}")
    print(f"Workers: {WORKERS}")
    print("Workload: sleep(0.0001 * (i % 20))")
    print()

    print(
        f"{'Strategy':<24} | {'Wall Time':>15} | "
        f"{'Completion Delta':>21}"
    )
    print("-" * 67)

    wall, delta = run_static()
    print_result("A: Static chunks", wall, delta)

    wall, delta = run_dynamic(chunk_size=1)
    print_result("B: Dynamic chunk=1", wall, delta)

    wall, delta = run_dynamic(chunk_size=5)
    print_result("B: Dynamic chunk=5", wall, delta)


if __name__ == "__main__":
    main()
