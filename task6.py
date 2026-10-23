
import platform
import threading
import time

THREADS = 4
INCREMENTS_PER_THREAD = 2_000_000
EXPECTED = THREADS * INCREMENTS_PER_THREAD


def run_without_lock(counter):
    for i in range(INCREMENTS_PER_THREAD):
        current = counter[0]

        if i % 1000 == 0:
            time.sleep(0)

        counter[0] = current + 1


def run_with_lock(counter, lock):
    for _ in range(INCREMENTS_PER_THREAD):
        with lock:
            counter[0] += 1


def benchmark(target, *args):
    threads = [
        threading.Thread(target=target, args=args)
        for _ in range(THREADS)
    ]

    start = time.perf_counter()

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    return (time.perf_counter() - start) * 1000


def main():
    print(f"Python version: {platform.python_version()}")
    print(f"OS: {platform.platform()}")
    print(f"Threads: {THREADS}")
    print(f"Increments per thread: {INCREMENTS_PER_THREAD}")
    print()

    # version 1: no synchronization
    unsafe_counter = [0]
    unsafe_time = benchmark(run_without_lock, unsafe_counter)

    lost_updates = EXPECTED - unsafe_counter[0]
    lost_percentage = max(0, lost_updates) / EXPECTED * 100

    print("=== Without Lock ===")
    print(f"Expected final value: {EXPECTED:,}")
    print(f"Actual final value:   {unsafe_counter[0]:,}")
    print(f"Lost updates:         {lost_updates:,}")
    print(f"Lost updates (%):     {lost_percentage:.4f}%")
    print(f"Wall time:            {unsafe_time:.3f} ms")
    print()

    # version 2: synchronized increments
    safe_counter = [0]
    lock = threading.Lock()
    safe_time = benchmark(run_with_lock, safe_counter, lock)

    print("=== With Lock ===")
    print(f"Expected final value: {EXPECTED:,}")
    print(f"Actual final value:   {safe_counter[0]:,}")
    print(f"Wall time:            {safe_time:.3f} ms")
    print()

    print("=== Synchronization Cost ===")
    print(f"Time penalty: {safe_time - unsafe_time:.3f} ms")
    if unsafe_time > 0:
        print(f"Time ratio (Lock / No Lock): {safe_time / unsafe_time:.3f}x")

    if safe_counter[0] != EXPECTED:
        raise RuntimeError("Synchronized counter is incorrect!")


if __name__ == "__main__":
    main()
