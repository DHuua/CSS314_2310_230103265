import math
import random
import threading
import time

STUDENT_ID = 230103265     
TOTAL = 40_000_000
THREADS = 4
PER_THREAD = TOTAL // THREADS
LOCK_BATCH = 1            


def run(worker_factory):
    hits = [0] * THREADS
    ts = [threading.Thread(target=worker_factory(i, hits)) for i in range(THREADS)]
    t0 = time.perf_counter()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    ms = (time.perf_counter() - t0) * 1000.0
    return 4.0 * sum(hits) / TOTAL, ms


def shared_factory():
    rng = random.Random(STUDENT_ID * 10)        
    lock = threading.Lock()

    def make(idx, hits):
        def work():
            h = 0
            done = 0
            while done < PER_THREAD:
                n = min(LOCK_BATCH, PER_THREAD - done)
                with lock:                       
                    pts = [(rng.random(), rng.random()) for _ in range(n)]
                for x, y in pts:
                    if x * x + y * y <= 1.0:
                        h += 1
                done += n
            hits[idx] = h
        return work
    return make


def independent_factory():
    def make(idx, hits):
        rng = random.Random(STUDENT_ID * 10 + idx)   # private generator, no locking
        rnd = rng.random

        def work():
            h = 0
            for _ in range(PER_THREAD):
                x = rnd()
                y = rnd()
                if x * x + y * y <= 1.0:
                    h += 1
            hits[idx] = h
        return work
    return make


if __name__ == "__main__":
    print(f"Student ID: {STUDENT_ID}")
    print(f"Samples: {TOTAL:,} | Threads: {THREADS} | Lock batch: {LOCK_BATCH} pair(s) per acquisition\n")

    pi_s, t_s = run(shared_factory())
    print(f"Shared PRNG (mutex)  : pi = {pi_s:.8f} | abs error = {abs(pi_s - math.pi):.8f} | time = {t_s:.1f} ms")

    pi_i, t_i = run(independent_factory())
    print(f"Independent PRNGs    : pi = {pi_i:.8f} | abs error = {abs(pi_i - math.pi):.8f} | time = {t_i:.1f} ms")

    print(f"\nSpeedup from removing generator locking: {t_s / t_i:.2f}x")