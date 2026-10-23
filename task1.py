
import os
import platform
import time

import numpy as np
from threadpoolctl import threadpool_limits

STUDENT_ID = "230103265"
ARRAY_SIZE = 20_000_000


def main():
    print(f"Student ID: {STUDENT_ID}")
    print(f"Operating System: {platform.platform()}")
    print(f"CPU Model: {platform.processor() or 'Not available'}")
    print(f"Logical CPU Cores: {os.cpu_count()}")

    seed = int(STUDENT_ID) if STUDENT_ID.isdigit() else sum(
        ord(char) for char in STUDENT_ID
    )

    with threadpool_limits(limits=1):
        start = time.perf_counter()

        rng = np.random.default_rng(seed)
        data = rng.random(ARRAY_SIZE, dtype=np.float64)

        total = np.sum(data, dtype=np.float64)
        std_dev = np.std(data, dtype=np.float64)

        elapsed_ms = (time.perf_counter() - start) * 1000

    print(f"Array size: {data.size}")
    print(f"Data type: {data.dtype}")
    print(f"Sum: {total:.10f}")
    print(f"Standard Deviation: {std_dev:.10f}")
    print(f"Execution Time: {elapsed_ms:.3f} ms")


if __name__ == "__main__":
    main()
