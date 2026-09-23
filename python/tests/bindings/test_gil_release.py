import threading

import numpy as np

import atlas


def test_long_native_computation_releases_the_gil() -> None:
    started = threading.Event()
    finished = threading.Event()
    progress = 0

    def worker() -> None:
        nonlocal progress
        started.wait()
        while not finished.is_set():
            progress += 1

    thread = threading.Thread(target=worker)
    thread.start()
    started.set()

    atlas._native._gil_free_spin(200)

    finished.set()
    thread.join()

    assert progress > 100


def test_public_array_operation_releases_the_gil() -> None:
    ready = threading.Event()
    start = threading.Event()
    finished = threading.Event()
    progress = 0

    def worker() -> None:
        nonlocal progress
        ready.set()
        start.wait()
        while not finished.is_set():
            progress += 1

    values = np.ones(8_000_000)
    thread = threading.Thread(target=worker)
    thread.start()
    ready.wait()
    start.set()

    atlas.add(values, 1.0)

    finished.set()
    thread.join()

    assert progress > 100
