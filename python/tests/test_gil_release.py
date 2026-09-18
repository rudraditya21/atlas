import threading

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
