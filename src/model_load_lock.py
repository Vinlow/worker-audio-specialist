"""Process-wide coordination for cold PyTorch model construction.

Several optional worker components load large models lazily.  Their inference
can run independently, but their constructors may temporarily change PyTorch
module-initialization state (for example, meta/no-init contexts).  Serializing
only the cold load boundary prevents one component from observing another
component's transient construction state without reducing steady-state
inference concurrency.
"""

from contextlib import contextmanager
import threading
import time


MODEL_LOAD_LOCK = threading.RLock()


@contextmanager
def serialized_model_load(component: str):
    """Serialize a cold model load and expose contention in worker logs."""
    wait_started = time.perf_counter()
    MODEL_LOAD_LOCK.acquire()
    waited_sec = time.perf_counter() - wait_started
    try:
        if waited_sec >= 0.01:
            print(
                f"[ModelLoadLock] {component} waited {waited_sec:.3f}s "
                "for another cold model load",
                flush=True,
            )
        yield
    finally:
        MODEL_LOAD_LOCK.release()
