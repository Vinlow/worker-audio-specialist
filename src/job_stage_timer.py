"""Invocation-local timings that cannot fail a paid job or skip its cleanup."""
import json
import time


class JobStageTimer:
    """Keep timing state on the invocation, never in RunPod's global registry."""

    def __init__(self, stage):
        self.stage = stage
        self.started = None

    def __enter__(self):
        self.started = time.perf_counter()
        return self

    def __exit__(self, exception_type, _exception, _traceback):
        # Logging is optional evidence. A closed output stream must not replace
        # the original inference exception or prevent a finally block running.
        try:
            print(json.dumps({
                'event': 'audio_worker_stage',
                'stage': self.stage,
                'duration_ms': round((time.perf_counter() - self.started) * 1000, 3),
                'failed': exception_type is not None,
            }), flush=True)
        except (OSError, ValueError):
            pass
        return False
