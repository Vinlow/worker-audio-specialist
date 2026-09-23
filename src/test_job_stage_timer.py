import io
import json
import unittest
from contextlib import redirect_stdout
from unittest import mock

from job_stage_timer import JobStageTimer


class JobStageTimerTests(unittest.TestCase):
    def test_same_stage_can_repeat_and_overlap_without_shared_registration(self):
        output = io.StringIO()
        with redirect_stdout(output):
            for _ in range(3):
                with JobStageTimer('cleanup_step'):
                    with JobStageTimer('cleanup_step'):
                        pass
        rows = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(row['stage'] == 'cleanup_step' for row in rows))
        self.assertTrue(all(row['duration_ms'] >= 0 for row in rows))
        self.assertTrue(all(row['failed'] is False for row in rows))

    def test_original_exception_survives_failed_logging_and_cleanup_runs(self):
        original = RuntimeError('model failure')
        cleanup = mock.Mock()
        with mock.patch('builtins.print', side_effect=BrokenPipeError('closed log')):
            with self.assertRaises(RuntimeError) as caught:
                try:
                    with JobStageTimer('prediction_step'):
                        raise original
                finally:
                    with JobStageTimer('cleanup_step'):
                        cleanup()
        self.assertIs(caught.exception, original)
        cleanup.assert_called_once_with()

    def test_success_survives_closed_logging(self):
        with mock.patch('builtins.print', side_effect=ValueError('closed stream')):
            with JobStageTimer('cleanup_step'):
                completed = True
        self.assertTrue(completed)


if __name__ == '__main__':
    unittest.main()
