import unittest
from unittest import mock


try:
    import runpod
    from runpod.serverless.modules import rp_job
except ImportError:
    runpod = None
    rp_job = None


@unittest.skipUnless(runpod is not None, 'runpod is not installed')
class RunPodStreamContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_span_error_then_success_stream_before_exact_terminal_failure(self):
        self.assertEqual(runpod.__version__, '1.8.2')

        span_error = {
            'event': 'span_error',
            'failed_span_index': 7,
            'message': 'MEDIA_FETCH_FAILED: could not fetch audio',
            'code': 'MEDIA_FETCH_FAILED',
            'stage': 'span_download',
            'retryable': True,
        }
        successful_span = {
            'mode': 'final',
            'span_index': 8,
            'start_sec': 20.0,
            'transcription': 'later span still completed',
        }
        terminal_error = (
            'SPAN_STREAM_PARTIAL_FAILURE: failed_span_indexes=7,9; '
            'codes=MEDIA_FETCH_FAILED,SPAN_PROCESSING_FAILED'
        )
        later_span_error = {
            'event': 'span_error',
            'failed_span_index': 9,
            'message': 'SPAN_PROCESSING_FAILED: RuntimeError',
            'code': 'SPAN_PROCESSING_FAILED',
            'stage': 'prediction',
            'retryable': False,
        }
        handler_progress = []

        def span_stream_handler(_job):
            handler_progress.append('span_error')
            yield span_error
            handler_progress.append('successful_span')
            yield successful_span
            handler_progress.append('later_span_error')
            yield later_span_error
            handler_progress.append('terminal_error')
            yield {'error': terminal_error}
            handler_progress.append('unexpected_later_span')
            yield {'span_index': 10, 'transcription': 'must not be consumed'}

        streamed = []
        completed = []

        async def capture_stream(_session, stream_output, job):
            streamed.append((stream_output, job))

        async def capture_result(
            _session,
            job_result,
            job,
            is_stream=False,
        ):
            completed.append((job_result, job, is_stream))

        job = {'id': 'stream-contract-job', 'input': {}}
        config = {
            'handler': span_stream_handler,
            'return_aggregate_stream': True,
            'refresh_worker': False,
            'rp_args': {},
        }
        with (
            mock.patch.object(rp_job, 'stream_result', new=capture_stream),
            mock.patch.object(rp_job, 'send_result', new=capture_result),
        ):
            await rp_job.handle_job(object(), config, job)

        self.assertNotIn('error', span_error)
        self.assertEqual(
            handler_progress,
            [
                'span_error',
                'successful_span',
                'later_span_error',
                'terminal_error',
            ],
        )
        self.assertEqual(
            streamed,
            [
                ({'output': span_error}, job),
                ({'output': successful_span}, job),
                ({'output': later_span_error}, job),
            ],
        )
        self.assertEqual(
            completed,
            [
                (
                    {'error': terminal_error},
                    job,
                    True,
                ),
            ],
        )

    async def test_all_success_has_no_terminal_error(self):
        self.assertEqual(runpod.__version__, '1.8.2')

        successful_span = {
            'mode': 'final',
            'span_index': 8,
            'transcription': 'completed',
        }

        def successful_handler(_job):
            yield successful_span

        streamed = []
        completed = []

        async def capture_stream(_session, stream_output, job):
            streamed.append((stream_output, job))

        async def capture_result(_session, job_result, job, is_stream=False):
            completed.append((job_result, job, is_stream))

        job = {'id': 'all-success-contract-job', 'input': {}}
        config = {
            'handler': successful_handler,
            'return_aggregate_stream': True,
            'refresh_worker': False,
            'rp_args': {},
        }
        with (
            mock.patch.object(rp_job, 'stream_result', new=capture_stream),
            mock.patch.object(rp_job, 'send_result', new=capture_result),
        ):
            await rp_job.handle_job(object(), config, job)

        self.assertEqual(streamed, [({'output': successful_span}, job)])
        self.assertEqual(
            completed,
            [({'output': [successful_span]}, job, True)],
        )

    async def test_truthy_top_level_error_remains_terminal(self):
        handler_progress = []

        def terminal_error_handler(_job):
            handler_progress.append('terminal_error')
            yield {'error': 'terminal failure'}
            handler_progress.append('unexpected_later_span')
            yield {'span_index': 9, 'transcription': 'must not be consumed'}

        streamed = []
        completed = []

        async def capture_stream(_session, stream_output, job):
            streamed.append((stream_output, job))

        async def capture_result(
            _session,
            job_result,
            job,
            is_stream=False,
        ):
            completed.append((job_result, job, is_stream))

        job = {'id': 'terminal-contract-job', 'input': {}}
        config = {
            'handler': terminal_error_handler,
            'return_aggregate_stream': True,
            'refresh_worker': False,
            'rp_args': {},
        }
        with (
            mock.patch.object(rp_job, 'stream_result', new=capture_stream),
            mock.patch.object(rp_job, 'send_result', new=capture_result),
        ):
            await rp_job.handle_job(object(), config, job)

        self.assertEqual(handler_progress, ['terminal_error'])
        self.assertEqual(streamed, [])
        self.assertEqual(
            completed,
            [({'error': 'terminal failure'}, job, True)],
        )


if __name__ == '__main__':
    unittest.main()
