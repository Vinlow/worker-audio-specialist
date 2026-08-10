import importlib.util
import os
import sys
import tempfile
import time
import types
import unittest
import urllib.error
from pathlib import Path
from unittest import mock


SRC_DIR = Path(__file__).parent
HANDLER_PATH = SRC_DIR / 'rp_handler.py'


class NullTimer:
    def __init__(self, *_args, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class StubPredictor:
    def __init__(self):
        self.loaded_models = []

    def setup(self):
        pass

    def ensure_model_loaded(self, model_name):
        self.loaded_models.append(model_name)

    def predict(self, **_kwargs):
        return {
            'transcription': 'ok',
            'segments': [],
            'word_timestamps': [],
        }

    def predict_punctuation_window(self, payload):
        return payload

    def predict_punctuation_batch(self, payload):
        return payload


def stub_validate(values, schema):
    validated = {}
    errors = []
    for field_name, rules in schema.items():
        if field_name in values:
            value = values[field_name]
            expected_type = rules['type']
            if value is not None and not isinstance(value, expected_type):
                errors.append(f'{field_name} has invalid type')
            validated[field_name] = value
        else:
            validated[field_name] = rules.get('default')
    if errors:
        return {'errors': errors}
    return {'validated_input': validated}


def load_handler_with_stubs():
    numpy_module = types.ModuleType('numpy')
    numpy_module.floating = type('floating', (float,), {})
    numpy_module.integer = type('integer', (int,), {})
    numpy_module.ndarray = type('ndarray', (), {})

    predict_module = types.ModuleType('predict')
    predict_module.Predictor = StubPredictor
    predict_module.AVAILABLE_ASR_BACKENDS = {'whisper', 'parakeet'}

    clap_scorer_module = types.ModuleType('clap_scorer')
    clap_scorer_module.MAX_QUERY_COUNT = 256
    clap_scorer_module.MAX_QUERY_NAME_CHARS = 256
    clap_scorer_module.MAX_QUERY_TEXT_CHARS = 2048
    clap_scorer_module.MAX_TOTAL_QUERY_TEXT_CHARS = 131072

    runpod_module = types.ModuleType('runpod')
    serverless_module = types.ModuleType('runpod.serverless')
    utils_module = types.ModuleType('runpod.serverless.utils')
    validator_module = types.ModuleType('runpod.serverless.utils.rp_validator')
    utils_module.download_files_from_urls = lambda *_args, **_kwargs: [None]
    utils_module.rp_cleanup = types.SimpleNamespace(clean=lambda *_args: None)
    utils_module.rp_debugger = types.SimpleNamespace(LineTimer=NullTimer)
    validator_module.validate = stub_validate
    serverless_module.utils = utils_module
    serverless_module.start = lambda *_args, **_kwargs: None
    runpod_module.serverless = serverless_module

    stub_modules = {
        'numpy': numpy_module,
        'predict': predict_module,
        'clap_scorer': clap_scorer_module,
        'runpod': runpod_module,
        'runpod.serverless': serverless_module,
        'runpod.serverless.utils': utils_module,
        'runpod.serverless.utils.rp_validator': validator_module,
    }
    previous_modules = {
        name: sys.modules.get(name)
        for name in stub_modules
    }
    sys.modules.update(stub_modules)
    sys.path.insert(0, str(SRC_DIR))
    try:
        spec = importlib.util.spec_from_file_location(
            'rp_handler_hardening_test_subject',
            HANDLER_PATH,
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SRC_DIR))
        for name, previous in previous_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


class FakeResponse:
    def __init__(self, status=200, headers=None, body=b''):
        self.status = status
        self.headers = headers or {}
        self.body = body
        self.offset = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def getcode(self):
        return self.status

    def read(self, size=-1):
        if size < 0:
            result = self.body[self.offset:]
            self.offset = len(self.body)
            return result
        result = self.body[self.offset:self.offset + size]
        self.offset += len(result)
        return result


class HandlerHardeningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.handler = load_handler_with_stubs()

    def validated_input(self, **overrides):
        values = {
            name: rules.get('default')
            for name, rules in self.handler.INPUT_VALIDATIONS.items()
        }
        values.update(overrides)
        return values

    def test_final_span_validation_is_bounded_unique_finite_and_http_only(self):
        valid_span = {
            'index': 0,
            'audio': 'https://example.test/span.wav?signature=secret',
            'start_sec': 0.0,
        }
        self.assertIsNone(self.handler.validate_span_stream({
            'mode': 'final',
            'spans': [valid_span],
        }))

        too_many = [dict(valid_span, index=index) for index in range(65)]
        self.assertIn('at most 64', self.handler.validate_span_stream({
            'mode': 'final',
            'spans': too_many,
        }))
        self.assertIn('unique', self.handler.validate_span_stream({
            'mode': 'final',
            'spans': [valid_span, dict(valid_span)],
        }))
        self.assertIn('non-negative number', self.handler.validate_span_stream({
            'mode': 'final',
            'spans': [dict(valid_span, start_sec=float('nan'))],
        }))
        self.assertIn('http or https', self.handler.validate_span_stream({
            'mode': 'final',
            'spans': [dict(valid_span, audio='file:///tmp/span.wav')],
        }))

    def test_clap_queries_reject_empty_wrong_types_and_unbounded_values(self):
        self.assertIsNone(self.handler.validate_clap_queries({
            'laughter': 'people laughing',
        }))
        for invalid in ({}, [], None, {'laughter': 3}, {3: 'description'}):
            self.assertIsNotNone(
                self.handler.validate_clap_queries(invalid),
                invalid,
            )
        self.assertIn('at most 256', self.handler.validate_clap_queries({
            'x' * 257: 'description',
        }))
        self.assertIn('at most 2048', self.handler.validate_clap_queries({
            'query': 'x' * 2049,
        }))
        self.assertIn('256', self.handler.validate_clap_queries({
            str(index): 'description'
            for index in range(257)
        }))
        self.assertIn('combined', self.handler.validate_clap_queries({
            str(index): 'x' * 2048
            for index in range(65)
        }))

    def test_job_input_is_copied_and_bad_span_shape_does_not_crash_keepalive(self):
        original_input = {'span_stream': ['not', 'an', 'object']}
        events = list(self.handler.run_whisper_job({
            'id': 'copy-test',
            'input': original_input,
        }))

        self.assertEqual(original_input, {'span_stream': ['not', 'an', 'object']})
        self.assertEqual(events[0]['event'], 'started')
        self.assertEqual(events[0]['mode'], 'span_stream')
        self.assertIn('must be an object', events[1]['error'])

        for malformed_job in ({}, {'input': None}, {'input': []}, None):
            malformed_events = list(self.handler.run_whisper_job(malformed_job))
            self.assertEqual(malformed_events[0]['event'], 'started')
            self.assertEqual(malformed_events[-1]['error'], 'input must be an object')

    def test_explicit_invalid_clap_queries_are_handler_validation_errors(self):
        for invalid in ({}, [], None):
            events = list(self.handler.run_whisper_job({
                'id': 'clap-test',
                'input': {'clap_queries': invalid},
            }))
            self.assertEqual(events[0]['event'], 'started')
            self.assertIn('clap_queries', events[1]['error'])

    def test_span_stream_rejects_classic_audio_and_diarization_bounds(self):
        span_stream = {'mode': 'draft_warmup'}
        events = list(self.handler.run_whisper_job({
            'id': 'exclusive-test',
            'input': {
                'span_stream': span_stream,
                'audio': 'https://example.test/audio.wav',
            },
        }))
        self.assertIn('mutually exclusive', events[-1]['error'])

        for hints in (
            {'diarize_min_speakers': 65},
            {'diarize_min_speakers': 4, 'diarize_max_speakers': 2},
        ):
            events = list(self.handler.run_whisper_job({
                'id': 'diarize-test',
                'input': {'audio_base64': 'eA==', **hints},
            }))
            self.assertIn('diarize_', events[-1]['error'])

    def test_draft_warmup_defaults_to_turbo_not_schema_base(self):
        model = StubPredictor()
        with mock.patch.object(self.handler, 'MODEL', model):
            events = list(self.handler.run_whisper_job({
                'id': 'warmup-test',
                'input': {'span_stream': {'mode': 'draft_warmup'}},
            }))

        self.assertEqual(model.loaded_models, ['turbo'])
        self.assertEqual(events[-1]['model'], 'turbo')

    def test_media_fetch_failure_redacts_url_secrets_and_has_stable_fields(self):
        input_payload = {
            'audio': (
                'https://user:password@example.test/private/audio.wav'
                '?token=super-secret#fragment-secret'
            ),
        }
        original = dict(input_payload)
        with mock.patch.object(
            self.handler,
            'download_audio_url',
            side_effect=self.handler.MediaFetchFailure(
                input_payload['audio'],
                'classic_download',
            ),
        ):
            events = list(self.handler.run_whisper_job({
                'id': 'redaction-test',
                'input': input_payload,
            }))

        failure = events[-1]
        self.assertEqual(input_payload, original)
        self.assertEqual(failure['code'], 'MEDIA_FETCH_FAILED')
        self.assertEqual(failure['stage'], 'classic_download')
        self.assertTrue(failure['retryable'])
        self.assertEqual(
            failure['error'],
            'MEDIA_FETCH_FAILED: could not fetch audio from https://example.test',
        )
        for secret in (
            'super-secret',
            'fragment-secret',
            'user:password',
            '?token=',
            'audio.wav',
            '/private/',
        ):
            self.assertNotIn(secret, failure['error'])

    def test_classic_media_rejects_non_http_urls_and_strictly_caps_base64(self):
        url_events = list(self.handler.run_whisper_job({
            'id': 'classic-url-test',
            'input': {'audio': 'file:///private/audio.wav'},
        }))
        self.assertEqual(url_events[-1]['code'], 'INVALID_AUDIO_URL')
        self.assertNotIn('/private/', url_events[-1]['error'])

        for invalid_base64 in ('not-base64!', 'eA==\n'):
            events = list(self.handler.run_whisper_job({
                'id': 'classic-base64-test',
                'input': {'audio_base64': invalid_base64},
            }))
            self.assertEqual(events[-1]['code'], 'INVALID_AUDIO_BASE64')

        with mock.patch.object(self.handler, 'MAX_AUDIO_BYTES', 2):
            oversized_events = list(self.handler.run_whisper_job({
                'id': 'classic-base64-size-test',
                'input': {'audio_base64': 'QUJD'},
            }))
        self.assertEqual(oversized_events[-1]['code'], 'INVALID_AUDIO_BASE64')
        self.assertIn('2-byte limit', oversized_events[-1]['error'])

    def test_job_cleanup_rejects_traversal_and_resolves_exact_child(self):
        with mock.patch.object(self.handler.shutil, 'rmtree') as rmtree:
            for unsafe_id in (
                '../victim',
                'nested/job',
                r'..\victim',
                '.',
                '..',
                'job id',
            ):
                self.handler.cleanup_job_artifacts(unsafe_id)
            rmtree.assert_not_called()

            self.handler.cleanup_job_artifacts('job-123_abc')

        expected = os.path.join(
            os.path.realpath(os.path.abspath('jobs')),
            'job-123_abc',
        )
        rmtree.assert_called_once_with(expected, ignore_errors=True)

    def test_job_cleanup_is_best_effort_when_runpod_helper_raises(self):
        with mock.patch.object(
            self.handler.rp_cleanup,
            'clean',
            side_effect=OSError('private cleanup path'),
        ):
            self.assertFalse(
                self.handler.cleanup_job_artifacts('safe-job')
            )

    def test_draft_poll_retries_transient_http_statuses_with_backoff(self):
        signed_url = 'https://example.test/next?token=secret#fragment'
        transient_errors = [
            urllib.error.HTTPError(
                signed_url,
                429,
                'rate limited',
                {},
                None,
            )
            for _index in range(2)
        ]
        response = FakeResponse(status=204)
        with (
            mock.patch.object(
                self.handler.urllib.request,
                'urlopen',
                side_effect=[*transient_errors, response],
            ) as urlopen,
            mock.patch.object(self.handler.time, 'sleep') as sleep,
        ):
            result = self.handler.fetch_draft_audio('job', signed_url, [])

        self.assertFalse(result['available'])
        self.assertEqual(urlopen.call_count, 3)
        self.assertEqual(
            [call.args[0] for call in sleep.call_args_list],
            [0.25, 0.5],
        )
        for transient_error in transient_errors:
            transient_error.close()

    def test_draft_poll_urlerror_exhaustion_is_bounded_and_redacted(self):
        signed_url = 'https://example.test/next?token=secret#fragment'
        network_error = urllib.error.URLError(
            f'failed to reach {signed_url}',
        )
        with (
            mock.patch.object(
                self.handler.urllib.request,
                'urlopen',
                side_effect=network_error,
            ) as urlopen,
            mock.patch.object(self.handler.time, 'sleep'),
        ):
            with self.assertRaisesRegex(RuntimeError, 'DRAFT_POLL_FAILED') as raised:
                self.handler.fetch_draft_audio('job', signed_url, [])

        self.assertEqual(urlopen.call_count, 3)
        self.assertIn('attempts=3', str(raised.exception))
        self.assertNotIn('token=secret', str(raised.exception))
        self.assertNotIn('fragment', str(raised.exception))

    def test_draft_poll_does_not_start_after_deadline(self):
        poll_url = 'https://example.test/next'
        with mock.patch.object(
            self.handler.urllib.request,
            'urlopen',
        ) as urlopen:
            with self.assertRaisesRegex(
                RuntimeError,
                'DRAFT_POLL_DEADLINE_EXCEEDED',
            ):
                self.handler.fetch_draft_audio(
                    'job',
                    poll_url,
                    [],
                    deadline=time.monotonic() - 1,
                )
        urlopen.assert_not_called()

    def test_draft_poll_caps_response_body(self):
        signed_url = 'https://example.test/next?token=secret'
        response = FakeResponse(
            status=200,
            headers={'content-type': 'audio/aac'},
            body=b'123456789',
        )
        with (
            mock.patch.object(
                self.handler,
                'DRAFT_POLL_MAX_BODY_BYTES',
                8,
            ),
            mock.patch.object(
                self.handler.urllib.request,
                'urlopen',
                return_value=response,
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                'DRAFT_POLL_RESPONSE_TOO_LARGE',
            ) as raised:
                self.handler.fetch_draft_audio('job', signed_url, [])

        self.assertNotIn('token=secret', str(raised.exception))

    def test_url_downloader_caps_declared_and_streamed_bytes_and_cleans_partial(self):
        signed_url = 'https://user:pass@example.test/audio.wav?token=secret'
        declared_response = FakeResponse(
            headers={'content-length': '9', 'content-type': 'audio/wav'},
            body=b'123456789',
        )
        with mock.patch.object(
            self.handler.urllib.request,
            'urlopen',
            return_value=declared_response,
        ):
            with self.assertRaises(self.handler.MediaFetchFailure) as declared:
                self.handler.download_audio_url(
                    signed_url,
                    'test_download',
                    max_bytes=8,
                )
        self.assertEqual(declared.exception.payload['code'], 'MEDIA_FETCH_TOO_LARGE')
        self.assertFalse(declared.exception.payload['retryable'])
        self.assertEqual(
            declared.exception.payload['error'],
            (
                'MEDIA_FETCH_FAILED (MEDIA_FETCH_TOO_LARGE): audio from '
                'https://example.test exceeds the 67108864-byte limit'
            ),
        )

        created_paths = []
        real_named_tempfile = tempfile.NamedTemporaryFile

        def tracked_tempfile(*args, **kwargs):
            temp_file = real_named_tempfile(*args, **kwargs)
            created_paths.append(temp_file.name)
            return temp_file

        streamed_response = FakeResponse(
            headers={'content-type': 'audio/wav'},
            body=b'123456789',
        )
        with (
            mock.patch.object(
                self.handler.urllib.request,
                'urlopen',
                return_value=streamed_response,
            ),
            mock.patch.object(
                self.handler.tempfile,
                'NamedTemporaryFile',
                side_effect=tracked_tempfile,
            ),
        ):
            with self.assertRaises(self.handler.MediaFetchFailure) as streamed:
                self.handler.download_audio_url(
                    signed_url,
                    'test_download',
                    max_bytes=8,
                )
        self.assertEqual(streamed.exception.payload['code'], 'MEDIA_FETCH_TOO_LARGE')
        self.assertTrue(created_paths)
        self.assertTrue(all(not os.path.exists(path) for path in created_paths))
        for secret in ('user:pass', 'token=secret', '/audio.wav'):
            self.assertNotIn(secret, str(streamed.exception))

    def test_media_fetch_deadline_keeps_subtype_and_legacy_error_marker(self):
        failure = self.handler.media_fetch_failure_payload(
            'https://user:pass@example.test/private.wav?token=secret',
            'span_download',
            code='MEDIA_FETCH_DEADLINE_EXCEEDED',
            retryable=True,
        )

        self.assertEqual(failure['code'], 'MEDIA_FETCH_DEADLINE_EXCEEDED')
        self.assertEqual(failure['stage'], 'span_download')
        self.assertTrue(failure['retryable'])
        self.assertEqual(
            failure['error'],
            (
                'MEDIA_FETCH_FAILED (MEDIA_FETCH_DEADLINE_EXCEEDED): '
                'audio fetch from https://example.test exceeded its deadline'
            ),
        )
        for secret in ('user:pass', 'token=secret', '/private.wav'):
            self.assertNotIn(secret, failure['error'])

    def test_url_downloader_retries_transient_failure_and_returns_exact_file(self):
        signed_url = 'https://example.test/audio.wav?token=secret'
        transient = urllib.error.HTTPError(
            signed_url,
            503,
            'unavailable',
            {},
            None,
        )
        response = FakeResponse(
            headers={'content-length': '4', 'content-type': 'audio/wav'},
            body=b'RIFF',
        )
        with (
            mock.patch.object(
                self.handler.urllib.request,
                'urlopen',
                side_effect=[transient, response],
            ) as urlopen,
            mock.patch.object(self.handler.time, 'sleep') as sleep,
        ):
            path = self.handler.download_audio_url(
                signed_url,
                'test_download',
            )
        try:
            self.assertEqual(Path(path).read_bytes(), b'RIFF')
            self.assertEqual(Path(path).suffix, '.wav')
            self.assertEqual(urlopen.call_count, 2)
            sleep.assert_called_once_with(0.25)
        finally:
            self.handler.unlink_exact_file(path)
            transient.close()

    def test_draft_json_preserves_zero_cursor_and_rejects_bad_geometry_and_audio(self):
        poll_url = 'https://example.test/next'

        def json_response(payload):
            return FakeResponse(
                headers={'content-type': 'application/json'},
                body=bytes(__import__('json').dumps(payload), 'utf-8'),
            )

        with mock.patch.object(
            self.handler.urllib.request,
            'urlopen',
            return_value=json_response({'cursor': 0}),
        ):
            result = self.handler.fetch_draft_audio('job', poll_url, [])
        self.assertEqual(result['cursor'], 0)

        for payload in (
            {'audio_url': 42},
            {'audio_base64': []},
            {'audio_url': 'https://example.test/a.wav', 'start_sec': 'NaN'},
            {
                'audio_url': 'https://example.test/a.wav',
                'start_sec': 4,
                'end_sec': 3,
            },
        ):
            with (
                mock.patch.object(
                    self.handler.urllib.request,
                    'urlopen',
                    return_value=json_response(payload),
                ),
                mock.patch.object(
                    self.handler,
                    'download_audio_url',
                    return_value='/tmp/not-created',
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, 'DRAFT_POLL_INVALID_RESPONSE'):
                    self.handler.fetch_draft_audio('job', poll_url, [])

    def test_final_span_failures_continue_then_end_terminal_after_cleanup(self):
        paths = []
        for contents in (b'second', b'third'):
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(contents)
                paths.append(temp_file.name)

        class SucceedThenFailModel(StubPredictor):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def predict(self, **_kwargs):
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError('third span failed')
                return {
                    'transcription': 'second span',
                    'speaker_diarization': {
                        'turns': [
                            {'speaker': 'SPEAKER_00', 'start_sec': 0.5, 'end_sec': 1.0},
                        ],
                    },
                }

        spans = {
            'mode': 'final',
            'spans': [
                {'index': 7, 'audio': 'https://example.test/7.wav', 'start_sec': 10},
                {'index': 8, 'audio': 'https://example.test/8.wav', 'start_sec': 20},
                {'index': 9, 'audio': 'https://example.test/9.wav', 'start_sec': 30},
            ],
        }
        model = SucceedThenFailModel()
        fetch_failure = self.handler.MediaFetchFailure(
            spans['spans'][0]['audio'],
            'span_download',
        )
        with (
            mock.patch.object(self.handler, 'MODEL', model),
            mock.patch.object(
                self.handler,
                'download_audio_url',
                side_effect=[fetch_failure, *paths],
            ),
            mock.patch.object(
                self.handler,
                'cleanup_job_artifacts',
            ) as cleanup_job_artifacts,
        ):
            stream = self.handler.run_final_span_stream_job(
                {'id': 'final-test'},
                self.validated_input(),
                spans,
            )
            first = next(stream)
            second = next(stream)
            self.assertFalse(os.path.exists(paths[0]))
            third = next(stream)
            self.assertFalse(os.path.exists(paths[1]))
            cleanup_job_artifacts.assert_not_called()
            terminal = next(stream)
            cleanup_job_artifacts.assert_called_once_with('final-test')
            with self.assertRaises(StopIteration):
                next(stream)

        self.assertEqual(first['event'], 'span_error')
        self.assertEqual(first['failed_span_index'], 7)
        self.assertNotIn('span_index', first)
        self.assertNotIn('error', first)
        self.assertEqual(first['code'], 'MEDIA_FETCH_FAILED')
        self.assertEqual(second['span_index'], 8)
        self.assertEqual(third['event'], 'span_error')
        self.assertEqual(third['failed_span_index'], 9)
        self.assertNotIn('error', third)
        self.assertEqual(third['message'], 'SPAN_PROCESSING_FAILED: RuntimeError')
        self.assertEqual(
            terminal,
            {
                'error': (
                    'SPAN_STREAM_PARTIAL_FAILURE: failed_span_indexes=7,9; '
                    'codes=MEDIA_FETCH_FAILED,SPAN_PROCESSING_FAILED'
                ),
            },
        )
        sidecar = second['speaker_diarization']
        self.assertEqual(sidecar['timebase'], 'SPAN_RELATIVE_SECONDS')
        self.assertEqual(sidecar['span_index'], 8)
        self.assertEqual(sidecar['span_start_sec'], 20.0)
        self.assertEqual(
            sidecar['turns'],
            [{'speaker': 'SPEAKER_00', 'start_sec': 0.5, 'end_sec': 1.0}],
        )

    def test_final_span_all_success_has_no_terminal_error(self):
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(b'audio')
            path = temp_file.name

        with (
            mock.patch.object(self.handler, 'MODEL', StubPredictor()),
            mock.patch.object(
                self.handler,
                'download_audio_url',
                return_value=path,
            ),
            mock.patch.object(
                self.handler,
                'cleanup_job_artifacts',
            ) as cleanup_job_artifacts,
        ):
            events = list(self.handler.run_final_span_stream_job(
                {'id': 'all-success'},
                self.validated_input(),
                {
                    'mode': 'final',
                    'spans': [{
                        'index': 4,
                        'audio': 'https://example.test/4.wav',
                        'start_sec': 1,
                    }],
                },
            ))

        self.assertFalse(os.path.exists(path))
        cleanup_job_artifacts.assert_called_once_with('all-success')
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['span_index'], 4)
        self.assertNotIn('error', events[0])

    def test_draft_segment_file_is_deleted_before_segment_yield(self):
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(b'draft')
            path = temp_file.name

        model = StubPredictor()
        draft_audio = {
            'available': True,
            'audio_input': path,
            'start_sec': 0.0,
            'end_sec': 1.0,
            'timing': {},
        }
        with (
            mock.patch.object(self.handler, 'MODEL', model),
            mock.patch.object(
                self.handler,
                'fetch_draft_audio',
                return_value=draft_audio,
            ) as fetch_draft_audio,
        ):
            stream = self.handler.run_draft_span_stream_job(
                {'id': 'draft-test'},
                self.validated_input(),
                {
                    'mode': 'draft',
                    'next_url': 'https://example.test/next',
                    'poll_ms': 100,
                    'budget_sec': 5,
                    'idle_timeout_sec': 5,
                },
            )
            segment = next(stream)
            self.assertFalse(os.path.exists(path))
            self.assertIn(
                'deadline',
                fetch_draft_audio.call_args.kwargs,
            )
            stream.close()

        self.assertEqual(segment['event'], 'segment')

    def test_worker_build_sha_is_strictly_sanitized(self):
        valid_sha = 'a' * 40
        with mock.patch.dict(
            self.handler.os.environ,
            {'AUDIO_WORKER_BUILD_SHA': valid_sha},
        ):
            self.assertEqual(self.handler.get_worker_build_sha(), valid_sha)
        for unsafe_value in ('', 'abc', 'g' * 40, 'a' * 39, 'a' * 65, 'x\nleak'):
            with self.subTest(value=unsafe_value), mock.patch.dict(
                self.handler.os.environ,
                {'AUDIO_WORKER_BUILD_SHA': unsafe_value},
            ):
                self.assertEqual(
                    self.handler.get_worker_build_sha(),
                    'unknown',
                )


if __name__ == '__main__':
    unittest.main()
