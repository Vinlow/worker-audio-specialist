"""
rp_handler.py for runpod worker

rp_debugger:
- Utility that provides additional debugging information.
The handler must be called with --rp_debugger flag to enable it.
"""
import base64
import json
import os
import shutil
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

from rp_schema import INPUT_VALIDATIONS
from runpod.serverless.utils import download_files_from_urls, rp_cleanup, rp_debugger
from runpod.serverless.utils.rp_validator import validate
import runpod
import predict


MODEL = predict.Predictor()
MODEL.setup()


def base64_to_tempfile(base64_file: str) -> str:
    '''
    Convert base64 file to tempfile.

    Parameters:
    base64_file (str): Base64 file

    Returns:
    str: Path to tempfile
    '''
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_file.write(base64.b64decode(base64_file))

    return temp_file.name


def bytes_to_tempfile(data: bytes, suffix=".aac") -> str:
    '''
    Write binary audio bytes to a tempfile.
    '''
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        temp_file.write(data)

    return temp_file.name


def cleanup_job_artifacts(job_id, base64_temp_path=None):
    '''
    Remove all per-job disk artifacts.

    download_files_from_urls saves into jobs/<job_id>/downloaded_files/ and the
    runpod SDK never deletes that directory — rp_cleanup.clean() only touches
    input_objects/, output_objects/, job_files/ and output.zip, none of which
    this worker uses. Before this function existed, every URL job leaked its
    full audio chunk (~10-25 MB per 120s WAV) on the warm worker until the
    container disk filled and jobs started failing with opaque disk errors.
    Base64 jobs leaked their tempfile the same way.
    '''
    if job_id:
        shutil.rmtree(os.path.join("jobs", str(job_id)), ignore_errors=True)
    if base64_temp_path:
        try:
            os.unlink(base64_temp_path)
        except OSError:
            pass
    rp_cleanup.clean(['input_objects'])


def to_jsonable(o):
    '''Convert numpy types to plain Python so json.dumps doesn't choke.'''
    if isinstance(o, dict):
        return {k: to_jsonable(v) for k, v in o.items()}
    if isinstance(o, list):
        return [to_jsonable(x) for x in o]
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return to_jsonable(o.tolist())
    return o


def validate_final_span_stream(span_stream):
    '''
    Validate the Holy Grale final-tier span-stream input shape.
    '''
    spans = span_stream.get('spans')
    if not isinstance(spans, list) or len(spans) == 0:
        return 'span_stream.spans must be a non-empty list'
    for pos, span in enumerate(spans):
        if not isinstance(span, dict):
            return f'span_stream.spans[{pos}] must be an object'
        index = span.get('index')
        if not isinstance(index, int) or index < 0:
            return f'span_stream.spans[{pos}].index must be a non-negative integer'
        audio = span.get('audio')
        if not isinstance(audio, str) or audio.strip() == '':
            return f'span_stream.spans[{pos}].audio must be a non-empty URL'
        start_sec = span.get('start_sec')
        if not isinstance(start_sec, (int, float)) or start_sec < 0:
            return f'span_stream.spans[{pos}].start_sec must be a non-negative number'
    return None


def validate_draft_span_stream(span_stream):
    '''
    Validate the Holy Grale draft ticker span-stream input shape.
    '''
    next_url = span_stream.get('next_url')
    if not isinstance(next_url, str) or next_url.strip() == '':
        return 'span_stream.next_url must be a non-empty URL'

    poll_ms = span_stream.get('poll_ms', 500)
    if not isinstance(poll_ms, (int, float)) or poll_ms < 100 or poll_ms > 5000:
        return 'span_stream.poll_ms must be a number between 100 and 5000'

    budget_sec = span_stream.get('budget_sec', 480)
    if not isinstance(budget_sec, (int, float)) or budget_sec <= 0 or budget_sec > 540:
        return 'span_stream.budget_sec must be a number between 0 and 540'

    idle_timeout_sec = span_stream.get('idle_timeout_sec', 30)
    if not isinstance(idle_timeout_sec, (int, float)) or idle_timeout_sec <= 0 or idle_timeout_sec > 120:
        return 'span_stream.idle_timeout_sec must be a number between 0 and 120'

    return None


def validate_draft_warmup_span_stream(span_stream):
    '''
    Validate the Holy Grale draft warmup input shape.
    '''
    model = span_stream.get('model')
    if model is not None and (not isinstance(model, str) or model.strip() == ''):
        return 'span_stream.model must be a non-empty string when provided'
    return None


def validate_span_stream(span_stream):
    '''
    Validate the Holy Grale span-stream input shape.

    Returns an error string on invalid input, otherwise None.
    '''
    if not isinstance(span_stream, dict):
        return 'span_stream must be an object'
    mode = span_stream.get('mode')
    if mode == 'final':
        return validate_final_span_stream(span_stream)
    if mode == 'draft':
        return validate_draft_span_stream(span_stream)
    if mode == 'draft_warmup':
        return validate_draft_warmup_span_stream(span_stream)
    return 'span_stream.mode must be "final", "draft", or "draft_warmup"'


def update_url_cursor(next_url, cursor):
    '''
    Add or replace the `after` query param used by the draft polling endpoint.
    '''
    if cursor is None:
        return next_url
    parsed = urllib.parse.urlparse(next_url)
    query = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
    query['after'] = [str(cursor)]
    return urllib.parse.urlunparse(parsed._replace(query=urllib.parse.urlencode(query, doseq=True)))


def parse_float(value, default=None):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def elapsed_ms(start):
    return int(max(0.0, (time.monotonic() - start) * 1000.0))


def fetch_draft_audio(job_id, next_url, temp_paths):
    '''
    Poll the draft next-audio endpoint.

    Supported responses:
    - 204: no new audio yet
    - application/json: {audio|audio_url|audio_base64, cursor, next_url, start_sec, end_sec, done}
    - audio bytes: body is the next micro-segment; cursor/start/end can be response headers
    '''
    request_started = time.monotonic()
    timing = {
        'request_ms': 0,
        'body_bytes': 0,
        'audio_download_ms': 0,
    }
    request = urllib.request.Request(
        next_url,
        headers={
            'Accept': 'application/json,audio/aac,audio/wav,*/*',
            'User-Agent': 'web2labs-audio-specialist-draft/1',
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            status = response.getcode()
            headers = response.headers
            body = response.read()
            timing['request_ms'] = elapsed_ms(request_started)
            timing['body_bytes'] = len(body or b'')
    except urllib.error.HTTPError as error:
        timing['request_ms'] = elapsed_ms(request_started)
        if error.code in (204, 404):
            return {'available': False, 'timing': timing}
        raise

    if status == 204:
        return {'available': False, 'timing': timing}

    content_type = (headers.get('content-type') or '').lower()
    if 'application/json' in content_type:
        payload = json.loads(body.decode('utf-8') or '{}')
        if payload.get('done'):
            return {
                'available': False,
                'done': True,
                'cursor': payload.get('cursor') or payload.get('next_cursor'),
                'next_url': payload.get('next_url'),
                'timing': timing,
            }

        audio_input = None
        audio_url = payload.get('audio') or payload.get('audio_url')
        if isinstance(audio_url, str) and audio_url.strip():
            download_started = time.monotonic()
            audio_input = download_files_from_urls(job_id, [audio_url])[0]
            timing['audio_download_ms'] = elapsed_ms(download_started)
            if not audio_input:
                raise RuntimeError(f"MEDIA_FETCH_FAILED: could not download audio from {audio_url}")
        else:
            audio_base64 = payload.get('audio_base64') or payload.get('audio_b64')
            if isinstance(audio_base64, str) and audio_base64.strip():
                audio_input = base64_to_tempfile(audio_base64)
                temp_paths.append(audio_input)

        if not audio_input:
            return {
                'available': False,
                'cursor': payload.get('cursor') or payload.get('next_cursor'),
                'next_url': payload.get('next_url'),
                'timing': timing,
            }

        return {
            'available': True,
            'audio_input': audio_input,
            'cursor': payload.get('cursor') or payload.get('next_cursor'),
            'next_url': payload.get('next_url'),
            'start_sec': parse_float(payload.get('start_sec'), 0.0),
            'end_sec': parse_float(payload.get('end_sec')),
            'timing': timing,
        }

    if not body:
        return {'available': False, 'timing': timing}

    suffix = '.wav' if 'wav' in content_type else '.aac'
    audio_input = bytes_to_tempfile(body, suffix=suffix)
    temp_paths.append(audio_input)
    return {
        'available': True,
        'audio_input': audio_input,
        'cursor': headers.get('x-next-cursor') or headers.get('x-cursor'),
        'next_url': headers.get('x-next-url'),
        'start_sec': parse_float(headers.get('x-start-sec'), 0.0),
        'end_sec': parse_float(headers.get('x-end-sec')),
        'timing': timing,
    }


def run_final_span_stream_job(job, job_input, span_stream):
    '''
    Run final-tier transcription for multiple ready spans and yield each result.

    The wrapper handler returns this generator only for span-stream jobs; classic
    single-audio jobs still return a plain dict from run_whisper_job.
    '''
    try:
        for span_pos, span in enumerate(span_stream['spans']):
            span_index = int(span['index'])
            audio_url = span['audio']
            start_sec = float(span['start_sec'])
            with rp_debugger.LineTimer(f'span_{span_pos}_download_step'):
                audio_input = download_files_from_urls(job['id'], [audio_url])[0]
            if not audio_input:
                raise RuntimeError(f"MEDIA_FETCH_FAILED: could not download audio from {audio_url}")

            with rp_debugger.LineTimer(f'span_{span_pos}_prediction_step'):
                whisper_results = MODEL.predict(
                    audio=audio_input,
                    model_name=job_input["model"],
                    transcription=job_input["transcription"],
                    translation=job_input["translation"],
                    translate=job_input["translate"],
                    language=job_input["language"],
                    temperature=job_input["temperature"],
                    best_of=job_input["best_of"],
                    beam_size=job_input["beam_size"],
                    patience=job_input["patience"],
                    length_penalty=job_input["length_penalty"],
                    suppress_tokens=job_input.get("suppress_tokens", "-1"),
                    initial_prompt=job_input["initial_prompt"],
                    condition_on_previous_text=job_input["condition_on_previous_text"],
                    temperature_increment_on_fallback=job_input["temperature_increment_on_fallback"],
                    compression_ratio_threshold=job_input["compression_ratio_threshold"],
                    logprob_threshold=job_input["logprob_threshold"],
                    no_speech_threshold=job_input["no_speech_threshold"],
                    enable_vad=job_input["enable_vad"],
                    word_timestamps=job_input["word_timestamps"],
                    clap_queries=job_input.get("clap_queries"),
                    force_align=job_input.get("force_align", False),
                )
            whisper_results["span_index"] = span_index
            whisper_results["start_sec"] = start_sec
            yield to_jsonable(whisper_results)
    finally:
        with rp_debugger.LineTimer('span_stream_cleanup_step'):
            cleanup_job_artifacts(job.get('id'))


def run_draft_span_stream_job(job, job_input, span_stream):
    '''
    Run draft-tier pull-loop transcription and yield ticker batches.
    '''
    next_url = span_stream['next_url']
    cursor = span_stream.get('cursor')
    poll_ms = float(span_stream.get('poll_ms', 500))
    budget_sec = float(span_stream.get('budget_sec', 480))
    idle_timeout_sec = float(span_stream.get('idle_timeout_sec', 30))
    job_started = time.monotonic()
    budget_deadline = time.monotonic() + budget_sec
    idle_deadline = time.monotonic() + idle_timeout_sec
    poll_index = 0
    yield_index = 0
    temp_paths = []
    model_warmup_ms = None
    model_warmup_error = None

    def warm_turbo_model():
        nonlocal model_warmup_ms, model_warmup_error
        warm_started = time.monotonic()
        try:
            MODEL.ensure_model_loaded('turbo')
        except Exception as error:
            model_warmup_error = str(error)
        finally:
            model_warmup_ms = elapsed_ms(warm_started)

    warmup_thread = threading.Thread(
        target=warm_turbo_model,
        name='draft-turbo-warmup',
        daemon=True,
    )
    warmup_thread.start()

    def closed(reason):
        return {
            'mode': 'draft',
            'event': 'closed',
            'reason': reason,
            'cursor': cursor,
            'next_url': next_url,
            'yield_index': yield_index,
            'timing': {
                'job_elapsed_ms': elapsed_ms(job_started),
                'model_warmup_ms': model_warmup_ms,
            },
        }

    try:
        while time.monotonic() < budget_deadline:
            poll_url = update_url_cursor(next_url, cursor)
            current_poll_index = poll_index
            poll_index += 1
            with rp_debugger.LineTimer(f'draft_poll_step_{current_poll_index}'):
                draft_audio = fetch_draft_audio(job.get('id'), poll_url, temp_paths)

            if draft_audio.get('next_url'):
                next_url = draft_audio['next_url']
            if draft_audio.get('cursor') is not None:
                cursor = draft_audio['cursor']

            if draft_audio.get('done'):
                yield closed('eof')
                return

            if not draft_audio.get('available'):
                if time.monotonic() >= idle_deadline:
                    yield closed('idle_timeout')
                    return
                time.sleep(poll_ms / 1000.0)
                continue

            idle_deadline = time.monotonic() + idle_timeout_sec
            start_sec = float(draft_audio.get('start_sec') or 0.0)
            end_sec = draft_audio.get('end_sec')
            model_wait_started = time.monotonic()
            if warmup_thread.is_alive():
                warmup_thread.join()
            model_warmup_wait_ms = elapsed_ms(model_wait_started)
            if model_warmup_error:
                raise RuntimeError(f"draft turbo warmup failed: {model_warmup_error}")

            prediction_started = time.monotonic()
            with rp_debugger.LineTimer(f'draft_prediction_step_{yield_index}'):
                whisper_results = MODEL.predict(
                    audio=draft_audio['audio_input'],
                    model_name='turbo',
                    transcription='plain_text',
                    translation='plain_text',
                    translate=False,
                    language=job_input["language"],
                    temperature=0,
                    best_of=1,
                    beam_size=1,
                    patience=1.0,
                    length_penalty=job_input["length_penalty"],
                    suppress_tokens=job_input.get("suppress_tokens", "-1"),
                    initial_prompt=job_input["initial_prompt"],
                    condition_on_previous_text=job_input["condition_on_previous_text"],
                    temperature_increment_on_fallback=job_input["temperature_increment_on_fallback"],
                    compression_ratio_threshold=job_input["compression_ratio_threshold"],
                    logprob_threshold=job_input["logprob_threshold"],
                    no_speech_threshold=job_input["no_speech_threshold"],
                    enable_vad=job_input["enable_vad"],
                    word_timestamps=True,
                    clap_queries=None,
                    force_align=False,
                )
            prediction_ms = elapsed_ms(prediction_started)

            words = whisper_results.get('word_timestamps') or []
            if end_sec is None:
                last_word_end = max([parse_float(word.get('end'), 0.0) for word in words], default=0.0)
                end_sec = start_sec + last_word_end
            fetch_timing = draft_audio.get('timing') or {}

            yield to_jsonable({
                'mode': 'draft',
                'event': 'segment',
                'yield_index': yield_index,
                'cursor': cursor,
                'next_url': next_url,
                'start_sec': start_sec,
                'end_sec': end_sec,
                'words': words,
                'transcription': whisper_results.get('transcription', ''),
                'segments': whisper_results.get('segments', []),
                'detected_language': whisper_results.get('detected_language'),
                'model': whisper_results.get('model'),
                'timing': {
                    'job_elapsed_ms': elapsed_ms(job_started),
                    'poll_index': current_poll_index,
                    'poll_ms': fetch_timing.get('request_ms'),
                    'poll_body_bytes': fetch_timing.get('body_bytes'),
                    'audio_download_ms': fetch_timing.get('audio_download_ms'),
                    'model_warmup_ms': model_warmup_ms,
                    'model_warmup_wait_ms': model_warmup_wait_ms,
                    'prediction_ms': prediction_ms,
                },
            })
            yield_index += 1

        yield closed('budget_exhausted')
    finally:
        with rp_debugger.LineTimer('draft_span_stream_cleanup_step'):
            for path in temp_paths:
                try:
                    os.unlink(path)
                except OSError:
                    pass
            cleanup_job_artifacts(job.get('id'))


def run_draft_warmup_span_stream_job(job, job_input, span_stream):
    '''
    Load the draft ASR model without polling audio so Studio can hide cold start
    behind draft creation/upload time.
    '''
    job_started = time.monotonic()
    model_name = span_stream.get('model') or job_input.get('model') or 'turbo'
    warmup_started = time.monotonic()
    try:
        MODEL.ensure_model_loaded(model_name)
        model_warmup_ms = elapsed_ms(warmup_started)
        yield to_jsonable({
            'mode': 'draft_warmup',
            'event': 'warmed',
            'model': model_name,
            'yield_index': 0,
            'timing': {
                'job_elapsed_ms': elapsed_ms(job_started),
                'model_warmup_ms': model_warmup_ms,
            },
        })
    finally:
        with rp_debugger.LineTimer('draft_warmup_cleanup_step'):
            cleanup_job_artifacts(job.get('id'))


def run_span_stream_job(job, job_input, span_stream):
    if span_stream.get('mode') == 'draft':
        return run_draft_span_stream_job(job, job_input, span_stream)
    if span_stream.get('mode') == 'draft_warmup':
        return run_draft_warmup_span_stream_job(job, job_input, span_stream)
    return run_final_span_stream_job(job, job_input, span_stream)


def run_whisper_job(job):
    '''
    Run inference on the model.

    Parameters:
    job (dict): Input job containing the model parameters

    Yields:
    dict: Streaming results. Runpod detects streaming support from this function
    being a generator function via inspect.isgeneratorfunction(). Do not wrap
    this function with class decorators such as rp_debugger.FunctionTimer; that
    hides the generator shape and disables /stream support.
    '''
    job_input = job['input']

    # Extract clap_queries before validation — rp_validator chokes on dict types
    raw_clap_queries = job_input.pop('clap_queries', None)
    raw_span_stream = job_input.pop('span_stream', None)

    # RunPod serverless streaming jobs can be retried as timed out if a cold
    # worker spends too long loading models before the first stream item.
    # Emit a cheap control item immediately; Studio ignores unknown events.
    if raw_span_stream is not None:
        yield {
            'mode': raw_span_stream.get('mode'),
            'event': 'started',
            'yield_index': -1,
        }
    else:
        yield {
            'mode': 'classic',
            'event': 'started',
            'yield_index': -1,
        }

    with rp_debugger.LineTimer('validation_step'):
        input_validation = validate(job_input, INPUT_VALIDATIONS)

        if 'errors' in input_validation:
            yield {"error": input_validation['errors']}
            return
        job_input = input_validation['validated_input']

    # Restore clap_queries after validation
    if raw_clap_queries and isinstance(raw_clap_queries, dict):
        job_input['clap_queries'] = raw_clap_queries

    if raw_span_stream is not None:
        span_error = validate_span_stream(raw_span_stream)
        if span_error:
            yield {'error': span_error}
            return
        yield from run_span_stream_job(job, job_input, raw_span_stream)
        return

    if not job_input.get('audio', False) and not job_input.get('audio_base64', False):
        yield {'error': 'Must provide either audio or audio_base64'}
        return

    if job_input.get('audio', False) and job_input.get('audio_base64', False):
        yield {'error': 'Must provide either audio or audio_base64, not both'}
        return

    base64_temp_path = None
    try:
        if job_input.get('audio', False):
            with rp_debugger.LineTimer('download_step'):
                audio_input = download_files_from_urls(job['id'], [job_input['audio']])[0]
            if not audio_input:
                # download_files_from_urls returns None for a failed download
                # (after 3 internal retries with backoff). Without this guard the
                # None reaches av.open('None') deep inside faster_whisper and
                # crashes with a misleading FileNotFoundError (2026-07-03
                # web2labs abstain-matrix post-mortem). Fail with a clear,
                # classifiable signature instead — the web2labs server
                # recognizes MEDIA_FETCH_FAILED and skips its model-fallback retry.
                yield {'error': f"MEDIA_FETCH_FAILED: could not download audio from {job_input['audio']}"}
                return

        if job_input.get('audio_base64', False):
            base64_temp_path = base64_to_tempfile(job_input['audio_base64'])
            audio_input = base64_temp_path

        with rp_debugger.LineTimer('prediction_step'):
            whisper_results = MODEL.predict(
                audio=audio_input,
                model_name=job_input["model"],
                transcription=job_input["transcription"],
                translation=job_input["translation"],
                translate=job_input["translate"],
                language=job_input["language"],
                temperature=job_input["temperature"],
                best_of=job_input["best_of"],
                beam_size=job_input["beam_size"],
                patience=job_input["patience"],
                length_penalty=job_input["length_penalty"],
                suppress_tokens=job_input.get("suppress_tokens", "-1"),
                initial_prompt=job_input["initial_prompt"],
                condition_on_previous_text=job_input["condition_on_previous_text"],
                temperature_increment_on_fallback=job_input["temperature_increment_on_fallback"],
                compression_ratio_threshold=job_input["compression_ratio_threshold"],
                logprob_threshold=job_input["logprob_threshold"],
                no_speech_threshold=job_input["no_speech_threshold"],
                enable_vad=job_input["enable_vad"],
                word_timestamps=job_input["word_timestamps"],
                clap_queries=job_input.get("clap_queries"),
                force_align=job_input.get("force_align", False),
                diarize=job_input.get("diarize", False),
                diarize_min_speakers=job_input.get("diarize_min_speakers") or None,
                diarize_max_speakers=job_input.get("diarize_max_speakers") or None,
            )
    finally:
        # Always clean up job artifacts — success, MEDIA_FETCH_FAILED return, or
        # a predict() exception. Before the try/finally, any exception skipped
        # cleanup and leaked the downloaded audio on the warm worker.
        with rp_debugger.LineTimer('cleanup_step'):
            cleanup_job_artifacts(job.get('id'), base64_temp_path)

    # If TEST_OUTPUT_PATH is set (local Docker test mode), dump the full result
    # as JSON to that path so we can inspect all word timestamps without hitting
    # the docker stdout buffer limit on long audio.
    output_path = os.environ.get("TEST_OUTPUT_PATH")
    if output_path:
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(to_jsonable(whisper_results), f)
            print(f"[Test] Full result written to {output_path}", flush=True)
        except Exception as e:
            print(f"[Test] Failed to write {output_path}: {e}", flush=True)

    yield to_jsonable(whisper_results)
    return


runpod.serverless.start({
    "handler": run_whisper_job,
    "return_aggregate_stream": True,
})
