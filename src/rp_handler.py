"""
rp_handler.py for runpod worker

rp_debugger:
- Utility that provides additional debugging information.
The handler must be called with --rp_debugger flag to enable it.
"""
import base64
import binascii
import http.client
import json
import math
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
from runpod.serverless.utils import rp_cleanup, rp_debugger
from runpod.serverless.utils.rp_validator import validate
import runpod
import predict
from clap_scorer import (
    MAX_QUERY_COUNT,
    MAX_QUERY_NAME_CHARS,
    MAX_QUERY_TEXT_CHARS,
    MAX_TOTAL_QUERY_TEXT_CHARS,
)


def get_worker_build_sha():
    '''Return a safe immutable build identity for logs, or ``unknown``.'''
    value = os.environ.get('AUDIO_WORKER_BUILD_SHA', '').strip().lower()
    if len(value) not in (40, 64):
        return 'unknown'
    if not all(char in '0123456789abcdef' for char in value):
        return 'unknown'
    return value


WORKER_BUILD_SHA = get_worker_build_sha()
print(f'[AudioWorkerStartup] build_sha={WORKER_BUILD_SHA}', flush=True)


MODEL = predict.Predictor()
MODEL.setup()


_MISSING = object()
MAX_CLAP_QUERIES = MAX_QUERY_COUNT
MAX_CLAP_QUERY_NAME_CHARS = MAX_QUERY_NAME_CHARS
MAX_CLAP_QUERY_DESCRIPTION_CHARS = MAX_QUERY_TEXT_CHARS
MAX_CLAP_QUERY_TOTAL_DESCRIPTION_CHARS = MAX_TOTAL_QUERY_TEXT_CHARS
MAX_FINAL_SPANS = 64
MAX_DIARIZATION_SPEAKERS = 64
MAX_URL_CHARS = 8192
MAX_AUDIO_BYTES = 64 * 1024 * 1024
MAX_AUDIO_BASE64_CHARS = 4 * ((MAX_AUDIO_BYTES + 2) // 3)
DRAFT_POLL_MAX_ATTEMPTS = 3
DRAFT_POLL_BACKOFF_SECONDS = 0.25
DRAFT_POLL_MAX_BACKOFF_SECONDS = 2.0
DRAFT_POLL_MAX_BODY_BYTES = 64 * 1024 * 1024
MEDIA_FETCH_MAX_ATTEMPTS = 3
MEDIA_FETCH_BACKOFF_SECONDS = 0.25
MEDIA_FETCH_MAX_BACKOFF_SECONDS = 2.0
MEDIA_FETCH_READ_BYTES = 1024 * 1024
MEDIA_FETCH_ALLOWED_SUFFIXES = {
    '.aac',
    '.flac',
    '.m4a',
    '.mp3',
    '.mp4',
    '.ogg',
    '.opus',
    '.wav',
    '.webm',
}


def safe_url_for_error(url):
    '''Return only a URL origin, without credentials, path, query, or fragment.'''
    try:
        parsed = urllib.parse.urlsplit(str(url))
        hostname = parsed.hostname
        if not hostname or parsed.scheme.lower() not in ('http', 'https'):
            return '<redacted-url>'
        host = f'[{hostname}]' if ':' in hostname else hostname
        try:
            port = parsed.port
        except ValueError:
            port = None
        if port is not None:
            host = f'{host}:{port}'
        return f'{parsed.scheme.lower()}://{host}'
    except (TypeError, ValueError):
        return '<redacted-url>'


def validate_http_url(value, field_name):
    '''Validate a bounded HTTP(S) URL without resolving or fetching it.'''
    if not isinstance(value, str) or value.strip() == '':
        return f'{field_name} must be a non-empty URL'
    if len(value) > MAX_URL_CHARS:
        return f'{field_name} must be at most {MAX_URL_CHARS} characters'
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        return f'{field_name} must be a valid HTTP(S) URL'
    try:
        parsed = urllib.parse.urlsplit(value)
        has_valid_port = parsed.port is None or 0 < parsed.port <= 65535
    except ValueError:
        return f'{field_name} must be a valid HTTP(S) URL'
    if (
        parsed.scheme.lower() not in ('http', 'https')
        or not parsed.hostname
        or not has_valid_port
    ):
        return f'{field_name} must use an http or https URL'
    return None


def media_fetch_failure_payload(
    url,
    stage,
    code='MEDIA_FETCH_FAILED',
    retryable=True,
):
    safe_url = safe_url_for_error(url)
    error_label = (
        'MEDIA_FETCH_FAILED'
        if code == 'MEDIA_FETCH_FAILED'
        else f'MEDIA_FETCH_FAILED ({code})'
    )
    if code == 'MEDIA_FETCH_TOO_LARGE':
        message = (
            f'{error_label}: audio from {safe_url} exceeds the '
            f'{MAX_AUDIO_BYTES}-byte limit'
        )
    elif code == 'MEDIA_FETCH_DEADLINE_EXCEEDED':
        message = (
            f'{error_label}: audio fetch from {safe_url} exceeded its deadline'
        )
    else:
        message = f'{error_label}: could not fetch audio from {safe_url}'
    return {
        'error': message,
        'code': code,
        'stage': stage,
        'retryable': retryable,
    }


class MediaFetchFailure(RuntimeError):
    def __init__(
        self,
        url,
        stage,
        code='MEDIA_FETCH_FAILED',
        retryable=True,
    ):
        self.payload = media_fetch_failure_payload(
            url,
            stage,
            code=code,
            retryable=retryable,
        )
        super().__init__(self.payload['error'])


def unlink_exact_file(path):
    '''Best-effort deletion of one exact per-segment file.'''
    if not path:
        return True
    try:
        os.unlink(path)
        return True
    except FileNotFoundError:
        return True
    except (OSError, TypeError, ValueError):
        return False


def is_finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def base64_to_tempfile(base64_file: str) -> str:
    '''
    Convert base64 file to tempfile.

    Parameters:
    base64_file (str): Base64 file

    Returns:
    str: Path to tempfile
    '''
    if not isinstance(base64_file, str) or base64_file == '':
        raise ValueError('audio_base64 must be a non-empty base64 string')
    if len(base64_file) > MAX_AUDIO_BASE64_CHARS:
        raise ValueError(f'audio_base64 exceeds the {MAX_AUDIO_BYTES}-byte limit')
    try:
        decoded_audio = base64.b64decode(base64_file, validate=True)
    except (binascii.Error, TypeError, ValueError):
        raise ValueError('audio_base64 must be strictly valid base64') from None
    if not decoded_audio:
        raise ValueError('audio_base64 must decode to non-empty audio bytes')
    if len(decoded_audio) > MAX_AUDIO_BYTES:
        raise ValueError(f'audio_base64 exceeds the {MAX_AUDIO_BYTES}-byte limit')

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(decoded_audio)
        return temp_path
    except Exception:
        unlink_exact_file(temp_path)
        raise


def bytes_to_tempfile(data: bytes, suffix=".aac") -> str:
    '''
    Write binary audio bytes to a tempfile.
    '''
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(data)
        return temp_path
    except Exception:
        unlink_exact_file(temp_path)
        raise


def media_suffix(url, content_type=''):
    '''Choose a harmless decoder hint without trusting arbitrary URL paths.'''
    content_type = str(content_type or '').split(';', 1)[0].strip().lower()
    content_type_suffixes = {
        'audio/aac': '.aac',
        'audio/flac': '.flac',
        'audio/m4a': '.m4a',
        'audio/mp4': '.m4a',
        'audio/mpeg': '.mp3',
        'audio/ogg': '.ogg',
        'audio/opus': '.opus',
        'audio/wav': '.wav',
        'audio/x-wav': '.wav',
        'video/mp4': '.mp4',
        'video/webm': '.webm',
    }
    if content_type in content_type_suffixes:
        return content_type_suffixes[content_type]
    try:
        suffix = os.path.splitext(urllib.parse.urlsplit(url).path)[1].lower()
    except (TypeError, ValueError):
        suffix = ''
    return suffix if suffix in MEDIA_FETCH_ALLOWED_SUFFIXES else '.audio'


def remaining_timeout(url, stage, deadline, attempts):
    if deadline is None:
        return 15.0
    remaining_sec = deadline - time.monotonic()
    if remaining_sec <= 0:
        raise MediaFetchFailure(
            url,
            stage,
            code='MEDIA_FETCH_DEADLINE_EXCEEDED',
            retryable=True,
        )
    return min(15.0, max(0.001, remaining_sec))


def sleep_before_media_retry(url, stage, attempt, deadline):
    delay_sec = min(
        MEDIA_FETCH_MAX_BACKOFF_SECONDS,
        MEDIA_FETCH_BACKOFF_SECONDS * (2 ** (attempt - 1)),
    )
    if deadline is not None:
        remaining_sec = deadline - time.monotonic()
        if remaining_sec <= 0:
            raise MediaFetchFailure(
                url,
                stage,
                code='MEDIA_FETCH_DEADLINE_EXCEEDED',
                retryable=True,
            )
        delay_sec = min(delay_sec, remaining_sec)
    time.sleep(delay_sec)


def download_audio_url(url, stage, deadline=None, max_bytes=None):
    '''Download one media URL with retries, a hard byte ceiling, and no URL logs.'''
    max_bytes = MAX_AUDIO_BYTES if max_bytes is None else int(max_bytes)
    url_error = validate_http_url(url, 'audio URL')
    if url_error:
        raise MediaFetchFailure(url, stage, retryable=False)

    for attempt in range(1, MEDIA_FETCH_MAX_ATTEMPTS + 1):
        temp_path = None
        request = urllib.request.Request(
            url,
            headers={
                'Accept': 'audio/*,video/mp4,video/webm,application/octet-stream',
                'User-Agent': 'web2labs-audio-specialist/1',
            },
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=remaining_timeout(url, stage, deadline, attempt),
            ) as response:
                status = response.getcode()
                if not isinstance(status, int):
                    raise MediaFetchFailure(url, stage, retryable=False)
                if status == 408 or status == 429 or status >= 500:
                    raise urllib.error.HTTPError(
                        url,
                        status,
                        'retryable media response',
                        response.headers,
                        None,
                    )
                if status >= 400:
                    raise MediaFetchFailure(url, stage, retryable=False)

                declared_value = response.headers.get('content-length')
                try:
                    declared_bytes = (
                        int(declared_value)
                        if declared_value is not None
                        else None
                    )
                except (TypeError, ValueError):
                    declared_bytes = None
                if declared_bytes is not None and declared_bytes > max_bytes:
                    raise MediaFetchFailure(
                        url,
                        stage,
                        code='MEDIA_FETCH_TOO_LARGE',
                        retryable=False,
                    )

                suffix = media_suffix(
                    url,
                    response.headers.get('content-type'),
                )
                total_bytes = 0
                with tempfile.NamedTemporaryFile(
                    suffix=suffix,
                    delete=False,
                ) as temp_file:
                    temp_path = temp_file.name
                    while True:
                        # urlopen timeouts are per blocking socket operation.
                        # This check also enforces the whole-operation deadline
                        # between bounded reads.
                        remaining_timeout(url, stage, deadline, attempt)
                        chunk = response.read(
                            min(
                                MEDIA_FETCH_READ_BYTES,
                                max_bytes - total_bytes + 1,
                            )
                        )
                        if not chunk:
                            break
                        total_bytes += len(chunk)
                        if total_bytes > max_bytes:
                            raise MediaFetchFailure(
                                url,
                                stage,
                                code='MEDIA_FETCH_TOO_LARGE',
                                retryable=False,
                            )
                        temp_file.write(chunk)
                if total_bytes <= 0:
                    raise MediaFetchFailure(url, stage, retryable=False)
                return temp_path
        except MediaFetchFailure:
            unlink_exact_file(temp_path)
            raise
        except urllib.error.HTTPError as error:
            unlink_exact_file(temp_path)
            retryable = (
                error.code == 408
                or error.code == 429
                or error.code >= 500
            )
            if retryable and attempt < MEDIA_FETCH_MAX_ATTEMPTS:
                sleep_before_media_retry(url, stage, attempt, deadline)
                continue
            raise MediaFetchFailure(url, stage, retryable=retryable) from None
        except (
            OSError,
            TimeoutError,
            urllib.error.URLError,
            http.client.HTTPException,
        ):
            unlink_exact_file(temp_path)
            if attempt < MEDIA_FETCH_MAX_ATTEMPTS:
                sleep_before_media_retry(url, stage, attempt, deadline)
                continue
            raise MediaFetchFailure(url, stage, retryable=True) from None

    raise MediaFetchFailure(url, stage, retryable=True)


def resolve_job_artifact_dir(job_id):
    '''Resolve only an ordinary, exact child of the local jobs directory.'''
    if job_id is None:
        return None
    job_id_text = str(job_id)
    if (
        not 1 <= len(job_id_text) <= 128
        or job_id_text in ('.', '..')
        or not all(
            char.isascii() and (char.isalnum() or char in '._-')
            for char in job_id_text
        )
    ):
        return None
    jobs_root = os.path.realpath(os.path.abspath('jobs'))
    expected_child = os.path.join(jobs_root, job_id_text)
    if os.path.realpath(expected_child) != expected_child:
        return None
    if os.path.dirname(expected_child) != jobs_root:
        return None
    return expected_child


def cleanup_job_artifacts(job_id, base64_temp_path=None):
    '''
    Remove all per-job disk artifacts.

    Older images used RunPod's downloader under jobs/<job_id>/downloaded_files/.
    Keep removing that legacy directory during rolling upgrades; the bounded
    downloader and base64 path now use exact temporary files as well.
    '''
    job_artifact_dir = resolve_job_artifact_dir(job_id)
    if job_artifact_dir:
        shutil.rmtree(job_artifact_dir, ignore_errors=True)
    if base64_temp_path:
        try:
            os.unlink(base64_temp_path)
        except OSError:
            pass
    try:
        rp_cleanup.clean(['input_objects'])
    except Exception as error:
        # Cleanup is best-effort.  In particular, never let RunPod's helper
        # replace a stable, sanitized terminal result with an SDK traceback.
        print(
            '[AudioWorkerCleanup] status=FAILED '
            f'error_code={type(error).__name__}'
        )
        return False
    return True


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
    if len(spans) > MAX_FINAL_SPANS:
        return f'span_stream.spans must contain at most {MAX_FINAL_SPANS} spans'
    seen_indexes = set()
    for pos, span in enumerate(spans):
        if not isinstance(span, dict):
            return f'span_stream.spans[{pos}] must be an object'
        index = span.get('index')
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            return f'span_stream.spans[{pos}].index must be a non-negative integer'
        if index in seen_indexes:
            return f'span_stream.spans[{pos}].index must be unique'
        seen_indexes.add(index)
        audio = span.get('audio')
        audio_error = validate_http_url(
            audio,
            f'span_stream.spans[{pos}].audio',
        )
        if audio_error:
            return audio_error
        start_sec = span.get('start_sec')
        if (
            not is_finite_number(start_sec)
            or start_sec < 0
        ):
            return f'span_stream.spans[{pos}].start_sec must be a non-negative number'
    return None


def validate_draft_span_stream(span_stream):
    '''
    Validate the Holy Grale draft ticker span-stream input shape.
    '''
    next_url = span_stream.get('next_url')
    next_url_error = validate_http_url(next_url, 'span_stream.next_url')
    if next_url_error:
        return next_url_error

    poll_ms = span_stream.get('poll_ms', 500)
    if (
        not is_finite_number(poll_ms)
        or poll_ms < 100
        or poll_ms > 5000
    ):
        return 'span_stream.poll_ms must be a number between 100 and 5000'

    budget_sec = span_stream.get('budget_sec', 480)
    if (
        not is_finite_number(budget_sec)
        or budget_sec <= 0
        or budget_sec > 540
    ):
        return 'span_stream.budget_sec must be a number between 0 and 540'

    idle_timeout_sec = span_stream.get('idle_timeout_sec', 30)
    if (
        not is_finite_number(idle_timeout_sec)
        or idle_timeout_sec <= 0
        or idle_timeout_sec > 120
    ):
        return 'span_stream.idle_timeout_sec must be a number between 0 and 120'

    return None


def validate_draft_warmup_span_stream(span_stream):
    '''
    Validate the Holy Grale draft warmup input shape.
    '''
    model = span_stream.get('model')
    if model is not None and (not isinstance(model, str) or model.strip() == ''):
        return 'span_stream.model must be a non-empty string when provided'
    if isinstance(model, str) and len(model) > 128:
        return 'span_stream.model must be at most 128 characters'
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


def validate_clap_queries(clap_queries):
    '''Validate the bounded CLAP query map handled outside rp_validator.'''
    if not isinstance(clap_queries, dict):
        return 'clap_queries must be an object'
    if not 1 <= len(clap_queries) <= MAX_CLAP_QUERIES:
        return f'clap_queries must contain between 1 and {MAX_CLAP_QUERIES} entries'
    total_description_chars = 0
    for query_name, description in clap_queries.items():
        if not isinstance(query_name, str) or query_name.strip() == '':
            return 'clap_queries keys must be non-empty strings'
        if len(query_name) > MAX_CLAP_QUERY_NAME_CHARS:
            return (
                'clap_queries keys must be at most '
                f'{MAX_CLAP_QUERY_NAME_CHARS} characters'
            )
        if not isinstance(description, str) or description.strip() == '':
            return f'clap_queries[{query_name!r}] must be a non-empty string'
        if len(description) > MAX_CLAP_QUERY_DESCRIPTION_CHARS:
            return (
                f'clap_queries[{query_name!r}] must be at most '
                f'{MAX_CLAP_QUERY_DESCRIPTION_CHARS} characters'
            )
        total_description_chars += len(description)
        if total_description_chars > MAX_CLAP_QUERY_TOTAL_DESCRIPTION_CHARS:
            return (
                'combined clap_queries descriptions must be at most '
                f'{MAX_CLAP_QUERY_TOTAL_DESCRIPTION_CHARS} characters'
            )
    return None


def validate_diarization_hints(job_input):
    '''Keep optional pyannote speaker bounds small and internally consistent.'''
    min_speakers = job_input.get('diarize_min_speakers', 0)
    max_speakers = job_input.get('diarize_max_speakers', 0)
    for field_name, value in (
        ('diarize_min_speakers', min_speakers),
        ('diarize_max_speakers', max_speakers),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or value > MAX_DIARIZATION_SPEAKERS
        ):
            return (
                f'{field_name} must be an integer between 0 and '
                f'{MAX_DIARIZATION_SPEAKERS}'
            )
    if min_speakers and max_speakers and min_speakers > max_speakers:
        return 'diarize_min_speakers must be less than or equal to diarize_max_speakers'
    return None


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
        if isinstance(value, bool):
            return default
        parsed = float(value)
        return parsed if math.isfinite(parsed) else default
    except (OverflowError, TypeError, ValueError):
        return default


def first_non_none(mapping, *field_names):
    '''Return the first explicitly present, non-null alias (preserving 0).'''
    for field_name in field_names:
        if field_name in mapping and mapping[field_name] is not None:
            return mapping[field_name]
    return None


def draft_audio_alias(mapping, field_names, label):
    '''Strictly select one response alias instead of silently ignoring bad data.'''
    present_values = [
        mapping[field_name]
        for field_name in field_names
        if field_name in mapping and mapping[field_name] is not None
    ]
    if not present_values:
        return None
    if len(present_values) > 1 and any(
        value != present_values[0]
        for value in present_values[1:]
    ):
        raise RuntimeError(
            f'DRAFT_POLL_INVALID_RESPONSE: conflicting {label} aliases'
        )
    value = present_values[0]
    if not isinstance(value, str) or value.strip() == '':
        raise RuntimeError(
            f'DRAFT_POLL_INVALID_RESPONSE: {label} must be a non-empty string'
        )
    return value


def validated_draft_geometry(payload):
    raw_start = payload.get('start_sec')
    raw_end = payload.get('end_sec')
    start_sec = parse_float(raw_start, 0.0 if raw_start is None else None)
    end_sec = parse_float(raw_end)
    if start_sec is None or start_sec < 0:
        raise RuntimeError(
            'DRAFT_POLL_INVALID_RESPONSE: start_sec must be a finite '
            'non-negative number'
        )
    if end_sec is not None and end_sec < start_sec:
        raise RuntimeError(
            'DRAFT_POLL_INVALID_RESPONSE: end_sec must be finite and greater '
            'than or equal to start_sec'
        )
    return start_sec, end_sec


def elapsed_ms(start):
    return int(max(0.0, (time.monotonic() - start) * 1000.0))


def draft_poll_failure(next_url, code, stage, retryable, attempts, status=None):
    status_detail = f' status={status}' if status is not None else ''
    return RuntimeError(
        f'{code}: stage={stage} retryable={str(retryable).lower()} '
        f'attempts={attempts}{status_detail} url={safe_url_for_error(next_url)}'
    )


def enforce_draft_deadline(next_url, deadline, attempts, stage):
    if deadline is not None and time.monotonic() >= deadline:
        raise draft_poll_failure(
            next_url,
            'DRAFT_POLL_DEADLINE_EXCEEDED',
            stage,
            True,
            attempts,
        )


def draft_poll_timeout(next_url, deadline, attempts):
    if deadline is None:
        return 15.0
    remaining_sec = deadline - time.monotonic()
    if remaining_sec <= 0:
        raise draft_poll_failure(
            next_url,
            'DRAFT_POLL_DEADLINE_EXCEEDED',
            'poll',
            True,
            attempts,
        )
    return min(15.0, max(0.001, remaining_sec))


def sleep_before_draft_retry(next_url, attempt, deadline):
    delay_sec = min(
        DRAFT_POLL_MAX_BACKOFF_SECONDS,
        DRAFT_POLL_BACKOFF_SECONDS * (2 ** (attempt - 1)),
    )
    if deadline is not None:
        remaining_sec = deadline - time.monotonic()
        if remaining_sec <= 0:
            raise draft_poll_failure(
                next_url,
                'DRAFT_POLL_DEADLINE_EXCEEDED',
                'retry_backoff',
                True,
                attempt,
            )
        delay_sec = min(delay_sec, remaining_sec)
    time.sleep(delay_sec)


def read_bounded_response_body(response, next_url, attempts, deadline=None):
    content_length = response.headers.get('content-length')
    try:
        declared_bytes = int(content_length) if content_length is not None else None
    except (TypeError, ValueError):
        declared_bytes = None
    if declared_bytes is not None and declared_bytes > DRAFT_POLL_MAX_BODY_BYTES:
        raise draft_poll_failure(
            next_url,
            'DRAFT_POLL_RESPONSE_TOO_LARGE',
            'response_body',
            False,
            attempts,
        )
    enforce_draft_deadline(next_url, deadline, attempts, 'response_body')
    body = response.read(DRAFT_POLL_MAX_BODY_BYTES + 1)
    enforce_draft_deadline(next_url, deadline, attempts, 'response_body')
    if len(body or b'') > DRAFT_POLL_MAX_BODY_BYTES:
        raise draft_poll_failure(
            next_url,
            'DRAFT_POLL_RESPONSE_TOO_LARGE',
            'response_body',
            False,
            attempts,
        )
    return body


def validated_draft_response_url(value, field_name):
    if value is None:
        return None
    url_error = validate_http_url(value, field_name)
    if url_error:
        raise RuntimeError(f'DRAFT_POLL_INVALID_RESPONSE: {url_error}')
    return value


def fetch_draft_audio(job_id, next_url, temp_paths, deadline=None):
    '''
    Poll the draft next-audio endpoint.

    Supported responses:
    - 204: no new audio yet
    - application/json: {audio|audio_url|audio_base64, cursor, next_url, start_sec, end_sec, done}
    - audio bytes: body is the next micro-segment; cursor/start/end can be response headers
    '''
    next_url_error = validate_http_url(next_url, 'draft poll URL')
    if next_url_error:
        raise RuntimeError(f'DRAFT_POLL_INVALID_URL: {next_url_error}')
    request_started = time.monotonic()
    timing = {
        'request_ms': 0,
        'body_bytes': 0,
        'audio_download_ms': 0,
    }
    timing['request_attempts'] = 0
    status = None
    headers = None
    body = None
    for attempt in range(1, DRAFT_POLL_MAX_ATTEMPTS + 1):
        timing['request_attempts'] = attempt
        request_timeout = draft_poll_timeout(next_url, deadline, attempt)
        request = urllib.request.Request(
            next_url,
            headers={
                'Accept': 'application/json,audio/aac,audio/wav,*/*',
                'User-Agent': 'web2labs-audio-specialist-draft/1',
            },
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=request_timeout,
            ) as response:
                status = response.getcode()
                headers = response.headers
                if status in (204, 404):
                    timing['request_ms'] = elapsed_ms(request_started)
                    return {'available': False, 'timing': timing}
                if not isinstance(status, int):
                    raise draft_poll_failure(
                        next_url,
                        'DRAFT_POLL_FAILED',
                        'poll',
                        False,
                        attempt,
                    )
                retryable_status = status == 408 or status == 429 or status >= 500
                if retryable_status:
                    if attempt < DRAFT_POLL_MAX_ATTEMPTS:
                        sleep_before_draft_retry(next_url, attempt, deadline)
                        continue
                    raise draft_poll_failure(
                        next_url,
                        'DRAFT_POLL_FAILED',
                        'poll',
                        True,
                        attempt,
                        status,
                    )
                if status >= 400:
                    raise draft_poll_failure(
                        next_url,
                        'DRAFT_POLL_FAILED',
                        'poll',
                        False,
                        attempt,
                        status,
                    )
                body = read_bounded_response_body(
                    response,
                    next_url,
                    attempt,
                    deadline=deadline,
                )
                timing['request_ms'] = elapsed_ms(request_started)
                timing['body_bytes'] = len(body or b'')
                break
        except urllib.error.HTTPError as error:
            timing['request_ms'] = elapsed_ms(request_started)
            if error.code in (204, 404):
                return {'available': False, 'timing': timing}
            retryable = error.code == 408 or error.code == 429 or error.code >= 500
            if retryable and attempt < DRAFT_POLL_MAX_ATTEMPTS:
                sleep_before_draft_retry(next_url, attempt, deadline)
                continue
            raise draft_poll_failure(
                next_url,
                'DRAFT_POLL_FAILED',
                'poll',
                retryable,
                attempt,
                error.code,
            ) from None
        except (urllib.error.URLError, TimeoutError):
            timing['request_ms'] = elapsed_ms(request_started)
            if attempt < DRAFT_POLL_MAX_ATTEMPTS:
                sleep_before_draft_retry(next_url, attempt, deadline)
                continue
            raise draft_poll_failure(
                next_url,
                'DRAFT_POLL_FAILED',
                'poll',
                True,
                attempt,
            ) from None

    content_type = (headers.get('content-type') or '').lower()
    if 'application/json' in content_type:
        try:
            payload = json.loads(body.decode('utf-8') or '{}')
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f'DRAFT_POLL_INVALID_RESPONSE: {type(error).__name__}'
            ) from None
        if not isinstance(payload, dict):
            raise RuntimeError('DRAFT_POLL_INVALID_RESPONSE: JSON body must be an object')
        response_next_url = validated_draft_response_url(
            payload.get('next_url'),
            'draft response next_url',
        )
        cursor_value = first_non_none(payload, 'cursor', 'next_cursor')
        if payload.get('done'):
            return {
                'available': False,
                'done': True,
                'cursor': cursor_value,
                'next_url': response_next_url,
                'timing': timing,
            }

        audio_input = None
        audio_url = draft_audio_alias(
            payload,
            ('audio', 'audio_url'),
            'audio URL',
        )
        audio_base64 = draft_audio_alias(
            payload,
            ('audio_base64', 'audio_b64'),
            'audio_base64',
        )
        if audio_url is not None and audio_base64 is not None:
            raise RuntimeError(
                'DRAFT_POLL_INVALID_RESPONSE: provide either audio URL or '
                'audio_base64, not both'
            )
        if audio_url is not None:
            audio_url_error = validate_http_url(audio_url, 'draft response audio URL')
            if audio_url_error:
                raise MediaFetchFailure(audio_url, 'draft_audio_download')
            download_started = time.monotonic()
            audio_input = download_audio_url(
                audio_url,
                'draft_audio_download',
                deadline=deadline,
            )
            timing['audio_download_ms'] = elapsed_ms(download_started)
            temp_paths.append(audio_input)
        elif audio_base64 is not None:
            try:
                audio_input = base64_to_tempfile(audio_base64)
            except ValueError as error:
                raise RuntimeError(
                    f'DRAFT_POLL_INVALID_RESPONSE: {error}'
                ) from None
            temp_paths.append(audio_input)

        if not audio_input:
            return {
                'available': False,
                'cursor': cursor_value,
                'next_url': response_next_url,
                'timing': timing,
            }

        start_sec, end_sec = validated_draft_geometry(payload)

        return {
            'available': True,
            'audio_input': audio_input,
            'cursor': cursor_value,
            'next_url': response_next_url,
            'start_sec': start_sec,
            'end_sec': end_sec,
            'timing': timing,
        }

    if not body:
        return {'available': False, 'timing': timing}

    suffix = '.wav' if 'wav' in content_type else '.aac'
    audio_input = bytes_to_tempfile(body, suffix=suffix)
    temp_paths.append(audio_input)
    response_next_url = validated_draft_response_url(
        headers.get('x-next-url'),
        'draft response x-next-url',
    )
    start_sec, end_sec = validated_draft_geometry({
        'start_sec': headers.get('x-start-sec'),
        'end_sec': headers.get('x-end-sec'),
    })
    return {
        'available': True,
        'audio_input': audio_input,
        'cursor': first_non_none(headers, 'x-next-cursor', 'x-cursor'),
        'next_url': response_next_url,
        'start_sec': start_sec,
        'end_sec': end_sec,
        'timing': timing,
    }


def run_final_span_stream_job(job, job_input, span_stream):
    '''
    Run final-tier transcription for multiple ready spans and yield each result.

    The wrapper handler returns this generator only for span-stream jobs; classic
    single-audio jobs still return a plain dict from run_whisper_job.
    '''
    failed_span_indexes = set()
    failure_codes = set()
    try:
        for span_pos, span in enumerate(span_stream['spans']):
            span_index = int(span['index'])
            audio_url = span['audio']
            start_sec = float(span['start_sec'])
            audio_input = None
            span_result = None
            span_error = None
            span_stage = 'span_download'
            try:
                with rp_debugger.LineTimer(f'span_{span_pos}_download_step'):
                    audio_input = download_audio_url(audio_url, span_stage)

                span_stage = 'prediction'
                with rp_debugger.LineTimer(f'span_{span_pos}_prediction_step'):
                    whisper_results = MODEL.predict(
                        audio=audio_input,
                        model_name=job_input["model"],
                        asr_backend=job_input["asr_backend"],
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
                        # Final spans are the authoritative incremental transcript
                        # tier. Keep draft ticker latency unchanged, but allow the
                        # same opt-in speaker sidecar as classic jobs here.
                        diarize=job_input.get("diarize", False),
                        diarize_min_speakers=job_input.get("diarize_min_speakers") or None,
                        diarize_max_speakers=job_input.get("diarize_max_speakers") or None,
                    )
                speaker_diarization = whisper_results.get('speaker_diarization')
                if isinstance(speaker_diarization, dict):
                    # Turns remain chunk-local. Stamp their coordinate system
                    # so the consumer can translate them exactly once.
                    speaker_diarization = dict(speaker_diarization)
                    speaker_diarization.update({
                        'timebase': 'SPAN_RELATIVE_SECONDS',
                        'span_index': span_index,
                        'span_start_sec': start_sec,
                    })
                    whisper_results['speaker_diarization'] = speaker_diarization
                whisper_results["span_index"] = span_index
                whisper_results["start_sec"] = start_sec
                span_result = to_jsonable(whisper_results)
            except Exception as error:
                if isinstance(error, MediaFetchFailure):
                    failure = error.payload
                else:
                    failure = {
                        'error': (
                            'SPAN_PROCESSING_FAILED: '
                            f'{type(error).__name__}'
                        ),
                        'code': 'SPAN_PROCESSING_FAILED',
                        'stage': span_stage,
                        'retryable': span_stage == 'span_download',
                    }
                span_error = {
                    'mode': 'final',
                    'event': 'span_error',
                    'failed_span_index': span_index,
                    'start_sec': start_sec,
                    # RunPod 1.8.2 treats any streamed output containing a
                    # truthy top-level `error` as terminal and stops consuming
                    # this generator. A recoverable per-span event must use
                    # `message` so later spans are still processed.
                    'message': failure['error'],
                    'code': failure['code'],
                    'stage': failure['stage'],
                    'retryable': failure['retryable'],
                }
            finally:
                # Do not keep all spans on disk until the job-level cleanup.
                # The exact downloaded file is gone before either result yields.
                unlink_exact_file(audio_input)

            if span_error is not None:
                failed_span_indexes.add(span_index)
                failure_codes.add(span_error['code'])
                yield span_error
                continue
            yield span_result
    finally:
        with rp_debugger.LineTimer('span_stream_cleanup_step'):
            cleanup_job_artifacts(job.get('id'))

    if failed_span_indexes:
        indexes = ','.join(str(index) for index in sorted(failed_span_indexes))
        codes = ','.join(sorted(failure_codes))
        yield {
            'error': (
                'SPAN_STREAM_PARTIAL_FAILURE: '
                f'failed_span_indexes={indexes}; codes={codes}'
            ),
        }


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
                draft_audio = fetch_draft_audio(
                    job.get('id'),
                    poll_url,
                    temp_paths,
                    deadline=budget_deadline,
                )

            if draft_audio.get('next_url'):
                next_url = draft_audio['next_url']
            if draft_audio.get('cursor') is not None:
                cursor = draft_audio['cursor']

            if draft_audio.get('done'):
                yield closed('eof')
                return

            if not draft_audio.get('available'):
                now = time.monotonic()
                if now >= idle_deadline:
                    yield closed('idle_timeout')
                    return
                if now >= budget_deadline:
                    yield closed('budget_exhausted')
                    return
                time.sleep(min(
                    poll_ms / 1000.0,
                    idle_deadline - now,
                    budget_deadline - now,
                ))
                continue

            idle_deadline = time.monotonic() + idle_timeout_sec
            start_sec = float(draft_audio.get('start_sec') or 0.0)
            end_sec = draft_audio.get('end_sec')
            audio_input = draft_audio['audio_input']
            try:
                model_wait_started = time.monotonic()
                if warmup_thread.is_alive():
                    warmup_thread.join(timeout=max(
                        0.0,
                        budget_deadline - time.monotonic(),
                    ))
                model_warmup_wait_ms = elapsed_ms(model_wait_started)
                if warmup_thread.is_alive():
                    raise RuntimeError(
                        'DRAFT_POLL_DEADLINE_EXCEEDED: turbo warmup exceeded '
                        'the draft budget'
                    )
                if model_warmup_error:
                    raise RuntimeError(f"draft turbo warmup failed: {model_warmup_error}")

                if time.monotonic() >= budget_deadline:
                    raise RuntimeError(
                        'DRAFT_POLL_DEADLINE_EXCEEDED: no budget remains for '
                        'draft inference'
                    )

                prediction_started = time.monotonic()
                with rp_debugger.LineTimer(f'draft_prediction_step_{yield_index}'):
                    whisper_results = MODEL.predict(
                        audio=audio_input,
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
            finally:
                # A long-running draft ticker must not accumulate every prior
                # micro-segment. Retry at whole-job cleanup if this unlink fails.
                if unlink_exact_file(audio_input) and audio_input in temp_paths:
                    temp_paths.remove(audio_input)

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
    # The top-level schema defaults classic transcription jobs to `base`.
    # Draft warmup has its own contract and must default independently.
    model_name = span_stream.get('model') or 'turbo'
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
    # RunPod may reuse the original job object for diagnostics/retries. Special
    # fields are popped only from this private copy. Malformed envelopes still
    # receive a keepalive and a stable validation error instead of raising
    # before the generator's first yield.
    raw_job_input = job.get('input') if isinstance(job, dict) else None
    if not isinstance(raw_job_input, dict):
        yield {
            'mode': 'classic',
            'event': 'started',
            'yield_index': -1,
        }
        yield {'error': 'input must be an object'}
        return
    job_input = dict(raw_job_input)

    # Extract clap_queries before validation — rp_validator chokes on dict types
    raw_clap_queries = job_input.pop('clap_queries', _MISSING)
    raw_span_stream = job_input.pop('span_stream', _MISSING)
    clap_queries_present = raw_clap_queries is not _MISSING
    span_stream_present = raw_span_stream is not _MISSING
    clap_queries_error = (
        validate_clap_queries(raw_clap_queries)
        if clap_queries_present
        else None
    )
    # Validate before the keepalive chooses a mode; malformed list/null/string
    # values must never reach an unchecked `.get()`.
    span_stream_error = (
        validate_span_stream(raw_span_stream)
        if span_stream_present
        else None
    )
    raw_sat_punctuation_probe = job_input.pop(
        'sat_punctuation_probe',
        None,
    )
    raw_sat_punctuation_batch_probe = job_input.pop(
        'sat_punctuation_batch_probe',
        None,
    )

    # RunPod serverless streaming jobs can be retried as timed out if a cold
    # worker spends too long loading models before the first stream item.
    # Emit a cheap control item immediately; Studio ignores unknown events.
    if raw_sat_punctuation_batch_probe is not None:
        yield {
            'mode': 'sat_punctuation_batch_probe',
            'event': 'started',
            'yield_index': -1,
        }
    elif raw_sat_punctuation_probe is not None:
        yield {
            'mode': 'sat_punctuation_probe',
            'event': 'started',
            'yield_index': -1,
        }
    elif span_stream_present:
        yield {
            'mode': (
                raw_span_stream.get('mode')
                if isinstance(raw_span_stream, dict)
                else 'span_stream'
            ),
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

    if clap_queries_error:
        yield {'error': clap_queries_error}
        return
    if span_stream_error:
        yield {'error': span_stream_error}
        return

    # Restore the already-validated special field after rp_validator.
    if clap_queries_present:
        job_input['clap_queries'] = raw_clap_queries

    diarization_error = validate_diarization_hints(job_input)
    if diarization_error:
        yield {'error': diarization_error}
        return

    if (
        span_stream_present
        and (
            job_input.get('audio') is not None
            or job_input.get('audio_base64') is not None
        )
    ):
        yield {
            'error': 'span_stream is mutually exclusive with audio and audio_base64'
        }
        return

    if job_input["asr_backend"] not in predict.AVAILABLE_ASR_BACKENDS:
        yield {
            'error': (
                f"Invalid ASR backend: {job_input['asr_backend']}. "
                f"Available backends are: {sorted(predict.AVAILABLE_ASR_BACKENDS)}"
            )
        }
        return

    if (
        raw_sat_punctuation_probe is not None
        and raw_sat_punctuation_batch_probe is not None
    ):
        yield {
            'error': (
                "sat_punctuation_probe and "
                "sat_punctuation_batch_probe are mutually exclusive"
            )
        }
        return

    if (
        raw_sat_punctuation_probe is not None
        or raw_sat_punctuation_batch_probe is not None
    ):
        if span_stream_present or clap_queries_present:
            yield {
                'error': (
                    "SaT punctuation probes cannot be combined with "
                    "span_stream or clap_queries"
                )
            }
            return
        if job_input.get('audio') or job_input.get('audio_base64'):
            yield {
                'error': (
                    "SaT punctuation probes accept source tokens, not audio"
                )
            }
            return
        try:
            if raw_sat_punctuation_batch_probe is not None:
                result = MODEL.predict_punctuation_batch(
                    raw_sat_punctuation_batch_probe
                )
            else:
                result = MODEL.predict_punctuation_window(
                    raw_sat_punctuation_probe
                )
        except Exception as error:
            yield {'error': str(error)}
            return
        yield to_jsonable(result)
        return

    if span_stream_present:
        if (
            job_input["asr_backend"] != "whisper"
            and raw_span_stream.get("mode") != "final"
        ):
            yield {
                'error': (
                    "Experimental Parakeet backend supports classic and final "
                    "span jobs only; draft paths remain Whisper"
                )
            }
            return
        yield from run_span_stream_job(job, job_input, raw_span_stream)
        return

    if not job_input.get('audio', False) and not job_input.get('audio_base64', False):
        yield {'error': 'Must provide either audio or audio_base64'}
        return

    if job_input.get('audio', False) and job_input.get('audio_base64', False):
        yield {'error': 'Must provide either audio or audio_base64, not both'}
        return

    if job_input.get('audio', False):
        audio_url_error = validate_http_url(job_input['audio'], 'audio')
        if audio_url_error:
            yield {
                'error': audio_url_error,
                'code': 'INVALID_AUDIO_URL',
                'stage': 'input_validation',
                'retryable': False,
            }
            return

    audio_temp_path = None
    try:
        if job_input.get('audio', False):
            try:
                with rp_debugger.LineTimer('download_step'):
                    audio_temp_path = download_audio_url(
                        job_input['audio'],
                        'classic_download',
                    )
                    audio_input = audio_temp_path
            except MediaFetchFailure as error:
                yield error.payload
                return

        if job_input.get('audio_base64', False):
            try:
                audio_temp_path = base64_to_tempfile(job_input['audio_base64'])
            except ValueError as error:
                yield {
                    'error': str(error),
                    'code': 'INVALID_AUDIO_BASE64',
                    'stage': 'input_validation',
                    'retryable': False,
                }
                return
            audio_input = audio_temp_path

        with rp_debugger.LineTimer('prediction_step'):
            whisper_results = MODEL.predict(
                audio=audio_input,
                model_name=job_input["model"],
                asr_backend=job_input["asr_backend"],
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
            cleanup_job_artifacts(job.get('id'), audio_temp_path)

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
