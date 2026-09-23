"""Acoustically align caller-supplied, independently recognized text.

The caller supplies ordered speech segments with source-relative bounds. Those
bounds are routing hints only. Wav2vec2 creates every accepted word boundary,
and per-word acoustic authority remains explicit in the result.
"""

import hashlib
import math
import re

from aligner import ALIGNMENT_MODEL_ID, ALIGNMENT_SCHEMA_VERSION


SUPPLIED_TEXT_ALIGNMENT_POLICY_VERSION = "w2l-supplied-text-alignment-v1"
MAX_DURATION_SEC = 120.25
MAX_SEGMENTS = 256
MAX_WORDS = 800
MAX_TEXT_CHARS = 32_000
MINIMUM_WORD_MEAN_SCORE = -4.0
MINIMUM_COMMAND_MEAN_SCORE = -2.5
MAXIMUM_WORD_DURATION_SEC = 1.25
SEGMENT_CONTAINMENT_TOLERANCE_SEC = 0.75
COMMAND_TOKENS = frozenset({
    "again", "cancel", "cut", "delete", "redo", "remove", "restart",
    "retake", "start", "stop", "undo",
})


class SuppliedTextAligner:
    """Build provisional routing geometry, then replace it with CTC evidence."""

    def __init__(self, aligner):
        self.aligner = aligner

    def align(self, audio_path, segments, language_code="en"):
        normalized_segments = self.validate_segments(segments)
        words = self._provisional_words(normalized_segments)
        aligned = self.aligner.align(
            audio_path,
            words,
            language_code=language_code,
        )
        if len(aligned) != len(words):
            raise ValueError("Supplied-text alignment changed the word count")
        for ordinal, (expected, actual) in enumerate(zip(words, aligned)):
            if (
                actual.get("word") != expected["word"]
                or actual.get("supplied_word_ordinal") != ordinal
                or actual.get("supplied_segment_index")
                != expected["supplied_segment_index"]
            ):
                raise ValueError(
                    f"Supplied-text alignment changed word identity at {ordinal}"
                )

        aligned_words = sum(
            word.get("alignment_status") == "ALIGNED_SUPPORTED"
            and word.get("alignment_authority") is True
            for word in aligned
        )
        fallback_words = len(aligned) - aligned_words
        admission = self._admission(aligned, normalized_segments)
        supplied_text_sha256 = hashlib.sha256(
            " ".join(word["word"] for word in words).encode("utf-8")
        ).hexdigest()
        return {
            "segments": normalized_segments,
            "detected_language": language_code,
            "transcription": " ".join(word["word"] for word in aligned),
            "translation": None,
            "model": ALIGNMENT_MODEL_ID,
            "word_timestamps": aligned,
            "word_timestamps_aligned": aligned_words > 0,
            "alignment": {
                "schema_version": ALIGNMENT_SCHEMA_VERSION,
                "status": (
                    "ALIGNED_SUPPORTED" if fallback_words == 0 else "PARTIAL"
                ),
                "model_id": ALIGNMENT_MODEL_ID,
                "detected_language": language_code,
                "supported_languages": ["en"],
                "total_words": len(aligned),
                "aligned_words": aligned_words,
                "fallback_words": fallback_words,
                "aligned_word_fraction": (
                    aligned_words / len(aligned) if aligned else 0.0
                ),
                "per_word_authority": aligned_words > 0,
                "transcript_geometry_mutated": False,
                "source_text_kind": "CALLER_SUPPLIED_SEGMENTS",
                "supplied_text_policy_version": (
                    SUPPLIED_TEXT_ALIGNMENT_POLICY_VERSION
                ),
                "supplied_text_sha256": supplied_text_sha256,
                "supplied_text_admission": admission,
            },
        }

    @classmethod
    def _admission(cls, words, segments):
        violations = []
        for ordinal, word in enumerate(words):
            segment_index = word.get("supplied_segment_index")
            segment = (
                segments[segment_index]
                if isinstance(segment_index, int)
                and not isinstance(segment_index, bool)
                and 0 <= segment_index < len(segments)
                else None
            )
            start = word.get("start")
            end = word.get("end")
            mean_score = word.get("alignment_score_mean")
            token = re.sub(r"[^a-z0-9]", "", str(word.get("word", "")).lower())
            reasons = []
            if (
                word.get("alignment_authority") is not True
                or word.get("alignment_status") != "ALIGNED_SUPPORTED"
            ):
                reasons.append("MISSING_ACOUSTIC_AUTHORITY")
            if (
                isinstance(start, bool)
                or isinstance(end, bool)
                or not isinstance(start, (int, float))
                or not isinstance(end, (int, float))
                or not math.isfinite(start)
                or not math.isfinite(end)
                or end <= start
            ):
                reasons.append("INVALID_WORD_GEOMETRY")
            elif end - start > MAXIMUM_WORD_DURATION_SEC:
                reasons.append("IMPLAUSIBLE_WORD_DURATION")
            if segment is None:
                reasons.append("INVALID_SEGMENT_BINDING")
            elif (
                isinstance(start, (int, float))
                and isinstance(end, (int, float))
                and (
                    start < segment["start"] - SEGMENT_CONTAINMENT_TOLERANCE_SEC
                    or end > segment["end"] + SEGMENT_CONTAINMENT_TOLERANCE_SEC
                )
            ):
                reasons.append("OUTSIDE_SEGMENT_WINDOW")
            if (
                isinstance(mean_score, bool)
                or not isinstance(mean_score, (int, float))
                or not math.isfinite(mean_score)
                or mean_score < MINIMUM_WORD_MEAN_SCORE
            ):
                reasons.append("LOW_ACOUSTIC_SCORE")
            if (
                token in COMMAND_TOKENS
                and isinstance(mean_score, (int, float))
                and not isinstance(mean_score, bool)
                and mean_score < MINIMUM_COMMAND_MEAN_SCORE
            ):
                reasons.append("LOW_COMMAND_ACOUSTIC_SCORE")
            if reasons:
                violations.append({
                    "word_ordinal": ordinal,
                    "word": word.get("word"),
                    "reason_codes": reasons,
                    "start": start,
                    "end": end,
                    "alignment_score_mean": mean_score,
                })
        return {
            "policy_version": SUPPLIED_TEXT_ALIGNMENT_POLICY_VERSION,
            "status": "ACCEPTED" if not violations else "REJECTED",
            "minimum_word_mean_score": MINIMUM_WORD_MEAN_SCORE,
            "minimum_command_mean_score": MINIMUM_COMMAND_MEAN_SCORE,
            "maximum_word_duration_sec": MAXIMUM_WORD_DURATION_SEC,
            "segment_containment_tolerance_sec": (
                SEGMENT_CONTAINMENT_TOLERANCE_SEC
            ),
            "violations": violations,
        }

    @classmethod
    def validate_segments(cls, segments):
        if not isinstance(segments, list) or not segments:
            raise ValueError("alignment_segments must be a non-empty array")
        if len(segments) > MAX_SEGMENTS:
            raise ValueError("alignment_segments exceeds the segment limit")
        normalized = []
        previous_end = 0.0
        text_chars = 0
        word_count = 0
        for index, segment in enumerate(segments):
            if not isinstance(segment, dict) or set(segment) != {
                "start",
                "end",
                "text",
            }:
                raise ValueError(
                    f"alignment_segments[{index}] must contain start, end, text"
                )
            start = segment["start"]
            end = segment["end"]
            text = segment["text"]
            if (
                isinstance(start, bool)
                or isinstance(end, bool)
                or not isinstance(start, (int, float))
                or not isinstance(end, (int, float))
                or not math.isfinite(start)
                or not math.isfinite(end)
                or start < previous_end
                or end <= start
                or end > MAX_DURATION_SEC
            ):
                raise ValueError(
                    f"alignment_segments[{index}] has invalid ordered geometry"
                )
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"alignment_segments[{index}] has empty text"
                )
            clean_text = " ".join(text.split())
            tokens = re.findall(r"\S+", clean_text)
            text_chars += len(clean_text)
            word_count += len(tokens)
            normalized.append(
                {"start": float(start), "end": float(end), "text": clean_text}
            )
            previous_end = float(end)
        if text_chars > MAX_TEXT_CHARS:
            raise ValueError("alignment_segments exceeds the text limit")
        if word_count == 0 or word_count > MAX_WORDS:
            raise ValueError("alignment_segments has an invalid word count")
        return normalized

    @classmethod
    def _provisional_words(cls, segments):
        words = []
        for segment_index, segment in enumerate(segments):
            tokens = re.findall(r"\S+", segment["text"])
            weights = [max(1, len(re.sub(r"[^A-Za-z']", "", token))) for token in tokens]
            total_weight = sum(weights)
            duration = segment["end"] - segment["start"]
            cursor = 0
            for token, weight in zip(tokens, weights):
                start = segment["start"] + duration * cursor / total_weight
                cursor += weight
                end = segment["start"] + duration * cursor / total_weight
                words.append(
                    {
                        "word": token,
                        "start": start,
                        "end": end,
                        "supplied_word_ordinal": len(words),
                        "supplied_segment_index": segment_index,
                    }
                )
        return words
