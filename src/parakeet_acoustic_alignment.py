"""Optional acoustic enrichment of direct Parakeet recognition, never supplied text."""

import copy
import math

from aligner import ALIGNMENT_MODEL_ID, ALIGNMENT_SCHEMA_VERSION


class ParakeetAcousticAlignment:
    policy_version = "w2l-parakeet-acoustic-alignment-v1"

    @classmethod
    def apply(cls, audio_path, recognized, aligner, device):
        result = copy.deepcopy(recognized)
        original = result.get("word_timestamps", [])
        recognition = result.get("asr_backend_evidence", {})
        evidence = {
            "schema_version": ALIGNMENT_SCHEMA_VERSION,
            "policy_version": cls.policy_version,
            "status": "UNVERIFIED",
            "model_id": ALIGNMENT_MODEL_ID,
            "detected_language": recognition.get("model_detected_language"),
            "language_hint": recognition.get("language_hint"),
            "language_authority": recognition.get("language_authority"),
            "supported_languages": ["en"],
            "total_words": len(original),
            "aligned_words": 0,
            "fallback_words": len(original),
            "aligned_word_fraction": 0.0,
            "per_word_authority": False,
            "natural_landing_authority": False,
            "transcript_geometry_mutated": False,
        }
        result["alignment"] = evidence
        result["word_timestamps_aligned"] = False
        if (result.get("asr_backend") != "parakeet"
                or recognition.get("language_hint") != "en"
                or recognition.get("model_detected_language") not in (None, "en")
                or recognition.get("language_status") == "MODEL_HINT_MISMATCH"):
            evidence["status"] = "UNSUPPORTED_LANGUAGE"
            return result
        if not original:
            evidence["status"] = "NO_WORDS"
            return result
        try:
            aligner.setup(device=device)
            aligned = aligner.align(str(audio_path), copy.deepcopy(original), language_code="en")
            if len(aligned) != len(original) or any(
                word.get("word") != native.get("word")
                for word, native in zip(aligned, original)
            ):
                raise ValueError("Acoustic alignment changed recognition word identity")
            duration = recognition.get("audio_duration_seconds")
            words = [cls._word(word, native, duration)
                     for word, native in zip(aligned, original)]
            if any(left["end"] > right["start"] for left, right in zip(words, words[1:])):
                raise ValueError("Acoustic alignment returned overlapping word geometry")
        except Exception as error:
            # Recognition is settled work. Keep it and disclose failed enrichment;
            # never retry recognition or grant native timing acoustic authority.
            evidence["status"] = "FAILED"
            evidence["failure_type"] = type(error).__name__
            return result
        supported = sum(word.get("alignment_authority") is True for word in words)
        evidence.update({
            "status": "ALIGNED_SUPPORTED" if supported == len(words) else "PARTIAL",
            "aligned_words": supported,
            "fallback_words": len(words) - supported,
            "aligned_word_fraction": supported / len(words),
            "per_word_authority": supported > 0,
        })
        result["word_timestamps"] = words
        result["word_timestamps_aligned"] = supported > 0
        return result

    @staticmethod
    def _word(word, native, duration):
        result = copy.deepcopy(native)
        if word.get("alignment_authority") is not True or word.get("alignment_status") != "ALIGNED_SUPPORTED":
            result.update({"alignment_authority": False,
                           "alignment_status": "FALLBACK_UNALIGNED",
                           "alignment_reason": word.get("alignment_reason", "NO_ACOUSTIC_SUPPORT")})
            return result
        geometry = [word.get(key) for key in ("onset_start", "start", "end", "offset_end")]
        if (not all(isinstance(value, (float, int)) and not isinstance(value, bool)
                    and math.isfinite(value) for value in [*geometry, duration])
                or not 0 <= geometry[0] <= geometry[1] < geometry[2] <= geometry[3] <= duration):
            raise ValueError("Acoustic alignment returned invalid word geometry")
        result.update(word)
        result["probability"] = native.get("probability")
        result["native_timing"] = {
            key: native.get(key) for key in
            ("start", "end", "timestamp_source", "timestamp_authority")
        }
        result["timestamp_source"] = "WAV2VEC2_CTC"
        result["timestamp_authority"] = "NP_SBV2_ACOUSTIC"
        return result
