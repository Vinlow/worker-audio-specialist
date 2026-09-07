"""Choose source-ordered word geometry from overlapping acoustic windows.

Every candidate is an unmodified CTC measurement. A window switch must preserve
lexical order and the order of both acoustic envelopes. No clamping, averaging,
token deletion, or synthetic timing is permitted to manufacture a join.
"""

from dataclasses import dataclass
import math
from typing import Optional


@dataclass(frozen=True)
class AlignmentWindowCandidate:
    word: dict
    window_index: int
    window_start: float
    window_end: float

    @property
    def context(self) -> float:
        return min(
            self.word["start"] - self.window_start,
            self.window_end - self.word["end"],
        )


class AlignmentWindowStitcher:
    """Maximum-context path through actual per-word alignment candidates."""

    @classmethod
    def stitch(
        cls,
        candidates: list[list[AlignmentWindowCandidate]],
        fallbacks: list[Optional[dict]],
    ) -> list[dict]:
        if len(candidates) != len(fallbacks):
            raise ValueError("Alignment candidate/fallback count mismatch")
        rows: list[list[tuple[float, Optional[int]]]] = []
        ordinals: list[int] = []
        previous: list[AlignmentWindowCandidate] = []
        for ordinal, choices in enumerate(candidates):
            if not choices:
                if fallbacks[ordinal] is None:
                    raise ValueError(f"No complete acoustic candidate for word {ordinal}")
                fallback = fallbacks[ordinal]
                if (fallback.get("alignment_authority") is not False
                        or fallback.get("alignment_status") != "FALLBACK_UNALIGNED"):
                    raise ValueError(f"Fallback lacks explicit non-authority at word {ordinal}")
                continue
            if fallbacks[ordinal] is not None:
                raise ValueError(f"Ambiguous acoustic/fallback authority for word {ordinal}")
            cls._validate_choices(choices, ordinal)
            row = cls._advance(choices, previous, rows[-1] if rows else [])
            if not any(math.isfinite(score) for score, _ in row):
                raise ValueError(f"No coherent acoustic window join at word {ordinal}")
            rows.append(row)
            ordinals.append(ordinal)
            previous = choices

        output = [dict(word) if word is not None else None for word in fallbacks]
        if rows:
            choice = max(range(len(rows[-1])), key=lambda index: rows[-1][index][0])
            for row_index in range(len(rows) - 1, -1, -1):
                ordinal = ordinals[row_index]
                output[ordinal] = dict(candidates[ordinal][choice].word)
                predecessor = rows[row_index][choice][1]
                if row_index > 0:
                    if predecessor is None:
                        raise ValueError("Acoustic candidate path lost its predecessor")
                    choice = predecessor
        if any(word is None for word in output):
            raise ValueError("Acoustic candidate path did not cover every word")
        return [word for word in output if word is not None]

    @classmethod
    def _advance(
        cls,
        choices: list[AlignmentWindowCandidate],
        previous: list[AlignmentWindowCandidate],
        previous_scores: list[tuple[float, Optional[int]]],
    ) -> list[tuple[float, Optional[int]]]:
        row: list[tuple[float, Optional[int]]] = []
        for candidate in choices:
            if not previous:
                row.append((candidate.context, None))
                continue
            compatible = [
                index for index, earlier in enumerate(previous)
                if math.isfinite(previous_scores[index][0])
                and cls._compatible(earlier, candidate)
            ]
            if not compatible:
                row.append((-math.inf, None))
                continue
            predecessor = max(compatible, key=lambda index: previous_scores[index][0])
            row.append((previous_scores[predecessor][0] + candidate.context, predecessor))
        return row

    @staticmethod
    def _compatible(previous: AlignmentWindowCandidate, current: AlignmentWindowCandidate) -> bool:
        left, right = previous.word, current.word
        return (
            current.window_index >= previous.window_index
            and right["start"] >= left["end"]
            and right["onset_start"] >= left["onset_start"]
            and right["offset_end"] >= left["offset_end"]
        )

    @staticmethod
    def _validate_choices(choices: list[AlignmentWindowCandidate], ordinal: int) -> None:
        text = choices[0].word.get("word")
        for candidate in choices:
            word = candidate.word
            values = [word.get(key) for key in ("onset_start", "start", "end", "offset_end")]
            if (
                not isinstance(text, str) or not text.strip()
                or word.get("word") != text
                or word.get("alignment_authority") is not True
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not math.isfinite(value) for value in values)
                or not (candidate.window_start <= values[0] <= values[1] < values[2] <= values[3] <= candidate.window_end)
            ):
                raise ValueError(f"Invalid acoustic candidate at word {ordinal}")
