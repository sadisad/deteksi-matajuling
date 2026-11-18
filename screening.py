"""Decision logic for the two-stage eye-screening prototype.

Kept separate from the UI and from Ultralytics so it can be tested without
model weights, a camera, or Streamlit.

The important rule here: a model that produces no boxes has not said
"normal". It has said nothing. That case is reported as INCONCLUSIVE with
no confidence value, never as NORMAL with a placeholder score.
"""

from dataclasses import dataclass, field
from typing import Iterable, Sequence

NORMAL = "NORMAL"
STRABISMUS = "STRABISMUS (JULING)"
INCONCLUSIVE = "TIDAK DAPAT DISIMPULKAN"


@dataclass(frozen=True)
class Detection:
    """One classified eye region from stage 2."""

    label: str
    confidence: float
    box: Sequence[int] = field(default_factory=tuple)

    def is_strabismus(self) -> bool:
        label = self.label.upper()
        return "JULING" in label or "STRABISMUS" in label


@dataclass(frozen=True)
class Outcome:
    """What the pipeline is allowed to claim, and why."""

    status: str
    confidence: float | None
    reason: str = ""
    detections: tuple = ()

    @property
    def is_conclusive(self) -> bool:
        return self.confidence is not None


def decide(detections: Iterable[Detection], eye_regions_found: bool = True) -> Outcome:
    """Map stage-2 detections onto a status.

    ``eye_regions_found`` is False when stage 1 fell back to the whole frame,
    which means stage 2 classified something other than eyes.
    """
    items = tuple(detections)

    if not eye_regions_found:
        return Outcome(INCONCLUSIVE, None, "area mata tidak terdeteksi", ())

    if not items:
        return Outcome(INCONCLUSIVE, None, "model klasifikasi tidak mengeluarkan kotak", ())

    strabismus = [d for d in items if d.is_strabismus()]
    if strabismus:
        return Outcome(STRABISMUS, max(d.confidence for d in strabismus), "", items)

    return Outcome(NORMAL, max(d.confidence for d in items), "", items)


def format_result(outcome: Outcome) -> str:
    """Human-readable line. Never prints a percentage for an abstention."""
    if outcome.confidence is None:
        return outcome.status
    return f"{outcome.status} ({outcome.confidence:.2%})"
