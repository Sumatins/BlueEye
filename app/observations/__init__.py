"""Persistent observation records for BlueEye.

This package turns a finished detection run into a durable, searchable
*observation record* that survives application reruns and restarts. It never
runs inference itself: it stores what the existing pipeline already produced
(annotated media, the JSON report, per-object detections and confidences).

Design
------
* :mod:`app.observations.store` - SQLite persistence, always scoped to the
  authenticated account id (ownership is enforced in every query).
* :mod:`app.observations.builder` - build an :class:`Analysis` from a real
  detection report, without fabricating any field the pipeline did not give us.
* :mod:`app.observations.export` - structured CSV / JSON / printable HTML
  reports for a single record or a whole collection.
* :mod:`app.observations.compare` - side-by-side comparison of saved records.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

from app.observations import builder, compare, export, store
from app.observations.models import (
    REVIEW_STATUSES,
    Analysis,
    Collection,
    Correction,
    DetectionRecord,
    ObservationError,
)

__all__ = [
    "REVIEW_STATUSES",
    "Analysis",
    "Collection",
    "Correction",
    "DetectionRecord",
    "ObservationError",
    "builder",
    "compare",
    "export",
    "store",
]
