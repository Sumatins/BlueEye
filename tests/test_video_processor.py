"""Unit tests for the video pipeline (model mocked)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import pytest

from app.processing.video_processor import VideoProcessor, probe_video
from app.utils.validation import ValidationError
from tests.conftest import MockDetector


def _open_frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened()
    count = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        count += 1
    cap.release()
    return count


# --------------------------------------------------------------------------- #
# Probing
# --------------------------------------------------------------------------- #
def test_probe_video_reads_metadata(sample_video: Path) -> None:
    info = probe_video(sample_video)
    assert info.width == 160
    assert info.height == 120
    assert info.fps == pytest.approx(10.0, abs=0.5)
    assert info.frame_count == 8
    assert info.duration_seconds == pytest.approx(0.8, abs=0.1)
    assert info.file_size_bytes > 0
    assert info.to_dict()["filename"] == sample_video.name


def test_probe_video_rejects_corrupt_file(tmp_path: Path) -> None:
    fake = tmp_path / "broken.mp4"
    fake.write_bytes(b"garbage")
    with pytest.raises(ValidationError):
        probe_video(fake)


# --------------------------------------------------------------------------- #
# Processing
# --------------------------------------------------------------------------- #
def test_process_creates_annotated_video_with_same_frame_count(
    settings, sample_video: Path
) -> None:
    detector = MockDetector()
    processor = VideoProcessor(detector, settings=settings)
    result = processor.process(sample_video)

    assert result.output_path.exists()
    assert result.output_path.suffix == ".mp4"
    assert result.output_path.parent == settings.videos_output_dir
    assert result.output_path.stat().st_size > 0

    # Dimensions and frame count are preserved.
    assert result.info.width == 160 and result.info.height == 120
    assert _open_frame_count(result.output_path) == 8
    assert result.frames_processed == 8
    assert len(detector.calls) == 8  # one inference per frame


def test_process_writes_json_report_with_statistics(settings, sample_video: Path) -> None:
    result = VideoProcessor(MockDetector(), settings=settings).process(sample_video)
    assert result.report_path is not None and result.report_path.exists()

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["input"] == sample_video.name
    assert report["output_video"] == result.output_path.name
    assert report["video"]["width"] == 160

    stats = report["statistics"]
    # MockDetector yields 2 detections (>= 0.5) per frame x 8 frames.
    assert stats["total_detections"] == 16
    assert stats["unique_classes"] == 2
    assert stats["detections_per_class"] == {"fish": 8, "shark": 8}

    processing = report["processing"]
    assert processing["frames_processed"] == 8
    assert processing["frames_with_detections"] == 8
    assert processing["processing_fps"] > 0

    # Per-frame breakdown only for frames with detections.
    assert len(report["frame_detections"]) == 8
    assert report["frame_detections"][0]["frame"] == 0


def test_progress_callback_is_invoked_per_frame(settings, sample_video: Path) -> None:
    events: list[tuple[int, int | None]] = []
    VideoProcessor(MockDetector(), settings=settings).process(
        sample_video,
        progress_callback=lambda done, total: events.append((done, total)),
    )
    assert [done for done, _ in events] == list(range(1, 9))
    assert all(total == 8 for _, total in events)


def test_progress_callback_errors_do_not_abort(settings, sample_video: Path) -> None:
    def exploding_callback(done: int, total) -> None:
        raise RuntimeError("UI exploded")

    result = VideoProcessor(MockDetector(), settings=settings).process(
        sample_video, progress_callback=exploding_callback
    )
    assert result.frames_processed == 8


def test_max_frames_limits_processing(settings, sample_video: Path) -> None:
    detector = MockDetector()
    result = VideoProcessor(detector, settings=settings).process(sample_video, max_frames=3)
    assert result.frames_processed == 3
    assert _open_frame_count(result.output_path) == 3
    assert len(detector.calls) == 3


def test_confidence_threshold_applies_to_every_frame(settings, sample_video: Path) -> None:
    result = VideoProcessor(MockDetector(), settings=settings).process(
        sample_video, confidence=0.8, save_json=True
    )
    # Only the 0.92 detection survives on each frame.
    assert result.total_detections == 8
    assert result.statistics["detections_per_class"] == {"shark": 8}


def test_no_json_option(settings, sample_video: Path) -> None:
    result = VideoProcessor(MockDetector(), settings=settings).process(
        sample_video, save_json=False
    )
    assert result.report_path is None
    assert result.output_path.exists()


def test_enhancement_flag_is_propagated(settings, sample_video: Path) -> None:
    from app.processing.enhancement import EnhancementConfig

    detector = MockDetector()
    result = VideoProcessor(detector, settings=settings).process(
        sample_video, enhancement=EnhancementConfig(enabled=True)
    )
    assert result.enhanced is True
    assert all(call["enhanced"] for call in detector.calls)


def test_video_rejects_missing_file(settings, tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="not found"):
        VideoProcessor(MockDetector(), settings=settings).process(tmp_path / "ghost.mp4")


def test_result_summary_stats_shape(settings, sample_video: Path) -> None:
    result = VideoProcessor(MockDetector(), settings=settings).process(sample_video)
    summary = result.summary_stats()
    for key in (
        "total_detections",
        "unique_classes",
        "detections_per_class",
        "average_confidence",
        "max_confidence",
        "frames_total",
        "frames_processed",
        "frames_with_detections",
        "processing_fps",
        "elapsed_seconds",
        "detections_per_frame",
    ):
        assert key in summary
    assert summary["frames_processed"] == 8
    assert not result.is_empty
