"""Video pipeline: probe -> frame-by-frame detection -> annotated video.

Adapted for BlueEye from the video inference loop of *marine-detect* by
Orange Business Services SA (AGPL-3.0-only) - extended with streaming
progress callbacks, aggregated statistics and JSON reports.

Frames are processed incrementally with OpenCV ``VideoCapture`` /
``VideoWriter``; the video is never loaded into memory as a whole.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

import cv2
import numpy as np

from app.config import Settings, get_settings
from app.detection.results import compute_statistics
from app.processing.drawing import draw_detections
from app.processing.enhancement import EnhancementConfig, enhance
from app.processing.image_processor import ProcessingError
from app.utils.file_utils import ensure_dir, save_json as write_json, unique_output_path
from app.utils.validation import validate_video_file

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.detection.detector import MarineDetector

logger = logging.getLogger(__name__)

#: progress_callback(frames_processed, total_frames_or_None)
ProgressCallback = Callable[[int, Optional[int]], None]

_FALLBACK_FPS = 25.0
_MIN_FPS = 0.5
_MAX_FPS = 240.0

# OpenCV 4 calls it CAP_PROP_FRAME_FPS, OpenCV 5 renamed it to CAP_PROP_FPS.
CAP_PROP_FPS = getattr(cv2, "CAP_PROP_FRAME_FPS", None) or getattr(cv2, "CAP_PROP_FPS", 5)


@dataclass(frozen=True)
class VideoInfo:
    """Metadata of a video file."""

    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float
    file_size_bytes: int
    fourcc: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.path.name,
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 3),
            "frame_count": self.frame_count,
            "duration_seconds": round(self.duration_seconds, 2),
            "file_size_bytes": self.file_size_bytes,
            "codec": self.fourcc,
        }


def probe_video(path: str | Path) -> VideoInfo:
    """Validate a video and read its metadata.

    Raises :class:`app.utils.validation.ValidationError` when the file is
    missing, unsupported, empty or corrupt.
    """
    file_path = validate_video_file(path)
    cap = cv2.VideoCapture(str(file_path))
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
        fourcc = "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)).strip()
    finally:
        cap.release()

    if frame_count < 0:
        frame_count = 0
    duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
    return VideoInfo(
        path=file_path,
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_seconds=duration,
        file_size_bytes=file_path.stat().st_size,
        fourcc=fourcc or "unknown",
    )


@dataclass
class VideoProcessResult:
    """Outcome of one processed video."""

    input_path: Path
    output_path: Path
    report_path: Path | None
    info: VideoInfo
    frames_processed: int
    frames_with_detections: int
    models_used: list[str] = field(default_factory=list)
    statistics: dict[str, Any] = field(default_factory=dict)
    frame_detections: list[dict[str, Any]] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    enhanced: bool = False

    @property
    def total_detections(self) -> int:
        return int(self.statistics.get("total_detections", 0))

    @property
    def processing_fps(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.frames_processed / self.elapsed_seconds

    @property
    def is_empty(self) -> bool:
        return self.total_detections == 0

    def summary_stats(self) -> dict[str, Any]:
        """Statistics dict including video-specific fields."""
        stats = dict(self.statistics)
        stats.update(
            {
                "frames_total": self.info.frame_count or self.frames_processed,
                "frames_processed": self.frames_processed,
                "frames_with_detections": self.frames_with_detections,
                "processing_fps": round(self.processing_fps, 2),
                "elapsed_seconds": round(self.elapsed_seconds, 2),
            }
        )
        if self.frames_processed:
            stats["detections_per_frame"] = round(
                self.total_detections / self.frames_processed, 3
            )
        return stats


class _FFmpegH264Writer:
    """Write an H.264 MP4 by piping raw frames to the ffmpeg binary bundled
    with *imageio-ffmpeg*.

    OpenCV's default ``mp4v`` (MPEG-4 Part 2) output cannot be decoded by
    Chromium, so the annotated video would not play in the Streamlit
    preview. H.264 with ``yuv420p`` plays in every browser and player.

    ``write()`` / ``release()`` mirror the OpenCV ``VideoWriter`` interface;
    ``release()`` never raises - failures are recorded in :attr:`error` and
    surfaced by the caller, so a finally-block release can never mask an
    exception already in flight.
    """

    codec = "h264"

    def __init__(self, path: Path, fps: float, size: tuple[int, int]) -> None:
        import imageio_ffmpeg  # lazy import: keeps OpenCV-only fallback possible

        width, height = size
        cmd = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{fps:g}",
            "-i",
            "pipe:0",
            "-an",
        ]
        if width % 2 or height % 2:  # yuv420p needs even dimensions
            cmd += ["-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2"]
        cmd += [
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-movflags",
            "+faststart",
            str(path),
        ]
        self.error: str | None = None
        self._stderr_lines: list[str] = []
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        threading.Thread(target=self._drain_stderr, daemon=True).start()

    def _drain_stderr(self) -> None:
        assert self._proc.stderr is not None
        for raw in iter(self._proc.stderr.readline, b""):
            line = raw.decode("utf-8", errors="replace").strip()
            if line:
                self._stderr_lines = (self._stderr_lines + [line])[-20:]

    def _fail_reason(self, code: int | None) -> str:
        tail = "; ".join(self._stderr_lines[-5:])
        detail = f": {tail}" if tail else ""
        return f"ffmpeg encoder exited with code {code}{detail}"

    def write(self, frame: np.ndarray) -> None:
        if self._proc.poll() is not None:
            raise RuntimeError(self._fail_reason(self._proc.returncode))
        try:
            self._proc.stdin.write(  # type: ignore[union-attr]
                np.ascontiguousarray(frame, dtype=np.uint8).tobytes()
            )
        except BrokenPipeError as exc:
            raise RuntimeError(self._fail_reason(self._proc.poll())) from exc

    def release(self) -> None:
        if self._proc.stdin is not None and not self._proc.stdin.closed:
            try:
                self._proc.stdin.close()
            except OSError:
                pass
        try:
            code = self._proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
            self.error = "ffmpeg did not finish writing the video in time."
            return
        if code != 0:
            self.error = self._fail_reason(code)


class _OpenCVWriter:
    """Fallback writer using OpenCV's ``mp4v`` codec (not browser-playable)."""

    codec = "mp4v"

    def __init__(self, path: Path, fps: float, size: tuple[int, int]) -> None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(path), fourcc, fps, size)
        if not self._writer.isOpened():
            self._writer.release()
            raise OSError("cv2.VideoWriter (mp4v) could not be opened")

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def release(self) -> None:
        self._writer.release()


def _open_writer(path: Path, fps: float, size: tuple[int, int]) -> Any:
    """Create the best available video writer.

    Prefers H.264 (browser/playable everywhere) via *imageio-ffmpeg* and
    falls back to OpenCV's ``mp4v`` so that video processing keeps working
    even when the optional ffmpeg helper is not installed.
    """
    try:
        writer = _FFmpegH264Writer(path, fps, size)
        logger.debug("Using H.264 (ffmpeg) video writer for '%s'.", path.name)
        return writer
    except Exception as exc:  # noqa: BLE001 - any failure means fall back
        logger.warning(
            "H.264 writer unavailable (%s); falling back to OpenCV 'mp4v'. "
            "The annotated video may not play in a browser preview.",
            exc,
        )
        return _OpenCVWriter(path, fps, size)


class VideoProcessor:
    """Processes a video frame by frame."""

    def __init__(self, detector: "MarineDetector", settings: Settings | None = None) -> None:
        self.detector = detector
        self.settings = settings or get_settings()

    def process(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
        model_selection: str | None = None,
        confidence: float | None = None,
        enhancement: EnhancementConfig | None = None,
        progress_callback: ProgressCallback | None = None,
        save_json: bool = True,
        max_frames: int | None = None,
    ) -> VideoProcessResult:
        """Run the full video pipeline and return the stored artefacts.

        Parameters
        ----------
        progress_callback:
            Called after every processed frame as
            ``(frames_processed, total_frames_or_None)``. Exceptions raised
            by the callback are logged and ignored so that UI progress
            rendering can never abort a long job.
        max_frames:
            Optional cap on processed frames (useful for demos/tests).
        """
        settings = self.settings
        info = probe_video(input_path)

        # Fail fast (before creating any output file) if weights are absent.
        keys = self.detector.load_model(model_selection)

        fps = info.fps if _MIN_FPS <= info.fps <= _MAX_FPS else _FALLBACK_FPS
        if fps != info.fps:
            logger.warning(
                "Unusable FPS value %s in '%s'; writing output at %.1f FPS.",
                info.fps,
                info.path.name,
                fps,
            )

        if output_path is None:
            out_path = unique_output_path(settings.videos_output_dir, "detection", ".mp4")
        else:
            out_path = Path(output_path)
            ensure_dir(out_path.parent)

        cap = cv2.VideoCapture(str(info.path))
        if not cap.isOpened():
            raise ProcessingError(f"Could not open video '{info.path.name}' for reading.")

        try:
            writer = _open_writer(out_path, fps, (info.width, info.height))
        except OSError as exc:
            cap.release()
            raise ProcessingError(
                f"Could not open a video writer for '{out_path.name}': {exc} "
                "Check available disk space and output permissions."
            ) from exc

        enhanced_flag = bool(enhancement is not None and enhancement.enabled)
        selection = model_selection or settings.default_model_selection

        per_class: Counter = Counter()
        confidences: list[float] = []
        frame_detections: list[dict[str, Any]] = []
        models_used: set[str] = set()
        frames_processed = 0
        frames_with_detections = 0
        total_hint = info.frame_count if info.frame_count > 0 else None
        started = time.perf_counter()

        logger.info(
            "Processing video '%s' (%sx%s, %s frames, models=%s)",
            info.path.name,
            info.width,
            info.height,
            total_hint or "unknown",
            ", ".join(keys),
        )

        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                if frame.shape[1] != info.width or frame.shape[0] != info.height:
                    frame = cv2.resize(frame, (info.width, info.height))

                det_frame = enhance(frame, enhancement) if enhanced_flag else frame
                detection = self.detector.predict_frame(
                    det_frame,
                    model_selection=model_selection,
                    confidence=confidence,
                    frame_index=frames_processed,
                    enhanced=enhanced_flag,
                )
                models_used.update(detection.models_used)

                annotated = draw_detections(frame, detection.detections)
                writer.write(np.ascontiguousarray(annotated, dtype=np.uint8))

                if detection.detections:
                    frames_with_detections += 1
                    frame_detections.append(
                        {
                            "frame": frames_processed,
                            "detections": [
                                det.to_report_dict() for det in detection.sorted_detections()
                            ],
                        }
                    )
                    for det in detection.detections:
                        per_class[det.class_name] += 1
                        confidences.append(det.confidence)

                frames_processed += 1
                if progress_callback is not None:
                    try:
                        progress_callback(frames_processed, total_hint)
                    except Exception:  # noqa: BLE001 - progress must not abort the job
                        logger.debug("Progress callback raised; continuing.", exc_info=True)
                if max_frames is not None and frames_processed >= max_frames:
                    logger.info("Stopped after max_frames=%d as requested.", max_frames)
                    break
        except RuntimeError:
            raise
        except Exception as exc:  # noqa: BLE001 - friendly message
            raise ProcessingError(
                f"Video processing failed after {frames_processed} frame(s): {exc}"
            ) from exc
        finally:
            cap.release()
            writer.release()

        elapsed = time.perf_counter() - started

        if frames_processed == 0:
            out_path.unlink(missing_ok=True)
            raise ProcessingError(
                f"No frames could be read from '{info.path.name}'. "
                "The video may be corrupt or use an unsupported codec."
            )
        writer_error = getattr(writer, "error", None)
        if writer_error:
            out_path.unlink(missing_ok=True)
            raise ProcessingError(f"The annotated video could not be written ({writer_error}).")
        if not out_path.exists() or out_path.stat().st_size == 0:
            raise ProcessingError(
                "The annotated video could not be written. "
                "Check free disk space and output permissions."
            )

        stats = compute_statistics(confidences, per_class)
        result = VideoProcessResult(
            input_path=info.path,
            output_path=out_path,
            report_path=None,
            info=info,
            frames_processed=frames_processed,
            frames_with_detections=frames_with_detections,
            models_used=sorted(models_used),
            statistics=stats,
            frame_detections=frame_detections,
            elapsed_seconds=elapsed,
            enhanced=enhanced_flag,
        )

        if save_json:
            report = {
                "input": info.path.name,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "models": result.models_used,
                "confidence_threshold": confidence,
                "enhanced": enhanced_flag,
                "video": info.to_dict(),
                "output_video": out_path.name,
                "output_codec": getattr(writer, "codec", "unknown"),
                "processing": result.summary_stats(),
                "statistics": stats,
                "frame_detections": frame_detections,
            }
            report_path = settings.reports_output_dir / f"{out_path.stem}.json"
            try:
                write_json(report_path, report)
                result.report_path = report_path
            except OSError as exc:
                raise ProcessingError(
                    f"Could not save the JSON report to '{settings.reports_output_dir}': {exc}"
                ) from exc

        logger.info(
            "Video '%s': %d/%d frames, %d detection(s) in %.2f s -> %s",
            info.path.name,
            frames_processed,
            total_hint or frames_processed,
            result.total_detections,
            elapsed,
            out_path,
        )
        return result
