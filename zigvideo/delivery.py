from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
import time

import imageio_ffmpeg


@dataclass(frozen=True)
class DeliveryReport:
    input: str
    output: str
    source_frames: int
    source_duration_seconds: float
    source_fps: float
    target_fps: int
    target_width: int
    target_height: int
    interpolation: str
    output_frames: int
    output_duration_seconds: float
    elapsed_seconds: float

    def to_dict(self) -> dict:
        return asdict(self)


def build_video_filter(
    target_fps: int = 24,
    target_width: int = 720,
    target_height: int = 1280,
    interpolation: str = "motion",
    source_fps: float | None = None,
    source_duration: float | None = None,
) -> str:
    if target_fps < 1:
        raise ValueError("target_fps must be >= 1")
    if target_width < 16 or target_height < 16:
        raise ValueError("target dimensions are too small")
    if interpolation not in {"motion", "duplicate"}:
        raise ValueError("interpolation must be 'motion' or 'duplicate'")

    if interpolation == "motion":
        minterpolate = (
            f"minterpolate=fps={target_fps}:mi_mode=mci:"
            "mc_mode=aobmc:me_mode=bidir:vsbmc=1"
        )
        if source_fps and source_duration:
            # minterpolate needs future frames to synthesize the final interval.
            # Clone enough tail frames to provide that context, then trim back to
            # the exact source duration so delivery does not become shorter.
            tail_padding = 2.0 / source_fps
            temporal = (
                f"tpad=stop_mode=clone:stop_duration={tail_padding:.6f},"
                f"{minterpolate},trim=duration={source_duration:.6f}"
            )
        else:
            temporal = minterpolate
    else:
        temporal = f"fps={target_fps}"

    spatial = f"scale={target_width}:{target_height}:flags=lanczos"
    return f"{temporal},{spatial}"


def deliver_video(
    input_path: str | Path,
    output_path: str | Path,
    target_fps: int = 24,
    target_width: int = 720,
    target_height: int = 1280,
    interpolation: str = "motion",
) -> DeliveryReport:
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_frames, source_seconds = imageio_ffmpeg.count_frames_and_secs(
        str(input_path)
    )
    source_fps = source_frames / source_seconds if source_seconds else 0.0

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    video_filter = build_video_filter(
        target_fps=target_fps,
        target_width=target_width,
        target_height=target_height,
        interpolation=interpolation,
        source_fps=source_fps,
        source_duration=source_seconds,
    )

    command = [
        ffmpeg,
        "-y",
        "-i",
        str(input_path),
        "-vf",
        video_filter,
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]

    print(
        "[ZigVideo] Delivery: "
        f"{source_fps:.2f} fps -> {target_fps} fps, "
        f"{target_width}x{target_height}, interpolation={interpolation}"
    )

    started = time.perf_counter()
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - started

    if completed.returncode != 0:
        tail = completed.stderr[-4000:]
        raise RuntimeError(f"FFmpeg delivery failed:\n{tail}")

    output_frames, output_seconds = imageio_ffmpeg.count_frames_and_secs(
        str(output_path)
    )

    report = DeliveryReport(
        input=str(input_path.resolve()),
        output=str(output_path.resolve()),
        source_frames=int(source_frames),
        source_duration_seconds=round(float(source_seconds), 3),
        source_fps=round(float(source_fps), 3),
        target_fps=target_fps,
        target_width=target_width,
        target_height=target_height,
        interpolation=interpolation,
        output_frames=int(output_frames),
        output_duration_seconds=round(float(output_seconds), 3),
        elapsed_seconds=round(elapsed, 2),
    )

    report_path = output_path.with_suffix(output_path.suffix + ".json")
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[ZigVideo] Delivery saved: {output_path.resolve()}")
    print(f"[ZigVideo] Delivery report: {report_path.resolve()}")
    return report
