from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .global_pad_export import (
    DEFAULT_TARGET_HEIGHTS,
    _float32_output_bytes,
    build_geometry_plan,
)


DEFAULT_DATA_ROOT = Path("/mnt/t9/vindr-data/vindr")
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/t9/vindr-data/preprocessed-vindr-global-pad-multires-v1"
)
DEFAULT_SPLIT_ASSIGNMENTS = DEFAULT_DATA_ROOT / "global_pad_split_assignments.csv"
OPTIONAL_TARGET_HEIGHTS = (*DEFAULT_TARGET_HEIGHTS, 2048)

_LAUNCH_LOCK = threading.Lock()
_MANIFEST_CACHE_LOCK = threading.Lock()
_MANIFEST_LINE_CACHE: dict[str, tuple[int, int, int, int]] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _parse_timestamp(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _nearest_existing_ancestor(path: Path) -> Path:
    candidate = path.expanduser().resolve(strict=False)
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def exporter_process_alive(pid: Any, *, proc_root: Path = Path("/proc")) -> bool:
    """Return true only when PID still belongs to this exporter.

    Reading procfs is deliberately non-invasive: monitoring never sends a signal
    to the running exporter.
    """
    try:
        numeric_pid = int(pid)
    except (TypeError, ValueError):
        return False
    if numeric_pid <= 0:
        return False
    try:
        command = (proc_root / str(numeric_pid) / "cmdline").read_bytes()
    except OSError:
        return False
    text = command.replace(b"\x00", b" ").decode("utf-8", errors="replace")
    return (
        "export_global_padded_multires.py" in text
        or "vindr_mammo.global_pad_export" in text
    )


def _manifest_data_rows(path: Path) -> int:
    try:
        stat = path.stat()
    except OSError:
        return 0
    cache_key = str(path)
    identity = (int(stat.st_dev), int(stat.st_ino))
    with _MANIFEST_CACHE_LOCK:
        cached = _MANIFEST_LINE_CACHE.get(cache_key)
        if (
            cached is not None
            and cached[:2] == identity
            and int(stat.st_size) >= cached[2]
        ):
            offset = cached[2]
            line_count = cached[3]
        else:
            offset = 0
            line_count = 0
        try:
            with path.open("rb") as handle:
                handle.seek(offset)
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    line_count += block.count(b"\n")
                consumed_size = handle.tell()
        except OSError:
            return max(0, line_count - 1)
        _MANIFEST_LINE_CACHE[cache_key] = (
            identity[0],
            identity[1],
            consumed_size,
            line_count,
        )
        return max(0, line_count - 1)


def _tail_text(path: Path, *, maximum_bytes: int = 12_000) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - maximum_bytes))
            value = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    return value[-maximum_bytes:]


def _geometry_variants(payload: dict[str, Any]) -> list[dict[str, Any]]:
    geometry = payload.get("geometry", {})
    if not isinstance(geometry, dict):
        return []
    variants = geometry.get("variants", [])
    if not isinstance(variants, list):
        return []
    return [item for item in variants if isinstance(item, dict)]


def read_global_pad_status(output_root: str | Path) -> dict[str, Any]:
    """Read progress and capacity without touching the exporter process."""
    root = Path(output_root).expanduser().resolve(strict=False)
    metadata_root = root / "metadata"
    status = _read_json(metadata_root / "status.json")
    launcher = _read_json(metadata_root / "gui_run.json")
    resolved = _read_json(metadata_root / "resolved_config.json")

    status_pid = status.get("pid")
    launcher_pid = launcher.get("pid")
    status_pid_alive = exporter_process_alive(status_pid)
    launcher_pid_alive = exporter_process_alive(launcher_pid)
    active = status_pid_alive or launcher_pid_alive
    active_pid = status_pid if status_pid_alive else (launcher_pid if launcher_pid_alive else None)

    reported = str(status.get("status") or "").strip()
    if launcher_pid_alive and not status_pid_alive and reported not in {
        "completed",
        "completed_with_errors",
        "failed",
    }:
        effective_status = "starting_or_scanning"
    elif reported == "running" and not status_pid_alive:
        effective_status = "interrupted"
    elif reported:
        effective_status = reported
    elif launcher_pid_alive:
        effective_status = "starting_or_scanning"
    elif launcher:
        effective_status = "interrupted"
    else:
        effective_status = "not_started"

    # A live launcher always wins over stale terminal state from an older run.
    if launcher_pid_alive and status.get("pid") != launcher_pid:
        effective_status = "starting_or_scanning"

    variants = _geometry_variants(status) or _geometry_variants(resolved)
    image_count = int(
        status.get("image_count")
        or resolved.get("selected_image_count")
        or resolved.get("source_image_count")
        or 0
    )
    variant_count = len(variants)
    if effective_status in {"completed", "completed_with_errors"}:
        manifest_path = metadata_root / "manifest.csv"
    else:
        manifest_path = metadata_root / "manifest.partial.csv"
    manifest_rows = _manifest_data_rows(manifest_path)
    expected_rows = image_count * variant_count
    processed_images = (
        min(image_count, manifest_rows // variant_count)
        if variant_count and image_count
        else 0
    )
    progress = (
        min(1.0, manifest_rows / expected_rows) if expected_rows > 0 else None
    )

    started_at = status.get("started_at") or launcher.get("launched_at")
    started = _parse_timestamp(started_at)
    elapsed_seconds: float | None = None
    eta_seconds: float | None = None
    if started is not None:
        elapsed_seconds = max(
            0.0,
            (datetime.now(timezone.utc) - started).total_seconds(),
        )
        if (
            active
            and progress is not None
            and 0.0 < progress < 1.0
            and elapsed_seconds > 0
        ):
            eta_seconds = elapsed_seconds * (1.0 - progress) / progress

    try:
        disk = shutil.disk_usage(_nearest_existing_ancestor(root))
        disk_payload: dict[str, int | None] = {
            "total_bytes": int(disk.total),
            "used_bytes": int(disk.used),
            "free_bytes": int(disk.free),
        }
    except OSError:
        disk_payload = {
            "total_bytes": None,
            "used_bytes": None,
            "free_bytes": None,
        }

    return {
        "output_root": str(root),
        "status": effective_status,
        "reported_status": reported or None,
        "active": active,
        "pid": int(active_pid) if active_pid is not None else None,
        "workers": status.get("workers") or resolved.get("workers"),
        "image_count": image_count,
        "variant_count": variant_count,
        "variants": variants,
        "manifest_path": str(manifest_path),
        "manifest_rows": manifest_rows,
        "expected_rows": expected_rows,
        "processed_images": processed_images,
        "progress": progress,
        "started_at": started_at,
        "elapsed_seconds": elapsed_seconds,
        "eta_seconds": eta_seconds,
        "error": status.get("error"),
        "error_count": status.get("error_count"),
        "log_path": str(metadata_root / "gui_export.log"),
        "log_tail": _tail_text(metadata_root / "gui_export.log"),
        "estimated_allocated_output_bytes": resolved.get(
            "estimated_allocated_output_bytes"
        ),
        **disk_payload,
    }


def validate_target_heights(values: Sequence[int]) -> tuple[int, ...]:
    heights = tuple(dict.fromkeys(int(value) for value in values))
    if not heights:
        raise ValueError("Select at least one compact output height.")
    unsupported = [value for value in heights if value not in OPTIONAL_TARGET_HEIGHTS]
    if unsupported:
        raise ValueError(
            "The GUI supports these compact heights only: "
            + ", ".join(str(value) for value in OPTIONAL_TARGET_HEIGHTS)
        )
    return tuple(sorted(heights))


def build_global_pad_command(
    *,
    data_root: str | Path,
    output_root: str | Path,
    split_assignments: str | Path | None,
    target_heights: Sequence[int],
    workers: int,
    overwrite: bool = False,
    rescan: bool = False,
    allow_low_space: bool = False,
    python_executable: str | Path | None = None,
) -> list[str]:
    heights = validate_target_heights(target_heights)
    worker_count = int(workers)
    if worker_count < 1:
        raise ValueError("Workers must be at least 1.")
    repository_root = Path(__file__).resolve().parents[2]
    command = [
        str(python_executable or sys.executable),
        str(repository_root / "scripts" / "export_global_padded_multires.py"),
        "--data-root",
        str(Path(data_root).expanduser().resolve(strict=False)),
        "--output-root",
        str(Path(output_root).expanduser().resolve(strict=False)),
        "--split-assignments",
        (
            str(Path(split_assignments).expanduser().resolve(strict=False))
            if split_assignments
            else "official"
        ),
        "--target-heights",
        *(str(value) for value in heights),
        "--workers",
        str(worker_count),
    ]
    if overwrite:
        command.append("--overwrite")
    if rescan:
        command.append("--rescan")
    if allow_low_space:
        command.append("--allow-low-space")
    return command


def command_preview(**kwargs: Any) -> str:
    command = build_global_pad_command(**kwargs)
    repository_root = Path(__file__).resolve().parents[2]
    return f"cd {shlex.quote(str(repository_root))}\nPYTHONPATH=src {shlex.join(command)}"


def estimate_selected_payload(
    output_root: str | Path,
    target_heights: Sequence[int],
) -> dict[str, Any]:
    """Estimate dense float32 bytes using already-scanned output metadata."""
    heights = validate_target_heights(target_heights)
    root = Path(output_root).expanduser().resolve(strict=False)
    status = _read_json(root / "metadata" / "status.json")
    resolved = _read_json(root / "metadata" / "resolved_config.json")
    geometry = status.get("geometry") or resolved.get("geometry") or {}
    maximum_height = int(geometry.get("maximum_source_height") or 0)
    maximum_width = int(geometry.get("maximum_source_width") or 0)
    image_count = int(
        status.get("image_count")
        or resolved.get("selected_image_count")
        or resolved.get("source_image_count")
        or 0
    )
    if maximum_height <= 0 or maximum_width <= 0 or image_count <= 0:
        return {}
    plan = build_geometry_plan(maximum_height, maximum_width, heights)
    return {
        "image_count": image_count,
        "payload_bytes": _float32_output_bytes(image_count, plan.variants),
        "variants": [
            {
                "name": variant.name,
                "height": variant.output_height,
                "width": variant.output_width,
            }
            for variant in plan.variants
        ],
    }


def launch_global_pad_export(
    *,
    data_root: str | Path,
    output_root: str | Path,
    split_assignments: str | Path | None,
    target_heights: Sequence[int],
    workers: int,
    overwrite: bool = False,
    rescan: bool = False,
    allow_low_space: bool = False,
    python_executable: str | Path | None = None,
) -> dict[str, Any]:
    """Start a detached, logged exporter without managing or signaling it later."""
    data = Path(data_root).expanduser().resolve(strict=False)
    output = Path(output_root).expanduser().resolve(strict=False)
    split = (
        Path(split_assignments).expanduser().resolve(strict=False)
        if split_assignments
        else None
    )
    if not data.is_dir():
        raise FileNotFoundError(f"VinDr data root not found: {data}")
    for required in ("images", "breast-level_annotations.csv", "finding_annotations.csv"):
        if not (data / required).exists():
            raise FileNotFoundError(f"Required source is missing: {data / required}")
    if split is not None and not split.is_file():
        raise FileNotFoundError(f"Split-assignment CSV not found: {split}")
    if output == data or output == data / "images" or data / "images" in output.parents:
        raise ValueError("Output must not replace the original VinDr data or images folder.")

    with _LAUNCH_LOCK:
        current = read_global_pad_status(output)
        if current["active"]:
            raise RuntimeError(
                f"Exporter PID {current['pid']} is already active for {output}. "
                "The GUI will monitor it and will not start a competing process."
            )
        command = build_global_pad_command(
            data_root=data,
            output_root=output,
            split_assignments=split,
            target_heights=target_heights,
            workers=workers,
            overwrite=overwrite,
            rescan=rescan,
            allow_low_space=allow_low_space,
            python_executable=python_executable,
        )
        metadata_root = output / "metadata"
        metadata_root.mkdir(parents=True, exist_ok=True)
        log_path = metadata_root / "gui_export.log"
        repository_root = Path(__file__).resolve().parents[2]
        environment = os.environ.copy()
        source_root = str(repository_root / "src")
        previous_pythonpath = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = (
            source_root + os.pathsep + previous_pythonpath
            if previous_pythonpath
            else source_root
        )
        with log_path.open("ab") as log_handle:
            log_handle.write(
                f"\n[{_now()}] GUI launch: {shlex.join(command)}\n".encode("utf-8")
            )
            log_handle.flush()
            process = subprocess.Popen(
                command,
                cwd=repository_root,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
        launch_record = {
            "status": "launched",
            "launched_at": _now(),
            "pid": int(process.pid),
            "output_root": str(output),
            "command": command,
            "log_path": str(log_path),
        }
        _atomic_write_json(metadata_root / "gui_run.json", launch_record)
        return launch_record
