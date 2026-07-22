from __future__ import annotations

import copy
import os
import threading
import time
import uuid
from collections import OrderedDict, deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ProgressCallback = Callable[[dict[str, Any]], None]
ExportRunner = Callable[..., Any]


class DuplicateOutputRootError(ValueError):
    """Raised when two retained queue jobs target the same output directory."""


class QueueJobNotFoundError(KeyError):
    """Raised when a queue operation references an unknown job id."""


class InvalidJobStateError(RuntimeError):
    """Raised when an operation is invalid for the job's current state."""


@dataclass
class _QueueJob:
    job_id: str
    name: str
    config: dict[str, Any]
    output_root: str
    estimated_bytes: int | None
    metadata: dict[str, Any]
    status: str = "queued"
    attempts: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    progress_fraction: float | None = None
    progress: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: str | None = None


class ExportQueueManager:
    """Thread-safe FIFO export queue backed by exactly one worker thread.

    The runner defaults to :func:`vindr_mammo.export.export_from_config`, loaded
    lazily so importing this backend does not import the DICOM/PyTorch stack.
    Custom runners are useful for tests and for alternate extraction pipelines.

    Configs are deep-copied at enqueue time and copied again before invoking the
    runner. Mutating either the caller's config or the runner's argument cannot
    change the retained reproducibility snapshot.
    """

    TERMINAL_STATUSES = frozenset({"completed", "failed"})

    def __init__(
        self,
        runner: ExportRunner | None = None,
        *,
        auto_start: bool = False,
        id_factory: Callable[[], str] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._runner = runner or _default_export_runner
        self._id_factory = id_factory or (lambda: uuid.uuid4().hex)
        self._clock = clock
        self._condition = threading.Condition(threading.RLock())
        self._jobs: OrderedDict[str, _QueueJob] = OrderedDict()
        self._pending: deque[str] = deque()
        self._output_roots: dict[str, str] = {}
        self._worker: threading.Thread | None = None
        self._running_job_id: str | None = None
        self._started = False
        self._shutdown_requested = False
        if auto_start:
            self.start()

    def enqueue(
        self,
        config: Mapping[str, Any],
        *,
        name: str | None = None,
        estimated_bytes: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Append a deep-copied export config and return its stable job id."""

        config_copy = copy.deepcopy(dict(config))
        output_root = _output_root_from_config(config_copy)
        # Freeze the resolved absolute path as part of the retained config too.
        # Otherwise a relative output_root could resolve differently if the
        # process working directory changes between enqueue and execution.
        paths_copy = dict(config_copy.get("paths", {}) or {})
        paths_copy["output_root"] = output_root
        config_copy["paths"] = paths_copy
        normalized_root = _normalize_output_root(output_root)
        estimate = None if estimated_bytes is None else max(0, int(estimated_bytes))
        metadata_copy = copy.deepcopy(dict(metadata or {}))

        with self._condition:
            self._ensure_accepting_locked()
            existing_id = self._output_roots.get(normalized_root)
            if existing_id is not None:
                existing = self._jobs.get(existing_id)
                status = existing.status if existing is not None else "retained"
                raise DuplicateOutputRootError(
                    f"Output root {output_root!r} is already owned by queue job "
                    f"{existing_id!r} ({status}). Remove that job before reusing the path."
                )

            job_id = str(self._id_factory())
            if not job_id:
                raise ValueError("id_factory returned an empty job id")
            if job_id in self._jobs:
                raise ValueError(f"id_factory returned duplicate job id {job_id!r}")
            now = float(self._clock())
            display_name = str(name or Path(output_root).name or job_id)
            job = _QueueJob(
                job_id=job_id,
                name=display_name,
                config=config_copy,
                output_root=output_root,
                estimated_bytes=estimate,
                metadata=metadata_copy,
                created_at=now,
                updated_at=now,
            )
            self._jobs[job_id] = job
            self._pending.append(job_id)
            self._output_roots[normalized_root] = job_id
            self._condition.notify_all()
            return job_id

    def start(self) -> None:
        """Start the single FIFO worker. Calling this more than once is safe."""

        with self._condition:
            if self._shutdown_requested:
                raise RuntimeError("Cannot start an export queue after shutdown")
            if self._worker is not None and self._worker.is_alive():
                self._started = True
                self._condition.notify_all()
                return
            self._started = True
            self._worker = threading.Thread(
                target=self._worker_loop,
                name="vindr-export-queue",
                daemon=True,
            )
            self._worker.start()
            self._condition.notify_all()

    def remove(self, job_id: str) -> bool:
        """Remove a queued or terminal job.

        Running jobs cannot be removed because the current exporter has no safe
        transactional cancellation boundary. Removing a retained terminal job
        also releases its output root for a new queue entry.
        """

        with self._condition:
            job = self._job_locked(job_id)
            if job.status == "running":
                raise InvalidJobStateError(f"Cannot remove running job {job_id!r}")
            if job.status == "queued":
                self._pending = deque(value for value in self._pending if value != job_id)
            self._jobs.pop(job_id, None)
            self._output_roots.pop(_normalize_output_root(job.output_root), None)
            self._condition.notify_all()
            return True

    def retry(self, job_id: str) -> str:
        """Append a failed job to the end of the FIFO queue using the same id."""

        with self._condition:
            self._ensure_accepting_locked()
            job = self._job_locked(job_id)
            if job.status != "failed":
                raise InvalidJobStateError(
                    f"Only failed jobs can be retried; {job_id!r} is {job.status!r}"
                )
            now = float(self._clock())
            job.status = "queued"
            job.updated_at = now
            job.started_at = None
            job.finished_at = None
            job.progress_fraction = None
            job.progress = {}
            job.result = None
            job.error = None
            self._pending.append(job_id)
            self._condition.notify_all()
            return job_id

    def snapshot(self, *, include_config: bool = False) -> dict[str, Any]:
        """Return a detached, JSON-oriented view of all retained jobs."""

        with self._condition:
            pending_ids = list(self._pending)
            positions = {job_id: index for index, job_id in enumerate(pending_ids, start=1)}
            jobs = [
                self._job_snapshot_locked(
                    job,
                    queue_position=positions.get(job.job_id),
                    include_config=include_config,
                )
                for job in self._jobs.values()
            ]
            return {
                "started": bool(self._started),
                "accepting_jobs": not self._shutdown_requested,
                "running_job_id": self._running_job_id,
                "pending_job_ids": pending_ids,
                "jobs": jobs,
            }

    def get_job(self, job_id: str, *, include_config: bool = False) -> dict[str, Any]:
        """Return one detached job snapshot."""

        with self._condition:
            job = self._job_locked(job_id)
            try:
                queue_position = list(self._pending).index(job_id) + 1
            except ValueError:
                queue_position = None
            return self._job_snapshot_locked(
                job,
                queue_position=queue_position,
                include_config=include_config,
            )

    def wait_for_idle(self, timeout: float | None = None) -> bool:
        """Wait until no job is queued or running; return ``False`` on timeout."""

        deadline = None if timeout is None else time.monotonic() + max(0.0, float(timeout))
        with self._condition:
            while self._pending or self._running_job_id is not None:
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def shutdown(self, *, wait: bool = True, timeout: float | None = None) -> bool:
        """Stop accepting jobs and let the worker drain the existing FIFO queue."""

        with self._condition:
            self._shutdown_requested = True
            worker = self._worker
            self._condition.notify_all()
        if wait and worker is not None and worker is not threading.current_thread():
            worker.join(timeout=timeout)
        return worker is None or not worker.is_alive()

    def __enter__(self) -> ExportQueueManager:
        self.start()
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.shutdown(wait=True)

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                while not self._pending:
                    if self._shutdown_requested:
                        self._started = False
                        self._condition.notify_all()
                        return
                    self._condition.wait()
                job_id = self._pending.popleft()
                job = self._jobs.get(job_id)
                if job is None or job.status != "queued":
                    continue
                now = float(self._clock())
                job.status = "running"
                job.attempts += 1
                job.started_at = now
                job.updated_at = now
                job.finished_at = None
                job.error = None
                self._running_job_id = job_id
                runner_config = copy.deepcopy(job.config)
                self._condition.notify_all()

            try:
                result = self._runner(
                    runner_config,
                    progress_callback=self._progress_callback(job_id),
                )
                result_snapshot = _result_snapshot(result)
            except Exception as exc:
                with self._condition:
                    current = self._jobs.get(job_id)
                    if current is not None:
                        now = float(self._clock())
                        current.status = "failed"
                        current.error = f"{type(exc).__name__}: {exc}"
                        current.result = None
                        current.finished_at = now
                        current.updated_at = now
                    self._running_job_id = None
                    self._condition.notify_all()
                continue

            with self._condition:
                current = self._jobs.get(job_id)
                if current is not None:
                    now = float(self._clock())
                    current.status = "completed"
                    current.progress_fraction = 1.0
                    current.result = result_snapshot
                    current.error = None
                    current.finished_at = now
                    current.updated_at = now
                self._running_job_id = None
                self._condition.notify_all()

    def _progress_callback(self, job_id: str) -> ProgressCallback:
        def update(event: dict[str, Any]) -> None:
            payload = copy.deepcopy(dict(event or {}))
            with self._condition:
                job = self._jobs.get(job_id)
                if job is None or job.status != "running":
                    return
                job.progress = payload
                fraction = _progress_fraction(payload)
                if fraction is not None:
                    job.progress_fraction = fraction
                job.updated_at = float(self._clock())
                self._condition.notify_all()

        return update

    def _job_locked(self, job_id: str) -> _QueueJob:
        try:
            return self._jobs[str(job_id)]
        except KeyError as exc:
            raise QueueJobNotFoundError(str(job_id)) from exc

    def _ensure_accepting_locked(self) -> None:
        if self._shutdown_requested:
            raise RuntimeError("Export queue is shutting down and no longer accepts jobs")

    @staticmethod
    def _job_snapshot_locked(
        job: _QueueJob,
        *,
        queue_position: int | None,
        include_config: bool,
    ) -> dict[str, Any]:
        snapshot = {
            "job_id": job.job_id,
            "name": job.name,
            "output_root": job.output_root,
            "estimated_bytes": job.estimated_bytes,
            "metadata": copy.deepcopy(job.metadata),
            "status": job.status,
            "attempts": int(job.attempts),
            "queue_position": queue_position,
            "created_at": float(job.created_at),
            "updated_at": float(job.updated_at),
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "progress_fraction": job.progress_fraction,
            "progress": copy.deepcopy(job.progress),
            "result": copy.deepcopy(job.result),
            "error": job.error,
        }
        if include_config:
            snapshot["config"] = copy.deepcopy(job.config)
        return snapshot


def _default_export_runner(
    config: dict[str, Any], *, progress_callback: ProgressCallback
) -> Any:
    from .export import export_from_config

    return export_from_config(config, progress_callback=progress_callback)


def _output_root_from_config(config: Mapping[str, Any]) -> str:
    paths = config.get("paths", {}) or {}
    if not isinstance(paths, Mapping):
        raise ValueError("config.paths must be a mapping containing output_root")
    value = paths.get("output_root")
    if value is None or not str(value).strip():
        raise ValueError("config.paths.output_root is required for queued exports")
    return str(Path(str(value)).expanduser().resolve(strict=False))


def _normalize_output_root(path: str) -> str:
    return os.path.normcase(os.path.abspath(os.path.normpath(path)))


def _progress_fraction(event: Mapping[str, Any]) -> float | None:
    for key in ("progress_fraction", "fraction", "progress"):
        value = event.get(key)
        if isinstance(value, (int, float)):
            return min(max(float(value), 0.0), 1.0)
    processed = event.get("processed")
    total = event.get("total")
    try:
        processed_float = float(processed)
        total_float = float(total)
    except (TypeError, ValueError):
        return None
    if total_float <= 0:
        return None
    return min(max(processed_float / total_float, 0.0), 1.0)


def _result_snapshot(result: Any) -> Any:
    if result is None or isinstance(result, (str, int, float, bool)):
        return result
    if isinstance(result, Mapping):
        return copy.deepcopy(dict(result))
    output_root = getattr(result, "output_root", None)
    summary = getattr(result, "summary", None)
    if output_root is not None or summary is not None:
        return {
            "output_root": str(output_root) if output_root is not None else None,
            "summary": copy.deepcopy(summary),
        }
    return {"repr": repr(result)}


__all__ = [
    "DuplicateOutputRootError",
    "ExportQueueManager",
    "InvalidJobStateError",
    "QueueJobNotFoundError",
]
