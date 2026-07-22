from __future__ import annotations

import threading
from collections import defaultdict

import pytest

from vindr_mammo.export_queue import (
    DuplicateOutputRootError,
    ExportQueueManager,
    InvalidJobStateError,
)


def _config(output_root, marker: str) -> dict:
    return {
        "paths": {"output_root": str(output_root)},
        "pipeline": {"marker": marker, "nested": {"values": [marker]}},
    }


def test_queue_runs_fifo_with_one_worker_and_detached_configs(tmp_path) -> None:
    order: list[str] = []
    active = 0
    max_active = 0
    lock = threading.Lock()

    def runner(config, *, progress_callback):
        nonlocal active, max_active
        marker = config["pipeline"]["marker"]
        with lock:
            active += 1
            max_active = max(max_active, active)
            order.append(marker)
        progress_callback({"event": "image_progress", "processed": 1, "total": 2})
        config["pipeline"]["nested"]["values"].append("runner-mutated")
        with lock:
            active -= 1
        return {"marker": marker}

    ids = iter(["job-a", "job-b", "job-c"])
    manager = ExportQueueManager(runner, id_factory=lambda: next(ids))
    first_config = _config(tmp_path / "a", "a")
    first_id = manager.enqueue(first_config, estimated_bytes=123, metadata={"kind": "first"})
    manager.enqueue(_config(tmp_path / "b", "b"))
    manager.enqueue(_config(tmp_path / "c", "c"))

    # The caller's later mutation must not affect the queued snapshot.
    first_config["pipeline"]["marker"] = "changed-after-enqueue"
    first_config["pipeline"]["nested"]["values"].append("caller-mutated")

    manager.start()
    manager.start()  # idempotent; must not create another worker
    assert manager.wait_for_idle(timeout=3.0)

    snapshot = manager.snapshot(include_config=True)
    assert order == ["a", "b", "c"]
    assert max_active == 1
    assert [job["status"] for job in snapshot["jobs"]] == [
        "completed",
        "completed",
        "completed",
    ]
    first = manager.get_job(first_id, include_config=True)
    assert first["estimated_bytes"] == 123
    assert first["metadata"] == {"kind": "first"}
    assert first["config"]["pipeline"]["marker"] == "a"
    assert first["config"]["pipeline"]["nested"]["values"] == ["a"]
    assert first["progress_fraction"] == 1.0
    assert first["result"] == {"marker": "a"}

    # Returned snapshots are detached too.
    first["config"]["pipeline"]["nested"]["values"].append("snapshot-mutated")
    assert manager.get_job(first_id, include_config=True)["config"]["pipeline"][
        "nested"
    ]["values"] == ["a"]
    assert manager.shutdown(timeout=3.0)


def test_queue_freezes_relative_output_root_as_an_absolute_path(
    tmp_path, monkeypatch
) -> None:
    seen: list[str] = []

    def runner(config, *, progress_callback):
        seen.append(config["paths"]["output_root"])

    monkeypatch.chdir(tmp_path)
    manager = ExportQueueManager(runner, id_factory=lambda: "relative")
    job_id = manager.enqueue(_config("relative-output", "relative"))

    frozen = manager.get_job(job_id, include_config=True)
    expected = str((tmp_path / "relative-output").resolve())
    assert frozen["output_root"] == expected
    assert frozen["config"]["paths"]["output_root"] == expected

    # Execution uses the enqueue-time absolute path even if cwd later changes.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    manager.start()
    assert manager.wait_for_idle(timeout=3.0)
    assert seen == [expected]
    assert manager.shutdown(timeout=3.0)


def test_queue_rejects_duplicate_normalized_output_root_until_job_removed(tmp_path) -> None:
    manager = ExportQueueManager(
        lambda config, *, progress_callback: None,
        id_factory=iter(["first", "second"]).__next__,
    )
    first_id = manager.enqueue(_config(tmp_path / "folder" / ".." / "output", "a"))

    with pytest.raises(DuplicateOutputRootError):
        manager.enqueue(_config(tmp_path / "output", "b"))

    assert manager.remove(first_id)
    second_id = manager.enqueue(_config(tmp_path / "output", "b"))
    assert second_id == "second"
    assert manager.snapshot()["pending_job_ids"] == ["second"]
    assert manager.remove(second_id)
    assert manager.shutdown()


def test_failed_job_does_not_block_following_job_and_can_be_retried(tmp_path) -> None:
    attempts = defaultdict(int)
    order: list[tuple[str, int]] = []

    def runner(config, *, progress_callback):
        marker = config["pipeline"]["marker"]
        attempts[marker] += 1
        order.append((marker, attempts[marker]))
        progress_callback({"fraction": 0.25, "stage": "mock"})
        if marker == "bad" and attempts[marker] == 1:
            raise RuntimeError("synthetic failure")
        return {"ok": marker}

    ids = iter(["bad-id", "good-id"])
    manager = ExportQueueManager(runner, id_factory=lambda: next(ids))
    bad_id = manager.enqueue(_config(tmp_path / "bad", "bad"))
    good_id = manager.enqueue(_config(tmp_path / "good", "good"))
    manager.start()
    assert manager.wait_for_idle(timeout=3.0)

    bad = manager.get_job(bad_id)
    good = manager.get_job(good_id)
    assert bad["status"] == "failed"
    assert bad["attempts"] == 1
    assert "synthetic failure" in bad["error"]
    assert good["status"] == "completed"
    assert order == [("bad", 1), ("good", 1)]

    assert manager.retry(bad_id) == bad_id
    assert manager.wait_for_idle(timeout=3.0)
    retried = manager.get_job(bad_id)
    assert retried["status"] == "completed"
    assert retried["attempts"] == 2
    assert retried["error"] is None
    assert order == [("bad", 1), ("good", 1), ("bad", 2)]
    assert manager.shutdown(timeout=3.0)


def test_running_job_cannot_be_removed(tmp_path) -> None:
    entered = threading.Event()
    release = threading.Event()

    def runner(config, *, progress_callback):
        entered.set()
        assert release.wait(timeout=3.0)
        return {"ok": True}

    manager = ExportQueueManager(runner, id_factory=lambda: "blocking")
    job_id = manager.enqueue(_config(tmp_path / "blocking", "blocking"))
    manager.start()
    assert entered.wait(timeout=3.0)

    with pytest.raises(InvalidJobStateError):
        manager.remove(job_id)

    release.set()
    assert manager.wait_for_idle(timeout=3.0)
    assert manager.get_job(job_id)["status"] == "completed"
    assert manager.shutdown(timeout=3.0)
