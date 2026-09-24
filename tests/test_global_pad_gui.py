from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vindr_mammo import global_pad_gui


def test_build_command_keeps_2048_optional_and_uses_no_shell() -> None:
    command = global_pad_gui.build_global_pad_command(
        data_root="/data/vindr",
        output_root="/data/export",
        split_assignments=None,
        target_heights=[640, 512],
        workers=4,
        python_executable="/python",
    )

    assert command[0] == "/python"
    assert command[command.index("--split-assignments") + 1] == "official"
    height_index = command.index("--target-heights")
    assert command[height_index + 1 : height_index + 3] == ["512", "640"]
    assert "2048" not in command


def test_read_status_reports_manifest_progress_without_signaling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "export"
    metadata = output_root / "metadata"
    metadata.mkdir(parents=True)
    (metadata / "status.json").write_text(
        json.dumps(
            {
                "status": "running",
                "pid": 123,
                "image_count": 10,
                "started_at": "2026-08-20T00:00:00+00:00",
                "geometry": {
                    "variants": [
                        {"output_height": 64, "output_width": 32},
                        {"output_height": 32, "output_width": 32},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    with (metadata / "manifest.partial.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", "image_id"])
        for index in range(6):
            writer.writerow(["variant", index])
    monkeypatch.setattr(
        global_pad_gui,
        "exporter_process_alive",
        lambda pid: pid is not None and int(pid) == 123,
    )

    status = global_pad_gui.read_global_pad_status(output_root)

    assert status["active"] is True
    assert status["pid"] == 123
    assert status["manifest_rows"] == 6
    assert status["processed_images"] == 3
    assert status["progress"] == pytest.approx(0.3)


def test_gui_launch_is_detached_logged_and_recorded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "vindr"
    (data_root / "images").mkdir(parents=True)
    for filename in ("breast-level_annotations.csv", "finding_annotations.csv"):
        (data_root / filename).write_text("header\n", encoding="utf-8")
    split_path = data_root / "split.csv"
    split_path.write_text("study_id,image_id,export_split\n", encoding="utf-8")
    output_root = tmp_path / "export"
    captured: dict[str, object] = {}

    def fake_popen(command: list[str], **kwargs: object) -> SimpleNamespace:
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(pid=456)

    monkeypatch.setattr(global_pad_gui.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        global_pad_gui,
        "read_global_pad_status",
        lambda _root: {"active": False, "pid": None},
    )

    result = global_pad_gui.launch_global_pad_export(
        data_root=data_root,
        output_root=output_root,
        split_assignments=split_path,
        target_heights=[512, 640],
        workers=2,
        python_executable="/python",
    )

    assert result["pid"] == 456
    assert captured["kwargs"]["start_new_session"] is True
    assert captured["kwargs"]["stdin"] is global_pad_gui.subprocess.DEVNULL
    assert "--overwrite" not in captured["command"]
    assert (output_root / "metadata" / "gui_export.log").exists()
    launch_record = json.loads(
        (output_root / "metadata" / "gui_run.json").read_text(encoding="utf-8")
    )
    assert launch_record["pid"] == 456


def test_gui_refuses_competing_process_for_same_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "vindr"
    (data_root / "images").mkdir(parents=True)
    for filename in ("breast-level_annotations.csv", "finding_annotations.csv"):
        (data_root / filename).write_text("header\n", encoding="utf-8")
    output_root = tmp_path / "export"
    monkeypatch.setattr(
        global_pad_gui,
        "read_global_pad_status",
        lambda _root: {"active": True, "pid": 789},
    )

    with pytest.raises(RuntimeError, match="already active"):
        global_pad_gui.launch_global_pad_export(
            data_root=data_root,
            output_root=output_root,
            split_assignments=None,
            target_heights=[512],
            workers=1,
        )
