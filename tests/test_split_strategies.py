from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from vindr_mammo.dash_app import _filter_records
from vindr_mammo.export import make_train_val_test_split, normalize_split_strategy_kwargs
from vindr_mammo.gui_app import _load_split_records


def _records() -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for study_index in range(6):
        for view_index in range(2):
            records.append(
                {
                    "study_id": f"train-{study_index}",
                    "image_id": f"train-{study_index}-{view_index}",
                    "split": "training",
                    "breast_birads": str(1 + study_index % 3),
                }
            )
    for view_index in range(2):
        records.append(
            {
                "study_id": "official-test",
                "image_id": f"test-{view_index}",
                "split": "test",
                "breast_birads": "4",
            }
        )
    return records


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (
            {
                "strategy": "official_only",
                "val_fraction_from_training": 0.9,
                "validation_study_count": 4,
                "validation_image_count": 8,
            },
            (0.0, 0, 0),
        ),
        (
            {
                "strategy": "random_study_fraction",
                "val_fraction_from_training": 0.5,
                "validation_study_count": 4,
                "validation_image_count": 8,
            },
            (0.5, None, None),
        ),
        (
            {
                "strategy": "exact_study_count",
                "val_fraction_from_training": 0.9,
                "validation_study_count": 2,
                "validation_image_count": 4,
            },
            (0.0, 2, 4),
        ),
    ],
)
def test_explicit_split_strategies_are_mutually_exclusive(config, expected) -> None:
    kwargs = normalize_split_strategy_kwargs(config)
    assert (
        kwargs["val_fraction"],
        kwargs["validation_study_count"],
        kwargs["validation_image_count"],
    ) == expected


def test_legacy_split_strategy_inference_prefers_exact_counts() -> None:
    exact = normalize_split_strategy_kwargs(
        {
            "val_fraction_from_training": 0.9,
            "validation_study_count": 2,
            "validation_image_count": 4,
        }
    )
    random_fraction = normalize_split_strategy_kwargs(
        {"val_fraction_from_training": 0.5}
    )
    official = normalize_split_strategy_kwargs(
        {"val_fraction_from_training": 0.0}
    )

    assert exact["validation_study_count"] == 2
    assert random_fraction["val_fraction"] == 0.5
    assert random_fraction["validation_study_count"] is None
    assert official["validation_study_count"] == 0


def test_strategies_preserve_official_test_and_study_grouping() -> None:
    records = _records()
    expected_val_images = {
        "official_only": 0,
        "random_study_fraction": 6,
        "exact_study_count": 4,
    }
    configs = {
        "official_only": {"strategy": "official_only"},
        "random_study_fraction": {
            "strategy": "random_study_fraction",
            "val_fraction_from_training": 0.5,
            "seed": 7,
        },
        "exact_study_count": {
            "strategy": "exact_study_count",
            "validation_study_count": 2,
            "validation_image_count": 4,
            "seed": 7,
        },
    }

    for strategy, config in configs.items():
        splits, _ = make_train_val_test_split(
            records, **normalize_split_strategy_kwargs(config)
        )
        train_studies = {row["study_id"] for row in splits["train"]}
        val_studies = {row["study_id"] for row in splits["val"]}
        assert len(splits["val"]) == expected_val_images[strategy]
        assert not train_studies & val_studies
        assert {row["study_id"] for row in splits["test"]} == {"official-test"}


def test_gui_split_loader_honors_exact_count_strategy() -> None:
    dataset = SimpleNamespace(image_records=_records())
    splits, table = _load_split_records(
        dataset,
        {
            "splits": {
                "strategy": "exact_study_count",
                "validation_study_count": 2,
                "validation_image_count": 4,
                "seed": 11,
            }
        },
    )

    assert len(splits["val"]) == 4
    assert len({row["study_id"] for row in splits["val"]}) == 2
    assert int((table["export_split"] == "test").sum()) == 2


def test_exact_strategy_requires_study_count() -> None:
    with pytest.raises(ValueError, match="requires validation_study_count"):
        normalize_split_strategy_kwargs(
            {"strategy": "exact_study_count", "validation_image_count": 4}
        )


def test_dash_source_filter_uses_current_split_controls() -> None:
    frame = pd.DataFrame(_records())
    frame["export_split"] = "train"
    frame["has_mass"] = True

    common = {
        "filter_split": "val",
        "filter_positive": "all images",
        "split_val_fraction": 0.5,
        "split_validation_study_count": 0,
        "split_validation_image_count": 0,
        "split_seed": 7,
        "split_stratify_birads": [],
    }
    official = _filter_records(frame, {**common, "split_strategy": "official_only"})
    random_fraction = _filter_records(
        frame, {**common, "split_strategy": "random_study_fraction"}
    )

    assert official.empty
    assert len(random_fraction) == 6
    assert len(set(random_fraction["study_id"])) == 3
