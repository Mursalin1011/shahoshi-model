"""Tests for config loading and run manifests.

The validation tests matter because these are the failures that otherwise show
up as a wasted training run: a fall head with no fall data trains on nothing but
masked rows and reports a perfect-looking loss of zero.
"""

import json
from pathlib import Path

import pytest
import yaml

from shahoshi import manifest
from shahoshi.config import Config

REPO = Path(__file__).resolve().parents[1]


class TestShippedConfigs:
    @pytest.mark.parametrize("name", ["movement.yaml", "movement_augmented.yaml"])
    def test_they_load_and_validate(self, name):
        cfg = Config.load(REPO / "configs" / name)
        cfg.validate()
        assert cfg.name

    def test_baseline_has_augmentation_off(self):
        assert Config.load(REPO / "configs" / "movement.yaml").augment.times == 0

    def test_augmented_variant_has_it_on(self):
        assert Config.load(REPO / "configs" / "movement_augmented.yaml").augment.times == 2

    def test_hop_matches_the_firmware_arithmetic(self):
        cfg = Config.load(REPO / "configs" / "movement.yaml")
        assert cfg.data.hop_seconds == pytest.approx(1.28)


class TestConfigLoad:
    def write(self, tmp_path, raw):
        p = tmp_path / "c.yaml"
        p.write_text(yaml.safe_dump(raw), encoding="utf-8")
        return p

    def test_defaults_fill_in_omitted_sections(self, tmp_path):
        cfg = Config.load(self.write(tmp_path, {"name": "x"}))
        assert cfg.data.win == 128
        assert cfg.train.monitor == "val_macro_f1"

    def test_nested_values_are_read(self, tmp_path):
        cfg = Config.load(self.write(tmp_path, {"data": {"win": 256, "stride": 128}}))
        assert cfg.data.win == 256 and cfg.data.stride == 128

    def test_empty_file_gives_all_defaults(self, tmp_path):
        p = tmp_path / "c.yaml"
        p.write_text("", encoding="utf-8")
        assert Config.load(p).data.win == 128

    def test_dotted_overrides_apply(self, tmp_path):
        cfg = Config.load(
            self.write(tmp_path, {"name": "x"}),
            **{"augment.times": 3, "train.epochs": 5},
        )
        assert cfg.augment.times == 3 and cfg.train.epochs == 5

    def test_unknown_top_level_key_is_rejected(self, tmp_path):
        """A typo in a config key must not silently do nothing -- that is a
        wasted training run that looks like it worked."""
        with pytest.raises(ValueError, match="unknown config key"):
            Config.load(self.write(tmp_path, {"epochs": 10}))

    def test_unknown_nested_key_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="unknown key"):
            Config.load(self.write(tmp_path, {"train": {"learning_rate": 0.1}}))

    def test_unknown_dotted_override_is_rejected(self, tmp_path):
        with pytest.raises(KeyError):
            Config.load(self.write(tmp_path, {}), **{"train.lr_schedule": "cosine"})

    def test_round_trips_through_save(self, tmp_path):
        cfg = Config.load(self.write(tmp_path, {"name": "rt", "data": {"win": 64}}))
        out = cfg.save(tmp_path / "out.yaml")
        assert Config.load(out).data.win == 64

    def test_to_dict_is_json_serializable(self, tmp_path):
        cfg = Config.load(self.write(tmp_path, {"name": "x"}))
        json.dumps(cfg.to_dict())  # must not raise


class TestConfigValidate:
    def test_rejects_unknown_split_protocol(self):
        cfg = Config()
        cfg.split.protocol = "random"
        with pytest.raises(ValueError, match="split.protocol"):
            cfg.validate()

    def test_random_window_split_is_not_even_an_option(self):
        """Window-level splitting is the single most common way HAR results get
        inflated, so it is absent from the vocabulary rather than discouraged."""
        cfg = Config()
        cfg.split.protocol = "window"
        with pytest.raises(ValueError):
            cfg.validate()

    def test_leave_dataset_out_requires_naming_the_corpus(self):
        cfg = Config()
        cfg.split.protocol = "leave_dataset_out"
        with pytest.raises(ValueError, match="test_dataset"):
            cfg.validate()

    def test_leave_dataset_out_accepts_a_named_corpus(self):
        cfg = Config()
        cfg.split.protocol = "leave_dataset_out"
        cfg.split.test_dataset = "sisfall"
        cfg.validate()

    def test_rejects_stride_larger_than_window(self):
        cfg = Config()
        cfg.data.stride = 256
        with pytest.raises(ValueError, match="skip samples"):
            cfg.validate()

    def test_rejects_fall_head_without_a_fall_corpus(self):
        """Otherwise the fall head trains on nothing but masked rows and reports
        a loss of zero, which reads as success."""
        cfg = Config()
        cfg.model.with_fall_head = True
        with pytest.raises(ValueError, match="fall labels"):
            cfg.validate()

    def test_accepts_fall_head_once_sisfall_is_loaded(self):
        cfg = Config()
        cfg.model.with_fall_head = True
        cfg.data.datasets = ["uci", "ms", "sisfall"]
        cfg.validate()

    def test_rejects_non_positive_export_far(self):
        cfg = Config()
        cfg.scoring.export_far = 0.0
        with pytest.raises(ValueError, match="export_far"):
            cfg.validate()


class TestManifest:
    def test_writes_a_readable_json_file(self, tmp_path):
        p = manifest.write(
            tmp_path, "run-a", {"data": {"win": 128}}, {"accuracy": 0.81},
            timestamp="20260101T000000Z",
        )
        payload = json.loads(p.read_text())
        assert payload["name"] == "run-a"
        assert payload["metrics"]["accuracy"] == pytest.approx(0.81)
        assert payload["config"]["data"]["win"] == 128

    def test_records_git_and_environment(self, tmp_path):
        p = manifest.write(tmp_path, "r", {}, {}, timestamp="20260101T000000Z")
        payload = json.loads(p.read_text())
        assert "git" in payload and "commit" in payload["git"]
        assert payload["environment"]["numpy"] is not None

    def test_coerces_numpy_scalars(self, tmp_path):
        import numpy as np

        p = manifest.write(
            tmp_path, "r",
            {"x": np.float32(1.5)},
            {"acc": np.float64(0.9), "cm": np.zeros((2, 2))},
            timestamp="20260101T000000Z",
        )
        payload = json.loads(p.read_text())
        assert payload["metrics"]["acc"] == pytest.approx(0.9)
        assert payload["metrics"]["cm"] == [[0, 0], [0, 0]]

    def test_repeated_runs_accumulate_rather_than_overwrite(self, tmp_path):
        manifest.write(tmp_path, "r", {}, {"a": 1}, timestamp="20260101T000000Z")
        manifest.write(tmp_path, "r", {}, {"a": 2}, timestamp="20260101T000001Z")
        assert len(list(tmp_path.glob("*.json"))) == 2

    def test_extra_payload_is_kept(self, tmp_path):
        p = manifest.write(
            tmp_path, "r", {}, {}, extra={"ops": ["CONV_2D"]},
            timestamp="20260101T000000Z",
        )
        assert json.loads(p.read_text())["extra"]["ops"] == ["CONV_2D"]

    def test_collect_returns_newest_first(self, tmp_path):
        manifest.write(tmp_path, "old", {}, {}, timestamp="20260101T000000Z")
        manifest.write(tmp_path, "new", {}, {}, timestamp="20260102T000000Z")
        assert [r["name"] for r in manifest.collect(tmp_path)] == ["new", "old"]

    def test_collect_on_missing_directory(self, tmp_path):
        assert manifest.collect(tmp_path / "nope") == []

    def test_collect_skips_unreadable_files(self, tmp_path):
        manifest.write(tmp_path, "good", {}, {"a": 1}, timestamp="20260101T000000Z")
        (tmp_path / "20260102T000000Z_broken.json").write_text("{not json", encoding="utf-8")
        assert [r["name"] for r in manifest.collect(tmp_path)] == ["good"]

    def test_compare_builds_a_table(self, tmp_path):
        manifest.write(tmp_path, "a", {}, {"acc": 0.81}, timestamp="20260101T000000Z")
        manifest.write(tmp_path, "b", {}, {"acc": 0.85}, timestamp="20260102T000000Z")
        table = manifest.compare(tmp_path)
        assert "acc" in table and "0.8100" in table and "0.8500" in table

    def test_compare_with_no_runs(self, tmp_path):
        assert "no manifests" in manifest.compare(tmp_path)

    def test_compare_tolerates_a_missing_metric(self, tmp_path):
        manifest.write(tmp_path, "a", {}, {"acc": 0.8}, timestamp="20260101T000000Z")
        manifest.write(tmp_path, "b", {}, {"other": 1.0}, timestamp="20260102T000000Z")
        assert "-" in manifest.compare(tmp_path)
