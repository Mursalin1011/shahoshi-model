"""Tests for the ESP-IDF export path.

`export.py` is deliberately TensorFlow-free so this can run without a TF
install. The resolver tests encode the specific deployment blocker in the
pre-refactor notebook: eight operators registered by hand against a model that
used thirteen, which fails at AllocateTensors() on the device.
"""

import json

import pytest

from shahoshi.export import (
    RESOLVER_METHODS,
    check_ops,
    device_ops,
    emit_c_array,
    firmware_notes,
    normalization_source,
    resolver_source,
    write_config,
)

# What the pre-refactor notebook's cell 28 actually printed.
BASELINE_OPS = [
    "ADD", "BATCH_TO_SPACE_ND", "CONV_2D", "DELEGATE", "DEPTHWISE_CONV_2D",
    "EXPAND_DIMS", "FULLY_CONNECTED", "MAX_POOL_2D", "MEAN", "MUL", "RESHAPE",
    "SOFTMAX", "SPACE_TO_BATCH_ND",
]

# What the dilation-free Stage 0 model is expected to need.
CLEAN_OPS = [
    "CONV_2D", "DEPTHWISE_CONV_2D", "MAX_POOL_2D", "MEAN", "FULLY_CONNECTED",
    "SOFTMAX", "QUANTIZE", "DEQUANTIZE", "RESHAPE",
]


class TestDeviceOps:
    def test_strips_host_only_delegate(self):
        """DELEGATE is how the host interpreter reports a delegated subgraph.
        There is no such node on device and TFLM has nothing to register."""
        assert "DELEGATE" not in device_ops(BASELINE_OPS)

    def test_deduplicates_and_sorts(self):
        assert device_ops(["MEAN", "CONV_2D", "MEAN"]) == ["CONV_2D", "MEAN"]

    def test_empty_input(self):
        assert device_ops([]) == []


class TestCheckOps:
    def test_flags_nothing_for_the_clean_model(self):
        report = check_ops(CLEAN_OPS)
        assert report["unmapped"] == []
        assert report["not_accelerated"] == []

    def test_flags_the_dilation_ops_as_unaccelerated(self):
        """The reason Stage 0 dropped dilated convolutions: ESP-NN has no int8
        kernel for the space-to-batch pair, so they run as reference C++ while
        the convolutions around them are vectorized."""
        report = check_ops(BASELINE_OPS)
        assert set(report["not_accelerated"]) == {"SPACE_TO_BATCH_ND", "BATCH_TO_SPACE_ND"}

    def test_reports_unmapped_without_raising(self):
        report = check_ops(["CONV_2D", "SOME_NEW_OP"])
        assert report["unmapped"] == ["SOME_NEW_OP"]

    def test_every_baseline_op_is_mapped(self):
        """The old model's ops must all be known, or we cannot even diagnose it."""
        assert check_ops(BASELINE_OPS)["unmapped"] == []


class TestResolverSource:
    def test_template_parameter_matches_the_op_count(self):
        """MicroMutableOpResolver<N> must have N >= the number of registered ops,
        or registration silently fails at runtime."""
        src = resolver_source(CLEAN_OPS)
        assert f"MicroMutableOpResolver<{len(CLEAN_OPS)}>" in src

    def test_registers_every_op_exactly_once(self):
        src = resolver_source(CLEAN_OPS)
        for op in CLEAN_OPS:
            assert f"{RESOLVER_METHODS[op]}();" in src
        assert src.count("resolver.Add") == len(CLEAN_OPS)

    def test_covers_the_baseline_model_the_old_sketch_missed(self):
        """The old sketch registered 8 ops for a 13-op model. Generated from the
        real op list, the resolver covers all 12 device ops."""
        src = resolver_source(BASELINE_OPS)
        assert "MicroMutableOpResolver<12>" in src
        assert "AddSpaceToBatchNd();" in src
        assert "AddBatchToSpaceNd();" in src
        assert "AddMul();" in src
        assert "AddExpandDims();" in src

    def test_does_not_register_the_host_only_delegate(self):
        assert "Delegate" not in resolver_source(BASELINE_OPS)

    def test_warns_about_unaccelerated_ops(self):
        assert "no ESP-NN int8 kernel" in resolver_source(BASELINE_OPS)

    def test_no_warning_for_the_clean_model(self):
        assert "WARNING" not in resolver_source(CLEAN_OPS)

    def test_raises_on_an_unmapped_op_rather_than_skipping_it(self):
        """Silently skipping an op yields a device that fails at AllocateTensors,
        which is far harder to diagnose than a build-time error."""
        with pytest.raises(KeyError, match="SOME_NEW_OP"):
            resolver_source(["CONV_2D", "SOME_NEW_OP"])

    def test_custom_variable_name(self):
        src = resolver_source(["CONV_2D"], var="op_resolver")
        assert "static tflite::MicroMutableOpResolver<1> op_resolver;" in src
        assert "op_resolver.AddConv2D();" in src


class TestEmitCArray:
    def test_writes_both_files(self, tmp_path):
        h, cc = emit_c_array(bytes(range(32)), tmp_path / "model_data")
        assert h.exists() and cc.exists()
        assert h.name == "model_data.h" and cc.name == "model_data.cc"

    def test_header_declares_the_symbol_and_length(self, tmp_path):
        h, _ = emit_c_array(bytes(16), tmp_path / "m", symbol="g_test")
        text = h.read_text()
        assert "extern const unsigned char g_test[];" in text
        assert "extern const unsigned int g_test_len;" in text
        assert "#pragma once" in text

    def test_source_is_sixteen_byte_aligned(self, tmp_path):
        """TFLM reads the flatbuffer in place; an unaligned buffer faults or
        misparses on Xtensa."""
        _, cc = emit_c_array(bytes(16), tmp_path / "m", symbol="g_test")
        assert "alignas(16) const unsigned char g_test[]" in cc.read_text()

    def test_length_matches_the_blob(self, tmp_path):
        _, cc = emit_c_array(bytes(1234), tmp_path / "m", symbol="g_test")
        assert "const unsigned int g_test_len = 1234;" in cc.read_text()

    def test_bytes_round_trip_exactly(self, tmp_path):
        blob = bytes([0x00, 0xFF, 0x7F, 0x80, 0x01, 0xAB])
        _, cc = emit_c_array(blob, tmp_path / "m", symbol="g_test")
        body = cc.read_text().split("{", 1)[1].rsplit("}", 1)[0]
        recovered = bytes(int(t, 16) for t in body.replace(",", " ").split() if t.startswith("0x"))
        assert recovered == blob

    def test_includes_its_own_header(self, tmp_path):
        _, cc = emit_c_array(bytes(8), tmp_path / "movement_model_data")
        assert '#include "movement_model_data.h"' in cc.read_text()

    def test_creates_missing_directories(self, tmp_path):
        h, _ = emit_c_array(bytes(8), tmp_path / "deep" / "nested" / "m")
        assert h.exists()

    def test_empty_blob_does_not_crash(self, tmp_path):
        _, cc = emit_c_array(b"", tmp_path / "m", symbol="g_test")
        assert "g_test_len = 0;" in cc.read_text()


class TestWriteConfig:
    def make(self, tmp_path, **kw):
        args = dict(
            win=128, channels=6, fs=50, stride=64,
            classes=["walk", "sit"], mean=[0.0] * 6, std=[1.0] * 6,
            thresholds={"entropy": 0.5, "mahalanobis": 15.2},
        )
        args.update(kw)
        return json.loads(write_config(tmp_path / "cfg.json", **args).read_text())

    def test_round_trips_the_fields(self, tmp_path):
        cfg = self.make(tmp_path)
        assert cfg["win"] == 128 and cfg["channels"] == 6 and cfg["fs"] == 50
        assert cfg["classes"] == ["walk", "sit"]
        assert cfg["thresholds"]["mahalanobis"] == pytest.approx(15.2)

    def test_derives_hop_and_windows_per_hour(self, tmp_path):
        """The firmware and the calibration must agree on the hop, and the
        alarms-per-hour figure follows from it -- so both are recorded, not
        recomputed by hand at two sites."""
        cfg = self.make(tmp_path)
        assert cfg["hop_seconds"] == pytest.approx(1.28)
        assert cfg["windows_per_hour"] == pytest.approx(2812.5)

    def test_extra_fields_are_merged(self, tmp_path):
        cfg = self.make(tmp_path, extra={"git_sha": "abc123"})
        assert cfg["git_sha"] == "abc123"

    def test_values_are_json_native(self, tmp_path):
        """numpy floats do not serialize; the writer must coerce."""
        import numpy as np

        cfg = self.make(
            tmp_path,
            mean=np.zeros(6, dtype=np.float32),
            std=np.ones(6, dtype=np.float32),
            thresholds={"entropy": np.float32(0.5)},
        )
        assert isinstance(cfg["mean"][0], float)
        assert isinstance(cfg["thresholds"]["entropy"], float)


class TestNormalizationSource:
    def test_emits_both_arrays_with_float_suffixes(self):
        src = normalization_source([0.1] * 6, [1.5] * 6)
        assert "static const float kMean[6]" in src
        assert "static const float kStd[6]" in src
        assert "0.100000f" in src and "1.500000f" in src

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError):
            normalization_source([0.0] * 6, [1.0] * 3)


class TestFirmwareNotes:
    def test_mentions_sizes_and_esp_nn(self):
        text = firmware_notes(
            CLEAN_OPS,
            {"flash_kb": 40.0, "largest_tensor_kb": 4.0, "sum_of_tensors_kb": 60.0},
            "movement_config.json",
        )
        assert "40.0 KB" in text
        assert "arena_used_bytes()" in text
        assert "CONFIG_NN_OPTIMIZATIONS=y" in text

    def test_warns_when_ops_are_unaccelerated(self):
        text = firmware_notes(
            BASELINE_OPS,
            {"flash_kb": 46.8, "largest_tensor_kb": 4.0, "sum_of_tensors_kb": 60.0},
            "movement_config.json",
        )
        assert "no ESP-NN int8 kernel" in text
        assert "SPACE_TO_BATCH_ND" in text
