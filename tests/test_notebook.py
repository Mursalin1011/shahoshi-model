"""Static checks on the driver notebook.

The notebook is the one artifact in this project that cannot be executed in the
local test environment -- it needs TensorFlow, which has no wheels for the local
Python. So it is the one place where a stale call site survives review, and it
already has twice: a keyword argument passed to a function that does not accept
it, and outputs indexed positionally when the converter had reordered them.

These tests parse the notebook and check every call into `shahoshi` against the
real signatures, and every `cfg.section.field` reference against the real Config.
They cannot prove the notebook runs. They do prove it is not calling things that
do not exist, which is the failure mode that costs a Colab round trip.
"""

from __future__ import annotations

import ast
import json
from dataclasses import fields, is_dataclass
from pathlib import Path

import pytest

from shahoshi.config import Config

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "notebooks" / "01_movement.ipynb"
SRC = REPO / "src" / "shahoshi"

# Modules the notebook imports as `from shahoshi import x` and calls as `x.fn()`.
TRACKED = {
    "augment": SRC / "augment.py",
    "export": SRC / "export.py",
    "fusion": SRC / "fusion.py",
    "manifest": SRC / "manifest.py",
    "quantize": SRC / "quantize.py",
    "scoring": SRC / "scoring.py",
    "splits": SRC / "splits.py",
    "windows": SRC / "windows.py",
    "movement": SRC / "models" / "movement.py",
}


def load_cells():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return nb["cells"]


def code_cells():
    return [
        (i, "".join(c["source"]))
        for i, c in enumerate(load_cells())
        if c["cell_type"] == "code"
    ]


def signatures(path: Path) -> dict[str, set[str]]:
    """Public function name -> accepted keyword-argument names."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: dict[str, set[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            out[node.name] = (
                {a.arg for a in node.args.args}
                | {a.arg for a in node.args.kwonlyargs}
            )
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            out[node.name] = set()   # constructor args not checked
    return out


ALL_SIGS = {name: signatures(path) for name, path in TRACKED.items()}


class TestNotebookIsWellFormed:
    def test_notebook_exists_and_parses(self):
        cells = load_cells()
        assert cells, "notebook has no cells"

    def test_every_code_cell_is_valid_python(self):
        bad = []
        for i, src in code_cells():
            try:
                ast.parse(src)
            except SyntaxError as exc:
                bad.append(f"cell[{i}] line {exc.lineno}: {exc.msg}")
        assert not bad, "\n".join(bad)

    def test_has_a_bootstrap_cell(self):
        assert any("REPO_URL" in src for _, src in code_cells())


class TestCallsMatchSignatures:
    def collect_calls(self):
        """Every `module.function(...)` call into a tracked shahoshi module."""
        found = []
        for i, src in code_cells():
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                fn = node.func
                if not (isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name)):
                    continue
                if fn.value.id not in ALL_SIGS:
                    continue
                found.append((i, fn.value.id, fn.attr, node))
        return found

    def test_notebook_actually_calls_the_package(self):
        """Guard the guard: if this finds nothing, the checks below are vacuous."""
        assert len(self.collect_calls()) >= 10

    def test_called_functions_exist(self):
        missing = [
            f"cell[{i}] {mod}.{fn}() does not exist"
            for i, mod, fn, _ in self.collect_calls()
            if fn not in ALL_SIGS[mod]
        ]
        assert not missing, "\n".join(missing)

    def test_keyword_arguments_are_accepted(self):
        """The exact bug this catches: bounded_relu= passed to compile_model(),
        which does not take it -- a TypeError several cells into a Colab run."""
        bad = []
        for i, mod, fn, node in self.collect_calls():
            accepted = ALL_SIGS[mod].get(fn)
            if accepted is None:
                continue
            for kw in node.keywords:
                if kw.arg and kw.arg not in accepted:
                    bad.append(
                        f"cell[{i}] {mod}.{fn}({kw.arg}=...) -- accepted: "
                        f"{sorted(accepted)}"
                    )
        assert not bad, "\n".join(bad)

    def test_positional_argument_counts_fit(self):
        bad = []
        for i, mod, fn, node in self.collect_calls():
            accepted = ALL_SIGS[mod].get(fn)
            if not accepted:
                continue
            n_pos = len([a for a in node.args if not isinstance(a, ast.Starred)])
            if any(isinstance(a, ast.Starred) for a in node.args):
                continue
            if n_pos > len(accepted):
                bad.append(
                    f"cell[{i}] {mod}.{fn}() got {n_pos} positional args, "
                    f"function takes at most {len(accepted)}"
                )
        assert not bad, "\n".join(bad)


class TestConfigReferences:
    def config_paths(self) -> set[str]:
        """Every `cfg.a.b` attribute chain used in the notebook."""
        found: set[str] = set()
        for _, src in code_cells():
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Attribute):
                    continue
                chain = []
                cur: ast.expr = node
                while isinstance(cur, ast.Attribute):
                    chain.append(cur.attr)
                    cur = cur.value
                if isinstance(cur, ast.Name) and cur.id == "cfg" and len(chain) >= 1:
                    found.add(".".join(reversed(chain)))
        return found

    def test_notebook_reads_the_config(self):
        assert len(self.config_paths()) >= 10

    def test_every_config_reference_exists(self):
        """A typo in cfg.quantize.n_represenative would otherwise surface as an
        AttributeError mid-run, after the training has already happened."""
        cfg = Config()
        bad = []
        for path in sorted(self.config_paths()):
            target = cfg
            for part in path.split("."):
                if not hasattr(target, part):
                    bad.append(f"cfg.{path} -- no attribute {part!r}")
                    break
                target = getattr(target, part)
        assert not bad, "\n".join(bad)

    def test_every_config_field_is_a_real_dataclass_field(self):
        """Sanity check on the checker: the sections it walks are dataclasses."""
        cfg = Config()
        assert is_dataclass(cfg)
        for f in fields(cfg):
            sub = getattr(cfg, f.name)
            if is_dataclass(sub):
                assert fields(sub)


class TestShippedConfigsAreLoadable:
    @pytest.mark.parametrize(
        "name",
        ["movement.yaml", "movement_augmented.yaml", "movement_relu6.yaml"],
    )
    def test_loads_and_validates(self, name):
        Config.load(REPO / "configs" / name).validate()

    def test_relu6_variant_differs_only_in_the_activation(self):
        """An A/B needs exactly one knob changed, or the comparison is confounded."""
        base = Config.load(REPO / "configs" / "movement.yaml")
        relu6 = Config.load(REPO / "configs" / "movement_relu6.yaml")
        assert base.model.bounded_relu is False
        assert relu6.model.bounded_relu is True
        assert base.name != relu6.name
        assert base.quantize.n_representative == relu6.quantize.n_representative
        assert base.augment.times == relu6.augment.times
        assert base.train.deterministic == relu6.train.deterministic
        assert base.data.datasets == relu6.data.datasets
