"""
Unit tests for core infrastructure:
  - StrategyRegistry  (biosieve/core/registry.py)
  - instantiate_strategy  (biosieve/core/factory.py)
  - lazy_import_class / StrategySpec  (biosieve/core/spec.py)
  - load_params / params_for_strategy  (biosieve/io/params.py)
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from biosieve.core.factory import instantiate_strategy
from biosieve.core.registry import StrategyRegistry
from biosieve.core.spec import StrategySpec, lazy_import_class
from biosieve.io.params import load_params, params_for_strategy

# ---------------------------------------------------------------------------
# StrategyRegistry
# ---------------------------------------------------------------------------


class _DummyReducer:
    pass


class _DummyReducer2:
    pass


class _DummySplitter:
    pass


def test_add_and_get_reducer():
    reg = StrategyRegistry()
    reg.add_reducer("dummy", _DummyReducer)
    assert reg.get_reducer_class("dummy") is _DummyReducer


def test_add_and_get_splitter():
    reg = StrategyRegistry()
    reg.add_splitter("dummy", _DummySplitter)
    assert reg.get_splitter_class("dummy") is _DummySplitter


def test_unknown_reducer_raises():
    reg = StrategyRegistry()
    with pytest.raises(KeyError, match="nope"):
        reg.get_reducer_class("nope")


def test_unknown_splitter_raises():
    reg = StrategyRegistry()
    with pytest.raises(KeyError, match="nope"):
        reg.get_splitter_class("nope")


def test_has_reducer_true_false():
    reg = StrategyRegistry()
    reg.add_reducer("r", _DummyReducer)
    assert reg.has_reducer("r") is True
    assert reg.has_reducer("missing") is False


def test_has_splitter_true_false():
    reg = StrategyRegistry()
    reg.add_splitter("s", _DummySplitter)
    assert reg.has_splitter("s") is True
    assert reg.has_splitter("missing") is False


def test_list_reducers_returns_registered():
    reg = StrategyRegistry()
    reg.add_reducer("a", _DummyReducer)
    reg.add_reducer("b", _DummyReducer2)
    listed = reg.list_reducers()
    assert "a" in listed
    assert "b" in listed
    assert "c" not in listed


def test_lazy_spec_resolves_to_real_class():
    reg = StrategyRegistry()
    spec = StrategySpec("exact", "reducer", "biosieve.reduction.exact:ExactDedupReducer")
    reg.add_reducer("exact", spec)
    cls = reg.get_reducer_class("exact")
    from biosieve.reduction.exact import ExactDedupReducer

    assert cls is ExactDedupReducer


def test_full_registry_has_expected_strategies(registry):
    """build_registry() exposes all documented strategies."""
    expected_reducers = {
        "exact",
        "kmer_jaccard",
        "identity_greedy",
        "mmseqs2",
        "embedding_cosine",
        "descriptor_euclidean",
        "structural_distance",
    }
    expected_splitters = {
        "random",
        "stratified",
        "group",
        "time",
        "distance_aware",
        "cluster_aware",
        "homology_aware",
        "random_kfold",
        "stratified_kfold",
        "group_kfold",
        "stratified_numeric",
        "stratified_numeric_kfold",
        "distance_aware_kfold",
    }
    assert expected_reducers.issubset(registry.list_reducers())
    assert expected_splitters.issubset(registry.list_splitters())


# ---------------------------------------------------------------------------
# instantiate_strategy (factory)
# ---------------------------------------------------------------------------


@dataclass
class _Params:
    threshold: float = 0.9
    k: int = 5


def test_instantiate_dataclass_with_valid_params():
    obj = instantiate_strategy(_Params, {"threshold": 0.5})
    assert obj.threshold == 0.5
    assert obj.k == 5  # default preserved


def test_instantiate_dataclass_empty_params_uses_defaults():
    obj = instantiate_strategy(_Params, {})
    assert obj.threshold == 0.9
    assert obj.k == 5


def test_instantiate_dataclass_unknown_param_raises():
    with pytest.raises(ValueError, match="Unknown parameters"):
        instantiate_strategy(_Params, {"nonexistent": 1})


def test_instantiate_non_dataclass_ok():
    class _Plain:
        def __init__(self, x: int = 1):
            self.x = x

    obj = instantiate_strategy(_Plain, {"x": 42})
    assert obj.x == 42


def test_instantiate_non_dataclass_unknown_param_raises():
    class _Plain:
        def __init__(self, x: int = 1):
            self.x = x

    with pytest.raises(ValueError, match="Unknown parameters"):
        instantiate_strategy(_Plain, {"z": 99})


# ---------------------------------------------------------------------------
# lazy_import_class / StrategySpec
# ---------------------------------------------------------------------------


def test_lazy_import_valid_path():
    cls = lazy_import_class("biosieve.reduction.exact:ExactDedupReducer")
    from biosieve.reduction.exact import ExactDedupReducer

    assert cls is ExactDedupReducer


def test_lazy_import_bad_format_raises():
    with pytest.raises(ValueError, match="Invalid import_path"):
        lazy_import_class("biosieve.reduction.exact.ExactDedupReducer")  # missing ':'


def test_lazy_import_bad_module_raises():
    with pytest.raises(ImportError):
        lazy_import_class("biosieve.nonexistent_module:SomeClass")


def test_lazy_import_bad_class_raises():
    with pytest.raises(AttributeError):
        lazy_import_class("biosieve.reduction.exact:NonExistentClass")


# ---------------------------------------------------------------------------
# load_params / params_for_strategy
# ---------------------------------------------------------------------------


def test_load_params_from_json(tmp_path):
    data = {"embedding_cosine": {"threshold": 0.95, "n_jobs": 4}}
    p = tmp_path / "params.json"
    p.write_text(json.dumps(data))
    result = load_params(str(p))
    assert result == data


def test_load_params_none_returns_empty():
    assert load_params(None) == {}


def test_load_params_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_params("/nonexistent/path/params.json")


def test_load_params_unsupported_extension_raises(tmp_path):
    p = tmp_path / "params.txt"
    p.write_text("{}")
    with pytest.raises(ValueError, match="Unsupported"):
        load_params(str(p))


def test_load_params_override_simple():
    result = load_params(None, overrides=["exact.threshold=0.5"])
    assert result["exact"]["threshold"] == 0.5


def test_load_params_override_type_coercion():
    result = load_params(None, overrides=["x.threshold=0.95"])
    assert isinstance(result["x"]["threshold"], float)
    assert result["x"]["threshold"] == pytest.approx(0.95)


def test_load_params_override_bool_coercion():
    result = load_params(None, overrides=["x.flag=true"])
    assert result["x"]["flag"] is True


def test_load_params_override_merges_with_file(tmp_path):
    data = {"exact": {"threshold": 0.9}}
    p = tmp_path / "params.json"
    p.write_text(json.dumps(data))
    result = load_params(str(p), overrides=["exact.threshold=0.5"])
    assert result["exact"]["threshold"] == pytest.approx(0.5)


def test_load_params_override_missing_equals_raises():
    with pytest.raises(ValueError, match="missing '='"):
        load_params(None, overrides=["exact.threshold"])


def test_params_for_strategy_hit():
    all_params = {"random": {"seed": 99, "test_size": 0.3}}
    result = params_for_strategy(all_params, "random")
    assert result == {"seed": 99, "test_size": 0.3}


def test_params_for_strategy_miss_returns_empty():
    result = params_for_strategy({}, "random")
    assert result == {}


def test_params_for_strategy_none_value_returns_empty():
    result = params_for_strategy({"random": None}, "random")
    assert result == {}


def test_params_for_strategy_wrong_type_raises():
    with pytest.raises(ValueError, match="must be a dict"):
        params_for_strategy({"random": "not_a_dict"}, "random")
