"""YAML validation rejects discarded settings while preserving safe merge semantics."""

from pathlib import Path
from functools import partial
from types import SimpleNamespace

import pytest
import yaml
from yaml.constructor import ConstructorError

from boltzgen.cli.boltzgen import execute_command
from boltzgen.data.parse.schema import YamlDesignParser, parse_redesign_yaml
from boltzgen.data.parse.yaml import load_yaml
from test_structure_transform_boundaries import molecule


@pytest.mark.parametrize(
    "document",
    [
        "id: A\nid: B\n",
        "file: {structure_groups: [], structure_groups: []}",
        "entities: [{protein: {id: A, id: B, sequence: GG}}]",
        "{id: A, 'id': B}",
        "{true: A, 1: B}",
        "{null: A, ~: B}",
        "{1: A, 01: B}",
        "{=: A, '=': B}",
        "a: &a {id: A, id: B}\nb: {<<: *a}",
        "a: &a {id: A}\nb: {<<: *a, <<: *a}",
        "a: &a {id: A, id: B}\nb: *a",
    ],
)
def test_duplicate_keys_are_rejected(document: str) -> None:
    with pytest.raises(ConstructorError, match="duplicate key"):
        load_yaml(document)


@pytest.mark.parametrize(
    "document",
    [
        "",
        "# comment only\n",
        "null",
        "[]",
        "{}",
        "{id: A, ID: B}",
        "{true: A, 'true': B}",
        "{1: A, '1': B}",
        "- {id: A}\n- {id: B}",
        "a: &a {id: A}\nb: *a",
        "a: &a {id: A, sequence: GG}\nb: {<<: *a, id: B}",
        "a: &a {id: A}\nb: &b {id: B}\nc: {<<: [*a, *b]}",
        "a: &a {id: A}\nb: &b {<<: *a, id: B}\nc: {<<: *b, id: C}",
        "a: &a {id: A}\nb: &b {<<: *a, id: B}\nc: {<<: *b}\nd: {<<: *b}",
        "a: &a {id: A}\nb: {<<: *a, '<<': ordinary}",
        "{=: ordinary}",
    ],
)
def test_safe_yaml_values_and_merge_precedence_are_preserved(document: str) -> None:
    assert load_yaml(document) == yaml.safe_load(document)


def test_recursive_value_alias_retains_identity() -> None:
    result = load_yaml("&a {self: *a}")
    assert result["self"] is result


@pytest.mark.parametrize(
    "document",
    [
        "{[a, b]: value}",
        "{<<: 1}",
        "a: [",
        "a: *undefined",
        "---\na: 1\n---\nb: 2",
        "!!python/name:os.system",
        "!unknown value",
    ],
)
def test_safe_yaml_restrictions_remain(document: str) -> None:
    with pytest.raises(yaml.YAMLError):
        load_yaml(document)


@pytest.mark.parametrize("entrypoint", ["design", "scaffold", "redesign", "steps"])
def test_file_entrypoints_report_duplicate_key_location(
    tmp_path: Path, entrypoint: str
) -> None:
    path = tmp_path / ("steps.yaml" if entrypoint == "steps" else "input.yaml")
    if entrypoint == "design":
        path.write_text(
            "entities:\n  - protein:\n      id: A\n      id: B\n      sequence: GG\n"
        )
        invoke = partial(YamlDesignParser(tmp_path).parse_yaml, path, {}, tmp_path)
        line = 4
    elif entrypoint == "scaffold":
        path.write_text(
            "path: target.pdb\ninclude:\n  - chain:\n      id: A\n      id: B\n"
        )
        invoke = partial(
            YamlDesignParser(tmp_path).parse_file,
            {"file": {"path": str(path)}},
            {},
            tmp_path,
            1,
            tmp_path,
        )
        line = 5
    elif entrypoint == "redesign":
        path.write_text(
            "restrictions:\n  design:\n    - chain:\n        id: A\n        id: B\n"
        )
        invoke = partial(parse_redesign_yaml, path, None)
        line = 5
    else:
        path.write_text("steps: []\nsteps: []\n")
        invoke = partial(execute_command, SimpleNamespace(output=tmp_path, steps=None))
        line = 2
    with pytest.raises(ConstructorError, match="duplicate key") as caught:
        invoke()
    assert str(path) in str(caught.value)
    assert f"line {line}" in str(caught.value)


def test_merged_entities_reach_real_design_parser(tmp_path: Path) -> None:
    path = tmp_path / "input.yaml"
    path.write_text(
        "entities:\n  - protein: &template {id: A, sequence: GG}\n  - protein: {<<: *template, id: B}\n"
    )
    target = YamlDesignParser(tmp_path).parse_yaml(path, {"GLY": molecule()}, tmp_path)
    assert target.structure.chains["name"].tolist() == ["A", "B"]
    assert target.structure.chains["res_num"].tolist() == [2, 2]


def test_repository_examples_remain_valid_yaml() -> None:
    paths = sorted((Path(__file__).parents[1] / "example").rglob("*.yaml"))
    assert paths
    for path in paths:
        with path.open() as stream:
            actual = load_yaml(stream)
        assert actual == yaml.safe_load(path.read_text()), str(path)


def test_validation_does_not_modify_pyyaml_global_loader() -> None:
    with pytest.raises(ConstructorError):
        load_yaml("id: A\nid: B")
    assert yaml.safe_load("id: A\nid: B") == {"id": "B"}
