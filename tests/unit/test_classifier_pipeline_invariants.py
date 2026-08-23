"""
Structural invariants of the three-module classifier pipeline model.

The condition data model, the node model, and the pipeline object live in
separate modules (``classifier_pipeline_conditions``,
``classifier_pipeline_nodes``, ``classifier_pipeline``) with a deliberately
acyclic import graph: conditions imports nothing from compare/, nodes imports
only conditions, and the pipeline module imports both and re-exports their
names so existing call sites keep resolving.

Two properties hold that up, and neither is checked by the per-condition
behaviour tests in test_classifier_pipeline.py:

1. Every condition type is deserializable.  ``_condition_from_dict`` is a
   hand-maintained branch table, so a condition class added without a matching
   branch raises ValueError at load time -- which ``ClassifierPipelines.load``
   catches and logs, meaning a stored pipeline silently disappears with only a
   log line.  The tests below drive the table from the condition classes
   themselves, so a new type is covered the moment it is defined.
2. The import graph stays acyclic.  Conditions and nodes are mutually recursive
   as data (a GroupCondition holds child PipelineNodes; a PipelineNode holds a
   condition), and the only thing keeping that from becoming an import cycle is
   the rule that the conditions module imports nothing from compare/ -- a rule a
   docstring alone cannot enforce.
"""

import ast
import dataclasses
from pathlib import Path

import pytest

import compare.classifier_pipeline_conditions as conditions_module
import compare.classifier_pipeline_nodes as nodes_module

# Deliberately imported from the parent module rather than from
# classifier_pipeline_nodes where it is defined: call sites across the app and
# test suite reach it through this re-export, so importing it this way keeps the
# re-export itself covered.
from compare.classifier_pipeline import _condition_from_dict


def _all_condition_classes():
    """Every condition dataclass, discovered by its ``condition_type`` marker.

    Discovered rather than listed so a newly added condition type is picked up
    without editing this file -- the point of these tests is to cover types
    nobody remembered to write a test for.
    """
    found = [
        obj
        for name in dir(conditions_module)
        if isinstance(obj := getattr(conditions_module, name), type)
        and dataclasses.is_dataclass(obj)
        and isinstance(getattr(obj, "condition_type", None), str)
    ]
    return sorted(found, key=lambda cls: cls.condition_type)


def _imported_compare_modules(module) -> set:
    """Names of compare/ modules *module* imports, including inside functions.

    Parses the source rather than inspecting the imported module so that
    function-local imports are caught too: deferring the import is exactly how a
    cycle between conditions and nodes would be reintroduced without any
    top-level import to grep for.
    """
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if (node.module or "").startswith("compare"):
                imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(
                alias.name for alias in node.names if alias.name.startswith("compare")
            )
    return imported


class TestConditionDeserializerCompleteness:
    def test_discovery_is_not_silently_empty(self):
        """Guards the parametrized test below.

        If discovery ever returns nothing, parametrize yields an empty list and
        every branch check vanishes without a single failure.
        """
        discovered = {cls.condition_type for cls in _all_condition_classes()}
        assert {"embedding", "group", "composite"} <= discovered

    @pytest.mark.parametrize(
        "condition_cls",
        _all_condition_classes(),
        ids=lambda cls: cls.condition_type,
    )
    def test_every_condition_type_round_trips(self, condition_cls):
        """A default-constructed condition survives to_dict -> _condition_from_dict.

        Fails with ValueError("Unknown condition type") when a condition class
        exists with no branch in the deserializer.
        """
        restored = _condition_from_dict(condition_cls().to_dict())
        assert isinstance(restored, condition_cls)
        assert restored.condition_type == condition_cls.condition_type


class TestPipelineModuleImportGraph:
    def test_conditions_module_imports_nothing_from_compare(self):
        """The conditions module is the base of the graph and must stay there.

        A condition type needing a runtime dependency should resolve it at
        evaluation time in the runner instead of importing it here.
        """
        assert _imported_compare_modules(conditions_module) == set()

    def test_nodes_module_imports_only_conditions(self):
        """Nodes may depend on conditions and nothing else in compare/.

        In particular not on classifier_pipeline itself, which imports nodes.
        """
        assert _imported_compare_modules(nodes_module) == {
            "compare.classifier_pipeline_conditions"
        }
