"""
Per-file decision records for ClassifierPipeline runs.

A decision record is what one pipeline evaluation concluded about one file:
the action that fired, and every node's matched flag and score along the way.
The pipeline engine builds these dicts while walking the node graph but has
historically discarded them at return -- ``run_pipeline`` returns only a
ClassifierActionType, and ``_dispatch_action`` never sees the node results.
Recording them is what makes a run auditable after the fact: which node routed
a file, how close a score was to its threshold, and which nodes never ran.

Records carry no pipeline config of their own.  The run dump they are written
into already stores the full serialized pipeline alongside them, so repeating a
config digest per file would be redundant -- ``pipeline_name`` is enough to tie
a record back to it.

Records are plain JSON-serializable dicts rather than a dataclass: they are
written straight to the run dump and read back by external tooling, so the
schema is the file format and a class would only add an encode/decode step.
"""

from __future__ import annotations

from typing import Optional


# Schema version for the record dicts below.  External tooling reading a run
# dump should check this before assuming field names; bump it whenever a field
# changes meaning or is removed (adding a field does not require a bump).
DECISION_RECORD_VERSION = 1


def _json_safe_score(score) -> Optional[object]:
    """Normalize a node score for the record.

    Condition evaluators return a float where one is meaningful, but several
    return a descriptive string instead (the matched media type, the filename
    pattern that hit) and most return None.  Strings are kept as-is rather than
    dropped -- they are the only score those conditions have -- and anything
    else becomes None so a record is always JSON-serializable.
    """
    if isinstance(score, bool):
        # bool is a subclass of int; a score of True/False carries no more
        # information than the matched flag already does.
        return None
    if isinstance(score, (int, float, str)):
        return score
    return None


def build_decision_record(
    pipeline_name: str,
    image_path: str,
    action: Optional[object],
    node_results: dict,
    node_scores: dict,
) -> dict:
    """Build one file's decision record.

    *action* is the ClassifierActionType that fired (or None for no action);
    it is stored by value so the record stays JSON-serializable.

    *node_results* and *node_scores* are the runner's own working dicts, keyed
    by node name -- group children included, under their ``"<group>/<child>"``
    keys.  Both are copied, since the runner reuses neither after this call but
    callers should not depend on that.

    Nodes that never ran (the walk halted, jumped past them, or they were
    disabled) are simply absent from the record rather than recorded as
    no-match: "did not run" and "ran and did not match" are different facts and
    conflating them would misrepresent the run.
    """
    verdicts: dict[str, dict] = {}
    for node_name, matched in node_results.items():
        verdicts[node_name] = {
            "matched": bool(matched),
            "score": _json_safe_score(node_scores.get(node_name)),
        }
    return {
        "version": DECISION_RECORD_VERSION,
        "pipeline_name": pipeline_name,
        "path": image_path,
        "action": getattr(action, "value", None),
        "node_verdicts": verdicts,
    }
