#!/usr/bin/env python3
"""
Explain the routing decisions recorded in a classifier pipeline run dump.

A run dump carries one decision per evaluated file, with every node's verdict
(matched, score), when the pipeline sets record_node_verdicts. This script
derives everything from the pipeline stored in the dump, so it applies to any
pipeline built from vote nodes (nodes that only record a verdict) and routing
nodes (nodes whose outcome ends the run).

Reports:
  - the node whose outcome ended the run, per file;
  - node_result references that can never hold;
  - votes whose verdict never changed across the run (always no-match, or
    always match);
  - for each routing node, how often each of its required votes held, and
    how often a file missed the route by exactly one vote;
  - for routing nodes that fire when a combined vote fails, which of its
    votes failed;
  - score distributions of embedding and classifier-rank votes next to their
    thresholds, to calibrate them from.

A route's requirements are the node_result conditions of its condition. A
referenced node whose condition is an AND of node_result conditions, or an
AND group, is expanded into those votes; other conditions inside a route are
not recorded separately and are counted as unrecorded. GOTO outcomes are not
followed: files are assumed to visit nodes in list order.

Reads the JSON file only; imports nothing from the app.

Usage:
  python scripts/analyze_consensus_dump.py <pipeline_run_...json> [--top N]
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter

TERMINAL_OUTCOMES = {"EXECUTE", "ACCEPT", "REJECT"}


# ---------------------------------------------------------------------------
# Pipeline structure
# ---------------------------------------------------------------------------

def _outcome_type(outcome) -> str:
    return (outcome or {}).get("outcome_type") or "CONTINUE"


def _is_terminal(outcome) -> bool:
    return _outcome_type(outcome) in TERMINAL_OUTCOMES


def _describe_outcome(outcome) -> str:
    kind = _outcome_type(outcome)
    modifier = (outcome or {}).get("action_modifier") or ""
    action = (outcome or {}).get("action_type")
    if kind == "EXECUTE":
        return f"{action} {modifier}".strip()
    return kind


def _node_result_refs(condition) -> list:
    """(node_name, expected) of every node_result in *condition*, through
    composites. Group children are separate verdicts and are not searched."""
    if not condition:
        return []
    if condition.get("condition_type") == "node_result":
        return [(condition.get("node_name"), bool(condition.get("expected_result", True)))]
    refs = []
    for sub in condition.get("sub_conditions") or []:
        refs += _node_result_refs(sub)
    return refs


class Pipeline:
    def __init__(self, pipeline: dict):
        self.nodes = pipeline.get("nodes") or []
        self.by_name = {n["name"]: n for n in self.nodes}
        self.routes = [n for n in self.nodes
                       if _is_terminal(n.get("on_match")) or _is_terminal(n.get("on_no_match"))]

    def expand(self, name: str, seen=()) -> "list | None":
        """The votes a True verdict of node *name* consists of, as verdict
        keys, or None when the node is itself the vote."""
        node = self.by_name.get(name)
        if node is None or name in seen:
            return None
        cond = node.get("condition") or {}
        kind = cond.get("condition_type")
        if kind == "group" and cond.get("operator") == "AND" and cond.get("nodes"):
            return [f"{name}/{child['name']}" for child in cond["nodes"]]
        if kind == "composite" and cond.get("operator") == "AND":
            subs = cond.get("sub_conditions") or []
            if not subs or any(s.get("condition_type") != "node_result"
                               or not s.get("expected_result", True) for s in subs):
                return None
            leaves = []
            for s in subs:
                leaves += self.expand(s["node_name"], (*seen, name)) or [s["node_name"]]
            return leaves
        return None

    def requirements(self, route: dict) -> "tuple[list, int]":
        """(requirements, unrecorded) for *route* to match: requirements are
        (verdict key, expected) pairs; unrecorded counts conditions inside the
        route that have no verdict of their own."""
        cond = route.get("condition") or {}
        kind = cond.get("condition_type")
        if kind == "node_result":
            subs = [cond]
        elif kind == "composite" and cond.get("operator") == "AND":
            subs = cond.get("sub_conditions") or []
        else:
            return [], 0
        reqs, unrecorded = [], 0
        for s in subs:
            if s.get("condition_type") != "node_result":
                unrecorded += 1
                continue
            name, expected = s.get("node_name"), bool(s.get("expected_result", True))
            leaves = self.expand(name) if expected else None
            reqs += [(leaf, True) for leaf in leaves] if leaves else [(name, expected)]
        return reqs, unrecorded

    def thresholds(self, condition_type: str) -> dict:
        """Verdict key -> condition, for every *condition_type* condition that
        is a node's whole condition or a group child's."""
        found = {}
        for node in self.nodes:
            cond = node.get("condition") or {}
            if cond.get("condition_type") == condition_type:
                found[node["name"]] = cond
            for child in cond.get("nodes") or []:
                child_cond = child.get("condition") or {}
                if child_cond.get("condition_type") == condition_type:
                    found[f"{node['name']}/{child['name']}"] = child_cond
        return found


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------

def _matched(verdicts: dict, key: str) -> "bool | None":
    v = verdicts.get(key)
    return None if v is None else bool(v.get("matched"))


def _label(requirement) -> str:
    key, expected = requirement
    return key if expected else f"not {key}"


def _ended_by(pipeline: Pipeline, verdicts: dict) -> "tuple[str, str]":
    """(node name, outcome) that ended the run for this file."""
    for node in pipeline.nodes:
        matched = _matched(verdicts, node["name"])
        if matched is None:
            continue
        outcome = node.get("on_match") if matched else node.get("on_no_match")
        if _is_terminal(outcome):
            return node["name"], _describe_outcome(outcome)
    return "(no node ended the run)", "default action"


def _quantiles(values: list) -> str:
    if not values:
        return "no values"
    values = sorted(values)

    def q(p):
        return values[min(len(values) - 1, int(p * (len(values) - 1)))]
    return (f"n={len(values)} min={values[0]:.3f} p10={q(0.1):.3f} p50={q(0.5):.3f} "
            f"p90={q(0.9):.3f} p99={q(0.99):.3f} max={values[-1]:.3f} mean={statistics.mean(values):.3f}")


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def report_ended_by(pipeline, decisions):
    ended = Counter(_ended_by(pipeline, d.get("node_verdicts") or {}) for d in decisions)
    print("== Node that ended the run")
    for (name, outcome), count in ended.most_common():
        print(f"   {count:6d}  {name}  -> {outcome}")
    ended_names = {name for name, _outcome in ended}
    never = [n["name"] for n in pipeline.routes if n["name"] not in ended_names]
    if never:
        print("   never ended a run: " + ", ".join(never))
    print()
    return ended_names


def report_dead_references(pipeline):
    dead = []
    for node in pipeline.nodes:
        for name, expected in _node_result_refs(node.get("condition")):
            ref = pipeline.by_name.get(name)
            if ref is None:
                continue
            # A later node only runs when the referenced one did not end the run.
            ends_run = ref.get("on_match") if expected else ref.get("on_no_match")
            if _is_terminal(ends_run):
                dead.append((node["name"], _label((name, expected))))
    if dead:
        print("== node_result references that can never hold (that verdict ends the run)")
        for node, ref in dead:
            print(f"   {node!r} requires {ref!r}")
        print()


def report_constant_votes(pipeline, decisions):
    route_names = {n["name"] for n in pipeline.routes}
    seen: dict = {}
    for d in decisions:
        for key, v in (d.get("node_verdicts") or {}).items():
            if key in route_names:
                continue
            seen.setdefault(key, set()).add(bool(v.get("matched")))
    never = sorted(k for k, values in seen.items() if values == {False})
    always = sorted(k for k, values in seen.items() if values == {True})
    if never or always:
        print("== Votes whose verdict never changed (routes needing the other verdict are unreachable)")
        for key in never:
            print(f"   never matched:  {key}")
        for key in always:
            print(f"   always matched: {key}")
        print()


def report_routes(pipeline, decisions, top):
    for route in pipeline.routes:
        reqs, unrecorded = pipeline.requirements(route)
        if not reqs:
            continue
        missing_count, one_short = Counter(), Counter()
        reached = evaluated = 0
        for d in decisions:
            verdicts = d.get("node_verdicts") or {}
            if any(_matched(verdicts, key) is None for key, _ in reqs):
                continue  # ended before these votes ran
            evaluated += 1
            missing = [_label(r) for r in reqs if _matched(verdicts, r[0]) is not r[1]]
            if not missing:
                reached += 1
                continue
            missing_count.update(missing)
            if len(missing) == 1:
                one_short[missing[0]] += 1
        extra = f" (+{unrecorded} unrecorded condition(s))" if unrecorded else ""
        print(f"== {route['name']} -> {_describe_outcome(route.get('on_match'))}: every recorded "
              f"requirement held in {reached} of {evaluated} files that reached them{extra}")
        for label, count in one_short.most_common(top):
            print(f"   missed by only this: {label:40s} {count}")
        for label, count in missing_count.most_common(top):
            print(f"   not held (any file): {label:40s} {count}")
        print()


def report_failed_votes(pipeline, decisions, ended, top):
    """For routes requiring a combined vote to fail, which of its votes failed."""
    for route in pipeline.routes:
        if route["name"] not in ended:
            continue
        for name, expected in _node_result_refs(route.get("condition")):
            leaves = None if expected else pipeline.expand(name)
            if not leaves:
                continue
            combos = Counter()
            for d in decisions:
                verdicts = d.get("node_verdicts") or {}
                if _ended_by(pipeline, verdicts)[0] != route["name"]:
                    continue
                combos[tuple(k for k in leaves if _matched(verdicts, k) is False)] += 1
            print(f"== {route['name']}: which votes of {name!r} failed")
            for combo, count in combos.most_common(top):
                print(f"   {count:6d}  {', '.join(combo) or '(none recorded)'}")
            print()


def report_scores(pipeline, decisions):
    def scores(key):
        return [v["score"] for d in decisions
                for v in [(d.get("node_verdicts") or {}).get(key) or {}]
                if isinstance(v.get("score"), (int, float))]

    embeddings = pipeline.thresholds("embedding")
    if embeddings:
        print("== Embedding votes (mean cosine similarity over the prompts)")
        for key, cond in embeddings.items():
            values = scores(key)
            if values:
                threshold = cond.get("threshold")
                print(f"   {key:40s} {_quantiles(values)}  > {threshold}: "
                      f"{sum(1 for x in values if x > threshold)}")
        print()

    ranks = pipeline.thresholds("classifier_rank")
    if ranks:
        # The score is the highest-ranked listed category's in the rank
        # window, reaching min_confidence or not. Dumps from older app
        # versions carry it only when it reached min_confidence.
        print("== Classifier-rank votes (score of the highest-ranked listed category in the window)")
        for key, cond in ranks.items():
            values = scores(key)
            window = f"ranks {cond.get('min_rank', 1)}-{cond.get('max_rank', 1)}"
            negate = " negated" if cond.get("negate") else ""
            print(f"   {key:40s} {_quantiles(values)}  ({cond.get('classifier_name')}, {window}, "
                  f">= {cond.get('min_confidence')}{negate})")
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dump", help="pipeline_run_*.json from the Weidr log directory")
    parser.add_argument("--top", type=int, default=10,
                        help="Rows shown per breakdown (default: %(default)s).")
    args = parser.parse_args()

    with open(args.dump, "r", encoding="utf-8") as f:
        dump = json.load(f)
    decisions = [d for d in dump.get("decisions") or [] if d.get("node_verdicts")]
    if not decisions:
        print("The dump has no decisions (record_node_verdicts off, or no file was evaluated).")
        return 1
    pipeline = Pipeline(dump.get("pipeline") or {})
    print(f"{len(decisions)} decisions, actions: "
          f"{dict(Counter(str(d.get('action')) for d in decisions))}\n")

    ended = report_ended_by(pipeline, decisions)
    report_dead_references(pipeline)
    report_constant_votes(pipeline, decisions)
    report_routes(pipeline, decisions, args.top)
    report_failed_votes(pipeline, decisions, ended, args.top)
    report_scores(pipeline, decisions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
