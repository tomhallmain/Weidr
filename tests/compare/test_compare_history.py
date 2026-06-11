"""CompareHistory / CompareRunSettings persistence."""

from compare.compare_args import CompareArgs
from compare.compare_history import CompareHistory, CompareRunSettings
from compare.compare_manager import CompareManager, CombinationLogic
from utils.constants import CompareMode


def test_run_settings_round_trip_json():
    rs = CompareRunSettings(
        overwrite=True,
        store_checkpoints=True,
        use_matrix_comparison=False,
        threshold=0.88,
        counter_limit=5000,
    )
    assert CompareRunSettings.from_json(rs.to_json()) == rs


def test_history_from_json_legacy_top_level_matrix_flag():
    h = CompareHistory.from_json({
        "directory": "/tmp/pics",
        "timestamp": "2026-01-01T00:00:00",
        "instances": [],
        "combination_logic": "AND",
        "filter_dict": None,
        "use_matrix_comparison": False,
    })
    assert h is not None
    assert h.run_settings.use_matrix_comparison is False


def test_manager_snapshot_restores_run_settings():
    mgr = CompareManager()
    mgr.set_overwrite(True)
    mgr.set_store_checkpoints(True)
    mgr.set_use_matrix_comparison(False)
    mgr.set_threshold(0.91)
    mgr.set_counter_limit(1234)

    snap = mgr.snapshot(CompareArgs(base_dir="/data/photos"))
    assert snap.run_settings.use_matrix_comparison is False
    assert snap.run_settings.threshold == 0.91
    assert snap.run_settings.counter_limit == 1234

    mgr2 = CompareManager()
    mgr2.apply_snapshot(snap)
    assert mgr2.get_overwrite() is True
    assert mgr2.get_store_checkpoints() is True
    assert mgr2.get_use_matrix_comparison() is False
    assert mgr2.get_threshold() == 0.91
    assert mgr2.get_counter_limit() == 1234


def test_file_filter_serialises_under_new_key():
    h = CompareHistory(
        directory="/tmp",
        timestamp="2026-01-01T00:00:00",
        instances=[],
        combination_logic="AND",
        filter_dict=None,
        file_filter="cats",
    )
    d = h.to_json()
    assert d["file_filter"] == "cats"
    assert "inclusion_pattern" not in d


def test_file_filter_round_trips_via_from_json():
    h = CompareHistory(
        directory="/tmp",
        timestamp="2026-01-01T00:00:00",
        instances=[],
        combination_logic="AND",
        filter_dict=None,
        file_filter="cats;!_edit",
    )
    h2 = CompareHistory.from_json(h.to_json())
    assert h2.file_filter == "cats;!_edit"


def test_legacy_inclusion_pattern_key_loads_correctly():
    d = {
        "directory": "/tmp",
        "timestamp": "2026-01-01T00:00:00",
        "instances": [],
        "combination_logic": "AND",
        "filter_dict": None,
        "inclusion_pattern": "cats",
    }
    h = CompareHistory.from_json(d)
    assert h is not None
    assert h.file_filter == "cats"


def test_file_filter_key_takes_precedence_over_legacy_key():
    d = {
        "directory": "/tmp",
        "timestamp": "2026-01-01T00:00:00",
        "instances": [],
        "combination_logic": "AND",
        "filter_dict": None,
        "file_filter": "new_value",
        "inclusion_pattern": "old_value",
    }
    h = CompareHistory.from_json(d)
    assert h.file_filter == "new_value"


def test_history_json_identity_includes_run_settings():
    h1 = CompareHistory(
        directory="/a",
        timestamp="t1",
        instances=[{"compare_mode": CompareMode.CLIP_EMBEDDING.value}],
        combination_logic=CombinationLogic.AND.value,
        filter_dict=None,
        run_settings=CompareRunSettings(use_matrix_comparison=False),
    )
    h2 = CompareHistory(
        directory="/a",
        timestamp="t2",
        instances=[{"compare_mode": CompareMode.CLIP_EMBEDDING.value}],
        combination_logic=CombinationLogic.AND.value,
        filter_dict=None,
        run_settings=CompareRunSettings(use_matrix_comparison=True),
    )
    assert h1._identity_key() != h2._identity_key()
