import csv

import pytest

import evaluate

METRIC_PREFIXES = ("accuracy", "macro_f1", "f1_")


def _metrics(path):
    with open(path, newline="") as f:
        return {(r["model"], r["subset"]): {k: float(v) for k, v in r.items() if k.startswith(METRIC_PREFIXES)}
                for r in csv.DictReader(f)}


@pytest.fixture(scope="module")
def two_runs(tmp_path_factory):
    runs = []
    for i in range(2):
        d = tmp_path_factory.mktemp(f"run{i}")
        evaluate.main(["--out-dir", str(d), "--model-out", str(d / "model.json")])
        runs.append(d)
    return runs


def test_evaluate_is_deterministic(two_runs):
    a, b = (_metrics(d / "cv_summary.csv") for d in two_runs)
    assert a == b
    assert (two_runs[0] / "model.json").read_text() == (two_runs[1] / "model.json").read_text()


def test_evaluate_matches_committed_results(two_runs):
    new = _metrics(two_runs[0] / "cv_summary.csv")
    committed = _metrics("artifacts/results/cv_summary.csv")
    assert new.keys() == committed.keys()
    for key in committed:
        for metric, value in committed[key].items():
            assert new[key][metric] == pytest.approx(value, abs=1e-3), (key, metric)
