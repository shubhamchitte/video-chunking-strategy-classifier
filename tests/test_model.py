import numpy as np
import pytest

from models import CLASSES, JsonLogReg, load_dataset

MODEL = "artifacts/model/logreg_v1.json"
# Reference outputs of the committed model on the first rows of artifacts/features/cheap_features.csv.
EXPECTED = {
    "026dzf-vc5g": ("4_narrative", [0.0429, 0.0150, 0.3346, 0.6075]),
    "1NYQf_OXDqI": ("4_narrative", [0.0526, 0.0336, 0.2650, 0.6489]),
    "27YZLpw0M98": ("2_info_dense", [0.1433, 0.4387, 0.0047, 0.4133]),
    "2QTiAmvygC4": ("4_narrative", [0.0660, 0.0766, 0.2487, 0.6087]),
    "2gFxZmb0dMo": ("4_narrative", [0.2868, 0.1026, 0.2185, 0.3921]),
}


def test_model_artifact_loads_and_matches_feature_schema():
    m = JsonLogReg(MODEL)
    _, _, _, names, _ = load_dataset()
    assert m.classes == CLASSES
    assert m.features == names
    assert m.coef.shape == (4, len(names))


def test_model_reproduces_stored_predictions():
    m = JsonLogReg(MODEL)
    X, _, ids, _, _ = load_dataset()
    rows = [ids.index(v) for v in EXPECTED]
    proba = m.predict_proba(X[rows])
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
    for p, (vid, (cls, ref)) in zip(proba, EXPECTED.items()):
        assert m.classes[int(p.argmax())] == cls, vid
        np.testing.assert_allclose(p, ref, atol=1e-4)


def test_json_model_matches_sklearn_refit():
    from models import make_model
    X, y, _, _, _ = load_dataset()
    sk = make_model("logreg").fit(X, y)
    # Same data and settings; allow tiny solver differences across sklearn versions.
    np.testing.assert_allclose(sk.predict_proba(X), JsonLogReg(MODEL).predict_proba(X), atol=1e-3)


def test_model_features_are_exactly_the_documented_17():
    """No extra column (e.g. analysed_sec, extract_sec, timing) may slip into the model or the training data."""
    import csv

    from cheap import FEATURES
    assert len(FEATURES) == 17
    assert JsonLogReg(MODEL).features == FEATURES
    _, _, _, names, _ = load_dataset()
    assert names == FEATURES
    header = next(csv.reader(open("artifacts/features/cheap_features.csv")))
    assert set(header) - set(FEATURES) == {"video_id", "analysed_sec", "extract_sec"}
    leaky = [n for n in FEATURES if any(k in n for k in ("sec", "duration", "time", "cpu", "wall"))]
    assert leaky == []
