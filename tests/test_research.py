import numpy as np
import pytest
from scipy.special import softmax

from nlp_lab.research import fit_temperature, make_perturbations, macro_f1_fast, paired_bootstrap, uncertainty


def test_temperature_fit_preserves_predictions_and_improves_validation_nll():
    scores = np.array([[10., -10.]] * 5 + [[-10., 10.]] * 5)
    truth = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 0])
    temperature = fit_temperature(scores, truth)
    before = uncertainty(softmax(scores, axis=1), truth)
    after = uncertainty(softmax(scores / temperature, axis=1), truth)
    assert temperature > 1
    assert after["nll"] < before["nll"]
    assert np.array_equal(scores.argmax(1), (scores / temperature).argmax(1))
    assert sum(x["samples"] for x in after["reliability_bins"]) == len(truth)


def test_paired_bootstrap_identity_and_clear_improvement():
    truth = np.tile([0, 1], 50)
    reference = np.zeros((2, 100), dtype=int)
    candidate = np.stack([truth, truth])
    equal = paired_bootstrap(candidate, candidate, truth, 2, replicates=30)
    assert equal["delta_macro_f1"] == 0 and equal["ci95"] == [0, 0]
    result = paired_bootstrap(reference, candidate, truth, 2, replicates=30)
    assert result["delta_macro_f1"] == pytest.approx(2 / 3)
    assert result["ci95"][0] > 0
    assert macro_f1_fast(truth, truth, 2) == 1


def test_stress_selection_is_balanced_deterministic_and_paired():
    rows = [{"text": f"股票市场新闻标题第{i}条", "label": label, "source_line": i} for label in ["甲", "乙"] for i in range(10)]
    a = make_perturbations(rows, ["甲", "乙"], per_class=3)
    b = make_perturbations(rows, ["甲", "乙"], per_class=3)
    assert a == b
    assert len(a["indices"]) == 6
    for mode, cases in a["cases"].items():
        assert [x["test_index"] for x in cases] == a["indices"]
        assert sum(x["label"] == "甲" for x in cases) == 3
        assert all(x["text"] for x in cases)
    assert any(a != b for a, b in zip([x["text"] for x in a["cases"]["clean_subset"]],
                                     [x["text"] for x in a["cases"]["char_delete_5pct"]]))
