"""Automated validation test suite for FairRank-ID.
Run: python test_suite.py
"""
import os, json, tempfile
import numpy as np
from suspect_id import (
    make_population, make_case, score, rank_candidates, evaluate,
    audit, verify_audit_log, W_HAND, FEATS
)
from ranker import train, load_weights, save_weights


def test_synthetic_population():
    print("[1/5] Testing synthetic population generation...")
    rng = np.random.default_rng(7)
    pop = make_population(n=500, rng=rng)
    assert pop["face"].shape == (500, 128)
    assert len(pop["hair"]) == 500
    assert len(pop["group"]) == 500
    assert "Candidate_" in pop["names"][0]
    print("  ✓ Synthetic population passed.")


def test_scoring_and_ranking():
    print("[2/5] Testing hand-set and learned ranking...")
    rng = np.random.default_rng(7)
    pop = make_population(n=200, rng=rng)
    c = make_case(pop, rng)

    # Hand-set ranking
    rank_hand = rank_candidates(pop, c, k=5, mode="hand")
    assert len(rank_hand["candidates"]) == 5
    assert "face" in rank_hand["candidates"][0]["evidence"]

    # Learned ranking
    weights = {"face": 8.0, "face_x_quality": 7.0, "hair": 1.0, "build": 1.0, "age": 1.0, "zone": 2.0, "mo": 1.0}
    rank_learned = rank_candidates(pop, c, k=5, mode="learned", weights=weights)
    assert len(rank_learned["candidates"]) == 5
    assert "face_x_quality" in rank_learned["candidates"][0]["evidence"]
    print("  ✓ Dual ranking engine passed.")


def test_evidence_suppression():
    print("[3/5] Testing insufficient evidence suppression gate...")
    rng = np.random.default_rng(7)
    pop = make_population(n=200, rng=rng)
    c = make_case(pop, rng)

    # Set very high threshold to guarantee suppression
    res = rank_candidates(pop, c, k=5, min_threshold=99.0)
    assert res["insufficient_evidence"] is True
    assert "suppressed" in res["suppression_reason"].lower()

    # Set low threshold to guarantee pass
    res_pass = rank_candidates(pop, c, k=5, min_threshold=0.1)
    assert res_pass["insufficient_evidence"] is False
    print("  ✓ Insufficient evidence guardrail passed.")


def test_bias_and_fpr_audit():
    print("[4/5] Testing demographic bias and per-group FPR audit...")
    rng = np.random.default_rng(7)
    pop = make_population(n=300, rng=rng)
    cases = [make_case(pop, rng) for _ in range(30)]

    res = evaluate(pop, cases, mode="hand")
    assert "recall@1" in res
    assert "recall@10" in res
    assert "FPR@90%recall" in res
    assert "recall@10_by_group" in res
    assert "fpr@90%recall_by_group" in res
    assert "disparity_ratio_fpr" in res
    assert len(res["fpr@90%recall_by_group"]) > 0
    print(f"  ✓ Bias audit passed.")


def test_tamper_evident_audit_logging():
    print("[5/5] Testing SHA-256 hash-chained audit logging and verification...")
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Write 3 chained records
        rec1 = audit(case_id=0, ranked=[{"candidate": 1, "score": 5.2}], path=tmp_path, reviewer="Det. Walker", decision="Pursue as lead", reason="Corroborated")
        rec2 = audit(case_id=1, ranked=[{"candidate": 2, "score": 4.1}], path=tmp_path, reviewer="Det. Walker", decision="Reject", reason="Alibi verified")
        rec3 = audit(case_id=2, ranked=[], path=tmp_path, reviewer="Det. Walker", supervisor="Capt. Miller", action_type="insufficient_evidence_override", reason="Exigent")

        # Verify intact chain
        valid, count, msg = verify_audit_log(tmp_path)
        assert valid is True
        assert count == 3

        # Simulate tampering on record 2
        with open(tmp_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        tampered_rec = json.loads(lines[1])
        tampered_rec["decision"] = "Forged Decision"
        lines[1] = json.dumps(tampered_rec) + "\n"
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        valid_tampered, _, _ = verify_audit_log(tmp_path)
        assert valid_tampered is False, "Tamper-detection must catch modified records!"
        print("  ✓ SHA-256 hash chain and tamper-detection passed.")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING FAIRRANK-ID VALIDATION SUITE")
    print("=" * 60)
    test_synthetic_population()
    test_scoring_and_ranking()
    test_evidence_suppression()
    test_bias_and_fpr_audit()
    test_tamper_evident_audit_logging()
    print("=" * 60)
    print("ALL TESTS PASSED! System is demo-ready.")
    print("=" * 60)
