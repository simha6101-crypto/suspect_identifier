"""Suspect Identification: Multi-Modal Ranking, Safety Guardrails & Bias Auditing.

Pipeline:
  Synthetic population (128-D unit vectors with calibrated demographic noise)
  -> Surveillance probe (with CCTV degradation & witness error rates)
  -> Multi-signal scoring (Face similarity + Contextual Attributes + Activity Zones + MO)
  -> Ranker Engine (Hand-Set Heuristic Weights vs Learned Softmax Ranker)
  -> Safety Gate (Insufficient Evidence Suppression Threshold)
  -> Ranked candidates with explainable evidence contributions
  -> Tamper-evident hash-chained audit trail (Human-in-the-loop decisions)
  -> Bias & Safety Evaluation (Recall@K, FPR@90%recall, Per-Group FPR & Disparity Ratio).

The system only ranks investigative leads. It never makes legal identifications.
All contextual attributes (hair, build, age, zone, MO) are simulated.
"""
import argparse, json, time, os, hashlib
import numpy as np

ZONES, MO_TAGS = 30, 8
GROUP_NOISE = [1.0, 1.3, 1.8]   # Simulates embedding quality disparity across demographic cohorts
W_HAND = {"face": 8.0, "attr": 1.0, "zone": 1.5, "mo": 1.0}
FEATS = ["face", "face_x_quality", "hair", "build", "age", "zone", "mo"]


def unit(x):
    """Normalize vectors to unit sphere."""
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, 1e-12)


def make_population(n=5000, rng=None, path=None):
    """Create a synthetic population or load an external real_population.npz from Colab.
    All candidate identities are strictly anonymous identifiers (e.g. Candidate_0042).
    """
    if path and os.path.exists(path):
        data = np.load(path, allow_pickle=True)
        face = data["face"].astype(np.float32)
        n_actual = len(face)
        # Ensure candidate names are completely anonymous
        names = np.array([f"Subject_{i:04d}" for i in range(n_actual)])
        return {
            "face": face,
            "hair": data["hair"].astype(np.int64),
            "build": data["build"].astype(np.int64),
            "age": data["age"].astype(np.int64),
            "zones": data["zones"].astype(bool),
            "mo": data["mo"].astype(np.int64),
            "group": data["group"].astype(np.int64),
            "names": names,
            "is_real": True,
            "embedding_dim": face.shape[1],
            "dataset_name": f"External LFW Embeddings (Colab, {face.shape[1]}-D)"
        }

    if rng is None:
        rng = np.random.default_rng(7)

    return {
        "face": unit(rng.normal(size=(n, 128)).astype(np.float32)),
        "hair": rng.integers(5, size=n, dtype=np.int64),
        "build": rng.integers(3, size=n, dtype=np.int64),
        "age": rng.integers(4, size=n, dtype=np.int64),
        "zones": (rng.random((n, ZONES)) < 0.12),
        "mo": rng.integers(MO_TAGS, size=n, dtype=np.int64),
        "group": rng.choice(3, size=n, p=[0.6, 0.3, 0.1]).astype(np.int64),
        "names": np.array([f"Candidate_{i:04d}" for i in range(n)]),
        "is_real": False,
        "embedding_dim": 128,
        "dataset_name": "Synthetic Benchmark (128-D)"
    }


def make_case(pop, rng):
    """Generate a surveillance probe case with realistic degradation and witness uncertainty."""
    n = len(pop["face"])
    d = pop["face"].shape[1]
    s = int(rng.integers(n))
    quality = float(rng.uniform(0.3, 1.0))  # Surveillance video quality factor

    sigma = GROUP_NOISE[pop["group"][s]] / quality
    probe = unit(pop["face"][s] + rng.normal(size=d) * (sigma * (1.2 if d > 128 else 1.0)) / np.sqrt(d))

    def witness(true_val, k):
        if rng.random() > 0.7:  # 30% unobserved
            return None
        return int(true_val) if rng.random() < 0.8 else int(rng.integers(k))  # 80% witness reliability

    zs = np.flatnonzero(pop["zones"][s])
    zone = int(rng.choice(zs)) if len(zs) and rng.random() < 0.8 else int(rng.integers(ZONES))

    return {
        "truth": s,
        "quality": quality,
        "probe": probe,
        "zone": zone,
        "hair": witness(pop["hair"][s], 5),
        "build": witness(pop["build"][s], 3),
        "age": witness(pop["age"][s], 4),
        "mo": witness(pop["mo"][s], MO_TAGS),
    }


def features(pop, c):
    """Compute (N, 7) signed feature matrix."""
    face = pop["face"] @ c["probe"]

    def m(k):
        if c[k] is None:
            return np.zeros(len(face), dtype=np.float32)
        return np.where(pop[k] == c[k], 1.0, -1.0).astype(np.float32)

    return np.stack([
        face,
        face * c["quality"],
        m("hair"),
        m("build"),
        m("age"),
        pop["zones"][:, c["zone"]].astype(np.float32),
        m("mo")
    ], axis=1).astype(np.float32)


def score(pop, case, face_only=False, mode="hand", weights=None):
    """Compute candidate scores and explainable evidence contributions."""
    face = pop["face"] @ case["probe"]

    if face_only:
        sig = {"face": W_HAND["face"] * case["quality"] * face}
        return sig["face"], sig

    if mode == "learned" and weights is not None:
        if isinstance(weights, dict):
            w_arr = np.array([weights.get(f, 0.0) for f in FEATS], dtype=np.float32)
        else:
            w_arr = np.array(weights, dtype=np.float32)
        X = features(pop, case)
        total = X @ w_arr
        sig = {FEATS[i]: X[:, i] * w_arr[i] for i in range(len(FEATS))}
        return total, sig

    # Hand-set heuristic weights
    obs = [(k, case[k]) for k in ("hair", "build", "age") if case[k] is not None]
    a = np.zeros(len(face), dtype=np.float32)
    for k, v in obs:
        a += np.where(pop[k] == v, 1.0, -0.5).astype(np.float32)
    attr = W_HAND["attr"] * a / max(len(obs), 1)
    zone = W_HAND["zone"] * pop["zones"][:, case["zone"]].astype(np.float32)
    mo = W_HAND["mo"] * (np.zeros(len(face), dtype=np.float32) if case["mo"] is None
                         else (pop["mo"] == case["mo"]).astype(np.float32))

    sig = {
        "face": W_HAND["face"] * case["quality"] * face,
        "attr": attr,
        "zone": zone,
        "mo": mo
    }
    total = sig["face"] + sig["attr"] + sig["zone"] + sig["mo"]
    return total, sig


def rank_candidates(pop, case, k=10, mode="hand", weights=None, min_threshold=None):
    """Rank candidates with explainability and evidence confidence gate."""
    total, sig = score(pop, case, mode=mode, weights=weights)
    top_indices = np.argsort(-total)[:k]
    max_score = float(total[top_indices[0]]) if len(top_indices) else 0.0

    insufficient_evidence = False
    suppression_reason = None
    if min_threshold is not None and max_score < min_threshold:
        insufficient_evidence = True
        suppression_reason = (
            f"Top candidate score ({max_score:.2f}) is below the required evidence threshold ({min_threshold:.2f}). "
            f"Image quality is {case['quality']:.2f}. Automated ranking suppressed to prevent confirmation bias."
        )

    candidates = []
    for pos, i in enumerate(top_indices, 1):
        name = pop["names"][i] if "names" in pop else f"Candidate_{i:04d}"
        candidates.append({
            "candidate": int(i),
            "name": str(name),
            "rank": pos,
            "score": round(float(total[i]), 3),
            "evidence": {s: round(float(v[i]), 3) for s, v in sig.items()},
            "image_quality": round(case["quality"], 2),
            "group": int(pop["group"][i])
        })

    return {
        "candidates": candidates,
        "insufficient_evidence": insufficient_evidence,
        "suppression_reason": suppression_reason,
        "top_score": round(max_score, 3),
        "threshold": min_threshold,
        "mode": mode,
        "image_quality": round(case["quality"], 2)
    }


def get_last_audit_hash(path="audit_log.jsonl"):
    """Read the hash of the last record in the audit log for hash chaining."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return "0" * 64
    last_line = ""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                last_line = line
    if not last_line:
        return "0" * 64
    try:
        data = json.loads(last_line)
        return data.get("hash", "0" * 64)
    except Exception:
        return "0" * 64


def audit(case_id, ranked, path="audit_log.jsonl", reviewer=None, decision=None, reason=None, action_type="ranking", supervisor=None, threshold=None):
    """Write an append-only, tamper-evident hash-chained audit record."""
    prev_hash = get_last_audit_hash(path)
    rec = {
        "ts": time.time(),
        "type": action_type,
        "case": case_id,
        "reviewer": reviewer,
        "supervisor": supervisor,
        "decision": decision,
        "reason": reason,
        "threshold": threshold,
        "candidates": ranked,
        "prev_hash": prev_hash
    }
    # Deterministic SHA-256 hash over canonical JSON (excluding "hash" key)
    canonical = json.dumps(rec, sort_keys=True)
    rec["hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def verify_audit_log(path="audit_log.jsonl"):
    """Verify cryptographic hash-chain integrity of the append-only audit trail."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return True, 0, "Audit log is empty."
    expected_prev = "0" * 64
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            count += 1
            rec = json.loads(line)
            actual_hash = rec.get("hash")
            actual_prev = rec.get("prev_hash")
            if actual_prev != expected_prev:
                return False, line_num, f"Broken chain at line {line_num}: expected prev_hash {expected_prev[:8]}..., got {str(actual_prev)[:8]}..."
            rec_copy = {k: v for k, v in rec.items() if k != "hash"}
            canonical = json.dumps(rec_copy, sort_keys=True)
            computed_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            if actual_hash != computed_hash:
                return False, line_num, f"Tampered record at line {line_num}: computed hash {computed_hash[:8]}..., stored {str(actual_hash)[:8]}..."
            expected_prev = actual_hash
    return True, count, f"Hash chain valid across all {count} records."


def evaluate(pop, cases, mode="hand", weights=None, face_only=False, ks=(1, 5, 10, 20)):
    """Evaluate retrieval accuracy, FPR, and group fairness metrics."""
    n = len(pop["face"])
    ranks, truth_scores, all_scores, groups = [], [], [], []

    for c in cases:
        total, _ = score(pop, c, face_only=face_only, mode=mode, weights=weights)
        t = total[c["truth"]]
        ranks.append(1 + int((total > t).sum()))
        truth_scores.append(t)
        all_scores.append(total)
        groups.append(pop["group"][c["truth"]])

    ranks = np.array(ranks)
    groups = np.array(groups)
    truth_scores = np.array(truth_scores)

    # 90% Recall threshold across true suspects
    thr = np.percentile(truth_scores, 10)

    # Overall False Positive Rate (FPR) among innocent population
    fpr_list = []
    for c, s in zip(cases, all_scores):
        innocent_mask = np.ones(n, dtype=bool)
        innocent_mask[c["truth"]] = False
        fpr_list.append((s[innocent_mask] >= thr).mean())
    overall_fpr = float(np.mean(fpr_list))

    # Per-group FPR: False alarm rate broken down by demographic cohort
    unique_groups = np.unique(pop["group"])
    fpr_by_group = {}
    recall_by_group = {}
    for g in unique_groups:
        g_fprs = []
        for c, s in zip(cases, all_scores):
            innocent_mask = np.ones(n, dtype=bool)
            innocent_mask[c["truth"]] = False
            innocent_g = innocent_mask & (pop["group"] == g)
            if innocent_g.sum() > 0:
                g_fprs.append((s[innocent_g] >= thr).mean())
        fpr_by_group[int(g)] = float(np.mean(g_fprs)) if len(g_fprs) else 0.0
        recall_by_group[int(g)] = float((ranks[groups == g] <= 10).mean())

    max_fpr = max(fpr_by_group.values()) if fpr_by_group else 0.0
    min_fpr = min(fpr_by_group.values()) if fpr_by_group else 1e-6
    disparity_ratio = float(max_fpr / (min_fpr + 1e-6))

    out = {
        **{f"recall@{k}": float((ranks <= k).mean()) for k in ks},
        "FPR@90%recall": overall_fpr,
        "analyst_time_saved": float(1 - ranks.mean() / (n / 2)),
        "recall@10_by_group": recall_by_group,
        "fpr@90%recall_by_group": fpr_by_group,
        "disparity_ratio_fpr": round(disparity_ratio, 2)
    }
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--cases", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--path", type=str, default=None, help="Path to external real_population.npz if generated via Colab")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    pop = make_population(args.n, rng, path=args.path)
    cases = [make_case(pop, rng) for _ in range(args.cases)]

    print(f"Dataset: {pop['dataset_name']} ({len(pop['face'])} candidates, dim={pop['embedding_dim']})")

    # Evaluate Face-only baseline vs Fused
    print("\n" + "=" * 55)
    print("BENCHMARK EVALUATION")
    print("=" * 55)
    res_face = evaluate(pop, cases, face_only=True)
    res_hand = evaluate(pop, cases, face_only=False, mode="hand")

    print(f"{'Metric':<25} {'Face-Only':<15} {'Fused Hand-Set':<15}")
    print("-" * 55)
    for k in ["recall@1", "recall@5", "recall@10", "recall@20", "FPR@90%recall", "analyst_time_saved", "disparity_ratio_fpr"]:
        print(f"{k:<25} {res_face[k]:<15.3f} {res_hand[k]:<15.3f}")

    print("\nRecall@10 by demographic group:")
    print("  Face-only:", res_face["recall@10_by_group"])
    print("  Fused:    ", res_hand["recall@10_by_group"])

    print("\nFPR@90%recall by demographic group:")
    print("  Face-only:", res_face["fpr@90%recall_by_group"])
    print("  Fused:    ", res_hand["fpr@90%recall_by_group"])

    demo_rank = rank_candidates(pop, cases[0], k=3, min_threshold=3.5)
    print("\nDemo Case 0 (Threshold=3.5):")
    print(f"  Insufficient evidence suppressed: {demo_rank['insufficient_evidence']}")
    for c in demo_rank["candidates"]:
        print(f"  #{c['rank']} {c['name']} (ID: {c['candidate']}) Score: {c['score']} Evidence: {c['evidence']}")
