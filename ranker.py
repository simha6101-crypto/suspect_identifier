"""Learned ranker for suspect_id.py (numpy only).

Learns evidence weights with a listwise softmax loss: for each training case the model
should give the true candidate the highest probability among ALL candidates.
Compares against the hand-set weights on held-out cases with bias and FPR auditing.
Run: python ranker.py --n 5000 --train 400 --test 500
"""
import argparse, json, os
import numpy as np
from suspect_id import make_population, make_case, score, FEATS, features


def train(pop, cases, iters=300, lr=0.05, l2=1e-3):
    """Train listwise softmax ranker with Adam optimization."""
    X = [features(pop, c) for c in cases]
    truth = [c["truth"] for c in cases]
    w = np.zeros(len(FEATS), dtype=np.float32)
    m, v = np.zeros_like(w), np.zeros_like(w)  # Adam state
    for t in range(1, iters + 1):
        g, loss = np.zeros_like(w), 0.0
        for Xc, y in zip(X, truth):
            s = Xc @ w
            p = np.exp(s - s.max())
            p /= p.sum()
            g += Xc.T @ p - Xc[y]
            loss -= np.log(p[y] + 1e-12)
        g = g / len(X) + l2 * w
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        w -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        if t % 50 == 0:
            print(f"  iter {t:3d}  train loss {loss / len(X):.3f}")
    return w


def evaluate_fn(pop, cases, fn, ks=(1, 5, 10, 20)):
    """fn(case) -> score array over all candidates. Computes recall, FPR, and group fairness."""
    n = len(pop["face"])
    ranks, truth_scores, all_scores, groups = [], [], [], []
    for c in cases:
        s = fn(c)
        t = s[c["truth"]]
        ranks.append(1 + int((s > t).sum()))
        truth_scores.append(t)
        all_scores.append(s)
        groups.append(pop["group"][c["truth"]])

    ranks, groups = np.array(ranks), np.array(groups)
    thr = np.percentile(truth_scores, 10)  # catches 90% of true suspects

    # Overall FPR among innocent candidates
    fprs = []
    for c, s in zip(cases, all_scores):
        innocent_mask = np.ones(n, dtype=bool)
        innocent_mask[c["truth"]] = False
        fprs.append((s[innocent_mask] >= thr).mean())
    overall_fpr = float(np.mean(fprs))

    # Per-group FPR
    unique_groups = np.unique(pop["group"])
    fpr_by_group = {}
    recall_by_group = {}
    for g in unique_groups:
        gf = []
        for c, s in zip(cases, all_scores):
            innocent_mask = np.ones(n, dtype=bool)
            innocent_mask[c["truth"]] = False
            innocent_g = innocent_mask & (pop["group"] == g)
            if innocent_g.sum() > 0:
                gf.append((s[innocent_g] >= thr).mean())
        fpr_by_group[int(g)] = float(np.mean(gf)) if len(gf) else 0.0
        recall_by_group[int(g)] = float((ranks[groups == g] <= 10).mean())

    max_fpr = max(fpr_by_group.values()) if fpr_by_group else 0.0
    min_fpr = min(fpr_by_group.values()) if fpr_by_group else 1e-6
    disparity = float(max_fpr / (min_fpr + 1e-6))

    out = {f"recall@{k}": float((ranks <= k).mean()) for k in ks}
    out["FPR@90%recall"] = overall_fpr
    out["analyst_time_saved"] = float(1 - ranks.mean() / (n / 2))
    out["recall@10_by_group"] = recall_by_group
    out["fpr@90%recall_by_group"] = fpr_by_group
    out["disparity_ratio_fpr"] = round(disparity, 2)
    return out


def load_weights(path="weights.json"):
    """Load learned weights dictionary from JSON."""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def save_weights(w, path="weights.json"):
    """Save learned weights dictionary to JSON."""
    if isinstance(w, dict):
        w_dict = w
    else:
        w_dict = dict(zip(FEATS, map(float, w)))
    with open(path, "w") as f:
        json.dump(w_dict, f, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--train", type=int, default=400)
    ap.add_argument("--test", type=int, default=500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--path", type=str, default=None, help="Path to external real_population.npz if generated via Colab")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    pop = make_population(a.n, rng, path=a.path)
    print(f"Population: {pop['dataset_name']} ({len(pop['face'])} candidates, dim={pop['embedding_dim']})")

    train_cases = [make_case(pop, rng) for _ in range(a.train)]
    test_cases = [make_case(pop, rng) for _ in range(a.test)]  # held out, never seen in training

    print("Training Learned Softmax Ranker...")
    w = train(pop, train_cases)
    print("\nLearned weights:")
    for name, val in zip(FEATS, w):
        print(f"  {name:15s} {val:+.2f}")
    save_weights(w, "weights.json")

    hand = evaluate_fn(pop, test_cases, lambda c: score(pop, c, mode="hand")[0])
    learned = evaluate_fn(pop, test_cases, lambda c: features(pop, c) @ w)

    print(f"\n{'metric':24s}{'hand-set':>12s}{'learned':>12s}")
    for k in ["recall@1", "recall@5", "recall@10", "recall@20", "FPR@90%recall", "analyst_time_saved", "disparity_ratio_fpr"]:
        print(f"{k:24s}{hand[k]:12.3f}{learned[k]:12.3f}")

    print("\nRecall@10 by Demographic Group:")
    print("  Hand-set:", hand["recall@10_by_group"])
    print("  Learned: ", learned["recall@10_by_group"])

    print("\nFPR@90%recall by Demographic Group:")
    print("  Hand-set:", hand["fpr@90%recall_by_group"])
    print("  Learned: ", learned["fpr@90%recall_by_group"])
