"""Real Face Embeddings Pipeline: InsightFace (ArcFace 512-D) + LFW Dataset + FAISS.

This script extracts genuine 512-dimensional ArcFace face embeddings from the
Labeled Faces in the Wild (LFW) dataset using InsightFace on genuine photo pairs:
- Photo 1 of a person -> Gallery (population)
- Photo 2 of the same person (a completely different photo) -> Probe query
No artificial Gaussian noise is added to the probe embedding.
Candidate identities are strictly anonymized (Subject_0001, Subject_0002, etc.).

DISCLOSURE:
While face embeddings and probe pairs are genuine 512-D ArcFace vectors from LFW,
all contextual attributes (hair, build, age, zone, MO) are simulated.

Usage:
  python real_face_pipeline.py --min-faces 2 --output real_population.npz
"""
import os, sys, json, argparse, time
import numpy as np

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    import insightface
    from insightface.app import FaceAnalysis
    HAS_INSIGHTFACE = True
except ImportError:
    HAS_INSIGHTFACE = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

ZONES, MO_TAGS = 30, 8
GROUP_NOISE = [1.0, 1.3, 1.8]


def unit(x):
    """Normalize vectors to unit sphere for cosine similarity via inner product."""
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, 1e-12)


class FAISSFaceIndex:
    """FAISS-based vector index for vector similarity search."""
    def __init__(self, dim=512):
        self.dim = dim
        if HAS_FAISS:
            self.index = faiss.IndexFlatIP(dim)
        else:
            self.index = None
            self.vectors = None

    def add(self, embeddings):
        embeddings = embeddings.astype(np.float32)
        if HAS_FAISS:
            self.index.add(embeddings)
        else:
            self.vectors = embeddings if self.vectors is None else np.vstack([self.vectors, embeddings])

    def search(self, query_embeddings, k=10):
        query_embeddings = query_embeddings.astype(np.float32)
        if HAS_FAISS:
            distances, indices = self.index.search(query_embeddings, k)
            return distances, indices
        else:
            sims = query_embeddings @ self.vectors.T
            indices = np.argsort(-sims, axis=1)[:, :k]
            distances = np.take_along_axis(sims, indices, axis=1)
            return distances, indices


def features(pop, c):
    """Constructs signed feature matrix (N, 7)."""
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


def train_learned_ranker(pop, cases, iters=300, lr=0.05, l2=1e-3):
    """Trains listwise softmax ranker weights."""
    X = [features(pop, c) for c in cases]
    truth = [c["truth"] for c in cases]
    w = np.zeros(7, dtype=np.float32)
    m, v = np.zeros_like(w), np.zeros_like(w)
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
    return w


def evaluate_pipeline(pop, cases, score_fn, ks=(1, 5, 10, 20)):
    """Comprehensive evaluation computing recall, FPR, and group fairness."""
    n = len(pop["face"])
    ranks, truth_scores, all_scores, groups = [], [], [], []
    for c in cases:
        scores = score_fn(c)
        t = scores[c["truth"]]
        ranks.append(1 + int((scores > t).sum()))
        truth_scores.append(t)
        all_scores.append(scores)
        groups.append(pop["group"][c["truth"]])

    ranks, groups = np.array(ranks), np.array(groups)
    thr = np.percentile(truth_scores, 10)  # Threshold for 90% true positive recall

    # Compute overall FPR among all innocent candidates
    fpr_list = []
    for c, s in zip(cases, all_scores):
        innocent_mask = np.ones(n, dtype=bool)
        innocent_mask[c["truth"]] = False
        fpr_list.append((s[innocent_mask] >= thr).mean())
    overall_fpr = float(np.mean(fpr_list))

    # Per-group FPR
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
    disparity = float(max_fpr / (min_fpr + 1e-6))

    return {
        **{f"recall@{k}": float((ranks <= k).mean()) for k in ks},
        "FPR@90%recall": overall_fpr,
        "analyst_time_saved": float(1 - ranks.mean() / (n / 2)),
        "recall@10_by_group": recall_by_group,
        "fpr@90%recall_by_group": fpr_by_group,
        "disparity_ratio_fpr": round(disparity, 2)
    }


def extract_lfw_embeddings_with_insightface(min_faces=2):
    """Extracts ArcFace embeddings on genuine LFW photo pairs."""
    from sklearn.datasets import fetch_lfw_people
    print("Loading LFW dataset via scikit-learn...")
    lfw = fetch_lfw_people(min_faces_per_person=min_faces, resize=1.0, color=True)
    print(f"Loaded {len(lfw.images)} images across {len(lfw.target_names)} subjects.")

    print("Initializing InsightFace (buffalo_l)...")
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    gallery_embeddings = []
    probe_cases_raw = []  # (true_idx, probe_embedding)
    person_to_gallery_idx = {}

    for img, target in zip(lfw.images, lfw.target):
        img_bgr = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        faces = app.get(img_bgr)
        if len(faces) == 0:
            continue
        emb = unit(faces[0].embedding.astype(np.float32))

        if target not in person_to_gallery_idx:
            # Photo 1 -> Registered in Gallery
            person_to_gallery_idx[target] = len(gallery_embeddings)
            gallery_embeddings.append(emb)
        else:
            # Photo 2 (different real photo of same person) -> Probe query
            true_idx = person_to_gallery_idx[target]
            probe_cases_raw.append((true_idx, emb))

    gallery_embeddings = np.array(gallery_embeddings, dtype=np.float32)
    return gallery_embeddings, probe_cases_raw


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-faces", type=int, default=2)
    parser.add_argument("--output", type=str, default="real_population.npz")
    args = parser.parse_args()

    if not HAS_INSIGHTFACE or not HAS_CV2:
        print("Note: InsightFace or OpenCV is not installed locally. Run this pipeline in Google Colab using insightface_faiss_colab.ipynb with GPU acceleration.")
        sys.exit(0)

    gallery, probes = extract_lfw_embeddings_with_insightface(args.min_faces)
    n = len(gallery)
    rng = np.random.default_rng(42)

    # Simulated contextual attributes
    pop = {
        "face": gallery,
        "hair": rng.integers(5, size=n, dtype=np.int64),
        "build": rng.integers(3, size=n, dtype=np.int64),
        "age": rng.integers(4, size=n, dtype=np.int64),
        "zones": (rng.random((n, ZONES)) < 0.12),
        "mo": rng.integers(MO_TAGS, size=n, dtype=np.int64),
        "group": rng.choice(3, size=n, p=[0.6, 0.3, 0.1]).astype(np.int64),
        "names": np.array([f"Subject_{i:04d}" for i in range(n)])
    }

    np.savez_compressed(
        args.output,
        face=pop["face"],
        hair=pop["hair"],
        build=pop["build"],
        age=pop["age"],
        zones=pop["zones"],
        mo=pop["mo"],
        group=pop["group"],
        names=pop["names"]
    )
    print(f"Exported genuine LFW embeddings to {args.output} ({n} identities).")
