"""
Vigil AI — Text Autoencoder (Linear) Training v2
=================================================
Implements an "autoencoder-style" anomaly detector using TruncatedSVD
(a linear autoencoder) trained on SAFE text only.

At inference time, we compute reconstruction error on feature vectors and
flag unusually high-error chunks as "anomalous" (potentially novel dark patterns).

FIXES:
  H3: Proper evaluation against BOTH safe and dark-pattern examples.
      Produces ROC curve data and picks threshold via Youden's J statistic
      instead of arbitrary percentile.
  M4: Prints explained-variance ratio for TruncatedSVD components.
"""

import os
import sys
from typing import List, Tuple

import joblib
import numpy as np

# Reuse the exact same custom feature extractor as the classifier
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.abspath(PROJECT_ROOT))
from app.services.ml_analyzer import extract_custom_features  # noqa: E402

from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.pipeline import FeatureUnion  # noqa: E402
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler  # noqa: E402
from sklearn.decomposition import TruncatedSVD  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve, auc  # noqa: E402


def _load_dataset() -> List[Tuple[str, str]]:
    """
    Load the dataset used by the supervised model so we don't need a new file format.
    Importing this module is safe: it only defines DATASET at module scope.
    """
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    from train_ml_model import DATASET  # type: ignore  # noqa: E402

    return list(DATASET)


def _build_feature_union():
    # Keep these aligned with train_ml_model.py
    word_tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        max_features=5000,
        sublinear_tf=True,
        min_df=2,
        lowercase=True,
    )
    char_tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 6),
        max_features=4000,
        sublinear_tf=True,
        min_df=2,
        lowercase=True,
    )
    custom_feats = FunctionTransformer(extract_custom_features)

    return FeatureUnion(
        [
            ("word", word_tfidf),
            ("char", char_tfidf),
            ("custom", custom_feats),
        ]
    )


def _reconstruction_mse(x_scaled, xhat_scaled) -> np.ndarray:
    """
    Compute per-row mean squared reconstruction error.
    x_scaled can be sparse; xhat_scaled is dense.
    """
    if hasattr(x_scaled, "toarray"):
        x_dense = x_scaled.toarray()
    else:
        x_dense = np.asarray(x_scaled)
    diff = x_dense - xhat_scaled
    return np.mean(diff * diff, axis=1)


def train(
    n_components: int = 256,
    threshold_percentile: float = 99.0,
    random_state: int = 42,
):
    dataset = _load_dataset()
    safe_texts = [t for (t, y) in dataset if y == "safe"]
    dark_texts = [t for (t, y) in dataset if y != "safe"]

    if len(safe_texts) < 50:
        raise RuntimeError("Not enough SAFE examples to train autoencoder.")

    X_train, X_val = train_test_split(
        safe_texts, test_size=0.2, random_state=random_state
    )

    # Also split dark-pattern texts for evaluation
    dark_eval = dark_texts[:200]  # Use up to 200 dark-pattern examples for eval

    feature_union = _build_feature_union()
    scaler = MaxAbsScaler()
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)

    print("=" * 72)
    print("  Vigil AI — Text Autoencoder (Linear/SVD) Training v2")
    print("  Fixes: H3 (ROC evaluation), M4 (explained variance)")
    print("=" * 72)
    print(f"  SAFE samples: {len(safe_texts)} (train={len(X_train)}, val={len(X_val)})")
    print(f"  Dark-pattern eval samples: {len(dark_eval)}")
    print(f"  SVD components: {n_components}")

    # Fit transforms on SAFE train
    X_train_feats = feature_union.fit_transform(X_train)
    X_train_scaled = scaler.fit_transform(X_train_feats)
    svd.fit(X_train_scaled)

    # ── M4 FIX: Print explained variance ratio ──────────────────────────────
    explained_var = svd.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    total_explained = cumulative_var[-1]
    # Find how many components for 85% and 95%
    n_85 = np.searchsorted(cumulative_var, 0.85) + 1
    n_95 = np.searchsorted(cumulative_var, 0.95) + 1

    print(f"\n  ── SVD Explained Variance (M4 fix) ────────────────────────────")
    print(f"  Total explained variance ({n_components} components): {total_explained*100:.1f}%")
    print(f"  Components for 85% variance: {n_85}")
    print(f"  Components for 95% variance: {n_95}")

    if total_explained < 0.85:
        print(f"  ⚠ WARNING (M4): {n_components} components capture only {total_explained*100:.1f}% "
              f"of variance — consider increasing n_components.")
    else:
        print(f"  ✓ {n_components} components capture ≥85% of variance.")

    # Compute validation reconstruction errors (SAFE)
    X_val_feats = feature_union.transform(X_val)
    X_val_scaled = scaler.transform(X_val_feats)
    Z_val = svd.transform(X_val_scaled)
    Xhat_val_scaled = svd.inverse_transform(Z_val)
    val_err = _reconstruction_mse(X_val_scaled, Xhat_val_scaled)

    # ── H3 FIX: Evaluate against BOTH safe AND dark-pattern examples ─────────
    print(f"\n  ── Autoencoder ROC Evaluation (H3 fix) ────────────────────────")

    # Compute reconstruction errors for dark-pattern examples
    if dark_eval:
        X_dark_feats = feature_union.transform(dark_eval)
        X_dark_scaled = scaler.transform(X_dark_feats)
        Z_dark = svd.transform(X_dark_scaled)
        Xhat_dark_scaled = svd.inverse_transform(Z_dark)
        dark_err = _reconstruction_mse(X_dark_scaled, Xhat_dark_scaled)

        # Build labels: 0 = safe (normal), 1 = dark-pattern (anomaly)
        all_errors = np.concatenate([val_err, dark_err])
        all_labels = np.concatenate([np.zeros(len(val_err)), np.ones(len(dark_err))])

        # ROC curve
        fpr, tpr, roc_thresholds = roc_curve(all_labels, all_errors)
        roc_auc = auc(fpr, tpr)
        print(f"  ROC AUC: {roc_auc:.4f}")

        # Precision-Recall curve
        precision, recall, pr_thresholds = precision_recall_curve(all_labels, all_errors)
        pr_auc = auc(recall, precision)
        print(f"  PR AUC:  {pr_auc:.4f}")

        # Youden's J statistic to pick optimal threshold
        j_scores = tpr - fpr
        best_j_idx = np.argmax(j_scores)
        optimal_threshold = float(roc_thresholds[best_j_idx])
        print(f"  Optimal threshold (Youden's J): {optimal_threshold:.6f}")
        print(f"    at TPR={tpr[best_j_idx]:.3f}, FPR={fpr[best_j_idx]:.3f}")

        # Also compute the old percentile-based threshold for comparison
        percentile_threshold = float(np.percentile(val_err, threshold_percentile))
        print(f"  Percentile-based threshold (p{threshold_percentile:.0f}): {percentile_threshold:.6f}")

        # Use the ROC-optimal threshold instead of arbitrary percentile
        threshold = optimal_threshold

        # Report confusion-matrix-style stats at chosen threshold
        safe_flagged = np.sum(val_err > threshold)
        dark_flagged = np.sum(dark_err > threshold)
        print(f"\n  At chosen threshold ({threshold:.6f}):")
        print(f"    Safe correctly below threshold: {len(val_err) - safe_flagged}/{len(val_err)} "
              f"({(len(val_err) - safe_flagged)/len(val_err)*100:.1f}%)")
        print(f"    Dark correctly above threshold: {dark_flagged}/{len(dark_err)} "
              f"({dark_flagged/len(dark_err)*100:.1f}%)")
    else:
        # Fallback to percentile if no dark examples
        threshold = float(np.percentile(val_err, threshold_percentile))
        roc_auc = None
        pr_auc = None
        optimal_threshold = threshold
        print("  No dark-pattern examples available for ROC evaluation.")

    print(f"\n  Val reconstruction error: mean={val_err.mean():.6f}, "
          f"p99={np.percentile(val_err, 99):.6f}")
    print(f"  Final threshold: {threshold:.6f}")

    artifact = {
        "kind": "vigil_text_ae_svd",
        "version": 2,
        "feature_union": feature_union,
        "scaler": scaler,
        "svd": svd,
        "threshold": threshold,
        "threshold_method": "youden_j_statistic" if dark_eval else "percentile",
        "threshold_percentile": threshold_percentile,
        "n_components": n_components,
        "random_state": random_state,
        "explained_variance_total": float(total_explained),
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
        "pr_auc": float(pr_auc) if pr_auc is not None else None,
    }

    models_dir = os.path.join(os.path.dirname(__file__), "..", "app", "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "dp_text_ae.pkl")
    joblib.dump(artifact, model_path)

    print(f"\n  [OK] Autoencoder artifact saved -> {os.path.abspath(model_path)}")
    print("=" * 72)


if __name__ == "__main__":
    train()
