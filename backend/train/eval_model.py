"""
Vigil AI — Honest Evaluation Script v2
========================================
FIXED (C1, C2, H2): Uses the EXACT SAME split as train_ml_model.py
(test_size=0.15, random_state=42, stratify=labels) so the test set
contains ONLY examples the model has never seen during training.

FIXED (C3): Reports per-class sample counts and warns when n < 50.

NEW (M5): Includes adversarial test examples (Unicode lookalikes,
subtle wording, mixed-language, obfuscated text).

Run from backend/train/:
    python eval_model.py
"""

import os
import sys
import json
import time
import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from train_ml_model import DATASET

# ── ADVERSARIAL TEST SET (M5) ────────────────────────────────────────────────
# These examples test robustness against obfuscation and edge cases.
# They are NOT part of training. They are evaluated separately.
ADVERSARIAL_EXAMPLES = [
    # Unicode lookalikes (𝐅𝐑𝐄𝐄 instead of FREE)
    ("𝐅𝐑𝐄𝐄 trial — then $19.99/month automatically.", "forced_continuity"),
    ("𝐎𝐧𝐥𝐲 𝟐 left in stock!", "urgency"),

    # Subtle / hedged wording
    ("We may occasionally share some of your information with select partners.", "privacy"),
    ("Your preferences have been pre-configured for your convenience.", "preselection"),
    ("A small contribution has been included to support our mission.", "basket_sneaking"),

    # Mixed-language
    ("Oferta limitada — solo hoy! Limited time only.", "urgency"),
    ("Votre essai gratuit se termine bientôt. You will be charged $9.99.", "forced_continuity"),

    # CSS-invisible style text patterns
    ("color:transparent; Only 1 room left at this price!", "urgency"),
    ("display:none; We share your data with 200 partners.", "privacy"),

    # Intentionally subtle / borderline
    ("Most customers choose this option.", "social_proof"),
    ("We think you'll love the premium plan we've selected for you.", "preselection"),
    ("A processing fee applies — see details at checkout.", "hidden_costs"),
    ("No thanks, I'd rather not improve my results.", "confirmshaming"),

    # Very short / fragmented (tests min-length handling)
    ("Act now!", "urgency"),
    ("Join 10M+ users.", "social_proof"),

    # Long compound sentence (real-world blob)
    (
        "In high demand! We and 3 partners use cookies. Only 2 seats left — "
        "cancel anytime. Your trial converts to $29.99/month automatically. "
        "No thanks, I prefer to miss out.",
        "urgency",  # dominant pattern — but multi-class
    ),
]


def evaluate():
    print("=" * 70)
    print("  Vigil AI — Honest Model Evaluation v2")
    print("  Fixes: C1 (train-data eval), C2 (split mismatch), C3 (small test)")
    print("=" * 70)

    # ── Load model ────────────────────────────────────────────────────────────
    model_path = os.path.join(
        os.path.dirname(__file__), '..', 'app', 'models', 'dp_classifier.pkl'
    )
    if not os.path.exists(model_path):
        print(f"  [ERROR] Model not found: {model_path}")
        print("  Run train_ml_model.py first.")
        sys.exit(1)

    model = joblib.load(model_path)

    # ── EXACT SAME SPLIT as train_ml_model.py (FIX C1/C2/H2) ────────────────
    # Uses test_size=0.15, random_state=42, stratify=labels
    # This guarantees X_test here == X_test in training == truly unseen data
    texts  = [d[0] for d in DATASET]
    labels = [d[1] for d in DATASET]
    classes = sorted(set(labels))

    X_train_raw, X_test, y_train_raw, y_test = train_test_split(
        texts, labels, test_size=0.15, random_state=42, stratify=labels
    )

    print(f"\n  Dataset     : {len(DATASET)} examples, {len(classes)} classes")
    print(f"  Train split : {len(X_train_raw)} (same as training script)")
    print(f"  Test split  : {len(X_test)} (truly held-out, NO augmented data)")

    # ── Per-class sample count warning (FIX C3) ──────────────────────────────
    from collections import Counter
    test_dist = Counter(y_test)
    MIN_SAMPLES_PER_CLASS = 50
    print(f"\n  Per-class test sample counts (minimum recommended: {MIN_SAMPLES_PER_CLASS}):")
    insufficient_classes = []
    for cls in classes:
        n = test_dist.get(cls, 0)
        status = "OK" if n >= MIN_SAMPLES_PER_CLASS else "INSUFFICIENT"
        if n < MIN_SAMPLES_PER_CLASS:
            insufficient_classes.append((cls, n))
        print(f"    {cls:22s}: {n:4d}  [{status}]")

    if insufficient_classes:
        print(f"\n  ⚠ WARNING (C3): {len(insufficient_classes)} classes have fewer than "
              f"{MIN_SAMPLES_PER_CLASS} test samples.")
        print("    Per-class F1 scores are UNRELIABLE at this scale.")
        print("    A single misclassification can swing class F1 by ~8+ points.")
        print("    Recommendation: Add more labeled examples to reach 50+ per class.\n")

    # ── Main evaluation ──────────────────────────────────────────────────────
    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    inference_time = time.perf_counter() - t0

    test_acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"\n  ══ RESULTS ══════════════════════════════════════════════════════")
    print(f"  Test Accuracy  : {test_acc * 100:.1f}%")
    print(f"  Macro F1       : {macro_f1 * 100:.1f}%")
    print(f"  Weighted F1    : {weighted_f1 * 100:.1f}%")
    print(f"  Inference time : {inference_time * 1000:.1f}ms for {len(X_test)} samples "
          f"({inference_time / len(X_test) * 1000:.2f}ms/sample)")

    # ── Classification report ────────────────────────────────────────────────
    print(f"\n  Per-class Report:")
    print(classification_report(y_test, y_pred, target_names=classes, digits=3))

    # ── Confusion matrix ─────────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    print("  Confusion Matrix (rows=actual, cols=predicted):")
    header = f"  {'':22s}" + "  ".join(f"{c[:6]:>6s}" for c in classes)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {classes[i][:22]:22s}" + "  ".join(f"{v:6d}" for v in row))

    # ── Per-class F1 quality gate ────────────────────────────────────────────
    per_class_f1 = f1_score(y_test, y_pred, labels=classes, average=None)
    MIN_F1 = 0.70
    print(f"\n  Per-class F1 quality gate (minimum {MIN_F1:.0%}):")
    all_pass = True
    for cls_name, f1_val in zip(classes, per_class_f1):
        status = "[PASS]" if f1_val >= MIN_F1 else "[FAIL]"
        if f1_val < MIN_F1:
            all_pass = False
        print(f"    {cls_name:22s}: {f1_val:.3f}  {status}")

    if not all_pass:
        print("\n  ⚠ Some classes are below the 0.70 F1 threshold.")

    # ── ADVERSARIAL EVALUATION (M5) ──────────────────────────────────────────
    print(f"\n  ══ ADVERSARIAL TEST SET ({len(ADVERSARIAL_EXAMPLES)} examples) ═══════════")

    adv_texts  = [ex[0] for ex in ADVERSARIAL_EXAMPLES]
    adv_labels = [ex[1] for ex in ADVERSARIAL_EXAMPLES]

    # Filter to examples that meet the chunk length requirement (>15 chars)
    valid_mask = [len(t.strip()) > 15 for t in adv_texts]
    adv_texts_valid  = [t for t, v in zip(adv_texts, valid_mask) if v]
    adv_labels_valid = [l for l, v in zip(adv_labels, valid_mask) if v]
    skipped = sum(1 for v in valid_mask if not v)

    if skipped > 0:
        print(f"  ({skipped} examples skipped — too short for pipeline)")

    if adv_texts_valid:
        adv_pred = model.predict(adv_texts_valid)
        adv_acc = accuracy_score(adv_labels_valid, adv_pred)
        print(f"  Adversarial Accuracy: {adv_acc * 100:.1f}% ({sum(p == l for p, l in zip(adv_pred, adv_labels_valid))}/{len(adv_labels_valid)} correct)")
        print(f"\n  Detail:")
        for text, true, pred in zip(adv_texts_valid, adv_labels_valid, adv_pred):
            match = "✓" if true == pred else "✗"
            print(f"    {match} [{true:20s}] → [{pred:20s}] \"{text[:80]}\"")

    # ── INFERENCE LATENCY BENCHMARK (M2) ─────────────────────────────────────
    print(f"\n  ══ INFERENCE LATENCY BENCHMARK ═══════════════════════════════════")
    # Simulate realistic page payload (50 text chunks)
    benchmark_chunks = (X_test * 5)[:50]  # Repeat to get 50 chunks
    latencies = []
    for _ in range(10):  # 10 runs for P50/P95/P99
        t0 = time.perf_counter()
        _ = model.predict(benchmark_chunks)
        latencies.append((time.perf_counter() - t0) * 1000)  # ms

    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    print(f"  Payload: {len(benchmark_chunks)} chunks × 10 runs")
    print(f"  P50 latency: {p50:.1f}ms")
    print(f"  P95 latency: {p95:.1f}ms")
    print(f"  P99 latency: {p99:.1f}ms")

    if p95 > 300:
        print("  ⚠ P95 > 300ms — ensemble may be too slow for real-time browser extension use.")
    elif p95 > 100:
        print("  ⚠ P95 > 100ms — consider profiling individual models to find bottleneck.")
    else:
        print("  ✓ Latency within acceptable range for real-time use.")

    # ── Export evaluation results to JSON ────────────────────────────────────
    eval_results = {
        "eval_version": "v2.0",
        "split_params": {"test_size": 0.15, "random_state": 42, "stratify": True},
        "dataset_size": len(DATASET),
        "test_size": len(X_test),
        "test_accuracy": round(test_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class_f1": {cls: round(float(f1), 4) for cls, f1 in zip(classes, per_class_f1)},
        "per_class_test_counts": dict(test_dist),
        "insufficient_classes": [{"class": c, "count": n} for c, n in insufficient_classes],
        "adversarial_accuracy": round(adv_acc, 4) if adv_texts_valid else None,
        "latency_ms": {"p50": round(p50, 1), "p95": round(p95, 1), "p99": round(p99, 1)},
        "inference_time_per_sample_ms": round(inference_time / len(X_test) * 1000, 3),
    }

    eval_path = os.path.join(os.path.dirname(__file__), 'eval_results.json')
    with open(eval_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    print(f"\n  [OK] Evaluation results saved → {os.path.abspath(eval_path)}")
    print("=" * 70)

    return eval_results


if __name__ == "__main__":
    evaluate()
