"""
DeepFake Shield — Evaluation Pipeline
======================================
Evaluates the detector against a labeled dataset and produces:
  • Accuracy, Precision, Recall, F1, AUC-ROC
  • Confusion matrix plot
  • ROC curve plot
  • Score distribution plot
  • Per-image results CSV
  • Summary JSON

Usage
-----
  # Full ensemble (model + heuristics) on CIFAKE test split:
  python evaluate.py --dataset path/to/CIFAKE --output eval_results/

  # Model-only (fastest):
  python evaluate.py --dataset path/to/CIFAKE --mode model --output eval_results/

  # Heuristics-only (no GPU / internet needed):
  python evaluate.py --dataset path/to/CIFAKE --mode heuristics --output eval_results/

  # Limit to first 500 images for a quick sanity check:
  python evaluate.py --dataset path/to/CIFAKE --max-images 500

Expected dataset layout (auto-detected):
  Flat:          dataset/REAL/   dataset/FAKE/
  CIFAKE-style:  dataset/test/REAL/   dataset/test/FAKE/

The CIFAKE dataset can be downloaded free from Kaggle:
  https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DeepFake Shield — Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--dataset", required=True,
        help="Root directory of the evaluation dataset."
    )
    p.add_argument(
        "--output", default="eval_results",
        help="Directory to save results (default: eval_results/)."
    )
    p.add_argument(
        "--mode", choices=["ensemble", "model", "heuristics"],
        default="ensemble",
        help=(
            "ensemble = neural model + heuristics (default)\n"
            "model    = neural model output only\n"
            "heuristics = classical signals only (no model download)"
        ),
    )
    p.add_argument(
        "--max-images", type=int, default=0,
        help="Max images to evaluate (0 = all). Useful for quick tests."
    )
    p.add_argument(
        "--threshold", type=float, default=0.50,
        help="Decision threshold for fake/real classification (default: 0.50)."
    )
    p.add_argument(
        "--model-id", default=None,
        help="HuggingFace model ID OR local path to a fine-tuned model directory "
             "(e.g. ./my_deepfake_model). Default: uses model_manager default."
    )
    return p.parse_args()


# ── Dataset discovery ─────────────────────────────────────────────────────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def find_dataset_dirs(root: str) -> tuple[Path, Path]:
    """
    Locate REAL and FAKE image directories under root.
    Tries flat layout first, then CIFAKE-style split layout.
    """
    base = Path(root)

    candidates = [
        ("REAL", "FAKE"), ("real", "fake"), ("Real", "Fake"),
        ("authentic", "fake"), ("AUTHENTIC", "FAKE"),
    ]

    for rname, fname in candidates:
        rd, fd = base / rname, base / fname
        if rd.is_dir() and fd.is_dir():
            return rd, fd

    for split in ("test", "val", "validation", "Test", "Val"):
        split_dir = base / split
        if split_dir.is_dir():
            for rname, fname in candidates:
                rd, fd = split_dir / rname, split_dir / fname
                if rd.is_dir() and fd.is_dir():
                    print(f"  [dataset] Using '{split}' split.")
                    return rd, fd

    raise FileNotFoundError(
        f"Cannot find REAL/FAKE directories under '{root}'.\n"
        "Expected one of:\n"
        "  {root}/REAL/  +  {root}/FAKE/\n"
        "  {root}/test/REAL/  +  {root}/test/FAKE/"
    )


def collect_image_paths(
    real_dir: Path, fake_dir: Path, max_images: int
) -> tuple[list[Path], list[Path]]:
    """Collect image file paths from both directories."""
    def gather(d: Path) -> list[Path]:
        return sorted(
            p for p in d.rglob("*") if p.suffix.lower() in IMAGE_EXTS
        )

    real_paths = gather(real_dir)
    fake_paths = gather(fake_dir)

    if max_images > 0:
        # Balance the classes at max_images/2 each
        half = max_images // 2
        real_paths = real_paths[:half]
        fake_paths = fake_paths[:half]

    return real_paths, fake_paths


# ── Scoring functions ─────────────────────────────────────────────────────────

def score_model_only(image_path: str) -> float:
    """Return the neural model's fake_score directly."""
    from detector.model_manager import model_manager
    pred = model_manager.predict(image_path)
    return pred["fake_score"]


def score_heuristics_only(image_path: str) -> float:
    """Return the heuristic ensemble score (no neural model)."""
    from detector.image_detector import analyze_image
    result = analyze_image(image_path, use_model=False)
    if result.get("error") or result["ai_probability"] is None:
        return 0.5
    return result["ai_probability"]


def score_ensemble(image_path: str) -> float:
    """Return the full ensemble score (model + heuristics)."""
    from detector.image_detector import analyze_image
    result = analyze_image(image_path, use_model=True)
    if result.get("error") or result["ai_probability"] is None:
        return 0.5
    return result["ai_probability"]


# ── Main evaluation loop ──────────────────────────────────────────────────────

def run_evaluation(args: argparse.Namespace) -> None:
    print("\n" + "=" * 58)
    print("  DeepFake Shield — Evaluation Pipeline")
    print("=" * 58)

    # ── Setup output directory ────────────────────────────────────────────
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Find dataset ──────────────────────────────────────────────────────
    print(f"\n  Dataset : {Path(args.dataset).resolve()}")
    real_dir, fake_dir = find_dataset_dirs(args.dataset)
    real_paths, fake_paths = collect_image_paths(
        real_dir, fake_dir, args.max_images
    )

    n_real  = len(real_paths)
    n_fake  = len(fake_paths)
    n_total = n_real + n_fake

    print(f"  Images  : {n_total:,} ({n_real:,} real  +  {n_fake:,} fake)")
    print(f"  Mode    : {args.mode}")
    print(f"  Output  : {out_dir.resolve()}")

    # ── Load model if needed ──────────────────────────────────────────────
    if args.mode in ("ensemble", "model"):
        from detector.model_manager import model_manager, DEFAULT_MODEL_ID
        model_id = args.model_id or DEFAULT_MODEL_ID
        print(f"\n  Loading model: {model_id}")
        ok = model_manager.load(model_id)
        if not ok:
            print(f"  ERROR: {model_manager.load_error}")
            if args.mode == "model":
                sys.exit(1)
            print("  Falling back to heuristics-only mode.\n")
            args.mode = "heuristics"
        else:
            print("  Model ready.\n")

    # ── Pick scoring function ─────────────────────────────────────────────
    _SCORE_FN = {
        "ensemble":   score_ensemble,
        "model":      score_model_only,
        "heuristics": score_heuristics_only,
    }
    score_fn = _SCORE_FN[args.mode]

    # ── Evaluation loop ───────────────────────────────────────────────────
    try:
        from tqdm import tqdm
        _tqdm = tqdm
    except ImportError:
        # tqdm not installed — simple fallback
        def _tqdm(iterable, **kwargs):
            total = kwargs.get("total", "?")
            for i, item in enumerate(iterable):
                if i % 50 == 0:
                    print(f"    {i}/{total}", flush=True)
                yield item

    all_paths  = [(p, 0) for p in real_paths] + [(p, 1) for p in fake_paths]
    records    = []
    errors     = 0
    t_start    = time.time()

    for img_path, true_label in _tqdm(all_paths, total=n_total, desc="  Analyzing"):
        try:
            score = score_fn(str(img_path))
        except Exception as exc:
            score  = 0.5
            errors += 1

        pred_label = 1 if score >= args.threshold else 0
        records.append({
            "path":       str(img_path),
            "true_label": true_label,       # 0=real, 1=fake
            "pred_label": pred_label,
            "fake_score": round(score, 4),
        })

    elapsed = time.time() - t_start
    print(f"\n  Analyzed {n_total:,} images in {elapsed:.1f}s "
          f"({n_total/elapsed:.1f} img/s)  |  errors: {errors}")

    # ── Compute metrics ───────────────────────────────────────────────────
    df     = pd.DataFrame(records)
    y_true = df["true_label"].to_numpy()
    y_pred = df["pred_label"].to_numpy()
    y_prob = df["fake_score"].to_numpy()

    metrics = _compute_metrics(y_true, y_pred, y_prob)

    # ── Print summary ─────────────────────────────────────────────────────
    _print_summary(metrics, args, elapsed, n_total, errors)

    # ── Save artefacts ────────────────────────────────────────────────────
    _save_results(df, metrics, args, out_dir, elapsed)
    _save_plots(y_true, y_pred, y_prob, out_dir)

    print(f"\n  Results saved to: {out_dir.resolve()}\n")


# ── Metrics ───────────────────────────────────────────────────────────────────

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix,
    )

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")

    cm = confusion_matrix(y_true, y_pred).tolist()

    # Per-class breakdown
    prec_real = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    rec_real  = recall_score(y_true, y_pred, pos_label=0, zero_division=0)

    return {
        "accuracy":          round(acc,  4),
        "precision_fake":    round(prec, 4),
        "recall_fake":       round(rec,  4),
        "f1_fake":           round(f1,   4),
        "precision_real":    round(prec_real, 4),
        "recall_real":       round(rec_real,  4),
        "auc_roc":           round(auc, 4) if not np.isnan(auc) else None,
        "confusion_matrix":  cm,
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

def _save_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    out_dir: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix, roc_curve, auc
    except ImportError:
        print("  [plots] matplotlib/seaborn not installed — skipping plots.")
        return

    sns.set_theme(style="whitegrid", font_scale=1.1)

    # ── 1. Confusion matrix ───────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Pred Real", "Pred Fake"],
        yticklabels=["Act. Real", "Act. Fake"],
    )
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # ── 2. ROC curve ──────────────────────────────────────────────────────
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc     = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "roc_curve.png", dpi=150)
        plt.close(fig)
    except Exception:
        pass

    # ── 3. Score distribution ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    real_scores = y_prob[y_true == 0]
    fake_scores = y_prob[y_true == 1]
    ax.hist(real_scores, bins=40, alpha=0.65, color="#2196F3", label="Real")
    ax.hist(fake_scores, bins=40, alpha=0.65, color="#F44336", label="Fake")
    ax.axvline(x=0.50, color="black", linestyle="--", linewidth=1, label="Threshold 0.50")
    ax.set_xlabel("Predicted Fake Probability")
    ax.set_ylabel("Count")
    ax.set_title("Score Distribution — Real vs Fake")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "score_distribution.png", dpi=150)
    plt.close(fig)

    print("  Plots saved: confusion_matrix.png  roc_curve.png  score_distribution.png")


# ── Output helpers ────────────────────────────────────────────────────────────

def _print_summary(
    metrics: dict,
    args: argparse.Namespace,
    elapsed: float,
    n_total: int,
    errors: int,
) -> None:
    cm = metrics["confusion_matrix"]
    tn, fp = cm[0][0], cm[0][1]
    fn, tp = cm[1][0], cm[1][1]

    print("\n" + "─" * 46)
    print("  RESULTS")
    print("─" * 46)
    print(f"  Accuracy          : {metrics['accuracy']:.2%}")
    print(f"  Precision (Fake)  : {metrics['precision_fake']:.2%}")
    print(f"  Recall    (Fake)  : {metrics['recall_fake']:.2%}")
    print(f"  F1 Score  (Fake)  : {metrics['f1_fake']:.2%}")
    print(f"  Precision (Real)  : {metrics['precision_real']:.2%}")
    print(f"  Recall    (Real)  : {metrics['recall_real']:.2%}")
    auc_str = f"{metrics['auc_roc']:.4f}" if metrics["auc_roc"] else "n/a"
    print(f"  AUC-ROC           : {auc_str}")
    print("─" * 46)
    print(f"  Confusion Matrix")
    print(f"                 Pred Real  Pred Fake")
    print(f"  Actual Real :  {tn:>8,}   {fp:>8,}")
    print(f"  Actual Fake :  {fn:>8,}   {tp:>8,}")
    print("─" * 46)
    if errors:
        print(f"  Skipped (errors): {errors}")
    print(f"  Speed: {n_total/elapsed:.1f} images/sec\n")


def _save_results(
    df: pd.DataFrame,
    metrics: dict,
    args: argparse.Namespace,
    out_dir: Path,
    elapsed: float,
) -> None:
    # Per-image CSV
    df.to_csv(out_dir / "results.csv", index=False)
    print("  Saved: results.csv")

    # Summary JSON
    summary = {
        "generated_at":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset":         str(Path(args.dataset).resolve()),
        "mode":            args.mode,
        "threshold":       args.threshold,
        "n_images":        len(df),
        "elapsed_seconds": round(elapsed, 2),
        "images_per_sec":  round(len(df) / elapsed, 2),
        "metrics":         metrics,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("  Saved: summary.json")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args)
