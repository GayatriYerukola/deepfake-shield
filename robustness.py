"""
DeepFake Shield — Robustness Testing Pipeline
==============================================
Measures how detection accuracy degrades when deepfake images are
subjected to real-world post-processing (compression, blur, resize, etc.).

This is your RESEARCH CONTRIBUTION: most papers only evaluate on clean
deepfakes. You measure the detection gap under realistic sharing conditions.

Usage
-----
  # Full robustness sweep (uses ensemble by default):
  python robustness.py --dataset path/to/CIFAKE --output robustness_results/

  # Heuristics-only (no model download needed):
  python robustness.py --dataset path/to/CIFAKE --mode heuristics

  # Quick test with 200 images:
  python robustness.py --dataset path/to/CIFAKE --max-images 200

Output
------
  robustness_results/
  ├── results_by_attack.csv      ← accuracy per attack and intensity
  ├── summary.json               ← all metrics in one file
  ├── accuracy_vs_attack.png     ← line chart (main thesis figure)
  ├── f1_heatmap.png             ← F1 heatmap across attacks × intensities
  └── per_image/                 ← per-image scores for each attack (optional)

Expected dataset layout (same as evaluate.py):
  dataset/REAL/  +  dataset/FAKE/
  — or —
  dataset/test/REAL/  +  dataset/test/FAKE/
"""

import argparse
import io
import json
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

# ── Attack definitions ────────────────────────────────────────────────────────

def attack_jpeg(img: Image.Image, quality: int) -> Image.Image:
    """Simulate JPEG compression at a given quality (1-95)."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def attack_resize(img: Image.Image, scale: float) -> Image.Image:
    """Downscale then upscale — simulates thumbnail reconstruction."""
    w, h   = img.size
    small  = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    return small.resize((w, h), Image.LANCZOS)


def attack_gaussian_blur(img: Image.Image, sigma: float) -> Image.Image:
    """Apply Gaussian blur (simulates motion blur or low-resolution capture)."""
    arr     = np.array(img.convert("RGB"))
    blurred = cv2.GaussianBlur(arr, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return Image.fromarray(blurred)


def attack_brightness(img: Image.Image, factor: float) -> Image.Image:
    """Alter brightness — factor < 1 darkens, > 1 brightens."""
    return ImageEnhance.Brightness(img.convert("RGB")).enhance(factor)


def attack_contrast(img: Image.Image, factor: float) -> Image.Image:
    """Alter contrast."""
    return ImageEnhance.Contrast(img.convert("RGB")).enhance(factor)


def attack_whatsapp(img: Image.Image, _unused=None) -> Image.Image:
    """
    Simulate WhatsApp image sharing:
    - Downscale to max 1600px on the longest side
    - JPEG recompress at quality ≈ 82
    This is the most practically relevant attack — most viral deepfakes
    circulate via messaging apps.
    """
    MAX_DIM = 1600
    w, h    = img.size
    if max(w, h) > MAX_DIM:
        scale = MAX_DIM / max(w, h)
        img   = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return attack_jpeg(img, quality=82)


def attack_instagram(img: Image.Image, _unused=None) -> Image.Image:
    """
    Simulate Instagram upload:
    - Resize to max 1080px
    - JPEG at quality ≈ 75
    """
    MAX_DIM = 1080
    w, h    = img.size
    if max(w, h) > MAX_DIM:
        scale = MAX_DIM / max(w, h)
        img   = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return attack_jpeg(img, quality=75)


def attack_noise(img: Image.Image, std: float) -> Image.Image:
    """Add Gaussian noise (simulates sensor noise from screenshot)."""
    arr   = np.array(img.convert("RGB"), dtype=np.float32)
    noise = np.random.normal(0, std, arr.shape).astype(np.float32)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


# ── Attack registry ───────────────────────────────────────────────────────────
# Structure:  attack_name -> list of (label, function, parameter) tuples
# Parameter is passed as the second argument to the attack function.

ATTACKS = {
    "JPEG Compression": [
        ("q=90", attack_jpeg,          90),
        ("q=75", attack_jpeg,          75),
        ("q=60", attack_jpeg,          60),
        ("q=45", attack_jpeg,          45),
        ("q=30", attack_jpeg,          30),
    ],
    "Downscale + Upscale": [
        ("×0.9",  attack_resize,       0.9),
        ("×0.75", attack_resize,       0.75),
        ("×0.5",  attack_resize,       0.5),
        ("×0.35", attack_resize,       0.35),
        ("×0.25", attack_resize,       0.25),
    ],
    "Gaussian Blur": [
        ("σ=0.5", attack_gaussian_blur, 0.5),
        ("σ=1.0", attack_gaussian_blur, 1.0),
        ("σ=1.5", attack_gaussian_blur, 1.5),
        ("σ=2.0", attack_gaussian_blur, 2.0),
        ("σ=3.0", attack_gaussian_blur, 3.0),
    ],
    "Brightness Shift": [
        ("+20%",  attack_brightness,   1.20),
        ("+40%",  attack_brightness,   1.40),
        ("-20%",  attack_brightness,   0.80),
        ("-40%",  attack_brightness,   0.60),
        ("-60%",  attack_brightness,   0.40),
    ],
    "Gaussian Noise": [
        ("σ=5",   attack_noise,        5.0),
        ("σ=10",  attack_noise,       10.0),
        ("σ=20",  attack_noise,       20.0),
        ("σ=30",  attack_noise,       30.0),
        ("σ=50",  attack_noise,       50.0),
    ],
    "Social Media": [
        ("WhatsApp",  attack_whatsapp,  None),
        ("Instagram", attack_instagram, None),
        ("JPEG q=50", attack_jpeg,      50),
        ("JPEG q=35", attack_jpeg,      35),
        ("JPEG q=20", attack_jpeg,      20),
    ],
}


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DeepFake Shield — Robustness Testing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--dataset",    required=True, help="Dataset root directory.")
    p.add_argument("--output",     default="robustness_results")
    p.add_argument("--mode",       choices=["ensemble","model","heuristics"], default="ensemble")
    p.add_argument("--max-images", type=int, default=200,
                   help="Images per attack variant (default 200 — keeps runtime reasonable).")
    p.add_argument("--threshold",  type=float, default=0.50)
    p.add_argument("--attacks",    nargs="+", default=None,
                   help="Which attack categories to run (default: all). "
                        f"Choices: {list(ATTACKS.keys())}")
    return p.parse_args()


# ── Dataset helpers (reused from evaluate.py) ─────────────────────────────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def find_dataset_dirs(root: str):
    base = Path(root)
    for rn, fn in [("REAL","FAKE"),("real","fake"),("Real","Fake")]:
        if (base/rn).is_dir() and (base/fn).is_dir():
            return base/rn, base/fn
    for split in ("test","val","Test","Val"):
        sd = base/split
        if sd.is_dir():
            for rn, fn in [("REAL","FAKE"),("real","fake")]:
                if (sd/rn).is_dir() and (sd/fn).is_dir():
                    print(f"  [dataset] Using '{split}' split.")
                    return sd/rn, sd/fn
    raise FileNotFoundError(f"No REAL/FAKE dirs found under '{root}'.")


def sample_paths(real_dir: Path, fake_dir: Path, n: int):
    def gather(d):
        return sorted(p for p in d.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    real = gather(real_dir)[:n//2]
    fake = gather(fake_dir)[:n//2]
    return real, fake


# ── Scoring helpers ───────────────────────────────────────────────────────────

def get_score_fn(mode: str):
    if mode == "model":
        def fn(path):
            from detector.model_manager import model_manager
            return model_manager.predict(path)["fake_score"]
    elif mode == "heuristics":
        def fn(path):
            from detector.image_detector import analyze_image
            r = analyze_image(path, use_model=False)
            return r["ai_probability"] if not r.get("error") else 0.5
    else:
        def fn(path):
            from detector.image_detector import analyze_image
            r = analyze_image(path, use_model=True)
            return r["ai_probability"] if not r.get("error") else 0.5
    return fn


def evaluate_batch(
    paths_labels: list[tuple[Path, int]],
    attack_fn,
    attack_param,
    score_fn,
    threshold: float,
    tmp_dir: Path,
) -> dict:
    """Apply one attack variant to all images and compute metrics."""
    from sklearn.metrics import (
        accuracy_score, f1_score, roc_auc_score,
    )

    y_true, y_pred, y_prob = [], [], []

    for img_path, label in paths_labels:
        try:
            img        = Image.open(img_path).convert("RGB")
            attacked   = attack_fn(img, attack_param) if attack_param is not None \
                         else attack_fn(img)
            # Save to temp file for score_fn (which expects a path)
            tmp = tmp_dir / f"tmp_{img_path.stem}.jpg"
            attacked.save(tmp, "JPEG", quality=95)
            score = score_fn(str(tmp))
            tmp.unlink(missing_ok=True)
        except Exception:
            score = 0.5

        y_true.append(label)
        y_pred.append(1 if score >= threshold else 0)
        y_prob.append(score)

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")

    return {"accuracy": round(acc, 4), "f1": round(f1, 4), "auc": round(auc, 4)}


# ── Main ──────────────────────────────────────────────────────────────────────

def run_robustness(args: argparse.Namespace):
    print("\n" + "=" * 60)
    print("  DeepFake Shield — Robustness Testing Pipeline")
    print("=" * 60)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    real_dir, fake_dir = find_dataset_dirs(args.dataset)
    real_paths, fake_paths = sample_paths(real_dir, fake_dir, args.max_images)
    all_pairs = [(p, 0) for p in real_paths] + [(p, 1) for p in fake_paths]

    print(f"\n  Images/variant : {len(all_pairs)} ({len(real_paths)} real + {len(fake_paths)} fake)")
    print(f"  Mode           : {args.mode}")

    # ── Load model if needed ──────────────────────────────────────────────
    if args.mode in ("ensemble", "model"):
        from detector.model_manager import model_manager, DEFAULT_MODEL_ID
        print(f"\n  Loading model: {DEFAULT_MODEL_ID}")
        ok = model_manager.load()
        if not ok:
            print(f"  ERROR: {model_manager.load_error}")
            if args.mode == "model":
                sys.exit(1)
            print("  Falling back to heuristics.\n")
            args.mode = "heuristics"
        else:
            print("  Model ready.\n")

    score_fn  = get_score_fn(args.mode)
    tmp_dir   = Path(tempfile.mkdtemp())

    active_attacks = (
        {k: ATTACKS[k] for k in args.attacks if k in ATTACKS}
        if args.attacks else ATTACKS
    )

    # ── Baseline (no attack) ──────────────────────────────────────────────
    print("  Running baseline (no attack)…")
    identity = lambda img, _=None: img
    baseline = evaluate_batch(all_pairs, identity, None, score_fn, args.threshold, tmp_dir)
    print(f"    Accuracy={baseline['accuracy']:.2%}  F1={baseline['f1']:.2%}  AUC={baseline['auc']:.4f}")

    # ── Attack variants ───────────────────────────────────────────────────
    records = []

    for cat_name, variants in active_attacks.items():
        print(f"\n  Attack: {cat_name}")
        for label, fn, param in variants:
            t0 = time.time()
            m  = evaluate_batch(all_pairs, fn, param, score_fn, args.threshold, tmp_dir)
            elapsed = time.time() - t0

            drop_acc = baseline["accuracy"] - m["accuracy"]
            records.append({
                "Attack Category": cat_name,
                "Intensity":       label,
                "Accuracy":        m["accuracy"],
                "F1":              m["f1"],
                "AUC-ROC":         m["auc"],
                "Accuracy Drop":   round(drop_acc, 4),
            })
            print(f"    {label:<12}  Acc={m['accuracy']:.2%}  "
                  f"F1={m['f1']:.2%}  drop={drop_acc:+.2%}  ({elapsed:.1f}s)")

    shutil.rmtree(tmp_dir, ignore_errors=True)

    # ── Save results ──────────────────────────────────────────────────────
    df = pd.DataFrame(records)
    df.to_csv(out_dir / "results_by_attack.csv", index=False)

    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset":      str(Path(args.dataset).resolve()),
        "mode":         args.mode,
        "n_per_variant":len(all_pairs),
        "threshold":    args.threshold,
        "baseline":     baseline,
        "results":      records,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    _save_plots(df, baseline, out_dir)

    # ── Print worst performers ────────────────────────────────────────────
    worst = df.nlargest(5, "Accuracy Drop")[["Attack Category","Intensity","Accuracy","Accuracy Drop"]]
    print("\n  Top 5 most damaging attacks:")
    print(worst.to_string(index=False))
    print(f"\n  Results saved to: {out_dir.resolve()}\n")


# ── Plot generation ───────────────────────────────────────────────────────────

def _save_plots(df: pd.DataFrame, baseline: dict, out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style="whitegrid", font_scale=1.05)
    except ImportError:
        print("  [plots] matplotlib/seaborn not installed — skipping.")
        return

    categories = df["Attack Category"].unique()
    colors     = sns.color_palette("tab10", len(categories))

    # ── 1. Accuracy vs attack intensity (line chart — main thesis figure) ──
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.axhline(baseline["accuracy"], color="black", linestyle="--",
               linewidth=1.5, label=f"Baseline (no attack) {baseline['accuracy']:.1%}")

    for cat, color in zip(categories, colors):
        sub   = df[df["Attack Category"] == cat]
        accs  = sub["Accuracy"].tolist()
        ticks = range(len(accs))
        ax.plot(ticks, accs, marker="o", label=cat, color=color, linewidth=2)

    ax.set_xticks([])
    ax.set_ylabel("Accuracy")
    ax.set_title("Detection Accuracy Under Post-Processing Attacks\n"
                 "(left = mild, right = severe within each attack category)")
    ax.set_ylim(0, 1.05)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_vs_attack.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 2. Accuracy drop bar chart ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    df_sorted = df.sort_values("Accuracy Drop", ascending=False)
    bar_labels = df_sorted["Attack Category"] + " (" + df_sorted["Intensity"] + ")"
    colors_bar = [
        "#d32f2f" if d > 0.10 else ("#f57c00" if d > 0.05 else "#388e3c")
        for d in df_sorted["Accuracy Drop"]
    ]
    ax.barh(bar_labels, df_sorted["Accuracy Drop"], color=colors_bar)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Accuracy Drop from Baseline")
    ax.set_title("Accuracy Drop Per Attack\n(red > 10%, orange > 5%, green ≤ 5%)")
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_drop.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 3. F1 heatmap across attacks × intensities ────────────────────────
    try:
        pivot = df.pivot_table(index="Attack Category", columns="Intensity", values="F1")
        fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns)*1.5), max(4, len(pivot)*0.9)))
        sns.heatmap(
            pivot, annot=True, fmt=".2f", cmap="RdYlGn",
            vmin=0, vmax=1, ax=ax, linewidths=0.5,
        )
        ax.set_title("F1 Score Heatmap (Attack Category × Intensity)")
        fig.tight_layout()
        fig.savefig(out_dir / "f1_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        pass

    print("  Plots: accuracy_vs_attack.png  accuracy_drop.png  f1_heatmap.png")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    run_robustness(args)
