"""
DeepFake Shield — Model Fine-Tuning Script
==========================================
Fine-tunes a Vision Transformer (ViT-base) on the CIFAKE dataset.

Why fine-tune instead of using a pre-trained model directly?
  - The pre-trained model was trained on someone else's data with unknown splits.
  - YOUR fine-tuned model is trained and evaluated on a known, reproducible setup.
  - You can report exact train/val/test splits — required for academic work.
  - Comparing pre-trained vs fine-tuned shows your contribution clearly.

Recommended: Run on Google Colab (free T4 GPU, ~2-3 hours for full dataset)
  See COLAB_INSTRUCTIONS at the bottom of this file.

Usage
-----
  # Quick test (500 images, ~5 min on CPU):
  python train.py --dataset path/to/CIFAKE --max-train 500 --epochs 2 --output my_model/

  # Full training (100k images, needs GPU):
  python train.py --dataset path/to/CIFAKE --epochs 5 --output my_model/

  # After training, use your model in the app:
  # Sidebar → type the local path into the Model ID box → Load Model
"""

import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune ViT for deepfake detection on CIFAKE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--dataset",   required=True,
                   help="Root of CIFAKE dataset (contains train/ and test/).")
    p.add_argument("--output",    default="my_deepfake_model",
                   help="Directory to save the trained model.")
    p.add_argument("--base-model", default="google/vit-base-patch16-224",
                   help="HuggingFace model to fine-tune (default: ViT-base-patch16-224).")
    p.add_argument("--epochs",    type=int,   default=5)
    p.add_argument("--batch-size", type=int,  default=32)
    p.add_argument("--lr",        type=float, default=2e-5,
                   help="Learning rate (default 2e-5 is standard for ViT fine-tuning).")
    p.add_argument("--max-train", type=int,   default=0,
                   help="Max training images (0 = all). Use 5000 for a quick test run.")
    p.add_argument("--max-val",   type=int,   default=0,
                   help="Max validation images (0 = all).")
    p.add_argument("--seed",      type=int,   default=42)
    p.add_argument("--fp16",      action="store_true",
                   help="Use mixed precision (requires CUDA GPU). Auto-enabled on Colab.")
    return p.parse_args()


# ── Dataset ───────────────────────────────────────────────────────────────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
LABELS     = {"REAL": 0, "FAKE": 1}   # 0 = real, 1 = fake


def load_file_list(split_dir: Path, max_images: int) -> list[tuple[Path, int]]:
    """
    Collect (image_path, label) pairs from a CIFAKE split directory.
    split_dir must contain REAL/ and FAKE/ subdirectories.
    """
    pairs = []
    for label_name, label_idx in LABELS.items():
        label_dir = split_dir / label_name
        if not label_dir.exists():
            # Try lowercase
            label_dir = split_dir / label_name.lower()
        if not label_dir.exists():
            raise FileNotFoundError(f"Expected {split_dir / label_name}")

        files = [
            p for p in label_dir.rglob("*")
            if p.suffix.lower() in IMAGE_EXTS
        ]
        pairs.extend((f, label_idx) for f in files)

    random.shuffle(pairs)

    if max_images > 0:
        # Balance classes
        real = [(p, l) for p, l in pairs if l == 0][:max_images // 2]
        fake = [(p, l) for p, l in pairs if l == 1][:max_images // 2]
        pairs = real + fake
        random.shuffle(pairs)

    return pairs


class CIFAKEDataset:
    """
    PyTorch Dataset that loads CIFAKE images and applies ViT preprocessing.

    Each item: (pixel_values tensor, label int)
    """

    def __init__(self, file_list: list[tuple[Path, int]], processor):
        self.file_list = file_list
        self.processor = processor

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx: int) -> dict:
        path, label = self.file_list[idx]

        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # Return a blank image if file is corrupted
            img = Image.new("RGB", (224, 224), color=0)

        # ViTImageProcessor handles resize, normalize, convert to tensor
        encoded = self.processor(images=img, return_tensors="pt")

        return {
            "pixel_values": encoded["pixel_values"].squeeze(0),
            "labels":       label,
        }


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred) -> dict:
    """
    Called by HuggingFace Trainer after each evaluation epoch.
    Returns accuracy, F1, precision, recall.
    """
    from sklearn.metrics import (
        accuracy_score, f1_score,
        precision_score, recall_score, roc_auc_score,
    )
    import torch

    logits, labels = eval_pred
    preds  = np.argmax(logits, axis=1)
    probs  = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]

    acc  = accuracy_score(labels, preds)
    f1   = f1_score(labels, preds, zero_division=0)
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels, preds, zero_division=0)

    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = float("nan")

    return {
        "accuracy":  round(acc,  4),
        "f1":        round(f1,   4),
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
        "auc_roc":   round(auc,  4),
    }


# ── Training ──────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace):
    print("\n" + "=" * 58)
    print("  DeepFake Shield — Model Training")
    print("=" * 58)

    # ── Seed for reproducibility ──────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ── Import heavy deps here (so --help works without them) ─────────────
    try:
        import torch
        from transformers import (
            ViTForImageClassification,
            ViTImageProcessor,
            TrainingArguments,
            Trainer,
        )
        from torch.utils.data import DataLoader
    except ImportError as e:
        print(f"\n  ERROR: {e}")
        print("  Run: pip install torch transformers scikit-learn")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device    : {device.upper()}")
    if device == "cpu":
        print("  WARNING   : Training on CPU is very slow.")
        print("              Use Google Colab for free GPU access.")
        print("              See COLAB_INSTRUCTIONS at the bottom of this file.\n")

    # ── Find dataset splits ───────────────────────────────────────────────
    dataset_root = Path(args.dataset)
    train_dir    = dataset_root / "train"
    test_dir     = dataset_root / "test"

    if not train_dir.exists():
        # Maybe user passed the test/ dir directly — use it for both
        train_dir = dataset_root
        test_dir  = dataset_root
        print("  Note: no train/ split found — using full dataset for both train and val.")

    print(f"  Loading training files…")
    train_files = load_file_list(train_dir, args.max_train)
    print(f"  Loading validation files…")
    val_files   = load_file_list(test_dir,  args.max_val)

    # 80/20 split if train and test are the same directory
    if train_dir == test_dir:
        split_at  = int(len(train_files) * 0.8)
        val_files  = train_files[split_at:]
        train_files = train_files[:split_at]

    print(f"\n  Train images : {len(train_files):,} "
          f"({sum(1 for _,l in train_files if l==0):,} real + "
          f"{sum(1 for _,l in train_files if l==1):,} fake)")
    print(f"  Val images   : {len(val_files):,}")
    print(f"  Base model   : {args.base_model}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  Learning rate: {args.lr}")

    # ── Load processor and model ──────────────────────────────────────────
    print(f"\n  Loading base model: {args.base_model}…")
    processor = ViTImageProcessor.from_pretrained(args.base_model)

    model = ViTForImageClassification.from_pretrained(
        args.base_model,
        num_labels=2,
        id2label={0: "Real", 1: "Fake"},
        label2id={"Real": 0, "Fake": 1},
        ignore_mismatched_sizes=True,  # replaces the classification head
    )
    model.to(device)

    # ── Build datasets ────────────────────────────────────────────────────
    train_dataset = CIFAKEDataset(train_files, processor)
    val_dataset   = CIFAKEDataset(val_files,   processor)

    # ── Training arguments ────────────────────────────────────────────────
    out_dir    = Path(args.output)
    use_fp16   = args.fp16 or (device == "cuda")

    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),

        # Training schedule
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,               # 10% of steps as warmup

        # Evaluation
        eval_strategy="epoch",          # evaluate after every epoch
        save_strategy="epoch",
        load_best_model_at_end=True,    # keep the best checkpoint
        metric_for_best_model="f1",
        greater_is_better=True,

        # Performance
        fp16=use_fp16,
        dataloader_num_workers=0,       # 0 is safest on Windows

        # Logging
        logging_dir=str(out_dir / "logs"),
        logging_steps=50,
        report_to="none",               # disable wandb/tensorboard by default

        # Reproducibility
        seed=args.seed,
    )

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    print("\n  Starting training…\n")
    t_start = datetime.now()

    train_result = trainer.train()

    elapsed = (datetime.now() - t_start).total_seconds()
    print(f"\n  Training complete in {elapsed/60:.1f} minutes.")

    # ── Final evaluation on test set ──────────────────────────────────────
    print("\n  Running final evaluation on test set…")
    eval_result = trainer.evaluate(val_dataset)

    print("\n" + "─" * 46)
    print("  FINAL TEST RESULTS")
    print("─" * 46)
    for k, v in eval_result.items():
        if not k.startswith("eval_runtime"):
            label = k.replace("eval_", "").replace("_", " ").title()
            print(f"  {label:<20}: {v:.4f}")
    print("─" * 46)

    # ── Save model ────────────────────────────────────────────────────────
    print(f"\n  Saving model to: {out_dir}/")
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))
    processor.save_pretrained(str(out_dir))

    # Save training summary
    summary = {
        "trained_at":    datetime.now().isoformat(timespec="seconds"),
        "base_model":    args.base_model,
        "dataset":       str(Path(args.dataset).resolve()),
        "n_train":       len(train_files),
        "n_val":         len(val_files),
        "epochs":        args.epochs,
        "batch_size":    args.batch_size,
        "learning_rate": args.lr,
        "device":        device,
        "train_runtime_sec": round(elapsed, 1),
        "final_metrics": {
            k.replace("eval_", ""): v
            for k, v in eval_result.items()
            if not k.startswith("eval_runtime")
        },
    }
    (out_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"\n  Model saved. To use it in the app:")
    print(f"  Sidebar → Model ID box → type: {out_dir.resolve()}")
    print(f"  Or run: python evaluate.py --dataset {args.dataset} "
          f"--model-id {out_dir.resolve()} --output eval_finetuned/\n")

    # ── Training loss plot ────────────────────────────────────────────────
    _plot_training_history(trainer, out_dir)


# ── Training history plot ─────────────────────────────────────────────────────

def _plot_training_history(trainer, out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        history = trainer.state.log_history

        train_loss = [(h["epoch"], h["loss"])
                      for h in history if "loss" in h and "eval_loss" not in h]
        eval_acc   = [(h["epoch"], h["eval_accuracy"])
                      for h in history if "eval_accuracy" in h]
        eval_f1    = [(h["epoch"], h["eval_f1"])
                      for h in history if "eval_f1" in h]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        if train_loss:
            ep, loss = zip(*train_loss)
            ax1.plot(ep, loss, marker="o", color="#1e5799")
            ax1.set_title("Training Loss")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")

        if eval_acc and eval_f1:
            ep_a, acc = zip(*eval_acc)
            ep_f, f1  = zip(*eval_f1)
            ax2.plot(ep_a, acc, marker="o", label="Accuracy", color="#28a745")
            ax2.plot(ep_f, f1,  marker="s", label="F1 Score",  color="#1e5799")
            ax2.set_title("Validation Metrics")
            ax2.set_xlabel("Epoch")
            ax2.set_ylim(0, 1.05)
            ax2.legend()

        fig.tight_layout()
        fig.savefig(out_dir / "training_history.png", dpi=150)
        plt.close(fig)
        print(f"  Training history plot saved to {out_dir / 'training_history.png'}")

    except Exception:
        pass   # Plotting is optional — don't crash training over it


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    train(args)


# ══════════════════════════════════════════════════════════════════════════════
# COLAB INSTRUCTIONS
# ══════════════════════════════════════════════════════════════════════════════
#
# 1. Go to https://colab.research.google.com
# 2. New notebook → Runtime → Change runtime type → T4 GPU
# 3. Run these cells:
#
#   Cell 1 — Upload your project:
#   ─────────────────────────────
#   from google.colab import files
#   import zipfile, os
#
#   # Zip your deepfake-shield folder on your PC first:
#   # Right-click deepfake-shield → Send to → Compressed (zipped) folder
#
#   uploaded = files.upload()   # upload deepfake-shield.zip
#   with zipfile.ZipFile("deepfake-shield.zip") as z:
#       z.extractall("/content/")
#   os.chdir("/content/deepfake-shield")
#
#   Cell 2 — Install dependencies:
#   ───────────────────────────────
#   !pip install torch transformers scikit-learn matplotlib tqdm -q
#
#   Cell 3 — Download CIFAKE via kagglehub:
#   ────────────────────────────────────────
#   !pip install kagglehub -q
#   import kagglehub
#   path = kagglehub.dataset_download("birdy654/cifake-real-and-ai-generated-synthetic-images")
#   print("Dataset:", path)
#
#   Cell 4 — Train (5 epochs, full dataset, ~2.5 hours on T4):
#   ────────────────────────────────────────────────────────────
#   !python train.py \
#       --dataset {path} \
#       --output /content/my_deepfake_model \
#       --epochs 5 \
#       --batch-size 32 \
#       --fp16
#
#   Cell 5 — Download the trained model:
#   ──────────────────────────────────────
#   import shutil
#   shutil.make_archive("/content/my_deepfake_model", "zip", "/content/my_deepfake_model")
#   files.download("/content/my_deepfake_model.zip")
#
#   Then unzip on your PC into deepfake-shield/my_deepfake_model/
#
#   Cell 6 — Evaluate your fine-tuned model:
#   ──────────────────────────────────────────
#   !python evaluate.py \
#       --dataset {path} \
#       --model-id /content/my_deepfake_model \
#       --output /content/eval_finetuned
#
# ══════════════════════════════════════════════════════════════════════════════
