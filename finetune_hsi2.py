"""
HSI-only fine-tuning baseline for MUUFL Gulfport camouflage/target detection.
UPGRADED VERSION -- brings the same fixes applied to finetune_lidar.py after
its checkpoint-collapse issue, so HSI-only and LiDAR-only are now judged by
an identical, fair standard:

  - Checkpoint selection by target-class F1 (not raw recall) -- the original
    script selected on recall alone, which is vulnerable to a degenerate
    model that just predicts "target" constantly (perfect recall, useless
    precision). Your original test result (precision 0.675, recall 1.000)
    has exactly that fingerprint.
  - Gradient clipping for training stability.
  - LR warmup + cosine decay instead of a flat learning rate.
  - Sqrt-weighted oversampling of rare classes (gentler than full inverse
    frequency) in addition to the existing loss class-weighting.
  - A short frozen-backbone warmup before fine-tuning the full encoder.
  - Early stopping + per-epoch CSV logging.

Saves to a NEW output directory by default (runs/hsi_only_v2) so your
original runs/hsi_only/best_model.pt is left untouched for comparison.

Usage:
    python finetune_hsi_v2.py --data_dir preprocessed --hsimae_repo pretrained_models\\HSIMAE \
                               --pretrained pretrained_models\\HSIMAE\\checkpoints\\HSIMAE_L_m0.8_d12_dim256_dd1_ddim72_9x9_10e.pkl \
                               --out runs\\hsi_only_v2

Requires: torch, numpy, scikit-learn
"""

import argparse
import csv
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import classification_report

CLASS_NAMES = [
    "Trees", "Mostly Grass", "Mixed Ground Surface", "Dirt and Sand",
    "Road", "Water", "Shadow of Buildings", "Buildings", "Sidewalk",
    "Yellow Curb", "Cloth Panels (TARGET)",
]
TARGET_CLASS_IDX = 10


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_warmup_cosine_schedule(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class HSIPatchDataset(Dataset):
    def __init__(self, npz_path, augment=False):
        data = np.load(npz_path)
        self.hsi = data["hsi"]                 # (N, 9, 9, 32)
        self.labels = data["labels"] - 1        # shift from 1-11 to 0-10
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        patch = self.hsi[idx].copy()

        if self.augment:
            if np.random.rand() < 0.5:
                patch = np.flip(patch, axis=0).copy()
            if np.random.rand() < 0.5:
                patch = np.flip(patch, axis=1).copy()

        patch = torch.tensor(patch, dtype=torch.float32)
        patch = patch.unsqueeze(0).permute(0, 3, 1, 2)  # (1, Bands, H, W)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return patch, label


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(num_bands, num_classes, patch_size=9, b_patch_size=8,
                 embed_dim=256, depth=12, s_depth=9, device="cuda"):
    from Models import HSIViT

    model = HSIViT(
        img_size=patch_size,
        patch_size=3,
        in_chans=1,
        bands=num_bands,
        b_patch_size=b_patch_size,
        num_class=num_classes,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=embed_dim // 16,
        s_depth=s_depth,
        trunc_init=True,
        drop_path=0.2,
    ).to(device)
    return model


def load_pretrained_weights(model, pretrained_path, device):
    if not pretrained_path or not os.path.exists(pretrained_path):
        print(f"WARNING: pretrained checkpoint not found at '{pretrained_path}'. "
              f"Training from random initialization instead.")
        return model

    checkpoint = torch.load(pretrained_path, map_location=device)
    model_dict = model.state_dict()

    matched, skipped = [], []
    state_dict = {}
    for key, value in checkpoint.items():
        if key in model_dict and model_dict[key].shape == value.shape:
            state_dict[key] = value
            matched.append(key)
        else:
            skipped.append(key)

    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

    print(f"Loaded {len(matched)} pretrained tensors, skipped {len(skipped)}.")
    if len(matched) == 0:
        print("WARNING: zero keys matched -- check checkpoint/config pairing.")
    return model


def set_encoder_trainable(model, trainable):
    """Freezes/unfreezes everything except the classification head."""
    for name, param in model.named_parameters():
        if "cls_head" not in name:
            param.requires_grad = trainable


# ---------------------------------------------------------------------------
# Class balancing
# ---------------------------------------------------------------------------
def compute_class_weights(labels, num_classes, device):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def make_weighted_sampler(labels, num_classes):
    """Sqrt inverse-frequency oversampling -- gentler than full inverse
    frequency, avoiding the training/validation distribution mismatch that
    caused instability in the LiDAR script before this same fix was applied."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1
    class_weights = 1.0 / np.sqrt(counts)
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(labels), replacement=True)


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------
def target_precision_recall_f1(preds, labels):
    target_mask_true = labels == TARGET_CLASS_IDX
    target_mask_pred = preds == TARGET_CLASS_IDX

    tp = np.logical_and(target_mask_true, target_mask_pred).sum()
    fp = np.logical_and(~target_mask_true, target_mask_pred).sum()
    fn = np.logical_and(target_mask_true, ~target_mask_pred).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def run_epoch(model, dataloader, criterion, optimizer, device, train=True,
              grad_clip=1.0, scheduler=None):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            total_loss += loss.item() * x.size(0)
            preds = outputs.argmax(dim=1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss, np.array(all_preds), np.array(all_labels)


def evaluate_and_report(preds, labels, split_name):
    print(f"\n--- {split_name} results ---")
    print(classification_report(
        labels, preds, target_names=CLASS_NAMES, digits=3, zero_division=0
    ))
    precision, recall, f1 = target_precision_recall_f1(preds, labels)
    print(f"Target class (Cloth Panels): precision {precision:.3f}, "
          f"recall {recall:.3f}, F1 {f1:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--hsimae_repo", required=True)
    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--out", default="runs/hsi_only_v2",
                         help="Separate from your original runs/hsi_only, so the "
                              "old checkpoint is preserved for comparison.")
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--s_depth", type=int, default=9)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--grad_clip", type=float, default=0.5)
    parser.add_argument("--freeze_epochs", type=int, default=5)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    sys.path.insert(0, args.hsimae_repo)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.out, exist_ok=True)

    # --- Data ---
    train_ds = HSIPatchDataset(os.path.join(args.data_dir, "train.npz"), augment=True)
    val_ds = HSIPatchDataset(os.path.join(args.data_dir, "val.npz"), augment=False)
    test_ds = HSIPatchDataset(os.path.join(args.data_dir, "test.npz"), augment=False)

    num_bands = train_ds.hsi.shape[-1]
    num_classes = len(CLASS_NAMES)
    print(f"Bands: {num_bands}, classes: {num_classes}")
    print(f"Train/val/test sizes: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

    train_sampler = make_weighted_sampler(train_ds.labels, num_classes)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    # --- Model ---
    model = build_model(
        num_bands=num_bands, num_classes=num_classes,
        embed_dim=args.embed_dim, depth=args.depth, s_depth=args.s_depth,
        device=device,
    )
    model = load_pretrained_weights(model, args.pretrained, device)

    class_weights = compute_class_weights(train_ds.labels, num_classes, device)
    print("Per-class weights:")
    for name, w in zip(CLASS_NAMES, class_weights.cpu().numpy()):
        print(f"  {name:<25s} {w:.3f}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = max(1, int(0.05 * total_steps))
    scheduler = make_warmup_cosine_schedule(optimizer, warmup_steps, total_steps)

    # --- Training loop ---
    best_val_target_f1 = -1
    best_epoch = -1
    epochs_without_improvement = 0
    best_model_path = os.path.join(args.out, "best_model.pt")
    last_model_path = os.path.join(args.out, "last_model.pt")
    log_path = os.path.join(args.out, "training_log.csv")

    with open(log_path, "w", newline="") as log_file:
        log_writer = csv.writer(log_file)
        log_writer.writerow([
            "epoch", "train_loss", "val_loss", "val_acc",
            "val_target_precision", "val_target_recall", "val_target_f1",
            "encoder_frozen",
        ])

        for epoch in range(args.epochs):
            encoder_frozen = epoch < args.freeze_epochs
            set_encoder_trainable(model, trainable=not encoder_frozen)
            if epoch == args.freeze_epochs:
                print(f"Unfreezing encoder at epoch {epoch+1}.")

            train_loss, _, _ = run_epoch(
                model, train_loader, criterion, optimizer, device,
                train=True, grad_clip=args.grad_clip, scheduler=scheduler,
            )
            val_loss, val_preds, val_labels = run_epoch(
                model, val_loader, criterion, optimizer, device, train=False
            )

            val_precision, val_recall, val_f1 = target_precision_recall_f1(val_preds, val_labels)
            val_acc = (val_preds == val_labels).mean()
            current_lr = optimizer.param_groups[0]["lr"]

            print(f"Epoch {epoch+1}/{args.epochs} | lr {current_lr:.2e} | "
                  f"{'[encoder frozen] ' if encoder_frozen else ''}"
                  f"train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
                  f"val_acc {val_acc:.3f} | val_target_P {val_precision:.3f} | "
                  f"val_target_R {val_recall:.3f} | val_target_F1 {val_f1:.3f}")

            log_writer.writerow([
                epoch + 1, train_loss, val_loss, val_acc,
                val_precision, val_recall, val_f1, encoder_frozen,
            ])
            log_file.flush()

            torch.save(model.state_dict(), last_model_path)

            if val_f1 > best_val_target_f1:
                best_val_target_f1 = val_f1
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                torch.save(model.state_dict(), best_model_path)
                print(f"  -> New best model saved (val_target_F1={val_f1:.3f})")
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= args.early_stop_patience:
                print(f"\nNo improvement in val_target_F1 for "
                      f"{args.early_stop_patience} epochs. Stopping early at "
                      f"epoch {epoch+1} (best was epoch {best_epoch}).")
                break

    print(f"\nBest epoch: {best_epoch} (val_target_F1={best_val_target_f1:.3f})")
    print(f"Full per-epoch metrics logged to: {log_path}")

    # --- Final test evaluation ---
    print("\nLoading best checkpoint for final test evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    _, test_preds, test_labels = run_epoch(
        model, test_loader, criterion, optimizer, device, train=False
    )
    evaluate_and_report(test_preds, test_labels, "TEST (HSI-only, v2)")

    print(f"\nBest model saved to: {best_model_path}")
    print(f"Original (v1) checkpoint untouched at: runs/hsi_only/best_model.pt")


if __name__ == "__main__":
    main()