"""
HSI-only fine-tuning baseline for MUUFL Gulfport camouflage/target detection.

Uses HSIMAE's HSIViT class (spectral-spatial transformer encoder + a linear
classification head) initialized from the pretrained HSIMAE-Large checkpoint,
then fine-tuned on your preprocessed HSI patches.

This is intentionally the SIMPLER classification-only setup (not the
semi-supervised dual-branch reconstruction+classification approach in
HSIMAE's own Model_Finetuning.py), since we have fully labeled data and no
prepared unlabeled patch set. This gives you a clean HSI-only baseline to
compare against your LiDAR-only and fusion results later.

Usage:
    python train_hsi_only.py --data_dir preprocessed --hsimae_repo pretrained_models/HSIMAE \
                              --pretrained pretrained_models/HSIMAE/checkpoints/HSIMAE_L.pkl \
                              --out runs/hsi_only

Requires: torch, numpy, scikit-learn
    pip install torch numpy scikit-learn
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix

# MUUFL Gulfport class names, 0-indexed after subtracting 1 from raw labels
CLASS_NAMES = [
    "Trees", "Mostly Grass", "Mixed Ground Surface", "Dirt and Sand",
    "Road", "Water", "Shadow of Buildings", "Buildings", "Sidewalk",
    "Yellow Curb", "Cloth Panels (TARGET)",
]
TARGET_CLASS_IDX = 10  # "Cloth Panels" after 0-indexing


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class HSIPatchDataset(Dataset):
    """
    Wraps a preprocessed .npz split, using only the 'hsi' field.
    Converts each (H, W, Bands) patch to the (1, Bands, H, W) layout HSIViT
    expects: 1 input channel, bands as the "temporal"/spectral axis.
    """

    def __init__(self, npz_path, augment=False):
        data = np.load(npz_path)
        self.hsi = data["hsi"]                 # (N, 9, 9, 32)
        self.labels = data["labels"] - 1        # shift from 1-11 to 0-10
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        patch = self.hsi[idx].copy()  # (H, W, Bands)

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
# Model loading
# ---------------------------------------------------------------------------
def build_model(num_bands, num_classes, patch_size=9, b_patch_size=8,
                 embed_dim=256, depth=12, s_depth=9, device="cuda"):
    """
    Builds HSIMAE-Large's HSIViT classifier with the shapes matching your
    preprocessed data (9x9 spatial patch, `num_bands` spectral bands).

    embed_dim=256, depth=12, s_depth=9 is the "Large" config used in
    HSIMAE's own Model_Finetuning.py. Use embed_dim=128 for the "Base"
    config if you downloaded the Base checkpoint instead.
    """
    from Models import HSIViT  # imported from the cloned HSIMAE repo

    model = HSIViT(
        img_size=patch_size,
        patch_size=3,             # spatial sub-patch size within the 9x9 window
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
    """
    Loads only the keys that exist in both the pretrained checkpoint and our
    HSIViT model (encoder layers: patch_embed, pos_embed, blocks_1, blocks_2,
    blocks, norm). The classification head (cls_head) is intentionally left
    randomly initialized, since the pretrained model wasn't trained on your
    11 classes.
    """
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

    print(f"Loaded {len(matched)} pretrained tensors, skipped {len(skipped)} "
          f"(expected: skipped keys should mostly be decoder/reconstruction "
          f"layers and the classification head).")
    if len(matched) == 0:
        print("WARNING: zero keys matched -- check that the checkpoint file "
              "and model config (embed_dim, depth, s_depth) actually correspond "
              "to the same HSIMAE variant (Base vs Large).")
    return model


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------
def compute_class_weights(labels, num_classes, device):
    """Inverse-frequency class weights, so the rare target class isn't drowned out."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1  # avoid div by zero
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_epoch(model, dataloader, criterion, optimizer, device, train=True):
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
                optimizer.step()

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

    target_mask = labels == TARGET_CLASS_IDX
    if target_mask.sum() > 0:
        target_recall = (preds[target_mask] == TARGET_CLASS_IDX).mean()
        print(f"Target class (Cloth Panels) recall: {target_recall:.3f} "
              f"({(preds[target_mask] == TARGET_CLASS_IDX).sum()}/{target_mask.sum()} correctly found)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Directory with train.npz/val.npz/test.npz")
    parser.add_argument("--hsimae_repo", required=True, help="Path to the cloned HSIMAE repo (contains Models.py)")
    parser.add_argument("--pretrained", default=None, help="Path to the pretrained .pkl checkpoint")
    parser.add_argument("--out", default="runs/hsi_only", help="Where to save the fine-tuned model")
    parser.add_argument("--embed_dim", type=int, default=256, help="256 for HSIMAE-Large, 128 for Base")
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--s_depth", type=int, default=9)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    args = parser.parse_args()

    sys.path.insert(0, args.hsimae_repo)  # so `from Models import HSIViT` works

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

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    # --- Model ---
    model = build_model(
        num_bands=num_bands, num_classes=num_classes,
        embed_dim=args.embed_dim, depth=args.depth, s_depth=args.s_depth,
        device=device,
    )
    model = load_pretrained_weights(model, args.pretrained, device)

    # --- Loss with class weighting for the rare target class ---
    class_weights = compute_class_weights(train_ds.labels, num_classes, device)
    print("Per-class weights (higher = rarer class, weighted more heavily):")
    for name, w in zip(CLASS_NAMES, class_weights.cpu().numpy()):
        print(f"  {name:<25s} {w:.3f}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # --- Training loop ---
    best_val_target_recall = -1
    best_model_path = os.path.join(args.out, "best_model.pt")

    for epoch in range(args.epochs):
        train_loss, train_preds, train_labels = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_preds, val_labels = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )

        target_mask = val_labels == TARGET_CLASS_IDX
        val_target_recall = (
            (val_preds[target_mask] == TARGET_CLASS_IDX).mean()
            if target_mask.sum() > 0 else 0.0
        )
        val_acc = (val_preds == val_labels).mean()

        print(f"Epoch {epoch+1}/{args.epochs} | train_loss {train_loss:.4f} | "
              f"val_loss {val_loss:.4f} | val_acc {val_acc:.3f} | "
              f"val_target_recall {val_target_recall:.3f}")

        # Save the checkpoint that's best at finding the target class,
        # not just best overall accuracy -- overall accuracy can look great
        # while completely missing the rare target class.
        if val_target_recall > best_val_target_recall:
            best_val_target_recall = val_target_recall
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best model saved (val_target_recall={val_target_recall:.3f})")

    # --- Final test evaluation using the best checkpoint ---
    print("\nLoading best checkpoint for final test evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    _, test_preds, test_labels = run_epoch(
        model, test_loader, criterion, optimizer, device, train=False
    )
    evaluate_and_report(test_preds, test_labels, "TEST (HSI-only)")

    print(f"\nBest model saved to: {best_model_path}")
    print("Use this same evaluation approach for your LiDAR-only and fusion "
          "models so all three are compared fairly on the same test set.")


if __name__ == "__main__":
    main()