"""
LiDAR-only fine-tuning baseline for MUUFL Gulfport camouflage/target detection.

Uses Utonia's real PointTransformerV3 backbone, loaded via utonia.load(),
with a custom classification head added on top. Each of your preprocessed
9x9x4 LiDAR elevation patches is treated as a small local point cloud
(81 points), fed through the pretrained backbone, globally pooled per
sample, then classified into one of the 11 MUUFL Gulfport classes.

IMPORTANT DESIGN NOTE: Utonia's transform pipeline (GridSample, Collect,
NormalizeColor, etc.) is built for large, RGB-colored, meter-scale scene
point clouds. For small colorless 81-point patches like ours, we skip that
pipeline entirely and build the Point dict directly from coord + feat +
offset + grid_size, which Utonia's own Point structure supports natively
(see structure.py -- grid_coord is auto-derived from coord + grid_size if
not provided). This is simpler and more robust for our use case.

Since MUUFL's LiDAR has no true RGB color, the 4 elevation channels are
split: channel 0 becomes the z-coordinate (height), the remaining 3
channels are used as a color-like feature (not real color) so the model
still receives a coord+color-shaped feature vector similar to what it
expects. This is a genuine domain mismatch worth stating explicitly in your
results/limitations section -- Utonia was never trained on data like this.

Usage:
    python finetune_lidar.py --data_dir preprocessed \
                              --utonia_repo pretrained_models\\Utonia \
                              --pretrained pretrained_models\\Utonia\\checkpoints\\utonia.pth \
                              --out runs\\lidar_only

Requires: run setup_lidar_env.py first.
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

try:
    import torch_scatter
except ImportError:
    torch_scatter = None

CLASS_NAMES = [
    "Trees", "Mostly Grass", "Mixed Ground Surface", "Dirt and Sand",
    "Road", "Water", "Shadow of Buildings", "Buildings", "Sidewalk",
    "Yellow Curb", "Cloth Panels (TARGET)",
]
TARGET_CLASS_IDX = 10
GRID_SIZE = 0.05  # small, since our coordinate spread is only a few units


def set_seed(seed=42):
    """Fixes randomness across numpy/torch/random so results are reproducible
    and honestly comparable between runs, rather than each run being a
    slightly different roll of the dice."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_warmup_cosine_schedule(optimizer, warmup_steps, total_steps):
    """
    A short linear warmup followed by cosine decay. This directly targets
    the instability seen around epochs 9-14 in the previous run: a constant
    learning rate applied from step 1 onward can push a large pretrained
    backbone too hard before its gradients have stabilized. Warmup ramps the
    LR up gradually; cosine decay then eases it down smoothly instead of
    letting it stay high (and noisy) for the whole run.
    """
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Dataset: converts each (9, 9, 4) elevation patch into a small point cloud
# ---------------------------------------------------------------------------
class LidarPatchDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.lidar = data["lidar"]           # (N, 9, 9, 4)
        self.labels = data["labels"] - 1      # shift 1-11 -> 0-10

        h, w = self.lidar.shape[1], self.lidar.shape[2]
        # relative pixel offsets, shared across all samples
        ys, xs = np.meshgrid(
            np.arange(h) - h // 2, np.arange(w) - w // 2, indexing="ij"
        )
        self.rel_x = xs.flatten().astype(np.float32)  # (81,)
        self.rel_y = ys.flatten().astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        patch = self.lidar[idx]                  # (9, 9, 4)
        flat = patch.reshape(-1, 4)               # (81, 4)

        z = flat[:, 0]                             # primary elevation -> height
        color_like = flat[:, 1:4]                   # remaining 3 channels -> pseudo-color

        coord = np.stack([self.rel_x, self.rel_y, z], axis=1).astype(np.float32)  # (81, 3)
        color_like = color_like.astype(np.float32)                                # (81, 3)
        normal = np.zeros_like(coord, dtype=np.float32)                           # (81, 3), unknown

        label = int(self.labels[idx])
        return coord, color_like, normal, label


def collate_points(batch, use_normal):
    """
    Builds a single batched Point-ready dict from a list of
    (coord, color, normal, label) tuples.
    """
    coords, feats, labels = [], [], []
    counts = []

    for coord, color, normal, label in batch:
        coord_t = torch.from_numpy(coord)
        color_t = torch.from_numpy(color)
        normal_t = torch.from_numpy(normal)

        feat_parts = [coord_t, color_t]
        if use_normal:
            feat_parts.append(normal_t)
        feat_t = torch.cat(feat_parts, dim=1)  # (81, 6) or (81, 9)

        coords.append(coord_t)
        feats.append(feat_t)
        counts.append(coord_t.shape[0])
        labels.append(label)

    coord_batch = torch.cat(coords, dim=0).float()
    feat_batch = torch.cat(feats, dim=0).float()
    offset = torch.cumsum(torch.tensor(counts, dtype=torch.long), dim=0)
    labels = torch.tensor(labels, dtype=torch.long)

    point_dict = {
        "coord": coord_batch,
        "feat": feat_batch,
        "offset": offset,
        "grid_size": GRID_SIZE,
    }
    return point_dict, labels


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class LidarClassifier(nn.Module):
    """Wraps the pretrained PointTransformerV3 backbone with a pooled classification head."""

    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.head = None  # built lazily once we know the backbone's output width
        self.num_classes = num_classes

    def _build_head(self, feat_dim, device):
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, self.num_classes),
        ).to(device)
        print(f"Classification head built for backbone output dim {feat_dim}.")

    def forward(self, point_dict):
        from utonia.utils import offset2batch

        point = self.backbone(point_dict)  # returns a Point structure with .feat
        feat = point.feat                   # (total_points_in_batch, C)

        if self.head is None:
            self._build_head(feat.shape[-1], feat.device)

        batch_idx = offset2batch(point.offset)  # (total_points,) which sample each point belongs to
        if torch_scatter is not None:
            pooled = torch_scatter.scatter_mean(feat, batch_idx, dim=0)
        else:
            # fallback if torch_scatter isn't available: manual loop (slower, but works)
            num_samples = int(batch_idx.max().item()) + 1
            pooled = torch.stack([
                feat[batch_idx == i].mean(dim=0) for i in range(num_samples)
            ])

        return self.head(pooled)


def load_backbone(pretrained_path, device):
    import utonia

    try:
        import flash_attn  # noqa
        has_flash = True
    except ImportError:
        has_flash = False

    if has_flash:
        model = utonia.load(pretrained_path).to(device)
    else:
        print("flash-attn not available -- loading with enable_flash=False fallback.")
        custom_config = dict(
            enc_patch_size=[1024 for _ in range(5)],
            enable_flash=False,
        )
        model = utonia.load(pretrained_path, custom_config=custom_config).to(device)

    return model


def infer_in_channels(backbone):
    """Reads the backbone's expected input feature dimension from its embedding stem."""
    try:
        return backbone.embedding.stem.linear.in_features
    except AttributeError:
        print("Could not auto-detect backbone input channels; defaulting to 6 (coord+color).")
        return 6


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------
def compute_class_weights(labels, num_classes, device):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def make_weighted_sampler(labels, num_classes):
    """
    Oversamples rare classes, but using SQRT inverse frequency rather than
    full inverse frequency. Full inverse frequency was found to distort the
    training distribution too aggressively (e.g. the rarest class getting
    sampled ~127x more than its true frequency relative to the majority
    class), which likely contributed to training instability and a model
    that recovers accuracy after a bad epoch but not target precision.
    Sqrt-weighting still meaningfully boosts rare-class exposure with a much
    gentler distortion.
    """
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1
    class_weights = 1.0 / np.sqrt(counts)
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=sample_weights, num_samples=len(labels), replacement=True
    )


def set_backbone_trainable(model, trainable):
    """Freezes or unfreezes the pretrained backbone's parameters."""
    for param in model.backbone.parameters():
        param.requires_grad = trainable


def move_point_dict(point_dict, device):
    return {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in point_dict.items()
    }


def run_epoch(model, dataloader, criterion, optimizer, device, train=True, grad_clip=1.0, scheduler=None):
    model.train() if train else model.eval()
    total_loss = 0.0
    n_samples = 0
    all_preds, all_labels = [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for point_dict, y in dataloader:
            point_dict = move_point_dict(point_dict, device)
            y = y.to(device)

            outputs = model(point_dict)
            loss = criterion(outputs, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping: prevents the occasional exploding-gradient
                # spike that can otherwise wreck a training run for an epoch
                # or two (this is what happened around epoch 35 in the first run).
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            n_samples += batch_size
            preds = outputs.argmax(dim=1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / n_samples
    return avg_loss, np.array(all_preds), np.array(all_labels)


def target_precision_recall_f1(preds, labels):
    """
    Precision/recall/F1 for the target class specifically. Using F1 (not
    just recall) for checkpoint selection prevents a degenerate model that
    predicts "target" for everything from looking like the best model --
    such a model gets perfect recall but near-zero precision, and F1
    correctly penalizes that.
    """
    target_mask_true = labels == TARGET_CLASS_IDX
    target_mask_pred = preds == TARGET_CLASS_IDX

    tp = np.logical_and(target_mask_true, target_mask_pred).sum()
    fp = np.logical_and(~target_mask_true, target_mask_pred).sum()
    fn = np.logical_and(target_mask_true, ~target_mask_pred).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


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
    parser.add_argument("--utonia_repo", required=True, help="Path to the cloned Utonia repo")
    parser.add_argument("--pretrained", required=True, help="Path to the local .pth checkpoint file")
    parser.add_argument("--out", default="runs/lidar_only")
    parser.add_argument("--use_normal", action="store_true",
                         help="Include a (zero) normal feature. Only set this if the "
                              "printed 'expected input channels' is 9, not 6.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--grad_clip", type=float, default=0.5)
    parser.add_argument("--freeze_epochs", type=int, default=5,
                         help="Number of initial epochs to train only the "
                              "classification head, with the backbone frozen. "
                              "Helps stabilize training before fine-tuning the "
                              "full 137M-parameter backbone.")
    parser.add_argument("--early_stop_patience", type=int, default=10,
                         help="Stop if val_target_F1 hasn't improved for this "
                              "many epochs.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    sys.path.insert(0, args.utonia_repo)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.out, exist_ok=True)

    # --- Data ---
    train_ds = LidarPatchDataset(os.path.join(args.data_dir, "train.npz"))
    val_ds = LidarPatchDataset(os.path.join(args.data_dir, "val.npz"))
    test_ds = LidarPatchDataset(os.path.join(args.data_dir, "test.npz"))
    print(f"Train/val/test sizes: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

    def make_loader(ds, shuffle, sampler=None):
        return DataLoader(
            ds, batch_size=args.batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            collate_fn=lambda batch: collate_points(batch, args.use_normal),
        )

    train_sampler = make_weighted_sampler(train_ds.labels, len(CLASS_NAMES))
    train_loader = make_loader(train_ds, shuffle=False, sampler=train_sampler)
    val_loader = make_loader(val_ds, shuffle=False)
    test_loader = make_loader(test_ds, shuffle=False)

    # --- Model ---
    backbone = load_backbone(args.pretrained, device)
    expected_in_channels = infer_in_channels(backbone)
    print(f"Backbone expects {expected_in_channels} input channels "
          f"(6 = coord+color, 9 = coord+color+normal).")
    if expected_in_channels == 9 and not args.use_normal:
        print("WARNING: backbone expects 9 channels but --use_normal was not set. "
              "Re-run with --use_normal.")
    elif expected_in_channels == 6 and args.use_normal:
        print("WARNING: backbone expects 6 channels but --use_normal was set. "
              "Re-run without --use_normal.")

    model = LidarClassifier(backbone, num_classes=len(CLASS_NAMES)).to(device)

    # Run one dummy batch through to force-build the classification head
    # before constructing the optimizer, so the head's parameters are included.
    sample_point_dict, _ = collate_points([train_ds[i] for i in range(4)], args.use_normal)
    sample_point_dict = move_point_dict(sample_point_dict, device)
    with torch.no_grad():
        model(sample_point_dict)

    class_weights = compute_class_weights(train_ds.labels, len(CLASS_NAMES), device)
    print("Per-class weights:")
    for name, w in zip(CLASS_NAMES, class_weights.cpu().numpy()):
        print(f"  {name:<25s} {w:.3f}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = max(1, int(0.05 * total_steps))  # 5% warmup
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
            "backbone_frozen",
        ])

        for epoch in range(args.epochs):
            backbone_frozen = epoch < args.freeze_epochs
            set_backbone_trainable(model, trainable=not backbone_frozen)
            if epoch == args.freeze_epochs:
                print(f"Unfreezing backbone at epoch {epoch+1}.")

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
                  f"{'[backbone frozen] ' if backbone_frozen else ''}"
                  f"train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
                  f"val_acc {val_acc:.3f} | val_target_P {val_precision:.3f} | "
                  f"val_target_R {val_recall:.3f} | val_target_F1 {val_f1:.3f}")

            log_writer.writerow([
                epoch + 1, train_loss, val_loss, val_acc,
                val_precision, val_recall, val_f1, backbone_frozen,
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
    _, test_preds, test_labels = run_epoch(model, test_loader, criterion, optimizer, device, train=False)
    evaluate_and_report(test_preds, test_labels, "TEST (LiDAR-only)")

    print(f"\nBest model saved to: {best_model_path}")


if __name__ == "__main__":
    main()