"""
Cross-modal attention fusion model for MUUFL Gulfport camouflage/target
detection. Combines the HSI branch (HSIMAE/HSIViT) and the LiDAR branch
(PTv3/Utonia), each warm-started from your already fine-tuned unimodal
checkpoints, via a small cross-attention module -- not simple concatenation.

HOW FUSION WORKS HERE:
  1. HSI branch produces one pooled feature vector per patch (its
     classification head is removed -- we want the feature, not the logits).
  2. LiDAR branch produces one pooled feature vector per patch the same way.
  3. Both vectors are projected to a shared dimension and treated as a
     2-token sequence (HSI token, LiDAR token).
  4. A self-attention layer lets each token attend to the other -- so the
     HSI representation gets updated using LiDAR context and vice versa,
     which is what "learns relationships between spectral and geometric
     information" actually means architecturally.
  5. Both updated tokens are concatenated and passed through a final
     classification head.

WARM START: instead of fusing the raw pretrained HSIMAE/Utonia weights, this
loads your already fine-tuned best_model.pt from each unimodal run. This
gives fusion a much stronger starting point (both branches already know how
to classify these 11 classes) rather than re-learning everything from
scratch during fusion training.

Usage:
    python finetune_fusion.py --data_dir preprocessed \
        --hsimae_repo pretrained_models\\HSIMAE \
        --hsi_checkpoint runs\\hsi_only_v2\\best_model.pt \
        --utonia_repo pretrained_models\\Utonia \
        --lidar_pretrained pretrained_models\\Utonia\\checkpoints\\utonia.pth \
        --lidar_checkpoint runs\\lidar_only\\best_model.pt \
        --use_normal \
        --out runs\\fusion
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
GRID_SIZE = 0.05


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
# Dataset: yields BOTH modalities for the same pixel, from the same npz split
# ---------------------------------------------------------------------------
class FusionPatchDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.hsi = data["hsi"]        # (N, 9, 9, 32)
        self.lidar = data["lidar"]    # (N, 9, 9, 4)
        self.labels = data["labels"] - 1

        h, w = self.lidar.shape[1], self.lidar.shape[2]
        ys, xs = np.meshgrid(
            np.arange(h) - h // 2, np.arange(w) - w // 2, indexing="ij"
        )
        self.rel_x = xs.flatten().astype(np.float32)
        self.rel_y = ys.flatten().astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        hsi_patch = self.hsi[idx].copy()  # (9, 9, 32)
        hsi_tensor = torch.tensor(hsi_patch, dtype=torch.float32)
        hsi_tensor = hsi_tensor.unsqueeze(0).permute(0, 3, 1, 2)  # (1, 32, 9, 9)

        lidar_patch = self.lidar[idx]
        flat = lidar_patch.reshape(-1, 4)
        z = flat[:, 0]
        color_like = flat[:, 1:4].astype(np.float32)
        coord = np.stack([self.rel_x, self.rel_y, z], axis=1).astype(np.float32)
        normal = np.zeros_like(coord, dtype=np.float32)

        label = int(self.labels[idx])
        return hsi_tensor, coord, color_like, normal, label


def collate_fusion(batch, use_normal):
    hsi_tensors, coords, feats, labels, counts = [], [], [], [], []

    for hsi_tensor, coord, color, normal, label in batch:
        hsi_tensors.append(hsi_tensor)

        coord_t = torch.from_numpy(coord)
        color_t = torch.from_numpy(color)
        normal_t = torch.from_numpy(normal)
        feat_parts = [coord_t, color_t]
        if use_normal:
            feat_parts.append(normal_t)
        feat_t = torch.cat(feat_parts, dim=1)

        coords.append(coord_t)
        feats.append(feat_t)
        counts.append(coord_t.shape[0])
        labels.append(label)

    hsi_batch = torch.stack(hsi_tensors, dim=0)  # (B, 1, 32, 9, 9)
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
    return hsi_batch, point_dict, labels


# ---------------------------------------------------------------------------
# Feature extractors (branches with their classification heads stripped)
# ---------------------------------------------------------------------------
def build_hsi_extractor(hsimae_repo, checkpoint_path, num_bands=32,
                         embed_dim=256, depth=12, s_depth=9, device="cuda"):
    sys.path.insert(0, hsimae_repo)
    from Models import HSIViT

    model = HSIViT(
        img_size=9, patch_size=3, in_chans=1, bands=num_bands, b_patch_size=8,
        num_class=len(CLASS_NAMES), embed_dim=embed_dim, depth=depth,
        num_heads=embed_dim // 16, s_depth=s_depth, trunc_init=True, drop_path=0.2,
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Loaded fine-tuned HSI checkpoint from {checkpoint_path}")

    feature_dim = model.cls_head.in_features
    model.cls_head = nn.Identity()  # now forward() returns pooled features, not logits
    return model, feature_dim


class LidarFeatureExtractor(nn.Module):
    """PTv3/Utonia backbone + global mean pooling, no classification head."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, point_dict):
        from utonia.utils import offset2batch

        point = self.backbone(point_dict)
        feat = point.feat
        batch_idx = offset2batch(point.offset)

        if torch_scatter is not None:
            pooled = torch_scatter.scatter_mean(feat, batch_idx, dim=0)
        else:
            num_samples = int(batch_idx.max().item()) + 1
            pooled = torch.stack([
                feat[batch_idx == i].mean(dim=0) for i in range(num_samples)
            ])
        return pooled  # (B, C)


def build_lidar_extractor(utonia_repo, pretrained_path, checkpoint_path,
                           use_normal, device):
    sys.path.insert(0, utonia_repo)
    import utonia

    try:
        import flash_attn  # noqa
        backbone = utonia.load(pretrained_path).to(device)
    except ImportError:
        custom_config = dict(enc_patch_size=[1024 for _ in range(5)], enable_flash=False)
        backbone = utonia.load(pretrained_path, custom_config=custom_config).to(device)

    extractor = LidarFeatureExtractor(backbone).to(device)

    # The checkpoint was saved from LidarClassifier (backbone + head); we
    # only need the "backbone.*" keys here, so we filter for those.
    full_state_dict = torch.load(checkpoint_path, map_location=device)
    backbone_state_dict = {
        k[len("backbone."):]: v for k, v in full_state_dict.items()
        if k.startswith("backbone.")
    }
    missing, unexpected = extractor.backbone.load_state_dict(backbone_state_dict, strict=False)
    print(f"Loaded fine-tuned LiDAR backbone from {checkpoint_path} "
          f"({len(backbone_state_dict)} tensors; "
          f"{len(missing)} missing, {len(unexpected)} unexpected)")

    return extractor


# ---------------------------------------------------------------------------
# Cross-modal attention fusion
# ---------------------------------------------------------------------------
class CrossModalFusion(nn.Module):
    def __init__(self, hsi_dim, lidar_dim, fusion_dim=256, num_heads=4,
                 num_classes=11, dropout=0.2):
        super().__init__()
        self.proj_hsi = nn.Linear(hsi_dim, fusion_dim)
        self.proj_lidar = nn.Linear(lidar_dim, fusion_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(fusion_dim)
        self.ffn = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, fusion_dim),
        )
        self.norm2 = nn.LayerNorm(fusion_dim)

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes),
        )

    def forward(self, hsi_feat, lidar_feat):
        hsi_tok = self.proj_hsi(hsi_feat).unsqueeze(1)      # (B, 1, D)
        lidar_tok = self.proj_lidar(lidar_feat).unsqueeze(1)  # (B, 1, D)
        tokens = torch.cat([hsi_tok, lidar_tok], dim=1)      # (B, 2, D)

        attn_out, _ = self.attn(tokens, tokens, tokens)       # self-attention over the 2 tokens
        tokens = self.norm1(tokens + attn_out)
        tokens = self.norm2(tokens + self.ffn(tokens))

        fused = tokens.reshape(tokens.size(0), -1)             # (B, 2*D) -- concat both updated tokens
        return self.classifier(fused)


class FusionModel(nn.Module):
    def __init__(self, hsi_extractor, lidar_extractor, hsi_dim, lidar_dim,
                 fusion_dim=256, num_classes=11):
        super().__init__()
        self.hsi_extractor = hsi_extractor
        self.lidar_extractor = lidar_extractor
        self.fusion = CrossModalFusion(hsi_dim, lidar_dim, fusion_dim, num_classes=num_classes)

    def forward(self, hsi_x, point_dict):
        hsi_feat = self.hsi_extractor(hsi_x)
        lidar_feat = self.lidar_extractor(point_dict)
        return self.fusion(hsi_feat, lidar_feat)


def set_branches_trainable(model, trainable):
    for param in model.hsi_extractor.parameters():
        param.requires_grad = trainable
    for param in model.lidar_extractor.parameters():
        param.requires_grad = trainable


# ---------------------------------------------------------------------------
# Class balancing, metrics, training loop (same pattern as the other scripts)
# ---------------------------------------------------------------------------
def compute_class_weights(labels, num_classes, device):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def make_weighted_sampler(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1
    class_weights = 1.0 / np.sqrt(counts)
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(labels), replacement=True)


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


def move_point_dict(point_dict, device):
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in point_dict.items()}


def run_epoch(model, dataloader, criterion, optimizer, device, train=True,
              grad_clip=1.0, scheduler=None):
    model.train() if train else model.eval()
    total_loss = 0.0
    n_samples = 0
    all_preds, all_labels = [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for hsi_x, point_dict, y in dataloader:
            hsi_x = hsi_x.to(device)
            point_dict = move_point_dict(point_dict, device)
            y = y.to(device)

            outputs = model(hsi_x, point_dict)
            loss = criterion(outputs, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
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

    return total_loss / n_samples, np.array(all_preds), np.array(all_labels)


def evaluate_and_report(preds, labels, split_name):
    print(f"\n--- {split_name} results ---")
    print(classification_report(labels, preds, target_names=CLASS_NAMES, digits=3, zero_division=0))
    precision, recall, f1 = target_precision_recall_f1(preds, labels)
    print(f"Target class (Cloth Panels): precision {precision:.3f}, recall {recall:.3f}, F1 {f1:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--hsimae_repo", required=True)
    parser.add_argument("--hsi_checkpoint", required=True, help="Your fine-tuned HSI-only best_model.pt")
    parser.add_argument("--utonia_repo", required=True)
    parser.add_argument("--lidar_pretrained", required=True, help="Original pretrained utonia.pth")
    parser.add_argument("--lidar_checkpoint", required=True, help="Your fine-tuned LiDAR-only best_model.pt")
    parser.add_argument("--use_normal", action="store_true")
    parser.add_argument("--out", default="runs/fusion")
    parser.add_argument("--fusion_dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32,
                         help="Smaller than the unimodal scripts, since both "
                              "large backbones now run together in memory.")
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--grad_clip", type=float, default=0.5)
    parser.add_argument("--freeze_epochs", type=int, default=5,
                         help="Epochs to train only the fusion module, with "
                              "both branches frozen, before fine-tuning end-to-end.")
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.out, exist_ok=True)

    # --- Data ---
    train_ds = FusionPatchDataset(os.path.join(args.data_dir, "train.npz"))
    val_ds = FusionPatchDataset(os.path.join(args.data_dir, "val.npz"))
    test_ds = FusionPatchDataset(os.path.join(args.data_dir, "test.npz"))
    print(f"Train/val/test sizes: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

    def make_loader(ds, sampler=None, shuffle=False):
        return DataLoader(
            ds, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler,
            collate_fn=lambda batch: collate_fusion(batch, args.use_normal),
        )

    train_sampler = make_weighted_sampler(train_ds.labels, len(CLASS_NAMES))
    train_loader = make_loader(train_ds, sampler=train_sampler)
    val_loader = make_loader(val_ds)
    test_loader = make_loader(test_ds)

    # --- Model: warm-started from both fine-tuned unimodal checkpoints ---
    hsi_extractor, hsi_dim = build_hsi_extractor(
        args.hsimae_repo, args.hsi_checkpoint, device=device
    )
    lidar_extractor = build_lidar_extractor(
        args.utonia_repo, args.lidar_pretrained, args.lidar_checkpoint,
        args.use_normal, device
    )

    # Determine LiDAR feature dim with a dummy forward pass
    sample_hsi, sample_point_dict, _ = collate_fusion(
        [train_ds[i] for i in range(2)], args.use_normal
    )
    sample_point_dict = move_point_dict(sample_point_dict, device)
    with torch.no_grad():
        lidar_dim = lidar_extractor(sample_point_dict).shape[-1]
    print(f"HSI feature dim: {hsi_dim}, LiDAR feature dim: {lidar_dim}")

    model = FusionModel(
        hsi_extractor, lidar_extractor, hsi_dim, lidar_dim,
        fusion_dim=args.fusion_dim, num_classes=len(CLASS_NAMES),
    ).to(device)

    class_weights = compute_class_weights(train_ds.labels, len(CLASS_NAMES), device)
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
            "branches_frozen",
        ])

        for epoch in range(args.epochs):
            branches_frozen = epoch < args.freeze_epochs
            set_branches_trainable(model, trainable=not branches_frozen)
            if epoch == args.freeze_epochs:
                print(f"Unfreezing both branches at epoch {epoch+1} "
                      f"(fusion module trains alone before this point).")

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
                  f"{'[branches frozen] ' if branches_frozen else ''}"
                  f"train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
                  f"val_acc {val_acc:.3f} | val_target_P {val_precision:.3f} | "
                  f"val_target_R {val_recall:.3f} | val_target_F1 {val_f1:.3f}")

            log_writer.writerow([
                epoch + 1, train_loss, val_loss, val_acc,
                val_precision, val_recall, val_f1, branches_frozen,
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
    evaluate_and_report(test_preds, test_labels, "TEST (Fusion)")

    print(f"\nBest model saved to: {best_model_path}")
    print("\nCompare against:")
    print("  HSI-only:   test F1 = 0.876 (precision 0.791, recall 0.981)")
    print("  LiDAR-only: test F1 = 0.943 (precision 0.962, recall 0.926)")


if __name__ == "__main__":
    main()