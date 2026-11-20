"""
Run your EXISTING MUUFL-trained fusion model, as-is, on Houston 2013 data.
No fine-tuning, no retraining -- same approach as test_on_trento.py, just
adapted for Houston's 144 HSI bands (vs Trento's 63) and its 15 real
land-cover classes (vs Trento's 6).

IMPORTANT HONEST NOTE: Houston's 15 real classes (Healthy grass, Trees,
Soil, Water, Residential, Commercial, Road, Highway, Railway, Parking Lot 1,
Parking Lot 2, Tennis Court, Running Track, etc.) do NOT correspond to
MUUFL's 11 classes. Predictions printed below are MUUFL class labels forced
onto Houston data -- not a real accuracy benchmark, just a "does the
pipeline run on new data" check, same as the Trento test.

Houston IS a more structurally similar scene to MUUFL than Trento was (both
are urban campus scenes with roads/buildings/trees, vs Trento's rural
vineyards), so this is a genuinely informative second data point on how
well your model's learned features generalize.

Usage:
    python test_on_houston.py --houston_dir path\\to\\Houston2013 \
        --hsimae_repo pretrained_models\\HSIMAE --hsi_checkpoint runs\\hsi_only_v2\\best_model.pt \
        --utonia_repo pretrained_models\\Utonia --lidar_pretrained pretrained_models\\Utonia\\checkpoints\\utonia.pth \
        --lidar_checkpoint runs\\lidar_only\\best_model.pt --fusion_checkpoint runs\\fusion\\best_model.pt \
        --use_normal --out houston_test_output

Requires: torch, numpy, scipy, scikit-learn, matplotlib
"""

import argparse
import os
import sys

import numpy as np
import torch
from scipy.io import loadmat
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from finetune_fusion import (
    build_hsi_extractor, build_lidar_extractor, FusionModel, move_point_dict,
    CLASS_NAMES, GRID_SIZE,
)

TARGET_PATCH_SIZE = 9
GWPCA_BANDS = 32
GWPCA_GROUPS = 4
TARGET_LIDAR_CHANNELS = 4

HOUSTON_CLASS_NAMES = [
    "Healthy grass", "Stressed grass", "Synthetic grass", "Trees", "Soil",
    "Water", "Residential", "Commercial", "Road", "Highway", "Railway",
    "Parking Lot 1", "Parking Lot 2", "Tennis Court", "Running Track",
]


def _find_array(mat_dict, hint_keys):
    for key in hint_keys:
        if key in mat_dict:
            return np.array(mat_dict[key])
    for key, value in mat_dict.items():
        if not key.startswith("__"):
            return np.array(value)
    raise ValueError(f"No usable array found. Keys were: {list(mat_dict.keys())}")


def center_crop(patches, target_size):
    n, h, w, c = patches.shape
    if h == target_size and w == target_size:
        return patches
    start_h = (h - target_size) // 2
    start_w = (w - target_size) // 2
    return patches[:, start_h:start_h + target_size, start_w:start_w + target_size, :]


def apply_gwpca(patches, nc=32, num_groups=4):
    n, h, w, c = patches.shape
    flat = patches.reshape(-1, c).astype(np.float64)
    fmin, fmax = flat.min(), flat.max()
    flat = (flat - fmin) / (fmax - fmin + 1e-12)

    groups = [flat]
    for _ in range(int(np.log2(num_groups))):
        new_groups = []
        for g in groups:
            mid = g.shape[1] // 2
            new_groups.append(g[:, :mid])
            new_groups.append(g[:, mid:])
        groups = new_groups

    per_group = nc // num_groups
    outputs = []
    for g in groups:
        n_comp = min(per_group, g.shape[1])
        pca = PCA(n_components=n_comp, whiten=True, random_state=42)
        outputs.append(pca.fit_transform(g))

    result = np.concatenate(outputs, axis=-1).astype(np.float32)
    return result.reshape(n, h, w, -1)


def normalize(patches):
    n, h, w, c = patches.shape
    flat = patches.reshape(-1, c)
    mean = flat.mean(axis=0, keepdims=True)
    std = flat.std(axis=0, keepdims=True) + 1e-8
    return ((flat - mean) / std).reshape(n, h, w, c)


def load_houston_test_patches(houston_dir, max_samples=500):
    hsi_raw = loadmat(os.path.join(houston_dir, "HSI_Te.mat"), simplify_cells=True)
    print(f"HSI_Te.mat keys: {list(hsi_raw.keys())}")
    hsi = _find_array(hsi_raw, ["Data", "HSI", "hsi"])

    lidar_raw = loadmat(os.path.join(houston_dir, "LIDAR_Te.mat"), simplify_cells=True)
    print(f"LIDAR_Te.mat keys: {list(lidar_raw.keys())}")
    lidar = _find_array(lidar_raw, ["Data", "LIDAR", "Lidar"])

    label_raw = loadmat(os.path.join(houston_dir, "TeLabel.mat"), simplify_cells=True)
    print(f"TeLabel.mat keys: {list(label_raw.keys())}")
    labels = _find_array(label_raw, ["Label", "TeLabel"]).flatten()

    print(f"Raw shapes -- HSI: {hsi.shape}, LiDAR: {lidar.shape}, labels: {labels.shape}")

    if len(labels) > max_samples:
        idx = np.random.RandomState(42).choice(len(labels), max_samples, replace=False)
        hsi, lidar, labels = hsi[idx], lidar[idx], labels[idx]
        print(f"Subsampled to {max_samples} patches for a quick demo run.")

    if hsi.ndim == 3:
        hsi = hsi[:, :, :, None]
    if lidar.ndim == 3:
        lidar = lidar[:, :, :, None]

    hsi = center_crop(hsi, TARGET_PATCH_SIZE)
    lidar = center_crop(lidar, TARGET_PATCH_SIZE)

    hsi_raw_for_viz = hsi.copy()

    print(f"Applying GWPCA ({hsi.shape[-1]} -> {GWPCA_BANDS} bands)...")
    hsi = apply_gwpca(hsi, nc=GWPCA_BANDS, num_groups=GWPCA_GROUPS)
    hsi = normalize(hsi)

    lidar = normalize(lidar)
    c = lidar.shape[-1]
    if c < TARGET_LIDAR_CHANNELS:
        pad = TARGET_LIDAR_CHANNELS - c
        lidar = np.concatenate(
            [lidar, np.zeros((*lidar.shape[:3], pad), dtype=np.float32)], axis=-1
        )
        print(f"Padded LiDAR from {c} to {TARGET_LIDAR_CHANNELS} channels.")
    elif c > TARGET_LIDAR_CHANNELS:
        lidar = lidar[:, :, :, :TARGET_LIDAR_CHANNELS]

    return hsi, lidar, labels, hsi_raw_for_viz


def make_false_color(patch):
    n_bands = patch.shape[-1]
    idx_r, idx_g, idx_b = n_bands - 1, n_bands // 2, 0
    composite = patch[:, :, [idx_r, idx_g, idx_b]].astype(np.float32)
    composite = (composite - composite.min()) / (composite.max() - composite.min() + 1e-8)
    return composite


def save_before_after_grid(hsi_raw_patches, preds, true_labels, out_path, n_examples=16):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed -- skipping before/after grid.")
        return

    n = min(n_examples, len(hsi_raw_patches))
    idx = np.random.RandomState(0).choice(len(hsi_raw_patches), n, replace=False)

    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4.5 * rows))
    axes = axes.flatten()

    for ax_i, i in enumerate(idx):
        composite = make_false_color(hsi_raw_patches[i])
        ax = axes[ax_i]
        ax.imshow(composite)
        pred_name = CLASS_NAMES[preds[i]]
        true_id = int(true_labels[i])
        true_name = HOUSTON_CLASS_NAMES[true_id - 1] if 1 <= true_id <= len(HOUSTON_CLASS_NAMES) else f"id={true_id}"
        ax.set_title(f"Before: raw patch\nAfter: predicted '{pred_name}'\n"
                     f"(Houston's real class: {true_name})", fontsize=9)
        ax.axis("off")

    for ax_i in range(len(idx), len(axes)):
        axes[ax_i].axis("off")

    plt.suptitle("Per-patch before/after (no spatial map possible -- Houston "
                  "patches have no coordinate metadata)", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved before/after patch grid to {out_path}")


def build_batch(hsi_patches, lidar_patches, use_normal):
    win = TARGET_PATCH_SIZE
    ys, xs = np.meshgrid(np.arange(win) - win // 2, np.arange(win) - win // 2, indexing="ij")
    rel_x = xs.flatten().astype(np.float32)
    rel_y = ys.flatten().astype(np.float32)

    hsi_tensors, coords, feats, counts = [], [], [], []
    for i in range(len(hsi_patches)):
        hsi_tensor = torch.tensor(hsi_patches[i], dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)
        hsi_tensors.append(hsi_tensor)

        flat = lidar_patches[i].reshape(-1, 4)
        z = flat[:, 0]
        color_like = flat[:, 1:4].astype(np.float32)
        coord = np.stack([rel_x, rel_y, z], axis=1).astype(np.float32)
        normal = np.zeros_like(coord, dtype=np.float32)

        coord_t = torch.from_numpy(coord)
        color_t = torch.from_numpy(color_like)
        normal_t = torch.from_numpy(normal)
        feat_parts = [coord_t, color_t]
        if use_normal:
            feat_parts.append(normal_t)
        feats.append(torch.cat(feat_parts, dim=1))
        coords.append(coord_t)
        counts.append(coord_t.shape[0])

    hsi_batch = torch.stack(hsi_tensors, dim=0)
    coord_batch = torch.cat(coords, dim=0).float()
    feat_batch = torch.cat(feats, dim=0).float()
    offset = torch.cumsum(torch.tensor(counts, dtype=torch.long), dim=0)

    point_dict = {"coord": coord_batch, "feat": feat_batch, "offset": offset, "grid_size": GRID_SIZE}
    return hsi_batch, point_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--houston_dir", required=True)
    parser.add_argument("--hsimae_repo", default=r"pretrained_models\HSIMAE")
    parser.add_argument("--hsi_checkpoint", default=r"runs\hsi_only_v2\best_model.pt")
    parser.add_argument("--utonia_repo", default=r"pretrained_models\Utonia")
    parser.add_argument("--lidar_pretrained", default=r"pretrained_models\Utonia\checkpoints\utonia.pth")
    parser.add_argument("--lidar_checkpoint", default=r"runs\lidar_only\best_model.pt")
    parser.add_argument("--fusion_checkpoint", default=r"runs\fusion\best_model.pt")
    parser.add_argument("--use_normal", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--out", default="houston_test_output")
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.out, exist_ok=True)

    hsi_patches, lidar_patches, true_labels, hsi_raw_for_viz = load_houston_test_patches(
        args.houston_dir, max_samples=args.max_samples
    )
    print(f"Prepared {len(hsi_patches)} patches for inference.")

    hsi_extractor, hsi_dim = build_hsi_extractor(args.hsimae_repo, args.hsi_checkpoint, device=device)
    lidar_extractor = build_lidar_extractor(
        args.utonia_repo, args.lidar_pretrained, args.lidar_checkpoint, args.use_normal, device
    )
    with torch.no_grad():
        sample_hsi, sample_point_dict = build_batch(hsi_patches[:2], lidar_patches[:2], args.use_normal)
        sample_point_dict = move_point_dict(sample_point_dict, device)
        lidar_dim = lidar_extractor(sample_point_dict).shape[-1]

    model = FusionModel(hsi_extractor, lidar_extractor, hsi_dim, lidar_dim, num_classes=len(CLASS_NAMES)).to(device)
    model.load_state_dict(torch.load(args.fusion_checkpoint, map_location=device))
    model.eval()
    print(f"Loaded fusion model from {args.fusion_checkpoint}")

    all_preds = []
    with torch.no_grad():
        for start in range(0, len(hsi_patches), args.batch_size):
            end = min(start + args.batch_size, len(hsi_patches))
            hsi_batch, point_dict = build_batch(
                hsi_patches[start:end], lidar_patches[start:end], args.use_normal
            )
            hsi_batch = hsi_batch.to(device)
            point_dict = move_point_dict(point_dict, device)
            logits = model(hsi_batch, point_dict)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)

    all_preds = np.array(all_preds)

    print("\n" + "=" * 60)
    print("RESULT: pipeline ran successfully end-to-end on Houston 2013 data.")
    print("=" * 60)
    print(f"\nPredicted MUUFL-class distribution across {len(all_preds)} Houston patches "
          f"(these are NOT Houston's real classes -- see note at top of script):")
    unique, counts = np.unique(all_preds, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  {CLASS_NAMES[u]:<25s}: {c} patches ({c/len(all_preds):.1%})")

    print(f"\nFor reference, Houston's own true labels present in this sample "
          f"(15 real classes, unrelated to the above): {sorted(set(true_labels.tolist()))}")

    np.save(os.path.join(args.out, "houston_predictions.npy"), all_preds)
    print(f"\nRaw predictions saved to {args.out}/houston_predictions.npy")

    save_before_after_grid(
        hsi_raw_for_viz, all_preds, true_labels,
        os.path.join(args.out, "before_after_patches.png"),
        n_examples=16,
    )


if __name__ == "__main__":
    main()