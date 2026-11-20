"""
End-to-end inference pipeline for the camouflaged-target detection system.

Takes a RAW MUUFL Gulfport .mat file (the same kind used for training) and
runs the full pipeline: preprocessing (GWPCA + normalization + patch
extraction) -> the trained fusion model -> a full-scene class map and a
target-confidence heatmap. Unlike training, this predicts on EVERY pixel in
the scene, not just the labeled ones -- this is what real deployment on a
new/unlabeled scene would look like.

NEW IN THIS VERSION:
  Output folders are now auto-timestamped under --out, so every run gets
    its own subfolder and nothing from a previous run is ever overwritten.

If ground-truth labels are available (for sanity-checking on your existing
scene), it also prints accuracy/F1 restricted to the labeled pixels, so you
can confirm the pipeline reproduces your known test results.

Usage:
    python inference_pipeline.py \
        --data MUUFLGulfport-master\\MUUFLGulfportDataCollection\\muufl_gulfport_campus_w_lidar_1.mat \
        --labels MUUFLGulfport-master\\MUUFLGulfportSceneLabels\\muufl_gulfport_campus_1_hsi_220_label.mat \
        --hsimae_repo pretrained_models\\HSIMAE \
        --hsi_checkpoint runs\\hsi_only_v2\\best_model.pt \
        --utonia_repo pretrained_models\\Utonia \
        --lidar_pretrained pretrained_models\\Utonia\\checkpoints\\utonia.pth \
        --fusion_checkpoint runs\\fusion\\best_model.pt \
        --use_normal \
        --out detection_output

Requires: torch, numpy, scipy, scikit-learn, matplotlib
"""

import argparse
import datetime
import os
import sys
import time

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

import numpy as np
import torch
from scipy.io import loadmat
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from finetune_fusion import (
    build_hsi_extractor, build_lidar_extractor, FusionModel, move_point_dict,
    CLASS_NAMES, TARGET_CLASS_IDX, GRID_SIZE,
)

HSI_PATCH_SIZE = 9
LIDAR_PATCH_SIZE = 9
GWPCA_BANDS = 32
GWPCA_GROUPS = 4


# ---------------------------------------------------------------------------
# Preprocessing (same logic as preprocess.py, applied to the FULL scene,
# not just labeled pixels)
# ---------------------------------------------------------------------------
def _split_spectral_groups(data_2d, num_groups=4):
    groups = [data_2d]
    steps = int(np.log2(num_groups))
    for _ in range(steps):
        new_groups = []
        for g in groups:
            mid = g.shape[1] // 2
            new_groups.append(g[:, :mid])
            new_groups.append(g[:, mid:])
        groups = new_groups
    return groups


def apply_gwpca(hsi_cube, nc=32, num_groups=4, whiten=True):
    h, w, c = hsi_cube.shape
    flat = hsi_cube.reshape(-1, c).astype(np.float64)
    fmin, fmax = flat.min(), flat.max()
    flat = (flat - fmin) / (fmax - fmin + 1e-12)

    groups = _split_spectral_groups(flat, num_groups)
    components_per_group = nc // num_groups
    pca_outputs = []
    for group_data in groups:
        n_comp = min(components_per_group, group_data.shape[1])
        pca = PCA(n_components=n_comp, whiten=whiten, random_state=42)
        pca_outputs.append(pca.fit_transform(group_data))
    result = np.concatenate(pca_outputs, axis=-1)
    return result.reshape(h, w, -1).astype(np.float32)


def normalize_hsi(hsi_cube):
    h, w, bands = hsi_cube.shape
    flat = hsi_cube.reshape(-1, bands)
    mean = flat.mean(axis=0, keepdims=True)
    std = flat.std(axis=0, keepdims=True) + 1e-8
    return ((flat - mean) / std).reshape(h, w, bands)


def normalize_lidar(lidar):
    if lidar.ndim == 2:
        lidar = lidar[:, :, None]
    result = lidar.copy()
    for c in range(result.shape[-1]):
        channel = result[:, :, c]
        mean, std = channel.mean(), channel.std() + 1e-8
        result[:, :, c] = (channel - mean) / std
    return result


def load_custom_scene(hsi_path, lidar_path, labels_path=None):
    def load_array(path, common_keys):
        if path.lower().endswith(".npy"):
            arr = np.load(path)
            print(f"Loaded {path}: shape {arr.shape} (from .npy)")
            return arr
        elif path.lower().endswith(".mat"):
            raw = loadmat(path, simplify_cells=True)
            print(f"Top-level keys in {path}: {list(raw.keys())}")
            for key in common_keys:
                if key in raw:
                    arr = np.array(raw[key])
                    print(f"Using key '{key}': shape {arr.shape}")
                    return arr
                if "hsi" in raw and isinstance(raw["hsi"], dict) and key in raw["hsi"]:
                    arr = np.array(raw["hsi"][key])
                    print(f"Using nested key 'hsi.{key}': shape {arr.shape}")
                    return arr
            # Fallback: none of the guessed names matched -- just grab the
            # first real (non-metadata) key instead of failing. This handles
            # datasets that use unpredictable variable names (e.g. Trento's
            # "mask_test" for labels).
            for key, value in raw.items():
                if not key.startswith("__"):
                    arr = np.array(value)
                    print(f"None of {common_keys} matched -- falling back to "
                          f"first available key '{key}': shape {arr.shape}")
                    return arr
            raise ValueError(f"No usable array found at all in {path}. Keys were: {list(raw.keys())}")
        else:
            raise ValueError(f"Unsupported file type for {path}. Use .mat or .npy.")

    print(f"Loading custom HSI file: {hsi_path}")
    hsi_cube = load_array(hsi_path, common_keys=["Data", "hsi", "cube", "data"]).astype(np.float32)

    print(f"Loading custom LiDAR file: {lidar_path}")
    lidar_raw = load_array(lidar_path, common_keys=["Lidar", "lidar", "elevation", "z", "data", "Data"])

    if lidar_raw.ndim == 2:
        lidar = lidar_raw[:, :, None]
    elif isinstance(lidar_raw, np.ndarray) and lidar_raw.dtype == object:
        channels = []
        for entry in lidar_raw:
            z = np.array(entry["z"] if isinstance(entry, dict) else entry, dtype=np.float32)
            if z.ndim == 2:
                z = z[:, :, None]
            channels.append(z)
        lidar = np.concatenate(channels, axis=-1)
    else:
        lidar = lidar_raw

    target_channels = 4
    if lidar.shape[-1] < target_channels:
        pad_width = target_channels - lidar.shape[-1]
        print(f"WARNING: LiDAR has {lidar.shape[-1]} channel(s), model expects {target_channels}. "
              f"Padding with {pad_width} zero channel(s). Treat results as exploratory.")
        lidar = np.concatenate(
            [lidar, np.zeros((*lidar.shape[:2], pad_width), dtype=np.float32)], axis=-1
        )
    elif lidar.shape[-1] > target_channels:
        print(f"WARNING: LiDAR has {lidar.shape[-1]} channels, model expects {target_channels}. "
              f"Truncating to the first {target_channels}.")
        lidar = lidar[:, :, :target_channels]

    labels = None
    if labels_path:
        labels = load_array(labels_path, common_keys=["labels", "sceneLabels", "data", "Data", "gt", "groundtruth"]).astype(np.int64)

    shapes = [hsi_cube.shape[:2], lidar.shape[:2]] + ([labels.shape[:2]] if labels is not None else [])
    min_h = min(s[0] for s in shapes)
    min_w = min(s[1] for s in shapes)
    if any(s != (min_h, min_w) for s in shapes):
        print(f"Shapes don't match exactly ({shapes}); cropping all to the shared "
              f"top-left region ({min_h}, {min_w}). Verify your files are actually "
              f"co-registered -- this crop does NOT perform spatial alignment.")
        hsi_cube = hsi_cube[:min_h, :min_w, :]
        lidar = lidar[:min_h, :min_w, :]
        if labels is not None:
            labels = labels[:min_h, :min_w]

    return hsi_cube, lidar, labels, None


def load_raw_scene(data_path, labels_path=None):
    print(f"Loading scene data from: {data_path}")
    data_raw = loadmat(data_path, simplify_cells=True)
    hsi_data = data_raw["hsi"]
    print(f"Available fields in this .mat file: {list(hsi_data.keys())}")
    hsi_cube = np.array(hsi_data["Data"], dtype=np.float32)

    rgb_image = None
    for rgb_key in ("Rgb", "RGB", "rgb"):
        if rgb_key in hsi_data:
            rgb_image = np.array(hsi_data[rgb_key])
            print(f"Found RGB image under key '{rgb_key}', shape {rgb_image.shape}")
            break
    if rgb_image is None:
        print("No RGB field found -- will build a false-color composite from HSI bands instead.")

    lidar_returns = hsi_data["Lidar"]
    lidar_z0 = np.array(lidar_returns[0]["z"], dtype=np.float32)
    lidar_z1 = np.array(lidar_returns[1]["z"], dtype=np.float32)
    lidar = np.concatenate([lidar_z0, lidar_z1], axis=-1)

    labels = None
    if labels_path:
        print(f"Loading ground-truth labels from: {labels_path}")
        labels_raw = loadmat(labels_path, simplify_cells=True)
        labels = np.array(labels_raw["hsi"]["sceneLabels"]["labels"], dtype=np.int64)

        label_h, label_w = labels.shape
        hsi_h, hsi_w, _ = hsi_cube.shape
        if label_w < hsi_w:
            hsi_cube = hsi_cube[:label_h, :label_w, :]
            lidar = lidar[:label_h, :label_w, :]
        if label_h < hsi_h:
            hsi_cube = hsi_cube[:label_h, :, :]
            lidar = lidar[:label_h, :, :]

    return hsi_cube, lidar, labels, rgb_image


def extract_all_patches(hsi_cube, lidar):
    h, w, _ = hsi_cube.shape
    hsi_half = HSI_PATCH_SIZE // 2
    lidar_half = LIDAR_PATCH_SIZE // 2

    hsi_padded = np.pad(hsi_cube, ((hsi_half, hsi_half), (hsi_half, hsi_half), (0, 0)), mode="reflect")
    lidar_padded = np.pad(lidar, ((lidar_half, lidar_half), (lidar_half, lidar_half), (0, 0)), mode="reflect")

    rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    rows, cols = rows.flatten(), cols.flatten()

    return hsi_padded, lidar_padded, rows, cols, hsi_half, lidar_half


def build_batch(hsi_padded, lidar_padded, rows, cols, hsi_half, lidar_half, use_normal):
    hsi_tensors, coords, feats, counts = [], [], [], []

    win = LIDAR_PATCH_SIZE
    ys, xs = np.meshgrid(np.arange(win) - win // 2, np.arange(win) - win // 2, indexing="ij")
    rel_x = xs.flatten().astype(np.float32)
    rel_y = ys.flatten().astype(np.float32)

    for row, col in zip(rows, cols):
        r, c = row + hsi_half, col + hsi_half
        hsi_patch = hsi_padded[r - hsi_half:r + hsi_half + 1, c - hsi_half:c + hsi_half + 1, :]
        hsi_tensor = torch.tensor(hsi_patch, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)
        hsi_tensors.append(hsi_tensor)

        r, c = row + lidar_half, col + lidar_half
        lidar_patch = lidar_padded[r - lidar_half:r + lidar_half + 1, c - lidar_half:c + lidar_half + 1, :]
        flat = lidar_patch.reshape(-1, 4)
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
        feat_t = torch.cat(feat_parts, dim=1)

        coords.append(coord_t)
        feats.append(feat_t)
        counts.append(coord_t.shape[0])

    hsi_batch = torch.stack(hsi_tensors, dim=0)
    coord_batch = torch.cat(coords, dim=0).float()
    feat_batch = torch.cat(feats, dim=0).float()
    offset = torch.cumsum(torch.tensor(counts, dtype=torch.long), dim=0)

    point_dict = {"coord": coord_batch, "feat": feat_batch, "offset": offset, "grid_size": GRID_SIZE}
    return hsi_batch, point_dict


def main():
    # ---------------------------------------------------------------------
    # All paths default to your actual project layout below, so running
    # just "python inference_pipeline.py" with no arguments works as-is.
    # Every value can still be overridden with its flag if you ever need to
    # point at a different file.
    # ---------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", default=r"MUUFLGulfport-master\MUUFLGulfportDataCollection\muufl_gulfport_campus_w_lidar_1.mat",
        help="Single bundled MUUFL-style .mat with both HSI and LiDAR"
    )
    parser.add_argument("--hsi_file", default=None, help="Separate HSI file (.mat or .npy) -- use with --lidar_file")
    parser.add_argument("--lidar_file", default=None, help="Separate LiDAR file (.mat or .npy) -- use with --hsi_file")
    parser.add_argument(
        "--labels", default=r"MUUFLGulfport-master\MUUFLGulfportSceneLabels\muufl_gulfport_campus_1_hsi_220_label.mat",
        help="Optional, for sanity-check accuracy on known pixels"
    )
    parser.add_argument("--hsimae_repo", default=r"pretrained_models\HSIMAE")
    parser.add_argument("--hsi_checkpoint", default=r"runs\hsi_only_v2\best_model.pt")
    parser.add_argument("--utonia_repo", default=r"pretrained_models\Utonia")
    parser.add_argument("--lidar_pretrained", default=r"pretrained_models\Utonia\checkpoints\utonia.pth")
    parser.add_argument(
        "--lidar_checkpoint", default=r"runs\lidar_only\best_model.pt",
        help="Your fine-tuned LiDAR-only best_model.pt. Only used to build the correct "
             "architecture before loading the full fusion checkpoint, which then "
             "overwrites these weights anyway."
    )
    parser.add_argument("--fusion_checkpoint", default=r"runs\fusion\best_model.pt")
    parser.add_argument(
        "--use_normal", default=True, action=argparse.BooleanOptionalAction,
        help="Defaults to True to match how the model was trained. "
             "Pass --no-use_normal to disable."
    )
    parser.add_argument(
        "--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "OUTPUT"),
        help="Base output folder -- each run creates its own timestamped "
             "subfolder inside this, so previous runs are never overwritten."
    )
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_out = os.path.join(args.out, f"run_{timestamp}")
    os.makedirs(run_out, exist_ok=True)
    print(f"This run's output folder: {run_out}")

    if args.hsi_file and args.lidar_file:
        hsi_cube, lidar, labels, rgb_image = load_custom_scene(args.hsi_file, args.lidar_file, args.labels)
    elif args.data:
        hsi_cube, lidar, labels, rgb_image = load_raw_scene(args.data, args.labels)
    else:
        raise ValueError("Provide either --data (bundled .mat) or both --hsi_file and --lidar_file.")
    print(f"Scene shape: HSI {hsi_cube.shape}, LiDAR {lidar.shape}")

    if rgb_image is None:
        n_bands = hsi_cube.shape[-1]
        idx_r, idx_g, idx_b = n_bands - 1, n_bands // 2, 0
        composite = hsi_cube[:, :, [idx_r, idx_g, idx_b]]
        composite = (composite - composite.min()) / (composite.max() - composite.min() + 1e-8)
        rgb_image = composite

    print("Applying GWPCA (72 -> 32 bands)...")
    hsi_cube = apply_gwpca(hsi_cube, nc=GWPCA_BANDS, num_groups=GWPCA_GROUPS)
    hsi_cube = normalize_hsi(hsi_cube)
    lidar = normalize_lidar(lidar)

    h, w, _ = hsi_cube.shape
    hsi_padded, lidar_padded, rows, cols, hsi_half, lidar_half = extract_all_patches(hsi_cube, lidar)
    n_pixels = len(rows)
    print(f"Running inference on all {n_pixels:,d} pixels in the scene...")

    hsi_extractor, hsi_dim = build_hsi_extractor(args.hsimae_repo, args.hsi_checkpoint, device=device)
    lidar_extractor = build_lidar_extractor(
        args.utonia_repo, args.lidar_pretrained, args.lidar_checkpoint, args.use_normal, device
    )
    with torch.no_grad():
        sample_hsi, sample_point_dict = build_batch(
            hsi_padded, lidar_padded, rows[:2], cols[:2], hsi_half, lidar_half, args.use_normal
        )
        sample_point_dict = move_point_dict(sample_point_dict, device)
        lidar_dim = lidar_extractor(sample_point_dict).shape[-1]

    model = FusionModel(hsi_extractor, lidar_extractor, hsi_dim, lidar_dim, num_classes=len(CLASS_NAMES)).to(device)
    model.load_state_dict(torch.load(args.fusion_checkpoint, map_location=device))
    model.eval()
    print(f"Loaded fusion model from {args.fusion_checkpoint}")

    all_preds = np.zeros(n_pixels, dtype=np.int64)
    all_target_probs = np.zeros(n_pixels, dtype=np.float32)

    inference_start_time = time.time()
    with torch.no_grad():
        for start in tqdm(range(0, n_pixels, args.batch_size), desc="Inference batches"):
            end = min(start + args.batch_size, n_pixels)
            batch_rows = rows[start:end]
            batch_cols = cols[start:end]

            hsi_batch, point_dict = build_batch(
                hsi_padded, lidar_padded, batch_rows, batch_cols, hsi_half, lidar_half, args.use_normal
            )
            hsi_batch = hsi_batch.to(device)
            point_dict = move_point_dict(point_dict, device)

            logits = model(hsi_batch, point_dict)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1).cpu().numpy()
            target_probs = probs[:, TARGET_CLASS_IDX].cpu().numpy()

            all_preds[start:end] = preds
            all_target_probs[start:end] = target_probs

    inference_time = time.time() - inference_start_time
    print(f"Inference completed in {inference_time:.2f} seconds.")

    class_map = all_preds.reshape(h, w) + 1
    confidence_map = all_target_probs.reshape(h, w)

    np.save(os.path.join(run_out, "class_map.npy"), class_map)
    np.save(os.path.join(run_out, "target_confidence_map.npy"), confidence_map)
    print(f"\nSaved class_map.npy and target_confidence_map.npy to {run_out}/")

    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from matplotlib.colors import ListedColormap, BoundaryNorm
        from matplotlib.patches import Patch

        CLASS_COLORS_HEX = [
            "#000000", "#228B22", "#90EE90", "#C2B280", "#DEB887", "#505050",
            "#4169E1", "#483D8B", "#A9A9A9", "#D3D3D3", "#FFD700", "#FF0000",
        ]

        def _hex_to_rgb(hex_str):
            hex_str = hex_str.lstrip("#")
            return tuple(int(hex_str[i:i+2], 16) / 255.0 for i in (0, 2, 4))

        class_rgb = [_hex_to_rgb(h) + (1.0,) for h in CLASS_COLORS_HEX]
        class_cmap = ListedColormap(class_rgb, name="muufl_semantic")
        class_bounds = np.arange(-0.5, len(CLASS_COLORS_HEX) + 0.5, 1)
        class_norm = BoundaryNorm(class_bounds, class_cmap.N)

        legend_patches = [
            Patch(facecolor=CLASS_COLORS_HEX[i + 1], edgecolor="black",
                  linewidth=0.5, label=CLASS_NAMES[i])
            for i in range(len(CLASS_NAMES))
        ]

        unique, counts = np.unique(class_map, return_counts=True)
        total_pixels = class_map.size
        stats_lines = ["\nPredicted class distribution:"]
        stats_lines.append(f"  {'Class':<30s} {'Pixels':>10s}  {'%':>6s}")
        stats_lines.append("  " + "-" * 50)
        for cls_id, cls_name in enumerate(CLASS_NAMES, start=1):
            cnt = int(counts[unique == cls_id][0]) if cls_id in unique else 0
            pct = 100.0 * cnt / total_pixels
            marker = "  <<<" if cls_id == TARGET_CLASS_IDX + 1 else ""
            stats_lines.append(f"  {cls_name:<30s} {cnt:>10,d}  {pct:>5.1f}%{marker}")
        stats_text = "\n".join(stats_lines)
        print(stats_text)

        stats_path = os.path.join(run_out, "class_distribution.txt")
        with open(stats_path, "w") as f:
            f.write(stats_text + "\n")
        print(f"Saved class distribution to {stats_path}")

        has_gt = labels is not None
        n_top = 2 + (1 if has_gt else 0)

        fig = plt.figure(figsize=(7 * max(n_top, 3), 14))
        n_cols = max(n_top, 3)

        ax_idx = 0
        ax_input = fig.add_subplot(2, n_cols, ax_idx + 1)
        ax_input.imshow(rgb_image)
        ax_input.set_title("Input Scene (RGB / False-Color)", fontsize=12, fontweight="bold")
        ax_input.axis("off")
        ax_idx += 1

        if has_gt:
            ax_gt = fig.add_subplot(2, n_cols, ax_idx + 1)
            gt_display = np.where(labels == -1, 0, labels).astype(float)
            gt_display_masked = np.ma.masked_where(gt_display == 0, gt_display)
            ax_gt.imshow(np.zeros((*gt_display.shape, 3)), alpha=1.0)
            ax_gt.imshow(gt_display_masked, cmap=class_cmap, norm=class_norm, interpolation="nearest")
            ax_gt.set_title("Ground Truth", fontsize=12, fontweight="bold")
            ax_gt.axis("off")
            ax_idx += 1

        ax_pred = fig.add_subplot(2, n_cols, ax_idx + 1)
        ax_pred.imshow(class_map, cmap=class_cmap, norm=class_norm, interpolation="nearest")
        ax_pred.set_title("Predicted Class Map", fontsize=12, fontweight="bold")
        ax_pred.axis("off")
        ax_pred.legend(
            handles=legend_patches, loc="upper left", bbox_to_anchor=(1.02, 1.0),
            fontsize=8, title="Class Legend", title_fontsize=9,
            frameon=True, fancybox=True, shadow=True, borderpad=0.6, labelspacing=0.4,
        )

        ax_heat = fig.add_subplot(2, n_cols, n_cols + 1)
        im = ax_heat.imshow(confidence_map, cmap="hot", vmin=0, vmax=1)
        ax_heat.set_title("Target (Cloth Panels) Confidence", fontsize=12, fontweight="bold")
        ax_heat.axis("off")
        fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)

        ax_overlay = fig.add_subplot(2, n_cols, n_cols + 2)
        ax_overlay.imshow(rgb_image)
        hot_cmap = cm.colormaps.get("hot", cm.get_cmap("hot")) if hasattr(cm, "colormaps") else cm.get_cmap("hot")
        overlay = hot_cmap(confidence_map)
        overlay[..., 3] = np.where(confidence_map > 0.3, np.clip((confidence_map - 0.3) / 0.7, 0, 1), 0)
        ax_overlay.imshow(overlay)
        ax_overlay.set_title("Target Confidence Overlay", fontsize=12, fontweight="bold")
        ax_overlay.axis("off")

        ax_bar = fig.add_subplot(2, n_cols, n_cols + 3)
        bar_names, bar_counts, bar_colors = [], [], []
        for cls_id, cls_name in enumerate(CLASS_NAMES, start=1):
            cnt = int(counts[unique == cls_id][0]) if cls_id in unique else 0
            bar_names.append(cls_name)
            bar_counts.append(cnt)
            bar_colors.append(CLASS_COLORS_HEX[cls_id])

        y_pos = np.arange(len(bar_names))
        ax_bar.barh(y_pos, bar_counts, color=bar_colors, edgecolor="black", linewidth=0.5, height=0.7)
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(bar_names, fontsize=8)
        ax_bar.invert_yaxis()
        ax_bar.set_xlabel("Pixel Count", fontsize=10)
        ax_bar.set_title("Class Distribution", fontsize=12, fontweight="bold")
        for i, (cnt, name) in enumerate(zip(bar_counts, bar_names)):
            pct = 100.0 * cnt / total_pixels
            ax_bar.text(cnt + total_pixels * 0.005, i, f"{pct:.1f}%", va="center", fontsize=7, fontweight="bold")

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.35, hspace=0.15)
        out_png = os.path.join(run_out, "detection_visualization.png")
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved enhanced visualization to {out_png}")
    except ImportError:
        print("matplotlib not installed -- skipping visualization image (raw .npy files still saved).")

    if labels is not None:
        labeled_mask = labels != -1
        gt = labels[labeled_mask]
        pred = class_map[labeled_mask]
        acc = (gt == pred).mean()
        target_mask = gt == (TARGET_CLASS_IDX + 1)
        target_recall = (pred[target_mask] == (TARGET_CLASS_IDX + 1)).mean() if target_mask.sum() > 0 else 0.0
        print(f"\nSanity check on labeled pixels: accuracy = {acc:.3f}, "
              f"target recall = {target_recall:.3f} "
              f"(should roughly match your reported test results)")

        try:
            from sklearn.metrics import classification_report, confusion_matrix
            import matplotlib.pyplot as plt

            present_classes = sorted(set(gt.tolist()) | set(pred.tolist()))
            present_names = []
            for cls_id in present_classes:
                if 1 <= cls_id <= len(CLASS_NAMES):
                    present_names.append(CLASS_NAMES[cls_id - 1])
                else:
                    present_names.append(f"Class {cls_id}")

            report = classification_report(
                gt, pred, labels=present_classes, target_names=present_names,
                zero_division=0, digits=3,
            )
            print("\n" + "=" * 70)
            print(" PER-CLASS CLASSIFICATION REPORT")
            print("=" * 70)
            print(report)

            report_path = os.path.join(run_out, "classification_report.txt")
            with open(report_path, "w") as f:
                f.write("PER-CLASS CLASSIFICATION REPORT\n")
                f.write("=" * 70 + "\n")
                f.write(report)
                f.write(f"\nOverall accuracy: {acc:.4f}\n")
                f.write(f"Target (Cloth Panels) recall: {target_recall:.4f}\n")
            print(f"Classification report saved to {report_path}")

            cm = confusion_matrix(gt, pred, labels=present_classes)
            fig_cm, ax_cm = plt.subplots(
                figsize=(max(8, len(present_classes) * 0.9), max(7, len(present_classes) * 0.8))
            )
            cm_norm = cm.astype(np.float64)
            row_sums = cm_norm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            cm_pct = cm_norm / row_sums * 100

            im = ax_cm.imshow(cm_pct, interpolation="nearest", cmap="YlOrRd")
            fig_cm.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04, label="Row-normalized %")

            for i in range(len(present_classes)):
                for j in range(len(present_classes)):
                    count = cm[i, j]
                    pct = cm_pct[i, j]
                    if count > 0:
                        text_color = "white" if pct > 50 else "black"
                        ax_cm.text(j, i, f"{count}\n({pct:.1f}%)", ha="center", va="center",
                                   fontsize=6, color=text_color,
                                   fontweight="bold" if i == j else "normal")

            ax_cm.set_xticks(range(len(present_names)))
            ax_cm.set_yticks(range(len(present_names)))
            ax_cm.set_xticklabels(present_names, rotation=45, ha="right", fontsize=8)
            ax_cm.set_yticklabels(present_names, fontsize=8)
            ax_cm.set_xlabel("Predicted Class", fontsize=11, fontweight="bold")
            ax_cm.set_ylabel("True Class", fontsize=11, fontweight="bold")
            ax_cm.set_title(f"Confusion Matrix (Labeled Pixels, Accuracy={acc:.3f})",
                             fontsize=13, fontweight="bold")
            plt.tight_layout()

            cm_path = os.path.join(run_out, "confusion_matrix.png")
            fig_cm.savefig(cm_path, dpi=150, bbox_inches="tight")
            plt.close(fig_cm)
            print(f"Confusion matrix saved to {cm_path}")

        except ImportError:
            print("sklearn not available -- skipping detailed classification report.")
        except Exception as e:
            print(f"Classification report error: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nAll outputs for this run are in: {run_out}")


if __name__ == "__main__":
    main()