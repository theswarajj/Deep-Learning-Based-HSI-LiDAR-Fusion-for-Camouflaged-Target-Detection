"""
HSI-ONLY inference pipeline for the camouflaged-target detection system.

Uses ONLY the fine-tuned HSIViT model (from finetune_hsi2.py) -- no LiDAR
data is loaded or used at all.  This lets you evaluate what hyperspectral
information alone can detect, as a direct comparison against:
  - LiDAR-only   (lidar_inference.py)
  - Fusion model  (inference_pipelineV2.py)

The HSIViT model still has its ORIGINAL classification head (11 classes),
NOT the stripped feature-extractor version used inside the fusion model.
We load best_model.pt from runs/hsi_only_v2 (or whichever HSI checkpoint
you specify).

Outputs (in OUTPUT/HSI/<timestamped_run>/):
  - class_map.npy              : predicted class for every pixel
  - target_confidence_map.npy  : per-pixel confidence for the target class
  - detection_visualization.png: overview figure
  - classification_report.txt  : per-class metrics (if labels are provided)
  - confusion_matrix.png       : confusion matrix (if labels are provided)

Usage:
    python hsi_inference.py
    python hsi_inference.py --hsi_file INPUT\\HSI\\hsi_data.mat --labels INPUT\\HSI\\labels.mat
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

# -- project root on sys.path so we can import from finetune_hsi2 / finetune_hsi --
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

CLASS_NAMES = [
    "Trees", "Mostly Grass", "Mixed Ground Surface", "Dirt and Sand",
    "Road", "Water", "Shadow of Buildings", "Buildings", "Sidewalk",
    "Yellow Curb", "Cloth Panels (TARGET)",
]
TARGET_CLASS_IDX = 10
HSI_PATCH_SIZE = 9
GWPCA_BANDS = 32
GWPCA_GROUPS = 4


# ---------------------------------------------------------------------------
# Preprocessing (identical to inference_pipelineV2.py)
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


# ---------------------------------------------------------------------------
# Scene loading — supports .mat, .npy, ENVI BIL (.bil/.hdr), GeoTIFF (.tif)
# ---------------------------------------------------------------------------
def _parse_envi_wavelengths(hdr_path):
    """Extract wavelength list from an ENVI .hdr file, if present."""
    wavelengths = None
    try:
        with open(hdr_path, "r") as f:
            text = f.read()
        import re
        match = re.search(r"wavelength\s*=\s*\{([^}]+)\}", text, re.IGNORECASE)
        if match:
            wl_str = match.group(1)
            wavelengths = np.array([float(x.strip()) for x in wl_str.split(",")], dtype=np.float32)
    except Exception:
        pass
    return wavelengths


def _find_envi_data_file(hdr_path):
    """Given a .hdr path, find the corresponding binary data file (.bil, .bsq, .bip, or extensionless)."""
    base = hdr_path
    # Strip .hdr extension(s)
    for ext in (".bil.hdr", ".bsq.hdr", ".bip.hdr", ".hdr"):
        if base.lower().endswith(ext):
            base = base[:-len(ext)]
            break
    # Try common ENVI data extensions
    for data_ext in (".bil", ".bsq", ".bip", ".dat", ".raw", ".img", ""):
        candidate = base + data_ext
        if os.path.isfile(candidate) and candidate.lower() != hdr_path.lower():
            return candidate
    return None


def _make_rgb_from_hsi(hsi_cube, wavelengths=None):
    """Build a false-color RGB from an HSI cube.  If wavelengths are known,
    picks bands near 640nm/550nm/460nm (true-color-ish).  Otherwise uses
    last/mid/first bands."""
    n_bands = hsi_cube.shape[-1]
    if wavelengths is not None and len(wavelengths) == n_bands:
        # Find closest bands to red/green/blue
        def closest_band(target_nm):
            return int(np.argmin(np.abs(wavelengths - target_nm)))
        idx_r = closest_band(640)
        idx_g = closest_band(550)
        idx_b = closest_band(460)
        print(f"RGB composite: bands {idx_r}({wavelengths[idx_r]:.0f}nm), "
              f"{idx_g}({wavelengths[idx_g]:.0f}nm), {idx_b}({wavelengths[idx_b]:.0f}nm)")
    else:
        idx_r, idx_g, idx_b = n_bands - 1, n_bands // 2, 0
        print(f"RGB composite: bands [{idx_r}, {idx_g}, {idx_b}] (last/mid/first)")
    composite = hsi_cube[:, :, [idx_r, idx_g, idx_b]].copy()
    # Per-channel percentile stretch for better contrast
    for c in range(3):
        ch = composite[:, :, c]
        p2, p98 = np.percentile(ch, [2, 98])
        ch = np.clip((ch - p2) / (p98 - p2 + 1e-8), 0, 1)
        composite[:, :, c] = ch
    return composite


def load_envi_scene(hsi_path, labels_path=None):
    """Load HSI from ENVI BIL/BSQ/BIP format using rasterio.
    Supports: .bil, .bsq, .bip data files, or .hdr header files.
    Also supports GeoTIFF (.tif/.tiff) with multiple bands."""
    try:
        import rasterio
    except ImportError:
        raise ImportError(
            "rasterio is required to load ENVI/GeoTIFF files. "
            "Install with: pip install rasterio"
        )

    # If user pointed at .hdr, find the actual data file
    actual_path = hsi_path
    if hsi_path.lower().endswith(".hdr"):
        data_file = _find_envi_data_file(hsi_path)
        if data_file:
            actual_path = data_file
            print(f"ENVI header: {hsi_path}")
            print(f"ENVI data:   {data_file}")
        else:
            raise FileNotFoundError(
                f"Could not find ENVI data file for header: {hsi_path}\n"
                f"Expected a .bil/.bsq/.bip file next to the .hdr file.\n"
                f"Please download the data file (usually the largest file, "
                f"hundreds of MB) and place it next to the .hdr file."
            )

    print(f"Loading HSI via rasterio: {actual_path}")
    with rasterio.open(actual_path) as ds:
        # rasterio reads as (bands, height, width) → we need (height, width, bands)
        data = ds.read()  # (B, H, W)
        hsi_cube = np.moveaxis(data, 0, -1).astype(np.float32)  # (H, W, B)
        print(f"HSI cube shape: {hsi_cube.shape} ({hsi_cube.shape[-1]} bands), "
              f"dtype: {data.dtype}")

    # Parse wavelengths from .hdr if available
    wavelengths = None
    hdr_candidates = [
        hsi_path if hsi_path.lower().endswith(".hdr") else None,
        actual_path + ".hdr",
        os.path.splitext(actual_path)[0] + ".hdr",
        os.path.splitext(actual_path)[0] + ".bil.hdr",
    ]
    for hdr in hdr_candidates:
        if hdr and os.path.isfile(hdr):
            wavelengths = _parse_envi_wavelengths(hdr)
            if wavelengths is not None:
                print(f"Wavelengths: {wavelengths[0]:.1f} - {wavelengths[-1]:.1f} nm "
                      f"({len(wavelengths)} bands)")
                break

    # Build RGB composite
    rgb_image = _make_rgb_from_hsi(hsi_cube, wavelengths)

    # Check for a co-located preview image (_image.png)
    base_dir = os.path.dirname(hsi_path)
    for f in os.listdir(base_dir):
        if f.lower().endswith(("_image.png", "_image.jpg", "_rgb.png")):
            try:
                from PIL import Image
                preview = np.array(Image.open(os.path.join(base_dir, f)).convert("RGB"))
                # Only use if it's a reasonable size
                if preview.shape[0] > 50 and preview.shape[1] > 50:
                    print(f"Found preview image: {f} (shape {preview.shape})")
                    # Normalize to [0,1]
                    rgb_image = preview.astype(np.float32) / 255.0
                    break
            except Exception:
                pass

    labels = None
    if labels_path:
        lp = labels_path.lower()
        if lp.endswith(".npy"):
            labels = np.load(labels_path).astype(np.int64)
        elif lp.endswith(".mat"):
            raw = loadmat(labels_path, simplify_cells=True)
            for key in ["labels", "sceneLabels", "gt", "groundtruth", "data", "Data"]:
                if key in raw:
                    labels = np.array(raw[key], dtype=np.int64)
                    break
            if labels is None:
                for key, value in raw.items():
                    if not key.startswith("__"):
                        labels = np.array(value, dtype=np.int64)
                        break
        if labels is not None:
            min_h = min(hsi_cube.shape[0], labels.shape[0])
            min_w = min(hsi_cube.shape[1], labels.shape[1])
            hsi_cube = hsi_cube[:min_h, :min_w, :]
            labels = labels[:min_h, :min_w]

    return hsi_cube, labels, rgb_image


def load_hsi_scene(hsi_path, labels_path=None):
    """Load HSI cube from .mat, .npy, ENVI BIL (.bil/.hdr), or GeoTIFF.
    No LiDAR needed."""

    lp = hsi_path.lower()
    # Route ENVI and GeoTIFF formats to the rasterio-based loader
    envi_exts = (".bil", ".bsq", ".bip", ".hdr", ".dat", ".img")
    tiff_exts = (".tif", ".tiff")
    if any(lp.endswith(ext) for ext in envi_exts + tiff_exts):
        return load_envi_scene(hsi_path, labels_path)

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
            for key, value in raw.items():
                if not key.startswith("__"):
                    arr = np.array(value)
                    print(f"Fallback to first key '{key}': shape {arr.shape}")
                    return arr
            raise ValueError(f"No usable array in {path}")
        else:
            raise ValueError(f"Unsupported file type for {path}")

    print(f"Loading HSI file: {hsi_path}")
    hsi_cube = load_array(hsi_path, common_keys=["Data", "hsi", "cube", "data"]).astype(np.float32)

    # Try to extract RGB for visualization
    rgb_image = None
    if hsi_path.lower().endswith(".mat"):
        raw = loadmat(hsi_path, simplify_cells=True)
        hsi_data = raw.get("hsi", raw)
        if isinstance(hsi_data, dict):
            for rgb_key in ("Rgb", "RGB", "rgb"):
                if rgb_key in hsi_data:
                    rgb_image = np.array(hsi_data[rgb_key])
                    print(f"Found RGB image under key '{rgb_key}'")
                    break

    labels = None
    if labels_path:
        labels = load_array(labels_path, common_keys=[
            "labels", "sceneLabels", "data", "Data", "gt", "groundtruth"
        ]).astype(np.int64)
        # Handle nested MUUFL label format
        if labels_path.lower().endswith(".mat"):
            try:
                raw = loadmat(labels_path, simplify_cells=True)
                if "hsi" in raw and isinstance(raw["hsi"], dict):
                    if "sceneLabels" in raw["hsi"]:
                        sl = raw["hsi"]["sceneLabels"]
                        if isinstance(sl, dict) and "labels" in sl:
                            labels = np.array(sl["labels"], dtype=np.int64)
                            print(f"Using nested MUUFL label format: shape {labels.shape}")
            except Exception:
                pass

        # Crop to match
        min_h = min(hsi_cube.shape[0], labels.shape[0])
        min_w = min(hsi_cube.shape[1], labels.shape[1])
        hsi_cube = hsi_cube[:min_h, :min_w, :]
        labels = labels[:min_h, :min_w]

    return hsi_cube, labels, rgb_image


def load_bundled_muufl(data_path, labels_path=None):
    """Load from a bundled MUUFL .mat file (HSI only, ignore LiDAR)."""
    print(f"Loading scene data from: {data_path}")
    data_raw = loadmat(data_path, simplify_cells=True)
    hsi_data = data_raw["hsi"]
    hsi_cube = np.array(hsi_data["Data"], dtype=np.float32)

    rgb_image = None
    for rgb_key in ("Rgb", "RGB", "rgb"):
        if rgb_key in hsi_data:
            rgb_image = np.array(hsi_data[rgb_key])
            break

    labels = None
    if labels_path:
        labels_raw = loadmat(labels_path, simplify_cells=True)
        labels = np.array(labels_raw["hsi"]["sceneLabels"]["labels"], dtype=np.int64)
        label_h, label_w = labels.shape
        hsi_h, hsi_w, _ = hsi_cube.shape
        if label_w < hsi_w:
            hsi_cube = hsi_cube[:label_h, :label_w, :]
        if label_h < hsi_h:
            hsi_cube = hsi_cube[:label_h, :, :]

    return hsi_cube, labels, rgb_image


# ---------------------------------------------------------------------------
# HSI-only model (full classifier, NOT the feature-extractor variant)
# ---------------------------------------------------------------------------
def build_hsi_classifier(hsimae_repo, checkpoint_path, num_bands=32,
                         embed_dim=256, depth=12, s_depth=9, device="cuda"):
    """Build HSIViT with its classification head intact, then load the
    fine-tuned checkpoint.  This is the standalone classifier, NOT the
    headless feature extractor used inside FusionModel."""
    sys.path.insert(0, hsimae_repo)
    from Models import HSIViT

    model = HSIViT(
        img_size=9, patch_size=3, in_chans=1, bands=num_bands, b_patch_size=8,
        num_class=len(CLASS_NAMES), embed_dim=embed_dim, depth=depth,
        num_heads=embed_dim // 16, s_depth=s_depth, trunc_init=True, drop_path=0.2,
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded HSI-only classifier from {checkpoint_path}")
    return model


# ---------------------------------------------------------------------------
# Patch extraction & batched inference
# ---------------------------------------------------------------------------
def extract_hsi_patches(hsi_cube):
    h, w, _ = hsi_cube.shape
    half = HSI_PATCH_SIZE // 2
    padded = np.pad(hsi_cube, ((half, half), (half, half), (0, 0)), mode="reflect")
    rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    return padded, rows.flatten(), cols.flatten(), half


def build_hsi_batch(hsi_padded, rows, cols, half):
    """Build a batch of HSI patches in the (B, 1, Bands, H, W) layout."""
    tensors = []
    for row, col in zip(rows, cols):
        r, c = row + half, col + half
        patch = hsi_padded[r - half:r + half + 1, c - half:c + half + 1, :]
        t = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)
        tensors.append(t)
    return torch.stack(tensors, dim=0)  # (B, 1, 32, 9, 9)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="HSI-ONLY Inference Pipeline")
    parser.add_argument(
        "--data", default=os.path.join(
            PROJECT_ROOT,
            r"MUUFLGulfport-master\MUUFLGulfportDataCollection\muufl_gulfport_campus_w_lidar_1.mat"
        ),
        help="Bundled MUUFL-style .mat (HSI will be extracted, LiDAR ignored)"
    )
    parser.add_argument("--hsi_file", default=None,
                        help="Separate HSI file (.mat, .npy, .bil, .hdr, .tif)")
    parser.add_argument(
        "--labels", default=os.path.join(
            PROJECT_ROOT,
            r"MUUFLGulfport-master\MUUFLGulfportSceneLabels\muufl_gulfport_campus_1_hsi_220_label.mat"
        ),
        help="Optional ground-truth labels for accuracy check"
    )
    parser.add_argument("--hsimae_repo", default=os.path.join(PROJECT_ROOT, "pretrained_models", "HSIMAE"))
    parser.add_argument(
        "--hsi_checkpoint", default=os.path.join(PROJECT_ROOT, "runs", "hsi_only_v2", "best_model.pt"),
        help="Fine-tuned HSI-only best_model.pt (from finetune_hsi2.py)"
    )
    parser.add_argument(
        "--out", default=os.path.join(SCRIPT_DIR, "OUTPUT", "HSI"),
        help="Base output folder — each run creates a timestamped subfolder"
    )
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("=" * 70)
    print("  HSI-ONLY INFERENCE PIPELINE")
    print("  (No LiDAR data used — pure hyperspectral detection)")
    print("=" * 70)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_out = os.path.join(args.out, f"run_{timestamp}")
    os.makedirs(run_out, exist_ok=True)
    print(f"Output folder: {run_out}")

    # --- Load scene (HSI only) ---
    if args.hsi_file:
        hsi_cube, labels, rgb_image = load_hsi_scene(args.hsi_file, args.labels)
    elif args.data:
        hsi_cube, labels, rgb_image = load_bundled_muufl(args.data, args.labels)
    else:
        raise ValueError("Provide either --data (bundled .mat) or --hsi_file.")
    print(f"Scene shape: HSI {hsi_cube.shape}")

    if rgb_image is None:
        n_bands = hsi_cube.shape[-1]
        idx_r, idx_g, idx_b = n_bands - 1, n_bands // 2, 0
        composite = hsi_cube[:, :, [idx_r, idx_g, idx_b]]
        composite = (composite - composite.min()) / (composite.max() - composite.min() + 1e-8)
        rgb_image = composite

    # --- Preprocess ---
    print("Applying GWPCA (spectral dimensionality reduction)...")
    hsi_cube = apply_gwpca(hsi_cube, nc=GWPCA_BANDS, num_groups=GWPCA_GROUPS)
    hsi_cube = normalize_hsi(hsi_cube)

    h, w, _ = hsi_cube.shape
    hsi_padded, rows, cols, half = extract_hsi_patches(hsi_cube)
    n_pixels = len(rows)
    print(f"Running HSI-only inference on all {n_pixels:,d} pixels...")

    # --- Model ---
    model = build_hsi_classifier(args.hsimae_repo, args.hsi_checkpoint, device=device)

    # --- Inference ---
    all_preds = np.zeros(n_pixels, dtype=np.int64)
    all_target_probs = np.zeros(n_pixels, dtype=np.float32)

    inference_start = time.time()
    with torch.no_grad():
        for start in tqdm(range(0, n_pixels, args.batch_size), desc="HSI-only inference"):
            end = min(start + args.batch_size, n_pixels)
            batch = build_hsi_batch(hsi_padded, rows[start:end], cols[start:end], half)
            batch = batch.to(device)

            logits = model(batch)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1).cpu().numpy()
            target_probs = probs[:, TARGET_CLASS_IDX].cpu().numpy()

            all_preds[start:end] = preds
            all_target_probs[start:end] = target_probs

    inference_time = time.time() - inference_start
    print(f"HSI-only inference completed in {inference_time:.2f} seconds.")

    class_map = all_preds.reshape(h, w) + 1  # back to 1-indexed
    confidence_map = all_target_probs.reshape(h, w)

    np.save(os.path.join(run_out, "class_map.npy"), class_map)
    np.save(os.path.join(run_out, "target_confidence_map.npy"), confidence_map)
    print(f"Saved class_map.npy and target_confidence_map.npy to {run_out}/")

    # --- Visualization ---
    try:
        import matplotlib
        matplotlib.use("Agg")
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

        class_rgb = [_hex_to_rgb(h_) + (1.0,) for h_ in CLASS_COLORS_HEX]
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
        stats_lines = ["\nPredicted class distribution (HSI-ONLY):"]
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
        ax_pred.set_title("Predicted Class Map (HSI-ONLY)", fontsize=12, fontweight="bold")
        ax_pred.axis("off")
        ax_pred.legend(
            handles=legend_patches, loc="upper left", bbox_to_anchor=(1.02, 1.0),
            fontsize=8, title="Class Legend", title_fontsize=9,
            frameon=True, fancybox=True, shadow=True,
        )

        ax_heat = fig.add_subplot(2, n_cols, n_cols + 1)
        im = ax_heat.imshow(confidence_map, cmap="hot", vmin=0, vmax=1)
        ax_heat.set_title("Target Confidence (HSI-ONLY)", fontsize=12, fontweight="bold")
        ax_heat.axis("off")
        fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)

        ax_overlay = fig.add_subplot(2, n_cols, n_cols + 2)
        ax_overlay.imshow(rgb_image)
        hot_cmap = cm.colormaps.get("hot", cm.get_cmap("hot")) if hasattr(cm, "colormaps") else cm.get_cmap("hot")
        overlay = hot_cmap(confidence_map)
        overlay[..., 3] = np.where(confidence_map > 0.3, np.clip((confidence_map - 0.3) / 0.7, 0, 1), 0)
        ax_overlay.imshow(overlay)
        ax_overlay.set_title("Target Confidence Overlay (HSI-ONLY)", fontsize=12, fontweight="bold")
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
        ax_bar.set_title("Class Distribution (HSI-ONLY)", fontsize=12, fontweight="bold")
        for i, (cnt, name) in enumerate(zip(bar_counts, bar_names)):
            pct = 100.0 * cnt / total_pixels
            ax_bar.text(cnt + total_pixels * 0.005, i, f"{pct:.1f}%", va="center", fontsize=7, fontweight="bold")

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.35, hspace=0.15)
        out_png = os.path.join(run_out, "detection_visualization.png")
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved visualization to {out_png}")
    except ImportError:
        print("matplotlib not installed -- skipping visualization.")

    # --- Accuracy check on labeled pixels ---
    if labels is not None:
        labeled_mask = labels != -1
        gt = labels[labeled_mask]
        pred = class_map[labeled_mask]
        acc = (gt == pred).mean()
        target_mask = gt == (TARGET_CLASS_IDX + 1)
        target_recall = (pred[target_mask] == (TARGET_CLASS_IDX + 1)).mean() if target_mask.sum() > 0 else 0.0
        print(f"\nSanity check (HSI-ONLY): accuracy = {acc:.3f}, "
              f"target recall = {target_recall:.3f}")

        try:
            from sklearn.metrics import classification_report, confusion_matrix
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            present_classes = sorted(set(gt.tolist()) | set(pred.tolist()))
            present_names = [
                CLASS_NAMES[c - 1] if 1 <= c <= len(CLASS_NAMES) else f"Class {c}"
                for c in present_classes
            ]

            report = classification_report(
                gt, pred, labels=present_classes, target_names=present_names,
                zero_division=0, digits=3,
            )
            print("\n" + "=" * 70)
            print(" HSI-ONLY PER-CLASS CLASSIFICATION REPORT")
            print("=" * 70)
            print(report)

            report_path = os.path.join(run_out, "classification_report.txt")
            with open(report_path, "w") as f:
                f.write("HSI-ONLY PER-CLASS CLASSIFICATION REPORT\n")
                f.write("=" * 70 + "\n")
                f.write(report)
                f.write(f"\nOverall accuracy: {acc:.4f}\n")
                f.write(f"Target recall: {target_recall:.4f}\n")
            print(f"Report saved to {report_path}")

            cm_data = confusion_matrix(gt, pred, labels=present_classes)
            fig_cm, ax_cm = plt.subplots(
                figsize=(max(8, len(present_classes) * 0.9), max(7, len(present_classes) * 0.8))
            )
            cm_norm = cm_data.astype(np.float64)
            row_sums = cm_norm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            cm_pct = cm_norm / row_sums * 100

            im = ax_cm.imshow(cm_pct, interpolation="nearest", cmap="YlOrRd")
            fig_cm.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04, label="Row-normalized %")

            for i in range(len(present_classes)):
                for j in range(len(present_classes)):
                    count = cm_data[i, j]
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
            ax_cm.set_title(f"HSI-ONLY Confusion Matrix (Accuracy={acc:.3f})",
                            fontsize=13, fontweight="bold")
            plt.tight_layout()
            cm_path = os.path.join(run_out, "confusion_matrix.png")
            fig_cm.savefig(cm_path, dpi=150, bbox_inches="tight")
            plt.close(fig_cm)
            print(f"Confusion matrix saved to {cm_path}")

        except Exception as e:
            print(f"Report/confusion-matrix error: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nAll HSI-ONLY outputs saved to: {run_out}")


if __name__ == "__main__":
    main()
