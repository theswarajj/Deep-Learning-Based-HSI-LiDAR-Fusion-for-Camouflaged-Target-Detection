"""
FULL PIPELINE  (inference_pipeline + identify_targets + ACE + SAM, in one script)

Stages, all in one run, all saved into a fresh timestamped folder:

  1. Load the scene (bundled MUUFL .mat, or separate --hsi_file/--lidar_file)
  2. Fusion inference (GWPCA -> HSIMAE + PTv3 -> cross-modal attention) for
     every pixel  ->  class_map.npy + target_confidence_map.npy
     - Optional: Test-Time Augmentation (TTA) for more robust predictions
     - Optional: Mixed-precision (fp16) inference for ~2x speedup on GPU
  3. Spatial confidence refinement (Gaussian smoothing to suppress noise)
  4. Classic ACE detector (Adaptive Coherence/Cosine Estimator) on the RAW
     hyperspectral cube -- no training, HSI only -- used as a baseline and as
     an independent cross-check of the fusion model
     - Optional: Multi-scale ACE (ensemble of regularization levels)
  5. SAM detector (Spectral Angle Mapper) -- another HSI-only, training-free
     baseline that measures the angular distance between each pixel spectrum
     and the target signature
  6. Ensemble score: weighted combination of fusion + ACE + SAM
  7. Target identification: connected components of the fusion confidence map,
     each described by colour, surrounding land cover, size, quality score,
     spectral contrast ratio, and cross-check by ACE/SAM/ensemble
  8. Proximity-based target merging to avoid over-counting fragments
  9. Reports, figures, ROC/PR comparison, calibration plot, and a standalone
     HTML summary report with embedded images

ACE:  ACE(x) = (s'^T S^-1 x')^2 / ( (s'^T S^-1 s') (x'^T S^-1 x') )
      x' = x - mu,  s' = target_spectrum - mu,  mu / S = scene mean / covariance

SAM:  SAM(x) = arccos( x·s / (||x|| ||s||) )   -> similarity = 1 - SAM/pi

Fair comparison on MUUFL: the script reproduces your exact preprocess.py
train/val/test split (seed 42). The ACE/SAM target signatures and thresholds
are chosen from train+val pixels ONLY, and all detectors are scored on the
held-out test pixels. Check that the printed test set is 10,738 pixels with
54 targets.

Usage (MUUFL, defaults baked in -- just run it):
    python full_pipeline.py

Skip the ~10 minute model pass by reusing an earlier run's maps:
    python full_pipeline.py --reuse_run "C:\\...\\OUTPUT\\run_20260912_183528"

Another scene (kind is auto-set to 'other'; no MUUFL label evaluation):
    python full_pipeline.py --hsi_file X_hsi.mat --lidar_file X_lidar.mat --labels X_gt.mat

Requires: numpy, scipy, scikit-learn, matplotlib (+ torch and your finetune_fusion.py
for the model pass).  Optional: opencv-python + ultralytics for --yolo.
"""

import argparse
import base64
import csv
import datetime
import json
import os
import sys
import time
import warnings
from collections import Counter
from io import BytesIO

import numpy as np
from scipy import ndimage
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Constants (mirror finetune_fusion.py / preprocess.py)
# ---------------------------------------------------------------------------
HSI_PATCH_SIZE = 9
LIDAR_PATCH_SIZE = 9
GWPCA_BANDS = 32
GWPCA_GROUPS = 4
GRID_SIZE = 0.05
IGNORE_LABEL = -1
MUUFL_TARGET_LABEL = 11          # "Cloth Panels" in the 1-indexed MUUFL label map
EXPECTED_TEST_SIZE = 10738       # your reported MUUFL test split
EXPECTED_TEST_TARGETS = 54

MUUFL_CLASS_NAMES = [
    "Trees", "Mostly Grass", "Mixed Ground Surface", "Dirt and Sand",
    "Road", "Water", "Shadow of Buildings", "Buildings", "Sidewalk",
    "Yellow Curb", "Cloth Panels (TARGET)",
]
MUUFL_CLASS_COLORS_HEX = [
    "#000000", "#228B22", "#90EE90", "#C2B280", "#DEB887", "#505050",
    "#4169E1", "#483D8B", "#A9A9A9", "#D3D3D3", "#FFD700", "#FF0000",
]

# Trento dataset (Italy) constants - 6 ground truth classes
TRENTO_CLASS_NAMES = [
    "Apple trees", "Buildings (TARGET)", "Ground", "Wood", "Vineyard", "Roads",
]
TRENTO_TARGET_LABEL = 2          # "Buildings" in the 1-indexed Trento label map
TRENTO_CLASS_COLORS_HEX = [
    "#000000", "#22c55e", "#ef4444", "#d97706", "#854d0e", "#a855f7", "#64748b",
]

# Default aliases for backwards compatibility
CLASS_NAMES = MUUFL_CLASS_NAMES
TARGET_CLASS_IDX = 10
CLASS_COLORS_HEX = MUUFL_CLASS_COLORS_HEX

# Multi-scale ACE regularization levels
ACE_MULTISCALE_REGS = [1e-4, 1e-3, 1e-2]

# TTA augmentation types
TTA_AUGMENTATIONS = ["original", "hflip", "vflip", "rot180"]


def get_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip("#")
    return tuple(int(hex_str[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


# ===========================================================================
# 1. LOADING
# ===========================================================================
def _find_hsi_cube_array(hsi_data):
    """MUUFL uses 'Data', Trento's raw scene uses 'data', etc."""
    for key in ("Data", "data", "hsi", "HSI", "cube"):
        if key in hsi_data:
            return np.array(hsi_data[key], dtype=np.float32)
    for key, value in hsi_data.items():
        if not str(key).startswith("__"):
            return np.array(value, dtype=np.float32)
    raise ValueError(f"Could not find an HSI array. Keys were: {list(hsi_data.keys())}")


def _find_wavelengths(hsi_data, n_bands):
    for key in ("wavelength", "Wavelength", "wl", "lambda"):
        if key in hsi_data:
            wl = np.array(hsi_data[key], dtype=np.float32).flatten()
            if len(wl) == n_bands:
                return wl
    return None


def load_raw_scene(data_path, labels_path=None):
    """Bundled MUUFL-style .mat (HSI + LiDAR together)."""
    print(f"Loading scene data from: {data_path}")
    data_raw = loadmat(data_path, simplify_cells=True)
    hsi_data = data_raw["hsi"]
    print(f"Available fields in this .mat file: {list(hsi_data.keys())}")
    hsi_cube = _find_hsi_cube_array(hsi_data)
    wavelengths = _find_wavelengths(hsi_data, hsi_cube.shape[-1])

    rgb_image = None
    for rgb_key in ("Rgb", "RGB", "rgb"):
        if rgb_key in hsi_data:
            rgb_image = np.array(hsi_data[rgb_key])
            print(f"Found RGB image under key '{rgb_key}', shape {rgb_image.shape}")
            break
    if rgb_image is None:
        print("No RGB field found -- a false-color composite will be built from the HSI bands.")

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

    return hsi_cube, lidar, labels, rgb_image, wavelengths


def load_custom_scene(hsi_path, lidar_path, labels_path=None):
    """Separate HSI / LiDAR (/ labels) files, .mat or .npy."""

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
                    print(f"None of {common_keys} matched -- falling back to "
                          f"first available key '{key}': shape {arr.shape}")
                    return arr
            raise ValueError(f"No usable array found in {path}. Keys were: {list(raw.keys())}")
        raise ValueError(f"Unsupported file type for {path}. Use .mat or .npy.")

    print(f"Loading custom HSI file: {hsi_path}")
    hsi_cube = load_array(hsi_path, ["Data", "hsi", "cube", "data"]).astype(np.float32)

    print(f"Loading custom LiDAR file: {lidar_path}")
    lidar_raw = load_array(lidar_path, ["Lidar", "lidar", "elevation", "z", "data", "Data"])

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
        lidar = lidar_raw.astype(np.float32)

    target_channels = 4
    if lidar.shape[-1] < target_channels:
        pad_width = target_channels - lidar.shape[-1]
        print(f"WARNING: LiDAR has {lidar.shape[-1]} channel(s), model expects {target_channels}. "
              f"Padding with {pad_width} zero channel(s). Treat results as exploratory.")
        lidar = np.concatenate(
            [lidar, np.zeros((*lidar.shape[:2], pad_width), dtype=np.float32)], axis=-1)
    elif lidar.shape[-1] > target_channels:
        print(f"WARNING: LiDAR has {lidar.shape[-1]} channels, model expects {target_channels}. "
              f"Truncating to the first {target_channels}.")
        lidar = lidar[:, :, :target_channels]

    labels = None
    if labels_path:
        labels = load_array(
            labels_path, ["labels", "sceneLabels", "data", "Data", "gt", "groundtruth", "mask_test", "mask_train", "allgrd"]
        ).astype(np.int64)

    shapes = [hsi_cube.shape[:2], lidar.shape[:2]] + ([labels.shape[:2]] if labels is not None else [])
    min_h = min(s[0] for s in shapes)
    min_w = min(s[1] for s in shapes)
    if any(s != (min_h, min_w) for s in shapes):
        print(f"Shapes don't match exactly ({shapes}); cropping all to the shared top-left "
              f"region ({min_h}, {min_w}). This does NOT perform spatial alignment.")
        hsi_cube = hsi_cube[:min_h, :min_w, :]
        lidar = lidar[:min_h, :min_w, :]
        if labels is not None:
            labels = labels[:min_h, :min_w]

    return hsi_cube, lidar, labels, None, None


def load_trento_scene(hsi_path=None, lidar_path=None, labels_path=None):
    """Load Trento hyperspectral and LiDAR scene at FULL SIZE (166x600) with NO CROPPING.

    Returns:
        hsi_cube: (166, 600, 63) float32
        lidar: (166, 600, 4) float32 (2 original channels: DSM + Intensity, padded with 2 zeros)
        labels: (166, 600) int64 ground truth (1..6 classes, 0 unlabeled)
        rgb_image: None (display image will be computed as false-color composite)
        wavelengths: 63 floats (AISA Eagle wavelengths: 402.89 nm to 989.09 nm)
    """
    if hsi_path is None:
        hsi_path = r"Trento-main\Italy_hsi.mat"
    if lidar_path is None:
        lidar_path = r"Trento-main\Italy_lidar.mat"
    if labels_path is None:
        labels_path = r"Trento-main\allgrd.mat"

    print("\n" + "=" * 78)
    print(" LOADING TRENTO DATASET (FULL SIZE 166x600 - NO CROPPING)")
    print("=" * 78)
    print(f"  HSI File   : {hsi_path}")
    print(f"  LiDAR File : {lidar_path}")
    print(f"  Labels File: {labels_path}")

    # 1. HSI
    if not os.path.exists(hsi_path):
        raise FileNotFoundError(f"Trento HSI file not found: {hsi_path}")
    raw_hsi = loadmat(hsi_path, simplify_cells=True)
    hsi_key = "data" if "data" in raw_hsi else [k for k in raw_hsi.keys() if not k.startswith("__")][0]
    hsi_cube = np.array(raw_hsi[hsi_key], dtype=np.float32)

    # 2. LiDAR
    if not os.path.exists(lidar_path):
        raise FileNotFoundError(f"Trento LiDAR file not found: {lidar_path}")
    raw_lidar = loadmat(lidar_path, simplify_cells=True)
    lidar_key = "data" if "data" in raw_lidar else [k for k in raw_lidar.keys() if not k.startswith("__")][0]
    lidar_2ch = np.array(raw_lidar[lidar_key], dtype=np.float32)
    # Pad to 4 channels for model compatibility (ch0: DSM elevation, ch1: Intensity, ch2: 0, ch3: 0)
    h, w = lidar_2ch.shape[:2]
    pad_zeros = np.zeros((h, w, 2), dtype=np.float32)
    lidar = np.concatenate([lidar_2ch, pad_zeros], axis=-1)

    # 3. Ground Truth Labels
    labels = None
    if labels_path and os.path.exists(labels_path):
        raw_labels = loadmat(labels_path, simplify_cells=True)
        if "mask_test" in raw_labels:
            labels = np.array(raw_labels["mask_test"], dtype=np.int64)
        else:
            for k, v in raw_labels.items():
                if not k.startswith("__"):
                    labels = np.array(v, dtype=np.int64)
                    break

    # 4. Wavelengths (AISA Eagle: 63 bands from 402.89 nm to 989.09 nm)
    wavelengths = np.linspace(402.89, 989.09, hsi_cube.shape[-1], dtype=np.float32)

    # Verify no cropping and exact dimensions
    assert hsi_cube.shape[:2] == lidar.shape[:2] == (166, 600), (
        f"Unexpected Trento dimensions: HSI {hsi_cube.shape}, LiDAR {lidar.shape}"
    )
    if labels is not None:
        assert labels.shape == (166, 600), f"Unexpected Trento labels dimension: {labels.shape}"

    print(f"  [Trento] Scene loaded at FULL SIZE without any cutting or cropping:")
    print(f"    * HSI cube   : {hsi_cube.shape} (63 bands, 402.9 nm - 989.1 nm)")
    print(f"    * LiDAR cube : {lidar.shape} (DSM height range: {lidar[:, :, 0].min():.2f}m - {lidar[:, :, 0].max():.2f}m)")
    if labels is not None:
        labeled_cnt = int((labels > 0).sum())
        print(f"    * GroundTruth: {labels.shape} (Labeled: {labeled_cnt:,d} px across 6 land-cover classes)")
    print("=" * 78 + "\n")

    return hsi_cube, lidar, labels, None, wavelengths


def prompt_dataset_selection(default_choice="muufl"):
    """Interactive terminal menu allowing user to select between MUUFL and Trento datasets."""
    if not sys.stdin.isatty():
        print(f"[Selection Menu] Non-interactive terminal detected: auto-selecting [{default_choice.upper()}].")
        return default_choice

    menu = (
        "\n" + "=" * 78 + "\n"
        "                 HYPERSPECTRAL & LIDAR DATASET SELECTION\n"
        "=" * 78 + "\n"
        "  [1] MUUFL Gulfport Campus (Default)\n"
        "      * Spatial Resolution : 325 x 220 pixels\n"
        "      * Hyperspectral Cube : 64 bands (375 nm - 1055 nm)\n"
        "      * LiDAR Elevation    : 2 returns (first & last surface height)\n"
        "      * Target of Interest : Cloth Panels (Class 11)\n"
        "\n"
        "  [2] Trento Dataset, Italy (FULL SIZE 166x600 - NO CROPPING)\n"
        "      * Spatial Resolution : 166 x 600 pixels (Full uncropped scene)\n"
        "      * Hyperspectral Cube : 63 bands (AISA Eagle, 402.9 nm - 989.1 nm)\n"
        "      * LiDAR Data         : 2 channels (DSM height up to 20.2m + Intensity)\n"
        "      * Target of Interest : Buildings (Class 2) / Man-made structures\n"
        "=" * 78
    )
    print(menu)
    while True:
        try:
            choice = input("Select dataset [1 or 2] (Press Enter for Default [1]): ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\nSelection interrupted: defaulting to [{default_choice.upper()}].")
            return default_choice

        if choice in ("", "1", "muufl", "m", "gulfport"):
            print("[Selection Menu] Selected: MUUFL Gulfport Campus")
            return "muufl"
        elif choice in ("2", "trento", "t", "italy"):
            print("[Selection Menu] Selected: Trento Dataset (FULL SIZE 166x600 - NO CROPPING)")
            return "trento"
        else:
            print("Invalid choice. Please enter '1' for MUUFL Gulfport or '2' for Trento.")


def prepare_display_image(rgb_image, hsi_raw):
    """RGB image cropped to the scene, or a false-colour composite."""
    h, w = hsi_raw.shape[:2]
    if rgb_image is not None:
        rgb = np.asarray(rgb_image)
        if rgb.ndim == 3 and rgb.shape[-1] >= 3:
            rgb = rgb[:h, :w, :3]
            if rgb.shape[:2] == (h, w):
                return rgb
        print("RGB image doesn't match the scene -- building a false-colour composite instead.")
    n = hsi_raw.shape[-1]
    comp = np.nan_to_num(hsi_raw[:, :, [n - 1, n // 2, 0]].astype(np.float32))
    lo, hi = np.percentile(comp, (2, 98))
    return np.clip((comp - lo) / (hi - lo + 1e-8), 0, 1)


# ===========================================================================
# 2. FUSION INFERENCE (with TTA + mixed-precision support)
# ===========================================================================
def _split_spectral_groups(data_2d, num_groups=4):
    groups = [data_2d]
    for _ in range(int(np.log2(num_groups))):
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
    per_group = nc // num_groups
    outputs = []
    for g in groups:
        n_comp = min(per_group, g.shape[1])
        outputs.append(PCA(n_components=n_comp, whiten=whiten, random_state=42).fit_transform(g))
    return np.concatenate(outputs, axis=-1).reshape(h, w, -1).astype(np.float32)


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
        ch = result[:, :, c]
        result[:, :, c] = (ch - ch.mean()) / (ch.std() + 1e-8)
    return result


def extract_all_patches(hsi_cube, lidar):
    h, w, _ = hsi_cube.shape
    hh, lh = HSI_PATCH_SIZE // 2, LIDAR_PATCH_SIZE // 2
    hsi_padded = np.pad(hsi_cube, ((hh, hh), (hh, hh), (0, 0)), mode="reflect")
    lidar_padded = np.pad(lidar, ((lh, lh), (lh, lh), (0, 0)), mode="reflect")
    rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    return hsi_padded, lidar_padded, rows.flatten(), cols.flatten(), hh, lh


def build_batch(hsi_padded, lidar_padded, rows, cols, hsi_half, lidar_half, use_normal):
    import torch
    hsi_tensors, coords, feats, counts = [], [], [], []
    win = LIDAR_PATCH_SIZE
    ys, xs = np.meshgrid(np.arange(win) - win // 2, np.arange(win) - win // 2, indexing="ij")
    rel_x = xs.flatten().astype(np.float32)
    rel_y = ys.flatten().astype(np.float32)

    for row, col in zip(rows, cols):
        r, c = row + hsi_half, col + hsi_half
        hsi_patch = hsi_padded[r - hsi_half:r + hsi_half + 1, c - hsi_half:c + hsi_half + 1, :]
        hsi_tensors.append(torch.tensor(hsi_patch, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2))

        r, c = row + lidar_half, col + lidar_half
        lp = lidar_padded[r - lidar_half:r + lidar_half + 1, c - lidar_half:c + lidar_half + 1, :]
        flat = lp.reshape(-1, 4)
        coord = np.stack([rel_x, rel_y, flat[:, 0]], axis=1).astype(np.float32)
        color_like = flat[:, 1:4].astype(np.float32)
        normal = np.zeros_like(coord, dtype=np.float32)

        coord_t = torch.from_numpy(coord)
        parts = [coord_t, torch.from_numpy(color_like)]
        if use_normal:
            parts.append(torch.from_numpy(normal))
        coords.append(coord_t)
        feats.append(torch.cat(parts, dim=1))
        counts.append(coord_t.shape[0])

    point_dict = {
        "coord": torch.cat(coords, dim=0).float(),
        "feat": torch.cat(feats, dim=0).float(),
        "offset": torch.cumsum(torch.tensor(counts, dtype=torch.long), dim=0),
        "grid_size": GRID_SIZE,
    }
    return torch.stack(hsi_tensors, dim=0), point_dict


def _apply_tta_transform(hsi_padded, lidar_padded, aug_type):
    """Apply a spatial augmentation to the padded arrays for TTA."""
    if aug_type == "original":
        return hsi_padded, lidar_padded
    elif aug_type == "hflip":
        return hsi_padded[:, ::-1, :].copy(), lidar_padded[:, ::-1, :].copy()
    elif aug_type == "vflip":
        return hsi_padded[::-1, :, :].copy(), lidar_padded[::-1, :, :].copy()
    elif aug_type == "rot180":
        return hsi_padded[::-1, ::-1, :].copy(), lidar_padded[::-1, ::-1, :].copy()
    return hsi_padded, lidar_padded


def _invert_tta_map(pred_map, conf_map, aug_type, h, w):
    """Undo the spatial augmentation on (H, W) maps."""
    if aug_type == "hflip":
        return pred_map[:, ::-1].copy(), conf_map[:, ::-1].copy()
    elif aug_type == "vflip":
        return pred_map[::-1, :].copy(), conf_map[::-1, :].copy()
    elif aug_type == "rot180":
        return pred_map[::-1, ::-1].copy(), conf_map[::-1, ::-1].copy()
    return pred_map, conf_map


def run_fusion_inference(args, hsi_raw, lidar_raw, target_class_idx=None):
    """Returns (class_map 1-indexed HxW, target_confidence_map HxW).

    Improvements over original:
    - Support configurable target class index (e.g. Buildings for Trento, Cloth Panels for MUUFL)
    - weights_only=True for secure model loading (suppresses FutureWarning)
    - Optional mixed-precision (fp16) inference via --mixed_precision
    - Optional Test-Time Augmentation (TTA) via --tta for more robust predictions
    """
    import torch
    from finetune_fusion import (
        build_hsi_extractor, build_lidar_extractor, FusionModel, move_point_dict,
        CLASS_NAMES as FUSION_CLASS_NAMES, TARGET_CLASS_IDX as FUSION_TARGET_IDX,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    use_amp = args.mixed_precision and device.type == "cuda"
    if use_amp:
        print("Mixed-precision (fp16) inference enabled.")

    print(f"Applying GWPCA ({hsi_raw.shape[-1]} -> {GWPCA_BANDS} bands)...")
    hsi = normalize_hsi(apply_gwpca(hsi_raw, nc=GWPCA_BANDS, num_groups=GWPCA_GROUPS))
    lidar = normalize_lidar(lidar_raw)

    h, w, _ = hsi.shape

    # Build the model once
    hsi_padded, lidar_padded, rows, cols, hh, lh = extract_all_patches(hsi, lidar)
    n_pixels = len(rows)
    print(f"Running inference on all {n_pixels:,d} pixels in the scene...")

    hsi_extractor, hsi_dim = build_hsi_extractor(args.hsimae_repo, args.hsi_checkpoint, device=device)
    lidar_extractor = build_lidar_extractor(
        args.utonia_repo, args.lidar_pretrained, args.lidar_checkpoint, args.use_normal, device)
    with torch.no_grad():
        _, sample_pd = build_batch(hsi_padded, lidar_padded, rows[:2], cols[:2], hh, lh, args.use_normal)
        lidar_dim = lidar_extractor(move_point_dict(sample_pd, device)).shape[-1]

    model = FusionModel(hsi_extractor, lidar_extractor, hsi_dim, lidar_dim,
                        num_classes=len(FUSION_CLASS_NAMES)).to(device)

    # Secure model loading -- suppresses FutureWarning
    try:
        state_dict = torch.load(args.fusion_checkpoint, map_location=device, weights_only=True)
    except TypeError:
        # Older PyTorch versions don't support weights_only
        state_dict = torch.load(args.fusion_checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded fusion model from {args.fusion_checkpoint}")

    # Determine TTA augmentations
    aug_types = TTA_AUGMENTATIONS if args.tta else ["original"]
    n_augs = len(aug_types)
    if n_augs > 1:
        print(f"Test-Time Augmentation enabled with {n_augs} views: {aug_types}")

    # Accumulate softmax probabilities across augmentations
    probs_accum = np.zeros((h, w, len(FUSION_CLASS_NAMES)), dtype=np.float64)

    t0 = time.time()
    for aug_idx, aug_type in enumerate(aug_types):
        hsi_aug, lidar_aug = _apply_tta_transform(hsi_padded, lidar_padded, aug_type)

        # Re-extract pixel coordinates for augmented arrays
        if aug_type != "original":
            hsi_aug_core = hsi_aug[hh:hh + h, hh:hh + w, :]
            lidar_aug_core = lidar_aug[lh:lh + h, lh:lh + w, :]
            hsi_aug_p, lidar_aug_p, rows_a, cols_a, _, _ = extract_all_patches(hsi_aug_core, lidar_aug_core)
        else:
            hsi_aug_p, lidar_aug_p, rows_a, cols_a = hsi_aug, lidar_aug, rows, cols

        preds_aug = np.zeros((n_pixels, len(FUSION_CLASS_NAMES)), dtype=np.float32)

        desc = f"Inference [{aug_type}]" if n_augs > 1 else "Inference batches"
        with torch.no_grad():
            for start in tqdm(range(0, n_pixels, args.batch_size), desc=desc):
                end = min(start + args.batch_size, n_pixels)
                hsi_batch, point_dict = build_batch(
                    hsi_aug_p, lidar_aug_p, rows_a[start:end], cols_a[start:end], hh, lh, args.use_normal)

                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits = model(hsi_batch.to(device), move_point_dict(point_dict, device))
                else:
                    logits = model(hsi_batch.to(device), move_point_dict(point_dict, device))

                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds_aug[start:end] = probs

        # Reshape to H x W x C and invert augmentation
        prob_map = preds_aug.reshape(h, w, -1)
        if aug_type != "original":
            # Invert spatial transform on the probability map
            if aug_type == "hflip":
                prob_map = prob_map[:, ::-1, :].copy()
            elif aug_type == "vflip":
                prob_map = prob_map[::-1, :, :].copy()
            elif aug_type == "rot180":
                prob_map = prob_map[::-1, ::-1, :].copy()

        probs_accum += prob_map

    # Average across augmentations
    probs_avg = (probs_accum / n_augs).astype(np.float32)
    class_map = probs_avg.argmax(axis=-1).reshape(h, w) + 1
    tgt_idx = target_class_idx if target_class_idx is not None else FUSION_TARGET_IDX
    confidence_map = probs_avg[:, :, tgt_idx].reshape(h, w)

    elapsed = time.time() - t0
    print(f"Inference completed in {elapsed:.2f} seconds"
          f"{f' ({n_augs} TTA views)' if n_augs > 1 else ''}.")

    return class_map, confidence_map


# ===========================================================================
# 2b. SPATIAL CONFIDENCE REFINEMENT
# ===========================================================================
def spatial_smooth_confidence(confidence_map, sigma=1.0):
    """Gaussian smoothing of the confidence map to suppress isolated noise pixels
    while reinforcing clustered detections. Returns the smoothed map."""
    smoothed = gaussian_filter(confidence_map.astype(np.float64), sigma=sigma)
    return smoothed.astype(np.float32)


def adaptive_threshold_otsu(confidence_map, n_bins=256):
    """Compute an Otsu-like optimal threshold on the confidence distribution.
    Returns (threshold, method_description)."""
    vals = confidence_map.flatten()
    vals = vals[vals > 0.01]  # ignore near-zero background
    if len(vals) < 100:
        return 0.5, "fallback (too few non-zero pixels for Otsu)"

    hist, bin_edges = np.histogram(vals, bins=n_bins, range=(0.01, 1.0))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    total = hist.sum()

    best_thresh, best_var = 0.5, -1.0
    w0, sum0, sum_total = 0.0, 0.0, float((hist * bin_centers).sum())

    for i in range(len(hist)):
        w0 += hist[i]
        if w0 == 0:
            continue
        w1 = total - w0
        if w1 == 0:
            break
        sum0 += hist[i] * bin_centers[i]
        mu0 = sum0 / w0
        mu1 = (sum_total - sum0) / w1
        var_between = w0 * w1 * (mu0 - mu1) ** 2
        if var_between > best_var:
            best_var = var_between
            best_thresh = bin_centers[i]

    return float(best_thresh), "Otsu (bimodal confidence split)"


# ===========================================================================
# 3. SPLIT REPRODUCTION + METRIC HELPERS
# ===========================================================================
def reproduce_split(labels, train_frac=0.6, val_frac=0.2, seed=42):
    """Re-creates preprocess.py's stratified 60/20/20 split as boolean HxW masks."""
    from sklearn.model_selection import train_test_split
    rows, cols = np.where(labels > 0)
    cls = labels[rows, cols]
    idx = np.arange(len(cls))
    train_idx, temp_idx = train_test_split(
        idx, train_size=train_frac, random_state=seed, stratify=cls)
    val_size = val_frac / (1 - train_frac)
    val_idx, test_idx = train_test_split(
        temp_idx, train_size=val_size, random_state=seed, stratify=cls[temp_idx])
    masks = {}
    for name, ids in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        m = np.zeros(labels.shape, dtype=bool)
        m[rows[ids], cols[ids]] = True
        masks[name] = m
    return masks


def prf(y_true, y_pred):
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def best_f1_threshold(y_true, scores):
    from sklearn.metrics import precision_recall_curve
    p, r, t = precision_recall_curve(y_true, scores)
    f1 = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-12)
    i = int(np.argmax(f1))
    return float(t[i]), float(f1[i])


# ===========================================================================
# 4. ACE DETECTOR
# ===========================================================================
def estimate_background_stats(x2d, chunk=100_000):
    n, b = x2d.shape
    mean = np.zeros(b, dtype=np.float64)
    for s in range(0, n, chunk):
        mean += x2d[s:s + chunk].sum(axis=0, dtype=np.float64)
    mean /= n
    cov = np.zeros((b, b), dtype=np.float64)
    for s in range(0, n, chunk):
        xc = x2d[s:s + chunk].astype(np.float64) - mean
        cov += xc.T @ xc
    cov /= max(n - 1, 1)
    return mean, cov


def build_whitener(cov, reg):
    """W such that W cov W^T ~ I; `reg` shrinks small eigenvalues (covariance is near-singular)."""
    cov = (cov + cov.T) / 2
    evals, evecs = np.linalg.eigh(cov)
    evals = np.clip(evals, 0, None)
    evals = evals + reg * evals.mean() + 1e-12
    return (evecs / np.sqrt(evals)).T


def compute_ace_map(hsi_cube, target_spectrum, reg=1e-3, chunk=100_000):
    """ACE score in [0, 1] for every pixel of an (H, W, B) cube."""
    h, w, b = hsi_cube.shape
    x = np.nan_to_num(hsi_cube.reshape(-1, b).astype(np.float32))
    s = np.asarray(target_spectrum, dtype=np.float64).reshape(-1)
    if s.shape[0] != b:
        raise ValueError(f"Target spectrum has {s.shape[0]} bands but the cube has {b}.")
    mean, cov = estimate_background_stats(x, chunk)
    W = build_whitener(cov, reg)
    sw = W @ (s - mean)
    s_norm2 = float(sw @ sw)
    if s_norm2 < 1e-12:
        raise ValueError("Target spectrum equals the scene mean -- ACE is undefined.")
    out = np.empty(x.shape[0], dtype=np.float32)
    for st in range(0, x.shape[0], chunk):
        xw = (x[st:st + chunk].astype(np.float64) - mean) @ W.T
        num = (xw @ sw) ** 2
        den = s_norm2 * np.einsum("ij,ij->i", xw, xw) + 1e-12
        out[st:st + chunk] = np.clip(num / den, 0.0, 1.0)
    return out.reshape(h, w)


def compute_multiscale_ace(hsi_cube, target_spectrum, regs=None, chunk=100_000):
    """Run ACE at multiple regularization levels and average the score maps.
    This captures both subtle and strong spectral anomalies."""
    if regs is None:
        regs = ACE_MULTISCALE_REGS
    maps = []
    for reg in regs:
        try:
            m = compute_ace_map(hsi_cube, target_spectrum, reg=reg, chunk=chunk)
            maps.append(m)
        except ValueError:
            continue
    if not maps:
        raise ValueError("All multi-scale ACE runs failed.")
    # Geometric mean preserves the dynamic range better than arithmetic mean
    stacked = np.stack(maps, axis=0)
    # Use log-space averaging to compute geometric mean safely
    log_maps = np.log(stacked + 1e-12)
    geo_mean = np.exp(log_maps.mean(axis=0))
    print(f"Multi-scale ACE: combined {len(maps)} scales (regs={[r for r in regs[:len(maps)]]})")
    return geo_mean.astype(np.float32)


# ===========================================================================
# 4b. SAM DETECTOR (Spectral Angle Mapper)
# ===========================================================================
def compute_sam_map(hsi_cube, target_spectrum, chunk=100_000):
    """Spectral Angle Mapper: similarity = 1 - arccos(cos_sim)/pi.
    Returns a score in [0, 1] where 1 = identical spectrum."""
    h, w, b = hsi_cube.shape
    x = np.nan_to_num(hsi_cube.reshape(-1, b).astype(np.float64))
    s = np.asarray(target_spectrum, dtype=np.float64).reshape(-1)
    if s.shape[0] != b:
        raise ValueError(f"Target spectrum has {s.shape[0]} bands but the cube has {b}.")
    s_norm = np.linalg.norm(s)
    if s_norm < 1e-12:
        raise ValueError("Target spectrum is all zeros -- SAM is undefined.")
    s_unit = s / s_norm

    out = np.empty(x.shape[0], dtype=np.float32)
    for st in range(0, x.shape[0], chunk):
        block = x[st:st + chunk]
        x_norms = np.linalg.norm(block, axis=1)
        x_norms = np.maximum(x_norms, 1e-12)
        cos_sim = (block @ s_unit) / x_norms
        cos_sim = np.clip(cos_sim, -1.0, 1.0)
        angles = np.arccos(cos_sim)  # 0 = identical, pi = opposite
        similarity = 1.0 - angles / np.pi
        out[st:st + chunk] = similarity.astype(np.float32)
    return out.reshape(h, w)


# ===========================================================================
# 4b2. RX ANOMALY DETECTOR (Reed-Xiaoli)
# ===========================================================================
def compute_rx_map(hsi_cube, reg=1e-3, chunk=100_000):
    """Reed-Xiaoli (RX) unsupervised anomaly detector:
    RX(x) = (x - mu)^T Sigma^-1 (x - mu).
    Evaluated using the whitener matrix W (where W Sigma W^T ~ I):
    x_w = (x - mu) @ W.T
    RX(x) = sum(x_w^2, axis=-1)
    Returns normalized anomaly score in [0, 1].
    """
    h, w, b = hsi_cube.shape
    x = np.nan_to_num(hsi_cube.reshape(-1, b).astype(np.float32))
    mean, cov = estimate_background_stats(x, chunk)
    W = build_whitener(cov, reg)
    out = np.empty(x.shape[0], dtype=np.float32)
    for st in range(0, x.shape[0], chunk):
        xw = (x[st:st + chunk].astype(np.float64) - mean) @ W.T
        rx_val = np.sum(xw ** 2, axis=1)
        out[st:st + chunk] = rx_val.astype(np.float32)
    lo, hi = float(out.min()), float(out.max())
    if hi - lo > 1e-12:
        out = (out - lo) / (hi - lo)
    return out.reshape(h, w)


# ===========================================================================
# 4c. ENSEMBLE SCORE
# ===========================================================================
def compute_ensemble_score(confidence_map, ace_map=None, sam_map=None,
                           w_fusion=0.5, w_ace=0.3, w_sam=0.2):
    """Weighted combination of fusion, ACE, and SAM scores.
    Each map is min-max normalized to [0, 1] before combining."""
    def _norm(m):
        lo, hi = m.min(), m.max()
        if hi - lo < 1e-12:
            return np.zeros_like(m)
        return (m - lo) / (hi - lo)

    score = w_fusion * _norm(confidence_map)
    total_w = w_fusion
    if ace_map is not None:
        score += w_ace * _norm(ace_map)
        total_w += w_ace
    if sam_map is not None:
        score += w_sam * _norm(sam_map)
        total_w += w_sam
    return (score / total_w).astype(np.float32)


# ===========================================================================
# 4d. SPECTRAL CONTRAST
# ===========================================================================
def compute_spectral_contrast(hsi_raw, target_rows, target_cols, pad=7):
    """Spectral contrast ratio: how spectrally distinct the target pixels are
    from their immediate surroundings (higher = more anomalous)."""
    h, w, b = hsi_raw.shape
    r0, r1 = max(0, target_rows.min() - pad), min(h, target_rows.max() + pad + 1)
    c0, c1 = max(0, target_cols.min() - pad), min(w, target_cols.max() + pad + 1)

    window = hsi_raw[r0:r1, c0:c1, :]
    inside = np.zeros(window.shape[:2], dtype=bool)
    inside[target_rows - r0, target_cols - c0] = True
    ring = window[~inside]

    target_spec = np.nan_to_num(hsi_raw[target_rows, target_cols]).mean(axis=0)
    bg_spec = np.nan_to_num(ring).mean(axis=0) if len(ring) > 0 else np.zeros(b)
    bg_std = np.nan_to_num(ring).std(axis=0) if len(ring) > 0 else np.ones(b)
    bg_std = np.maximum(bg_std, 1e-8)

    # Signal-to-background ratio (average across bands)
    contrast = np.abs(target_spec - bg_spec) / bg_std
    return float(np.mean(contrast))


# ---------------------------------------------------------------------------
def _iter_arrays(obj, name=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).startswith("__"):
                continue
            yield from _iter_arrays(v, f"{name}.{k}" if name else str(k))
    elif isinstance(obj, (list, tuple)) or (isinstance(obj, np.ndarray) and obj.dtype == object):
        for i, v in enumerate(obj):
            yield from _iter_arrays(v, f"{name}[{i}]")
    elif isinstance(obj, np.ndarray) and obj.dtype.kind in "fiu":
        yield name, obj


def spectrum_from_file(path, n_bands):
    """Best-effort: find a numeric array with one axis == n_bands and average the rest."""
    raw = loadmat(path, simplify_cells=True) if path.lower().endswith(".mat") else {"arr": np.load(path)}
    cands = [(n, a) for n, a in _iter_arrays(raw) if 1 <= a.ndim <= 2 and n_bands in a.shape and a.size < 100_000]
    if not cands:
        raise ValueError(f"No array with a {n_bands}-band axis found in {path}.")
    cands.sort(key=lambda t: t[1].ndim)  # prefer 1-D
    name, arr = cands[0]
    if arr.ndim == 2:
        arr = arr.mean(axis=1) if arr.shape[0] == n_bands else arr.mean(axis=0)
    print(f"ACE target spectrum taken from '{name}' in {path}")
    return arr.astype(np.float64)


def get_ace_signature(args, kind, hsi_raw, labels, confidence_map, sig_mask):
    """Returns (spectrum, description) or (None, reason)."""
    n_bands = hsi_raw.shape[-1]
    src = args.ace_target_source
    if src == "auto":
        if args.ace_target_file:
            src = "file"
        elif kind in ("muufl", "trento") and labels is not None:
            src = "labels"
        else:
            src = "model"

    if src == "file":
        if not args.ace_target_file:
            return None, "--ace_target_source file needs --ace_target_file"
        return spectrum_from_file(args.ace_target_file, n_bands), f"file {args.ace_target_file}"

    if src == "labels":
        tid = args.target_label_id
        if labels is None or tid is None:
            return None, "no labels / target_label_id for label-based signature"
        m = labels == tid
        if sig_mask is not None:
            m &= sig_mask
        if m.sum() == 0:
            return None, "no labeled target pixels available for the signature"
        scope = "train+val target pixels" if sig_mask is not None else "ALL labeled target pixels"
        return np.nan_to_num(hsi_raw[m]).mean(axis=0), f"mean of {int(m.sum())} {scope}"

    # src == "model": self-supervised -- mean spectrum of the fusion model's most confident pixels
    if confidence_map.max() < 0.3:
        return None, "fusion model found no plausible target (max confidence < 0.3); use --ace_target_file"
    m = confidence_map >= 0.9
    if m.sum() < 3:
        k = min(20, confidence_map.size)
        thr = np.sort(confidence_map.flatten())[-k]
        m = confidence_map >= thr
    return np.nan_to_num(hsi_raw[m]).mean(axis=0), f"mean of {int(m.sum())} most-confident fusion pixels (self-supervised)"


def default_ace_threshold(ace_map, confidence_map):
    """No labels to tune on: ACE must exceed the 99th percentile of 'clutter', i.e. of the
    ACE scores at pixels the fusion model is confident are NOT targets (falls back to the
    scene-wide 99th percentile)."""
    bg = confidence_map < 0.1
    if bg.sum() > 100:
        return float(np.percentile(ace_map[bg], 99)), "99th percentile of ACE over fusion-background pixels"
    return float(np.percentile(ace_map, 99)), "99th percentile of all ACE scores"


def evaluate_detectors(labels, target_id, eval_mask, select_mask, select_desc,
                       ace_map, confidence_map, class_map, sam_map=None,
                       ensemble_map=None, ace_threshold=None, rx_map=None):
    """ACE vs fusion (+ SAM + RX + ensemble) on eval pixels. ACE threshold is chosen on select pixels only."""
    from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

    y_eval = labels[eval_mask] == target_id
    out = {"n_eval": int(y_eval.size), "n_eval_targets": int(y_eval.sum())}
    if y_eval.sum() == 0 or y_eval.all():
        out["error"] = "evaluation set has no positives (or no negatives)"
        return out, None

    conf_eval = confidence_map[eval_mask]
    fus_pred = class_map[eval_mask] == target_id
    p, r, f1 = prf(y_eval, fus_pred)
    fpr, tpr, _ = roc_curve(y_eval, conf_eval)
    out["fusion"] = {
        "auc": float(roc_auc_score(y_eval, conf_eval)),
        "ap": float(average_precision_score(y_eval, conf_eval)),
        "pd_at_1pct_fpr": float(np.interp(0.01, fpr, tpr)),
        "precision": p, "recall": r, "f1": f1, "operating_point": "argmax class",
    }
    curves = {"y": y_eval, "fusion": conf_eval, "ace": None, "sam": None, "rx": None, "ensemble": None}

    if ace_map is not None:
        ace_eval = ace_map[eval_mask]
        thr, thr_src = ace_threshold, "user-specified"
        if thr is None:
            y_sel = labels[select_mask] == target_id
            if y_sel.sum() > 0:
                thr, _ = best_f1_threshold(y_sel, ace_map[select_mask])
                thr_src = f"best-F1 on {select_desc}"
        if thr is None:
            thr, thr_src = default_ace_threshold(ace_map, confidence_map)
        ap_, ar_, af1 = prf(y_eval, ace_eval >= thr)
        fpr_a, tpr_a, _ = roc_curve(y_eval, ace_eval)
        out["ace"] = {
            "auc": float(roc_auc_score(y_eval, ace_eval)),
            "ap": float(average_precision_score(y_eval, ace_eval)),
            "pd_at_1pct_fpr": float(np.interp(0.01, fpr_a, tpr_a)),
            "precision": ap_, "recall": ar_, "f1": af1,
            "threshold": float(thr), "threshold_source": thr_src,
        }
        curves["ace"] = ace_eval

    # SAM detector metrics
    if sam_map is not None:
        sam_eval = sam_map[eval_mask]
        sam_thr, _ = best_f1_threshold(labels[select_mask] == target_id, sam_map[select_mask]) \
            if (labels[select_mask] == target_id).sum() > 0 else (0.9, 0.0)
        sp, sr, sf1 = prf(y_eval, sam_eval >= sam_thr)
        fpr_s, tpr_s, _ = roc_curve(y_eval, sam_eval)
        out["sam"] = {
            "auc": float(roc_auc_score(y_eval, sam_eval)),
            "ap": float(average_precision_score(y_eval, sam_eval)),
            "pd_at_1pct_fpr": float(np.interp(0.01, fpr_s, tpr_s)),
            "precision": sp, "recall": sr, "f1": sf1,
            "threshold": float(sam_thr), "threshold_source": f"best-F1 on {select_desc}",
        }
        curves["sam"] = sam_eval

    # RX anomaly detector metrics
    if rx_map is not None:
        rx_eval = rx_map[eval_mask]
        rx_thr, _ = best_f1_threshold(labels[select_mask] == target_id, rx_map[select_mask]) \
            if (labels[select_mask] == target_id).sum() > 0 else (0.5, 0.0)
        rp, rr_, rf1 = prf(y_eval, rx_eval >= rx_thr)
        fpr_r, tpr_r, _ = roc_curve(y_eval, rx_eval)
        out["rx"] = {
            "auc": float(roc_auc_score(y_eval, rx_eval)),
            "ap": float(average_precision_score(y_eval, rx_eval)),
            "pd_at_1pct_fpr": float(np.interp(0.01, fpr_r, tpr_r)),
            "precision": rp, "recall": rr_, "f1": rf1,
            "threshold": float(rx_thr), "threshold_source": f"best-F1 on {select_desc}",
        }
        curves["rx"] = rx_eval

    # Ensemble metrics
    if ensemble_map is not None:
        ens_eval = ensemble_map[eval_mask]
        ens_thr, _ = best_f1_threshold(labels[select_mask] == target_id, ensemble_map[select_mask]) \
            if (labels[select_mask] == target_id).sum() > 0 else (0.5, 0.0)
        ep, er, ef1 = prf(y_eval, ens_eval >= ens_thr)
        fpr_e, tpr_e, _ = roc_curve(y_eval, ens_eval)
        out["ensemble"] = {
            "auc": float(roc_auc_score(y_eval, ens_eval)),
            "ap": float(average_precision_score(y_eval, ens_eval)),
            "pd_at_1pct_fpr": float(np.interp(0.01, fpr_e, tpr_e)),
            "precision": ep, "recall": er, "f1": ef1,
            "threshold": float(ens_thr), "threshold_source": f"best-F1 on {select_desc}",
        }
        curves["ensemble"] = ens_eval

    return out, curves


def format_metrics_table(metrics, split_name):
    lines = [f"Detector comparison on {split_name} pixels "
             f"(n={metrics['n_eval']:,d}, targets={metrics['n_eval_targets']})"]
    if "error" in metrics:
        lines.append(f"  {metrics['error']}")
        return "\n".join(lines)
    lines.append(f"  {'Method':<34s}{'AUC':>7s}{'AP':>7s}{'PD@1%FPR':>10s}{'Prec':>7s}{'Rec':>7s}{'F1':>7s}")
    lines.append("  " + "-" * 79)
    detector_labels = [
        ("fusion", "Fusion model (argmax class)"),
        ("ace", "ACE (HSI only, no training)"),
        ("sam", "SAM (HSI only, no training)"),
        ("rx", "RX Anomaly (HSI only, unsuperv)"),
        ("ensemble", "Ensemble (Fusion+ACE+SAM)"),
    ]
    for key, label in detector_labels:
        m = metrics.get(key)
        if m:
            lines.append(f"  {label:<34s}{m['auc']:>7.3f}{m['ap']:>7.3f}{m['pd_at_1pct_fpr']:>10.3f}"
                         f"{m['precision']:>7.3f}{m['recall']:>7.3f}{m['f1']:>7.3f}")
    if metrics.get("ace"):
        lines.append(f"  ACE threshold = {metrics['ace']['threshold']:.4f} ({metrics['ace']['threshold_source']})")
    if metrics.get("sam"):
        lines.append(f"  SAM threshold = {metrics['sam']['threshold']:.4f} ({metrics['sam']['threshold_source']})")
    if metrics.get("rx"):
        lines.append(f"  RX threshold = {metrics['rx']['threshold']:.4f} ({metrics['rx']['threshold_source']})")
    if metrics.get("ensemble"):
        lines.append(f"  Ensemble threshold = {metrics['ensemble']['threshold']:.4f} ({metrics['ensemble']['threshold_source']})")
    return "\n".join(lines)


# ===========================================================================
# 5. TARGET IDENTIFICATION
# ===========================================================================
def _classify_color(r, g, b):
    r, g, b = int(r), int(g), int(b)
    brightness = 0.299 * r + 0.587 * g + 0.114 * b
    saturation = max(r, g, b) - min(r, g, b)
    if brightness < 40:
        return "Black/Dark"
    if brightness > 220 and saturation < 30:
        return "White/Bright"
    if saturation < 25:
        return "Dark Gray" if brightness < 120 else "Light Gray"
    if r > g and r > b:
        if g > 150 and r > 180:
            return "Yellow/Gold"
        if g > 100 and b < 80:
            return "Orange/Brown"
        return "Red"
    if g > r and g > b:
        if r > 150:
            return "Yellow/Gold"
        return "Green" if g - r > 30 else "Light Green"
    if b > r and b > g:
        return "Purple/Violet" if (r > 100 and b > 150) else "Blue"
    if abs(r - g) < 20 and b < r:
        return "Brown/Tan"
    return "Mixed Color"


def identify_target_color(rgb_image, rows, cols):
    if rgb_image is None:
        return "Unknown"
    pixels = rgb_image[rows, cols]
    if len(pixels) == 0:
        return "Unknown"
    pixels = (pixels * 255).astype(np.uint8) if pixels.max() <= 1.0 else pixels.astype(np.uint8)
    if len(pixels) < 20:
        m = pixels.mean(axis=0)
        return _classify_color(*m)
    if len(pixels) > 500:
        pixels = pixels[np.random.RandomState(0).choice(len(pixels), 500, replace=False)]
    votes = Counter(_classify_color(*px) for px in pixels)
    total = len(pixels)
    dominant = [c for c, n in votes.most_common(4) if n / total >= 0.10][:3]
    if not dominant:
        return _classify_color(*pixels.mean(axis=0))
    return " + ".join(dominant)


def identify_surrounding_context(class_map, rows, cols, pad=5, class_names=None):
    """Dominant land-cover class in the ring around (not inside) the object."""
    if class_map is None:
        return "Unknown", 0.0
    c_names = class_names if class_names is not None else CLASS_NAMES
    h, w = class_map.shape
    r0, r1 = max(0, rows.min() - pad), min(h, rows.max() + pad + 1)
    c0, c1 = max(0, cols.min() - pad), min(w, cols.max() + pad + 1)
    window = class_map[r0:r1, c0:c1]
    inside = np.zeros(window.shape, dtype=bool)
    inside[rows - r0, cols - c0] = True
    ring = window[~inside]
    if len(ring) == 0:
        return "Unknown", 0.0
    cls, cnt = Counter(ring.tolist()).most_common(1)[0]
    name = c_names[cls - 1] if 1 <= cls <= len(c_names) else f"Class {cls}"
    return name, 100.0 * cnt / len(ring)


def _size_category(n):
    return "Large" if n >= 100 else "Medium" if n >= 20 else "Small" if n >= 5 else "Tiny"


def compute_target_score(size, mean_conf, max_conf, context, kind,
                         ace_confirmed=False, spectral_contrast=0.0):
    """Quality score with bonus for ACE confirmation and high spectral contrast."""
    score = min(size / 50.0, 1.0) * 25 + mean_conf * 35 + max_conf * 15
    if kind == "muufl":
        if context in {"Mostly Grass", "Mixed Ground Surface", "Dirt and Sand", "Trees", "Road"}:
            score += 8
        elif context in {"Buildings", "Sidewalk", "Yellow Curb"}:
            score -= 5
        else:
            score += 3
    elif kind == "trento":
        if context in {"Ground", "Roads", "Vineyard", "Apple trees", "Wood"}:
            score += 8
        else:
            score += 3
    else:
        score += 3  # context plausibility is MUUFL/Trento-specific; stay neutral elsewhere

    # Bonus for ACE cross-check confirmation
    if ace_confirmed:
        score += 7

    # Bonus for high spectral contrast (target stands out from background)
    score += min(spectral_contrast / 5.0, 1.0) * 10

    return max(0.0, min(100.0, score))


def _quality_tier(score):
    if score >= 75:
        return "HIGH CONFIDENCE"
    if score >= 50:
        return "MEDIUM CONFIDENCE"
    if score >= 30:
        return "LOW CONFIDENCE"
    return "LIKELY FALSE POSITIVE"


def identify_detected_targets(confidence_map, class_map, rgb_image, hsi_raw,
                              ace_map, ace_thr, sam_map, ensemble_map,
                              args, kind, lidar_raw=None, rx_map=None, rx_thr=None,
                              class_names=None):
    binary = confidence_map >= args.confidence_threshold
    if args.use_morphology:
        print("Applying morphological filtering...")
        binary = ndimage.binary_closing(binary, structure=np.ones((3, 3)))
        binary = ndimage.binary_opening(binary, structure=np.ones((2, 2)))
    labeled, n_obj = ndimage.label(binary)
    print(f"\nFound {n_obj} candidate object(s) above confidence {args.confidence_threshold} "
          f"before size filtering.")

    noun = "Building" if kind == "trento" else ("Cloth Panel" if kind == "muufl" else "Candidate Object")
    results, coords = [], []
    for i, sl in enumerate(ndimage.find_objects(labeled), start=1):
        if sl is None:
            continue
        sub = labeled[sl] == i
        size = int(sub.sum())
        if size < args.min_size:
            continue
        rr, cc = np.nonzero(sub)
        rr = rr + sl[0].start
        cc = cc + sl[1].start
        conf = confidence_map[rr, cc]
        color = identify_target_color(rgb_image, rr, cc)
        ctx, ctx_pct = identify_surrounding_context(class_map, rr, cc, class_names=class_names)

        # Spectral contrast ratio
        spec_contrast = 0.0
        if hsi_raw is not None:
            try:
                spec_contrast = compute_spectral_contrast(hsi_raw, rr, cc)
            except Exception:
                pass

        # LiDAR 3D Topographic Profiling
        lidar_info = {}
        if lidar_raw is not None:
            try:
                lr = lidar_raw if lidar_raw.ndim == 3 else lidar_raw[:, :, None]
                z0_tgt = lr[rr, cc, 0]
                h_sc, w_sc = lr.shape[:2]
                r0_bg, r1_bg = max(0, int(rr.min()) - 7), min(h_sc, int(rr.max()) + 8)
                c0_bg, c1_bg = max(0, int(cc.min()) - 7), min(w_sc, int(cc.max()) + 8)
                bg_sub = lr[r0_bg:r1_bg, c0_bg:c1_bg, 0]
                mask_sub = np.zeros(bg_sub.shape, dtype=bool)
                mask_sub[rr - r0_bg, cc - c0_bg] = True
                ring_z0 = bg_sub[~mask_sub]
                ground_z = float(np.median(ring_z0)) if len(ring_z0) > 0 else float(np.median(z0_tgt))

                mean_z0 = float(np.mean(z0_tgt))
                rel_h = float(mean_z0 - ground_z)
                std_z0 = float(np.std(z0_tgt))
                dz = float(np.mean(z0_tgt - lr[rr, cc, 1])) if lr.shape[-1] > 1 else 0.0

                lidar_info = {
                    "lidar_elevation_m": round(mean_z0, 2),
                    "lidar_relative_height_m": round(rel_h, 3),
                    "lidar_roughness_std_m": round(std_z0, 3),
                    "lidar_multi_return_dz_m": round(dz, 3),
                }
            except Exception:
                pass

        ace_confirmed = False
        rec = {
            "name": f"{color} {noun} ({_size_category(size)})",
            "dominant_color": color,
            "size_category": _size_category(size),
            "surrounding_context": ctx,
            "surrounding_pct": round(float(ctx_pct), 1),
            "size_pixels": size,
            "centroid": (round(float(rr.mean()), 1), round(float(cc.mean()), 1)),
            "bbox": (int(rr.min()), int(rr.max()), int(cc.min()), int(cc.max())),
            "mean_confidence": float(conf.mean()),
            "max_confidence": float(conf.max()),
            "spectral_contrast": round(spec_contrast, 3),
            **lidar_info,
        }
        if ace_map is not None:
            a = ace_map[rr, cc]
            rec["ace_mean"] = round(float(a.mean()), 4)
            rec["ace_max"] = round(float(a.max()), 4)
            ace_confirmed = a.mean() >= ace_thr if ace_thr else False
            rec["ace_verdict"] = ("CONFIRMED by ACE" if ace_confirmed else "FUSION ONLY (ACE below threshold)")
        if sam_map is not None:
            s = sam_map[rr, cc]
            rec["sam_mean"] = round(float(s.mean()), 4)
            rec["sam_max"] = round(float(s.max()), 4)
        if rx_map is not None:
            r_vals = rx_map[rr, cc]
            rec["rx_mean"] = round(float(r_vals.mean()), 4)
            rec["rx_max"] = round(float(r_vals.max()), 4)
            rx_confirmed = r_vals.mean() >= rx_thr if rx_thr else False
            rec["rx_verdict"] = ("ANOMALOUS (confirmed by RX)" if rx_confirmed else "Standard background spectral profile")
        if ensemble_map is not None:
            e = ensemble_map[rr, cc]
            rec["ensemble_mean"] = round(float(e.mean()), 4)
            rec["ensemble_max"] = round(float(e.max()), 4)

        score = compute_target_score(size, float(conf.mean()), float(conf.max()), ctx, kind,
                                     ace_confirmed=ace_confirmed, spectral_contrast=spec_contrast)
        rec["quality_score"] = round(score, 1)
        rec["quality_tier"] = _quality_tier(score)

        results.append(rec)
        coords.append((rr, cc))

    order = sorted(range(len(results)), key=lambda k: results[k]["quality_score"], reverse=True)
    results = [results[k] for k in order]
    coords = [coords[k] for k in order]
    for new_id, rec in enumerate(results, start=1):
        rec["id"] = new_id
        rec["name"] = rec["name"]
    # put id first for readable reports
    results = [{"id": r.pop("id"), **r} for r in results]
    return results, coords


# ===========================================================================
# 5b. PROXIMITY-BASED TARGET MERGING
# ===========================================================================
def merge_nearby_targets(results, coords, merge_radius=8):
    """Merge targets whose centroids are within merge_radius pixels of each other.
    Keeps the higher-quality target's metadata, combines pixel counts."""
    if len(results) <= 1:
        return results, coords

    centroids = np.array([r["centroid"] for r in results])
    merged = [False] * len(results)
    new_results, new_coords = [], []

    for i in range(len(results)):
        if merged[i]:
            continue
        group_ids = [i]
        for j in range(i + 1, len(results)):
            if merged[j]:
                continue
            dist = np.sqrt((centroids[i][0] - centroids[j][0]) ** 2 +
                           (centroids[i][1] - centroids[j][1]) ** 2)
            if dist <= merge_radius:
                group_ids.append(j)
                merged[j] = True

        if len(group_ids) == 1:
            new_results.append(results[i])
            new_coords.append(coords[i])
        else:
            # Merge: keep the highest-quality entry, combine pixels
            best_idx = max(group_ids, key=lambda k: results[k]["quality_score"])
            best = results[best_idx].copy()
            all_rr = np.concatenate([coords[k][0] for k in group_ids])
            all_cc = np.concatenate([coords[k][1] for k in group_ids])
            # De-duplicate pixels
            pixel_set = set(zip(all_rr.tolist(), all_cc.tolist()))
            all_rr = np.array([p[0] for p in pixel_set])
            all_cc = np.array([p[1] for p in pixel_set])

            best["size_pixels"] = len(pixel_set)
            best["size_category"] = _size_category(len(pixel_set))
            best["centroid"] = (round(float(all_rr.mean()), 1), round(float(all_cc.mean()), 1))
            best["bbox"] = (int(all_rr.min()), int(all_rr.max()), int(all_cc.min()), int(all_cc.max()))
            best["merged_from"] = len(group_ids)
            new_results.append(best)
            new_coords.append((all_rr, all_cc))

    # Re-number
    for new_id, rec in enumerate(new_results, start=1):
        rec["id"] = new_id
    n_merged = len(results) - len(new_results)
    if n_merged > 0:
        print(f"Proximity merging: {len(results)} -> {len(new_results)} targets "
              f"({n_merged} merged, radius={merge_radius}px)")
    return new_results, new_coords


# ===========================================================================
# 6. REPORTS AND FIGURES
# ===========================================================================
def save_target_reports(out_dir, results, threshold, ace_thr):
    print(f"\n{'=' * 70}\n IDENTIFIED TARGETS: {len(results)} object(s)\n{'=' * 70}\n")
    lines = [f"Detected {len(results)} candidate target(s), confidence threshold = {threshold}",
             "=" * 70, ""]
    for o in results:
        warn = "  *** WARNING ***" if o["quality_tier"] == "LIKELY FALSE POSITIVE" else ""
        block = [
            f"Target #{o['id']}: {o['name']}",
            f"  Quality Score: {o['quality_score']}/100 -- {o['quality_tier']}{warn}",
            f"  Location: in {o['surrounding_context']} area ({o['surrounding_pct']:.0f}% of surroundings)",
            f"  Color: {o['dominant_color']}",
            f"  Size: {o['size_pixels']}px ({o['size_category']})",
            f"  Centroid (row, col): {o['centroid']}   Bbox: {o['bbox']}",
            f"  Fusion confidence: mean={o['mean_confidence']:.3f}, max={o['max_confidence']:.3f}",
            f"  Spectral contrast ratio: {o.get('spectral_contrast', 0):.3f}",
        ]
        if "lidar_relative_height_m" in o:
            block.append(f"  LiDAR 3D Profile: elevation={o['lidar_elevation_m']:.2f}m, "
                         f"rel_height={o['lidar_relative_height_m']:+.2f}m (vs ground), "
                         f"roughness={o['lidar_roughness_std_m']:.3f}m, multi-return dz={o['lidar_multi_return_dz_m']:.3f}m")
        if "ace_mean" in o:
            block.append(f"  ACE score: mean={o['ace_mean']:.3f}, max={o['ace_max']:.3f} "
                         f"(threshold {ace_thr:.3f}) -> {o['ace_verdict']}")
        if "sam_mean" in o:
            block.append(f"  SAM score: mean={o['sam_mean']:.3f}, max={o['sam_max']:.3f}")
        if "rx_mean" in o:
            block.append(f"  RX anomaly: mean={o['rx_mean']:.3f}, max={o['rx_max']:.3f} -> {o.get('rx_verdict', '')}")
        if "ensemble_mean" in o:
            block.append(f"  Ensemble score: mean={o['ensemble_mean']:.3f}, max={o['ensemble_max']:.3f}")
        if o.get("merged_from", 1) > 1:
            block.append(f"  (Merged from {o['merged_from']} nearby fragments)")
        block.append("")
        print("\n".join(block))
        lines += block
    with open(os.path.join(out_dir, "targets_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(os.path.join(out_dir, "targets_report.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    if results:
        all_keys = []
        for r in results:
            for k in r.keys():
                if k not in all_keys:
                    all_keys.append(k)
        with open(os.path.join(out_dir, "targets_report.csv"), "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            wr.writeheader()
            wr.writerows(results)
    print(f"Reports saved to {out_dir}")


def _crop(rgb_image, bbox, pad):
    r0, r1, c0, c1 = bbox
    return rgb_image[max(0, r0 - pad):min(rgb_image.shape[0], r1 + pad + 1),
                     max(0, c0 - pad):min(rgb_image.shape[1], c1 + pad + 1)]


def save_target_figures(out_dir, rgb_image, results, args):
    if rgb_image is None or not results:
        return
    plt = get_plt()
    import matplotlib.patches as patches

    visual = results[:args.max_visual_targets]
    n = len(visual)
    ncols = min(n, 6)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.8 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, o in zip(axes.flat, visual):
        ax.imshow(_crop(rgb_image, o["bbox"], args.crop_padding))
        extra = f" | ACE={o['ace_mean']:.2f}" if "ace_mean" in o else ""
        if "ensemble_mean" in o:
            extra += f" | Ens={o['ensemble_mean']:.2f}"
        ax.set_title(f"#{o['id']}: {o['name']}\nin {o['surrounding_context']} | "
                     f"conf={o['mean_confidence']:.2f}{extra}", fontsize=8, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "target_crops.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 14))
    ax.imshow(rgb_image)
    for o in results[:200]:
        r0, r1, c0, c1 = o["bbox"]
        conf = o["mean_confidence"]
        colr = "#00FF00" if conf >= 0.8 else "#FFD700" if conf >= 0.6 else "#FF4444"
        ax.add_patch(patches.Rectangle((c0, r0), c1 - c0, r1 - r0, linewidth=2.5,
                                       edgecolor=colr, facecolor="none"))
        if o["id"] <= args.max_visual_targets:
            tag = f"#{o['id']}: {o['dominant_color']} ({o['size_category']}) conf={conf:.2f}"
            if "ace_mean" in o:
                tag += f" ACE={o['ace_mean']:.2f}"
            ax.text(c0, max(0, r0 - 4), tag, color="white", fontsize=7, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=colr, alpha=0.85, edgecolor="none"))
    ax.axis("off")
    ax.set_title("Full Scene -- Identified Targets", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "full_scene_boxes.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Target crops + full-scene boxes saved to {out_dir}")


def save_enhanced_yolo(out_dir, rgb_image, results, args):
    """Optional: upscale/denoise/sharpen each crop and run YOLOv8 on it."""
    if rgb_image is None or not results:
        return
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as ie:
        print(f"Skipping enhanced analysis (missing dependency: {ie}). "
              f"Install with: pip install opencv-python ultralytics")
        return
    plt = get_plt()
    try:
        print("\n--- Enhanced Target Analysis (Super-Resolution + YOLOv8) ---")
        model = YOLO("yolov8n.pt")
        targets = results[:min(args.max_visual_targets, 6)]
        fig, axes = plt.subplots(2, len(targets), figsize=(5 * len(targets), 10), squeeze=False)
        for j, o in enumerate(targets):
            crop = _crop(rgb_image, o["bbox"], args.crop_padding).copy()
            crop = (crop * 255).astype(np.uint8) if crop.max() <= 1.0 else crop.astype(np.uint8)  # RGB
            axes[0, j].imshow(crop)
            axes[0, j].set_title(f"BEFORE\n#{o['id']}: {o['name']}", fontsize=8, fontweight="bold", color="red")
            axes[0, j].axis("off")

            hc, wc = crop.shape[:2]
            scale = max(8, 128 // max(hc, 1), 128 // max(wc, 1))
            enh = cv2.resize(crop, (wc * scale, hc * scale), interpolation=cv2.INTER_CUBIC)
            enh = cv2.fastNlMeansDenoisingColored(enh, None, 6, 6, 7, 21)
            enh = cv2.addWeighted(enh, 1.5, cv2.GaussianBlur(enh, (0, 0), sigmaX=2.0), -0.5, 0)

            # ultralytics treats numpy input as BGR; `enh` is RGB
            det = model(cv2.cvtColor(enh, cv2.COLOR_RGB2BGR), conf=0.25, verbose=False)
            shown, labels_txt = enh.copy(), []
            if len(det) > 0 and det[0].boxes is not None:
                for box in det[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    name, cf = model.names[int(box.cls[0])], float(box.conf[0])
                    labels_txt.append(f"{name} ({cf:.2f})")
                    cv2.rectangle(shown, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(shown, f"{name} {cf:.2f}", (x1 + 2, max(12, y1 - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
            axes[1, j].imshow(shown)
            axes[1, j].set_title("AFTER (Enhanced + YOLO)\n" + ("\n".join(labels_txt[:3]) if labels_txt else
                                 "no standard (COCO) objects detected"), fontsize=7, fontweight="bold", color="green")
            axes[1, j].axis("off")
            cv2.imwrite(os.path.join(out_dir, f"target_{o['id']}_enhanced.png"), cv2.cvtColor(shown, cv2.COLOR_RGB2BGR))
        fig.suptitle("Before vs After: Super-Resolution + YOLOv8", fontsize=14, fontweight="bold")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "target_before_after.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved target_before_after.png and per-target enhanced crops.")
    except Exception as e:
        print(f"Enhanced analysis error: {e}")


def save_spectral_plots(out_dir, hsi_raw, wavelengths, results, coords, confidence_map, max_targets):
    if not results:
        return
    plt = get_plt()
    n_bands = hsi_raw.shape[-1]
    x_axis = wavelengths if wavelengths is not None else np.arange(n_bands)
    x_label = "Wavelength (nm)" if wavelengths is not None else "Band Index"
    palette = ["#E53935", "#43A047", "#1E88E5", "#FB8C00", "#8E24AA", "#00ACC1", "#F4511E", "#3949AB"]

    bg_mask = confidence_map < 0.1
    bg = None
    if bg_mask.sum() > 100:
        rr, cc = np.nonzero(bg_mask)
        sel = np.random.RandomState(0).choice(len(rr), min(20000, len(rr)), replace=False)
        px = np.nan_to_num(hsi_raw[rr[sel], cc[sel]])
        bg = (px.mean(axis=0), px.std(axis=0))

    for fname, with_bg in (("spectral_signatures.png", False), ("spectral_target_vs_background.png", True)):
        if with_bg and bg is None:
            continue
        fig, ax = plt.subplots(figsize=(12, 6))
        if with_bg:
            ax.plot(x_axis, bg[0], color="#888888", linewidth=2, linestyle="--", label="Background (non-target)")
            ax.fill_between(x_axis, bg[0] - bg[1], bg[0] + bg[1], alpha=0.1, color="#888888")
        for idx, (o, (rr, cc)) in enumerate(list(zip(results, coords))[:max_targets]):
            px = np.nan_to_num(hsi_raw[rr, cc])
            mean, std = px.mean(axis=0), px.std(axis=0)
            col = palette[idx % len(palette)]
            ax.plot(x_axis, mean, color=col, linewidth=1.5, label=f"#{o['id']}: {o['name']} [{o['quality_score']}/100]")
            if not with_bg:
                ax.fill_between(x_axis, mean - std, mean + std, alpha=0.15, color=col)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Reflectance / Intensity")
        ax.set_title("Target vs Background Spectral Comparison" if with_bg else "Spectral Signatures of Detected Targets",
                     fontsize=14, fontweight="bold")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches="tight")
        plt.close(fig)
    print("Spectral signature plots saved.")


def save_target_lidar_profiles(out_dir, lidar_raw, results, coords, max_targets=6):
    """Generates 2D LiDAR height crops and cross-sectional elevation profiles for detected targets."""
    if lidar_raw is None or not results:
        return
    plt = get_plt()
    targets = results[:max_targets]
    n = len(targets)
    lr = lidar_raw if lidar_raw.ndim == 3 else lidar_raw[:, :, None]
    z0 = lr[:, :, 0]
    h, w = z0.shape

    fig, axes = plt.subplots(n, 2, figsize=(11, 3.2 * n), squeeze=False)
    for i, (o, (rr, cc)) in enumerate(zip(targets, coords[:n])):
        r0, r1, c0, c1 = o["bbox"]
        pad = 8
        r_start, r_end = max(0, r0 - pad), min(h, r1 + pad + 1)
        c_start, c_end = max(0, c0 - pad), min(w, c1 + pad + 1)

        crop_z0 = z0[r_start:r_end, c_start:c_end]

        # 1. 2D Height Map with target boundary
        im = axes[i, 0].imshow(crop_z0, cmap="viridis", aspect="auto")
        t_mask = np.zeros(crop_z0.shape, dtype=bool)
        t_mask[rr - r_start, cc - c_start] = True
        try:
            axes[i, 0].contour(t_mask, levels=[0.5], colors=["#FF1744"], linewidths=[2])
        except Exception:
            pass
        axes[i, 0].set_title(f"Target #{o['id']} LiDAR Elevation (m)\n{o['name']}", fontsize=9, fontweight="bold")
        fig.colorbar(im, ax=axes[i, 0], fraction=0.046, pad=0.04)
        axes[i, 0].axis("off")

        # 2. Cross-section profile across target centroid
        cent_r = int(round(o["centroid"][0]))
        if 0 <= cent_r < h:
            profile_x = np.arange(c_start, c_end)
            profile_z = z0[cent_r, c_start:c_end]
            axes[i, 1].plot(profile_x, profile_z, color="#1565C0", linewidth=2, label="Elevation profile")
            tgt_cols_in_row = cc[rr == cent_r]
            if len(tgt_cols_in_row) > 0:
                axes[i, 1].axvspan(tgt_cols_in_row.min(), tgt_cols_in_row.max(), color="#FF1744", alpha=0.3, label="Target extent")
            axes[i, 1].axhline(o.get("lidar_elevation_m", profile_z.mean()), color="black", linestyle="--", alpha=0.6,
                               label=f"Avg Target H: {o.get('lidar_elevation_m', 0):.2f}m")
        axes[i, 1].set_xlabel("Column Index", fontsize=8)
        axes[i, 1].set_ylabel("Height (m)", fontsize=8)
        axes[i, 1].set_title(f"Elevation Profile (rel: {o.get('lidar_relative_height_m', 0):+.2f}m vs ground)", fontsize=9, fontweight="bold")
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "target_lidar_profiles.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Target LiDAR 3D elevation profiles saved.")


# ===========================================================================
# 7. FIGURES: detection overview, label evaluation, ACE, calibration
# ===========================================================================
def save_detection_visualization(run_out, rgb_image, labels, class_map, confidence_map,
                                 class_names=None, class_colors_hex=None,
                                 target_name=None, target_class_idx=None):
    plt = get_plt()
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch

    c_names = class_names if class_names is not None else CLASS_NAMES
    c_colors = class_colors_hex if class_colors_hex is not None else CLASS_COLORS_HEX
    tgt_name = target_name if target_name is not None else "Cloth Panels"
    tgt_idx = target_class_idx if target_class_idx is not None else TARGET_CLASS_IDX

    cmap = ListedColormap([hex_to_rgb(c) + (1.0,) for c in c_colors], name="semantic_palette")
    norm = BoundaryNorm(np.arange(-0.5, len(c_colors) + 0.5, 1), cmap.N)
    legend = [Patch(facecolor=c_colors[i + 1], edgecolor="black", linewidth=0.5, label=c_names[i])
              for i in range(len(c_names))]

    unique, counts = np.unique(class_map, return_counts=True)
    total = class_map.size
    lines = ["\nPredicted class distribution:", f"  {'Class':<30s} {'Pixels':>10s}  {'%':>6s}", "  " + "-" * 50]
    bar_counts = []
    for cid, cname in enumerate(c_names, start=1):
        cnt = int(counts[unique == cid][0]) if cid in unique else 0
        bar_counts.append(cnt)
        mark = "  <<<" if cid == tgt_idx + 1 else ""
        lines.append(f"  {cname:<30s} {cnt:>10,d}  {100.0 * cnt / total:>5.1f}%{mark}")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(run_out, "class_distribution.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")

    has_gt = labels is not None
    n_cols = 3
    fig = plt.figure(figsize=(21, 14))
    ax = fig.add_subplot(2, n_cols, 1)
    ax.imshow(rgb_image)
    ax.set_title("Input Scene (RGB / False-Color)", fontsize=12, fontweight="bold")
    ax.axis("off")
    pos = 2
    if has_gt:
        ax = fig.add_subplot(2, n_cols, pos)
        gt = np.ma.masked_where(labels <= 0, labels.astype(float))
        ax.imshow(np.zeros((*labels.shape, 3)))
        ax.imshow(gt, cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title("Ground Truth", fontsize=12, fontweight="bold")
        ax.axis("off")
        pos += 1
    ax = fig.add_subplot(2, n_cols, pos)
    ax.imshow(class_map, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_title("Predicted Class Map", fontsize=12, fontweight="bold")
    ax.axis("off")
    ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, title="Class Legend")

    ax = fig.add_subplot(2, n_cols, n_cols + 1)
    im = ax.imshow(confidence_map, cmap="hot", vmin=0, vmax=1)
    ax.set_title(f"Target ({tgt_name}) Confidence", fontsize=12, fontweight="bold")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(2, n_cols, n_cols + 2)
    ax.imshow(rgb_image)
    overlay = plt.get_cmap("hot")(confidence_map)
    overlay[..., 3] = np.where(confidence_map > 0.3, np.clip((confidence_map - 0.3) / 0.7, 0, 1), 0)
    ax.imshow(overlay)
    ax.set_title("Target Confidence Overlay", fontsize=12, fontweight="bold")
    ax.axis("off")

    ax = fig.add_subplot(2, n_cols, n_cols + 3)
    ypos = np.arange(len(c_names))
    ax.barh(ypos, bar_counts, color=c_colors[1:len(c_names) + 1], edgecolor="black", linewidth=0.5, height=0.7)
    ax.set_yticks(ypos)
    ax.set_yticklabels(c_names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Pixel Count")
    ax.set_title("Class Distribution", fontsize=12, fontweight="bold")
    for i, cnt in enumerate(bar_counts):
        ax.text(cnt + total * 0.005, i, f"{100.0 * cnt / total:.1f}%", va="center", fontsize=7, fontweight="bold")
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.35, hspace=0.15)
    fig.savefig(os.path.join(run_out, "detection_visualization.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved detection_visualization.png")


def save_label_evaluation(run_out, labels, class_map, kind, split_masks, class_names=None):
    from sklearn.metrics import classification_report, confusion_matrix
    plt = get_plt()
    c_names = class_names if class_names is not None else CLASS_NAMES
    labeled = labels > 0
    gt, pred = labels[labeled], class_map[labeled]
    acc = float((gt == pred).mean())
    print(f"\nAccuracy on all labeled pixels: {acc:.3f}")
    if kind not in ("muufl", "trento"):
        print("NOTE: this scene's class space differs from MUUFL's 11 classes, so the numbers below "
              "are NOT a meaningful accuracy -- they only show the MUUFL-trained model's raw behaviour.")

    test_acc = None
    if split_masks is not None:
        tm = split_masks["test"]
        test_acc = float((labels[tm] == class_map[tm]).mean())
        print(f"Accuracy on the held-out TEST split ({int(tm.sum()):,d} px): "
              f"{test_acc:.3f}  (all-labeled number above includes training pixels)")

    present = sorted(set(gt.tolist()) | set(pred.tolist()))
    names = [c_names[c - 1] if 1 <= c <= len(c_names) else f"Class {c}" for c in present]
    report = classification_report(gt, pred, labels=present, target_names=names, zero_division=0, digits=3)
    print("\n" + "=" * 70 + f"\n PER-CLASS CLASSIFICATION REPORT ({kind.upper()} - all labeled pixels)\n" + "=" * 70)
    print(report)

    # Per-class test-split report
    if split_masks is not None:
        tm = split_masks["test"]
        gt_test, pred_test = labels[tm], class_map[tm]
        present_test = sorted(set(gt_test.tolist()) | set(pred_test.tolist()))
        names_test = [c_names[c - 1] if 1 <= c <= len(c_names) else f"Class {c}" for c in present_test]
        report_test = classification_report(gt_test, pred_test, labels=present_test, target_names=names_test,
                                            zero_division=0, digits=3)
        print("\n" + "=" * 70 + f"\n PER-CLASS CLASSIFICATION REPORT ({kind.upper()} - held-out TEST split only)\n" + "=" * 70)
        print(report_test)
        with open(os.path.join(run_out, "classification_report_test_split.txt"), "w", encoding="utf-8") as f:
            f.write(f"Test split: {int(tm.sum()):,d} pixels\nAccuracy: {test_acc:.4f}\n\n{report_test}\n")

    with open(os.path.join(run_out, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report + f"\nOverall accuracy: {acc:.4f}\n")

    conf_mat = confusion_matrix(gt, pred, labels=present)
    pct = conf_mat / np.where(conf_mat.sum(axis=1, keepdims=True) == 0, 1, conf_mat.sum(axis=1, keepdims=True)) * 100
    fig, ax = plt.subplots(figsize=(max(8, len(present) * 0.9), max(7, len(present) * 0.8)))
    im = ax.imshow(pct, interpolation="nearest", cmap="YlOrRd")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized %")
    for i in range(len(present)):
        for j in range(len(present)):
            if conf_mat[i, j] > 0:
                ax.text(j, i, f"{conf_mat[i, j]}\n({pct[i, j]:.1f}%)", ha="center", va="center", fontsize=6,
                        color="white" if pct[i, j] > 50 else "black", fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Predicted Class", fontweight="bold")
    ax.set_ylabel("True Class", fontweight="bold")
    ax.set_title(f"Confusion Matrix ({kind.upper()}, accuracy={acc:.3f})", fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(run_out, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved classification_report.txt and confusion_matrix.png")
    return acc, test_acc


# ===========================================================================
# 7b. CALIBRATION DIAGRAM
# ===========================================================================
def save_calibration_plot(run_out, labels, confidence_map, target_id, eval_mask, n_bins=10):
    """Reliability diagram + Expected Calibration Error (ECE).
    Shows whether the model's confidence values are well-calibrated."""
    plt = get_plt()
    y_true = (labels[eval_mask] == target_id).astype(np.float32)
    y_score = confidence_map[eval_mask]

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_acc, bin_conf, bin_count = [], [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_score >= lo) & (y_score < hi)
        if mask.sum() == 0:
            continue
        bin_acc.append(float(y_true[mask].mean()))
        bin_conf.append(float(y_score[mask].mean()))
        bin_count.append(int(mask.sum()))

    if not bin_acc:
        return None

    # ECE
    total = sum(bin_count)
    ece = sum(abs(a - c) * n / total for a, c, n in zip(bin_acc, bin_conf, bin_count))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar(bin_conf, bin_acc, width=0.08, alpha=0.7, color="#1E88E5", edgecolor="black", label="Actual accuracy")
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax1.set_xlabel("Mean predicted confidence")
    ax1.set_ylabel("Fraction of positives (accuracy)")
    ax1.set_title(f"Reliability Diagram (ECE={ece:.4f})", fontweight="bold")
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(alpha=0.3)

    ax2.bar(bin_conf, bin_count, width=0.08, alpha=0.7, color="#43A047", edgecolor="black")
    ax2.set_xlabel("Mean predicted confidence")
    ax2.set_ylabel("Number of pixels")
    ax2.set_title("Confidence Distribution", fontweight="bold")
    ax2.set_xlim(0, 1)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(run_out, "calibration_diagram.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved calibration_diagram.png (ECE={ece:.4f})")
    return ece


def save_ace_figures(ace_dir, rgb_image, ace_map, confidence_map, fusion_thr, ace_thr,
                     curves, metrics, split_name, sam_map=None, rx_map=None):
    plt = get_plt()
    from sklearn.metrics import roc_curve, precision_recall_curve
    from matplotlib.patches import Patch

    fus = confidence_map >= fusion_thr
    ace = ace_map >= ace_thr
    agree = np.zeros((*ace_map.shape, 3), dtype=np.float32) + 0.08
    agree[fus & ace] = (0.0, 0.9, 0.2)
    agree[fus & ~ace] = (0.95, 0.15, 0.15)
    agree[~fus & ace] = (0.2, 0.4, 1.0)
    n_both, n_f, n_a = int((fus & ace).sum()), int((fus & ~ace).sum()), int((~fus & ace).sum())

    panels = [
        ("Input scene", rgb_image, None, None),
        (f"ACE score map (HSI only)\nthr={ace_thr:.3f}", ace_map, "magma", (0, max(float(np.percentile(ace_map, 99.9)), 1e-3))),
        (f"Fusion confidence\nthr={fusion_thr}", confidence_map, "hot", (0, 1)),
        (f"Agreement: {n_both}px both, {n_f}px fusion, {n_a}px ACE", agree, None, None),
    ]
    if sam_map is not None:
        panels.append(("SAM similarity map", sam_map, "viridis", (0, 1)))
    if rx_map is not None:
        panels.append(("RX anomaly map (unsupervised)", rx_map, "plasma", (0, 1)))

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 5.5))
    for ax, (title, img, cmap, vlims) in zip(axes, panels):
        if cmap is None:
            ax.imshow(img)
            if "Agreement" in title:
                ax.legend(handles=[Patch(color=(0, .9, .2), label="Fusion + ACE"),
                                   Patch(color=(.95, .15, .15), label="Fusion only"),
                                   Patch(color=(.2, .4, 1), label="ACE only")], loc="lower right", fontsize=7)
        else:
            im = ax.imshow(img, cmap=cmap, vmin=vlims[0], vmax=vlims[1])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontweight="bold", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(os.path.join(ace_dir, "ace_vs_fusion_maps.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    if curves is not None and curves.get("ace") is not None:
        y = curves["y"]
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        detector_styles = [
            ("fusion", "Fusion", "#d62728"),
            ("ace", "ACE", "#1f77b4"),
            ("sam", "SAM", "#2ca02c"),
            ("rx", "RX Anomaly", "#ff7f0e"),
            ("ensemble", "Ensemble", "#9467bd"),
        ]
        for key, label, col in detector_styles:
            if curves.get(key) is None:
                continue
            fpr, tpr, _ = roc_curve(y, curves[key])
            axes[0].plot(fpr, tpr, color=col, linewidth=2, label=f"{label} (AUC={metrics[key]['auc']:.3f})")
            p, r, _ = precision_recall_curve(y, curves[key])
            axes[1].plot(r, p, color=col, linewidth=2, label=f"{label} (AP={metrics[key]['ap']:.3f})")
        axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
        axes[0].set_xlabel("False positive rate")
        axes[0].set_ylabel("Detection rate")
        axes[0].set_title(f"ROC -- {split_name} pixels", fontweight="bold")
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        axes[1].set_title(f"Precision-Recall -- {split_name} pixels", fontweight="bold")
        for a in axes:
            a.legend()
            a.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(ace_dir, "ace_vs_fusion_curves.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
    return {"both_px": n_both, "fusion_only_px": n_f, "ace_only_px": n_a}


# ===========================================================================
# 7c. BAND IMPORTANCE (top discriminative HSI bands)
# ===========================================================================
def save_band_importance_plot(out_dir, hsi_raw, confidence_map, wavelengths, top_k=20):
    """Identifies the HSI bands most correlated with target confidence."""
    plt = get_plt()
    h, w, b = hsi_raw.shape
    flat_hsi = hsi_raw.reshape(-1, b)
    flat_conf = confidence_map.flatten()

    # Pearson correlation between each band and the target confidence map
    correlations = np.zeros(b)
    for i in range(b):
        band_vals = flat_hsi[:, i]
        if band_vals.std() < 1e-8:
            continue
        correlations[i] = np.corrcoef(band_vals, flat_conf)[0, 1]
    correlations = np.nan_to_num(correlations)

    # Top-k most correlated bands
    top_idx = np.argsort(np.abs(correlations))[-top_k:][::-1]
    x_axis = wavelengths if wavelengths is not None else np.arange(b)
    x_label = "Wavelength (nm)" if wavelengths is not None else "Band Index"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    # Full correlation spectrum
    colors = ["#E53935" if c > 0 else "#1E88E5" for c in correlations]
    ax1.bar(x_axis, correlations, color=colors, alpha=0.7, width=x_axis[1] - x_axis[0] if len(x_axis) > 1 else 1)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Pearson correlation with target confidence")
    ax1.set_title("Band-by-Band Correlation with Target Detection", fontweight="bold")
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.grid(alpha=0.3)

    # Top-k bands
    top_labels = [f"{x_axis[i]:.0f}" if wavelengths is not None else f"Band {i}" for i in top_idx]
    top_vals = [correlations[i] for i in top_idx]
    top_colors = ["#E53935" if v > 0 else "#1E88E5" for v in top_vals]
    ax2.barh(range(len(top_idx)), [abs(v) for v in top_vals], color=top_colors, edgecolor="black", linewidth=0.5)
    ax2.set_yticks(range(len(top_idx)))
    ax2.set_yticklabels(top_labels, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel("|Correlation|")
    ax2.set_title(f"Top {top_k} Most Discriminative Bands", fontweight="bold")
    for i, v in enumerate(top_vals):
        ax2.text(abs(v) + 0.005, i, f"{v:+.3f}", va="center", fontsize=7)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "band_importance.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved band_importance.png")
    return {f"band_{i}": round(float(correlations[i]), 4) for i in top_idx}


# ===========================================================================
# 7d. THRESHOLD SWEEP (Precision-Recall operating curve)
# ===========================================================================
def save_threshold_sweep(out_dir, labels, confidence_map, target_id, eval_mask, otsu_thr=None):
    """Evaluates precision, recall, F1, and false alarm rate across confidence thresholds."""
    plt = get_plt()
    y_true = (labels[eval_mask] == target_id)
    n_pos = int(y_true.sum())
    n_neg = int((~y_true).sum())
    if n_pos == 0:
        return []
    scores = confidence_map[eval_mask]
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    if otsu_thr is not None and 0.05 < otsu_thr < 0.95:
        thresholds.append(otsu_thr)
    thresholds = sorted(list(set(thresholds)))

    rows = []
    for thr in thresholds:
        pred = scores >= thr
        tp = int(np.sum(y_true & pred))
        fp = int(np.sum((~y_true) & pred))
        fn = int(np.sum(y_true & (~pred)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        far = fp / n_neg if n_neg > 0 else 0.0
        rows.append({
            "threshold": round(float(thr), 3),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
            "tp": tp, "fp": fp, "fn": fn,
            "false_alarm_rate": round(float(far), 5),
            "is_otsu": abs(thr - otsu_thr) < 1e-4 if otsu_thr else False
        })
    with open(os.path.join(out_dir, "threshold_sweep.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    thrs = [r["threshold"] for r in rows]
    precs = [r["precision"] for r in rows]
    recs = [r["recall"] for r in rows]
    f1s = [r["f1"] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thrs, precs, "o-", color="#1E88E5", label="Precision", linewidth=2)
    ax.plot(thrs, recs, "s-", color="#43A047", label="Recall", linewidth=2)
    ax.plot(thrs, f1s, "^-", color="#E53935", label="F1 Score", linewidth=2.5)
    if otsu_thr is not None:
        ax.axvline(otsu_thr, color="#8E24AA", linestyle="--", alpha=0.8, label=f"Otsu Optimal ({otsu_thr:.3f})")
    ax.set_xlabel("Confidence Threshold", fontweight="bold")
    ax.set_ylabel("Metric Value", fontweight="bold")
    ax.set_title("Performance Trade-Off Across Confidence Thresholds", fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "threshold_sweep.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved threshold_sweep.csv and threshold_sweep.png")
    return rows


# ===========================================================================
# 8. HTML INTERACTIVE INTELLIGENCE DASHBOARD
# ===========================================================================
def _img_to_base64(path):
    """Read an image file and return a base64-encoded data URI."""
    try:
        with open(path, "rb") as f:
            data = f.read()
        ext = os.path.splitext(path)[1].lower().lstrip(".")
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(ext, "image/png")
        return f"data:{mime};base64,{base64.b64encode(data).decode()}"
    except Exception:
        return ""


def save_html_report(run_out, results, metrics_dict, summary, image_files, sweep_rows=None, test_report_text=None):
    """Generate a self-contained, publication-grade interactive HTML dashboard."""
    # Build categorized image dictionaries
    cat_maps = [("Detection Visualization", "detection_visualization.png"),
                ("ACE vs Fusion Maps", os.path.join("ace", "ace_vs_fusion_maps.png")),
                ("Full Scene Boxes", os.path.join("targets", "full_scene_boxes.png")),
                ("ROC & Precision-Recall Curves", os.path.join("ace", "ace_vs_fusion_curves.png"))]
    cat_eval = [("Confusion Matrix", "confusion_matrix.png"),
                ("Calibration Diagram (ECE)", "calibration_diagram.png"),
                ("Confidence Threshold Sweep", "threshold_sweep.png")]
    cat_spectral = [("Spectral Signatures of Detected Targets", os.path.join("targets", "spectral_signatures.png")),
                    ("Target vs Background Contrast", os.path.join("targets", "spectral_target_vs_background.png")),
                    ("Discriminative Bands Importance", "band_importance.png"),
                    ("LiDAR 3D Topographic Profiles", os.path.join("targets", "target_lidar_profiles.png"))]
    cat_yolo = [("Super-Resolution + YOLOv8 Analysis", os.path.join("targets", "target_before_after.png")),
                ("Target Optical Crops", os.path.join("targets", "target_crops.png"))]

    def render_gallery(pairs):
        cards = ""
        for title, fname in pairs:
            path = os.path.join(run_out, fname)
            if os.path.exists(path):
                b64 = _img_to_base64(path)
                if b64:
                    cards += f"""
                    <div class="card img-card" onclick="openLightbox('{b64}', '{title}')">
                      <div class="card-header">
                        <h4>{title}</h4>
                        <span class="zoom-badge">🔍 Zoom</span>
                      </div>
                      <div class="img-container">
                        <img src="{b64}" alt="{title}" loading="lazy">
                      </div>
                    </div>"""
        return cards

    gallery_maps_html = render_gallery(cat_maps)
    gallery_eval_html = render_gallery(cat_eval)
    gallery_spectral_html = render_gallery(cat_spectral)
    gallery_yolo_html = render_gallery(cat_yolo)

    # Build target table rows
    target_rows_html = ""
    for o in results:
        tier = o["quality_tier"]
        tier_badge = {
            "HIGH CONFIDENCE": '<span class="badge badge-success">HIGH CONFIDENCE</span>',
            "MEDIUM CONFIDENCE": '<span class="badge badge-warning">MEDIUM CONFIDENCE</span>',
            "LOW CONFIDENCE": '<span class="badge badge-orange">LOW CONFIDENCE</span>',
            "LIKELY FALSE POSITIVE": '<span class="badge badge-danger">FALSE POSITIVE</span>',
        }.get(tier, '<span class="badge">UNKNOWN</span>')

        score = o["quality_score"]
        score_bar = f"""
        <div class="score-container">
          <span class="score-val">{score:.1f}</span>
          <div class="progress-track"><div class="progress-bar" style="width:{score}%;"></div></div>
        </div>"""

        ace_v = o.get("ace_verdict", "N/A")
        ace_badge = '<span class="badge badge-confirmed">ACE CONFIRMED</span>' if "CONFIRMED" in ace_v else f'<span class="text-muted">{ace_v}</span>'

        rx_v = o.get("rx_verdict", "N/A")
        rx_badge = '<span class="badge badge-purple">RX ANOMALOUS</span>' if "ANOMALOUS" in rx_v else f'<span class="text-muted">{o.get("rx_mean", "N/A")}</span>'

        lidar_h = f"{o['lidar_relative_height_m']:+.2f}m" if "lidar_relative_height_m" in o else "N/A"
        lidar_elev = f"{o['lidar_elevation_m']:.2f}m" if "lidar_elevation_m" in o else "N/A"
        lidar_rough = f"±{o['lidar_roughness_std_m']:.2f}m" if "lidar_roughness_std_m" in o else ""

        color_dot = ""
        c_low = o["dominant_color"].lower()
        dot_color = "#e11d48" if "red" in c_low else "#2563eb" if "blue" in c_low else "#16a34a" if "green" in c_low else "#eab308" if "yellow" in c_low else "#64748b"
        color_dot = f'<span class="color-dot" style="background:{dot_color}"></span>'

        target_rows_html += f"""
        <tr class="target-row" data-tier="{tier}" data-confirmed="{'true' if 'CONFIRMED' in ace_v else 'false'}">
          <td class="bold">#{o['id']}</td>
          <td><div class="flex-align">{color_dot}<strong>{o['name']}</strong></div></td>
          <td>{score_bar}</td>
          <td>{tier_badge}</td>
          <td>{o['size_pixels']}px <span class="text-muted">({o['size_category']})</span></td>
          <td>{o['surrounding_context']} <span class="text-muted">({o['surrounding_pct']}%)</span></td>
          <td><span class="mono">{o['mean_confidence']:.3f}</span> <span class="text-muted">/ {o['max_confidence']:.3f}</span></td>
          <td><span class="mono bold text-cyan">{o.get('spectral_contrast', 0):.2f}</span></td>
          <td><strong>{lidar_h}</strong> <span class="text-muted small">({lidar_elev} {lidar_rough})</span></td>
          <td>{ace_badge}</td>
          <td>{rx_badge}</td>
        </tr>"""

    # Multi-detector benchmark table
    detector_table_html = ""
    det_names = [
        ("fusion", "Multimodal Fusion (HSIMAE + PTv3)", "Supervised Deep Learning"),
        ("ace", "ACE Detector (Multi-Scale Regularization)", "HSI Subspace Projection"),
        ("sam", "SAM (Spectral Angle Mapper)", "Angular Spectral Metric"),
        ("rx", "Reed-Xiaoli (RX) Anomaly Detector", "Unsupervised Background Whitening"),
        ("ensemble", "Ensemble Consensus (Fusion + ACE + SAM)", "Multi-Detector Fusion"),
    ]
    for key, name, tag in det_names:
        m = metrics_dict.get(key)
        if m:
            auc = f"{m.get('auc', 0):.4f}"
            ap = f"{m.get('ap', 0):.4f}"
            pd1 = f"{m.get('pd_at_1pct_fpr', 0):.4f}"
            pr = f"{m.get('precision', 0):.4f}"
            rc = f"{m.get('recall', 0):.4f}"
            f1 = f"{m.get('f1', 0):.4f}"
            thr = f"{m.get('threshold', 0):.4f}" if "threshold" in m else "0.5000 (argmax)"
            detector_table_html += f"""
            <tr>
              <td><strong>{name}</strong><br><span class="text-muted small">{tag}</span></td>
              <td><span class="badge badge-cyan">{auc}</span></td>
              <td>{ap}</td>
              <td>{pd1}</td>
              <td>{pr}</td>
              <td>{rc}</td>
              <td><span class="badge badge-success bold">{f1}</span></td>
              <td class="text-muted mono small">{thr}</td>
            </tr>"""

    # Threshold sweep rows
    sweep_table_html = ""
    if sweep_rows:
        for r in sweep_rows:
            is_otsu = r.get("is_otsu", False)
            row_cls = ' class="highlight-row"' if is_otsu else ""
            otsu_tag = ' <span class="badge badge-purple small">OTSU OPTIMAL</span>' if is_otsu else ""
            sweep_table_html += f"""
            <tr{row_cls}>
              <td class="mono"><strong>{r['threshold']:.3f}</strong>{otsu_tag}</td>
              <td>{r['precision']:.4f}</td>
              <td>{r['recall']:.4f}</td>
              <td><strong class="text-cyan">{r['f1']:.4f}</strong></td>
              <td>{r['tp']}</td>
              <td>{r['fp']}</td>
              <td>{r['false_alarm_rate']:.5f}</td>
            </tr>"""

    # Test report text if available
    test_report_html = ""
    if test_report_text:
        test_report_html = f'<div class="code-box"><pre>{test_report_text}</pre></div>'
    else:
        # Check if classification_report_test_split.txt exists
        cr_path = os.path.join(run_out, "classification_report_test_split.txt")
        if os.path.exists(cr_path):
            try:
                with open(cr_path, "r", encoding="utf-8") as f:
                    test_report_html = f'<div class="code-box"><pre>{f.read()}</pre></div>'
            except Exception:
                pass

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Target Detection & Analysis Dossier</title>
<style>
  :root {{
    --bg-base: #090d16;
    --bg-surface: #111827;
    --bg-card: #162032;
    --bg-hover: #1f2d44;
    --border-color: #24344d;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --accent-cyan: #38bdf8;
    --accent-emerald: #10b981;
    --accent-amber: #f59e0b;
    --accent-rose: #f43f5e;
    --accent-indigo: #6366f1;
    --accent-purple: #a855f7;
    --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background-color: var(--bg-base);
    color: var(--text-primary);
    font-family: var(--font-sans);
    line-height: 1.5;
    padding: 24px;
  }}

  /* Header */
  .header {{
    background: linear-gradient(135deg, #111827 0%, #172554 100%);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    padding: 28px;
    margin-bottom: 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 16px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
  }}
  .header-left h1 {{
    font-size: 26px;
    font-weight: 800;
    letter-spacing: -0.5px;
    display: flex;
    align-items: center;
    gap: 12px;
  }}
  .header-left p {{
    color: var(--text-secondary);
    font-size: 14px;
    margin-top: 6px;
  }}
  .meta-pills {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 12px;
  }}
  .pill {{
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.12);
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    color: var(--text-secondary);
  }}
  .pill strong {{ color: var(--text-primary); }}
  .header-actions {{
    display: flex;
    gap: 10px;
  }}
  .btn {{
    background: var(--bg-card);
    color: var(--text-primary);
    border: 1px solid var(--border-color);
    padding: 8px 16px;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s ease;
    text-decoration: none;
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }}
  .btn:hover {{
    background: var(--bg-hover);
    border-color: var(--accent-cyan);
    color: var(--accent-cyan);
  }}
  .btn-primary {{
    background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
    border: none;
  }}
  .btn-primary:hover {{
    background: linear-gradient(135deg, #38bdf8 0%, #0284c7 100%);
    color: white;
  }}

  /* Stat Cards */
  .kpi-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
  }}
  .kpi-card {{
    background: var(--bg-surface);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 18px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
  }}
  .kpi-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--card-accent, var(--accent-cyan));
  }}
  .kpi-title {{
    color: var(--text-secondary);
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }}
  .kpi-value {{
    font-size: 28px;
    font-weight: 800;
    color: var(--text-primary);
    margin: 8px 0 4px;
    letter-spacing: -0.5px;
  }}
  .kpi-subtext {{
    color: var(--text-muted);
    font-size: 12px;
  }}

  /* Tabs */
  .tabs-nav {{
    display: flex;
    gap: 8px;
    border-bottom: 1px solid var(--border-color);
    margin-bottom: 24px;
    overflow-x: auto;
    padding-bottom: 8px;
  }}
  .tab-btn {{
    background: transparent;
    color: var(--text-secondary);
    border: none;
    padding: 10px 18px;
    font-size: 14px;
    font-weight: 600;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s ease;
    white-space: nowrap;
  }}
  .tab-btn:hover {{
    color: var(--text-primary);
    background: var(--bg-surface);
  }}
  .tab-btn.active {{
    background: var(--accent-cyan);
    color: #0b0f19;
    font-weight: 700;
  }}
  .tab-pane {{
    display: none;
  }}
  .tab-pane.active {{
    display: block;
    animation: fadeIn 0.25s ease-in-out;
  }}
  @keyframes fadeIn {{
    from {{ opacity: 0; transform: translateY(6px); }}
    to {{ opacity: 1; transform: translateY(0); }}
  }}

  /* Cards & Sections */
  .section {{
    background: var(--bg-surface);
    border: 1px solid var(--border-color);
    border-radius: 14px;
    padding: 24px;
    margin-bottom: 24px;
  }}
  .section-title {{
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
  }}

  /* Table */
  .table-responsive {{
    overflow-x: auto;
    border-radius: 8px;
    border: 1px solid var(--border-color);
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    background: var(--bg-surface);
    font-size: 13px;
    text-align: left;
  }}
  th, td {{
    padding: 12px 14px;
    border-bottom: 1px solid var(--border-color);
  }}
  th {{
    background: #151f30;
    color: var(--text-secondary);
    font-weight: 700;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }}
  tr:hover {{
    background: var(--bg-card);
  }}
  .highlight-row {{
    background: rgba(168, 85, 247, 0.15) !important;
  }}

  /* Badges & Pills */
  .badge {{
    display: inline-block;
    padding: 3px 8px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.3px;
  }}
  .badge-success {{ background: rgba(16, 185, 129, 0.18); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }}
  .badge-warning {{ background: rgba(245, 158, 11, 0.18); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }}
  .badge-orange {{ background: rgba(249, 115, 22, 0.18); color: #fb923c; border: 1px solid rgba(249, 115, 22, 0.3); }}
  .badge-danger {{ background: rgba(244, 63, 94, 0.18); color: #fb7185; border: 1px solid rgba(244, 63, 94, 0.3); }}
  .badge-cyan {{ background: rgba(56, 189, 248, 0.18); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); }}
  .badge-purple {{ background: rgba(168, 85, 247, 0.18); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.3); }}
  .badge-confirmed {{ background: rgba(16, 185, 129, 0.25); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.5); font-weight: 800; }}

  /* Score progress bar */
  .score-container {{
    display: flex;
    align-items: center;
    gap: 8px;
  }}
  .score-val {{
    font-weight: 800;
    font-size: 13px;
    min-width: 32px;
  }}
  .progress-track {{
    flex: 1;
    height: 6px;
    background: #1f2d44;
    border-radius: 3px;
    overflow: hidden;
    min-width: 60px;
  }}
  .progress-bar {{
    height: 100%;
    background: linear-gradient(90deg, #0284c7, #10b981);
    border-radius: 3px;
  }}

  /* Filter toolbar */
  .toolbar {{
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    margin-bottom: 16px;
  }}
  .search-box {{
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    padding: 8px 14px;
    border-radius: 8px;
    color: var(--text-primary);
    font-size: 13px;
    width: 280px;
  }}
  .filter-chips {{
    display: flex;
    gap: 6px;
  }}
  .chip {{
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 12px;
    cursor: pointer;
    color: var(--text-secondary);
  }}
  .chip.active {{
    background: var(--border-color);
    color: var(--accent-cyan);
    border-color: var(--accent-cyan);
  }}

  /* Image gallery grid */
  .gallery-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(460px, 1fr));
    gap: 20px;
  }}
  .img-card {{
    background: var(--bg-surface);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 16px;
    cursor: pointer;
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
  }}
  .img-card:hover {{
    transform: translateY(-2px);
    border-color: var(--accent-cyan);
    box-shadow: 0 6px 20px rgba(56, 189, 248, 0.15);
  }}
  .img-card .card-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
  }}
  .img-card h4 {{
    font-size: 14px;
    font-weight: 700;
  }}
  .zoom-badge {{
    font-size: 11px;
    color: var(--accent-cyan);
    background: rgba(56, 189, 248, 0.1);
    padding: 2px 6px;
    border-radius: 4px;
  }}
  .img-container {{
    border-radius: 8px;
    overflow: hidden;
    background: #000;
    display: flex;
    justify-content: center;
  }}
  .img-container img {{
    max-width: 100%;
    height: auto;
    display: block;
  }}

  /* Code box */
  .code-box {{
    background: #0b0f19;
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 16px;
    font-family: Consolas, Monaco, "Courier New", monospace;
    font-size: 12px;
    overflow-x: auto;
    color: #e2e8f0;
    max-height: 400px;
  }}

  /* Color dot */
  .color-dot {{
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    margin-right: 6px;
  }}
  .flex-align {{
    display: flex;
    align-items: center;
  }}

  /* Utility */
  .bold {{ font-weight: 700; }}
  .mono {{ font-family: Consolas, monospace; }}
  .text-cyan {{ color: var(--accent-cyan); }}
  .text-muted {{ color: var(--text-muted); }}
  .small {{ font-size: 11px; }}

  /* Lightbox Modal */
  .lightbox {{
    display: none;
    position: fixed;
    z-index: 1000;
    left: 0; top: 0; width: 100%; height: 100%;
    background: rgba(4, 7, 13, 0.92);
    backdrop-filter: blur(8px);
    align-items: center;
    justify-content: center;
    flex-direction: column;
    padding: 24px;
  }}
  .lightbox.active {{
    display: flex;
  }}
  .lightbox-content {{
    max-width: 90vw;
    max-height: 85vh;
    border-radius: 8px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.8);
    border: 1px solid var(--border-color);
  }}
  .lightbox-caption {{
    color: var(--text-primary);
    font-size: 16px;
    font-weight: 700;
    margin-top: 14px;
  }}
  .lightbox-close {{
    position: absolute;
    top: 24px; right: 28px;
    font-size: 32px;
    color: var(--text-secondary);
    cursor: pointer;
  }}
  .lightbox-close:hover {{
    color: var(--text-primary);
  }}

  /* Footer */
  .footer {{
    margin-top: 40px;
    border-top: 1px solid var(--border-color);
    padding-top: 20px;
    text-align: center;
    color: var(--text-muted);
    font-size: 12px;
  }}
</style>
</head>
<body>

<!-- Header -->
<header class="header">
  <div class="header-left">
    <h1>🎯 Multimodal Target Detection & Intelligence Dossier</h1>
    <p>Automated Hyperspectral + LiDAR Target Exploitation, Verification & Spatial Analytics</p>
    <div class="meta-pills">
      <span class="pill">Dataset: <strong>{summary.get('dataset_kind', 'N/A').upper()}</strong></span>
      <span class="pill">Resolution: <strong>{summary.get('scene_shape', 'N/A')} ({summary.get('total_pixels', 0):,d} px)</strong></span>
      <span class="pill">Timestamp: <strong>{summary.get('timestamp', '')[:19]}</strong></span>
      <span class="pill" style="border-color: var(--accent-emerald); color: var(--accent-emerald);">● Pipeline Execution Complete</span>
    </div>
  </div>
  <div class="header-actions">
    <button class="btn" onclick="window.print()">🖨️ Print Dossier</button>
    <a class="btn" href="targets/targets_report.csv" download="targets.csv">📥 Targets CSV</a>
    <a class="btn btn-primary" href="run_summary.json" download="summary.json">📄 Summary JSON</a>
  </div>
</header>

<!-- Key Performance Indicators (KPIs) -->
<div class="kpi-grid">
  <div class="kpi-card" style="--card-accent: var(--accent-cyan);">
    <div class="kpi-title">Overall Accuracy</div>
    <div class="kpi-value">{summary.get('accuracy_all_labeled', 0)*100:.1f}%</div>
    <div class="kpi-subtext">All Labeled Pixels in Scene</div>
  </div>
  <div class="kpi-card" style="--card-accent: var(--accent-emerald);">
    <div class="kpi-title">Test Split Accuracy</div>
    <div class="kpi-value">{summary.get('accuracy_test_split', 0)*100:.1f}%</div>
    <div class="kpi-subtext">Held-out Evaluation Split</div>
  </div>
  <div class="kpi-card" style="--card-accent: var(--accent-indigo);">
    <div class="kpi-title">Ensemble F1 / AUC</div>
    <div class="kpi-value">{summary.get('ensemble_f1', 0):.3f}</div>
    <div class="kpi-subtext">AUC: {summary.get('ensemble_auc', 0):.5f}</div>
  </div>
  <div class="kpi-card" style="--card-accent: var(--accent-amber);">
    <div class="kpi-title">Targets Detected</div>
    <div class="kpi-value">{summary.get('n_targets_detected', len(results))}</div>
    <div class="kpi-subtext">{summary.get('n_high_confidence', 0)} High / {summary.get('n_medium_confidence', 0)} Med Tier</div>
  </div>
  <div class="kpi-card" style="--card-accent: var(--accent-rose);">
    <div class="kpi-title">Detection Threshold</div>
    <div class="kpi-value">{summary.get('confidence_threshold', 0.5):.2f}</div>
    <div class="kpi-subtext">Otsu Optimal: {summary.get('adaptive_threshold_otsu', 0.5):.3f}</div>
  </div>
  <div class="kpi-card" style="--card-accent: var(--accent-purple);">
    <div class="kpi-title">Multi-Detector Suite</div>
    <div class="kpi-value">4 Systems</div>
    <div class="kpi-subtext">Fusion + ACE + SAM + RX</div>
  </div>
</div>

<!-- Tabs Navigation -->
<div class="tabs-nav">
  <button class="tab-btn active" onclick="switchTab('tab-targets')">🎯 Target Catalog ({len(results)})</button>
  <button class="tab-btn" onclick="switchTab('tab-maps')">🗺️ Detection Maps & Spatial Verification</button>
  <button class="tab-btn" onclick="switchTab('tab-benchmarks')">📊 Benchmark & Calibration Metrics</button>
  <button class="tab-btn" onclick="switchTab('tab-spectral')">🔬 Spectral Signatures & LiDAR Topography</button>
  <button class="tab-btn" onclick="switchTab('tab-yolo')">🤖 Super-Resolution & YOLOv8</button>
</div>

<!-- Tab 1: Targets Catalog -->
<div id="tab-targets" class="tab-pane active">
  <div class="section">
    <div class="toolbar">
      <input type="text" id="targetSearch" class="search-box" placeholder="🔍 Search targets by name, context, color..." onkeyup="filterTargets()">
      <div class="filter-chips">
        <button class="chip active" onclick="filterTier('ALL')">All Targets</button>
        <button class="chip" onclick="filterTier('HIGH CONFIDENCE')">High Confidence</button>
        <button class="chip" onclick="filterTier('MEDIUM CONFIDENCE')">Medium</button>
        <button class="chip" onclick="filterTier('CONFIRMED')">ACE Confirmed</button>
      </div>
    </div>

    <div class="table-responsive">
      <table id="targetsTable">
        <thead>
          <tr>
            <th>ID</th>
            <th>Target Name & Color</th>
            <th style="min-width:130px;">Quality Score</th>
            <th>Tier</th>
            <th>Size</th>
            <th>Context (Surroundings)</th>
            <th>Fusion Conf</th>
            <th>Spectral Contrast</th>
            <th>LiDAR Rel H (vs Ground)</th>
            <th>ACE Verdict</th>
            <th>RX Anomaly</th>
          </tr>
        </thead>
        <tbody>
          {target_rows_html}
        </tbody>
      </table>
    </div>
  </div>
</div>

<!-- Tab 2: Detection Maps & Spatial Verification -->
<div id="tab-maps" class="tab-pane">
  <div class="gallery-grid">
    {gallery_maps_html}
  </div>
</div>

<!-- Tab 3: Benchmark & Calibration Metrics -->
<div id="tab-benchmarks" class="tab-pane">
  <div class="section">
    <h3 class="section-title">🏆 Multi-Detector Performance Benchmark (Held-out Test Split)</h3>
    <div class="table-responsive" style="margin-bottom:20px;">
      <table>
        <thead>
          <tr>
            <th>Detector / Model Architecture</th>
            <th>AUC</th>
            <th>Avg Precision (AP)</th>
            <th>PD @ 1% FPR</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1 Score</th>
            <th>Operating Threshold</th>
          </tr>
        </thead>
        <tbody>
          {detector_table_html}
        </tbody>
      </table>
    </div>
  </div>

  <div class="gallery-grid" style="margin-bottom: 24px;">
    {gallery_eval_html}
  </div>

  {f'<div class="section"><h3 class="section-title">📋 Held-out Test Split Per-Class Classification Report</h3>{test_report_html}</div>' if test_report_html else ''}

  {f'''<div class="section">
    <h3 class="section-title">📈 Confidence Threshold Sweep & Operating Trade-Off</h3>
    <div class="table-responsive">
      <table>
        <thead>
          <tr>
            <th>Threshold</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1 Score</th>
            <th>True Positives (TP)</th>
            <th>False Positives (FP)</th>
            <th>False Alarm Rate</th>
          </tr>
        </thead>
        <tbody>
          {sweep_table_html}
        </tbody>
      </table>
    </div>
  </div>''' if sweep_table_html else ''}
</div>

<!-- Tab 4: Spectral Signatures & LiDAR Topography -->
<div id="tab-spectral" class="tab-pane">
  <div class="gallery-grid">
    {gallery_spectral_html}
  </div>
</div>

<!-- Tab 5: Super-Resolution & YOLOv8 -->
<div id="tab-yolo" class="tab-pane">
  <div class="gallery-grid">
    {gallery_yolo_html}
  </div>
</div>

<!-- Lightbox Modal -->
<div id="lightboxModal" class="lightbox" onclick="closeLightbox(event)">
  <span class="lightbox-close" onclick="closeLightboxDirect()">&times;</span>
  <img id="lightboxImg" class="lightbox-content" src="" alt="Full view">
  <div id="lightboxCaption" class="lightbox-caption"></div>
</div>

<!-- Footer -->
<footer class="footer">
  Multimodal Hyperspectral & LiDAR Target Exploitation System &bull; Run Folder: {run_out} &bull; Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
</footer>

<script>
  function switchTab(tabId) {{
    document.querySelectorAll('.tab-pane').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
    document.getElementById(tabId).classList.add('active');
    event.currentTarget.classList.add('active');
  }}

  function filterTargets() {{
    const query = document.getElementById('targetSearch').value.toLowerCase();
    const rows = document.querySelectorAll('#targetsTable tbody tr');
    rows.forEach(row => {{
      const text = row.innerText.toLowerCase();
      row.style.display = text.includes(query) ? '' : 'none';
    }});
  }}

  function filterTier(tier) {{
    document.querySelectorAll('.filter-chips .chip').forEach(c => c.classList.remove('active'));
    event.currentTarget.classList.add('active');
    const rows = document.querySelectorAll('#targetsTable tbody tr');
    rows.forEach(row => {{
      if (tier === 'ALL') {{
        row.style.display = '';
      }} else if (tier === 'CONFIRMED') {{
        row.style.display = row.getAttribute('data-confirmed') === 'true' ? '' : 'none';
      }} else {{
        row.style.display = row.getAttribute('data-tier') === tier ? '' : 'none';
      }}
    }});
  }}

  function openLightbox(src, title) {{
    document.getElementById('lightboxImg').src = src;
    document.getElementById('lightboxCaption').innerText = title;
    document.getElementById('lightboxModal').classList.add('active');
  }}

  function closeLightbox(e) {{
    if (e.target.id === 'lightboxModal') {{
      closeLightboxDirect();
    }}
  }}

  function closeLightboxDirect() {{
    document.getElementById('lightboxModal').classList.remove('active');
  }}

  document.addEventListener('keydown', function(e) {{
    if (e.key === 'Escape') closeLightboxDirect();
  }});
</script>

</body>
</html>"""

    report_path = os.path.join(run_out, "report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Saved executive interactive HTML dashboard: {report_path}")


# ===========================================================================
# 9. MAIN
# ===========================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Fusion inference + ACE/SAM baseline + target identification")
    B = argparse.BooleanOptionalAction
    # --- dataset selection ---
    p.add_argument("--dataset", choices=["interactive", "muufl", "trento", "custom"], default="interactive",
                   help="Dataset selection: 'interactive' prompts the user in terminal, 'muufl' for MUUFL Gulfport, 'trento' for Trento (full size 166x600, no cropping), or 'custom'")
    # --- data (defaults = your MUUFL layout; run from the NASIM11 folder) ---
    p.add_argument("--data", default=r"MUUFLGulfport-master\MUUFLGulfportDataCollection\muufl_gulfport_campus_w_lidar_1.mat")
    p.add_argument("--hsi_file", default=None)
    p.add_argument("--lidar_file", default=None)
    p.add_argument("--labels", default=None,
                   help="Ground truth labels file (defaults to dataset standard if not specified)")
    p.add_argument("--dataset_kind", choices=["auto", "muufl", "trento", "other"], default="auto",
                   help="'auto' is resolved from --dataset or inputs")
    p.add_argument("--target_label_id", type=int, default=None,
                   help="1-indexed label id of the target class (auto: 11 for MUUFL, 2 for Trento)")
    # --- model ---
    p.add_argument("--hsimae_repo", default=r"pretrained_models\HSIMAE")
    p.add_argument("--hsi_checkpoint", default=r"runs\hsi_only_v2\best_model.pt")
    p.add_argument("--utonia_repo", default=r"pretrained_models\Utonia")
    p.add_argument("--lidar_pretrained", default=r"pretrained_models\Utonia\checkpoints\utonia.pth")
    p.add_argument("--lidar_checkpoint", default=r"runs\lidar_only\best_model.pt")
    p.add_argument("--fusion_checkpoint", default=r"runs\fusion\best_model.pt")
    p.add_argument("--use_normal", default=True, action=B)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--reuse_run", default=None,
                   help="Folder of an earlier run: loads its class_map.npy / target_confidence_map.npy and skips the model")
    # --- NEW: inference improvements ---
    p.add_argument("--tta", default=True, action=B,
                   help="Enable Test-Time Augmentation (hflip, vflip, rot180) for more robust predictions")
    p.add_argument("--mixed_precision", default=True, action=B,
                   help="Enable fp16 mixed-precision inference on GPU for ~2x speedup")
    p.add_argument("--spatial_smooth_sigma", type=float, default=1.0,
                   help="Gaussian sigma for spatial confidence smoothing (0 to disable)")
    # --- output ---
    p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "OUTPUT"),
                   help="Folder to save run results, figures, and reports")
    # --- target identification ---
    p.add_argument("--identify", default=True, action=B)
    p.add_argument("--confidence_threshold", type=float, default=0.5)
    p.add_argument("--adaptive_threshold", default=True, action=B,
                   help="Also compute and report Otsu-optimal threshold")
    p.add_argument("--min_size", type=int, default=2)
    p.add_argument("--use_morphology", action="store_true")
    p.add_argument("--crop_padding", type=int, default=10)
    p.add_argument("--max_visual_targets", type=int, default=12)
    p.add_argument("--merge_radius", type=int, default=8,
                   help="Merge target fragments within this pixel radius (0 to disable)")
    p.add_argument("--yolo", default=True, action=B, help="Optional YOLOv8 enhancement panel (needs opencv-python + ultralytics)")
    # --- ACE ---
    p.add_argument("--ace", default=True, action=B)
    p.add_argument("--ace_multiscale", default=True, action=B,
                   help="Use multi-scale ACE (ensemble of regularization levels)")
    p.add_argument("--ace_target_source", choices=["auto", "labels", "model", "file"], default="auto")
    p.add_argument("--ace_target_file", default=None, help=".mat/.npy with the target spectrum (e.g. MUUFL's tgt_img_spectra.mat)")
    p.add_argument("--ace_threshold", type=float, default=None)
    p.add_argument("--ace_reg", type=float, default=1e-3, help="Covariance shrinkage (relative to mean eigenvalue)")
    p.add_argument("--eval_split", choices=["auto", "test", "all"], default="auto")
    # --- NEW: SAM + RX + ensemble ---
    p.add_argument("--sam", default=True, action=B,
                   help="Enable Spectral Angle Mapper detector alongside ACE")
    p.add_argument("--rx", default=True, action=B,
                   help="Enable Reed-Xiaoli (RX) unsupervised anomaly detector")
    p.add_argument("--ensemble", default=True, action=B,
                   help="Compute ensemble score (fusion + ACE + SAM)")
    p.add_argument("--ensemble_weights", type=float, nargs=3, default=[0.5, 0.3, 0.2],
                   metavar=("W_FUSION", "W_ACE", "W_SAM"),
                   help="Weights for ensemble score [fusion, ace, sam]")
    # --- NEW: reports & profiling ---
    p.add_argument("--threshold_sweep", default=True, action=B,
                   help="Sweep confidence thresholds and generate precision-recall operating curve")
    p.add_argument("--lidar_profiles", default=True, action=B,
                   help="Generate 2D and cross-sectional LiDAR elevation profiles for targets")
    p.add_argument("--html_report", default=True, action=B,
                   help="Generate a standalone HTML summary report")
    p.add_argument("--band_importance", default=True, action=B,
                   help="Plot band-by-band correlation with target confidence")
    return p.parse_args()


def main():
    args = parse_args()
    t_start = time.time()
    timings = {}

    run_out = os.path.join(args.out, f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_out, exist_ok=True)
    print(f"This run's output folder: {run_out}")

    # Determine dataset selection
    custom = bool(args.hsi_file and args.lidar_file)
    if custom:
        selected_dataset = "custom"
    elif args.dataset == "interactive":
        selected_dataset = prompt_dataset_selection(default_choice="muufl")
    else:
        selected_dataset = args.dataset

    if selected_dataset == "trento":
        kind = "trento"
        active_class_names = TRENTO_CLASS_NAMES
        active_class_colors = TRENTO_CLASS_COLORS_HEX
        active_target_name = "Buildings"
        if args.target_label_id is None:
            args.target_label_id = TRENTO_TARGET_LABEL
        fusion_target_idx = 7 if args.target_label_id == 2 else TARGET_CLASS_IDX
    elif selected_dataset == "muufl":
        kind = "muufl"
        active_class_names = MUUFL_CLASS_NAMES
        active_class_colors = MUUFL_CLASS_COLORS_HEX
        active_target_name = "Cloth Panels"
        if args.target_label_id is None:
            args.target_label_id = MUUFL_TARGET_LABEL
        fusion_target_idx = TARGET_CLASS_IDX
    else:
        kind = args.dataset_kind if args.dataset_kind != "auto" else "other"
        active_class_names = MUUFL_CLASS_NAMES
        active_class_colors = MUUFL_CLASS_COLORS_HEX
        active_target_name = "Target"
        fusion_target_idx = TARGET_CLASS_IDX

    print(f"Dataset selected: {selected_dataset.upper()} (kind: {kind}, target: {active_target_name} [ID {args.target_label_id}])")

    # ---------------- load scene ----------------
    t_load = time.time()
    if selected_dataset == "trento":
        trento_hsi = args.hsi_file or r"Trento-main\Italy_hsi.mat"
        trento_lidar = args.lidar_file or r"Trento-main\Italy_lidar.mat"
        trento_labels = args.labels or r"Trento-main\allgrd.mat"
        hsi_raw, lidar_raw, labels, rgb_raw, wavelengths = load_trento_scene(trento_hsi, trento_lidar, trento_labels)
    elif selected_dataset == "custom":
        hsi_raw, lidar_raw, labels, rgb_raw, wavelengths = load_custom_scene(args.hsi_file, args.lidar_file, args.labels)
    elif selected_dataset == "muufl" or args.data:
        muufl_labels = args.labels or r"MUUFLGulfport-master\MUUFLGulfportSceneLabels\muufl_gulfport_campus_1_hsi_220_label.mat"
        hsi_raw, lidar_raw, labels, rgb_raw, wavelengths = load_raw_scene(args.data, muufl_labels)
    else:
        raise ValueError("Invalid dataset selection.")

    print(f"Scene shape: HSI {hsi_raw.shape}, LiDAR {lidar_raw.shape}")
    h, w = hsi_raw.shape[:2]
    rgb_image = prepare_display_image(rgb_raw, hsi_raw)
    timings["load_scene_s"] = round(time.time() - t_load, 2)

    # ---------------- fusion maps ----------------
    t_fusion = time.time()
    if args.reuse_run:
        print(f"Reusing maps from {args.reuse_run} (model pass skipped)")
        for fname in ("class_map.npy", "target_confidence_map.npy"):
            if not os.path.exists(os.path.join(args.reuse_run, fname)):
                raise FileNotFoundError(f"{fname} not found in --reuse_run folder '{args.reuse_run}'. "
                                        f"Point it at a run_<timestamp> folder from an earlier run.")
        class_map = np.load(os.path.join(args.reuse_run, "class_map.npy"))
        confidence_map = np.load(os.path.join(args.reuse_run, "target_confidence_map.npy"))
        if class_map.shape != (h, w) or confidence_map.shape != (h, w):
            raise ValueError(f"Reused maps are {class_map.shape} but the scene is {(h, w)} -- "
                             f"point --reuse_run at a run made from this same scene.")
    else:
        class_map, confidence_map = run_fusion_inference(
            args, hsi_raw, lidar_raw, target_class_idx=fusion_target_idx
        )
    timings["fusion_inference_s"] = round(time.time() - t_fusion, 2)

    # ---------------- spatial smoothing ----------------
    confidence_raw = confidence_map.copy()
    if args.spatial_smooth_sigma > 0:
        confidence_map = spatial_smooth_confidence(confidence_map, sigma=args.spatial_smooth_sigma)
        diff = np.abs(confidence_map - confidence_raw).mean()
        print(f"Spatial smoothing applied (sigma={args.spatial_smooth_sigma}, mean change={diff:.5f})")

    np.save(os.path.join(run_out, "class_map.npy"), class_map)
    np.save(os.path.join(run_out, "target_confidence_map.npy"), confidence_map)

    # ---------------- adaptive threshold ----------------
    otsu_thr = None
    if args.adaptive_threshold:
        otsu_thr, otsu_desc = adaptive_threshold_otsu(confidence_map)
        print(f"Adaptive threshold: {otsu_thr:.4f} ({otsu_desc}) vs fixed: {args.confidence_threshold}")

    # ---------------- overview figure + label evaluation ----------------
    try:
        save_detection_visualization(
            run_out, rgb_image, labels, class_map, confidence_map,
            class_names=active_class_names, class_colors_hex=active_class_colors,
            target_name=active_target_name, target_class_idx=fusion_target_idx,
        )
    except Exception as e:
        print(f"Visualization error: {e}")

    split_masks = None
    acc_all, acc_test = None, None
    if kind in ("muufl", "trento") and labels is not None and args.eval_split in ("auto", "test"):
        split_masks = reproduce_split(labels)
        n_test = int(split_masks["test"].sum())
        n_test_t = int((labels[split_masks["test"]] == args.target_label_id).sum())
        if kind == "muufl":
            print(f"\nReproduced preprocess.py split -> test set: {n_test:,d} pixels, {n_test_t} targets "
                  f"(expected {EXPECTED_TEST_SIZE:,d} / {EXPECTED_TEST_TARGETS})")
            if (n_test, n_test_t) != (EXPECTED_TEST_SIZE, EXPECTED_TEST_TARGETS):
                print("WARNING: this does not match your reported test set, so the 'test' numbers below may not be "
                      "your exact held-out pixels (different labels file or library version?).")
        else:
            print(f"\nStratified 60/20/20 split for {kind.upper()} -> test set: {n_test:,d} pixels, {n_test_t} targets (Class {args.target_label_id})")

    if labels is not None:
        try:
            acc_all, acc_test = save_label_evaluation(
                run_out, labels, class_map, kind, split_masks, class_names=active_class_names
            )
        except Exception as e:
            print(f"Label evaluation error: {e}")

    # ---------------- band importance ----------------
    band_importance = None
    if args.band_importance:
        try:
            band_importance = save_band_importance_plot(run_out, hsi_raw, confidence_map, wavelengths)
        except Exception as e:
            print(f"Band importance error: {e}")

    # ---------------- ACE + SAM + RX ----------------
    ace_map, ace_thr, sam_map, rx_map, ensemble_map = None, None, None, None, None
    ace_dir = os.path.join(run_out, "ace")
    metrics = None
    sweep_rows = []
    if args.ace:
        os.makedirs(ace_dir, exist_ok=True)
        print("\n" + "=" * 70 + "\n ACE (Adaptive Coherence Estimator) -- HSI-only baseline\n" + "=" * 70)
        try:
            use_split = split_masks is not None
            sig_mask = (split_masks["train"] | split_masks["val"]) if use_split else None
            spectrum, desc = get_ace_signature(args, kind, hsi_raw, labels, confidence_map, sig_mask)
            if spectrum is None:
                print(f"ACE skipped: {desc}")
            else:
                print(f"ACE target signature: {desc}")
                t_ace = time.time()
                if args.ace_multiscale:
                    ace_map = compute_multiscale_ace(hsi_raw, spectrum, chunk=100_000)
                else:
                    ace_map = compute_ace_map(hsi_raw, spectrum, reg=args.ace_reg)
                timings["ace_s"] = round(time.time() - t_ace, 2)
                np.save(os.path.join(ace_dir, "ace_score_map.npy"), ace_map)
                print(f"ACE score map computed (max={ace_map.max():.3f}, 99th pct={np.percentile(ace_map, 99):.3f})")

                # SAM detector
                if args.sam:
                    print("\n" + "=" * 70 + "\n SAM (Spectral Angle Mapper) -- HSI-only baseline\n" + "=" * 70)
                    t_sam = time.time()
                    sam_map = compute_sam_map(hsi_raw, spectrum)
                    timings["sam_s"] = round(time.time() - t_sam, 2)
                    np.save(os.path.join(ace_dir, "sam_score_map.npy"), sam_map)
                    print(f"SAM score map computed (max={sam_map.max():.3f}, 99th pct={np.percentile(sam_map, 99):.3f})")

                # RX anomaly detector
                if args.rx:
                    print("\n" + "=" * 70 + "\n RX (Reed-Xiaoli) -- Unsupervised Anomaly Detector\n" + "=" * 70)
                    t_rx = time.time()
                    rx_map = compute_rx_map(hsi_raw, reg=args.ace_reg)
                    timings["rx_s"] = round(time.time() - t_rx, 2)
                    np.save(os.path.join(ace_dir, "rx_score_map.npy"), rx_map)
                    print(f"RX anomaly map computed (max={rx_map.max():.3f}, 99th pct={np.percentile(rx_map, 99):.3f})")

                # Ensemble score
                if args.ensemble:
                    ensemble_map = compute_ensemble_score(
                        confidence_map, ace_map, sam_map,
                        w_fusion=args.ensemble_weights[0],
                        w_ace=args.ensemble_weights[1],
                        w_sam=args.ensemble_weights[2])
                    np.save(os.path.join(ace_dir, "ensemble_score_map.npy"), ensemble_map)
                    print(f"Ensemble score computed (weights: fusion={args.ensemble_weights[0]}, "
                          f"ace={args.ensemble_weights[1]}, sam={args.ensemble_weights[2]})")

                curves = None
                can_eval = (kind in ("muufl", "trento") or args.target_label_id is not None) and labels is not None
                if can_eval:
                    labeled = labels > 0
                    if use_split:
                        eval_mask, sel_mask, split_name = split_masks["test"], split_masks["train"] | split_masks["val"], "TEST"
                        sel_desc = "train+val pixels"
                    else:
                        eval_mask = sel_mask = labeled
                        split_name = "ALL-LABELED"
                        sel_desc = "all labeled pixels (optimistic: includes the evaluation pixels)"
                        print("NOTE: evaluating on ALL labeled pixels -- the fusion model was trained on most of them, "
                              "so its numbers are optimistic. ACE is unaffected by this (nothing is trained).")
                    metrics, curves = evaluate_detectors(labels, args.target_label_id, eval_mask, sel_mask, sel_desc,
                                                         ace_map, confidence_map, class_map,
                                                         sam_map=sam_map, ensemble_map=ensemble_map,
                                                         ace_threshold=args.ace_threshold, rx_map=rx_map)
                    table = format_metrics_table(metrics, split_name)
                    print("\n" + table)
                    with open(os.path.join(ace_dir, "ace_vs_fusion_metrics.txt"), "w", encoding="utf-8") as f:
                        f.write(table + f"\n\nACE signature: {desc}\n")
                    with open(os.path.join(ace_dir, "ace_vs_fusion_metrics.json"), "w", encoding="utf-8") as f:
                        json.dump({"signature": desc, "split": split_name, **metrics}, f, indent=4)
                    ace_thr = metrics.get("ace", {}).get("threshold")

                    # Calibration plot
                    try:
                        ece = save_calibration_plot(run_out, labels, confidence_map, args.target_label_id, eval_mask)
                    except Exception as e:
                        print(f"Calibration plot error: {e}")
                        ece = None

                    # Threshold sweep
                    if args.threshold_sweep:
                        try:
                            sweep_rows = save_threshold_sweep(run_out, labels, confidence_map, args.target_label_id, eval_mask, otsu_thr)
                        except Exception as e:
                            print(f"Threshold sweep error: {e}")

                if ace_thr is None:
                    if args.ace_threshold is not None:
                        ace_thr, thr_note = args.ace_threshold, "user-specified"
                    else:
                        ace_thr, thr_note = default_ace_threshold(ace_map, confidence_map)
                    print(f"ACE threshold for agreement analysis: {ace_thr:.4f} ({thr_note})")
                agreement = save_ace_figures(ace_dir, rgb_image, ace_map, confidence_map, args.confidence_threshold,
                                             ace_thr, curves, metrics,
                                             split_name if can_eval else "",
                                             sam_map=sam_map, rx_map=rx_map)
                print(f"Pixel agreement at the chosen thresholds: {agreement}")
                print(f"ACE outputs saved to {ace_dir}")
        except Exception as e:
            import traceback
            print(f"ACE stage error: {e}")
            traceback.print_exc()
            ace_map, ace_thr = None, None

    # ---------------- target identification ----------------
    results, coords = [], []
    if args.identify:
        tdir = os.path.join(run_out, "targets")
        os.makedirs(tdir, exist_ok=True)
        rx_thr_val = metrics.get("rx", {}).get("threshold") if metrics else None
        results, coords = identify_detected_targets(
            confidence_map, class_map, rgb_image, hsi_raw,
            ace_map, ace_thr, sam_map, ensemble_map, args, kind,
            lidar_raw=lidar_raw, rx_map=rx_map, rx_thr=rx_thr_val,
            class_names=active_class_names)

        # Proximity merging
        if args.merge_radius > 0 and len(results) > 1:
            results, coords = merge_nearby_targets(results, coords, merge_radius=args.merge_radius)

        save_target_reports(tdir, results, args.confidence_threshold, ace_thr if ace_thr is not None else float("nan"))
        if ace_map is not None and results:
            conf_n = sum(1 for o in results if o.get("ace_verdict", "").startswith("CONFIRMED"))
            print(f"ACE cross-check: {conf_n}/{len(results)} fusion detections are independently confirmed by ACE.")
        if rx_map is not None and results:
            rx_n = sum(1 for o in results if "ANOMALOUS" in o.get("rx_verdict", ""))
            print(f"RX cross-check: {rx_n}/{len(results)} fusion detections exhibit anomalous RX signatures.")
        try:
            save_target_figures(tdir, rgb_image, results, args)
            save_spectral_plots(tdir, hsi_raw, wavelengths, results, coords, confidence_map, args.max_visual_targets)
            if args.lidar_profiles:
                save_target_lidar_profiles(tdir, lidar_raw, results, coords, args.max_visual_targets)
            if args.yolo:
                save_enhanced_yolo(tdir, rgb_image, results, args)
        except Exception as e:
            import traceback
            print(f"Target figure error: {e}")
            traceback.print_exc()

    # ---------------- comprehensive run summary ----------------
    total_time = time.time() - t_start
    timings["total_s"] = round(total_time, 2)
    summary = {
        "run_folder": run_out,
        "timestamp": datetime.datetime.now().isoformat(),
        "dataset_kind": kind,
        "scene_shape": f"{h}x{w}x{hsi_raw.shape[-1]}",
        "total_pixels": h * w,
        "n_targets_detected": len(results),
        "n_high_confidence": sum(1 for r in results if r["quality_tier"] == "HIGH CONFIDENCE"),
        "n_medium_confidence": sum(1 for r in results if r["quality_tier"] == "MEDIUM CONFIDENCE"),
        "n_low_confidence": sum(1 for r in results if r["quality_tier"] in ("LOW CONFIDENCE", "LIKELY FALSE POSITIVE")),
        "confidence_threshold": args.confidence_threshold,
        "adaptive_threshold_otsu": otsu_thr,
        "spatial_smooth_sigma": args.spatial_smooth_sigma,
        "tta_enabled": args.tta,
        "mixed_precision": args.mixed_precision,
        "ace_multiscale": args.ace_multiscale if args.ace else False,
        "sam_enabled": args.sam if args.ace else False,
        "rx_enabled": args.rx if args.ace else False,
        "ensemble_enabled": args.ensemble if args.ace else False,
        "merge_radius": args.merge_radius,
        "threshold_sweep_enabled": args.threshold_sweep,
        "lidar_profiles_enabled": args.lidar_profiles,
    }
    if acc_all is not None:
        summary["accuracy_all_labeled"] = round(acc_all, 4)
    if acc_test is not None:
        summary["accuracy_test_split"] = round(acc_test, 4)
    if metrics:
        for det in ("fusion", "ace", "sam", "rx", "ensemble"):
            if det in metrics:
                summary[f"{det}_auc"] = metrics[det]["auc"]
                summary[f"{det}_f1"] = metrics[det]["f1"]
    summary["timings"] = timings

    with open(os.path.join(run_out, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    print(f"\nSaved run_summary.json")

    # ---------------- HTML report ----------------
    if args.html_report:
        try:
            image_files = [
                ("Detection Visualization", "detection_visualization.png"),
                ("Confusion Matrix", "confusion_matrix.png"),
                ("Calibration Diagram", "calibration_diagram.png"),
                ("Band Importance", "band_importance.png"),
            ]
            # Add ACE subfolder images
            for label, fname in [("ACE vs Fusion Maps", os.path.join("ace", "ace_vs_fusion_maps.png")),
                                 ("ROC / PR Curves", os.path.join("ace", "ace_vs_fusion_curves.png")),
                                 ("Target Crops", os.path.join("targets", "target_crops.png")),
                                 ("Full Scene Boxes", os.path.join("targets", "full_scene_boxes.png")),
                                 ("Spectral Signatures", os.path.join("targets", "spectral_signatures.png")),
                                 ("Target vs Background", os.path.join("targets", "spectral_target_vs_background.png")),
                                 ("Target LiDAR 3D Profiles", os.path.join("targets", "target_lidar_profiles.png"))]:
                image_files.append((label, fname))
            save_html_report(run_out, results, metrics if metrics else {}, summary, image_files, sweep_rows=sweep_rows)
        except Exception as e:
            import traceback
            print(f"HTML report error: {e}")
            traceback.print_exc()

    print(f"\nDone in {total_time:.1f}s. All outputs for this run are in: {run_out}")


if __name__ == "__main__":
    main()