

import argparse
import os
import sys

import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HSI_PATCH_SIZE = 9          # HSIMAE was pretrained on 9x9 spatial patches
LIDAR_PATCH_SIZE = 9        # same spatial window for LiDAR elevation
GWPCA_BANDS = 32            # HSIMAE-L expects 32 spectral bands
GWPCA_GROUPS = 4            # number of spectral groups for GWPCA
IGNORE_LABEL = -1           # unlabeled pixels in MUUFL's label map
TRAIN_FRAC = 0.6
VAL_FRAC = 0.2
# remaining ~0.2 goes to test
RANDOM_SEED = 42

# MUUFL Gulfport class names (1-indexed to match label values 1-11)
CLASS_NAMES = {
    1: "Trees",
    2: "Mostly Grass",
    3: "Mixed Ground Surface",
    4: "Dirt and Sand",
    5: "Road",
    6: "Water",
    7: "Shadow of Buildings",
    8: "Buildings",
    9: "Sidewalk",
    10: "Yellow Curb",
    11: "Cloth Panels",  # <-- camouflaged targets
}


# ---------------------------------------------------------------------------
# Group-Wise PCA (from HSIMAE repo, adapted)
# ---------------------------------------------------------------------------
def _split_spectral_groups(data_2d, num_groups=4):
    """Recursively bisect the spectral axis into `num_groups` sub-bands."""
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
    """
    Apply Group-Wise PCA to reduce HSI bands.

    Splits bands into `num_groups` sub-ranges, applies PCA independently
    to each, then concatenates. This preserves local spectral structure
    better than full PCA.

    Parameters
    ----------
    hsi_cube : ndarray (H, W, Bands)
    nc : int
        Total number of output components (nc // num_groups per group).
    num_groups : int
        Number of spectral sub-groups (must be power of 2).
    whiten : bool
        Whether to whiten PCA output.

    Returns
    -------
    ndarray (H, W, nc)
    """
    h, w, c = hsi_cube.shape
    flat = hsi_cube.reshape(-1, c).astype(np.float64)

    # Normalize to [0, 1] before PCA (same as HSIMAE repo)
    fmin, fmax = flat.min(), flat.max()
    flat = (flat - fmin) / (fmax - fmin + 1e-12)

    groups = _split_spectral_groups(flat, num_groups)
    components_per_group = nc // num_groups

    pca_outputs = []
    for i, group_data in enumerate(groups):
        # Ensure we don't request more components than features
        n_comp = min(components_per_group, group_data.shape[1])
        pca = PCA(n_components=n_comp, whiten=whiten, random_state=42)
        transformed = pca.fit_transform(group_data)
        pca_outputs.append(transformed)
        explained = pca.explained_variance_ratio_.sum()
        print(f"  GWPCA group {i}: {group_data.shape[1]} bands -> {n_comp} PCs "
              f"({explained:.1%} variance)")

    result = np.concatenate(pca_outputs, axis=-1)
    return result.reshape(h, w, -1).astype(np.float32)


# ---------------------------------------------------------------------------
# Data loading -- matches actual MUUFL Gulfport .mat structure
# ---------------------------------------------------------------------------
def load_raw(data_path, labels_path):
    """
    Load the MUUFL Gulfport .mat files.

    Data file structure:
        d['hsi']['Data']        -> (325, 337, 72) HSI cube
        d['hsi']['Lidar']       -> list of 2 dicts (first/last return),
                                   each with 'z' -> (325, 337, 2)

    Labels file structure:
        d['hsi']['sceneLabels']['labels'] -> (325, 220) int16
        d['hsi']['sceneLabels']['Materials_Type'] -> 11-element string array
    """
    print(f"Loading data from: {data_path}")
    data_raw = loadmat(data_path, simplify_cells=True)

    hsi_data = data_raw['hsi']
    hsi_cube = np.array(hsi_data['Data'], dtype=np.float32)
    print(f"  HSI cube shape: {hsi_cube.shape}")

    # LiDAR: two returns, each has 'z' with shape (H, W, 2)
    lidar_returns = hsi_data['Lidar']
    lidar_z0 = np.array(lidar_returns[0]['z'], dtype=np.float32)  # first return
    lidar_z1 = np.array(lidar_returns[1]['z'], dtype=np.float32)  # last return
    # Stack to (H, W, 4): [first_return_ch0, first_return_ch1, last_return_ch0, last_return_ch1]
    lidar = np.concatenate([lidar_z0, lidar_z1], axis=-1)
    print(f"  LiDAR stacked shape: {lidar.shape} "
          f"(first return {lidar_z0.shape} + last return {lidar_z1.shape})")

    # Labels
    print(f"Loading labels from: {labels_path}")
    labels_raw = loadmat(labels_path, simplify_cells=True)
    labels = np.array(
        labels_raw['hsi']['sceneLabels']['labels'],
        dtype=np.int64,
    )
    print(f"  Labels shape: {labels.shape}")

    # Class names
    class_names = labels_raw['hsi']['sceneLabels']['Materials_Type']
    print(f"  Classes: {list(class_names)}")

    # The labels cover only 220 columns (not the full 337).
    # Crop HSI and LiDAR to match.
    label_h, label_w = labels.shape
    hsi_h, hsi_w, _ = hsi_cube.shape

    if label_w < hsi_w:
        print(f"\n  Labels cover {label_w} columns, HSI/LiDAR have {hsi_w}. "
              f"Cropping to first {label_w} columns.")
        hsi_cube = hsi_cube[:label_h, :label_w, :]
        lidar = lidar[:label_h, :label_w, :]

    # Also crop rows if needed
    if label_h < hsi_h:
        print(f"  Labels cover {label_h} rows, HSI/LiDAR have {hsi_h}. "
              f"Cropping to first {label_h} rows.")
        hsi_cube = hsi_cube[:label_h, :, :]
        lidar = lidar[:label_h, :, :]

    assert hsi_cube.shape[:2] == labels.shape, (
        f"Shape mismatch after crop: HSI {hsi_cube.shape[:2]} vs labels {labels.shape}"
    )
    assert lidar.shape[:2] == labels.shape, (
        f"Shape mismatch after crop: LiDAR {lidar.shape[:2]} vs labels {labels.shape}"
    )

    return hsi_cube, lidar, labels


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
def normalize_hsi(hsi_cube):
    """Per-band normalization to zero mean, unit variance."""
    h, w, bands = hsi_cube.shape
    flat = hsi_cube.reshape(-1, bands)
    mean = flat.mean(axis=0, keepdims=True)
    std = flat.std(axis=0, keepdims=True) + 1e-8
    flat = (flat - mean) / std
    return flat.reshape(h, w, bands)


def normalize_lidar(lidar):
    """Normalize elevation to zero mean, unit variance (per channel)."""
    if lidar.ndim == 2:
        lidar = lidar[:, :, None]
    result = lidar.copy()
    for c in range(result.shape[-1]):
        channel = result[:, :, c]
        mean, std = channel.mean(), channel.std() + 1e-8
        result[:, :, c] = (channel - mean) / std
    return result


# ---------------------------------------------------------------------------
# Dataset building
# ---------------------------------------------------------------------------
def build_dataset(hsi_cube, lidar, labels):
    """Loop over every labeled pixel and build the patch/label arrays."""
    rows, cols = np.where(labels != IGNORE_LABEL)
    n_samples = len(rows)
    print(f"Found {n_samples} labeled pixels out of {labels.size} total "
          f"({n_samples/labels.size:.1%}).")

    # Print class distribution
    label_values = labels[rows, cols]
    unique, counts = np.unique(label_values, return_counts=True)
    print("\nClass distribution:")
    for val, cnt in zip(unique, counts):
        name = CLASS_NAMES.get(val, f"class_{val}")
        marker = " <-- TARGET" if val == 11 else ""
        print(f"  {val:>2}: {name:<25s} {cnt:>6,d} pixels{marker}")
    print()

    # --- Pad ONCE, then slice for each pixel ---
    hsi_half = HSI_PATCH_SIZE // 2
    lidar_half = LIDAR_PATCH_SIZE // 2

    hsi_padded = np.pad(
        hsi_cube,
        ((hsi_half, hsi_half), (hsi_half, hsi_half), (0, 0)),
        mode="reflect",
    )
    lidar_padded = np.pad(
        lidar,
        ((lidar_half, lidar_half), (lidar_half, lidar_half), (0, 0)),
        mode="reflect",
    )

    # Pre-allocate output arrays
    hsi_bands = hsi_cube.shape[2]
    lidar_ch = lidar.shape[2]
    hsi_patches = np.empty(
        (n_samples, HSI_PATCH_SIZE, HSI_PATCH_SIZE, hsi_bands), dtype=np.float32
    )
    lidar_patches = np.empty(
        (n_samples, LIDAR_PATCH_SIZE, LIDAR_PATCH_SIZE, lidar_ch), dtype=np.float32
    )
    class_labels = np.empty(n_samples, dtype=np.int64)

    for i, (row, col) in enumerate(zip(rows, cols)):
        # HSI patch (coordinates shifted by padding offset)
        r, c = row + hsi_half, col + hsi_half
        hsi_patches[i] = hsi_padded[
            r - hsi_half : r + hsi_half + 1,
            c - hsi_half : c + hsi_half + 1,
            :,
        ]
        # LiDAR patch
        r, c = row + lidar_half, col + lidar_half
        lidar_patches[i] = lidar_padded[
            r - lidar_half : r + lidar_half + 1,
            c - lidar_half : c + lidar_half + 1,
            :,
        ]
        class_labels[i] = labels[row, col]

        if (i + 1) % 10000 == 0:
            print(f"  Extracted {i+1:,d}/{n_samples:,d} patches...")

    # hsi_patches:   (N, 9, 9, 32)
    # lidar_patches: (N, 9, 9, 4)
    # class_labels:  (N,)
    return hsi_patches, lidar_patches, class_labels


def split_and_save(hsi_patches, lidar_patches, class_labels, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    idx = np.arange(len(class_labels))

    # Stratified split: train/val/test = 60/20/20
    # SAME split indices used for both HSI and LiDAR -- this is what keeps
    # the HSI-only / LiDAR-only / fusion comparison fair.
    train_idx, temp_idx = train_test_split(
        idx, train_size=TRAIN_FRAC, random_state=RANDOM_SEED, stratify=class_labels
    )
    val_size = VAL_FRAC / (1 - TRAIN_FRAC)
    val_idx, test_idx = train_test_split(
        temp_idx, train_size=val_size, random_state=RANDOM_SEED,
        stratify=class_labels[temp_idx]
    )

    splits = {"train": train_idx, "val": val_idx, "test": test_idx}
    for split_name, split_idx in splits.items():
        out_path = os.path.join(out_dir, f"{split_name}.npz")
        np.savez_compressed(
            out_path,
            hsi=hsi_patches[split_idx],
            lidar=lidar_patches[split_idx],
            labels=class_labels[split_idx],
        )
        unique, counts = np.unique(class_labels[split_idx], return_counts=True)
        print(f"  {split_name}: {len(split_idx):>6,d} samples -> {out_path}")
        # Show target class count
        target_mask = class_labels[split_idx] == 11
        print(f"    (cloth_panels/target: {target_mask.sum()} samples)")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess MUUFL Gulfport HSI+LiDAR for fine-tuning"
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to the HSI+LiDAR .mat file "
             "(e.g., muufl_gulfport_campus_w_lidar_1.mat)"
    )
    parser.add_argument(
        "--labels", required=True,
        help="Path to the scene labels .mat file "
             "(e.g., muufl_gulfport_campus_1_hsi_220_label.mat)"
    )
    parser.add_argument(
        "--out", default="preprocessed",
        help="Output directory for the .npz splits (default: preprocessed/)"
    )
    parser.add_argument(
        "--skip-gwpca", action="store_true",
        help="Skip GWPCA band reduction (not recommended, "
             "HSIMAE-L requires 32 bands)"
    )
    args = parser.parse_args()

    # --- 1. Load raw data ---
    print("=" * 60)
    print("Step 1: Loading raw data")
    print("=" * 60)
    hsi_cube, lidar, labels = load_raw(args.data, args.labels)
    print(f"\nAfter cropping:")
    print(f"  HSI:    {hsi_cube.shape}  (H x W x Bands)")
    print(f"  LiDAR:  {lidar.shape}  (H x W x Channels)")
    print(f"  Labels: {labels.shape}  (H x W)")

    # --- 2. GWPCA band reduction ---
    if not args.skip_gwpca:
        print("\n" + "=" * 60)
        print(f"Step 2: GWPCA band reduction ({hsi_cube.shape[2]} -> {GWPCA_BANDS})")
        print("=" * 60)
        hsi_cube = apply_gwpca(
            hsi_cube, nc=GWPCA_BANDS, num_groups=GWPCA_GROUPS, whiten=True
        )
        print(f"  HSI after GWPCA: {hsi_cube.shape}")
    else:
        print("\n  Skipping GWPCA (--skip-gwpca flag set)")
        print(f"  WARNING: HSIMAE-L expects 32 bands, but HSI has {hsi_cube.shape[2]}")

    # --- 3. Normalize ---
    print("\n" + "=" * 60)
    print("Step 3: Normalizing HSI and LiDAR")
    print("=" * 60)
    hsi_cube = normalize_hsi(hsi_cube)
    lidar = normalize_lidar(lidar)
    print(f"  HSI range after norm:   [{hsi_cube.min():.3f}, {hsi_cube.max():.3f}]")
    print(f"  LiDAR range after norm: [{lidar.min():.3f}, {lidar.max():.3f}]")

    # --- 4. Extract patches (HSI and LiDAR together, same pixel loop) ---
    print("\n" + "=" * 60)
    print("Step 4: Extracting patches for every labeled pixel")
    print("=" * 60)
    hsi_patches, lidar_patches, class_labels = build_dataset(hsi_cube, lidar, labels)
    print(f"Dataset shapes:")
    print(f"  HSI patches:   {hsi_patches.shape}")
    print(f"  LiDAR patches: {lidar_patches.shape}")
    print(f"  Labels:        {class_labels.shape}")
    print(f"  Label range:   [{class_labels.min()}, {class_labels.max()}]")

    # --- 5. Split and save (same split for both modalities) ---
    print("\n" + "=" * 60)
    print("Step 5: Splitting into train/val/test and saving")
    print("=" * 60)
    split_and_save(hsi_patches, lidar_patches, class_labels, args.out)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("Done! Summary of what was saved:")
    print("=" * 60)
    print(f"  Output directory: {os.path.abspath(args.out)}")
    print(f"  HSI patch shape:   ({HSI_PATCH_SIZE}, {HSI_PATCH_SIZE}, {hsi_cube.shape[2]})")
    print(f"  LiDAR patch shape: ({LIDAR_PATCH_SIZE}, {LIDAR_PATCH_SIZE}, {lidar.shape[2]})")
    print(f"  Number of classes: {len(np.unique(class_labels))}")
    print(f"\n  Each .npz file contains:")
    print(f"    'hsi'    -> (N, {HSI_PATCH_SIZE}, {HSI_PATCH_SIZE}, {hsi_cube.shape[2]})")
    print(f"    'lidar'  -> (N, {LIDAR_PATCH_SIZE}, {LIDAR_PATCH_SIZE}, {lidar.shape[2]})")
    print(f"    'labels' -> (N,)  values in {list(np.unique(class_labels))}")
    print(f"\n  Ready for fine-tuning! Labels use 1-indexing (1-11).")
    print(f"  Remap to 0-indexing (labels - 1) in your Dataset class.")
    print(f"\n  For HSI-only training: load these files, use only 'hsi' + 'labels'.")
    print(f"  For LiDAR-only training: load these files, use only 'lidar' + 'labels'.")
    print(f"  For fusion training: use both 'hsi' and 'lidar' + 'labels'.")


if __name__ == "__main__":
    main()