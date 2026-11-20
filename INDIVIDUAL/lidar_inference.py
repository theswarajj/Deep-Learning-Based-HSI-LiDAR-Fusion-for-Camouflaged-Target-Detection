"""
LiDAR-ONLY inference pipeline for the camouflaged-target detection system.

Uses ONLY the fine-tuned LidarClassifier model (from finetune_lidar.py) --
no HSI data is loaded or used at all.  This lets you evaluate what LiDAR
elevation / geometric information alone can detect, as a direct comparison
against:
  - HSI-only    (hsi_inference.py)
  - Fusion model (inference_pipelineV2.py)

The LidarClassifier wraps the Utonia/PTv3 backbone with a classification
head (backbone + pooling + Linear head).  We load best_model.pt from
runs/lidar_only.

Outputs (in OUTPUT/LIDAR/<timestamped_run>/):
  - class_map.npy              : predicted class for every pixel
  - target_confidence_map.npy  : per-pixel confidence for the target class
  - detection_visualization.png: overview figure
  - classification_report.txt  : per-class metrics (if labels are provided)
  - confusion_matrix.png       : confusion matrix (if labels are provided)

Usage:
    python lidar_inference.py
    python lidar_inference.py --lidar_file INPUT\\LIDAR\\lidar_data.mat
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

# -- project root on sys.path --
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

CLASS_NAMES = [
    "Trees", "Mostly Grass", "Mixed Ground Surface", "Dirt and Sand",
    "Road", "Water", "Shadow of Buildings", "Buildings", "Sidewalk",
    "Yellow Curb", "Cloth Panels (TARGET)",
]
TARGET_CLASS_IDX = 10
LIDAR_PATCH_SIZE = 9
GRID_SIZE = 0.05


# ---------------------------------------------------------------------------
# LiDAR preprocessing
# ---------------------------------------------------------------------------
def normalize_lidar(lidar):
    if lidar.ndim == 2:
        lidar = lidar[:, :, None]
    result = lidar.copy()
    for c in range(result.shape[-1]):
        channel = result[:, :, c]
        mean, std = channel.mean(), channel.std() + 1e-8
        result[:, :, c] = (channel - mean) / std
    return result


# ---------------------------------------------------------------------------
# Scene loading — supports .mat, .npy, LAS/LAZ, PLY, GeoTIFF, ENVI, XYZ/CSV
# ---------------------------------------------------------------------------
def _pointcloud_to_raster(points, resolution=0.1, channels=None):
    """Convert unstructured 3D point cloud to a 2D elevation raster.
    points: (N, 3+) array with at least [x, y, z].
    channels: if provided, (N, C) array of per-point features (color, intensity, etc.)
    Returns: (H, W, C) raster where C includes z + any extra channels."""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Compute grid
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    grid_w = int(np.ceil((x_max - x_min) / resolution)) + 1
    grid_h = int(np.ceil((y_max - y_min) / resolution)) + 1

    # Limit raster size (cap at ~4000×4000 for memory)
    max_dim = 4000
    if grid_w > max_dim or grid_h > max_dim:
        scale = max(grid_w / max_dim, grid_h / max_dim)
        resolution *= scale
        grid_w = int(np.ceil((x_max - x_min) / resolution)) + 1
        grid_h = int(np.ceil((y_max - y_min) / resolution)) + 1
        print(f"  Adjusted resolution to {resolution:.4f} to fit {grid_h}×{grid_w} grid")

    # Map points to grid cells
    col_idx = np.clip(((x - x_min) / resolution).astype(int), 0, grid_w - 1)
    row_idx = np.clip(((y_max - y) / resolution).astype(int), 0, grid_h - 1)  # flip y

    # Build z-raster (max elevation per cell)
    z_raster = np.full((grid_h, grid_w), np.nan, dtype=np.float32)
    # Use np.maximum.at for max-pooling into grid
    np.maximum.at(z_raster, (row_idx, col_idx), z.astype(np.float32))

    # Fill NaN holes with nearest-neighbor interpolation
    from scipy.ndimage import distance_transform_edt
    nan_mask = np.isnan(z_raster)
    if nan_mask.any():
        _, nearest_idx = distance_transform_edt(nan_mask, return_distances=True, return_indices=True)
        z_raster = z_raster[nearest_idx[0], nearest_idx[1]]

    n_extra = 0
    if channels is not None and channels.shape[1] > 0:
        n_extra = channels.shape[1]
        extra_rasters = np.full((grid_h, grid_w, n_extra), np.nan, dtype=np.float32)
        for c in range(n_extra):
            np.maximum.at(extra_rasters[:, :, c], (row_idx, col_idx), channels[:, c].astype(np.float32))
        if nan_mask.any():
            for c in range(n_extra):
                ch = extra_rasters[:, :, c]
                ch_nan = np.isnan(ch)
                if ch_nan.any():
                    _, nn_idx = distance_transform_edt(ch_nan, return_distances=True, return_indices=True)
                    extra_rasters[:, :, c] = ch[nn_idx[0], nn_idx[1]]
        result = np.concatenate([z_raster[:, :, None], extra_rasters], axis=-1)
    else:
        result = z_raster[:, :, None]

    print(f"  Point cloud → raster: {len(points):,d} points → {grid_h}×{grid_w} grid, "
          f"{result.shape[-1]} channel(s)")
    return result


def load_las_laz(path):
    """Load LAS/LAZ point cloud → raster."""
    try:
        import laspy
    except ImportError:
        raise ImportError("laspy is required for LAS/LAZ files. Install with: pip install laspy[lazrs]")
    print(f"Loading LAS/LAZ: {path}")
    las = laspy.read(path)
    xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)
    print(f"  Points: {len(xyz):,d}, bounds: x[{xyz[:,0].min():.1f}, {xyz[:,0].max():.1f}], "
          f"y[{xyz[:,1].min():.1f}, {xyz[:,1].max():.1f}], z[{xyz[:,2].min():.1f}, {xyz[:,2].max():.1f}]")

    # Extract extra channels (intensity, RGB if available)
    extras = []
    if hasattr(las, 'intensity') and las.intensity is not None:
        extras.append(las.intensity.astype(np.float32)[:, None])
    if hasattr(las, 'red') and las.red is not None:
        rgb = np.vstack([las.red, las.green, las.blue]).T.astype(np.float32) / 65535.0
        extras.append(rgb)
    channels = np.concatenate(extras, axis=1) if extras else None
    return _pointcloud_to_raster(xyz, channels=channels)


def load_ply(path):
    """Load PLY point cloud → raster."""
    # Try open3d first (fast), fall back to plyfile
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(path)
        xyz = np.asarray(pcd.points, dtype=np.float64)
        colors = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() else None
        print(f"Loading PLY (open3d): {path}, {len(xyz):,d} points")
        return _pointcloud_to_raster(xyz, channels=colors)
    except ImportError:
        pass
    try:
        from plyfile import PlyData
        ply = PlyData.read(path)
        vertex = ply['vertex']
        xyz = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T.astype(np.float64)
        colors = None
        if 'red' in vertex:
            colors = np.vstack([vertex['red'], vertex['green'], vertex['blue']]).T.astype(np.float32) / 255.0
        print(f"Loading PLY (plyfile): {path}, {len(xyz):,d} points")
        return _pointcloud_to_raster(xyz, channels=colors)
    except ImportError:
        raise ImportError("PLY files need open3d or plyfile. Install: pip install open3d  or  pip install plyfile")


def load_geotiff_lidar(path):
    """Load GeoTIFF (elevation/DSM raster) for LiDAR."""
    try:
        import rasterio
    except ImportError:
        raise ImportError("rasterio is required for GeoTIFF. Install: pip install rasterio")
    print(f"Loading GeoTIFF: {path}")
    with rasterio.open(path) as ds:
        data = ds.read()  # (bands, H, W)
        raster = np.moveaxis(data, 0, -1).astype(np.float32)
        print(f"  Raster shape: {raster.shape}, bands: {ds.count}, dtype: {ds.dtypes[0]}")
    return raster


def load_envi_lidar(path):
    """Load ENVI BIL/BSQ/BIP raster for LiDAR."""
    try:
        import rasterio
    except ImportError:
        raise ImportError("rasterio is required for ENVI files. Install: pip install rasterio")
    # If .hdr, find data file
    actual_path = path
    if path.lower().endswith(".hdr"):
        base = path
        for ext in (".bil.hdr", ".bsq.hdr", ".bip.hdr", ".hdr"):
            if base.lower().endswith(ext):
                base = base[:-len(ext)]
                break
        for data_ext in (".bil", ".bsq", ".bip", ".dat", ".raw", ".img", ""):
            candidate = base + data_ext
            if os.path.isfile(candidate) and candidate.lower() != path.lower():
                actual_path = candidate
                break
        else:
            raise FileNotFoundError(f"No ENVI data file found for header: {path}")
    print(f"Loading ENVI: {actual_path}")
    with rasterio.open(actual_path) as ds:
        data = ds.read()
        raster = np.moveaxis(data, 0, -1).astype(np.float32)
        print(f"  Raster shape: {raster.shape}, bands: {ds.count}")
    return raster


def load_xyz_csv(path):
    """Load ASCII XYZ / CSV point cloud → raster."""
    print(f"Loading XYZ/CSV: {path}")
    # Try comma, then space/tab delimiters
    for delimiter in [',', None, '\t', ' ']:
        try:
            data = np.loadtxt(path, delimiter=delimiter, comments='#', dtype=np.float64)
            if data.ndim == 2 and data.shape[1] >= 3:
                break
        except Exception:
            continue
    else:
        raise ValueError(f"Could not parse {path} as XYZ/CSV (need at least 3 columns: x, y, z)")
    xyz = data[:, :3]
    channels = data[:, 3:].astype(np.float32) if data.shape[1] > 3 else None
    print(f"  Points: {len(xyz):,d}, columns: {data.shape[1]}")
    return _pointcloud_to_raster(xyz, channels=channels)


def load_lidar_scene(lidar_path, labels_path=None, rgb_source=None):
    """Load LiDAR data from any supported format. No HSI needed.

    Supported formats:
      .mat       — MATLAB (MUUFL-style or generic)
      .npy       — NumPy array
      .las/.laz  — LAS/LAZ point clouds (via laspy)
      .ply       — PLY point clouds (via open3d or plyfile)
      .tif/.tiff — GeoTIFF elevation/DSM rasters (via rasterio)
      .bil/.bsq/.bip/.hdr — ENVI rasters (via rasterio)
      .xyz/.csv/.txt — ASCII XYZ text files
    """
    lp = lidar_path.lower()

    # ---- Route to format-specific loader ----
    if lp.endswith((".las", ".laz")):
        lidar = load_las_laz(lidar_path)
    elif lp.endswith(".ply"):
        lidar = load_ply(lidar_path)
    elif lp.endswith((".tif", ".tiff")):
        lidar = load_geotiff_lidar(lidar_path)
    elif lp.endswith((".bil", ".bsq", ".bip", ".hdr", ".dat", ".img")):
        lidar = load_envi_lidar(lidar_path)
    elif lp.endswith((".xyz", ".csv", ".txt", ".asc")):
        lidar = load_xyz_csv(lidar_path)
    elif lp.endswith(".npy"):
        lidar = np.load(lidar_path).astype(np.float32)
        print(f"Loaded {lidar_path}: shape {lidar.shape}")
        if lidar.ndim == 2:
            lidar = lidar[:, :, None]
    elif lp.endswith(".mat"):
        raw = loadmat(lidar_path, simplify_cells=True)
        print(f"Top-level keys in {lidar_path}: {list(raw.keys())}")
        lidar_raw = None
        for key in ["Lidar", "lidar", "elevation", "z", "data", "Data"]:
            if key in raw:
                lidar_raw = np.array(raw[key])
                print(f"Using key '{key}': shape {lidar_raw.shape}")
                break
            if "hsi" in raw and isinstance(raw["hsi"], dict) and key in raw["hsi"]:
                lidar_raw = np.array(raw["hsi"][key])
                print(f"Using nested key 'hsi.{key}': shape {lidar_raw.shape}")
                break
        if lidar_raw is None:
            for key, value in raw.items():
                if not key.startswith("__"):
                    lidar_raw = np.array(value)
                    print(f"Fallback to first key '{key}': shape {lidar_raw.shape}")
                    break
        if lidar_raw is None:
            raise ValueError(f"No usable array in {lidar_path}")

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
    else:
        raise ValueError(
            f"Unsupported LiDAR format: {lidar_path}\n"
            f"Supported: .mat .npy .las .laz .ply .tif .tiff .bil .hdr .xyz .csv .txt"
        )

    lidar = lidar.astype(np.float32)

    # Pad/truncate to 4 channels (model expects 4)
    target_channels = 4
    if lidar.shape[-1] < target_channels:
        pad_width = target_channels - lidar.shape[-1]
        print(f"LiDAR has {lidar.shape[-1]} channel(s), padding to {target_channels}")
        lidar = np.concatenate(
            [lidar, np.zeros((*lidar.shape[:2], pad_width), dtype=np.float32)], axis=-1
        )
    elif lidar.shape[-1] > target_channels:
        print(f"LiDAR has {lidar.shape[-1]} channels, truncating to {target_channels}")
        lidar = lidar[:, :, :target_channels]

    labels = None
    if labels_path:
        llp = labels_path.lower()
        if llp.endswith(".npy"):
            labels = np.load(labels_path).astype(np.int64)
        elif llp.endswith(".mat"):
            raw = loadmat(labels_path, simplify_cells=True)
            for key in ["labels", "sceneLabels", "data", "Data", "gt", "groundtruth"]:
                if key in raw:
                    labels = np.array(raw[key], dtype=np.int64)
                    break
            if labels is None:
                if "hsi" in raw and isinstance(raw["hsi"], dict):
                    if "sceneLabels" in raw["hsi"]:
                        sl = raw["hsi"]["sceneLabels"]
                        if isinstance(sl, dict) and "labels" in sl:
                            labels = np.array(sl["labels"], dtype=np.int64)
            if labels is None:
                for key, value in raw.items():
                    if not key.startswith("__"):
                        labels = np.array(value, dtype=np.int64)
                        break
        if labels is not None:
            min_h = min(lidar.shape[0], labels.shape[0])
            min_w = min(lidar.shape[1], labels.shape[1])
            lidar = lidar[:min_h, :min_w, :]
            labels = labels[:min_h, :min_w]

    # Try to load RGB for visualization (optional)
    rgb_image = None
    if rgb_source:
        try:
            rsp = rgb_source.lower()
            if rsp.endswith((".png", ".jpg", ".jpeg", ".bmp")):
                from PIL import Image
                rgb_image = np.array(Image.open(rgb_source).convert("RGB")).astype(np.float32) / 255.0
            elif rsp.endswith(".mat"):
                raw = loadmat(rgb_source, simplify_cells=True)
                hsi_data = raw.get("hsi", raw)
                if isinstance(hsi_data, dict):
                    for k in ("Rgb", "RGB", "rgb"):
                        if k in hsi_data:
                            rgb_image = np.array(hsi_data[k])
                            break
                    if rgb_image is None and "Data" in hsi_data:
                        cube = np.array(hsi_data["Data"], dtype=np.float32)
                        n_bands = cube.shape[-1]
                        composite = cube[:, :, [n_bands - 1, n_bands // 2, 0]]
                        composite = (composite - composite.min()) / (composite.max() - composite.min() + 1e-8)
                        rgb_image = composite
            elif rsp.endswith((".tif", ".tiff", ".npy")):
                if rsp.endswith(".npy"):
                    rgb_image = np.load(rgb_source).astype(np.float32)
                else:
                    import rasterio
                    with rasterio.open(rgb_source) as ds:
                        if ds.count >= 3:
                            rgb_data = ds.read([1, 2, 3])
                            rgb_image = np.moveaxis(rgb_data, 0, -1).astype(np.float32)
                if rgb_image is not None:
                    rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min() + 1e-8)
        except Exception:
            pass

    return lidar, labels, rgb_image


def load_bundled_muufl_lidar(data_path, labels_path=None):
    """Load from a bundled MUUFL .mat file (LiDAR only, also grab RGB for viz)."""
    print(f"Loading scene data from: {data_path}")
    data_raw = loadmat(data_path, simplify_cells=True)
    hsi_data = data_raw["hsi"]

    lidar_returns = hsi_data["Lidar"]
    lidar_z0 = np.array(lidar_returns[0]["z"], dtype=np.float32)
    lidar_z1 = np.array(lidar_returns[1]["z"], dtype=np.float32)
    lidar = np.concatenate([lidar_z0, lidar_z1], axis=-1)

    # Pad to 4 channels
    if lidar.shape[-1] < 4:
        pad = 4 - lidar.shape[-1]
        lidar = np.concatenate([lidar, np.zeros((*lidar.shape[:2], pad), dtype=np.float32)], axis=-1)

    rgb_image = None
    for rgb_key in ("Rgb", "RGB", "rgb"):
        if rgb_key in hsi_data:
            rgb_image = np.array(hsi_data[rgb_key])
            break
    if rgb_image is None:
        cube = np.array(hsi_data["Data"], dtype=np.float32)
        n_bands = cube.shape[-1]
        composite = cube[:, :, [n_bands - 1, n_bands // 2, 0]]
        composite = (composite - composite.min()) / (composite.max() - composite.min() + 1e-8)
        rgb_image = composite

    labels = None
    if labels_path:
        labels_raw = loadmat(labels_path, simplify_cells=True)
        labels = np.array(labels_raw["hsi"]["sceneLabels"]["labels"], dtype=np.int64)
        label_h, label_w = labels.shape
        lid_h, lid_w = lidar.shape[:2]
        if label_w < lid_w:
            lidar = lidar[:label_h, :label_w, :]
        if label_h < lid_h:
            lidar = lidar[:label_h, :, :]

    return lidar, labels, rgb_image


# ---------------------------------------------------------------------------
# LiDAR-only model (full LidarClassifier with classification head)
# ---------------------------------------------------------------------------
def build_lidar_classifier(utonia_repo, pretrained_path, checkpoint_path,
                           use_normal, device):
    """Build the standalone LidarClassifier (backbone + classification head)
    and load the fine-tuned checkpoint from finetune_lidar.py."""
    sys.path.insert(0, utonia_repo)
    import utonia

    try:
        import flash_attn  # noqa
        backbone = utonia.load(pretrained_path).to(device)
    except ImportError:
        custom_config = dict(enc_patch_size=[1024 for _ in range(5)], enable_flash=False)
        backbone = utonia.load(pretrained_path, custom_config=custom_config).to(device)

    # Import the LidarClassifier from our finetune_lidar module
    from finetune_lidar import LidarClassifier, move_point_dict, collate_points

    model = LidarClassifier(backbone, num_classes=len(CLASS_NAMES)).to(device)

    # Load fine-tuned weights
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"Loaded LiDAR-only classifier from {checkpoint_path}")
    return model


def move_point_dict(point_dict, device):
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in point_dict.items()}


# ---------------------------------------------------------------------------
# Patch extraction & batched inference
# ---------------------------------------------------------------------------
def extract_lidar_patches(lidar):
    h, w, _ = lidar.shape
    half = LIDAR_PATCH_SIZE // 2
    padded = np.pad(lidar, ((half, half), (half, half), (0, 0)), mode="reflect")
    rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    return padded, rows.flatten(), cols.flatten(), half


def build_lidar_batch(lidar_padded, rows, cols, half, use_normal):
    """Build a batched point_dict for the LiDAR classifier."""
    win = LIDAR_PATCH_SIZE
    ys, xs = np.meshgrid(np.arange(win) - win // 2, np.arange(win) - win // 2, indexing="ij")
    rel_x = xs.flatten().astype(np.float32)
    rel_y = ys.flatten().astype(np.float32)

    coords, feats, counts = [], [], []
    for row, col in zip(rows, cols):
        r, c = row + half, col + half
        patch = lidar_padded[r - half:r + half + 1, c - half:c + half + 1, :]
        flat = patch.reshape(-1, 4)
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

    coord_batch = torch.cat(coords, dim=0).float()
    feat_batch = torch.cat(feats, dim=0).float()
    offset = torch.cumsum(torch.tensor(counts, dtype=torch.long), dim=0)

    point_dict = {"coord": coord_batch, "feat": feat_batch, "offset": offset, "grid_size": GRID_SIZE}
    return point_dict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="LiDAR-ONLY Inference Pipeline")
    parser.add_argument(
        "--data", default=os.path.join(
            PROJECT_ROOT,
            r"MUUFLGulfport-master\MUUFLGulfportDataCollection\muufl_gulfport_campus_w_lidar_1.mat"
        ),
        help="Bundled MUUFL-style .mat (LiDAR will be extracted, HSI used only for RGB viz)"
    )
    parser.add_argument("--lidar_file", default=None,
                        help="Separate LiDAR file (.mat, .npy, .las, .laz, .ply, .tif, .bil, .hdr, .xyz, .csv)")
    parser.add_argument("--rgb_source", default=None,
                        help="Optional RGB source for visualization (can be the bundled .mat)")
    parser.add_argument(
        "--labels", default=os.path.join(
            PROJECT_ROOT,
            r"MUUFLGulfport-master\MUUFLGulfportSceneLabels\muufl_gulfport_campus_1_hsi_220_label.mat"
        ),
        help="Optional ground-truth labels for accuracy check"
    )
    parser.add_argument("--utonia_repo", default=os.path.join(PROJECT_ROOT, "pretrained_models", "Utonia"))
    parser.add_argument(
        "--lidar_pretrained", default=os.path.join(
            PROJECT_ROOT, "pretrained_models", "Utonia", "checkpoints", "utonia.pth"
        ),
        help="Original pretrained utonia.pth"
    )
    parser.add_argument(
        "--lidar_checkpoint", default=os.path.join(PROJECT_ROOT, "runs", "lidar_only", "best_model.pt"),
        help="Fine-tuned LiDAR-only best_model.pt (from finetune_lidar.py)"
    )
    parser.add_argument(
        "--use_normal", default=True, action=argparse.BooleanOptionalAction,
        help="Include normal features (must match how model was trained)"
    )
    parser.add_argument(
        "--out", default=os.path.join(SCRIPT_DIR, "OUTPUT", "LIDAR"),
        help="Base output folder — each run creates a timestamped subfolder"
    )
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("=" * 70)
    print("  LiDAR-ONLY INFERENCE PIPELINE")
    print("  (No HSI data used — pure elevation/geometry detection)")
    print("=" * 70)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_out = os.path.join(args.out, f"run_{timestamp}")
    os.makedirs(run_out, exist_ok=True)
    print(f"Output folder: {run_out}")

    # --- Load scene (LiDAR only) ---
    if args.lidar_file:
        lidar, labels, rgb_image = load_lidar_scene(
            args.lidar_file, args.labels, args.rgb_source
        )
    elif args.data:
        lidar, labels, rgb_image = load_bundled_muufl_lidar(args.data, args.labels)
    else:
        raise ValueError("Provide either --data (bundled .mat) or --lidar_file.")
    print(f"Scene shape: LiDAR {lidar.shape}")

    # --- Preprocess ---
    lidar = normalize_lidar(lidar)

    h, w, _ = lidar.shape
    lidar_padded, rows, cols, half = extract_lidar_patches(lidar)
    n_pixels = len(rows)
    print(f"Running LiDAR-only inference on all {n_pixels:,d} pixels...")

    # --- Model ---
    model = build_lidar_classifier(
        args.utonia_repo, args.lidar_pretrained, args.lidar_checkpoint,
        args.use_normal, device
    )

    # Force-build the classification head with a dummy forward pass
    with torch.no_grad():
        dummy_dict = build_lidar_batch(
            lidar_padded, rows[:4], cols[:4], half, args.use_normal
        )
        dummy_dict = move_point_dict(dummy_dict, device)
        _ = model(dummy_dict)
    print("LiDAR classifier head initialized.")

    # --- Inference ---
    all_preds = np.zeros(n_pixels, dtype=np.int64)
    all_target_probs = np.zeros(n_pixels, dtype=np.float32)

    inference_start = time.time()
    with torch.no_grad():
        for start in tqdm(range(0, n_pixels, args.batch_size), desc="LiDAR-only inference"):
            end = min(start + args.batch_size, n_pixels)
            point_dict = build_lidar_batch(
                lidar_padded, rows[start:end], cols[start:end], half, args.use_normal
            )
            point_dict = move_point_dict(point_dict, device)

            logits = model(point_dict)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1).cpu().numpy()
            target_probs = probs[:, TARGET_CLASS_IDX].cpu().numpy()

            all_preds[start:end] = preds
            all_target_probs[start:end] = target_probs

    inference_time = time.time() - inference_start
    print(f"LiDAR-only inference completed in {inference_time:.2f} seconds.")

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

        # If no RGB image available, create a LiDAR elevation visualization
        if rgb_image is None:
            rgb_image = lidar[:, :, 0]  # first channel (elevation)
            rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min() + 1e-8)

        unique, counts = np.unique(class_map, return_counts=True)
        total_pixels = class_map.size
        stats_lines = ["\nPredicted class distribution (LiDAR-ONLY):"]
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
        if rgb_image.ndim == 2:
            ax_input.imshow(rgb_image, cmap="terrain")
            ax_input.set_title("Input Scene (LiDAR Elevation)", fontsize=12, fontweight="bold")
        else:
            ax_input.imshow(rgb_image)
            ax_input.set_title("Input Scene (RGB from bundled data)", fontsize=12, fontweight="bold")
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
        ax_pred.set_title("Predicted Class Map (LiDAR-ONLY)", fontsize=12, fontweight="bold")
        ax_pred.axis("off")
        ax_pred.legend(
            handles=legend_patches, loc="upper left", bbox_to_anchor=(1.02, 1.0),
            fontsize=8, title="Class Legend", title_fontsize=9,
            frameon=True, fancybox=True, shadow=True,
        )

        ax_heat = fig.add_subplot(2, n_cols, n_cols + 1)
        im = ax_heat.imshow(confidence_map, cmap="hot", vmin=0, vmax=1)
        ax_heat.set_title("Target Confidence (LiDAR-ONLY)", fontsize=12, fontweight="bold")
        ax_heat.axis("off")
        fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)

        ax_overlay = fig.add_subplot(2, n_cols, n_cols + 2)
        if rgb_image.ndim == 2:
            ax_overlay.imshow(rgb_image, cmap="terrain")
        else:
            ax_overlay.imshow(rgb_image)
        hot_cmap = cm.colormaps.get("hot", cm.get_cmap("hot")) if hasattr(cm, "colormaps") else cm.get_cmap("hot")
        overlay = hot_cmap(confidence_map)
        overlay[..., 3] = np.where(confidence_map > 0.3, np.clip((confidence_map - 0.3) / 0.7, 0, 1), 0)
        ax_overlay.imshow(overlay)
        ax_overlay.set_title("Target Confidence Overlay (LiDAR-ONLY)", fontsize=12, fontweight="bold")
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
        ax_bar.set_title("Class Distribution (LiDAR-ONLY)", fontsize=12, fontweight="bold")
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
        print(f"\nSanity check (LiDAR-ONLY): accuracy = {acc:.3f}, "
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
            print(" LiDAR-ONLY PER-CLASS CLASSIFICATION REPORT")
            print("=" * 70)
            print(report)

            report_path = os.path.join(run_out, "classification_report.txt")
            with open(report_path, "w") as f:
                f.write("LiDAR-ONLY PER-CLASS CLASSIFICATION REPORT\n")
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
            ax_cm.set_title(f"LiDAR-ONLY Confusion Matrix (Accuracy={acc:.3f})",
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

    print(f"\nAll LiDAR-ONLY outputs saved to: {run_out}")


if __name__ == "__main__":
    main()
