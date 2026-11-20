"""
Identify individual hidden/camouflaged targets from the detection pipeline's
output. Takes the confidence heatmap (target_confidence_map.npy) and groups
high-confidence pixels into distinct objects (connected components), rather
than just showing a blurry heatmap -- this answers "how many objects, where
exactly, how big, how confident" for each one individually.

Each detected target is also **identified** by:
  - Analysing its dominant RGB color  (e.g. "Red", "Blue", "Green")
  - Reading the surrounding class_map context (e.g. "in Grass Area")
  - Estimating a size category (Small / Medium / Large)
so the output reads something like:
    "Red Cloth Panel (Large) — in Mostly Grass area"

Usage:
    python identify_targets.py --detection_dir detection_output \\
                                --confidence_threshold 0.5 \\
                                --rgb_source MUUFLGulfport-master\\\\MUUFLGulfportDataCollection\\\\muufl_gulfport_campus_w_lidar_1.mat

Requires: numpy, scipy, matplotlib
"""

import argparse
import os
import json
import csv

import numpy as np
from scipy import ndimage
from scipy.io import loadmat

# ---- MUUFL Gulfport class names (1-indexed in class_map.npy) ----
CLASS_NAMES = [
    "Trees", "Mostly Grass", "Mixed Ground Surface", "Dirt and Sand",
    "Road", "Water", "Shadow of Buildings", "Buildings", "Sidewalk",
    "Yellow Curb", "Cloth Panels (TARGET)",
]


# ---------------------------------------------------------------------------
# Helper: dominant-color identification from RGB pixels
# ---------------------------------------------------------------------------
def _classify_color(r, g, b):
    """Map an (R, G, B) triplet in [0, 255] to a human-readable color name."""
    r, g, b = int(r), int(g), int(b)  # avoid uint8 overflow
    brightness = 0.299 * r + 0.587 * g + 0.114 * b
    saturation = max(r, g, b) - min(r, g, b)

    # Very dark -> Black/Dark
    if brightness < 40:
        return "Black/Dark"
    # Very bright -> White/Bright
    if brightness > 220 and saturation < 30:
        return "White/Bright"
    # Low saturation -> Gray
    if saturation < 25:
        if brightness < 120:
            return "Dark Gray"
        return "Light Gray"

    # Chromatic identification
    if r > g and r > b:
        if g > 150 and r > 180:
            return "Yellow/Gold"
        if g > 100 and b < 80:
            return "Orange/Brown"
        return "Red"
    if g > r and g > b:
        if r > 150:
            return "Yellow/Gold"
        if g - r > 30:
            return "Green"
        return "Light Green"
    if b > r and b > g:
        if r > 100 and b > 150:
            return "Purple/Violet"
        return "Blue"
    # Ties
    if abs(r - g) < 20 and b < r:
        return "Brown/Tan"
    return "Mixed Color"


def identify_target_color(rgb_image, obj_mask):
    """Determine the dominant color of the target pixels using the RGB image.

    Returns a human-readable color string (e.g. "Red", "Blue", "Green").
    If the target contains multiple distinct colors (like the cloth-panel
    cluster), returns a comma-separated list of up to 3 dominant colors.
    """
    if rgb_image is None:
        return "Unknown"

    # Extract RGB values for the target pixels
    pixels = rgb_image[obj_mask]  # (N, 3)
    if len(pixels) == 0:
        return "Unknown"

    # Normalize to 0-255 range if image is float [0,1]
    if pixels.max() <= 1.0:
        pixels = (pixels * 255).astype(np.uint8)
    else:
        pixels = pixels.astype(np.uint8)

    # For small targets (< 20 pixels), just use the mean color
    if len(pixels) < 20:
        mean_r, mean_g, mean_b = pixels.mean(axis=0)
        return _classify_color(mean_r, mean_g, mean_b)

    # For larger targets, try to find multiple distinct color regions
    # using simple k-means-like clustering on the RGB values
    from collections import Counter
    color_votes = Counter()
    for px in pixels:
        color_votes[_classify_color(px[0], px[1], px[2])] += 1

    # Return up to 3 most common colors (covering at least 10% each)
    total = len(pixels)
    dominant = []
    for color, count in color_votes.most_common(4):
        if count / total >= 0.10:
            dominant.append(color)
        if len(dominant) >= 3:
            break

    if not dominant:
        mean_r, mean_g, mean_b = pixels.mean(axis=0)
        return _classify_color(mean_r, mean_g, mean_b)

    return " + ".join(dominant) if len(dominant) > 1 else dominant[0]


# ---------------------------------------------------------------------------
# Helper: surrounding context from class_map
# ---------------------------------------------------------------------------
def identify_surrounding_context(class_map, bbox, obj_mask, context_pad=5):
    """Look at the class_map pixels AROUND (not inside) the target to
    determine what environment it sits in (e.g. "Trees", "Mostly Grass").

    Returns: (dominant_class_name, percentage)
    """
    if class_map is None:
        return "Unknown", 0.0

    row_min, row_max, col_min, col_max = bbox
    h, w = class_map.shape

    # Expand the bounding box by context_pad
    r0 = max(0, row_min - context_pad)
    r1 = min(h, row_max + context_pad + 1)
    c0 = max(0, col_min - context_pad)
    c1 = min(w, col_max + context_pad + 1)

    # Build a mask of the surrounding ring (context area minus the target)
    surround_mask = np.zeros_like(class_map, dtype=bool)
    surround_mask[r0:r1, c0:c1] = True
    surround_mask[obj_mask] = False  # exclude the target itself

    surround_pixels = class_map[surround_mask]
    if len(surround_pixels) == 0:
        return "Unknown", 0.0

    # Find the most common class in the surroundings
    from collections import Counter
    counts = Counter(surround_pixels.tolist())
    most_common_cls, most_common_count = counts.most_common(1)[0]

    # Map 1-indexed class to name
    if 1 <= most_common_cls <= len(CLASS_NAMES):
        cls_name = CLASS_NAMES[most_common_cls - 1]
    else:
        cls_name = f"Class {most_common_cls}"

    pct = 100.0 * most_common_count / len(surround_pixels)
    return cls_name, pct


# ---------------------------------------------------------------------------
# Helper: size category
# ---------------------------------------------------------------------------
def _size_category(n_pixels):
    if n_pixels >= 100:
        return "Large"
    if n_pixels >= 20:
        return "Medium"
    if n_pixels >= 5:
        return "Small"
    return "Tiny"


# ---------------------------------------------------------------------------
# Helper: Detection Quality Score
# ---------------------------------------------------------------------------
def compute_target_score(size, mean_conf, max_conf, surrounding_context):
    """Compute a composite detection quality score (0-100).

    Combines size reliability, confidence strength, and contextual
    plausibility into a single score.  Higher = more trustworthy.
    """
    # Size contribution (0-30): bigger targets are more reliable
    size_score = min(size / 50.0, 1.0) * 30

    # Mean confidence contribution (0-40)
    conf_score = mean_conf * 40

    # Peak confidence contribution (0-20)
    peak_score = max_conf * 20

    # Context bonus/penalty (0-10)
    # Cloth panels are typically on ground surfaces / grass
    favorable_contexts = {
        "Mostly Grass", "Mixed Ground Surface", "Dirt and Sand",
        "Trees", "Road",
    }
    if surrounding_context in favorable_contexts:
        context_score = 10
    elif surrounding_context in {"Buildings", "Sidewalk", "Yellow Curb"}:
        context_score = -5   # unusual location for cloth panels
    else:
        context_score = 3

    return max(0, min(100, size_score + conf_score + peak_score + context_score))


def _quality_tier(score):
    """Map a quality score (0-100) to a human-readable tier."""
    if score >= 75:
        return "HIGH CONFIDENCE"
    if score >= 50:
        return "MEDIUM CONFIDENCE"
    if score >= 30:
        return "LOW CONFIDENCE"
    return "LIKELY FALSE POSITIVE"


def _tier_color(tier):
    """Return a color string for each quality tier."""
    return {
        "HIGH CONFIDENCE": "#00C853",
        "MEDIUM CONFIDENCE": "#FFD600",
        "LOW CONFIDENCE": "#FF6D00",
        "LIKELY FALSE POSITIVE": "#D50000",
    }.get(tier, "#888888")


def load_rgb_for_crops(rgb_source):
    """Optional: load an RGB/false-color image so we can crop and show each
    candidate target visually, not just report its coordinates."""
    if rgb_source is None:
        return None
    if rgb_source.lower().endswith(".mat"):
        raw = loadmat(rgb_source, simplify_cells=True)
        hsi_data = raw.get("hsi", raw)
        for key in ("Rgb", "RGB", "rgb"):
            if key in hsi_data:
                return np.array(hsi_data[key])
        # fallback: false-color from raw HSI bands
        cube = np.array(hsi_data["Data"], dtype=np.float32)
        n_bands = cube.shape[-1]
        composite = cube[:, :, [n_bands - 1, n_bands // 2, 0]]
        composite = (composite - composite.min()) / (composite.max() - composite.min() + 1e-8)
        return composite
    elif rgb_source.lower().endswith(".npy"):
        return np.load(rgb_source)
    return None


# ---------------------------------------------------------------------------
# Helper: Load raw HSI cube for spectral signature analysis
# ---------------------------------------------------------------------------
def load_hsi_cube(source_path):
    """Load the raw hyperspectral cube from a .mat or .npy file.

    Returns (cube, wavelengths) where wavelengths may be None if not
    available in the file.
    """
    if source_path is None:
        return None, None
    if source_path.lower().endswith(".mat"):
        raw = loadmat(source_path, simplify_cells=True)
        hsi_data = raw.get("hsi", raw)
        cube = np.array(hsi_data["Data"], dtype=np.float32)
        # Try to get wavelength info
        wavelengths = None
        for wl_key in ("wavelength", "Wavelength", "wl", "lambda"):
            if wl_key in hsi_data:
                wavelengths = np.array(hsi_data[wl_key], dtype=np.float32)
                break
        return cube, wavelengths
    elif source_path.lower().endswith(".npy"):
        return np.load(source_path), None
    return None, None


def extract_spectral_signature(hsi_cube, obj_mask):
    """Get the mean spectral profile of a detected target.

    Returns (mean_spectrum, std_spectrum) arrays of shape (n_bands,).
    """
    h, w = obj_mask.shape
    cube = hsi_cube[:h, :w, :]  # crop to match mask
    pixels = cube[obj_mask]     # (N, n_bands)
    if len(pixels) == 0:
        return None, None
    return pixels.mean(axis=0), pixels.std(axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detection_dir", required=True,
                         help="Folder containing class_map.npy and target_confidence_map.npy "
                              "from inference_pipeline.py")
    parser.add_argument("--confidence_threshold", type=float, default=0.5,
                         help="Minimum confidence for a pixel to count as part of a target")
    parser.add_argument("--min_size", type=int, default=2,
                         help="Minimum connected-pixel-count to count as a real object "
                              "(filters out single-pixel noise)")
    parser.add_argument("--rgb_source", default=None,
                         help="Optional: original .mat or .npy scene, used to crop and "
                              "show each detected object visually")
    parser.add_argument("--use_morphology", action="store_true",
                         help="Apply morphological closing/opening to clean up the detection mask")
    parser.add_argument("--crop_padding", type=int, default=10,
                         help="Extra pixels of context around each crop")
    parser.add_argument("--hsi_source", default=None,
                         help="Optional: original .mat or .npy HSI scene for spectral "
                              "signature analysis (same file as --rgb_source is fine)")
    parser.add_argument("--out", default=None, help="Defaults to <detection_dir>/targets")
    args = parser.parse_args()

    # If hsi_source not given but rgb_source is a .mat, reuse it
    if args.hsi_source is None and args.rgb_source and args.rgb_source.lower().endswith(".mat"):
        args.hsi_source = args.rgb_source

    out_dir = args.out or os.path.join(args.detection_dir, "targets")
    os.makedirs(out_dir, exist_ok=True)

    confidence_map = np.load(os.path.join(args.detection_dir, "target_confidence_map.npy"))
    print(f"Loaded confidence map: shape {confidence_map.shape}, "
          f"max confidence {confidence_map.max():.3f}")

    # Load class_map for surrounding-context identification
    class_map_path = os.path.join(args.detection_dir, "class_map.npy")
    class_map = None
    if os.path.exists(class_map_path):
        class_map = np.load(class_map_path)
        print(f"Loaded class_map: shape {class_map.shape}")
    else:
        print("WARNING: class_map.npy not found — surrounding-context "
              "identification will be unavailable.")

    # --- 1. Threshold + connected-component labeling ---
    binary_mask = confidence_map >= args.confidence_threshold
    
    if args.use_morphology:
        print("Applying morphological filtering...")
        binary_mask = ndimage.binary_closing(binary_mask, structure=np.ones((3, 3)))
        binary_mask = ndimage.binary_opening(binary_mask, structure=np.ones((2, 2)))

    labeled_array, num_objects = ndimage.label(binary_mask)
    print(f"\nFound {num_objects} candidate object(s) above confidence "
          f"{args.confidence_threshold} before size filtering.")

    rgb_image = load_rgb_for_crops(args.rgb_source)
    # Crop RGB to match confidence_map dimensions if needed
    if rgb_image is not None:
        h, w = confidence_map.shape
        rgb_image = rgb_image[:h, :w]

    # --- 2. Characterize & identify each object ---
    results = []
    for obj_id in range(1, num_objects + 1):
        obj_mask = labeled_array == obj_id
        size = obj_mask.sum()
        if size < args.min_size:
            continue

        rows, cols = np.where(obj_mask)
        centroid_row, centroid_col = rows.mean(), cols.mean()
        row_min, row_max = rows.min(), rows.max()
        col_min, col_max = cols.min(), cols.max()
        mean_conf = confidence_map[obj_mask].mean()
        max_conf = confidence_map[obj_mask].max()

        # --- NEW: Identify color & context ---
        dominant_color = identify_target_color(rgb_image, obj_mask)
        size_cat = _size_category(size)
        bbox = (int(row_min), int(row_max), int(col_min), int(col_max))

        surround_cls, surround_pct = identify_surrounding_context(
            class_map, bbox, obj_mask
        )

        # Build a descriptive name
        name = f"{dominant_color} Cloth Panel ({size_cat})"
        context_str = f"in {surround_cls} area ({surround_pct:.0f}%)"

        # --- Compute Detection Quality Score ---
        quality_score = compute_target_score(
            size, mean_conf, max_conf, surround_cls
        )
        quality_tier = _quality_tier(quality_score)

        results.append({
            "id": len(results) + 1,
            "name": name,
            "dominant_color": dominant_color,
            "size_category": size_cat,
            "surrounding_context": surround_cls,
            "surrounding_pct": round(surround_pct, 1),
            "size_pixels": int(size),
            "centroid": (round(centroid_row, 1), round(centroid_col, 1)),
            "bbox": bbox,
            "mean_confidence": float(mean_conf),
            "max_confidence": float(max_conf),
            "quality_score": round(quality_score, 1),
            "quality_tier": quality_tier,
        })

    print(f"\n{'='*70}")
    print(f" IDENTIFIED TARGETS: {len(results)} object(s)")
    print(f"{'='*70}\n")

    for obj in results:
        tier_marker = "  *** WARNING ***" if obj['quality_tier'] == "LIKELY FALSE POSITIVE" else ""
        print(f"  Target #{obj['id']}: {obj['name']}")
        print(f"    --> Score:      {obj['quality_score']}/100 -- {obj['quality_tier']}{tier_marker}")
        print(f"    --> Located {obj['surrounding_context']} area "
              f"({obj['surrounding_pct']:.0f}% of surroundings)")
        print(f"    --> Color:      {obj['dominant_color']}")
        print(f"    --> Size:       {obj['size_pixels']} pixels ({obj['size_category']})")
        print(f"    --> Centroid:   row={obj['centroid'][0]}, col={obj['centroid'][1]}")
        print(f"    --> Bbox:       {obj['bbox']}")
        print(f"    --> Confidence: mean={obj['mean_confidence']:.3f}, "
              f"max={obj['max_confidence']:.3f}")
        print()

    # --- 3. Save reports ---
    report_path = os.path.join(out_dir, "targets_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Detected {len(results)} candidate target(s), "
                f"confidence threshold = {args.confidence_threshold}\n")
        f.write(f"{'='*70}\n\n")
        for obj in results:
            tier_marker = "  *** WARNING ***" if obj['quality_tier'] == "LIKELY FALSE POSITIVE" else ""
            f.write(f"Target #{obj['id']}: {obj['name']}\n")
            f.write(f"  Quality Score: {obj['quality_score']}/100 -- {obj['quality_tier']}{tier_marker}\n")
            f.write(f"  Location: in {obj['surrounding_context']} area "
                    f"({obj['surrounding_pct']:.0f}%)\n")
            f.write(f"  Color: {obj['dominant_color']}\n")
            f.write(f"  Size: {obj['size_pixels']}px ({obj['size_category']})\n")
            f.write(f"  Centroid: {obj['centroid']}\n")
            f.write(f"  Bbox: {obj['bbox']}\n")
            f.write(f"  Confidence: mean={obj['mean_confidence']:.3f}, "
                    f"max={obj['max_confidence']:.3f}\n\n")
    print(f"Report saved to {report_path}")

    # Save JSON report
    json_path = os.path.join(out_dir, "targets_report.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"JSON report saved to {json_path}")

    # Save CSV report
    csv_path = os.path.join(out_dir, "targets_report.csv")
    if results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"CSV report saved to {csv_path}")

    # --- 4. Save visual crops with descriptive names ---
    if rgb_image is not None and len(results) > 0:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, len(results),
                                     figsize=(5 * len(results), 5))
            if len(results) == 1:
                axes = [axes]

            for ax, obj in zip(axes, results):
                row_min, row_max, col_min, col_max = obj["bbox"]
                pad = args.crop_padding
                r0 = max(0, row_min - pad)
                r1 = min(rgb_image.shape[0], row_max + pad + 1)
                c0 = max(0, col_min - pad)
                c1 = min(rgb_image.shape[1], col_max + pad + 1)

                crop = rgb_image[r0:r1, c0:c1]
                ax.imshow(crop)
                # Use the descriptive name instead of "Target #N"
                ax.set_title(
                    f"#{obj['id']}: {obj['name']}\n"
                    f"in {obj['surrounding_context']} | "
                    f"conf={obj['mean_confidence']:.2f}",
                    fontsize=9, fontweight="bold",
                )
                ax.axis("off")

            plt.tight_layout()
            crops_path = os.path.join(out_dir, "target_crops.png")
            plt.savefig(crops_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Visual crops saved to {crops_path}")
            
            import matplotlib.patches as patches
            
            # --- 5. Save full scene with bounding boxes & descriptive labels ---
            fig_full, ax_full = plt.subplots(figsize=(14, 14))
            ax_full.imshow(rgb_image)
            for obj in results:
                row_min, row_max, col_min, col_max = obj["bbox"]
                width = col_max - col_min
                height = row_max - row_min

                # Color-code by confidence: green=high, yellow=medium, red=low
                conf = obj["mean_confidence"]
                if conf >= 0.8:
                    box_color = "#00FF00"
                elif conf >= 0.6:
                    box_color = "#FFD700"
                else:
                    box_color = "#FF4444"

                rect = patches.Rectangle(
                    (col_min, row_min), width, height,
                    linewidth=2.5, edgecolor=box_color, facecolor="none"
                )
                ax_full.add_patch(rect)

                # Descriptive label with name
                label_text = (f"#{obj['id']}: {obj['dominant_color']} "
                              f"({obj['size_category']})  "
                              f"conf={obj['mean_confidence']:.2f}")
                ax_full.text(
                    col_min, max(0, row_min - 4), label_text,
                    color="white", fontsize=7, fontweight="bold",
                    bbox=dict(
                        boxstyle="round,pad=0.3", facecolor=box_color,
                        alpha=0.85, edgecolor="none"
                    ),
                )
            ax_full.axis("off")
            ax_full.set_title("Full Scene — Identified Camouflaged Targets",
                              fontsize=14, fontweight="bold")
            plt.tight_layout()
            full_scene_path = os.path.join(out_dir, "full_scene_boxes.png")
            fig_full.savefig(full_scene_path, dpi=150, bbox_inches="tight")
            plt.close(fig_full)
            print(f"Full scene bounding boxes saved to {full_scene_path}")

            # --- 6. Enhanced Target Analysis: Super-Resolution + YOLOv8 ---
            # Upscale each crop for clarity, run YOLO for object detection,
            # then save a before/after comparison figure.
            try:
                import cv2
                from ultralytics import YOLO

                print("\n--- Enhanced Target Analysis (Super-Resolution + YOLOv8) ---")

                # Load YOLOv8 model (nano for speed, works on CPU too)
                yolo_model = YOLO("yolov8n.pt")
                print("Loaded YOLOv8n model for object detection")

                UPSCALE_FACTOR = 8   # 8x upscale so tiny crops become visible
                MIN_CROP_SIZE = 128  # minimum output dimension

                n_targets = len(results)
                # 2 rows: top = original, bottom = enhanced + YOLO detections
                fig_ba, axes_ba = plt.subplots(
                    2, n_targets,
                    figsize=(5 * n_targets, 10)
                )
                if n_targets == 1:
                    axes_ba = axes_ba.reshape(2, 1)

                for col_idx, obj in enumerate(results):
                    row_min, row_max, col_min, col_max = obj["bbox"]
                    pad = args.crop_padding
                    r0 = max(0, row_min - pad)
                    r1 = min(rgb_image.shape[0], row_max + pad + 1)
                    c0 = max(0, col_min - pad)
                    c1 = min(rgb_image.shape[1], col_max + pad + 1)

                    crop = rgb_image[r0:r1, c0:c1].copy()

                    # Normalize crop to uint8 for OpenCV processing
                    if crop.max() <= 1.0:
                        crop_uint8 = (crop * 255).astype(np.uint8)
                    else:
                        crop_uint8 = crop.astype(np.uint8)

                    # --- Row 1: Original (Before) ---
                    axes_ba[0, col_idx].imshow(crop_uint8)
                    axes_ba[0, col_idx].set_title(
                        f"BEFORE\n#{obj['id']}: {obj['name']}",
                        fontsize=8, fontweight="bold", color="red"
                    )
                    axes_ba[0, col_idx].axis("off")

                    # --- Enhancement pipeline ---
                    h_crop, w_crop = crop_uint8.shape[:2]

                    # Calculate upscale to reach at least MIN_CROP_SIZE
                    scale = max(
                        UPSCALE_FACTOR,
                        MIN_CROP_SIZE // max(h_crop, 1),
                        MIN_CROP_SIZE // max(w_crop, 1),
                    )
                    new_h, new_w = h_crop * scale, w_crop * scale

                    # Step 1: Bicubic upscale
                    enhanced = cv2.resize(
                        crop_uint8, (new_w, new_h),
                        interpolation=cv2.INTER_CUBIC
                    )

                    # Step 2: Denoise (Non-Local Means) — positional args
                    # for cross-version OpenCV compatibility
                    enhanced = cv2.fastNlMeansDenoisingColored(
                        enhanced, None, 6, 6, 7, 21
                    )

                    # Step 3: Sharpen with unsharp mask
                    gaussian = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=2.0)
                    enhanced = cv2.addWeighted(
                        enhanced, 1.5, gaussian, -0.5, 0
                    )

                    # --- Run YOLOv8 on the enhanced crop ---
                    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB) \
                        if enhanced.shape[-1] == 3 else enhanced
                    yolo_results = yolo_model(
                        enhanced_rgb, conf=0.25, verbose=False
                    )

                    # Draw YOLO detections on the enhanced image
                    enhanced_display = enhanced.copy()
                    yolo_labels = []
                    if len(yolo_results) > 0 and yolo_results[0].boxes is not None:
                        for box in yolo_results[0].boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            conf_val = float(box.conf[0])
                            cls_id = int(box.cls[0])
                            cls_name = yolo_model.names[cls_id]
                            yolo_labels.append(f"{cls_name} ({conf_val:.2f})")

                            # Draw box
                            cv2.rectangle(
                                enhanced_display, (x1, y1), (x2, y2),
                                (0, 255, 0), 2
                            )
                            # Draw label background
                            label = f"{cls_name} {conf_val:.2f}"
                            (tw, th), _ = cv2.getTextSize(
                                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                            )
                            cv2.rectangle(
                                enhanced_display,
                                (x1, y1 - th - 6), (x1 + tw + 4, y1),
                                (0, 255, 0), -1
                            )
                            cv2.putText(
                                enhanced_display, label,
                                (x1 + 2, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 0), 1, cv2.LINE_AA
                            )

                    # --- Row 2: Enhanced + YOLO (After) ---
                    axes_ba[1, col_idx].imshow(
                        cv2.cvtColor(enhanced_display, cv2.COLOR_BGR2RGB)
                        if enhanced_display.shape[-1] == 3
                        else enhanced_display
                    )

                    if yolo_labels:
                        det_text = "\n".join(yolo_labels[:3])
                        after_title = f"AFTER (Enhanced + YOLO)\n{det_text}"
                    else:
                        after_title = (
                            f"AFTER (Enhanced)\n"
                            f"YOLO: Camouflaged (no standard\n"
                            f"objects detected -- target is hidden)"
                        )

                    axes_ba[1, col_idx].set_title(
                        after_title, fontsize=7, fontweight="bold",
                        color="green"
                    )
                    axes_ba[1, col_idx].axis("off")

                    # Save individual enhanced crop
                    indiv_path = os.path.join(
                        out_dir, f"target_{obj['id']}_enhanced.png"
                    )
                    cv2.imwrite(indiv_path, enhanced_display)

                # Add row labels
                fig_ba.text(
                    0.02, 0.75, "ORIGINAL",
                    fontsize=14, fontweight="bold", color="red",
                    rotation=90, va="center", ha="center"
                )
                fig_ba.text(
                    0.02, 0.25, "ENHANCED\n+ YOLO",
                    fontsize=14, fontweight="bold", color="green",
                    rotation=90, va="center", ha="center"
                )

                fig_ba.suptitle(
                    "Before vs After: Super-Resolution + YOLOv8 Detection",
                    fontsize=14, fontweight="bold", y=1.02
                )
                plt.tight_layout()
                plt.subplots_adjust(left=0.05, hspace=0.25)

                ba_path = os.path.join(out_dir, "target_before_after.png")
                fig_ba.savefig(ba_path, dpi=150, bbox_inches="tight")
                plt.close(fig_ba)
                print(f"Before/After comparison saved to {ba_path}")
                print(f"Individual enhanced crops saved to {out_dir}/")

            except ImportError as ie:
                print(f"Skipping enhanced analysis (missing dependency: {ie})")
            except Exception as e:
                print(f"Enhanced analysis encountered an error: {e}")
                import traceback
                traceback.print_exc()

            # --- 7. Spectral Signature Analysis ---
            try:
                hsi_cube, wavelengths = load_hsi_cube(args.hsi_source)
                if hsi_cube is not None:
                    print("\n--- Spectral Signature Analysis ---")
                    n_bands = hsi_cube.shape[-1]

                    # Build x-axis: use wavelengths if available, else band index
                    if wavelengths is not None and len(wavelengths) == n_bands:
                        x_axis = wavelengths
                        x_label = "Wavelength (nm)"
                    else:
                        x_axis = np.arange(n_bands)
                        x_label = "Band Index"

                    # Color palette for targets
                    spectral_colors = [
                        "#E53935", "#43A047", "#1E88E5", "#FB8C00",
                        "#8E24AA", "#00ACC1", "#F4511E", "#3949AB",
                    ]

                    # --- Individual + Combined spectral plots ---
                    fig_spec, ax_combined = plt.subplots(
                        figsize=(12, 6)
                    )

                    for idx, obj in enumerate(results):
                        row_min, row_max, col_min, col_max = obj["bbox"]
                        obj_mask = np.zeros(confidence_map.shape, dtype=bool)
                        # Reconstruct mask from labeled_array
                        # Use confidence threshold to get the pixels
                        obj_mask[
                            row_min:row_max + 1,
                            col_min:col_max + 1
                        ] = (
                            confidence_map[
                                row_min:row_max + 1,
                                col_min:col_max + 1
                            ] >= args.confidence_threshold
                        )

                        mean_spec, std_spec = extract_spectral_signature(
                            hsi_cube, obj_mask
                        )
                        if mean_spec is None:
                            continue

                        color = spectral_colors[idx % len(spectral_colors)]
                        tier = obj["quality_tier"]
                        label = (
                            f"#{obj['id']}: {obj['name']} "
                            f"[{obj['quality_score']}/100 {tier}]"
                        )

                        # Plot on combined chart
                        ax_combined.plot(
                            x_axis, mean_spec, color=color,
                            linewidth=1.5, label=label
                        )
                        ax_combined.fill_between(
                            x_axis,
                            mean_spec - std_spec,
                            mean_spec + std_spec,
                            alpha=0.15, color=color
                        )

                    ax_combined.set_xlabel(x_label, fontsize=11)
                    ax_combined.set_ylabel("Reflectance / Intensity", fontsize=11)
                    ax_combined.set_title(
                        "Spectral Signatures of Detected Targets",
                        fontsize=14, fontweight="bold"
                    )
                    ax_combined.legend(
                        fontsize=7, loc="upper right",
                        frameon=True, fancybox=True, shadow=True
                    )
                    ax_combined.grid(True, alpha=0.3)
                    plt.tight_layout()

                    spec_path = os.path.join(
                        out_dir, "spectral_signatures.png"
                    )
                    fig_spec.savefig(spec_path, dpi=150, bbox_inches="tight")
                    plt.close(fig_spec)
                    print(f"Spectral signatures saved to {spec_path}")

                    # --- Also extract background spectrum for comparison ---
                    bg_mask = confidence_map < 0.1  # low-confidence = background
                    if bg_mask.sum() > 100:
                        bg_mean, bg_std = extract_spectral_signature(
                            hsi_cube, bg_mask
                        )
                        if bg_mean is not None:
                            fig_comp, ax_comp = plt.subplots(figsize=(12, 6))

                            # Plot background
                            ax_comp.plot(
                                x_axis, bg_mean, color="#888888",
                                linewidth=2, label="Background (non-target)",
                                linestyle="--"
                            )
                            ax_comp.fill_between(
                                x_axis,
                                bg_mean - bg_std,
                                bg_mean + bg_std,
                                alpha=0.1, color="#888888"
                            )

                            # Plot each target
                            for idx, obj in enumerate(results):
                                row_min, row_max, col_min, col_max = obj["bbox"]
                                obj_mask = np.zeros(
                                    confidence_map.shape, dtype=bool
                                )
                                obj_mask[
                                    row_min:row_max + 1,
                                    col_min:col_max + 1
                                ] = (
                                    confidence_map[
                                        row_min:row_max + 1,
                                        col_min:col_max + 1
                                    ] >= args.confidence_threshold
                                )
                                mean_s, _ = extract_spectral_signature(
                                    hsi_cube, obj_mask
                                )
                                if mean_s is None:
                                    continue
                                color = spectral_colors[
                                    idx % len(spectral_colors)
                                ]
                                ax_comp.plot(
                                    x_axis, mean_s, color=color,
                                    linewidth=1.5,
                                    label=f"#{obj['id']}: {obj['name']}"
                                )

                            ax_comp.set_xlabel(x_label, fontsize=11)
                            ax_comp.set_ylabel(
                                "Reflectance / Intensity", fontsize=11
                            )
                            ax_comp.set_title(
                                "Target vs Background Spectral Comparison",
                                fontsize=14, fontweight="bold"
                            )
                            ax_comp.legend(
                                fontsize=7, loc="upper right",
                                frameon=True, fancybox=True, shadow=True
                            )
                            ax_comp.grid(True, alpha=0.3)
                            plt.tight_layout()

                            comp_path = os.path.join(
                                out_dir, "spectral_target_vs_background.png"
                            )
                            fig_comp.savefig(
                                comp_path, dpi=150, bbox_inches="tight"
                            )
                            plt.close(fig_comp)
                            print(
                                f"Target vs Background comparison saved "
                                f"to {comp_path}"
                            )
                else:
                    print(
                        "No HSI source provided -- skipping spectral "
                        "signature analysis."
                    )
            except Exception as e:
                print(f"Spectral analysis encountered an error: {e}")
                import traceback
                traceback.print_exc()

        except ImportError:
            print("matplotlib not installed -- skipping visual crops.")
    elif len(results) > 0:
        print("No --rgb_source provided -- skipping visual crops "
              "(coordinates/report above are still available).")


if __name__ == "__main__":
    main()