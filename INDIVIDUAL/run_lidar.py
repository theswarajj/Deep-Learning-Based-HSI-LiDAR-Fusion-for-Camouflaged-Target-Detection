"""
ONE-CLICK LiDAR-ONLY RUNNER
=============================
Runs the complete LiDAR-only pipeline end-to-end:
  Step 1: lidar_inference.py          → class_map + confidence heatmap + visualization
  Step 2: identify_targets_lidar.py   → individual target identification with:
          • Connected-component grouping
          • Color / context / quality scoring
          • Visual crops + bounding boxes
          • Super-Resolution + YOLOv8 (car/human/object detection)
          • Spectral signature analysis (if HSI source available)
          • JSON / CSV / TXT reports

Usage:
    python run_lidar.py
    python run_lidar.py --lidar_file INPUT\\LIDAR\\my_data.mat
    python run_lidar.py --skip_targets    (inference only, no target identification)
    python run_lidar.py --confidence_threshold 0.4 --use_morphology
"""

import argparse
import os
import sys
import subprocess
import glob


def find_latest_run(base_dir):
    """Find the most recent run_* folder inside base_dir."""
    if not os.path.isdir(base_dir):
        return None
    runs = sorted(glob.glob(os.path.join(base_dir, "run_*")))
    return runs[-1] if runs else None


def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

    parser = argparse.ArgumentParser(
        description="Run complete LiDAR-ONLY pipeline (inference + target identification)"
    )
    # --- Inference arguments ---
    parser.add_argument("--data", default=None,
                        help="Bundled MUUFL .mat file (defaults to project's MUUFL data)")
    parser.add_argument("--lidar_file", default=None,
                        help="Separate LiDAR file (.mat or .npy)")
    parser.add_argument("--labels", default=None,
                        help="Ground-truth labels file")
    parser.add_argument("--lidar_checkpoint", default=None,
                        help="Fine-tuned LiDAR checkpoint (defaults to runs/lidar_only/best_model.pt)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--use_normal", default=True, action=argparse.BooleanOptionalAction,
                        help="Include normal features (must match training)")

    # --- Target identification arguments ---
    parser.add_argument("--confidence_threshold", type=float, default=0.5,
                        help="Target identification confidence threshold")
    parser.add_argument("--min_size", type=int, default=2,
                        help="Minimum pixel count for a target")
    parser.add_argument("--skip_targets", action="store_true",
                        help="Skip target identification step (inference only)")
    parser.add_argument("--use_morphology", action="store_true",
                        help="Apply morphological filtering to clean detection mask")
    parser.add_argument("--crop_padding", type=int, default=10,
                        help="Extra context pixels around each visual crop")

    # --- Shared arguments ---
    parser.add_argument(
        "--rgb_source", default=None,
        help="RGB source for target crops + YOLOv8 (defaults to --data or bundled .mat)"
    )
    parser.add_argument(
        "--hsi_source", default=None,
        help="HSI source for spectral signature analysis "
             "(defaults to --rgb_source if it's a .mat)"
    )
    args = parser.parse_args()

    python = sys.executable

    # Determine the RGB/HSI source for target identification
    rgb_source = args.rgb_source
    if rgb_source is None:
        if args.data:
            rgb_source = args.data
        else:
            rgb_source = os.path.join(
                PROJECT_ROOT,
                r"MUUFLGulfport-master\MUUFLGulfportDataCollection\muufl_gulfport_campus_w_lidar_1.mat"
            )

    hsi_source = args.hsi_source
    if hsi_source is None and rgb_source and rgb_source.lower().endswith(".mat"):
        hsi_source = rgb_source

    print("=" * 70)
    print("  LiDAR-ONLY COMPLETE PIPELINE")
    print("  Step 1: LiDAR-only inference (PTv3/Utonia classifier)")
    print("  Step 2: Target identification + YOLOv8 enhancement")
    print("=" * 70)

    # =====================================================================
    # STEP 1: LiDAR-Only Inference
    # =====================================================================
    print("\n" + "-" * 70)
    print("  STEP 1 / 2 : LiDAR-ONLY INFERENCE")
    print("-" * 70 + "\n")

    inference_cmd = [python, os.path.join(SCRIPT_DIR, "lidar_inference.py")]
    if args.data:
        inference_cmd.extend(["--data", args.data])
    if args.lidar_file:
        inference_cmd.extend(["--lidar_file", args.lidar_file])
    if args.labels:
        inference_cmd.extend(["--labels", args.labels])
    if args.lidar_checkpoint:
        inference_cmd.extend(["--lidar_checkpoint", args.lidar_checkpoint])
    inference_cmd.extend(["--batch_size", str(args.batch_size)])
    if args.use_normal:
        inference_cmd.append("--use_normal")
    else:
        inference_cmd.append("--no-use_normal")

    result = subprocess.run(inference_cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"\nERROR: LiDAR inference failed with return code {result.returncode}")
        sys.exit(result.returncode)

    # =====================================================================
    # STEP 2: Target Identification (with YOLOv8 Enhancement)
    # =====================================================================
    if not args.skip_targets:
        print("\n" + "-" * 70)
        print("  STEP 2 / 2 : LiDAR-ONLY TARGET IDENTIFICATION")
        print("  (Super-Resolution + YOLOv8 for car/human/object detection)")
        print("-" * 70 + "\n")

        latest_run = find_latest_run(os.path.join(SCRIPT_DIR, "OUTPUT", "LIDAR"))
        if latest_run is None:
            print("WARNING: Could not find LiDAR inference output. Skipping target identification.")
        else:
            identify_cmd = [
                python, os.path.join(SCRIPT_DIR, "identify_targets_lidar.py"),
                "--detection_dir", latest_run,
                "--confidence_threshold", str(args.confidence_threshold),
                "--min_size", str(args.min_size),
                "--crop_padding", str(args.crop_padding),
            ]
            if rgb_source:
                identify_cmd.extend(["--rgb_source", rgb_source])
            if hsi_source:
                identify_cmd.extend(["--hsi_source", hsi_source])
            if args.use_morphology:
                identify_cmd.append("--use_morphology")

            result = subprocess.run(identify_cmd, cwd=PROJECT_ROOT)
            if result.returncode != 0:
                print(f"\nWARNING: Target identification exited with code {result.returncode}")

    # =====================================================================
    # SUMMARY
    # =====================================================================
    print("\n" + "=" * 70)
    print("  LiDAR-ONLY PIPELINE COMPLETE")
    latest = find_latest_run(os.path.join(SCRIPT_DIR, "OUTPUT", "LIDAR"))
    if latest:
        print(f"  Detection output : {latest}")
        targets_dir = os.path.join(latest, "targets")
        if os.path.isdir(targets_dir):
            print(f"  Target reports   : {targets_dir}")
        print("\n  Output files:")
        for f in sorted(os.listdir(latest)):
            fpath = os.path.join(latest, f)
            if os.path.isfile(fpath):
                size_kb = os.path.getsize(fpath) / 1024
                print(f"    {f:<45s} ({size_kb:>7.1f} KB)")
            elif os.path.isdir(fpath):
                n_files = len(os.listdir(fpath))
                print(f"    {f + '/':<45s} ({n_files} files)")
    print("=" * 70)


if __name__ == "__main__":
    main()
