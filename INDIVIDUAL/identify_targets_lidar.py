"""
Identify individual hidden/camouflaged targets from the LiDAR-ONLY detection
pipeline's output.  Calls the full Identify_targets.py via subprocess so
ALL features work correctly including:
  - Connected-component target grouping
  - Dominant color identification
  - Surrounding class-context analysis
  - Detection quality scoring (0-100)
  - Visual crops with descriptive labels
  - Full-scene bounding-box overlay
  - Super-Resolution + YOLOv8 enhancement (detects cars, humans, objects)
  - Spectral signature analysis (if HSI source available)
  - Text / JSON / CSV reports

Usage:
    python identify_targets_lidar.py
    python identify_targets_lidar.py --detection_dir OUTPUT\\LIDAR\\run_XXXXXXXX_XXXXXX
    python identify_targets_lidar.py --confidence_threshold 0.4 --use_morphology
"""

import argparse
import os
import sys
import subprocess


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RGB_SOURCE = os.path.join(
    PROJECT_ROOT,
    r"MUUFLGulfport-master\MUUFLGulfportDataCollection\muufl_gulfport_campus_w_lidar_1.mat"
)


def find_latest_run(base_dir):
    if not os.path.isdir(base_dir):
        return None
    runs = sorted([
        d for d in os.listdir(base_dir)
        if d.startswith("run_") and os.path.isdir(os.path.join(base_dir, d))
    ])
    return os.path.join(base_dir, runs[-1]) if runs else None


def main():
    lidar_output_base = os.path.join(SCRIPT_DIR, "OUTPUT", "LIDAR")
    default_detection_dir = find_latest_run(lidar_output_base)

    parser = argparse.ArgumentParser(
        description="Identify camouflaged targets from LiDAR-ONLY detection output.\n"
                    "Includes YOLOv8 enhancement for car/human/object detection."
    )
    parser.add_argument(
        "--detection_dir", default=default_detection_dir,
        help="Folder with class_map.npy + target_confidence_map.npy from lidar_inference.py"
    )
    parser.add_argument("--confidence_threshold", type=float, default=0.5,
                        help="Min confidence for a pixel to be part of a target")
    parser.add_argument("--min_size", type=int, default=2,
                        help="Min connected-pixel count to keep (filters noise)")
    parser.add_argument(
        "--rgb_source", default=DEFAULT_RGB_SOURCE,
        help="Original .mat or .npy for visual crops + YOLOv8 enhancement"
    )
    parser.add_argument(
        "--hsi_source", default=None,
        help="HSI source for spectral signature analysis "
             "(optional for LiDAR-only, defaults to --rgb_source if .mat)"
    )
    parser.add_argument("--use_morphology", action="store_true",
                        help="Clean up detection mask with morphological ops")
    parser.add_argument("--crop_padding", type=int, default=10,
                        help="Extra context pixels around each crop")
    parser.add_argument("--out", default=None,
                        help="Output folder (defaults to <detection_dir>/targets)")
    args = parser.parse_args()

    if args.detection_dir is None:
        print("ERROR: No LiDAR detection output found.")
        print("  Run lidar_inference.py first, or specify --detection_dir.")
        sys.exit(1)

    # Default hsi_source for spectral analysis (bonus analysis even for LiDAR-only)
    if args.hsi_source is None and args.rgb_source and args.rgb_source.lower().endswith(".mat"):
        args.hsi_source = args.rgb_source

    print("=" * 70)
    print("  LiDAR-ONLY TARGET IDENTIFICATION")
    print("  (Includes Super-Resolution + YOLOv8 for car/human detection)")
    print(f"  Detection dir: {args.detection_dir}")
    print("=" * 70)

    # Build subprocess command to call the ORIGINAL Identify_targets.py
    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "Identify_targets.py"),
        "--detection_dir", args.detection_dir,
        "--confidence_threshold", str(args.confidence_threshold),
        "--min_size", str(args.min_size),
        "--crop_padding", str(args.crop_padding),
    ]
    if args.rgb_source:
        cmd.extend(["--rgb_source", args.rgb_source])
    if args.hsi_source:
        cmd.extend(["--hsi_source", args.hsi_source])
    if args.use_morphology:
        cmd.append("--use_morphology")
    if args.out:
        cmd.extend(["--out", args.out])

    result = subprocess.run(cmd, cwd=PROJECT_ROOT)

    if result.returncode != 0:
        print(f"\nWARNING: Target identification exited with code {result.returncode}")
        sys.exit(result.returncode)

    # Print summary of what was produced
    out_dir = args.out or os.path.join(args.detection_dir, "targets")
    if os.path.isdir(out_dir):
        print("\n" + "=" * 70)
        print("  LiDAR-ONLY TARGET IDENTIFICATION COMPLETE")
        print("=" * 70)
        print(f"  Results saved to: {out_dir}")
        print("  Generated files:")
        for f in sorted(os.listdir(out_dir)):
            fpath = os.path.join(out_dir, f)
            size_kb = os.path.getsize(fpath) / 1024
            print(f"    {f:<40s} ({size_kb:>7.1f} KB)")
    print()


if __name__ == "__main__":
    main()
