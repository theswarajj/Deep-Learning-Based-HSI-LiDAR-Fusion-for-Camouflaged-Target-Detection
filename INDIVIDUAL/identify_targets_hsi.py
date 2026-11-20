"""
Identify individual hidden/camouflaged targets from the HSI-ONLY detection
pipeline's output.  Calls the full Identify_targets.py via subprocess so
ALL features work correctly including:
  - Connected-component target grouping
  - Dominant color identification
  - Surrounding class-context analysis
  - Detection quality scoring (0-100)
  - Visual crops with descriptive labels
  - Full-scene bounding-box overlay
  - Super-Resolution + YOLOv8 enhancement (detects cars, humans, objects)
  - Spectral signature analysis + target-vs-background comparison
  - Text / JSON / CSV reports

Usage:
    python identify_targets_hsi.py
    python identify_targets_hsi.py --detection_dir OUTPUT\\HSI\\run_XXXXXXXX_XXXXXX
    python identify_targets_hsi.py --confidence_threshold 0.4 --use_morphology
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
    hsi_output_base = os.path.join(SCRIPT_DIR, "OUTPUT", "HSI")
    default_detection_dir = find_latest_run(hsi_output_base)

    parser = argparse.ArgumentParser(
        description="Identify camouflaged targets from HSI-ONLY detection output.\n"
                    "Includes YOLOv8 enhancement for car/human/object detection."
    )
    parser.add_argument(
        "--detection_dir", default=default_detection_dir,
        help="Folder with class_map.npy + target_confidence_map.npy from hsi_inference.py"
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
             "(defaults to --rgb_source if it's a .mat file)"
    )
    parser.add_argument("--use_morphology", action="store_true",
                        help="Clean up detection mask with morphological ops")
    parser.add_argument("--crop_padding", type=int, default=10,
                        help="Extra context pixels around each crop")
    parser.add_argument("--out", default=None,
                        help="Output folder (defaults to <detection_dir>/targets)")
    args = parser.parse_args()

    if args.detection_dir is None:
        print("ERROR: No HSI detection output found.")
        print("  Run hsi_inference.py first, or specify --detection_dir.")
        sys.exit(1)

    # Default hsi_source to rgb_source for spectral analysis (HSI is the star here!)
    if args.hsi_source is None and args.rgb_source and args.rgb_source.lower().endswith(".mat"):
        args.hsi_source = args.rgb_source

    print("=" * 70)
    print("  HSI-ONLY TARGET IDENTIFICATION")
    print("  (Includes Super-Resolution + YOLOv8 for car/human detection)")
    print(f"  Detection dir: {args.detection_dir}")
    print("=" * 70)

    # Build subprocess command to call the ORIGINAL Identify_targets.py
    # Running as subprocess ensures:
    #   - CWD is PROJECT_ROOT (yolov8n.pt is found)
    #   - Clean argument parsing (no sys.argv hacks)
    #   - All features work identically to fusion pipeline
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
        print("  HSI-ONLY TARGET IDENTIFICATION COMPLETE")
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
