"""
Fetches the full raw Houston 2013 HSI + LiDAR scene using the fetch-houston2013
package, and saves it as .mat files in the same simple format your
inference_pipeline.py already knows how to read (a 'data' key holding the array).

Usage:
    python fetch_houston_full.py --out houston_full
"""

import argparse
import os

import numpy as np
from scipy.io import savemat
from fetch_houston2013 import fetch_houston2013

CACHE_FILENAME = "Houston2013.zip"


def find_cached_zip():
    """Searches the user's home directory for a cached Houston2013.zip."""
    home = os.path.expanduser("~")
    for root, dirs, files in os.walk(home):
        dirs[:] = [d for d in dirs if d.lower() not in
                   ("node_modules", "$recycle.bin", "windows")]
        if CACHE_FILENAME in files:
            return os.path.join(root, CACHE_FILENAME)
    return None


def clear_cached_zip():
    """Finds and deletes a cached Houston2013.zip, e.g. after a checksum failure."""
    found = find_cached_zip()
    if found:
        size_mb = os.path.getsize(found) / (1024 * 1024)
        print(f"Removing corrupted cached file: {found} ({size_mb:.1f} MB)")
        os.remove(found)
        return True
    print("No cached Houston2013.zip found to remove.")
    return False


def fetch_with_retry():
    """
    Tries fetch_houston2013() normally. If it fails on a checksum mismatch
    (a corrupted or incomplete previous download), automatically clears the
    bad cached file and retries once before giving up.
    """
    try:
        return fetch_houston2013()
    except AssertionError as e:
        if "SHA256" in str(e):
            print(f"\nDownload appears corrupted ({e}).")
            print("Clearing the bad cached file and retrying once...")
            clear_cached_zip()
            return fetch_houston2013()
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="houston_full",
                         help="Folder to save the fetched .mat files into")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("Fetching Houston 2013 dataset (this downloads from the official "
          "University of Houston source and may take a few minutes)...")
    hsi, dsm, train_label, test_label, info = fetch_with_retry()

    print(f"HSI shape: {hsi.shape}")
    print(f"DSM (LiDAR) shape: {dsm.shape}")
    print(f"Train label shape: {train_label.shape}")
    print(f"Test label shape: {test_label.shape}")
    print(f"Info: {info}")

    # The package returns arrays in (Bands, H, W) order (channel-first);
    # your pipeline expects (H, W, Bands) -- transpose to match.
    if hsi.ndim == 3 and hsi.shape[0] < hsi.shape[1]:
        hsi_hwc = np.transpose(hsi, (1, 2, 0))
    else:
        hsi_hwc = hsi

    if dsm.ndim == 3 and dsm.shape[0] < dsm.shape[1]:
        dsm_hwc = np.transpose(dsm, (1, 2, 0))
    elif dsm.ndim == 2:
        dsm_hwc = dsm[:, :, None]
    else:
        dsm_hwc = dsm

    print(f"Reshaped HSI to (H, W, Bands): {hsi_hwc.shape}")
    print(f"Reshaped DSM to (H, W, Channels): {dsm_hwc.shape}")

    # Combine train+test labels into one full ground-truth map, since your
    # pipeline expects a single label array covering the whole scene.
    try:
        train_dense = train_label.todense() if hasattr(train_label, "todense") else train_label
        test_dense = test_label.todense() if hasattr(test_label, "todense") else test_label
        full_labels = np.array(train_dense) + np.array(test_dense)
    except Exception as e:
        print(f"Could not merge train/test labels automatically ({e}); "
              f"saving them separately instead.")
        full_labels = None

    savemat(os.path.join(args.out, "houston_hsi.mat"), {"data": hsi_hwc})
    savemat(os.path.join(args.out, "houston_lidar.mat"), {"data": dsm_hwc})
    if full_labels is not None:
        savemat(os.path.join(args.out, "houston_labels.mat"), {"data": full_labels})
    else:
        savemat(os.path.join(args.out, "houston_train_labels.mat"), {"data": np.array(train_label)})
        savemat(os.path.join(args.out, "houston_test_labels.mat"), {"data": np.array(test_label)})

    print(f"\nSaved to {args.out}/:")
    print("  houston_hsi.mat")
    print("  houston_lidar.mat")
    print("  houston_labels.mat (or train/test separately if merge failed)")
    print("\nNow run your pipeline with:")
    print(f'  python inference_pipelineV2.py --hsi_file "{os.path.abspath(args.out)}\\houston_hsi.mat" '
          f'--lidar_file "{os.path.abspath(args.out)}\\houston_lidar.mat" '
          f'--labels "{os.path.abspath(args.out)}\\houston_labels.mat"')


if __name__ == "__main__":
    main()