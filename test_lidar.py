"""
Evaluate an already-trained LiDAR-only checkpoint on the test set.

Usage:
    python evaluate_lidar_only.py --data_dir preprocessed \
                                   --utonia_repo pretrained_models\\Utonia \
                                   --pretrained pretrained_models\\Utonia\\checkpoints\\utonia.pth \
                                   --checkpoint runs\\lidar_only\\best_model.pt \
                                   --use_normal
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from finetune_lidar import (
    LidarPatchDataset, collate_points, load_backbone, infer_in_channels,
    LidarClassifier, move_point_dict, target_precision_recall_f1,
    CLASS_NAMES, TARGET_CLASS_IDX,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--utonia_repo", required=True)
    parser.add_argument("--pretrained", required=True, help="Path to the local .pth backbone checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to your fine-tuned best_model.pt")
    parser.add_argument("--use_normal", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    sys.path.insert(0, args.utonia_repo)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_ds = LidarPatchDataset(os.path.join(args.data_dir, "test.npz"))
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda batch: collate_points(batch, args.use_normal),
    )
    print(f"Test size: {len(test_ds)}")

    # --- Rebuild the exact same model architecture used during training ---
    backbone = load_backbone(args.pretrained, device)
    expected_in_channels = infer_in_channels(backbone)
    print(f"Backbone expects {expected_in_channels} input channels.")
    if expected_in_channels == 9 and not args.use_normal:
        print("WARNING: backbone expects 9 channels but --use_normal was not set.")
    elif expected_in_channels == 6 and args.use_normal:
        print("WARNING: backbone expects 6 channels but --use_normal was set.")

    model = LidarClassifier(backbone, num_classes=len(CLASS_NAMES)).to(device)

    # The classification head is built lazily on first forward pass, so we
    # need to run one dummy batch through before loading the trained
    # state_dict, otherwise model.head is still None and load_state_dict
    # will fail to find those parameters.
    sample_point_dict, _ = collate_points([test_ds[i] for i in range(4)], args.use_normal)
    sample_point_dict = move_point_dict(sample_point_dict, device)
    with torch.no_grad():
        model(sample_point_dict)

    try:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    except TypeError:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # --- Run inference on the full test set ---
    all_preds, all_labels = [], []
    with torch.no_grad():
        for point_dict, y in test_loader:
            point_dict = move_point_dict(point_dict, device)
            outputs = model(point_dict)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    preds = np.array(all_preds)
    labels = np.array(all_labels)

    print("\n--- TEST (LiDAR-only) results ---")
    print(classification_report(
        labels, preds, target_names=CLASS_NAMES, digits=3, zero_division=0
    ))

    precision, recall, f1 = target_precision_recall_f1(preds, labels)
    print(f"Target class (Cloth Panels): precision {precision:.3f}, "
          f"recall {recall:.3f}, F1 {f1:.3f}")


if __name__ == "__main__":
    main()