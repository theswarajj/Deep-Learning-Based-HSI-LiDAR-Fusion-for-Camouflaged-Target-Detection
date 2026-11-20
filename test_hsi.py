"""
Evaluate an already-trained HSI-only checkpoint on the test set.

Use this when training was interrupted (or already finished) and you just
need the official test-set numbers from a saved best_model.pt, without
retraining from scratch.

Usage:
    python evaluate_hsi_only.py --data_dir preprocessed --hsimae_repo pretrained_models\HSIMAE \
                                 --checkpoint runs\hsi_only\best_model.pt
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from finetune_hsi import (
    HSIPatchDataset, build_model, CLASS_NAMES, TARGET_CLASS_IDX
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--hsimae_repo", required=True)
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--s_depth", type=int, default=9)
    args = parser.parse_args()

    sys.path.insert(0, args.hsimae_repo)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_ds = HSIPatchDataset(os.path.join(args.data_dir, "test.npz"), augment=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    num_bands = test_ds.hsi.shape[-1]
    num_classes = len(CLASS_NAMES)

    model = build_model(
        num_bands=num_bands, num_classes=num_classes,
        embed_dim=args.embed_dim, depth=args.depth, s_depth=args.s_depth,
        device=device,
    )
    try:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    except TypeError:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            outputs = model(x)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    preds = np.array(all_preds)
    labels = np.array(all_labels)

    print("\n--- TEST (HSI-only) results ---")
    print(classification_report(
        labels, preds, target_names=CLASS_NAMES, digits=3, zero_division=0
    ))

    target_mask = labels == TARGET_CLASS_IDX
    if target_mask.sum() > 0:
        target_recall = (preds[target_mask] == TARGET_CLASS_IDX).mean()
        print(f"Target class (Cloth Panels) recall: {target_recall:.3f} "
              f"({(preds[target_mask] == TARGET_CLASS_IDX).sum()}/{target_mask.sum()} correctly found)")


if __name__ == "__main__":
    main()