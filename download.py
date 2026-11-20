"""
Download the pretrained model code + checkpoints for:
  1. HSIMAE   (hyperspectral encoder)     -- HF repo: RyanWy/HSIMAE
  2. Utonia / PTv3  (LiDAR encoder)       -- HF repo: Pointcept/Utonia

Both checkpoints are hosted on Hugging Face, so both download the same way,
via huggingface_hub. No Google Drive file-ID hunting needed.

This version also handles the common failure mode where a repo's
requirements.txt pins an old CUDA-specific PyTorch build
(e.g. "torch~=1.12.1+cu116"), which modern pip rejects. It installs PyTorch
separately first, then strips any torch/torchvision/torchaudio lines out of
requirements.txt before installing the rest.

Requirements (installed automatically by this script, but you need pip itself):
    pip install huggingface_hub
"""

import os
import re
import subprocess

WORKDIR = "pretrained_models"
os.makedirs(WORKDIR, exist_ok=True)

HSIMAE_HF_REPO = "RyanWy/HSIMAE"
PTV3_HF_REPO = "Pointcept/Utonia"


def run(cmd, cwd=None, check=True):
    print(f"\n$ {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")
    return result.returncode


def install_torch():
    """Install PyTorch separately, before touching any repo's requirements.txt."""
    print("\n--- Installing PyTorch ---")
    has_nvidia_gpu = run("nvidia-smi", check=False) == 0

    if has_nvidia_gpu:
        print("NVIDIA GPU detected, installing CUDA build of PyTorch.")
        run("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    else:
        print("No NVIDIA GPU detected (or nvidia-smi unavailable), installing CPU build.")
        run("pip install torch torchvision")


def sanitize_requirements(req_path):
    """
    Remove torch/torchvision/torchaudio lines (already installed separately)
    and fix version pins that use '~=' with a local version label, which
    modern pip rejects.
    """
    if not os.path.exists(req_path):
        print(f"No requirements.txt found at {req_path}, skipping.")
        return

    with open(req_path, "r") as f:
        lines = f.readlines()

    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if re.match(r"^(torch|torchvision|torchaudio)\b", stripped):
            print(f"Skipping pinned line (installed separately): {stripped}")
            continue
        if "~=" in stripped and "+" in stripped:
            pkg = stripped.split("~=")[0]
            print(f"Rewriting problematic pin '{stripped}' -> '{pkg}'")
            cleaned.append(pkg + "\n")
            continue
        cleaned.append(line)

    backup_path = req_path + ".original"
    if not os.path.exists(backup_path):
        os.rename(req_path, backup_path)
    with open(req_path, "w") as f:
        f.writelines(cleaned)

    print(f"Cleaned requirements written to {req_path} (original backed up as {backup_path})")


def pip_install_requirements(repo_dir):
    req_path = os.path.join(repo_dir, "requirements.txt")
    sanitize_requirements(req_path)
    if os.path.exists(req_path):
        run("pip install -r requirements.txt", cwd=repo_dir)


# ---------------------------------------------------------------------------
# 1. HSIMAE (hyperspectral encoder)
# ---------------------------------------------------------------------------
def get_hsimae():
    repo_dir = os.path.join(WORKDIR, "HSIMAE")
    if not os.path.exists(repo_dir):
        run(f"git clone https://github.com/Ryan21wy/HSIMAE.git {repo_dir}")

    pip_install_requirements(repo_dir)

    from huggingface_hub import snapshot_download

    ckpt_dir = os.path.join(repo_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"\nDownloading HSIMAE checkpoint from Hugging Face repo: {HSIMAE_HF_REPO}")
    snapshot_download(repo_id=HSIMAE_HF_REPO, local_dir=ckpt_dir)
    print(f"HSIMAE checkpoint saved to {ckpt_dir}")


# ---------------------------------------------------------------------------
# 2. Utonia / Point Transformer V3 (LiDAR encoder)
# ---------------------------------------------------------------------------
def get_ptv3():
    repo_dir = os.path.join(WORKDIR, "Utonia")
    if not os.path.exists(repo_dir):
        run(f"git clone https://github.com/Pointcept/Utonia.git {repo_dir}")

    pip_install_requirements(repo_dir)

    from huggingface_hub import snapshot_download

    ckpt_dir = os.path.join(repo_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"\nDownloading Utonia/PTv3 checkpoint from Hugging Face repo: {PTV3_HF_REPO}")
    snapshot_download(repo_id=PTV3_HF_REPO, local_dir=ckpt_dir)
    print(f"PTv3 checkpoint saved to {ckpt_dir}")
    print(
        "\nNote: Utonia is a general cross-domain pretrained PTv3 encoder "
        "(indoor + outdoor point clouds), not one trained specifically on "
        "airborne/aerial LiDAR like DALES. It's a reasonable, easy-to-obtain "
        "starting point, but if your fine-tuning results are weak on the "
        "geometric branch, it's worth also checking Pointcept's original "
        "model zoo (https://github.com/Pointcept/Pointcept) for an "
        "outdoor-scene-specific checkpoint (e.g. SemanticKITTI) as an "
        "alternative to compare against."
    )


if __name__ == "__main__":
    print("pip installing huggingface_hub first...")
    run("pip install -U huggingface_hub")

    install_torch()

    print("\n=== Downloading HSIMAE ===")
    get_hsimae()

    print("\n=== Downloading Utonia / PTv3 ===")
    get_ptv3()

    print("\nDone. Check the 'pretrained_models/' folder for both repos and "
          "their checkpoint files.")