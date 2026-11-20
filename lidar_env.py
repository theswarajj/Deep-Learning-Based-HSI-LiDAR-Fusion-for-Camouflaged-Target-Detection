

import subprocess
import sys


def run(cmd):
    print(f"\n$ {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"WARNING: command exited with code {result.returncode}. "
              f"Read the error above -- you may need to adjust a version pin "
              f"manually before continuing.")
    return result.returncode


def main():
    print("=" * 60)
    print("Step 1: Reinstalling PyTorch for CUDA 12.4")
    print("=" * 60)
    print("NOTE: this replaces your current cu118 PyTorch build. This is "
          "required because spconv-cu124 and torch-scatter below are built "
          "specifically against CUDA 12.4 -- mixing CUDA versions across "
          "these packages causes cryptic import/runtime failures.")
    run("pip uninstall -y torch torchvision torchaudio")
    run("pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 "
        "--index-url https://download.pytorch.org/whl/cu124")

    print("\n" + "=" * 60)
    print("Step 2: Installing torch-scatter (matched to torch 2.5.0 + cu124)")
    print("=" * 60)
    run("pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu124.html")

    print("\n" + "=" * 60)
    print("Step 3: Installing spconv (cu124 build)")
    print("=" * 60)
    run("pip install spconv-cu124")

    print("\n" + "=" * 60)
    print("Step 4: Installing remaining dependencies")
    print("=" * 60)
    run("pip install addict timm huggingface_hub open3d \"numpy<=1.26.4\" "
        "scipy einops scikit-learn")

    print("\n" + "=" * 60)
    print("Step 5: Checking for flash-attention (optional)")
    print("=" * 60)
    try:
        import flash_attn  # noqa
        print("flash-attn is already installed and importable.")
    except ImportError:
        print("flash-attn is NOT installed (expected on native Windows -- "
              "it requires compiling from source). This is fine: "
              "finetune_lidar.py automatically falls back to "
              "enable_flash=False, which Utonia supports natively. "
              "You do not need to install it.")

    print("\n" + "=" * 60)
    print("Setup complete. Verifying the install...")
    print("=" * 60)
    run(f'{sys.executable} -c "import torch; print(\'torch:\', torch.__version__, \'cuda available:\', torch.cuda.is_available())"')
    run(f'{sys.executable} -c "import spconv; print(\'spconv OK\')"')
    run(f'{sys.executable} -c "import torch_scatter; print(\'torch_scatter OK\')"')
    run(f'{sys.executable} -c "import open3d; print(\'open3d OK\')"')

    print("\nIf all four checks above printed OK without errors, you're ready "
          "to run finetune_lidar.py.")


if __name__ == "__main__":
    main()