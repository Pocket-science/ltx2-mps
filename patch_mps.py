#!/usr/bin/env python3
"""
LTX-2 MPS Patcher

Patches the diffusers library to enable LTX-2 on Apple Silicon (MPS).
The issue is that LTX-2 uses float64 for RoPE calculations, but MPS doesn't support float64.
This script forces float32 which works fine for video generation.

Usage:
    python patch_mps.py

Requirements:
    - diffusers (dev version with LTX2Pipeline)
    - pip install git+https://github.com/huggingface/diffusers.git
"""

import os
import sys
import site


def find_diffusers_path():
    """Find the diffusers installation path."""
    for path in site.getsitepackages():
        diffusers_path = os.path.join(path, "diffusers")
        if os.path.exists(diffusers_path):
            return diffusers_path

    # Check user site-packages
    user_site = site.getusersitepackages()
    if user_site:
        diffusers_path = os.path.join(user_site, "diffusers")
        if os.path.exists(diffusers_path):
            return diffusers_path

    return None


def patch_file(filepath, old_text, new_text, description):
    """Patch a file by replacing text."""
    if not os.path.exists(filepath):
        print(f"  SKIP: {filepath} not found")
        return False

    with open(filepath, 'r') as f:
        content = f.read()

    if new_text in content:
        print(f"  OK: {description} (already patched)")
        return True

    if old_text not in content:
        print(f"  SKIP: {description} (pattern not found)")
        return False

    content = content.replace(old_text, new_text)

    with open(filepath, 'w') as f:
        f.write(content)

    print(f"  PATCHED: {description}")
    return True


def main():
    print("LTX-2 MPS Patcher")
    print("=" * 50)

    diffusers_path = find_diffusers_path()

    if not diffusers_path:
        print("ERROR: diffusers not found. Install it first:")
        print("  pip install git+https://github.com/huggingface/diffusers.git")
        sys.exit(1)

    print(f"Found diffusers at: {diffusers_path}")
    print()

    # Patch 1: connectors.py
    connectors_path = os.path.join(diffusers_path, "pipelines", "ltx2", "connectors.py")
    patch_file(
        connectors_path,
        "freqs_dtype = torch.float64 if self.double_precision else torch.float32",
        "# MPS fix: force float32 as MPS doesn't support float64\n        freqs_dtype = torch.float32",
        "connectors.py RoPE dtype"
    )

    # Patch 2: transformer_ltx2.py
    transformer_path = os.path.join(diffusers_path, "models", "transformers", "transformer_ltx2.py")
    patch_file(
        transformer_path,
        "        # 3. Create a 1D grid of frequencies for RoPE\n        freqs_dtype = torch.float64 if self.double_precision else torch.float32",
        "        # 3. Create a 1D grid of frequencies for RoPE\n        # MPS fix: force float32 as MPS doesn't support float64\n        freqs_dtype = torch.float32",
        "transformer_ltx2.py RoPE dtype"
    )

    print()
    print("Done! LTX-2 should now work on Apple Silicon MPS.")
    print()
    print("Test with:")
    print("  python generate.py 'A cat walking' -o test.mp4")


if __name__ == "__main__":
    main()
