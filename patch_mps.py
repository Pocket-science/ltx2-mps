#!/usr/bin/env python3
"""
patches diffusers to run ltx-2 on apple silicon (mps).

ltx-2 uses float64 for rope, but mps doesn't support it.
this forces float32 instead - works fine.

usage: python patch_mps.py
"""

import os
import sys
import site


def find_diffusers_path():
    """find where diffusers is installed"""
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
    """replace text in a file"""
    if not os.path.exists(filepath):
        print(f"  skip: {filepath} not found")
        return False

    with open(filepath, 'r') as f:
        content = f.read()

    if new_text in content:
        print(f"  ok: {description} (already patched)")
        return True

    if old_text not in content:
        print(f"  skip: {description} (pattern not found)")
        return False

    content = content.replace(old_text, new_text)

    with open(filepath, 'w') as f:
        f.write(content)

    print(f"  patched: {description}")
    return True


def main():
    print("ltx-2 mps patcher")
    print("-" * 40)

    diffusers_path = find_diffusers_path()

    if not diffusers_path:
        print("error: diffusers not found")
        print("  pip install git+https://github.com/huggingface/diffusers.git")
        sys.exit(1)

    print(f"found diffusers at: {diffusers_path}")
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
    print("done. ltx-2 should work on mps now.")
    print()
    print("test with:")
    print("  python generate.py 'a cat walking' -o test.mp4")


if __name__ == "__main__":
    main()
