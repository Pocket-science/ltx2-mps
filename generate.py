#!/usr/bin/env python3
"""
LTX-2 Video Generator for Apple Silicon (MPS)

Usage:
    python generate.py "Your prompt here" -o output.mp4 [options]

Options:
    --width      Video width (default: 512, must be divisible by 32)
    --height     Video height (default: 320, must be divisible by 32)
    --frames     Number of frames (default: 25, must be 8n+1)
    --steps      Inference steps (default: 20)
    --guidance   Guidance scale (default: 5.0)
    --fps        Output FPS (default: 24)
    --seed       Random seed (optional)
    -n           Negative prompt (optional)
"""

import argparse
import sys

import torch
from diffusers import LTX2Pipeline
from diffusers.utils import export_to_video


def main():
    parser = argparse.ArgumentParser(description="LTX-2 Video Generator for MPS")
    parser.add_argument("prompt", help="Text prompt for video generation")
    parser.add_argument("-o", "--output", default="output.mp4", help="Output video path")
    parser.add_argument("-n", "--negative-prompt", default="", help="Negative prompt")
    parser.add_argument("--steps", type=int, default=20, help="Inference steps")
    parser.add_argument("--guidance", type=float, default=5.0, help="Guidance scale")
    parser.add_argument("--width", type=int, default=512, help="Video width")
    parser.add_argument("--height", type=int, default=320, help="Video height")
    parser.add_argument("--frames", type=int, default=25, help="Number of frames")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    args = parser.parse_args()

    # Validate dimensions
    if args.width % 32 != 0:
        print(f"Error: width must be divisible by 32 (got {args.width})")
        sys.exit(1)
    if args.height % 32 != 0:
        print(f"Error: height must be divisible by 32 (got {args.height})")
        sys.exit(1)
    if (args.frames - 1) % 8 != 0:
        valid = [8*i + 1 for i in range(1, 13)]
        print(f"Error: frames must be 8n+1 (valid: {valid})")
        sys.exit(1)

    # Check MPS availability
    if not torch.backends.mps.is_available():
        print("Warning: MPS not available, falling back to CPU (will be slow)")
        device = "cpu"
    else:
        device = "mps"
        print(f"Using MPS (Apple Silicon GPU)")

    # Load model
    print("Loading LTX-2 model (this may take a while on first run)...")
    pipe = LTX2Pipeline.from_pretrained(
        "Lightricks/LTX-2",
        torch_dtype=torch.bfloat16
    )
    pipe.to(device)
    print("Model loaded!")

    # Set up generator
    if args.seed is None:
        args.seed = torch.randint(0, 2**31, (1,)).item()

    generator = torch.Generator(device="cpu")  # CPU generator more stable with MPS
    generator.manual_seed(args.seed)

    print(f"\nGenerating video...")
    print(f"  Prompt: {args.prompt}")
    print(f"  Size: {args.width}x{args.height}, {args.frames} frames")
    print(f"  Steps: {args.steps}, Guidance: {args.guidance}")
    print(f"  Seed: {args.seed}")
    print()

    # Generate
    result = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt if args.negative_prompt else None,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        width=args.width,
        height=args.height,
        num_frames=args.frames,
        generator=generator,
    )

    # Export video
    video_frames = result.frames[0]
    export_to_video(video_frames, args.output, fps=args.fps)

    print(f"\nVideo saved to: {args.output}")
    print(f"Seed: {args.seed}")


if __name__ == "__main__":
    main()
