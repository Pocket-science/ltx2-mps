#!/usr/bin/env python3
"""
ltx-2 video generator for mps

usage: python generate.py "your prompt" -o output.mp4
"""

import argparse
import sys

import numpy as np
import torch
from diffusers import LTX2Pipeline
from diffusers.pipelines.ltx2.export_utils import encode_video


def main():
    parser = argparse.ArgumentParser(description="ltx-2 video generator for mps")
    parser.add_argument("prompt", help="text prompt")
    parser.add_argument("-o", "--output", default="output.mp4", help="output path")
    parser.add_argument("-n", "--negative-prompt", default="", help="negative prompt")
    parser.add_argument("--steps", type=int, default=20, help="inference steps")
    parser.add_argument("--guidance", type=float, default=5.0, help="guidance scale")
    parser.add_argument("--width", type=int, default=512, help="video width")
    parser.add_argument("--height", type=int, default=320, help="video height")
    parser.add_argument("--frames", type=int, default=25, help="frame count")
    parser.add_argument("--fps", type=int, default=24, help="output fps")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--no-audio", action="store_true", help="disable audio generation")

    args = parser.parse_args()

    if args.width % 32 != 0:
        print(f"error: width must be divisible by 32 (got {args.width})")
        sys.exit(1)
    if args.height % 32 != 0:
        print(f"error: height must be divisible by 32 (got {args.height})")
        sys.exit(1)
    if (args.frames - 1) % 8 != 0:
        valid = [8*i + 1 for i in range(1, 13)]
        print(f"error: frames must be 8n+1 (valid: {valid})")
        sys.exit(1)

    if not torch.backends.mps.is_available():
        print("warning: mps not available, using cpu (slow)")
        device = "cpu"
    else:
        device = "mps"
        print("using mps")

    print("loading model...")
    pipe = LTX2Pipeline.from_pretrained(
        "Lightricks/LTX-2",
        torch_dtype=torch.bfloat16
    )
    pipe.to(device)
    print("model loaded")

    if args.seed is None:
        args.seed = torch.randint(0, 2**31, (1,)).item()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)

    print(f"\ngenerating{'...' if args.no_audio else ' with audio...'}")
    print(f"  prompt: {args.prompt}")
    print(f"  size: {args.width}x{args.height}, {args.frames} frames")
    print(f"  steps: {args.steps}, guidance: {args.guidance}")
    print(f"  seed: {args.seed}")
    print()

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

    # get video frames as tensor
    video_frames = result.frames[0]
    video_tensor = torch.stack([torch.from_numpy(np.array(f)) for f in video_frames])

    # get audio if available
    audio = None
    audio_sample_rate = None
    if not args.no_audio and result.audio is not None:
        audio = result.audio[0].float().cpu()
        audio_sample_rate = pipe.vocoder.config.output_sampling_rate
        print(f"audio generated ({audio_sample_rate}Hz)")

    # export with audio
    encode_video(
        video=video_tensor,
        fps=args.fps,
        audio=audio,
        audio_sample_rate=audio_sample_rate,
        output_path=args.output
    )

    print(f"\nsaved to: {args.output}")
    print(f"seed: {args.seed}")


if __name__ == "__main__":
    main()
