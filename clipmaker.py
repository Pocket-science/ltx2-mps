#!/usr/bin/env python3
"""
clipmaker - high quality video clip generator for ltx-2

usage:
    clipmaker "your prompt here"                    # quick preview
    clipmaker "your prompt" --preset hq             # high quality
    clipmaker "your prompt" --preset max            # maximum quality
    clipmaker --batch prompts.txt                   # batch from file
    clipmaker --interactive                         # interactive mode
"""

import argparse
import os
import sys
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# quality presets
PRESETS = {
    "preview": {
        "width": 512,
        "height": 320,
        "frames": 25,
        "steps": 10,
        "guidance": 4.0,
        "description": "fast preview (~1 min)"
    },
    "standard": {
        "width": 768,
        "height": 448,
        "frames": 49,
        "steps": 20,
        "guidance": 4.0,
        "description": "balanced quality (~5 min)"
    },
    "hq": {
        "width": 1024,
        "height": 576,
        "frames": 97,
        "steps": 25,
        "guidance": 4.0,
        "description": "high quality (~15 min)"
    },
    "max": {
        "width": 1024,
        "height": 576,
        "frames": 161,
        "steps": 30,
        "guidance": 4.0,
        "description": "maximum quality (~30 min)"
    },
    "cinematic": {
        "width": 1280,
        "height": 720,
        "frames": 97,
        "steps": 30,
        "guidance": 4.5,
        "description": "cinematic 720p (~25 min)"
    },
}

# default negative prompt based on ltx-2 guide
DEFAULT_NEGATIVE = "blurry, low quality, distorted, deformed, ugly, bad anatomy, text, watermark, signature, out of frame"

# prompt enhancement tips
PROMPT_TIPS = """
prompt tips (from ltx-2 guide):
  - write as flowing paragraph, 4-8 sentences
  - include: shot type, lighting, action, camera movement, audio
  - use cinematography terms: dolly, pan, track, handheld, close-up
  - describe sounds and dialogue in "quotes"
  - use present tense for actions

example:
  "A cinematic medium shot of a coffee cup on a wooden table, steam rising
   gently in soft morning light. The camera slowly pushes in as a hand
   reaches into frame to lift the cup. Warm ambient cafe sounds and soft
   jazz play in the background. Shallow depth of field, golden hour lighting."
"""


class ClipMaker:
    def __init__(self, output_dir="~/Desktop/clips"):
        self.output_dir = Path(output_dir).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pipe = None
        self.device = None

    def load_model(self):
        """load the ltx-2 pipeline"""
        if self.pipe is not None:
            return

        from diffusers import LTX2Pipeline

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"device: {self.device}")

        print("loading ltx-2 model...")
        self.pipe = LTX2Pipeline.from_pretrained(
            "Lightricks/LTX-2",
            torch_dtype=torch.bfloat16
        )
        self.pipe.to(self.device)
        print("model ready\n")

    def generate(self, prompt, preset="standard", negative_prompt=None,
                 seed=None, output_path=None, no_audio=False):
        """generate a video clip"""
        from diffusers.pipelines.ltx2.export_utils import encode_video

        self.load_model()

        # get preset settings
        if preset not in PRESETS:
            print(f"unknown preset: {preset}")
            print(f"available: {', '.join(PRESETS.keys())}")
            return None

        settings = PRESETS[preset]

        # generate output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"clip_{timestamp}.mp4"
        else:
            output_path = Path(output_path)

        # set seed
        if seed is None:
            seed = torch.randint(0, 2**31, (1,)).item()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

        # use default negative if not provided
        if negative_prompt is None:
            negative_prompt = DEFAULT_NEGATIVE

        print(f"{'='*60}")
        print(f"generating clip")
        print(f"{'='*60}")
        print(f"preset: {preset} ({settings['description']})")
        print(f"size: {settings['width']}x{settings['height']}")
        print(f"frames: {settings['frames']} (~{settings['frames']/24:.1f}s)")
        print(f"steps: {settings['steps']}")
        print(f"seed: {seed}")
        print(f"audio: {'no' if no_audio else 'yes'}")
        print(f"output: {output_path}")
        print(f"\nprompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        print(f"{'='*60}\n")

        # generate
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=settings["width"],
            height=settings["height"],
            num_frames=settings["frames"],
            num_inference_steps=settings["steps"],
            guidance_scale=settings["guidance"],
            generator=generator,
        )

        # get video frames as tensor
        video_frames = result.frames[0]
        video_tensor = torch.stack([torch.from_numpy(np.array(f)) for f in video_frames])

        # get audio
        audio = None
        audio_sr = None
        if not no_audio and result.audio is not None:
            audio = result.audio[0].float().cpu()
            audio_sr = self.pipe.vocoder.config.output_sampling_rate
            print(f"audio: {audio_sr}Hz")

        # export
        encode_video(
            video=video_tensor,
            fps=24,
            audio=audio,
            audio_sample_rate=audio_sr,
            output_path=str(output_path)
        )

        # save metadata
        meta_path = output_path.with_suffix(".json")
        metadata = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "preset": preset,
            "settings": settings,
            "seed": seed,
            "timestamp": datetime.now().isoformat(),
            "output": str(output_path),
        }
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\n{'='*60}")
        print(f"done!")
        print(f"video: {output_path}")
        print(f"metadata: {meta_path}")
        print(f"seed: {seed} (use --seed {seed} to reproduce)")
        print(f"{'='*60}\n")

        return output_path

    def batch_generate(self, prompts_file, preset="standard"):
        """generate multiple clips from a file"""
        prompts_path = Path(prompts_file)
        if not prompts_path.exists():
            print(f"file not found: {prompts_file}")
            return

        prompts = []
        with open(prompts_path) as f:
            current_prompt = []
            for line in f:
                line = line.strip()
                if line == "---":  # separator between prompts
                    if current_prompt:
                        prompts.append(" ".join(current_prompt))
                        current_prompt = []
                elif line and not line.startswith("#"):  # skip comments
                    current_prompt.append(line)
            if current_prompt:
                prompts.append(" ".join(current_prompt))

        print(f"found {len(prompts)} prompts in {prompts_file}")
        print(f"preset: {preset}")
        print()

        for i, prompt in enumerate(prompts):
            print(f"\n[{i+1}/{len(prompts)}]")
            self.generate(prompt, preset=preset)

    def interactive(self):
        """interactive prompt mode"""
        print("\n" + "="*60)
        print("clipmaker interactive mode")
        print("="*60)
        print(PROMPT_TIPS)
        print("\npresets:", ", ".join(PRESETS.keys()))
        print("commands: /preset <name>, /tips, /quit\n")

        current_preset = "standard"

        while True:
            try:
                prompt = input(f"[{current_preset}] > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nbye!")
                break

            if not prompt:
                continue
            elif prompt == "/quit":
                print("bye!")
                break
            elif prompt == "/tips":
                print(PROMPT_TIPS)
            elif prompt.startswith("/preset"):
                parts = prompt.split()
                if len(parts) > 1 and parts[1] in PRESETS:
                    current_preset = parts[1]
                    print(f"preset: {current_preset} - {PRESETS[current_preset]['description']}")
                else:
                    print(f"presets: {', '.join(PRESETS.keys())}")
            else:
                self.generate(prompt, preset=current_preset)


def main():
    parser = argparse.ArgumentParser(
        description="clipmaker - hq video clip generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
presets:
  preview   - {PRESETS['preview']['description']}
  standard  - {PRESETS['standard']['description']}
  hq        - {PRESETS['hq']['description']}
  max       - {PRESETS['max']['description']}
  cinematic - {PRESETS['cinematic']['description']}

examples:
  clipmaker "a cat sleeping on a couch"
  clipmaker "epic sunset timelapse" --preset hq
  clipmaker --batch prompts.txt --preset standard
  clipmaker --interactive
        """
    )

    parser.add_argument("prompt", nargs="?", help="video prompt")
    parser.add_argument("--preset", "-p", default="standard",
                        choices=PRESETS.keys(), help="quality preset")
    parser.add_argument("--output", "-o", help="output path")
    parser.add_argument("--seed", "-s", type=int, help="random seed")
    parser.add_argument("--negative", "-n", help="negative prompt")
    parser.add_argument("--no-audio", action="store_true", help="disable audio")
    parser.add_argument("--batch", "-b", help="batch generate from file")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="interactive mode")
    parser.add_argument("--output-dir", default="~/Desktop/clips",
                        help="output directory")
    parser.add_argument("--tips", action="store_true", help="show prompt tips")

    args = parser.parse_args()

    if args.tips:
        print(PROMPT_TIPS)
        return

    maker = ClipMaker(output_dir=args.output_dir)

    if args.interactive:
        maker.interactive()
    elif args.batch:
        maker.batch_generate(args.batch, preset=args.preset)
    elif args.prompt:
        maker.generate(
            prompt=args.prompt,
            preset=args.preset,
            negative_prompt=args.negative,
            seed=args.seed,
            output_path=args.output,
            no_audio=args.no_audio,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
