#!/usr/bin/env python3
"""
generate a continuous story video with ltx-2
uses image-to-video to maintain visual continuity between scenes
"""

import os
import subprocess
import numpy as np
import torch
from PIL import Image
from diffusers import LTX2Pipeline, LTX2ImageToVideoPipeline
from diffusers.pipelines.ltx2.export_utils import encode_video

# story scenes
SCENES = [
    "Sunrise over the Swiss Alps, snow-covered peaks glowing orange and pink, a white Swiss shepherd dog standing on a ridge, peaceful morning, cinematic wide shot",
    "The Swiss shepherd walking through deep powder snow, determined stride, pine trees, snow particles in air, morning light, tracking shot following the dog",
    "The shepherd stops suddenly, ears perked, alert pose, listening intently, snowy forest, something caught its attention",
    "A small white lamb alone in the snow, shivering, lost and scared, the shepherd approaches gently in the background",
    "The Swiss shepherd nuzzling the scared lamb, comforting gesture, warm breath visible, tender moment, shallow depth of field",
    "The shepherd leading the lamb through snowy alpine meadow, protective stance, walking together, mountains in background, golden hour",
    "Wide shot of dog and lamb crossing a snowy hill, vast white landscape, beautiful alpine scenery, afternoon light",
    "A cozy Swiss mountain village appearing in the distance, warm lights glowing, smoke from chimneys, dusk, hopeful atmosphere",
    "The shepherd and lamb arriving at a wooden barn, warm light spilling out, welcoming atmosphere, journey's end",
    "Night sky over the Alps with stars, the shepherd dog silhouette on a ridge, majestic ending, peaceful, cinematic finale",
]

def main():
    output_dir = os.path.expanduser("~/Desktop/mountain_guardian")
    os.makedirs(output_dir, exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"using {device}")

    # load both pipelines
    print("loading text-to-video pipeline...")
    t2v_pipe = LTX2Pipeline.from_pretrained(
        "Lightricks/LTX-2",
        torch_dtype=torch.bfloat16
    )
    t2v_pipe.to(device)

    print("loading image-to-video pipeline...")
    i2v_pipe = LTX2ImageToVideoPipeline.from_pretrained(
        "Lightricks/LTX-2",
        torch_dtype=torch.bfloat16
    )
    i2v_pipe.to(device)

    width, height = 768, 448
    frames = 97  # ~4 seconds per scene
    steps = 20

    all_video_frames = []
    all_audio = []
    last_frame = None

    for i, prompt in enumerate(SCENES):
        print(f"\n{'='*60}")
        print(f"scene {i+1}/{len(SCENES)}")
        print(f"prompt: {prompt[:60]}...")
        print(f"{'='*60}\n")

        if i == 0:
            # first scene: text-to-video
            result = t2v_pipe(
                prompt=prompt,
                negative_prompt="blurry, low quality, distorted, deformed",
                width=width,
                height=height,
                num_frames=frames,
                num_inference_steps=steps,
            )
        else:
            # subsequent scenes: image-to-video from last frame
            result = i2v_pipe(
                image=last_frame,
                prompt=prompt,
                negative_prompt="blurry, low quality, distorted, deformed",
                width=width,
                height=height,
                num_frames=frames,
                num_inference_steps=steps,
            )

        # get frames
        video_frames = result.frames[0]

        # save last frame for next scene
        last_frame = video_frames[-1]

        # collect frames (skip first frame for scenes 2+ to avoid duplicate)
        if i == 0:
            all_video_frames.extend(video_frames)
        else:
            all_video_frames.extend(video_frames[1:])  # skip first frame (duplicate of last)

        # collect audio
        if result.audio is not None:
            all_audio.append(result.audio[0])

        # save individual scene
        scene_path = os.path.join(output_dir, f"scene_{i+1:02d}.mp4")
        video_tensor = torch.stack([torch.from_numpy(np.array(f)) for f in video_frames])
        audio = result.audio[0].float().cpu() if result.audio is not None else None
        audio_sr = t2v_pipe.vocoder.config.output_sampling_rate if audio is not None else None
        encode_video(video_tensor, fps=24, audio=audio, audio_sample_rate=audio_sr, output_path=scene_path)
        print(f"saved: {scene_path}")

    # save full video
    print("\ncreating full video...")
    full_path = os.path.join(output_dir, "mountain_guardian_full.mp4")
    video_tensor = torch.stack([torch.from_numpy(np.array(f)) for f in all_video_frames])

    # concatenate audio
    if all_audio:
        full_audio = torch.cat(all_audio, dim=-1).float().cpu()
        audio_sr = t2v_pipe.vocoder.config.output_sampling_rate
    else:
        full_audio = None
        audio_sr = None

    encode_video(video_tensor, fps=24, audio=full_audio, audio_sample_rate=audio_sr, output_path=full_path)

    print(f"\n{'='*60}")
    print(f"done!")
    print(f"total frames: {len(all_video_frames)}")
    print(f"duration: ~{len(all_video_frames)/24:.1f} seconds")
    print(f"saved to: {full_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
