#!/usr/bin/env python3
"""
generate a multi-scene story video with ltx-2
"""

import os
import subprocess
import sys

# story scenes - each generates ~4 seconds
SCENES = [
    # Act 1: Introduction
    {
        "prompt": "Sunrise over the Swiss Alps, snow-covered peaks glowing orange and pink, peaceful morning atmosphere, cinematic wide shot, gentle wind blowing snow, 4K quality",
        "name": "01_sunrise"
    },
    {
        "prompt": "A majestic white Swiss shepherd dog standing on a snowy ridge overlooking a mountain valley, morning light, heroic pose, wind ruffling fur, cinematic portrait shot",
        "name": "02_hero_intro"
    },
    {
        "prompt": "Close-up of the Swiss shepherd's face, alert eyes scanning the horizon, breath visible in cold air, morning sunlight on fur, shallow depth of field",
        "name": "03_closeup"
    },
    {
        "prompt": "The Swiss shepherd walking through deep powder snow in the Alps, determined stride, pine trees in background, snow particles in air, tracking shot",
        "name": "04_walking"
    },

    # Act 2: Discovery
    {
        "prompt": "The Swiss shepherd stops suddenly, ears perked up, alert pose, listening intently, snowy forest background, dramatic lighting, tension building",
        "name": "05_alert"
    },
    {
        "prompt": "A small white lamb alone in the snow, shivering, lost and scared, soft snowfall, vulnerable, wide snowy landscape, emotional scene",
        "name": "06_lost_lamb"
    },
    {
        "prompt": "The Swiss shepherd approaching the lamb gently, careful steps through snow, compassionate body language, soft winter light, heartwarming moment",
        "name": "07_approach"
    },
    {
        "prompt": "Close-up of the shepherd dog nuzzling the scared lamb, comforting gesture, warm breath visible, tender moment, shallow depth of field, emotional",
        "name": "08_comfort"
    },

    # Act 3: Journey home
    {
        "prompt": "The Swiss shepherd leading the small lamb through snowy alpine meadow, protective stance, walking together, mountains in background, golden hour light",
        "name": "09_leading"
    },
    {
        "prompt": "Wide shot of dog and lamb crossing a snowy hill together, tiny figures in vast white landscape, beautiful alpine scenery, cinematic composition",
        "name": "10_journey"
    },
    {
        "prompt": "The shepherd and lamb walking past snow-covered pine trees, gentle snowfall, peaceful atmosphere, soft afternoon light filtering through branches",
        "name": "11_forest_path"
    },
    {
        "prompt": "A cozy Swiss mountain village appearing in the distance, warm lights glowing from windows, smoke from chimneys, dusk setting in, hopeful atmosphere",
        "name": "12_village_sight"
    },

    # Act 4: Reunion
    {
        "prompt": "The Swiss shepherd and lamb arriving at a wooden barn door, warm light spilling out, welcoming atmosphere, end of journey, relief",
        "name": "13_arrival"
    },
    {
        "prompt": "A farmer in traditional Swiss clothing opening the barn door, surprised and grateful expression, warm interior light, emotional reunion moment",
        "name": "14_farmer"
    },
    {
        "prompt": "The lamb running to join other sheep in a warm barn, happy reunion, straw on floor, cozy interior, heartwarming resolution",
        "name": "15_reunion"
    },
    {
        "prompt": "The Swiss shepherd sitting proudly outside the barn, farmer patting its head gratefully, twilight sky, village lights twinkling, satisfied hero",
        "name": "16_reward"
    },

    # Finale
    {
        "prompt": "Night sky over the Swiss Alps with stars and northern lights, the shepherd dog silhouette on a ridge, majestic ending, peaceful, cinematic wide shot",
        "name": "17_finale"
    },
]

def generate_scene(scene, output_dir, width=768, height=448, frames=97, steps=20):
    """generate a single scene"""
    output_path = os.path.join(output_dir, f"{scene['name']}.mp4")

    if os.path.exists(output_path):
        print(f"skipping {scene['name']} (already exists)")
        return output_path

    cmd = [
        sys.executable, "generate.py",
        scene["prompt"],
        "-o", output_path,
        "--width", str(width),
        "--height", str(height),
        "--frames", str(frames),
        "--steps", str(steps),
        "-n", "blurry, low quality, distorted, deformed, ugly, bad anatomy"
    ]

    print(f"\n{'='*60}")
    print(f"generating: {scene['name']}")
    print(f"prompt: {scene['prompt'][:80]}...")
    print(f"{'='*60}\n")

    subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    return output_path

def concatenate_videos(video_files, output_path):
    """concatenate all videos using ffmpeg"""
    # create file list
    list_path = "/tmp/video_list.txt"
    with open(list_path, "w") as f:
        for video in video_files:
            f.write(f"file '{video}'\n")

    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", list_path,
        "-c:v", "libx264", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        output_path
    ]

    print(f"\nconcatenating {len(video_files)} clips...")
    subprocess.run(cmd)
    print(f"saved to: {output_path}")

def main():
    output_dir = os.path.expanduser("~/Desktop/mountain_guardian")
    os.makedirs(output_dir, exist_ok=True)

    print(f"generating {len(SCENES)} scenes for 'The Mountain Guardian'")
    print(f"output directory: {output_dir}")
    print(f"estimated time: ~{len(SCENES) * 10} minutes\n")

    video_files = []
    for i, scene in enumerate(SCENES):
        print(f"\n[{i+1}/{len(SCENES)}]")
        video_path = generate_scene(scene, output_dir)
        video_files.append(video_path)

    # concatenate all videos
    final_path = os.path.join(output_dir, "mountain_guardian_full.mp4")
    concatenate_videos(video_files, final_path)

    print(f"\n{'='*60}")
    print(f"done! final video: {final_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
