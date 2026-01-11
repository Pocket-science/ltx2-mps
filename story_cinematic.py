#!/usr/bin/env python3
"""
the mountain guardian - a cinematic short film
generated with ltx-2 using proper prompting techniques
"""

import os
import numpy as np
import torch
from PIL import Image
from diffusers import LTX2Pipeline, LTX2ImageToVideoPipeline
from diffusers.pipelines.ltx2.export_utils import encode_video

# cinematic story prompts following ltx-2 prompting guide
SCENES = [
    # scene 1: opening - sunrise
    """EXT. SWISS ALPS - DAWN. A cinematic wide establishing shot of snow-covered mountain peaks as golden sunrise light spills over the ridgeline. The camera slowly pans right, revealing a vast alpine valley blanketed in fresh powder snow. Wisps of morning mist drift between the pine trees below. The warm orange glow gradually illuminates the pristine white landscape. Ambient sounds of gentle wind and distant bird calls fill the air. The shot lingers on the majestic scenery, peaceful and untouched.""",

    # scene 2: hero introduction
    """EXT. MOUNTAIN RIDGE - DAWN. The camera pushes in slowly on a magnificent white Swiss shepherd dog standing proudly on a snowy ridge, silhouetted against the golden sunrise. The dog's thick fur ruffles gently in the cold mountain breeze, breath visible in the frigid air. Medium shot, shallow depth of field with the valley soft in the background. The shepherd surveys the landscape below with alert, intelligent eyes, ears perked forward. Soft ambient wind and the dog's quiet breathing create an intimate atmosphere. Cinematic warm backlighting creates a heroic golden rim around the dog's form.""",

    # scene 3: the patrol begins
    """EXT. SNOWY FOREST - MORNING. Tracking shot following the white Swiss shepherd as it walks purposefully through deep powder snow, each step sending up small puffs of white. Pine trees tower on either side, their branches heavy with snow. The camera moves alongside the dog at eye level, handheld style with subtle movement. Morning light filters through the forest canopy in soft rays. The sound of snow crunching under paws and the dog's steady breathing. The shepherd moves with determination, nose low, following an invisible trail through the wilderness.""",

    # scene 4: something's wrong
    """EXT. FOREST CLEARING - MORNING. The Swiss shepherd stops abruptly mid-stride, head snapping to the right, ears rotating forward. Close-up on the dog's face showing intense focus, nostrils flaring as it catches a scent. The camera slowly pushes in on the shepherd's alert expression. A beat of tense silence, then a faint, distant bleating sound echoes through the trees. The dog's eyes widen slightly with recognition. The ambient forest sounds fade as the shepherd locks onto the direction of the cry. Shallow depth of field isolates the dog's concentrated expression.""",

    # scene 5: discovery
    """EXT. SNOWY HOLLOW - MORNING. Wide shot revealing a small white lamb huddled alone in a depression in the snow, shivering visibly, its wool matted and wet. The lamb lets out weak, frightened bleats, breath coming in short visible puffs. The camera slowly dollies forward, keeping the vulnerable creature centered. Snow continues to fall gently around it. In the background, barely visible through the snow, the Swiss shepherd appears at the edge of the clearing. Soft, melancholic ambient tones underscore the lamb's distress. The scene conveys isolation and vulnerability.""",

    # scene 6: gentle approach
    """EXT. SNOWY HOLLOW - MORNING. Medium shot as the Swiss shepherd approaches the frightened lamb with slow, deliberate steps, body low and non-threatening. The lamb looks up with wide, fearful eyes but doesn't flee. The dog pauses, then takes another careful step forward. The camera tracks alongside at ground level. Soft snow crunches beneath careful paws. The shepherd's expression is gentle, reassuring. Warm morning light breaks through the clouds above. The tension slowly dissolves as the lamb recognizes help has arrived. Ambient sounds of gentle wind and soft animal breathing.""",

    # scene 7: comfort
    """EXT. SNOWY HOLLOW - MORNING. Close-up intimate shot as the Swiss shepherd gently nuzzles the shivering lamb, warm breath creating a soft cloud between them. The lamb presses against the dog's thick fur, seeking warmth. The camera holds on this tender moment, shallow depth of field blurring the snowy background. The shepherd's eyes close briefly in a gesture of comfort. Soft, warm lighting wraps around both animals. The lamb's frightened bleating quiets to soft sounds of relief. A moment of connection between two creatures in the vast wilderness. Heartwarming and genuine.""",

    # scene 8: the journey begins
    """EXT. ALPINE MEADOW - MIDDAY. Wide cinematic shot as the Swiss shepherd leads the small lamb across a vast snowy meadow, the dog walking protectively alongside its small companion. Mountain peaks rise majestically in the background under a pale blue sky. The camera slowly cranes up to reveal the epic scale of their journey ahead. Both animals leave a trail of footprints in the pristine snow. Soft orchestral tones suggest hope and determination. The shepherd occasionally glances back to check on the lamb, who follows trustingly. Golden sunlight illuminates the pair as they traverse the white expanse.""",

    # scene 9: village in sight
    """EXT. MOUNTAIN OVERLOOK - LATE AFTERNOON. The camera pushes forward as the Swiss shepherd and lamb crest a snowy hill, revealing a picturesque Swiss village nestled in the valley below. Warm lights glow from windows of wooden chalets, smoke rising from chimneys into the golden hour sky. The shepherd pauses, tail wagging slightly at the sight. The lamb stands close beside, tired but hopeful. A sense of relief and accomplishment fills the frame. Church bells chime faintly in the distance. The camera slowly zooms toward the welcoming village as the sun sets behind the mountains.""",

    # scene 10: finale
    """EXT. VILLAGE BARN - DUSK. Medium shot as a farmer in traditional Swiss clothing opens a wooden barn door, warm golden light spilling out into the blue twilight. His weathered face shows surprise, then breaks into a warm smile as he sees the shepherd with the lost lamb. He kneels down, arms open. The lamb bounds forward into the warm barn interior where other sheep can be seen. The farmer reaches out to pat the shepherd's head gratefully, saying softly "Good dog... good dog." The shepherd sits proudly, breath visible in the cold air, mission complete. Warm interior light contrasts with the cold blue dusk outside.""",
]

def main():
    output_dir = os.path.expanduser("~/Desktop/mountain_guardian")
    os.makedirs(output_dir, exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"using {device}")
    print(f"\n{'='*60}")
    print("THE MOUNTAIN GUARDIAN")
    print("a cinematic short film")
    print(f"{'='*60}\n")

    # load pipelines
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

    # settings - 768x448, 97 frames (~4 sec per scene)
    width, height = 768, 448
    frames = 97
    steps = 25  # higher for better quality

    all_frames = []
    all_audio = []
    last_frame = None

    for i, prompt in enumerate(SCENES):
        print(f"\n{'='*60}")
        print(f"SCENE {i+1}/{len(SCENES)}")
        print(f"{'='*60}")
        # show first 100 chars of prompt
        print(f"{prompt[:100]}...")
        print()

        neg_prompt = "blurry, low quality, distorted, deformed, ugly, bad anatomy, text, watermark, signature"

        if i == 0 or last_frame is None:
            # first scene: text-to-video
            result = t2v_pipe(
                prompt=prompt,
                negative_prompt=neg_prompt,
                width=width,
                height=height,
                num_frames=frames,
                num_inference_steps=steps,
                guidance_scale=4.0,
            )
        else:
            # subsequent scenes: image-to-video for continuity
            result = i2v_pipe(
                image=last_frame,
                prompt=prompt,
                negative_prompt=neg_prompt,
                width=width,
                height=height,
                num_frames=frames,
                num_inference_steps=steps,
                guidance_scale=4.0,
            )

        video_frames = result.frames[0]
        last_frame = video_frames[-1]

        # collect frames (skip first for scenes 2+ to avoid duplicate)
        if i == 0:
            all_frames.extend(video_frames)
        else:
            all_frames.extend(video_frames[1:])

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

    # create full film
    print(f"\n{'='*60}")
    print("ASSEMBLING FINAL FILM...")
    print(f"{'='*60}\n")

    full_path = os.path.join(output_dir, "the_mountain_guardian.mp4")
    video_tensor = torch.stack([torch.from_numpy(np.array(f)) for f in all_frames])

    if all_audio:
        full_audio = torch.cat(all_audio, dim=-1).float().cpu()
        audio_sr = t2v_pipe.vocoder.config.output_sampling_rate
    else:
        full_audio = None
        audio_sr = None

    encode_video(video_tensor, fps=24, audio=full_audio, audio_sample_rate=audio_sr, output_path=full_path)

    duration = len(all_frames) / 24
    print(f"\n{'='*60}")
    print("PRODUCTION COMPLETE")
    print(f"{'='*60}")
    print(f"total frames: {len(all_frames)}")
    print(f"duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    print(f"output: {full_path}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
