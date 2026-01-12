# ltx2-mps

run [LTX-2](https://huggingface.co/Lightricks/LTX-2) video + audio generation on mac using MPS (metal).

## what's this about

LTX-2 uses float64 for rotary position embeddings, but MPS doesn't support float64. you get this error:

```
TypeError: Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64
```

this repo patches diffusers to use float32 instead. works fine, no noticeable quality loss.

## requirements

- mac with apple silicon (m1/m2/m3/m4)
- python 3.11+
- 64GB+ ram recommended (model is ~40GB)

## setup

```bash
git clone https://github.com/Pocket-science/ltx2-mps.git
cd ltx2-mps

python3 -m venv venv
source venv/bin/activate

pip install torch torchvision torchaudio
pip install git+https://github.com/huggingface/diffusers.git
pip install transformers accelerate safetensors sentencepiece
pip install imageio imageio-ffmpeg

python patch_mps.py
```

## usage

```bash
python generate.py "a cat walking through grass" -o output.mp4
```

### options

| flag | default | description |
|------|---------|-------------|
| `--width` | 512 | video width (divisible by 32) |
| `--height` | 320 | video height (divisible by 32) |
| `--frames` | 25 | frame count (must be 8n+1: 9, 17, 25, 33...) |
| `--steps` | 20 | inference steps |
| `--guidance` | 5.0 | guidance scale |
| `--fps` | 24 | output fps |
| `--seed` | random | seed for reproducibility |
| `-n` | "" | negative prompt |
| `--no-audio` | false | disable audio generation |

### examples

```bash
# quick test
python generate.py "sunset over mountains" -o test.mp4 --steps 10

# higher quality
python generate.py "dog running on beach" -o video.mp4 --frames 49 --steps 20 --width 768 --height 448

# max quality (needs 128GB ram, takes ~30 min)
python generate.py "cinematic forest shot" -o hq.mp4 --frames 97 --steps 30 --width 1024 --height 576
```

## performance

tested on m3 ultra:

| resolution | frames | steps | time |
|------------|--------|-------|------|
| 512x320 | 25 | 10 | ~1 min |
| 768x448 | 49 | 20 | ~10 min |
| 1024x576 | 97 | 30 | ~30 min |

## how the patch works

two files get patched in diffusers:

**diffusers/pipelines/ltx2/connectors.py**
```python
# before
freqs_dtype = torch.float64 if self.double_precision else torch.float32

# after
freqs_dtype = torch.float32
```

**diffusers/models/transformers/transformer_ltx2.py**
```python
# same change
freqs_dtype = torch.float32
```

## prompting guide

LTX-2 works best with detailed, flowing paragraph prompts rather than comma-separated tags. describe what happens in the video like you're writing a screenplay.

### prompt structure

write prompts as flowing paragraphs that include:

1. **scene setting** - location, time of day, weather
2. **camera work** - shot type, movement, framing
3. **subject action** - what's happening, how it moves
4. **visual style** - lighting, colors, atmosphere
5. **audio cues** - ambient sounds, music mood (LTX-2 generates audio too!)

### example prompts

**bad prompt:**
```
wolf, snow, forest, walking, cinematic
```

**good prompt:**
```
EXT. SNOWY FOREST - DUSK. A cinematic tracking shot follows a lone grey wolf
walking through deep powder snow between towering pine trees. The camera moves
alongside at eye level as soft blue twilight filters through the branches.
The wolf's breath is visible in the cold air, paws crunching softly in the snow.
Atmospheric and moody, shallow depth of field with gentle film grain.
```

### cinematography terms that work well

- **shot types:** wide establishing shot, medium shot, close-up, extreme close-up, overhead shot
- **camera movement:** tracking shot, dolly in/out, pan, crane up, handheld, steadicam
- **framing:** shallow depth of field, rack focus, silhouette, rule of thirds
- **lighting:** golden hour, blue hour, rim light, volumetric light, natural lighting
- **style:** cinematic, documentary style, film grain, anamorphic, photorealistic

### negative prompts

always include a negative prompt to avoid common issues:

```
blurry, low quality, distorted, deformed, ugly, bad anatomy, text, watermark, signature
```

if you're getting unwanted artistic styles, add:

```
cartoon, anime, illustration, painting, drawing, sketch, cgi, 3d render, digital art, stylized
```

## multi-scene films with image-to-video

LTX-2 supports image-to-video generation using `LTX2ImageToVideoPipeline`. you can create continuity between scenes by using the last frame of scene N as the input image for scene N+1.

### important warnings

- **style corruption can propagate** - if one scene produces artifacts or wrong style, it will affect all subsequent scenes
- **the prompt still applies** but the input image has strong influence on visual style
- **use higher guidance_scale (5.0+)** to give the prompt more weight over the image
- **if a scene goes wrong**, use the last frame from an earlier good scene instead

### example workflow

```python
from diffusers import LTX2Pipeline, LTX2ImageToVideoPipeline

# scene 1: text-to-video
t2v_pipe = LTX2Pipeline.from_pretrained("Lightricks/LTX-2", torch_dtype=torch.bfloat16)
result1 = t2v_pipe(prompt="...", guidance_scale=4.0, ...)
last_frame = result1.frames[0][-1]

# scene 2+: image-to-video for continuity
i2v_pipe = LTX2ImageToVideoPipeline.from_pretrained("Lightricks/LTX-2", torch_dtype=torch.bfloat16)
result2 = i2v_pipe(
    image=last_frame,
    prompt="...",  # prompt still matters!
    guidance_scale=5.0,  # higher to enforce prompt style
    ...
)
```

## distilled model warning

there's a distilled version available (`blanchon/LTX-2-Distilled-diffusers`) that promises faster generation with fewer steps.

**do not use it for production** - in our testing it produces severe artifacts, cartoon-style corruption, and generally unusable output. stick with the full `Lightricks/LTX-2` model.

## troubleshooting

**out of memory** - reduce resolution/frames or close other apps

**model download fails** - it's ~40GB, first run takes a while

**import errors** - make sure you installed diffusers from git, not pip

**cartoon/artistic style when you wanted photorealistic:**
- add "photorealistic, cinematic film look, real world footage" to your prompt
- add "cartoon, anime, illustration, painting, drawing" to negative prompt
- increase guidance_scale to 5.0 or higher
- if using image-to-video, check if the input image has style issues

**scene continuity problems in multi-scene films:**
- check each scene individually before combining
- if a scene has artifacts, regenerate it with text-to-video or use a different input frame
- style corruption from bad frames propagates to all subsequent scenes

## credits

- [lightricks](https://github.com/Lightricks) for ltx-2
- [@ivanfioravanti](https://twitter.com/ivanfioravanti) for the mps fix approach
- [huggingface](https://github.com/huggingface/diffusers) for diffusers

## license

MIT
