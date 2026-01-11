# ltx2-mps

run [LTX-2](https://huggingface.co/Lightricks/LTX-2) video generation on mac using MPS (metal).

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

## troubleshooting

**out of memory** - reduce resolution/frames or close other apps

**model download fails** - it's ~40GB, first run takes a while

**import errors** - make sure you installed diffusers from git, not pip

## credits

- [lightricks](https://github.com/Lightricks) for ltx-2
- [@ivanfioravanti](https://twitter.com/ivanfioravanti) for the mps fix approach
- [huggingface](https://github.com/huggingface/diffusers) for diffusers

## license

MIT
