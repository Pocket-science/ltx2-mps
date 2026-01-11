# LTX-2 on Apple Silicon (MPS)

Run [Lightricks LTX-2](https://huggingface.co/Lightricks/LTX-2) video generation on Mac with Apple Silicon using Metal Performance Shaders (MPS).

## The Problem

LTX-2 uses `float64` (double precision) for rotary position embeddings (RoPE), but Apple's MPS backend doesn't support float64 - only float32. This causes the error:

```
TypeError: Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64
```

## The Solution

This repo provides a patch that forces `float32` for RoPE calculations. The quality difference is negligible, and it enables LTX-2 to run on Mac.

## Requirements

- **macOS** with Apple Silicon (M1, M2, M3, M4 - any variant)
- **Python 3.11+**
- **64GB+ RAM recommended** (model is ~40GB, 128GB ideal for max settings)
- **PyTorch 2.0+**

## Quick Start

```bash
# 1. Clone this repo
git clone https://github.com/YOUR_USERNAME/ltx2-mps.git
cd ltx2-mps

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install torch torchvision torchaudio
pip install git+https://github.com/huggingface/diffusers.git
pip install transformers accelerate safetensors sentencepiece
pip install imageio imageio-ffmpeg

# 4. Apply MPS patches
python patch_mps.py

# 5. Generate a video!
python generate.py "A cat walking through grass" -o output.mp4
```

## Usage

```bash
python generate.py "Your prompt here" -o output.mp4 [options]
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--width` | 512 | Video width (must be divisible by 32) |
| `--height` | 320 | Video height (must be divisible by 32) |
| `--frames` | 25 | Number of frames (must be 8n+1: 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97) |
| `--steps` | 20 | Inference steps (more = better quality, slower) |
| `--guidance` | 5.0 | Guidance scale |
| `--fps` | 24 | Output video FPS |
| `--seed` | random | Random seed for reproducibility |
| `-n` | "" | Negative prompt |

### Examples

```bash
# Quick preview (fast)
python generate.py "A sunset over mountains" -o preview.mp4 --frames 25 --steps 10 --width 512 --height 320

# Standard quality
python generate.py "A dog running on the beach" -o standard.mp4 --frames 49 --steps 20 --width 768 --height 448

# High quality (slow, needs 128GB RAM)
python generate.py "Cinematic shot of a forest" -o hq.mp4 --frames 97 --steps 30 --width 1024 --height 576
```

## Performance

Tested on Mac with M-series chips:

| Resolution | Frames | Steps | Time (approx) | RAM Usage |
|------------|--------|-------|---------------|-----------|
| 512x320 | 25 | 10 | ~1 min | ~45GB |
| 768x448 | 49 | 20 | ~10 min | ~60GB |
| 1024x576 | 97 | 30 | ~30 min | ~80GB |

## How the Patch Works

Two files in diffusers are patched:

### 1. `diffusers/pipelines/ltx2/connectors.py`
```python
# Before:
freqs_dtype = torch.float64 if self.double_precision else torch.float32

# After:
freqs_dtype = torch.float32  # MPS fix
```

### 2. `diffusers/models/transformers/transformer_ltx2.py`
```python
# Before:
freqs_dtype = torch.float64 if self.double_precision else torch.float32

# After:
freqs_dtype = torch.float32  # MPS fix
```

## Troubleshooting

### "MPS backend out of memory"
- Reduce resolution, frames, or close other apps
- Try `--width 512 --height 320 --frames 25`

### Model download fails
- Check your internet connection
- The model is ~40GB, first run takes a while to download

### Import errors
- Make sure you installed diffusers from git (dev version needed for LTX2Pipeline)
- Run `pip install git+https://github.com/huggingface/diffusers.git`

## Credits

- [Lightricks](https://github.com/Lightricks) for LTX-2
- [Hugging Face](https://github.com/huggingface/diffusers) for diffusers
- MPS patch discovered while debugging with Claude

## License

MIT
