# PPGS watermarking

This is a compact reference implementation of **Proportion-Preserved Gaussian
Shading (PPGS)** from *Towards provenance-aware diffusion: Key-free
watermarking with Gaussian shading* (Wang and Tian, 2026).

Implemented components:

- Algorithm 1: public permutation, proportion-aware interval construction,
  and inverse-normal sampling.
- Algorithm 2: exponential soft-guidance during deterministic DDIM sampling.
- Algorithm 3: VAE encoding, null-prompt DDIM inversion, interval decoding,
  inverse permutation, and verification.
- Linear and cosine schedules from the guidance ablation.
- The 256-bit experimental payload interpretation via repeat-to-capacity
  embedding and majority-vote recovery.

## Install and test

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

For Stable Diffusion integration:

```bash
python -m pip install -e ".[diffusion]"
```

## Latent-only use

```python
import numpy as np
from ppgs_watermark import decode_latents, embed_watermark

watermark = np.random.default_rng(0).integers(0, 2, 256)
latent, metadata = embed_watermark(
    watermark,
    (1, 4, 64, 64),
    public_seed=2026,
    sampling_seed=42,
    repeat_payload=True,
)
recovered = decode_latents(latent, metadata)
assert np.array_equal(recovered, watermark)
```

`metadata` is public provenance data, not a secret. Extraction requires the
bit proportion (`gamma`), payload layout, and public permutation seed.

## Diffusers use

```python
import torch
from diffusers import StableDiffusionPipeline
from ppgs_watermark.diffusers_pipeline import PPGSDiffusers

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16
).to("cuda")
ppgs = PPGSDiffusers(pipe)

bits = [0, 1] * 128
result = ppgs.generate(
    "a studio photograph of a red fox",
    bits,
    num_inference_steps=50,
    maximum_guidance=7.5,
    guidance_decay=2.0,
    minimum_guidance=0.1,
    repeat_payload=True,
)
result.images[0].save("outputs/watermarked.png")

inverted = ppgs.invert(result.images[0], num_inference_steps=10)
accepted, accuracy, recovered = ppgs.verify(
    inverted, bits, result.metadata, threshold=0.9
)
```

The paper evaluates SD-v1.4, SD-v2.0, and SD-v2.1 at 512x512, with latent
shape 4x64x64, one bit per position, 50 DDIM generation steps, CFG 7.5, and
10 unconditional inversion steps. Attacks include JPEG quality 50, resizing,
Gaussian noise sigma 0.05, blur, crop up to 10%, rotation up to 15 degrees,
color shifts, and a composite pipeline.

## Reproduction notes

The paper contains implementation ambiguities that this code exposes:

1. Algorithm 1 requires `l=m*c*h*w`, while the experiments state a 256-bit
   payload for a 4x64x64 latent with `m=1`. `repeat_payload=True` tiles those
   bits over 16,384 positions and uses majority voting during recovery.
2. Equation (25) has `g_t = g_max exp(-decay*(T-t)/T)` without `g_min`, while
   Table 2 reports `g_min=0.1`. The code follows the equation and offers an
   explicit optional minimum clamp.
3. Equation (14) prints the zero-bit exponent as `l-k_i`; for an m-bit symbol
   it must be `m-k_i` for probabilities to normalize.

Latent-only tests do not download weights. A full image round trip requires a
compatible Stable Diffusion checkpoint, GPU memory, and optional dependencies.
