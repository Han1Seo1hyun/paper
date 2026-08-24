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
- The 256-bit Gaussian-Shading payload interpretation via a 4x8x8 payload
  grid, 8x8 spatial tiling, and block-wise majority-vote recovery.
- Public JSON manifests containing the extraction metadata and generation
  settings (no secret key material).
- Section 4.1 attacks: JPEG-50, random rescaling, Gaussian noise (sigma 0.05),
  blur, crop (up to 10% area), rotation (up to 15 degrees), color shift, and a
  deterministic composite pipeline.
- Bit accuracy, binomial FPR threshold, normalized inversion error, and
  Gaussian mean/std/KS diagnostics.
- Resumable SD-v1.4/v2.0/v2.1 experiment matrix and guidance/proportion
  ablations.
- Inception-feature Fréchet distance, OpenCLIP score, paired t statistics,
  exact binomial detection curves, and user-scale attribution.
- Executable keyed Gaussian Shading and Tree-Ring latent baselines, plus the
  official DwtDctSvd/RivaGAN image encoders at a pinned upstream commit.

## Install and test

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

For Stable Diffusion integration:

```bash
python -m pip install -e ".[diffusion]"
```

The validated Windows/CUDA 12.6 environment can instead be installed with:

```bash
python -m pip install -r requirements-paper-cu126.txt
```

This project was smoke-tested with PyTorch 2.10.0+cu126, Diffusers 0.40,
Transformers 5.15, and Accelerate 1.14 on an RTX 2060 6 GB GPU.

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
    payload_layout="spatial_tile",
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
    guidance_schedule="exponential",
    payload_layout="spatial_tile",
)
result.images[0].save("outputs/watermarked.png")
result.save_manifest("outputs/watermarked.json")

inverted = ppgs.invert(
    result.images[0], num_inference_steps=10, metadata=result.metadata
)
accepted, accuracy, recovered = ppgs.verify(
    inverted, bits, result.metadata, threshold=0.9
)
```

The paper evaluates SD-v1.4, SD-v2.0, and SD-v2.1 at 512x512, with latent
shape 4x64x64, one bit per position, 50 DDIM generation steps, CFG 7.5, and
10 unconditional inversion steps. Attacks include JPEG quality 50, resizing,
Gaussian noise sigma 0.05, blur, crop up to 10%, rotation up to 15 degrees,
color shifts, and a composite pipeline.

The canonical Stability AI SD 2.x repositories currently require an
authenticated Hugging Face session. `configs/paper.json` therefore keeps the
canonical names for reporting but maps downloads to public `sd-research`
fp16 mirrors. Every `run.json` records both `model` and `model_source`.

## Paper experiment loop

The checked-in configuration mirrors Section 4.1. Fetch the public prompt
split and run the three-model matrix with paired watermark-free images:

```bash
python experiments/fetch_prompts.py --count 1000
python experiments/run_matrix.py --limit 50 --quality-pairs
python experiments/analyze_results.py
python experiments/verify_run.py \
  --run-dir outputs/paper/main/compvis-stable-diffusion-v1-4 \
  --expected-prompts 50
```

Every generation and attack record is written immediately. A stopped job can
be continued with the same command because the matrix passes `--resume`.
Use `--include-ablations` to additionally execute exponential, linear, cosine,
literal-equation guidance, and payload-proportion runs.

The paper's 1,000-image detection curves can be generated without retaining
thousands of intermediate PNGs:

```bash
python experiments/run_matrix.py --limit 1000 --artifacts metrics
python experiments/analyze_results.py
```

The runner saves watermarked, unwatermarked, and attacked images, the terminal
latent, public manifest, `run.json`, `metrics.jsonl`, and `summary.json`. On a
6 GB GPU it enables attention/VAE slicing and model CPU offload. Evaluate the
paired images with:

```bash
python experiments/evaluate_quality.py \
  --run-dir outputs/paper/main/compvis-stable-diffusion-v1-4 \
  --prompts experiments/prompts-1000.txt --batch-size 10
```

When the real-image reference corpus is known, pass it with
`--reference-dir`. Without that option the report measures the distributional
shift from paired watermark-free outputs; it must not be presented as the
paper's absolute FID.

Install and run the public comparison implementations with:

```bash
python -m pip install -r requirements-baselines.txt
python experiments/run_image_baselines.py \
  --source-dir outputs/paper/main/compvis-stable-diffusion-v1-4 --resume
python experiments/run_latent_baselines.py --limit 50 --resume
```

See `BASELINES.md` for upstream repositories, pinned revisions, capacities,
and the checkpoint provenance that the paper leaves unspecified.

## Reproduction notes

The paper contains implementation ambiguities that this code exposes:

1. Algorithm 1 requires `l=m*c*h*w`, while the experiments state a 256-bit
   payload for a 4x64x64 latent with `m=1`. `payload_layout="spatial_tile"`
   follows the official Gaussian-Shading convention: it maps the payload to
   4x8x8, tiles it 8x8, and uses block-wise majority voting during recovery.
   `payload_layout="full"` is the literal Algorithm 1/3 interpretation.
2. Equation (25) has `g_t = g_max exp(-decay*(T-t)/T)` without `g_min`, while
   Table 2 reports `g_min=0.1` and the text requires a nearly unconditional
   final step. The default `exponential` schedule uses an endpoint-normalized
   curve from 7.5 to 0.1. Select `paper_exponential` to reproduce equation
   (25) literally.
3. Equation (14) prints the zero-bit exponent as `l-k_i`; for an m-bit symbol
   it must be `m-k_i` for probabilities to normalize.
4. Section 4.1 first states an FPR of `10^-10`, while Table 1, Table 2, and the
   conclusion use `10^-6`. The checked-in experiment config follows the tables
   and uses `10^-6`.
5. The paper reports FID over "50 batches" but gives neither batch size nor
   the real-image reference corpus, Inception preprocessing/version, or CLIP
checkpoint. These are explicit CLI inputs here; absolute Table 1 quality
numbers cannot be independently duplicated without the missing choices.
The runnable default is OpenCLIP `ViT-B-32/laion2b_s34b_b79k`; the selected
checkpoint is stored in `quality.json` and can be overridden explicitly.
6. The paper does not specify a common payload length or checkpoint provenance
   for all seven baselines. RivaGAN's public model supports 32 bits, Tree-Ring
   is a single tag, and the paper itself omits Tree-Ring bit accuracy. Runs
   record the actual payload capacity instead of silently treating them as
   256-bit methods.
7. "Normalized inversion error" is not defined mathematically. This project
   records relative L2 error, `||z_inv-z_T||_2 / ||z_T||_2`, and does not treat
   it as numerically interchangeable with Table 2's unpublished normalization.

Latent-only tests do not download weights. A full image round trip requires a
compatible Stable Diffusion checkpoint, GPU memory, and optional dependencies.
