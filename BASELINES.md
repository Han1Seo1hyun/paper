# Baseline provenance

The comparison layer uses public upstream implementations and records their
actual capacities. It does not relabel unlike checkpoints as the unpublished
setup used for Table 1.

| Method | Upstream | Pinned revision | Local status |
|---|---|---|---|
| Gaussian Shading | `bsmhmmlf/Gaussian-Shading` | `09c678fadc7545acf7be12647ddf2a5e66f6a9dc` | Sign-conditioned keyed sampler and block vote reproduced in `baselines.py` |
| Tree-Ring | `YuxinWenRick/tree-ring-watermark` | `3015283d9cf82e90b628f02ad2121bd37408ca9a` | Random Fourier patch and L1 detector reproduced in `baselines.py` |
| DwtDctSvd / RivaGAN | `ShieldMnt/invisible-watermark` | `68d0376d94a4701ed240af0841ec12e00676e325` | Official encoder/decoder called by `run_image_baselines.py` |
| Stable Signature | `facebookresearch/stable_signature` | upstream checkout in `work/baselines` | Extractor is public; the paper's per-backbone fine-tuned decoder is not identified |
| MBRS | `jzyustc/MBRS` | upstream checkout in `work/baselines` | Training code is public; paper-specific pretrained checkpoint is not identified |
| HiDDeN | `facebookresearch/HiDDeN` | not vendored | Original implementation uses Torch7; paper-specific PyTorch checkpoint is not identified |

The `work/` checkouts and generated weights are intentionally ignored by Git.
`requirements-baselines.txt` pins the official image-domain toolkit revision.
