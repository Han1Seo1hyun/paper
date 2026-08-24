"""Optional Hugging Face Diffusers integration for PPGS.

PyTorch and Diffusers are imported lazily so the sampler can be tested with
only NumPy installed.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

from .core import EmbeddingMetadata, bit_accuracy, decode_latents, embed_watermark
from .guidance import GuidanceSchedule, guidance_scale


@dataclass
class GenerationResult:
    images: list[Any]
    terminal_latents: Any
    metadata: EmbeddingMetadata
    settings: dict[str, Any]

    def save_manifest(self, path: str | Path) -> None:
        """Persist all public information needed for extraction/reproduction."""

        manifest = {
            "format": "ppgs-generation-v1",
            "embedding": self.metadata.to_dict(),
            "generation": self.settings,
        }
        Path(path).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


class PPGSDiffusers:
    """Add PPGS generation and null-prompt DDIM inversion to an SD pipeline."""

    def __init__(self, pipeline: Any):
        self.pipeline = pipeline

    @property
    def device(self) -> Any:
        import torch

        return getattr(self.pipeline, "_execution_device", torch.device("cpu"))

    def _encode_prompt(self, prompt: str, *, classifier_free: bool) -> tuple[Any, Any | None]:
        encoded = self.pipeline.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=classifier_free,
            negative_prompt="" if classifier_free else None,
        )
        if isinstance(encoded, tuple):
            return encoded[0], encoded[1] if classifier_free else None
        return encoded, None

    def generate(
        self,
        prompt: str,
        watermark: Sequence[int],
        *,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        maximum_guidance: float = 7.5,
        guidance_decay: float = 2.0,
        minimum_guidance: float = 0.1,
        guidance_schedule: GuidanceSchedule = "exponential",
        public_seed: int = 2026,
        sampling_seed: int | None = None,
        bits_per_position: int = 1,
        payload_layout: Literal["full", "repeat", "spatial_tile"] = "spatial_tile",
        spatial_copies: tuple[int, int, int] = (1, 8, 8),
    ) -> GenerationResult:
        """Generate with Algorithm 1 sampling and Algorithm 2 guidance."""

        import torch
        from diffusers import DDIMScheduler

        pipe = self.pipeline
        scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        scheduler.set_timesteps(num_inference_steps, device=self.device)
        channels = int(pipe.unet.config.in_channels)
        vae_scale = int(getattr(pipe, "vae_scale_factor", 8))
        latent_shape = (1, channels, height // vae_scale, width // vae_scale)
        latent_np, metadata = embed_watermark(
            watermark,
            latent_shape,
            bits_per_position=bits_per_position,
            public_seed=public_seed,
            sampling_seed=sampling_seed,
            payload_layout=payload_layout,
            spatial_copies=spatial_copies,
        )

        prompt_embeds, negative_embeds = self._encode_prompt(prompt, classifier_free=True)
        latents = torch.as_tensor(latent_np, device=self.device, dtype=prompt_embeds.dtype)
        terminal_latents = latents.clone()
        combined_embeds = torch.cat([negative_embeds, prompt_embeds])

        with torch.no_grad():
            for index, timestep in enumerate(scheduler.timesteps):
                model_input = scheduler.scale_model_input(
                    torch.cat([latents, latents]), timestep
                )
                prediction = pipe.unet(
                    model_input, timestep, encoder_hidden_states=combined_embeds
                ).sample
                unconditioned, conditioned = prediction.chunk(2)
                scale = guidance_scale(
                    index,
                    num_inference_steps,
                    maximum=maximum_guidance,
                    decay=guidance_decay,
                    minimum=minimum_guidance,
                    schedule=guidance_schedule,
                )
                blended = unconditioned + scale * (conditioned - unconditioned)
                latents = scheduler.step(blended, timestep, latents, eta=0.0).prev_sample

            scaling = float(pipe.vae.config.scaling_factor)
            decoded = pipe.vae.decode(latents / scaling, return_dict=False)[0]
            images = pipe.image_processor.postprocess(decoded, output_type="pil")
        settings = {
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "maximum_guidance": maximum_guidance,
            "minimum_guidance": minimum_guidance,
            "guidance_decay": guidance_decay,
            "guidance_schedule": guidance_schedule,
            "sampling_seed": sampling_seed,
            "spatial_copies": list(spatial_copies),
        }
        return GenerationResult(images, terminal_latents, metadata, settings)

    def invert(
        self,
        image: Any,
        *,
        num_inference_steps: int = 10,
        metadata: EmbeddingMetadata | None = None,
    ) -> Any:
        """Encode an image and perform Algorithm 3 null-prompt DDIM inversion."""

        import torch
        from diffusers import DDIMInverseScheduler

        pipe = self.pipeline
        scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
        scheduler.set_timesteps(num_inference_steps, device=self.device)
        prompt_embeds, _ = self._encode_prompt("", classifier_free=False)
        preprocess_kwargs: dict[str, int] = {}
        if metadata is not None:
            vae_scale = int(getattr(pipe, "vae_scale_factor", 8))
            preprocess_kwargs = {
                "height": metadata.latent_shape[-2] * vae_scale,
                "width": metadata.latent_shape[-1] * vae_scale,
            }
        image_tensor = pipe.image_processor.preprocess(image, **preprocess_kwargs).to(
            device=self.device, dtype=prompt_embeds.dtype
        )
        with torch.no_grad():
            distribution = pipe.vae.encode(image_tensor).latent_dist
            latents = distribution.mode() * float(pipe.vae.config.scaling_factor)
            for timestep in scheduler.timesteps:
                model_input = scheduler.scale_model_input(latents, timestep)
                prediction = pipe.unet(
                    model_input, timestep, encoder_hidden_states=prompt_embeds
                ).sample
                latents = scheduler.step(prediction, timestep, latents).prev_sample
        if metadata is not None and tuple(latents.shape) != metadata.latent_shape:
            raise ValueError(
                f"inversion produced {tuple(latents.shape)}, expected "
                f"{metadata.latent_shape}"
            )
        return latents

    @staticmethod
    def extract(inverted_latents: Any, metadata: EmbeddingMetadata) -> np.ndarray:
        values = inverted_latents.detach().float().cpu().numpy()
        return decode_latents(values, metadata)

    @staticmethod
    def verify(
        inverted_latents: Any,
        expected_watermark: Sequence[int],
        metadata: EmbeddingMetadata,
        *,
        threshold: float = 0.9,
    ) -> tuple[bool, float, np.ndarray]:
        recovered = PPGSDiffusers.extract(inverted_latents, metadata)
        accuracy = bit_accuracy(expected_watermark, recovered)
        return accuracy >= threshold, accuracy, recovered
