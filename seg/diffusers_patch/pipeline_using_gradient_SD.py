"""Local Stable Diffusion pipeline entry point with Stein variational transport guidance."""

import inspect
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
    StableDiffusionPipelineOutput,
    rescale_noise_cfg,
)
from diffusers.utils import deprecate
from diffusers.utils.torch_utils import randn_tensor


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """Retrieve and validate scheduler timesteps/sigmas."""
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")

    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        if timesteps is None:
            raise ValueError("Scheduler did not set timesteps.")
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accepts_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        if timesteps is None:
            raise ValueError("Scheduler did not set timesteps.")
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        if timesteps is None:
            raise ValueError("Scheduler did not set timesteps.")

    return timesteps, num_inference_steps


def _expand_prompts_for_particles(
    prompt: Optional[Union[str, List[str]]],
    base_sample_count: int,
    num_particles: int,
) -> Optional[List[str]]:
    if prompt is None:
        return None
    if isinstance(prompt, str):
        base_prompts = [prompt] * base_sample_count
    else:
        base_prompts = list(prompt)
        if len(base_prompts) != base_sample_count:
            raise ValueError(
                f"Prompt list length ({len(base_prompts)}) does not match base sample count ({base_sample_count})."
            )

    particle_prompts: List[str] = []
    for base_prompt in base_prompts:
        particle_prompts.extend([base_prompt] * num_particles)
    return particle_prompts


def _decode_latents_for_reward(pipe: StableDiffusionPipeline, latents: torch.Tensor) -> torch.Tensor:
    # Decode latent tensor to [0, 1] image tensor while preserving grad flow.
    needs_upcasting = pipe.vae.dtype == torch.float16 and pipe.vae.config.force_upcast

    if needs_upcasting:
        pipe.upcast_vae()
        latents = latents.to(next(iter(pipe.vae.post_quant_conv.parameters())).dtype)
    elif latents.dtype != pipe.vae.dtype:
        latents = latents.to(pipe.vae.dtype)

    has_latents_mean = hasattr(pipe.vae.config, "latents_mean") and pipe.vae.config.latents_mean is not None
    has_latents_std = hasattr(pipe.vae.config, "latents_std") and pipe.vae.config.latents_std is not None
    if has_latents_mean and has_latents_std:
        latents_mean = torch.tensor(pipe.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
        latents_std = torch.tensor(pipe.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
        latents = latents * latents_std / pipe.vae.config.scaling_factor + latents_mean
    else:
        latents = latents / pipe.vae.config.scaling_factor

    image = pipe.vae.decode(latents, return_dict=False)[0]

    if needs_upcasting:
        pipe.vae.to(dtype=torch.float16)

    return (image / 2 + 0.5).clamp(0, 1)


def _rbf_stein_vector_field(
    latents: torch.Tensor,
    score: torch.Tensor,
    base_sample_count: int,
    num_particles: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    # Compute Stein direction per prompt group to avoid cross-prompt particle interactions.
    if num_particles == 1:
        return score

    b, c, h, w = latents.shape
    if b != base_sample_count * num_particles:
        raise ValueError("Latent batch does not match base_sample_count * num_particles.")

    latents_grouped = latents.view(base_sample_count, num_particles, c, h, w)
    score_grouped = score.view(base_sample_count, num_particles, c, h, w)
    out_grouped = torch.zeros_like(score_grouped)

    for group_idx in range(base_sample_count):
        x = latents_grouped[group_idx].reshape(num_particles, -1)
        s = score_grouped[group_idx].reshape(num_particles, -1)

        dist2 = torch.cdist(x, x) ** 2
        positive_dist2 = dist2[dist2 > 0]
        if positive_dist2.numel() == 0:
            h_bandwidth = torch.tensor(1.0, device=latents.device, dtype=latents.dtype)
        else:
            h_bandwidth = positive_dist2.median() / (math.log(num_particles + 1.0) + eps)
            h_bandwidth = torch.clamp(h_bandwidth, min=eps)

        kernel = torch.exp(-dist2 / h_bandwidth)
        attraction = (kernel.t() @ s) / float(num_particles)

        weighted_sum = kernel.t() @ x
        kernel_sum = kernel.sum(dim=0, keepdim=True).t()
        repulsion = (2.0 / h_bandwidth) * (weighted_sum - x * kernel_sum) / float(num_particles)

        phi = attraction + repulsion
        out_grouped[group_idx] = phi.view(num_particles, c, h, w)

    return out_grouped.view(b, c, h, w)


def _to_timestep_int(t: Union[int, torch.Tensor]) -> int:
    return int(t.item()) if torch.is_tensor(t) else int(t)


@torch.no_grad()
def pipeline_using_gradient_sd(
    self: StableDiffusionPipeline,
    prompt: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    guidance_scale: float = 7.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: int = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.Tensor] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    ip_adapter_image: Optional[Any] = None,
    ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
    output_type: str = "pil",
    return_dict: bool = True,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    guidance_rescale: float = 0.0,
    clip_skip: Optional[int] = None,
    callback_on_step_end: Optional[Callable[..., Optional[Dict[str, torch.Tensor]]]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    # Stein parameters
    num_particles: int = 4,
    batch_p: int = 1,
    reward_fn: Optional[Callable[[torch.Tensor, List[str]], torch.Tensor]] = None,
    stein_step: float = 0.05,
    stein_loop: int = 1,
    stein_kernel: str = "rbf",
    stein_adagrad_eps: float = 1e-8,
    stein_adagrad_clip: Optional[Tuple[float, float]] = None,
    kl_coeff: float = 1.0,
    steer_start: Optional[int] = None,
    steer_end: Optional[int] = None,
    return_all_particles: bool = True,
    intermediate_rewards: bool = False,
    monitor_stein_delta: bool = False,
    **kwargs,
) -> Union[StableDiffusionPipelineOutput, Tuple]:
    """Run SD denoising with optional Stein particle transport guidance."""

    callback = kwargs.pop("callback", None)
    callback_steps = kwargs.pop("callback_steps", None)

    legacy_return_intermediate_rewards = kwargs.pop("return_intermediate_rewards", None)
    legacy_show_intermediate_rewards = kwargs.pop("show_intermediate_rewards", None)
    if legacy_return_intermediate_rewards is not None or legacy_show_intermediate_rewards is not None:
        intermediate_rewards = (
            intermediate_rewards
            or bool(legacy_return_intermediate_rewards)
            or bool(legacy_show_intermediate_rewards)
        )

    if callback is not None:
        deprecate(
            "callback",
            "1.0.0",
            "Passing `callback` as an input argument is deprecated, use `callback_on_step_end` instead.",
        )
    if callback_steps is not None:
        deprecate(
            "callback_steps",
            "1.0.0",
            "Passing `callback_steps` as an input argument is deprecated, use `callback_on_step_end` instead.",
        )

    if callback_on_step_end is not None and hasattr(callback_on_step_end, "tensor_inputs"):
        callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

    height = height or self.unet.config.sample_size * self.vae_scale_factor
    width = width or self.unet.config.sample_size * self.vae_scale_factor

    # 1. Check inputs
    if num_particles < 1:
        raise ValueError("num_particles must be >= 1")
    if batch_p < 1:
        raise ValueError("batch_p must be >= 1")
    if num_particles < batch_p:
        raise ValueError("num_particles should be greater than or equal to batch_p")
    if stein_loop < 0:
        raise ValueError("stein_loop must be >= 0")
    if stein_step < 0:
        raise ValueError("stein_step must be >= 0")

    check_params = inspect.signature(self.check_inputs).parameters
    check_kwargs: Dict[str, Any] = {
        "prompt": prompt,
        "height": height,
        "width": width,
        "callback_steps": callback_steps,
        "negative_prompt": negative_prompt,
        "prompt_embeds": prompt_embeds,
        "negative_prompt_embeds": negative_prompt_embeds,
        "ip_adapter_image": ip_adapter_image,
        "ip_adapter_image_embeds": ip_adapter_image_embeds,
        "callback_on_step_end_tensor_inputs": callback_on_step_end_tensor_inputs,
    }
    self.check_inputs(**{k: v for k, v in check_kwargs.items() if k in check_params})

    # 2. Define call parameters
    self._guidance_scale = guidance_scale
    self._guidance_rescale = guidance_rescale
    self._clip_skip = clip_skip
    self._cross_attention_kwargs = cross_attention_kwargs
    self._interrupt = False

    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        if prompt_embeds is None:
            raise ValueError("`prompt_embeds` must be provided when `prompt` is None.")
        batch_size = prompt_embeds.shape[0]

    # Diffusers expects prompt/negative_prompt container types to match.
    if isinstance(prompt, list) and isinstance(negative_prompt, str):
        negative_prompt = [negative_prompt] * batch_size
    elif isinstance(prompt, str) and isinstance(negative_prompt, list):
        if len(negative_prompt) == 1:
            negative_prompt = negative_prompt[0]
        else:
            raise TypeError(
                "`negative_prompt` must be a string when `prompt` is a string, "
                f"but got list of length {len(negative_prompt)}."
            )

    base_sample_count = batch_size * num_images_per_prompt
    particle_images_per_prompt = num_images_per_prompt * num_particles

    device = self._execution_device
    lora_scale = self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None

    # 3. Encode input prompt
    if hasattr(self, "encode_prompt"):
        encode_params = inspect.signature(self.encode_prompt).parameters
        encode_kwargs: Dict[str, Any] = {
            "prompt": prompt,
            "device": device,
            "num_images_per_prompt": particle_images_per_prompt,
            "do_classifier_free_guidance": self.do_classifier_free_guidance,
            "negative_prompt": negative_prompt,
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "lora_scale": lora_scale,
            "clip_skip": self.clip_skip,
        }
        encoded = self.encode_prompt(**{k: v for k, v in encode_kwargs.items() if k in encode_params})

        if isinstance(encoded, tuple) and len(encoded) >= 2:
            prompt_embeds, negative_prompt_embeds = encoded[0], encoded[1]
        elif torch.is_tensor(encoded):
            prompt_embeds = encoded
            negative_prompt_embeds = None
        else:
            raise RuntimeError("Unexpected encode_prompt output format.")
    else:
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            particle_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
        )
        negative_prompt_embeds = None

    assert prompt_embeds is not None
    prompt_embeds = cast(torch.Tensor, prompt_embeds)

    if self.do_classifier_free_guidance:
        if negative_prompt_embeds is None:
            raise ValueError("negative_prompt_embeds is required for classifier-free guidance.")
        negative_prompt_embeds = cast(torch.Tensor, negative_prompt_embeds)
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

    prompt_embeds = prompt_embeds.to(device)

    # 4. Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        timesteps,
        sigmas,
    )
    assert timesteps is not None

    # 5. Prepare latent variables
    num_channels_latents = self.unet.config.in_channels
    expected_particle_batch = base_sample_count * num_particles
    if latents is None:
        latents = cast(
            torch.Tensor,
            self.prepare_latents(
                expected_particle_batch,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                None,
            ),
        )
    else:
        latents = latents.to(device=device, dtype=prompt_embeds.dtype)
        expected_base_batch = base_sample_count
        if latents.shape[0] == expected_base_batch and num_particles > 1:
            latents = latents.repeat_interleave(num_particles, dim=0)
            jitter = randn_tensor(latents.shape, generator=generator, device=device, dtype=latents.dtype)
            latents = (latents + jitter) / math.sqrt(2.0)
        elif latents.shape[0] != expected_particle_batch:
            raise ValueError(
                f"Provided latents batch ({latents.shape[0]}) must be either {expected_base_batch} or {expected_particle_batch}."
            )
    latents = cast(torch.Tensor, latents)

    # 6. Prepare extra step kwargs
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    image_embeds = None
    if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
        image_embeds = self.prepare_ip_adapter_image_embeds(
            ip_adapter_image,
            ip_adapter_image_embeds,
            device,
            expected_particle_batch,
            self.do_classifier_free_guidance,
        )

    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

    timestep_cond = None
    if self.unet.config.time_cond_proj_dim is not None:
        guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(expected_particle_batch)
        timestep_cond = self.get_guidance_scale_embedding(
            guidance_scale_tensor,
            embedding_dim=self.unet.config.time_cond_proj_dim,
        ).to(device=device, dtype=latents.dtype)

    prompt_particles = _expand_prompts_for_particles(prompt, base_sample_count, num_particles)

    total_inference_steps = len(timesteps)

    if steer_start is None:
        steer_start_effective = 0
    else:
        steer_start_effective = int(steer_start)

    if steer_end is None:
        steer_end_effective = total_inference_steps - 1
    else:
        steer_end_effective = int(steer_end)

    if steer_start_effective < 0:
        steer_start_effective += total_inference_steps
    if steer_end_effective < 0:
        steer_end_effective += total_inference_steps

    steer_start_effective = max(0, min(total_inference_steps - 1, steer_start_effective))
    steer_end_effective = max(0, min(total_inference_steps - 1, steer_end_effective))

    has_steering_window = steer_start_effective <= steer_end_effective
    use_stein = has_steering_window and reward_fn is not None and stein_loop > 0 and stein_step > 0
    reward_chunk_size = max(1, int(batch_p) * base_sample_count)

    intermediate_rewards_data: Dict[str, List[float]] = {
        "step_indices": [],
        "timesteps": [],
        "pre_steer_mean": [],
        "pre_steer_max": [],
        "post_steer_mean": [],
        "post_steer_max": [],
    }

    def _slice_condition_tensor(
        condition: torch.Tensor,
        start_idx: Optional[int],
        end_idx: Optional[int],
    ) -> torch.Tensor:
        if start_idx is None or end_idx is None:
            return condition

        if self.do_classifier_free_guidance:
            expected_cfg_batch = expected_particle_batch * 2
            if condition.shape[0] == expected_cfg_batch:
                condition_uncond = condition[:expected_particle_batch]
                condition_text = condition[expected_particle_batch:]
                return torch.cat(
                    [
                        condition_uncond[start_idx:end_idx],
                        condition_text[start_idx:end_idx],
                    ],
                    dim=0,
                )
            raise ValueError(
                "Condition batch does not match classifier-free guidance layout: "
                f"got {condition.shape[0]}, expected {expected_cfg_batch}."
            )

        if condition.shape[0] == expected_particle_batch:
            return condition[start_idx:end_idx]

        raise ValueError(
            "Condition batch does not match particle batch layout: "
            f"got {condition.shape[0]}, expected {expected_particle_batch}."
        )

    def _predict_noise(
        current_latents: torch.Tensor,
        t: torch.Tensor,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
    ) -> torch.Tensor:
        latent_model_input = torch.cat([current_latents] * 2) if self.do_classifier_free_guidance else current_latents
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

        prompt_embeds_local = _slice_condition_tensor(prompt_embeds, start_idx, end_idx)

        added_cond_kwargs = None
        if image_embeds is not None:
            added_cond_kwargs = {
                "image_embeds": [
                    _slice_condition_tensor(embed, start_idx, end_idx)
                    for embed in image_embeds
                ]
            }

        noise_pred_local = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds_local,
            timestep_cond=timestep_cond,
            cross_attention_kwargs=self.cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]

        if self.do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred_local.chunk(2)
            noise_pred_local = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            if self.guidance_rescale > 0.0:
                noise_pred_local = rescale_noise_cfg(
                    noise_pred_local,
                    noise_pred_text,
                    guidance_rescale=self.guidance_rescale,
                )

        return noise_pred_local

    def _predict_x0(current_latents: torch.Tensor, t_int: int, noise_pred_local: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        alpha_bar_t = self.scheduler.alphas_cumprod[t_int].to(device=current_latents.device, dtype=current_latents.dtype)
        sqrt_alpha_bar_t = torch.sqrt(torch.clamp(alpha_bar_t, min=1e-6))
        sqrt_one_minus_alpha_bar_t = torch.sqrt(torch.clamp(1.0 - alpha_bar_t, min=1e-6))
        pred_x0 = (current_latents - sqrt_one_minus_alpha_bar_t * noise_pred_local) / sqrt_alpha_bar_t
        return pred_x0, sqrt_alpha_bar_t, sqrt_one_minus_alpha_bar_t

    def _compute_reward(current_latents: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if reward_fn is None or prompt_particles is None:
            return torch.zeros(current_latents.shape[0], device=current_latents.device, dtype=current_latents.dtype)

        rewards: List[torch.Tensor] = []
        for start_idx in range(0, current_latents.shape[0], reward_chunk_size):
            end_idx = start_idx + reward_chunk_size
            lat_chunk = current_latents[start_idx:end_idx]

            with torch.no_grad():
                noise_pred_chunk = _predict_noise(lat_chunk, t, start_idx=start_idx, end_idx=end_idx)
                pred_x0_chunk, _, _ = _predict_x0(lat_chunk, _to_timestep_int(t), noise_pred_chunk)
                images_chunk = _decode_latents_for_reward(self, pred_x0_chunk)
                reward_chunk = reward_fn(images_chunk, prompt_particles[start_idx:end_idx])

            if not torch.is_tensor(reward_chunk):
                reward_chunk = torch.tensor(reward_chunk, device=current_latents.device, dtype=current_latents.dtype)
            reward_chunk = reward_chunk.to(device=current_latents.device, dtype=current_latents.dtype).flatten()
            rewards.append(reward_chunk)

        return torch.cat(rewards, dim=0)

    def _compute_reward_grad(
        current_latents: torch.Tensor,
        t: torch.Tensor,
        return_rewards: bool = False,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        if reward_fn is None or prompt_particles is None:
            zero_rewards = torch.zeros(current_latents.shape[0], device=current_latents.device, dtype=current_latents.dtype)
            return (zero_rewards if return_rewards else None), torch.zeros_like(current_latents)

        all_rewards: List[torch.Tensor] = []
        all_grads: List[torch.Tensor] = []

        for start_idx in range(0, current_latents.shape[0], reward_chunk_size):
            end_idx = start_idx + reward_chunk_size
            lat_chunk = current_latents[start_idx:end_idx].detach().requires_grad_(True)

            with torch.enable_grad():
                noise_pred_chunk = _predict_noise(lat_chunk, t, start_idx=start_idx, end_idx=end_idx)
                pred_x0_chunk, _, _ = _predict_x0(lat_chunk, _to_timestep_int(t), noise_pred_chunk)
                images_chunk = _decode_latents_for_reward(self, pred_x0_chunk)
                reward_chunk = reward_fn(images_chunk, prompt_particles[start_idx:end_idx])

                if not torch.is_tensor(reward_chunk):
                    reward_chunk = torch.tensor(reward_chunk, device=lat_chunk.device, dtype=lat_chunk.dtype)
                reward_chunk = reward_chunk.to(device=lat_chunk.device, dtype=lat_chunk.dtype).flatten()
                scaled_reward_chunk = reward_chunk / max(kl_coeff, 1e-6)

                grad_chunk = torch.autograd.grad(scaled_reward_chunk.sum(), lat_chunk, allow_unused=True)[0]
                if grad_chunk is None:
                    grad_chunk = torch.zeros_like(lat_chunk)

            if return_rewards:
                all_rewards.append(reward_chunk.detach())
            all_grads.append(grad_chunk.detach())

        rewards_out = torch.cat(all_rewards, dim=0) if return_rewards else None
        return rewards_out, torch.cat(all_grads, dim=0)

    # Stein loop
    self._num_timesteps = len(timesteps)
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            t_int = _to_timestep_int(t)
            noise_pred = _predict_noise(latents, t)
            is_steered_step = use_stein and (steer_start_effective <= i <= steer_end_effective)

            pre_stein_latents = None
            post_stein_latents = None
            pre_stein_pred_x0 = None
            post_stein_pred_x0 = None

            if is_steered_step:
                latents_before_stein = None
                if monitor_stein_delta or "pre_stein_latents" in callback_on_step_end_tensor_inputs:
                    latents_before_stein = latents.detach().clone()

                if "pre_stein_latents" in callback_on_step_end_tensor_inputs:
                    pre_stein_latents = latents_before_stein

                should_log_rewards = intermediate_rewards
                pre_reward = None

                if should_log_rewards:
                    intermediate_rewards_data["step_indices"].append(float(i))
                    intermediate_rewards_data["timesteps"].append(float(t_int))

                grad_accumulator = torch.zeros_like(latents, dtype=torch.float32)

                for loop_idx in range(stein_loop):
                    noise_pred_for_score = _predict_noise(latents, t)
                    pred_x0_for_score, _, sqrt_one_minus_alpha_bar_t = _predict_x0(latents, t_int, noise_pred_for_score)
                    if loop_idx == 0:
                        pre_stein_pred_x0 = pred_x0_for_score.detach().clone()
                    prior_score = -noise_pred_for_score / torch.clamp(sqrt_one_minus_alpha_bar_t, min=1e-6)

                    reward_values, reward_grad = _compute_reward_grad(
                        latents,
                        t,
                        return_rewards=should_log_rewards and loop_idx == 0,
                    )
                    if should_log_rewards and loop_idx == 0 and reward_values is not None:
                        pre_reward = reward_values

                    score_q = prior_score.float() + reward_grad.float()

                    if stein_kernel != "rbf":
                        raise ValueError(f"Unsupported stein_kernel: {stein_kernel}. Only 'rbf' is currently supported.")

                    stein_direction = _rbf_stein_vector_field(
                        latents=latents.float(),
                        score=score_q,
                        base_sample_count=base_sample_count,
                        num_particles=num_particles,
                    )
                    stein_direction = torch.nan_to_num(stein_direction)

                    grad_accumulator = grad_accumulator + stein_direction * stein_direction
                    adaptive_step = stein_step / (torch.sqrt(grad_accumulator) + stein_adagrad_eps)
                    if stein_adagrad_clip is not None:
                        adaptive_step = adaptive_step.clamp(min=stein_adagrad_clip[0], max=stein_adagrad_clip[1])

                    latents = latents + (adaptive_step * stein_direction).to(latents.dtype)

                if monitor_stein_delta and latents_before_stein is not None:
                    delta = (latents - latents_before_stein).flatten(1).norm(dim=1)
                    base = latents_before_stein.flatten(1).norm(dim=1)
                    rel_delta = (delta / (base + 1e-8)).mean()
                    before_flat = latents_before_stein.flatten(1).float()
                    after_flat = latents.flatten(1).float()
                    cosine_sim = (
                        (before_flat * after_flat).sum(dim=1)
                        / (before_flat.norm(dim=1) * after_flat.norm(dim=1) + 1e-8)
                    ).mean()

                    print(
                        "i=", i,
                        "t=", t_int,
                        "rel_delta=", rel_delta.item(),
                        "abs_delta=", delta.mean().item(),
                        "cosine_sim=", cosine_sim.item(),
                    )

                if should_log_rewards:
                    if pre_reward is None:
                        pre_reward = _compute_reward(latents, t)

                    intermediate_rewards_data["pre_steer_mean"].append(float(pre_reward.mean().item()))
                    intermediate_rewards_data["pre_steer_max"].append(float(pre_reward.max().item()))

                    post_reward = _compute_reward(latents, t)
                    intermediate_rewards_data["post_steer_mean"].append(float(post_reward.mean().item()))
                    intermediate_rewards_data["post_steer_max"].append(float(post_reward.max().item()))

                    if intermediate_rewards:
                        print(
                            f"[t={t_int:04d}] pre_mean={pre_reward.mean().item():.6f} "
                            f"pre_max={pre_reward.max().item():.6f} "
                            f"post_mean={post_reward.mean().item():.6f} "
                            f"post_max={post_reward.max().item():.6f}"
                        )

                if "post_stein_latents" in callback_on_step_end_tensor_inputs:
                    post_stein_latents = latents.detach().clone()

            if hasattr(self.scheduler, "previous_timestep"):
                prev_t = self.scheduler.previous_timestep(t)
                prev_t_int = int(prev_t.item()) if torch.is_tensor(prev_t) else int(prev_t)
            else:
                if i + 1 < len(timesteps):
                    prev_t = timesteps[i + 1]
                    prev_t_int = _to_timestep_int(prev_t)
                else:
                    prev_t = -1
                    prev_t_int = -1

            alpha_bar_prev = (
                self.scheduler.alphas_cumprod[prev_t_int]
                if prev_t_int >= 0
                else self.scheduler.final_alpha_cumprod
            )
            if not torch.is_tensor(alpha_bar_prev):
                alpha_bar_prev = torch.tensor(alpha_bar_prev, device=latents.device, dtype=latents.dtype)
            alpha_bar_prev = alpha_bar_prev.to(device=latents.device, dtype=latents.dtype)

            if hasattr(self.scheduler, "_get_variance"):
                variance_t = self.scheduler._get_variance(t, prev_t)
                if not torch.is_tensor(variance_t):
                    variance_t = torch.tensor(variance_t, device=latents.device, dtype=latents.dtype)
                variance_t = variance_t.to(device=latents.device, dtype=latents.dtype)
            else:
                variance_t = torch.tensor(0.0, device=latents.device, dtype=latents.dtype)

            sigma_t = eta * torch.sqrt(torch.clamp(variance_t, min=0.0))
            pred_noise_coeff = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - sigma_t ** 2, min=0.0))
            white_noise = randn_tensor(latents.shape, generator=generator, device=latents.device, dtype=latents.dtype)

            # Pred x0|t = x0(steered_xt). Reuse noise prediction when no steering changed latents.
            steered_noise_pred = _predict_noise(latents, t) if is_steered_step else noise_pred
            pred_x0, _, _ = _predict_x0(latents, t_int, steered_noise_pred)
            if is_steered_step and callback_on_step_end is not None:
                post_stein_pred_x0 = pred_x0.detach()

            latents_dtype = latents.dtype
            latents = (
                torch.sqrt(torch.clamp(alpha_bar_prev, min=0.0)) * pred_x0
                + pred_noise_coeff * noise_pred
                + sigma_t * white_noise
            )
            if latents.dtype != latents_dtype and torch.backends.mps.is_available():
                latents = latents.to(latents_dtype)

            if callback_on_step_end is not None:
                callback_kwargs = {key: locals()[key] for key in callback_on_step_end_tensor_inputs if key in locals()}
                callback_kwargs["pre_stein_pred_x0"] = pre_stein_pred_x0
                callback_kwargs["post_stein_pred_x0"] = post_stein_pred_x0
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                if callback_outputs is not None:
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()
                if callback is not None and callback_steps is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, "order", 1)
                    callback(step_idx, t, latents)

    final_particle_rewards = None
    if num_particles > 1 and not return_all_particles:
        if reward_fn is not None and prompt_particles is not None:
            with torch.no_grad():
                final_images_for_select = _decode_latents_for_reward(self, latents)
                final_particle_rewards = reward_fn(final_images_for_select, prompt_particles)
                if not torch.is_tensor(final_particle_rewards):
                    final_particle_rewards = torch.tensor(final_particle_rewards, device=latents.device, dtype=latents.dtype)
                final_particle_rewards = final_particle_rewards.to(device=latents.device, dtype=latents.dtype).flatten()

            reward_grouped = final_particle_rewards.view(base_sample_count, num_particles)
            best_idx = reward_grouped.argmax(dim=1)
            base_idx = torch.arange(base_sample_count, device=latents.device)
            gather_idx = base_idx * num_particles + best_idx
            latents = latents[gather_idx]

            if intermediate_rewards:
                intermediate_rewards_data["final_best_particle_reward"] = [
                    float(v) for v in reward_grouped[base_idx, best_idx].detach().float().cpu().tolist()
                ]
        else:
            latents = latents.view(base_sample_count, num_particles, *latents.shape[1:])[:, 0]

    if output_type != "latent":
        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

        if needs_upcasting:
            self.upcast_vae()
            latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
        elif latents.dtype != self.vae.dtype and torch.backends.mps.is_available():
            self.vae = self.vae.to(latents.dtype)

        has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
        has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None
        if has_latents_mean and has_latents_std:
            latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
            latents_std = torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
            latents = latents * latents_std / self.vae.config.scaling_factor + latents_mean
        else:
            latents = latents / self.vae.config.scaling_factor

        image = self.vae.decode(latents, return_dict=False)[0]

        if needs_upcasting:
            self.vae.to(dtype=torch.float16)
    else:
        image = latents

    image = self.image_processor.postprocess(image, output_type=output_type)
    self.maybe_free_model_hooks()

    if not return_dict:
        if intermediate_rewards:
            return (image, intermediate_rewards_data)
        return (image,)

    if intermediate_rewards:
        return {"images": image, "intermediate_rewards": intermediate_rewards_data}

    return StableDiffusionPipelineOutput(images=image)


# Backward-compatible alias used in older callsites/docs.
pipeline_using_stein_sd = pipeline_using_gradient_sd
