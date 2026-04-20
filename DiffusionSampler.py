from collections import defaultdict
import contextlib
import os
import datetime
import time
from absl import app, flags
from ml_collections import config_flags
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import DiffusionPipeline, StableDiffusionPipeline, StableDiffusionXLPipeline, DDIMScheduler, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from seg.scorers.ImageReward_scorer import ImageRewardScorer
from seg.scorers.PickScore_scorer import PickScoreScorer
from seg.scorers.clip_scorer import CLIPScorer
import numpy as np
import torch
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
import json
import random
from seg.diffusers_patch.pipeline_using_Stein_SDXL import pipeline_using_stein_sdxl

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

class DiffusionModelSampler:
    def __init__(self, config, *args, **kwargs):
        """Initialize the Sampler with the given configuration."""
        self.config = config
        random.seed(self.config.seed)
        self.accelerator = None
        self.pipeline = None
        self.global_step = 0
        self.logger = get_logger(__name__)
        self.unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
        self.config.run_name = self.config.run_name or self.unique_id
        self.log_dir = f"logs/{self.config.project_name}/{self.config.reward_fn}/{self.config.run_name}/eval_vis"
        os.makedirs(self.log_dir, exist_ok=True)
        with open(f"logs/{self.config.project_name}/{self.config.reward_fn}/{self.config.run_name}/config.json", 'w') as json_file:
            json.dump(config.to_dict(), json_file, indent=4)

        # Setup the accelerator and environment
        self.setup_accelerator(*args, **kwargs)

        # Load models and scheduler
        self.load_models_and_scheduler()

        # Prepare prompts and rewards
        self.prepare_prompts_and_rewards()

        self.autocast = self.accelerator.autocast

        if "xl" in self.config.pretrained.model:
            print("Configuring SDXL")
            self.pipeline.vae.to(torch.float32)
            self.pipeline.text_encoder.to(dtype=self.inference_dtype)
            if hasattr(self.pipeline, "text_encoder_2") and self.pipeline.text_encoder_2 is not None:
                self.pipeline.text_encoder_2.to(dtype=self.inference_dtype)
            self.autocast = contextlib.nullcontext

    def _decode_latents_sdxl(self, latents):
        needs_upcasting = self.pipeline.vae.dtype == torch.float16 and self.pipeline.vae.config.force_upcast

        if needs_upcasting:
            self.pipeline.upcast_vae()
            latents = latents.to(next(iter(self.pipeline.vae.post_quant_conv.parameters())).dtype)
        elif latents.dtype != self.pipeline.vae.dtype:
            if torch.backends.mps.is_available():
                self.pipeline.vae = self.pipeline.vae.to(latents.dtype)
            else:
                latents = latents.to(self.pipeline.vae.dtype)

        has_latents_mean = hasattr(self.pipeline.vae.config, "latents_mean") and self.pipeline.vae.config.latents_mean is not None
        has_latents_std = hasattr(self.pipeline.vae.config, "latents_std") and self.pipeline.vae.config.latents_std is not None
        if has_latents_mean and has_latents_std:
            latents_mean = torch.tensor(self.pipeline.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
            latents_std = torch.tensor(self.pipeline.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
            latents = latents * latents_std / self.pipeline.vae.config.scaling_factor + latents_mean
        else:
            latents = latents / self.pipeline.vae.config.scaling_factor

        image = self.pipeline.vae.decode(latents, return_dict=False)[0]

        if needs_upcasting:
            self.pipeline.vae.to(dtype=torch.float16)

        do_denormalize = [True] * image.shape[0]
        return self.pipeline.image_processor.postprocess(image, output_type="pt", do_denormalize=do_denormalize)

    def setup_accelerator(self, *args, **kwargs):
        """Setup the Accelerate environment and logging."""

        accelerator_config = ProjectConfiguration(
            project_dir=os.path.join(self.config.logdir, self.config.run_name),
        )

        self.accelerator = Accelerator(
            log_with="wandb",
            mixed_precision=self.config.mixed_precision,
            project_config=accelerator_config,
            *args, **kwargs
        )

        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                project_name=self.config.project_name,
                config=self.config.to_dict(),
                init_kwargs={"wandb": {"name": self.config.run_name}},
            )
        self.logger.info(f"\n{self.config}")

        # Set seed
        set_seed(self.config.seed, device_specific=True)

        # Enable TF32 for faster training on Ampere GPUs
        if self.config.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

    def load_models_and_scheduler(self):
        """Load the Stable Diffusion models and the DDIM scheduler."""

        if "xl" in self.config.pretrained.model:
            print("Loading SDXL")
            pipeline = DiffusionPipeline.from_pretrained(
                self.config.pretrained.model, 
                torch_dtype=torch.float16, variant="fp16", use_safetensors=True
            ).to(self.accelerator.device)
        else:
            pipeline = StableDiffusionPipeline.from_pretrained(
                self.config.pretrained.model, 
                revision=self.config.pretrained.revision
            ).to(self.accelerator.device)

        # Freeze parameters of models to save more memory
        pipeline.vae.requires_grad_(False)
        pipeline.text_encoder.requires_grad_(False)
        if hasattr(pipeline, "text_encoder_2") and pipeline.text_encoder_2 is not None:
            pipeline.text_encoder_2.requires_grad_(False)

        # Disable safety checker
        pipeline.safety_checker = None

        # Switch to DDIM scheduler
        if not "lcm" in self.config.pretrained.model and not "LCM" in self.config.pretrained.model:
            pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        pipeline.scheduler.set_timesteps(self.config.sample.num_steps)

        # Set mixed precision for inference
        inference_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            inference_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            inference_dtype = torch.bfloat16
        self.inference_dtype = inference_dtype

        # Move unet, vae, and text_encoder to device and cast to inference_dtype
        pipeline.vae.to(self.accelerator.device, dtype=inference_dtype)
        pipeline.text_encoder.to(self.accelerator.device, dtype=inference_dtype)   
        if hasattr(pipeline, "text_encoder_2") and pipeline.text_encoder_2 is not None:
            pipeline.text_encoder_2.to(self.accelerator.device, dtype=inference_dtype)
        
        self.pipeline = pipeline

    def prepare_prompts_and_rewards(self):
        """Prepare the prompt and reward functions."""
        # Retrieve the prompt function from ddpo_pytorch.prompts using the config
        self.prompt_fn = getattr(prompts_file, self.config.prompt_fn)
        self.eval_prompts, self.eval_prompt_metadata = zip(
                *[
                    self.prompt_fn(i) 
                    for i in range(self.config.sample.batch_size * self.config.max_vis_images)
                ]
            ) # Use fixed set of evaluation prompts

        # Retrieve the reward function from implemented local scorers.
        print(f"Using reward function {self.config.reward_fn}")
        if self.config.reward_fn in {"image_reward", "imagereward", "image_reward_score"}:
            self.reward_fn = ImageRewardScorer(dtype=self.inference_dtype, device=self.accelerator.device)
        elif self.config.reward_fn in {"pick", "pick_score"}:
            self.reward_fn = PickScoreScorer(dtype=self.inference_dtype, device=self.accelerator.device)
        elif self.config.reward_fn in {"clip", "clip_score"}:
            self.reward_fn = CLIPScorer(dtype=self.inference_dtype, device=self.accelerator.device)
        else:
            raise NotImplementedError(
                f"Unsupported reward function: {self.config.reward_fn}. "
                "Supported scorers: image_reward, pick, clip"
            )

    def sample_images(self, train=False):
        """Sample images using the diffusion model."""
        self.pipeline.unet.eval()
        samples = []

        num_prompts_per_gpu = self.config.sample.batch_size

        # Generate prompts
        prompts, prompt_metadata = self.eval_prompts, self.eval_prompt_metadata
        print("prompts: ", prompts)

        sample_size = self.pipeline.unet.config.sample_size
        if isinstance(sample_size, int):
            latent_h, latent_w = sample_size, sample_size
        else:
            latent_h, latent_w = sample_size

        latents_0 = torch.randn(
            (
                self.config.sample.batch_size * self.config.max_vis_images,
                self.pipeline.unet.config.in_channels,
                latent_h,
                latent_w,
            ),
            device=self.accelerator.device,
            dtype=self.inference_dtype,
        )
        
        with torch.no_grad():
            for vis_idx in tqdm(
                range(self.config.max_vis_images),
                desc=f"Sampling images",
                disable=not self.accelerator.is_local_main_process,
                position=0,
            ):
                prompts_batch = prompts[vis_idx*num_prompts_per_gpu : (vis_idx+1)*num_prompts_per_gpu]

                latents_0_batch = latents_0[vis_idx*num_prompts_per_gpu : (vis_idx+1)*num_prompts_per_gpu]

                # Sample images
                with self.autocast():
                    if "xl" in self.config.pretrained.model:
                        result = pipeline_using_stein_sdxl(
                            self.pipeline,
                            prompt=prompts_batch,
                            num_inference_steps=self.config.sample.num_steps,
                            guidance_scale=self.config.sample.guidance_scale,
                            eta=self.config.sample.eta,
                            output_type="latent",
                            latents=latents_0_batch,
                            num_particles=getattr(self.config.sample, "num_particles", 1),
                            batch_p=getattr(self.config.sample, "batch_p", 1),
                            reward_fn=self.reward_fn,
                            stein_step=getattr(self.config.sample, "stein_step", 0.0),
                            stein_loop=getattr(self.config.sample, "stein_loop", 0),
                            stein_kernel=getattr(self.config.sample, "stein_kernel", "rbf"),
                            stein_adagrad_eps=getattr(self.config.sample, "stein_adagrad_eps", 1e-8),
                            stein_adagrad_clip=getattr(self.config.sample, "stein_adagrad_clip", None),
                            kl_coeff=getattr(self.config.sample, "kl_coeff", 1.0),
                            steer_start=getattr(self.config.sample, "steer_start", None),
                            steer_end=getattr(self.config.sample, "steer_end", None),
                            show_intermediate_rewards=getattr(self.config.sample, "show_intermediate_rewards", False),
                        )
                        latents_out = result.images if hasattr(result, "images") else result[0]
                        images = self._decode_latents_sdxl(latents_out)
                    else:
                        result = self.pipeline(
                            prompt=prompts_batch,
                            num_inference_steps=self.config.sample.num_steps,
                            guidance_scale=self.config.sample.guidance_scale,
                            eta=self.config.sample.eta,
                            output_type="pt",
                            latents=latents_0_batch,
                        )
                        images = result.images if hasattr(result, "images") else result[0]

                rewards = self.reward_fn(images, prompts_batch)

                self.info_eval_vis["eval_rewards_img"].append(rewards.clone().detach())
                self.info_eval_vis["eval_image"].append(images.clone().detach())
                self.info_eval_vis["eval_prompts"] = list(self.info_eval_vis["eval_prompts"]) + list(prompts_batch)

    def log_evaluation(self, epoch=None, inner_epoch=None):
        """Log results to the accelerator and external tracking systems."""

        self.info_eval = {k: torch.mean(torch.stack(v)) for k, v in self.info_eval.items()}
        self.info_eval = self.accelerator.reduce(self.info_eval, reduction="mean")

        ims = torch.cat(self.info_eval_vis["eval_image"])
        rewards = torch.cat(self.info_eval_vis["eval_rewards_img"])
        prompts = self.info_eval_vis["eval_prompts"]
        
        self.info_eval["eval_rewards"] = rewards.mean()
        self.info_eval["eval_rewards_std"] = rewards.std()

        self.accelerator.log(self.info_eval, step=self.global_step)

        images  = []
        for i, image in enumerate(ims):
            prompt = prompts[i]
            reward = rewards[i]
            if image.min() < 0: # normalize unnormalized images
                image = (image.clone().detach() / 2 + 0.5).clamp(0, 1)

            pil = Image.fromarray((image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
            if epoch is not None and inner_epoch is not None:
                caption = f"{epoch:03d}_{inner_epoch:03d}_{i:03d}_{prompt} | reward: {reward}"
            else:
                caption = f"{i:03d}_{prompt} | reward: {reward}"
            pil.save(f"{self.log_dir}/{caption}.png")

            pil = pil.resize((256, 256))
            caption = f"{prompt:.25} | {reward:.2f}"
            images.append(wandb.Image(pil, caption=caption)) 

        self.accelerator.log({"eval_images": images},step=self.global_step)

        # Log additional details if needed
        self.logger.info(f"Logged Evaluation results for step {self.global_step}")

    def run_evaluation(self):
        """Run sampling"""

        samples_per_eval = (
            self.config.sample.batch_size
            * self.accelerator.num_processes
            * self.config.max_vis_images
        )

        self.logger.info("***** Running Sampling *****")
        self.logger.info(f"  Sample batch size per device = {self.config.sample.batch_size}")

        self.logger.info("")
        self.logger.info(f"  Total number of samples for evaluation = {samples_per_eval}")

        self.logger.info(f"Using pre-trained model {self.config.pretrained.model}")



        self.info_eval = defaultdict(list)
        self.info_eval_vis = defaultdict(list)
        self.sample_images(train=False)

        # Log evaluation-related stuff
        self.log_evaluation()
