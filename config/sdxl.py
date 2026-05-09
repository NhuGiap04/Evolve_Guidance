import ml_collections
import os
from config.general import general


def _base_config():
    config = general()

    config.sample.num_steps = 100
    config.sample.eta = 1.

    config.sample.batch_size = 1
    config.max_vis_images = 2

    # Stein transport guidance parameters.
    config.sample.num_particles = 4
    config.sample.batch_p = 1
    config.sample.stein_step = 0.02
    config.sample.stein_loop = 2
    config.sample.stein_kernel = "rbf"
    config.sample.stein_adagrad_eps = 1e-8
    config.sample.stein_adagrad_clip = None
    config.sample.kl_coeff = 0.0001
    config.sample.prediction_model = "default"
    config.sample.predicted_samples = 1
    config.sample.lookahead_steps = 10
    config.sample.steer_start = None
    config.sample.steer_end = None
    config.sample.monitor_status = False

    config.pretrained.model = "stabilityai/stable-diffusion-xl-base-1.0"

    return config


def clip():
    print("CLIP Score")
    config = _base_config()
    config.reward_fn = "clip"
    config.prompt_fn = "eval_hps_v2_all"

    return config


def pick():
    print("PickScore")
    config = _base_config()
    config.reward_fn = "pick"
    config.prompt_fn = "eval_hps_v2_all"

    return config


def image_reward():
    print("ImageReward")
    config = _base_config()
    config.reward_fn = "image_reward"
    config.prompt_fn = "eval_hps_v2_all"

    return config


def aesthetic():
    print("Aesthetic Score")
    config = _base_config()
    config.reward_fn = "aesthetic"
    config.prompt_fn = "eval_hps_v2_all"

    return config


def hpsv2():
    print("HPSv2 Score")
    config = _base_config()
    config.reward_fn = "hpsv2"
    config.prompt_fn = "eval_hps_v2_all"

    return config


REWARD_CONFIGS = {
    "pick": pick,
    "clip": clip,
    "image_reward": image_reward,
    "aesthetic": aesthetic,
    "hpsv2": hpsv2,
}


def get_config(name):
    if name not in REWARD_CONFIGS:
        valid = ", ".join(REWARD_CONFIGS)
        raise ValueError(f"Unsupported SDXL reward config: {name}. Choose one of: {valid}")
    return REWARD_CONFIGS[name]()
