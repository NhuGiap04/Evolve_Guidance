import ml_collections

from .general import general


def _base_config():
    config = general()

    config.sample.num_steps = 100
    # eta=0 keeps the base DDIM dynamics deterministic, as required by the
    # interacting ODE. Nonzero eta remains available as a stochastic extension.
    config.sample.eta = 0.0

    config.sample.batch_size = 1
    config.max_vis_images = 2

    # Stein transport guidance parameters.
    config.sample.num_particles = 4
    config.sample.batch_p = 1
    config.sample.stein_step = 0.02
    config.sample.repulsion_schedule = "const"
    config.sample.stein_kernel = "rbf"
    config.sample.stein_repulsion = 1.0
    config.sample.kl_coeff = 0.0001
    config.sample.soft_temperature = None
    config.sample.start = 0
    config.sample.end = config.sample.num_steps

    config.pretrained.model = "runwayml/stable-diffusion-v1-5"
    config.pretrained.revision = "main"

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
        raise ValueError(f"Unsupported SD reward config: {name}. Choose one of: {valid}")
    return REWARD_CONFIGS[name]()
