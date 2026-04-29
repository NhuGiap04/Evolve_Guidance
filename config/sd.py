import ml_collections
from config.general import general


def seg():
    config = general()

    config.sample.num_steps = 100
    config.sample.eta = 1.0

    config.sample.batch_size = 1
    config.max_vis_images = 2

    # Stein transport guidance parameters.
    config.sample.num_particles = 4
    config.sample.batch_p = None
    config.sample.stein_step = 0.02
    config.sample.stein_loop = 2
    config.sample.stein_kernel = "rbf"
    config.sample.stein_adagrad_eps = 1e-8
    config.sample.stein_adagrad_clip = None
    config.sample.kl_coeff = 1.0
    config.sample.reward_guidance_rho = 1.0
    config.sample.steer_start = None
    config.sample.steer_end = None
    config.sample.intermediate_rewards = True

    config.pretrained.model = "runwayml/stable-diffusion-v1-5"
    config.pretrained.revision = "main"

    return config


def clip():
    print("CLIP Score")
    config = seg()
    config.reward_fn = "clip"
    config.prompt_fn = "eval_hps_v2_all"
    return config


def pick():
    print("PickScore")
    config = seg()
    config.reward_fn = "pick"
    config.prompt_fn = "eval_hps_v2_all"
    return config


def get_config(name):
    return globals()[name]()
