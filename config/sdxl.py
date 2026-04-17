import ml_collections
import os
from config.general import general

def seg():
    config = general()

    config.sample.num_steps = 100
    config.sample.eta = 1.

    config.sample.batch_size = 1
    config.max_vis_images = 2

    config.pretrained.model = "stabilityai/stable-diffusion-xl-base-1.0"

    return config


def clip():
    print("CLIP Score")
    config = seg()
    config.reward_fn = "clip"
    config.prompt_fn = "eval_hps_v2_all"
    
    config.smc.kl_coeff = 0.01

    return config

def pick():
    print("PickScore")
    config = seg()
    config.reward_fn = "pick"
    config.prompt_fn = "eval_hps_v2_all"
    
    config.smc.kl_coeff = 0.0001

    return config

def get_config(name):
    return globals()[name]()
