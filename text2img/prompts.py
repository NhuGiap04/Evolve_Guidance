from importlib import resources
import os
import functools
import random
import inflect

IE = inflect.engine()
PROMPTS_PATH = resources.files("text2img.prompts")

@functools.cache
def _load_lines(path):
    """
    Load lines from a file. First tries `path` directly, then bundled prompt files.
    """
    if not os.path.exists(path):
        newpath = PROMPTS_PATH.joinpath(path)
    else:
        newpath = path
    if not os.path.exists(newpath):
        raise FileNotFoundError(f"Could not find {path} or text2img/prompts/{path}")
    path = newpath
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]


def from_file(path, low=None, high=None, all=False):
    prompts = _load_lines(path)[low:high]
    return random.choice(prompts), {} 
    # return prompts, {}

def from_file_in_order(path, idx, low=None, high=None, all=False):
    prompts = _load_lines(path)[low:high]
    return prompts[idx % len(prompts)], {} 

def eval_hps_v2_all(idx):
    return from_file("hps_v2_all_eval.txt")
