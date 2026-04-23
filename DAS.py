from absl import app, flags
from ml_collections import config_flags

from DiffusionSampler import DiffusionModelSampler


FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("config", None, "Sampling configuration.", lock_config=False)


def main(_):
    sampler = DiffusionModelSampler(FLAGS.config)
    sampler.run_evaluation()


if __name__ == "__main__":
    app.run(main)
