"""Step-size schedules shared by the SD and SDXL guidance pipelines."""

from typing import Union


SUPPORTED_STEP_SCHEDULES = ("const", "linear_decay")


def step_schedule_scale(
    schedule: str,
    timestep: Union[int, float],
    first_timestep: Union[int, float],
    last_timestep: Union[int, float],
) -> float:
    """Return the multiplier for a reverse-diffusion guidance step.

    Flow-matching trajectory time runs from noise (tau=0) to data (tau=1),
    while diffusion timesteps run in the opposite direction.  For the linear
    schedule, mapping the current diffusion timestep ``t`` to that trajectory
    gives:

        tau = (t_first - t) / (t_first - t_last)
        gamma(t) = gamma_max * (1 - tau)

    Normalizing to the actual inference endpoints also supports custom,
    non-uniform timestep schedules and makes the last guidance step exactly
    zero.
    """

    if schedule == "const":
        return 1.0
    if schedule != "linear_decay":
        choices = ", ".join(SUPPORTED_STEP_SCHEDULES)
        raise ValueError(f"Unsupported step_schedule {schedule!r}. Choose one of: {choices}")

    denominator = float(first_timestep) - float(last_timestep)
    if denominator == 0.0:
        return 1.0

    scale = (float(timestep) - float(last_timestep)) / denominator
    return min(1.0, max(0.0, scale))
