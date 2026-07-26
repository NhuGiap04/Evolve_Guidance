"""Stein-repulsion schedules shared by the SD and SDXL guidance pipelines."""

from typing import Union


SUPPORTED_REPULSION_SCHEDULES = ("const", "linear_decay")


def repulsion_schedule_scale(
    schedule: str,
    timestep: Union[int, float],
    first_timestep: Union[int, float],
    last_timestep: Union[int, float],
) -> float:
    """Return the multiplier for Stein repulsion during reverse diffusion.

    Flow-matching trajectory time runs from noise (tau=0) to data (tau=1),
    while diffusion timesteps run in the opposite direction. For the linear
    schedule:

        tau = (t_first - t) / (t_first - t_last)
        repulsion(t) = repulsion_max * tau

    Normalizing to the actual inference endpoints also supports custom,
    non-uniform timestep schedules. Thus repulsion is zero at the initial
    noisy sample and reaches its configured strength at the final step.
    """

    if schedule == "const":
        return 1.0
    if schedule != "linear_decay":
        choices = ", ".join(SUPPORTED_REPULSION_SCHEDULES)
        raise ValueError(f"Unsupported repulsion_schedule {schedule!r}. Choose one of: {choices}")

    denominator = float(first_timestep) - float(last_timestep)
    if denominator == 0.0:
        return 0.0

    scale = (float(first_timestep) - float(timestep)) / denominator
    return min(1.0, max(0.0, scale))
