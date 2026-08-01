import ast
import math
from pathlib import Path

import pytest
import torch

from text2img.diffusers_patch.stein_ode import stein_guidance_field


ROOT = Path(__file__).resolve().parents[2]


def _manual_field(x, grad_log_h, kernel_name, gamma):
    particle_count = x.shape[0]
    x_flat = x.flatten(1).detach().clone().requires_grad_(True)
    grad_flat = grad_log_h.flatten(1)

    dist2 = torch.cdist(x_flat.detach(), x_flat.detach()).square()
    positive = dist2[dist2 > 0]
    bandwidth = positive.median() / (math.log(particle_count + 1.0) + 1e-8)

    def kernel(first, second):
        squared_distance = (first - second).square().sum()
        if kernel_name == "rbf":
            return torch.exp(-squared_distance / bandwidth)
        if kernel_name == "imq":
            return torch.rsqrt(1.0 + squared_distance)
        first_unit = first / first.norm()
        second_unit = second / second.norm()
        return torch.exp((first_unit * second_unit).sum() - 1.0)

    outputs = []
    for i in range(particle_count):
        value = torch.zeros_like(x_flat[i])
        for j in range(particle_count):
            attraction_kernel = kernel(x_flat[j].detach(), x_flat[i].detach())
            # Detaching the first argument gives the required partial
            # derivative with respect to x_j, including when i == j.
            repulsion_kernel = kernel(x_flat[i].detach(), x_flat[j])
            repulsion_grad = torch.autograd.grad(
                repulsion_kernel, x_flat, retain_graph=True
            )[0][j]
            value = value + attraction_kernel * grad_flat[j] + gamma * repulsion_grad
        outputs.append(value / float(particle_count))

    return torch.stack(outputs).reshape_as(x)


@pytest.mark.parametrize("kernel", ["rbf", "imq", "vmf"])
def test_field_matches_ode_term(kernel):
    latents = torch.tensor(
        [[[[1.0, 0.5]]], [[[0.1, 1.4]]], [[[-0.8, 0.7]]]],
        dtype=torch.float64,
    )
    grad_log_h = torch.tensor(
        [[[[0.3, -0.2]]], [[[0.7, 0.4]]], [[[-0.1, 0.9]]]],
        dtype=torch.float64,
    )

    actual = stein_guidance_field(
        latents,
        grad_log_h,
        base_sample_count=1,
        num_particles=3,
        kernel=kernel,
        repulsion_strength=0.6,
    )
    expected = _manual_field(latents, grad_log_h, kernel, gamma=0.6)

    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)


def test_rbf_repulsion_pushes_two_particles_apart():
    latents = torch.tensor([[[[0.0]]], [[[2.0]]]])
    field = stein_guidance_field(
        latents,
        torch.zeros_like(latents),
        base_sample_count=1,
        num_particles=2,
        kernel="rbf",
        repulsion_strength=1.0,
    )

    assert field[0].item() < 0.0
    assert field[1].item() > 0.0


def test_single_particle_is_reward_attraction_only():
    latents = torch.randn(2, 1, 2, 2)
    grad_log_h = torch.randn_like(latents)
    field = stein_guidance_field(
        latents,
        grad_log_h,
        base_sample_count=2,
        num_particles=1,
        kernel="rbf",
        repulsion_strength=10.0,
    )

    torch.testing.assert_close(field, grad_log_h)


def test_particle_interactions_do_not_cross_prompt_groups():
    first_group = torch.tensor([[[[0.0]]], [[[1.0]]]])
    second_group = torch.tensor([[[[100.0]]], [[[103.0]]]])
    latents = torch.cat([first_group, second_group])
    grad_log_h = torch.tensor([[[[1.0]]], [[[2.0]]], [[[7.0]]], [[[11.0]]]])

    grouped = stein_guidance_field(
        latents,
        grad_log_h,
        base_sample_count=2,
        num_particles=2,
        kernel="imq",
        repulsion_strength=0.4,
    )
    first_only = stein_guidance_field(
        first_group,
        grad_log_h[:2],
        base_sample_count=1,
        num_particles=2,
        kernel="imq",
        repulsion_strength=0.4,
    )
    second_only = stein_guidance_field(
        second_group,
        grad_log_h[2:],
        base_sample_count=1,
        num_particles=2,
        kernel="imq",
        repulsion_strength=0.4,
    )

    torch.testing.assert_close(grouped, torch.cat([first_only, second_only]))


@pytest.mark.parametrize(
    ("relative_path", "function_name"),
    [
        ("diffusers_patch/pipeline_using_gradient_SD.py", "pipeline_using_gradient_sd"),
        ("diffusers_patch/pipeline_using_gradient_SDXL.py", "pipeline_using_gradient_sdxl"),
    ],
)
def test_pipeline_has_one_stein_evaluation_and_no_legacy_solver_args(relative_path, function_name):
    module = ast.parse((ROOT / "text2img" / relative_path).read_text(encoding="utf-8"))
    function = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == function_name
    )

    argument_names = {argument.arg for argument in function.args.args}
    legacy_names = {
        "stein_" + "loop",
        "stein_" + "adagrad_eps",
        "stein_" + "adagrad_clip",
    }
    assert argument_names.isdisjoint(legacy_names)

    field_calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "stein_guidance_field"
    ]
    assert len(field_calls) == 1
