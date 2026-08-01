"""Stein guidance vector field shared by the SD and SDXL pipelines.

For particle ``i`` this module implements

    (1 / K) sum_j [k(x_j, x_i) grad log h(x_j)
                   + gamma grad_{x_j} k(x_i, x_j)].

The diffusion drift is intentionally not part of this vector field.  It is
integrated by the diffusion scheduler in the calling pipeline.
"""

import math

import torch


SUPPORTED_STEIN_KERNELS = ("rbf", "imq", "vmf")


def stein_guidance_field(
    latents: torch.Tensor,
    log_h_grad: torch.Tensor,
    base_sample_count: int,
    num_particles: int,
    kernel: str,
    repulsion_strength: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Evaluate the attraction-plus-repulsion term in the Stein ODE.

    Particle interactions are restricted to particles belonging to the same
    base sample/prompt.  ``repulsion_strength`` is gamma_t; the caller applies
    lambda_t when integrating the returned field.
    """

    if kernel not in SUPPORTED_STEIN_KERNELS:
        choices = ", ".join(SUPPORTED_STEIN_KERNELS)
        raise ValueError(f"stein_kernel must be one of: {choices}.")
    if num_particles < 1:
        raise ValueError("num_particles must be >= 1")
    if latents.shape != log_h_grad.shape:
        raise ValueError("latents and log_h_grad must have the same shape.")
    if latents.ndim != 4:
        raise ValueError("Expected latents with shape [batch, channels, height, width].")

    batch, channels, height, width = latents.shape
    if batch != base_sample_count * num_particles:
        raise ValueError("Latent batch does not match base_sample_count * num_particles.")

    # For K=1, k(x, x)=1 and the self-kernel gradient is zero for all
    # supported kernels.
    if num_particles == 1:
        return log_h_grad

    latents_grouped = latents.reshape(base_sample_count, num_particles, -1)
    grads_grouped = log_h_grad.reshape(base_sample_count, num_particles, -1)
    fields = torch.empty_like(grads_grouped)

    for group_idx in range(base_sample_count):
        x = latents_grouped[group_idx]
        grad_log_h = grads_grouped[group_idx]

        if kernel == "vmf":
            norms = x.norm(dim=1, keepdim=True).clamp_min(eps)
            unit = x / norms
            cosine = unit @ unit.t()
            kernel_matrix = torch.exp(cosine - 1.0)

            attraction = kernel_matrix.t() @ grad_log_h

            # For k(x_i, x_j) = exp(<u_i,u_j>-1),
            # grad_{x_j} k = k / ||x_j|| * (u_i - <u_i,u_j>u_j).
            inv_source_norm = norms.reciprocal().squeeze(1)
            weights = kernel_matrix * inv_source_norm.unsqueeze(0)
            repulsion = unit * weights.sum(dim=1, keepdim=True)
            repulsion = repulsion - (weights * cosine) @ unit
        else:
            dist2 = torch.cdist(x, x).square()
            if kernel == "rbf":
                positive_dist2 = dist2[dist2 > 0]
                if positive_dist2.numel() == 0:
                    bandwidth = x.new_tensor(1.0)
                else:
                    bandwidth = positive_dist2.median()
                    bandwidth = bandwidth / (math.log(num_particles + 1.0) + eps)
                    bandwidth = bandwidth.clamp_min(eps)
                kernel_matrix = torch.exp(-dist2 / bandwidth)
                grad_weight = (2.0 / bandwidth) * kernel_matrix
            else:
                kernel_matrix = torch.rsqrt(1.0 + dist2)
                grad_weight = torch.pow(1.0 + dist2, -1.5)

            attraction = kernel_matrix.t() @ grad_log_h

            # Output row i is sum_j grad_{x_j} k(x_i, x_j).  The sign here
            # matters: x_i - x_j pushes particles apart.
            source_sum = grad_weight.t() @ x
            weight_sum = grad_weight.sum(dim=0, keepdim=True).t()
            repulsion = x * weight_sum - source_sum

        fields[group_idx] = (attraction + repulsion_strength * repulsion) / float(num_particles)

    return fields.reshape(batch, channels, height, width)
