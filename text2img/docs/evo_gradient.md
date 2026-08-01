# Stein-Guided Sampling — Gradient-Based

The SD and SDXL pipelines implement the interacting ODE

$$
\frac{dX_t^{(i)}}{dt}
=u_t(X_t^{(i)})
+\frac{\lambda_t}{K}\sum_{j=1}^{K}
\left[
k(X_t^{(j)},X_t^{(i)})\nabla_{X_t^{(j)}}\log h_t(X_t^{(j)})
+\gamma_t\nabla_{X_t^{(j)}}k(X_t^{(i)},X_t^{(j)})
\right].
$$

## Parameter Mapping

- `stein_step` is the integrated $\lambda_t$ coefficient for one diffusion
  step.
- `stein_repulsion * repulsion_schedule` is $\gamma_t$.
- `reward / kl_coeff` defines $\log h_t$.
- `start <= step_index < end` selects the exclusive steering interval.
- `num_particles` is $K$; particles interact only within the same prompt
  group.

There is no inner Stein loop and no AdaGrad state or preconditioning.

## Per-Step Update

At each diffusion timestep:

1. Predict the base DDIM transition from the current particles. This supplies
   the discretized base drift $u_t$.
2. If the step is inside the steering interval, differentiate
   `reward / kl_coeff` with respect to each particle.
3. Evaluate the selected RBF, IMQ, or vMF Stein field once.
4. Apply the explicit guidance increment

   $$
   \Delta X_t=\texttt{stein\_step}\,\Phi_t(X_t).
   $$

5. Add this increment to the base DDIM proposal.

Thus the implemented first-order update is

$$
X_{t-1}=\operatorname{DDIM}(X_t)
+\texttt{stein\_step}\,\Phi_t(X_t).
$$

The default `eta=0` keeps the process deterministic as an ODE. A nonzero
`eta` remains available as an explicit stochastic DDIM extension.

## Numerical Details

- The RBF kernel uses the per-prompt median-distance bandwidth heuristic.
- Kernel bandwidths and particle norms are clamped away from zero.
- Non-finite field values are replaced with finite values using
  `torch.nan_to_num`.
- Reward gradients can be evaluated in particle micro-batches using `batch_p`.
- `end` is exclusive and defaults to the number of inference steps.
