# Offline RL Evaluation Guidance Formulas

This note documents the formulas used by:

- `run_scripts/eval_gradient.sh`
- `run_scripts/eval_stein.sh`
- `run_scripts/eval_smc.sh`

All three scripts call `run/eval.py`, which builds a `FlowPolicy` from
`gflower/models_flow/flow_policy.py`. The trajectory tensor is normalized and
has shape

```text
x_t in R^{B x H x C},    C = action_dim + state_dim,
x_t[h] = [a_t[h], o_t[h]].
```

The initial observation condition is written as `c` and is enforced at horizon
index `0` by setting the observation block of `x[:, 0, :]` to the normalized
environment observation. During integration, the corresponding velocity entries
are zeroed so the condition remains fixed.

Let:

- `v_theta(x,t)` be the learned flow model.
- `V_phi(x)` be the value model output over a trajectory.
- `J_phi(x) = V_phi(x)[:, -1, 0]` be the scalar terminal value used by guidance
  and final selection.
- `s(t)` be the selected gradient schedule.
- `alpha` be `--grad_scale`.
- `\Delta t` be the Euler step size from `torch.linspace(*ode_t_span,
  ode_t_steps)`.

## Shared Flow Step

Without guidance, sampling starts from Gaussian noise and follows the Euler
discretization of the learned flow:

```math
x_0 \sim N(0, I),
```

```math
x_{k+1}
= x_k + \Delta t \, C_c(v_\theta(x_k, t_k)),
```

where `C_c(...)` means "apply conditioning": the observation velocity at the
conditioned horizon index is set to zero.

With the default evaluation settings in the scripts:

```math
H = 20, \qquad t_k \in \operatorname{linspace}(0,1,10).
```

The scripts evaluate both `cfm` and `ot_cfm` trained flow checkpoints. At
evaluation time this changes which trained `v_theta` checkpoint is loaded; the
sampling formulas below are otherwise the same.

## Gradient Schedule

The implementation uses `get_scheduler` in `FlowPolicy`:

```math
s_{const}(t) = t,
```

```math
s_{linear_decay}(t) = 1 - t,
```

```math
s_{cosine_decay}(t) = 0.5 \, (1 + \cos(\pi t)),
```

```math
s_{exp_decay}(t)
= \frac{\exp(-t) - \exp(-1)}{1 - \exp(-1)}.
```

Note: `const` is named like a constant schedule in the script, but the current
code returns `s(t)=t`.

## Value Gradient Used By Both Methods

The helper `_compute_value_gradient(...)` supports two places to evaluate the
value model.

### `grad_compute_at = x_t`

The value is evaluated directly at the current ODE state:

```math
J_t = J_\phi(x_t).
```

The only supported derivative target in this case is `grad_wrt = x_t`:

```math
g_t
= \alpha \, s(t) \, \nabla_{x_t} J_\phi(x_t).
```

The combination `grad_compute_at = x_t` and `grad_wrt = x_1` is skipped by both
scripts because the implementation raises an error for it.

### `grad_compute_at = x_1`

The code first predicts a first-order endpoint using the unguided flow velocity:

```math
\hat{x}_1(x_t,t)
= x_t + (1 - t) \, v_\theta(x_t,t).
```

The scalar objective is:

```math
J_1 = J_\phi(\hat{x}_1).
```

If `grad_wrt = x_1`, the guidance vector is the endpoint-space value gradient:

```math
g_t
= \alpha \, s(t) \, \nabla_{\hat{x}_1} J_\phi(\hat{x}_1).
```

If `grad_wrt = x_t`, autograd differentiates through
`\hat{x}_1 = x_t + (1-t)v_\theta(x_t,t)`:

```math
g_t
= \alpha \, s(t) \, \nabla_{x_t} J_\phi(\hat{x}_1(x_t,t)).
```

Equivalently:

```math
g_t
= \alpha \, s(t)
   \left(I + (1-t) \frac{\partial v_\theta(x_t,t)}{\partial x_t}\right)^T
   \nabla_{\hat{x}_1} J_\phi(\hat{x}_1).
```

## `eval_smc.sh`: Sequential Monte Carlo

SMC evolves `K` trajectory particles through the unguided flow ODE. At ODE
time `t`, it scores a first-order endpoint prediction

```math
\hat{x}_1 = x_t + (1-t)v_\theta(x_t,t)
```

and defines the annealed potential

```math
G_t(x_t) = \alpha t J_\phi(\hat{x}_1).
```

Log importance weights are updated incrementally by

```math
\log w_t = \log w_{t-1} + G_t(x_t) - G_{t-1}(x_{t-1}).
```

At configured ODE intervals, the implementation computes
`ESS = 1 / sum(normalized_weight^2)` independently for each environment and
uses systematic resampling when `ESS <= smc_ess_threshold * K`. It does not
resample after the final integration step. As in the other locomotion methods,
the policy executes the first action of the surviving particle with the highest
final value-model score.

## `eval_gradient.sh`: Direct Value-Gradient Guidance

`eval_gradient.sh` sets:

```text
guidance_method = gradient
batch_size = grad_particles = 8
ode_t_steps = 10
grad_scale in {0.0, 0.01, 0.1, 1.0}
grad_schedule in {cosine_decay, const, linear_decay, exp_decay}
grad_compute_at in {x_1, x_t}
grad_wrt in {x_1, x_t}, except x_t -> x_1 is skipped
```

At each Euler step, the guidance vector `g_t` above is added directly to the
flow velocity:

```math
u_t^{gradient}
= v_\theta(x_t,t) + g_t.
```

The implemented conditioned Euler update is:

```math
x_{k+1}
= x_k + \Delta t \, C_c(u_{t_k}^{gradient}).
```

After integration, the policy evaluates all generated candidates using the
terminal value:

```math
j^* = \arg\max_j J_\phi(x^{(j)}).
```

The returned environment action is the first action of the best trajectory:

```math
a = a^{(j^*)}_0.
```

## `eval_stein.sh`: Grouped RBF Stein Particle Steering

`eval_stein.sh` sets:

```text
guidance_method = stein
stein_particles = K = 8
stein_loop = 1
stein_step = eta = 0.02
stein_kernel = rbf
ode_t_steps = 10
grad_scale in {0.01, 0.1, 1.0}
grad_schedule in {cosine_decay, const, linear_decay, exp_decay}
grad_compute_at in {x_1, x_t}
grad_wrt in {x_1, x_t}, except x_t -> x_1 is skipped
```

For each environment batch item, the method keeps `K` trajectory particles.
Write the flattened trajectory particle at a fixed ODE time as:

```math
z_i = \operatorname{flatten}(x_i) \in \mathbb{R}^{H C}, \qquad i = 1,\ldots,K.
```

First, the same value-gradient helper above is computed for each particle:

```math
q_i = \operatorname{flatten}(g_t(x_i)).
```

The RBF bandwidth uses the median heuristic over positive pairwise squared
distances:

```math
d_{ij}^2 = \lVert z_i - z_j \rVert_2^2,
```

```math
h =
\frac{\operatorname{median}(\{d_{ij}^2 : d_{ij}^2 > 0\})}
     {\log(K+1) + \epsilon},
```

with `h = 1` if all positive distances are absent, and `h` clamped below by
`epsilon`.

The kernel is:

```math
k_{ij} = \exp(-d_{ij}^2 / h).
```

The implemented Stein vector field for particle `j` is:

```math
\phi_j
= \frac{1}{K} \sum_{i=1}^K
   \left[ k_{ij} q_i + \frac{2}{h} k_{ij} (z_i - z_j) \right].
```

The first term transports particles along the value-gradient scores. The second
term is the RBF kernel-gradient term as implemented in `_rbf_stein_vector_field`
and named `repulsion` in code.

Inside each ODE time step, the Stein loop applies an AdaGrad-style update:

```math
G_j^{(l)} = G_j^{(l-1)} + \phi_j^{(l)} \odot \phi_j^{(l)},
```

```math
\rho_j^{(l)}
= \frac{\eta}{\sqrt{G_j^{(l)}} + \epsilon_{\mathrm{AdaGrad}}},
```

```math
z_j^{(l+1)}
= z_j^{(l)} + \rho_j^{(l)} \odot \phi_j^{(l)}.
```

The code then reshapes the particles back to trajectories and reapplies the
observation condition. With the script default `stein_loop = 1`, this happens
once before each flow step.

After Stein steering at time `t_k`, the method takes an unguided conditioned
Euler flow step:

```math
x_{k+1}^{(j)}
= x_k^{(j)} + \Delta t \, C_c(v_\theta(x_k^{(j)}, t_k)).
```

At the end, values are reshaped by environment batch and particle index:

```math
j_b^* = \arg\max_{j=1,\ldots,K} J_\phi(x_b^{(j)}).
```

For the single-environment rollout used by `run/eval.py`, the returned action is:

```math
a = a_0^{(j_0^*)}.
```

## Main Difference

Direct gradient guidance changes the ODE velocity:

```math
v_\theta(x,t) \rightarrow v_\theta(x,t) + g_t.
```

Stein guidance first moves a group of particles with the RBF Stein field, then
uses the original flow velocity for the Euler step:

```math
x \rightarrow x + \rho \odot \phi, \qquad
x \rightarrow x + \Delta t \, C_c(v_\theta(x,t)).
```
