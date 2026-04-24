# EVO v1 Plan: DAS -> Stein-Transport Sampling (No SMC)

This document specifies the implementation plan for replacing the current SMC-style sampling logic with a Stein-transport update while keeping the SDXL denoising structure from DAS.

## 1. Goal and Scope

Target behavior for sampling:

1. Initialize `K` particles per prompt.
2. Keep the same base diffusion sampling skeleton as DAS.
3. Remove SMC-specific components:
	- No particle weights.
	- No ESS tracking.
	- No resampling step.
4. At each steered timestep, run an inner Stein loop of length `M`:
	- Update particles with an SVGD-style vector field.
	- Use AdaGrad-style adaptive step size.
5. After the Stein refinement at timestep $t$, propose $x_{t-1}$ with the manifold-preserving form:

$$
x_{t-1} = \sqrt{\bar\alpha_{t-1}}\,x_{0|t} + \sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\,\epsilon_\theta(x_t^{\mathrm{pre}},t) + \sigma_t z,
\quad z \sim \mathcal N(0, I).
$$

6. Add steering range arguments `(start, end)` with defaults `start=T`, `end=0`.

## 2. Delta vs DAS (Current `dev/das.md`)

Keep:

- Prompt encoding, CFG handling, UNet noise prediction flow.
- DDIM scheduler/timestep handling.
- Reward model interface (`reward_fn(images, prompts)`).

Remove:

- Importance weight updates ($log_w$, $log_Z$, incremental log-weight math).
- ESS computation and thresholding.
- Resampling and ancestry bookkeeping.
- Diffusion/proposal log-prob tracking used only for SMC weighting.

Add:

- Stein transport update for particles at each steered timestep.
- Per-particle AdaGrad optimizer state (inside timestep Stein loop).
- Steering range gate from `start` to `end`.

## 3. API and Config Changes

## 3.1 Pipeline Call Arguments

In `seg/diffusers_patch/pipeline_using_Stein_SDXL.py`, keep existing Stein args and add/refine:

- `num_particles: int = 4`
- `stein_loop: int = 1` (this is `M`)
- `stein_step: float = 0.05` (base learning rate)
- `stein_adagrad_eps: float = 1e-8`
- `stein_adagrad_clip: Optional[Tuple[float, float]] = None` (optional min/max clamp for adaptive step)
- `steer_start: Optional[int] = None`
- `steer_end: Optional[int] = None`
- `reward_scale_mode: str = "das"` (use DAS-consistent scaling by default)

Default steering behavior:

- If `steer_start is None`: use first scheduler timestep (`T`).
- If `steer_end is None`: use last scheduler timestep (`0`-side).
- Effective steering interval on descending timesteps: `steer_end <= t <= steer_start`.

Validation:

- Ensure `steer_start >= steer_end` for descending schedules.
- Ensure both boundaries exist in or are clamped to scheduler timestep range.

## 3.2 Config Surface

Expose in config (via `config/general.py` and preset overrides):

- `sample.num_particles`
- `sample.stein_loop`
- `sample.stein_step`
- `sample.stein_adagrad_eps`
- `sample.steer_start`
- `sample.steer_end`

Pass through in runner callsites (`runs/single/gradient_sdxl.py`, `runs/single/gradient_sd.py`) when invoking pipeline.

## 4. Mathematical Formulation to Implement

Given particles $\{x_t^{(j)}\}_{j=1}^K$, define Stein vector field:

$$
\hat\phi_t(x)
= \frac{1}{K}\sum_{j=1}^{K}
\left[
k(x_t^{(j)},x)\,\nabla_{x_t^{(j)}}\log q_t(x_t^{(j)}\mid c)
+ \nabla_{x_t^{(j)}}k(x_t^{(j)},x)
\right].
$$

Intermediate target:

$$
q_t(x\mid c) \propto p_t(x\mid c)\,\exp\big(\beta_t\,r_{\text{int}}(x)\big),
$$

where $\beta_t$ follows DAS-style scaling (recommended to match existing `kl_coeff` convention).

Practical score decomposition:

$$
\nabla_x \log q_t(x\mid c)
= \nabla_x \log p_t(x\mid c)
+ \beta_t\,\nabla_x r_{\text{int}}(x).
$$

- Estimate $\nabla_x \log p_t$ from diffusion model outputs at timestep `t`.
- Compute $\nabla_x r_{\text{int}}$ by autograd through decode and reward scorer.

AdaGrad update inside Stein loop:

$$
G \leftarrow G + \hat\phi_t(x)^2,
\quad
\eta_{\text{adapt}} = \frac{\text{stein\_step}}{\sqrt{G} + \epsilon},
\quad
x_t \leftarrow x_t + \eta_{\text{adapt}}\odot \hat\phi_t(x).
$$

## 5. Per-Timestep Algorithm (New)

For each timestep `t` in scheduler order (descending):

1. Split particles by prompt group (`B x K`) to avoid cross-prompt kernel coupling.
2. If `t` is outside steering range:
	- Do regular DDIM-style transition to $x_{t-1}$.
3. If `t` is inside steering range:
	- Initialize timestep-local AdaGrad accumulator `G = 0` (shape matches particle latents).
	- Repeat `m = 1..M` (`M = stein_loop`):
	  - Compute diffusion score term from current particles at timestep `t`.
	  - Compute intermediate reward score by autograd on current particles.
	  - Build `score_q`.
	  - Compute SVGD direction using kernel over particles in the same prompt group.
	  - AdaGrad update: `x_t <- x_t + eta_adapt * step`.
	- Keep pre-Stein noise prediction $\epsilon_\theta(x_t^{\mathrm{pre}}, t)$ for the proposal noise term.
	- Propose $x_{t-1}$ via manifold-preserving transition:

$$
x_{t-1} = \sqrt{\bar\alpha_{t-1}}\,x_{0|t} + \sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\,\epsilon_\theta(x_t^{\mathrm{pre}},t) + \sigma_t z.
$$

4. Continue to next timestep.

At final step, decode particles and pick output policy:

- Default simple policy for v1: best reward particle per prompt.
- Optionally keep all particles/trace outputs for analysis.

## 6. Kernel and Numerical Details

Kernel recommendation for v1:

- RBF with median heuristic bandwidth per prompt group.

Let flattened particle vectors be $u_i$.

$$
k(u_i,u_j) = \exp\left(-\frac{\|u_i-u_j\|^2}{h}\right),
\quad
h = \frac{\mathrm{median}_{i\neq j}\|u_i-u_j\|^2}{\log(K+1)+\delta}.
$$

Stability controls:

- Clamp `h` to a minimum value.
- Replace NaN/Inf in scores and steps using `nan_to_num`.
- Optional norm clip on `step` before AdaGrad update.
- Keep reward-gradient compute in micro-batches (`batch_p`) to control memory.

## 7. Pseudocode (Implementation-Oriented)

```python
timesteps = scheduler.timesteps  # descending
steer_start_eff = timesteps[0] if steer_start is None else steer_start
steer_end_eff = timesteps[-1] if steer_end is None else steer_end

latents = init_latents(batch_size * num_particles)

for t in timesteps:
	 noise_pred_pre = predict_noise(latents, t)
	 is_steered = (steer_end_eff <= int(t) <= steer_start_eff)

	 if is_steered:
		  G = torch.zeros_like(latents)

		  for _ in range(stein_loop):
				score_prior = estimate_diffusion_score(latents, t)
				score_reward = grad_intermediate_reward(latents, t, reward_fn)
				score_q = score_prior + beta_t(t) * score_reward

				step = stein_vector_field(latents, score_q, kernel=stein_kernel)
				step = torch.nan_to_num(step)

				G = G + step * step
				eta_adapt = stein_step / (torch.sqrt(G) + stein_adagrad_eps)
				if stein_adagrad_clip is not None:
					 eta_adapt = eta_adapt.clamp(*stein_adagrad_clip)

				latents = latents + eta_adapt * step

	 x0_pred = predict_x0(latents, t, steered_noise_pred)  # x_(0|t)
	 sigma_t = scheduler_sigma(t, eta)
	 alpha_bar_prev = scheduler_alpha_bar_prev(t)

	 z = torch.randn_like(latents)
	 latents = (
		  torch.sqrt(alpha_bar_prev) * x0_pred
		  + torch.sqrt(1.0 - alpha_bar_prev - sigma_t**2) * noise_pred_pre
		  + sigma_t * z
	 )
```

## 8. Code Change Plan by File

1. `seg/diffusers_patch/pipeline_using_gradient_SDXL.py` and `seg/diffusers_patch/pipeline_using_gradient_SD.py`
	- Implement full particle-aware Stein loop in denoising steps.
	- Remove any remaining SMC bookkeeping assumptions.
	- Add steering range argument handling (`steer_start`, `steer_end`).
	- Implement manifold-preserving proposal update.
	- Return optional traces (rewards, step norms, particle latents) for debugging.

2. `runs/single/gradient_sdxl.py` and `runs/single/gradient_sd.py`
	- Pass sampling arguments from config to pipeline call.
	- Keep existing reward scorer wiring.

3. `config/general.py` and `config/sdxl.py`
	- Add defaults for Stein and steering parameters.
	- Optionally define presets for quick ablations (`stein_loop`, `stein_step`, steering range).

4. `README.md` (or future note)
	- Document new sampler mode and parameter meanings.

## 9. Validation Plan

Functional checks:

1. `steer_start=None, steer_end=None` applies steering across full trajectory.
2. `steer_start=t_mid, steer_end=t_low` only steers in that band.
3. `num_particles=1` reduces to single-particle guided trajectory and does not crash.
4. No SMC traces/weights are produced or required.

Numerical checks:

1. No NaN/Inf in latents, scores, or step sizes.
2. AdaGrad denominator stability with epsilon.
3. Proposal variance term is valid: `1 - alpha_bar_prev - sigma_t^2 >= 0` (with clamp to zero if small negative from float error).

Quality checks:

1. Compare reward traces against baseline DDIM and old DAS-SMC.
2. Inspect diversity collapse risk as `stein_loop`/`stein_step` increase.
3. Monitor manifold drift by latent norm and decode quality.

## 10. Suggested Initial Hyperparameters

- `num_particles = 4`
- `stein_loop = 2`
- `stein_step = 0.02`
- `stein_adagrad_eps = 1e-8`
- `steer_start = None` (defaults to `T`)
- `steer_end = None` (defaults to `0`)

Then tune in this order:

1. `stein_step`
2. `stein_loop`
3. steering interval `(start, end)`
4. reward scaling (`kl_coeff`/`beta_t` schedule)
