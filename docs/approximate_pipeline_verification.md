# Approximate EVO Pipeline Verification

This note records a code-level check of the Stable Diffusion approximation pipeline against `docs/evo_approximate.md`.

Checked files:

- `seg/diffusers_patch/pipeline_using_approximate_SD.py`
- `runs/single/approximate_sd.py`
- `run_sd_batch_approx_logged.sh`
- `config/sd.py`
- `config/general.py`

Validation run:

```bash
python -m py_compile seg/diffusers_patch/pipeline_using_approximate_SD.py runs/single/approximate_sd.py
```

Result: syntax check passed.

## Summary

The SD approximate pipeline does implement the main approximation idea: it builds clean `x0` anchors, evaluates rewards on decoded anchors, forms a reward-tilted Gaussian mixture score in log/softmax form, and uses that score inside an RBF SVGD update with AdaGrad.

There are still implementation gaps relative to the current document:

- The implementation uses exactly one anchor per particle. The documented `anchor_samples (L) = 1..4` is not exposed or implemented as multiple anchors per particle.
- The bandwidth floor and EMA schedule from Section 10 are not implemented in `pipeline_using_approximate_SD.py`.
- Runtime defaults differ from the documented recommended defaults. The pipeline default `stein_step` is `0.05`, `config/sd.py` uses `0.02`, and `run_sd_batch_approx_logged.sh` uses `0.005`.
- The pipeline contains unconditional `[DEBUG]` prints, so normal runs are noisy even when intermediate reward logging is not requested.
- `reward_guidance_rho` and `reward_scale_fixed` remain in the function signature, but they are ignored when `use_approximate_score=True`.

## What Matches The Approximation Spec

### Particle setup

`pipeline_using_approximate_sd` expands prompt embeddings and latents from `base_sample_count` to `base_sample_count * num_particles`. It groups particles by prompt before Stein interactions, so particles from different prompts do not share kernels or mixture scores.

Relevant code:

- `num_particles` default and input validation: `pipeline_using_approximate_SD.py:245`, `pipeline_using_approximate_SD.py:303`
- latent replication and jitter when base latents are supplied: `pipeline_using_approximate_SD.py:446`
- prompt expansion: `pipeline_using_approximate_SD.py:479`
- per-prompt Stein grouping: `pipeline_using_approximate_SD.py:137`

### Clean anchor generation

The pipeline supports these anchor modes:

- `base`: one base-model `x0` prediction from the current `x_t`
- `dpm`: a short DPMSolver rollout toward `x0`
- `lcm`: an LCMScheduler rollout, requiring `x0_anchor_lora_path`

Relevant code:

- anchor options and validation: `pipeline_using_approximate_SD.py:536`
- DPM/LCM scheduler setup: `pipeline_using_approximate_SD.py:545`
- anchor timestep construction: `pipeline_using_approximate_SD.py:684`
- anchor prediction: `pipeline_using_approximate_SD.py:719`
- reward evaluation on decoded anchors: `pipeline_using_approximate_SD.py:823`

This matches the spec's "base / DPM / LCM" anchor-source design, except that there is no multi-anchor sampling dimension `L`.

### Soft good-conditioned mixture score

The core approximation is implemented in `_approximate_score`.

The implementation computes:

- Gaussian forward log density using `x_t`, anchor `z`, `sqrt(alpha_bar_t)`, and `1 - alpha_bar_t`
- reward tilt using `reward / kl_coeff`
- normalized mixture weights with `torch.softmax(log_a, dim=1)`
- weighted score terms `-(x - sqrt(alpha_bar_t) * z) / (1 - alpha_bar_t)`

Relevant code:

- grouping latents, anchors, rewards: `pipeline_using_approximate_SD.py:863`
- reward temperature scale via `kl_coeff`: `pipeline_using_approximate_SD.py:868`
- log mixture terms and softmax weights: `pipeline_using_approximate_SD.py:874`
- weighted mixture score: `pipeline_using_approximate_SD.py:879`

This matches Sections 3 and 4 of `docs/evo_approximate.md`.

### Stein transport with AdaGrad

The approximate score is used as the target score in the RBF Stein vector field, then applied through an AdaGrad-preconditioned update.

Relevant code:

- approximate score selection: `pipeline_using_approximate_SD.py:974`
- RBF Stein vector field: `pipeline_using_approximate_SD.py:120`
- attraction and repulsion terms: `pipeline_using_approximate_SD.py:160`
- AdaGrad accumulator and adaptive step: `pipeline_using_approximate_SD.py:1040`
- latent update: `pipeline_using_approximate_SD.py:1045`

This matches Section 6 of the spec.

### Off-manifold correction

After Stein refinement, the code predicts `x0` again from the steered latent and reconstructs the next latent using a DDIM-style update.

Relevant code:

- new noise prediction from steered latent: `pipeline_using_approximate_SD.py:1093`
- clean prediction from steered latent: `pipeline_using_approximate_SD.py:1094`
- next latent from corrected `pred_x0`, original scheduler noise direction, and optional DDIM noise: `pipeline_using_approximate_SD.py:1099`

This is consistent with Section 7: the clean component comes from the Stein-refined latent, while the noise direction may come from the base prediction before or after correction depending on scheduler implementation. Here the noise direction used in the final transition is the pre-Stein `noise_pred` computed at the beginning of the timestep.

### Runner wiring

`runs/single/approximate_sd.py` imports and calls `pipeline_using_approximate_sd` with `use_approximate_score=True`, passes the anchor settings, and returns all particles by default.

Relevant code:

- import: `runs/single/approximate_sd.py:14`
- CLI anchor flags: `runs/single/approximate_sd.py:120`
- config override for anchor settings: `runs/single/approximate_sd.py:514`
- pipeline call kwargs: `runs/single/approximate_sd.py:625`
- `use_approximate_score=True`: `runs/single/approximate_sd.py:653`
- `return_all_particles=True`: `runs/single/approximate_sd.py:654`

`run_sd_batch_approx_logged.sh` points to `runs/single/approximate_sd.py` by default and sets DPM anchors with two anchor steps by default.

Relevant code:

- default single-run script: `run_sd_batch_approx_logged.sh:11`
- default anchor model and steps: `run_sd_batch_approx_logged.sh:35`

## Mismatches And Risks

### 1. No `anchor_samples` implementation

The spec describes `L = 1..4` anchors per particle. The code currently produces one anchor per particle, then reshapes rewards and anchors as `(base_sample_count, num_particles, ...)`.

Impact: the approximation is lower-sample than documented. It uses `N_a = K` anchors per prompt group, not `N_a = K * L`.

Recommended fix: add an `anchor_samples` argument, generate repeated anchor predictions per particle, evaluate all rewards, and update `_approximate_score` to group anchors as `(base_sample_count, num_particles * anchor_samples, ...)`.

### 2. Bandwidth floor and EMA are not implemented

Section 10 specifies:

- median heuristic
- optional `h_bandwidth` multiplier
- noise-schedule floor using `stein_bw_floor_scale`
- EMA smoothing using `stein_bw_ema_decay`

The SD approximate pipeline only implements median bandwidth. If `stein_bandwidth == "sigma_t"`, it multiplies the median bandwidth by scheduler `sigma_t`; it does not implement the documented floor or EMA.

Impact: late-timestep particle convergence can still collapse the kernel bandwidth. The behavior differs from the current document.

Recommended fix: add `h_bandwidth`, `stein_bw_floor_scale`, and `stein_bw_ema_decay` parameters, maintain per-prompt or scalar EMA state across steered timesteps, and apply the noise floor before passing bandwidth to the RBF field.

### 3. Defaults are inconsistent

Documented defaults:

- `stein_step = 0.002` to `0.005`
- `alpha = kl_coeff`
- `x0_anchor_model = "base"`
- `x0_anchor_steps = 1` to `4`

Current code/config defaults:

- pipeline function default: `stein_step = 0.05`
- `config/sd.py`: `stein_step = 0.02`, `kl_coeff = 0.0001`
- `run_sd_batch_approx_logged.sh`: `STEIN_STEP = 0.005`, `X0_ANCHOR_MODEL = dpm`, `X0_ANCHOR_STEPS = 2`

Impact: behavior depends heavily on the entry point. The shell script matches the recommended step range, but the Python defaults are much larger.

Recommended fix: align `pipeline_using_approximate_sd` and `config/sd.py` with the documented defaults, or document that the shell script is the canonical approximation entry point.

### 4. Unconditional debug prints

The pipeline prints setup and return messages even when verbose/intermediate reward logging is not requested.

Impact: batch logs are noisier and downstream scripts that parse stdout may be harder to use.

Recommended fix: gate these prints behind `intermediate_rewards` or a dedicated `verbose` argument.

### 5. Legacy gradient-guidance knobs are inert in approximate mode

When `use_approximate_score=True`, the code sets `reward_grad` and `reward_scale` to zero placeholders. `reward_guidance_rho` and `reward_scale_fixed` only affect the old gradient-score branch.

Impact: users may tune `reward_scale_fixed` expecting it to affect approximate guidance, but it will not.

Recommended fix: either remove/hide these knobs for approximate runs or document that approximate guidance uses `kl_coeff` as the reward temperature instead.

## Verification Conclusion

The active SD approximate pipeline is structurally implementing the soft good-probability Stein approximation. The most important mathematical path is present:

```text
x_t -> x0 anchors -> reward h(z) -> log-space reward-tilted Gaussian mixture score -> SVGD -> AdaGrad -> corrected DDIM transition
```

The main remaining work is not a syntax or wiring issue. It is alignment work between the document and implementation: multi-anchor sampling, bandwidth floor/EMA, and default cleanup.
