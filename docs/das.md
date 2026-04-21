# SMC for SDXL: Detailed Algorithm Documentation

This document explains the algorithm implemented in [references/das/pipeline_using_SMC_SDXL.py](references/das/pipeline_using_SMC_SDXL.py), with explicit mapping to the mathematical formulation you provided.

## 1. Objective and Target Distribution

The pipeline uses Sequential Monte Carlo (SMC) over diffusion latents to bias generation toward high-reward outputs while staying close to the diffusion dynamics.

Let:

- $x_t$: latent at diffusion time $t$.
- $p_t(x_t)$: forward diffusion marginal at time $t$.
- $\hat x_0(x_t)$: model-predicted clean sample from latent $x_t$.
- $r(\cdot)$: reward function on decoded images.
- $\alpha$: temperature / KL trade-off coefficient (implemented as kl_coeff).

From your Eq. (10), the approximate posterior used for guidance is:

$$
\hat p_t(x_t \mid \mathcal O) \propto p_t(x_t)\,\exp\!\left(\frac{\hat r(x_t)}{\alpha}\right),
$$

where $\hat r(x_t)$ is the reward estimated from the decoded prediction of $x_0$.

From your Eq. (11), tempering defines intermediate targets:

$$
\pi_t(x_t) \propto p_t(x_t)\,\exp\!\left(\lambda_t\frac{\hat r(x_t)}{\alpha}\right),
$$

with $0 = \lambda_T \le \cdots \le \lambda_0 = 1$.

In code:

- $\hat r/\alpha$ is implemented as:
	- lookforward_fn = lambda r: r / kl_coeff
	- log_twist_func = lookforward_fn(rewards)
- $\lambda_t$ is implemented as scale_factor (or min_scale_next depending on branch), then applied by:
	- log_twist_func *= scale_factor


## 2. Proposal: Locally Optimal Idea and Gaussian Approximation

Your Eq. (12) gives a locally optimal proposal density of the form:

$$
m^*_{t-1}(x_{t-1}\mid x_t) \propto
\exp\!\left(
-\frac{\|x_{t-1}-\mu_\theta(x_t,t)\|^2}{2\sigma_t^2}
+\lambda_{t-1}\frac{\hat r(x_{t-1})}{\alpha}
\right).
$$

Because direct sampling from $m^*$ is hard, your Eq. (13) uses a Gaussian approximation:

$$
m_{t-1}(x_{t-1}\mid x_t)
= \mathcal N\!\left(
\mu_\theta(x_t,t)
+ \sigma_t^2\lambda_{t-1}\frac{1}{\alpha}\nabla_{x_t}\hat r(x_t),
\sigma_t^2 I
\right).
$$

Implementation mapping:

1. Base DDIM mean / sample are computed with ddim_step_with_mean.
2. The reward gradient is approximated by autograd on latents:

	 $$
	 \mathrm{approx\_guidance} = \nabla_{x_t}\left(\frac{r}{\alpha}\right).
	 $$

3. Proposal mean shift is applied by:

	 $$
	 x_{t-1}^{\text{prop}} = x_{t-1}^{\text{ddim}} + \sigma_t^2\,\texttt{approx_guidance}.
	 $$

	 In code this is:
	 - variance = eta^2 * get_variance(...)
	 - prop_latents = prev_sample + variance * approx_guidance

Thus, the script implements Eq. (13) in latent space using DDIM variance and gradient-based reward guidance.


## 3. Full Per-Step SMC Procedure (as Implemented)

Assume batch size $B$ prompts and $K$ particles per prompt.

### 3.1 Initialization

At the start:

- Draw initial latents for all particles: shape $(B\cdot K, C, H, W)$.
- Initialize log weights log_w = 0.
- Initialize twist terms, diffusion/proposal log-prob terms, traces.
- Set start = int(num_inference_steps * tempering_start).

Before start, the algorithm behaves like base DDIM (no reward-gradient correction).

### 3.2 Guidance / Reward Evaluation at Timestep t

For each particle (micro-batched by batch_p):

1. Predict noise with UNet (with CFG if guidance_scale > 1).
2. Predict clean sample $\hat x_0$ via ddim_prediction_with_logprob.
3. Decode $\hat x_0$ to image via VAE.
4. Compute reward $r$ using reward_fn.
5. Compute log twist:

	 $$
	 \log \gamma_t(x_t) = \frac{r}{\alpha}.
	 $$

6. Compute approximate reward guidance:

	 $$
	 g_t = \nabla_{x_t}\log \gamma_t(x_t) = \nabla_{x_t}\left(\frac{r}{\alpha}\right).
	 $$

NaN values in log_twist_func or approx_guidance are replaced with finite numbers via nan_to_num.

### 3.3 Tempering Parameter Selection

If i >= start, compute scale factor (interpreted as $\lambda_t$) with one of:

- schedule:
	- numeric power schedule, or
	- exp schedule using (1 + tempering_gamma) growth.
- adaptive:
	- adaptive_tempering(...) chooses factor based on ESS condition.
- FreeDoM:
	- ratio of norms between CFG guidance and reward guidance.
- fallback:
	- 1.0 (no tempering).

Then:

- log_twist_func *= scale_factor
- approx_guidance *= min_scale_next

This is the implementation of annealing from easy target $p_t$ to reward-tilted targets.

### 3.4 Importance Weight Update

The incremental log-weight update is:

$$
\Delta \log w
= \log p_{\text{diff}}(x_{t-1}\mid x_t)
+ \log \gamma_t(x_t)
- \log q_{\text{prop}}(x_{t-1}\mid x_t)
- \log \gamma_{t+1}(x_t),
$$

implemented exactly as:

$$
\mathrm{incremental\_log\_w} = \mathrm{log\_prob\_diffusion} + \mathrm{log\_twist\_func} - \mathrm{log\_prob\_proposal} - \mathrm{log\_twist\_func\_prev}.
$$

Then:

- log_w += incremental_log_w
- log_Z += logsumexp(log_w)

where log_Z is a running estimate of normalizing constants.

### 3.5 ESS and Resampling

- ESS is computed from log weights per prompt.
- resample_fn (strategy from resample_strategy, threshold from ess_threshold) decides whether/when to resample.
- If resampled, particle-indexed tensors are reindexed with resample_indices:
	- latents
	- noise_pred
	- pred_original_sample
	- manifold_deviation_trace
	- log_prob_diffusion_trace

### 3.6 Proposal Sampling and Transition Log-Probabilities

After resampling, proposal propagation is:

1. Compute DDIM step mean and sample:
	 - prev_sample, prev_sample_mean
2. Compute variance $\sigma_t^2$ from scheduler and eta.
3. Apply reward-gradient shift:

	 $$
	 x_{t-1}^{\text{prop}} = \texttt{prev_sample} + \sigma_t^2 g_t.
	 $$

4. Compute log-probability under diffusion kernel and proposal kernel:

	 $$
	 \log p_{\text{diff}} = \log \mathcal N(x_{t-1}^{\text{prop}};\,\mu_\theta,\sigma_t^2 I),
	 $$

	 $$
	 \log q_{\text{prop}} = \log \mathcal N(x_{t-1}^{\text{prop}};\,\mu_\theta + \sigma_t^2 g_t,\sigma_t^2 I).
	 $$

5. Store these for the next timestep weight update.


## 4. Finalization at t = 0

After the denoising loop:

1. Decode final particle latents to images.
2. Compute terminal rewards.
3. Final twist update with latest scale factor.
4. Final weight update using the same incremental formula.
5. Normalize weights.
6. Return the maximum-weight sample as output image/latent.

So the final returned sample is MAP-like over particles under SMC weights, not a random draw from normalized_w.


## 5. What the Returned Values Mean

The function returns:

1. output: chosen best sample (format depends on output_type).
2. log_w: final unnormalized log weights for all particles.
3. normalized_w: normalized particle weights.
4. all_latents: per-step latent states.
5. all_log_w: per-step log weights.
6. all_resample_indices: particle ancestry after each resample.
7. ess_trace: ESS across timesteps.
8. scale_factor_trace: tempering scale ($\lambda$-like) per step.
9. rewards_trace: per-step best reward per prompt.
10. manifold_deviation_trace: diagnostic of proposal shift relative to noise direction.
11. log_prob_diffusion_trace: cumulative negative diffusion log-prob diagnostic.


## 6. Practical Interpretation of Hyperparameters

- num_particles:
	- More particles improve exploration but increase compute/memory linearly.
- batch_p:
	- Micro-batch for reward-gradient computation; lower values reduce memory.
- kl_coeff:
	- Larger value weakens reward influence (acts like higher temperature $\alpha$).
- tempering and tempering_schedule:
	- Control how fast reward influence turns on.
- ess_threshold and resample_strategy:
	- Control particle degeneracy management.
- eta:
	- Controls DDIM stochasticity and variance scale in both diffusion and proposal kernels.


## 7. Important Implementation Notes

1. Reward gradients are taken through decode($\hat x_0$), so reward_fn must be differentiable with respect to image tensors.
2. The script uses gradient checkpointing for UNet and VAE to reduce memory usage.
3. Before tempering starts (i < start), only standard noise prediction is used and reward guidance is skipped.
4. CFG guidance and reward guidance are separate:
	 - CFG influences noise prediction.
	 - Reward gradient shifts the proposal mean and affects SMC weights.
5. The final selected sample is argmax(log_w), not a stochastic sample from normalized_w.


## 8. Compact Pseudocode

For each prompt, initialize K particles from initial noise.

For each diffusion timestep t:

1. Evaluate UNet noise predictions.
2. If t is after tempering_start:
	 - Decode predicted $\hat x_0$.
	 - Compute rewards and gradients $\nabla_{x_t}(r/\alpha)$.
	 - Choose tempering factor $\lambda_t$.
	 - Update twist and incremental importance weights.
	 - Compute ESS and resample if needed.
3. Propagate particles with Gaussian proposal mean shifted by variance times reward gradient.
4. Compute log probabilities under diffusion and proposal kernels for next weight update.

At the final step:

1. Decode final particles and compute final rewards.
2. Apply final weight update and normalize weights.
3. Return highest-weight particle and all diagnostics.

