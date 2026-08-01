# Stein-Transport Sampling - Approximation-based

This document specifies the approximate EVO sampler that steers diffusion particles with a soft reward-tilted good probability. It removes the hard Good/Bad particle split and replaces it with a Monte Carlo estimate of the good-conditioned diffusion density.

## 1. Objective

Given a pretrained conditional diffusion score model

$$
s_\theta(x_t,t\mid c)\approx\nabla_{x_t}\log p_t(x_t\mid c),
$$

we steer sampling toward samples that satisfy a soft good event $y=1$ under prompt/context $c$. The event likelihood is defined by a reward or verifier score $h(x_0)$ and temperature $\alpha$:

$$
p(y=1\mid x_0=z,c)\propto \exp\left(\frac{h(z)}{\alpha}\right).
$$

The sampler uses particles and Stein updates, without SMC weighting, resampling, or hard thresholding.

## 2. Setup and Notation

- $K$: number of diffusion particles per prompt.
- $N$: number of denoising timesteps.
- $(\text{start},\text{end})$: steering window in inference-step index space, with $\text{start}\le i\le\text{end}$.
- $h(x_0)$: reward/verifier score on decoded clean samples.
- $\alpha$: reward temperature / KL trade-off coefficient.
- $z$: clean data-space or latent clean sample used as an $x_0$ anchor.
- $Z$: normalizing constant for the reward-tilted clean distribution. It is not needed for Stein guidance because it is constant with respect to $x_t$.

There is no Good/Bad set construction. In particular, we do not compute a reward mean, relaxation margin, or hard threshold. All sampled $x_0$ anchors can contribute, with contribution strength controlled continuously by $\exp(h(z)/\alpha)$.

## 3. Soft Good-Conditioned Density

For the good event $y=1$, the target intermediate density at timestep $t$ is

$$
p(x_t=x\mid y=1,c) =
\int p(x_t=x,x_0=z\mid y=1,c)\,dz.
$$

Using the forward diffusion marginal conditioned on $x_0$:

$$
p(x_t=x\mid y=1,c) =
\int p(x_t=x\mid x_0=z)\,p(x_0=z\mid y=1,c)\,dz.
$$

The clean good-conditioned distribution is reward tilted:

$$
p(x_0=z\mid y=1,c) =
\frac{1}{Z}p_{\text{data}}(z\mid c)\exp\left(\frac{h(z)}{\alpha}\right).
$$

Therefore,

$$
p(x_t=x\mid y=1,c) =
\frac{1}{Z}
\mathbb E_{z\sim p_{\text{data}}(\cdot\mid c)}
\left[
p(x_t=x\mid x_0=z)
\exp\left(\frac{h(z)}{\alpha}\right)
\right].
$$

Taking the score removes the unknown constant $Z$:

$$
\nabla_{x_t}\log p(x_t=x\mid y=1,c) =
\nabla_{x_t}\log
\mathbb E_{z\sim p_{\text{data}}(\cdot\mid c)}
\left[
p(x_t=x\mid x_0=z)
\exp\left(\frac{h(z)}{\alpha}\right)
\right].
$$

## 4. Monte Carlo Approximation

Approximate the expectation with $N_a$ clean anchors $z^{(i)}$:

$$
\nabla_{x_t}\log p(x_t=x\mid y=1,c)
\approx
\nabla_{x_t}\log
\left[
\frac{1}{N_a}\sum_{i=1}^{N_a}
p(x_t=x\mid x_0=z^{(i)})
\exp\left(\frac{h(z^{(i)})}{\alpha}\right)
\right].
$$

Under the standard diffusion forward marginal,

$$
p(x_t=x\mid x_0=z^{(i)}) =
\mathcal N\left(
x;\sqrt{\bar\alpha_t}\,z^{(i)},(1-\bar\alpha_t)I
\right),
$$

and

$$
\nabla_x\log p(x_t=x\mid x_0=z^{(i)}) =
-\frac{x-\sqrt{\bar\alpha_t}\,z^{(i)}}{1-\bar\alpha_t}.
$$

This gives the normalized mixture-score form

$$
\nabla_{x_t}\log p(x_t=x\mid y=1,c)
\approx
\sum_{i=1}^{N_a}
\omega_i(x,t)
\left(
-\frac{x-\sqrt{\bar\alpha_t}\,z^{(i)}}{1-\bar\alpha_t}
\right),
$$

where

$$
\omega_i(x,t) =
\frac{
p(x_t=x\mid x_0=z^{(i)})
\exp\left(h(z^{(i)})/\alpha\right)
}{
\sum_{j=1}^{N_a}
p(x_t=x\mid x_0=z^{(j)})
\exp\left(h(z^{(j)})/\alpha\right)
}.
$$

For numerical stability, compute the mixture weights in log space:

$$
\log a_i =
\log p(x_t=x\mid x_0=z^{(i)})
+\frac{h(z^{(i)})}{\alpha},
\qquad
\omega_i=\mathrm{softmax}_i(\log a_i).
$$

In high-dimensional latent space, the Gaussian log forward term is a sum over all latent dimensions. Even modest per-dimension differences can accumulate into logit gaps of hundreds or thousands, causing the softmax to become effectively one-hot. To avoid one anchor dominating purely because of logit scale, the implementation applies a bounded adaptive temperature before the softmax.

For a current particle $x_t^{(j)}$, define the raw log weights over anchors:

$$
\ell_{j,i} =
\log p(x_t^{(j)}\mid x_0=z^{(i)})
+\frac{h(z^{(i)})}{\alpha}.
$$

Then compute a row-wise temperature from the empirical spread of the raw logits:

$$
\tau_j =
\max\left(\mathrm{Std}_{i}(\ell_{j,i}),\tau_{\min}\right),
\qquad
\tau_{\min}=1.
$$

The normalized mixture weights are then

$$
\omega_{j,i} =
\mathrm{softmax}_i\left(\frac{\ell_{j,i}}{\tau_j}\right).
$$

In code this adaptive behavior corresponds to `soft_temperature=None`. If `soft_temperature` is set, it disables the adaptive rule and uses the provided fixed temperature:

$$
\tau_j =
\texttt{soft\_temperature}.
$$

This temperature is a numerical stabilization heuristic for the Monte Carlo mixture; it is not part of the exact probabilistic density.

## 5. Getting $x_0$ Anchors While Steering at $x_t$

At a steered latent $x_t$, the clean anchors $z^{(i)}$ are approximations of possible $x_0$ outcomes conditioned on the current state and prompt. To obtain them cheaply, use a distilled or fast-sampling model rather than a full denoising rollout.

Recommended anchor sources:

- DPM-style fast solver from $x_t$ to an $x_0$ prediction.
- LCM / consistency model prediction from $x_t$.
- DMD or another distilled one/few-step sampler.
- The base model's $\hat x_{0|t}$ prediction for the default clean estimate.

For each current particle $x_t^{(j)}$, produce one or more anchors

$$
z^{(j,\ell)}\approx x_0
\quad\text{from}\quad
x_t^{(j)},t,c,
\qquad \ell=1,\dots,L.
$$

Then evaluate $h(z^{(j,\ell)})$ after decoding the anchor to image space if the reward model expects images. These anchors replace the old Good set anchors; they are not selected by threshold.

## 6. Single-Step Stein Transport

Use the soft good-conditioned score from Section 4 as the target score inside an SVGD field. For particles $\{x_t^{(j)}\}_{j=1}^{K}$ in the same prompt group:

$$
\hat\phi_t(x) =
\frac{1}{K}\sum_{j=1}^{K}
\left[
k(x_t^{(j)},x)\,
\nabla_{x_t^{(j)}}\log p(x_t^{(j)}\mid y=1,c)
+
\nabla_{x_t^{(j)}}k(x_t^{(j)},x)
\right].
$$

Default kernel:

$$
k(x,x')=\exp\left(-\frac{\lVert x-x'\rVert_2^2}{h\sigma_t}\right),
\quad
h = \frac{\mathrm{median}_{i\neq j}\|u_i-u_j\|^2}{\log(K+1)+\delta}.
\quad\text{(RBF)}.
$$

Evaluate the field once and apply one explicit update at each selected
diffusion step:

$$
\tilde x_t=x_t+\texttt{stein\_step}\,\hat\phi_t(x_t).
$$

## 7. Off-Manifold Correction for $x_{t-1}$

A Stein update directly modifies $x_t$ and can move the latent away from the diffusion manifold expected by the base scheduler. To correct this before computing the next latent, decode or predict a clean sample from the steered latent and use that clean prediction in the DDIM-style transition.

Let the Stein-refined latent be

$$
\tilde x_t = x_t + \texttt{stein\_step}\,\hat\phi_t(x_t).
$$

Predict a clean sample from the refined latent:

$$
\bar x_{0|t}=\mathrm{DecodeToClean}(\tilde x_t,t,c),
$$

where `DecodeToClean` is the scheduler's $x_0$ prediction.

Then compute the next latent with the clean corrected term:

$$
x_{t-1} =
\sqrt{\bar\alpha_{t-1}}\,\bar x_{0|t}
+
\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\,
\epsilon_\theta(x_t,t,c)
+
\sigma_t z,
\qquad z\sim\mathcal N(0,I).
$$

The correction uses $\bar x_{0|t}$ from the Stein-refined latent, while the noise direction can use the base model prediction at the pre-correction or current timestep latent, depending on scheduler implementation. The key constraint is that $x_{t-1}$ is reconstructed through a clean $x_0$ estimate instead of treating the off-manifold $\tilde x_t$ as the clean component.

## 8. Full Algorithm

1. Initialize $K$ particles and run diffusion for $N$ timesteps.
2. For each timestep $i$ with diffusion time $t=\text{timesteps}[i]$:
   1. If $i$ is outside the steering window $[\text{start},\text{end}]$, run the base scheduler transition.
   2. If $i$ is inside the steering window:
      1. For each particle, obtain one or more approximate $x_0$ anchors using a distilled or fast-sampling model such as DPM, LCM, or DMD.
      2. Evaluate rewards $h(z^{(i)})$ for all anchors.
      3. Estimate the soft good-conditioned score with the Monte Carlo mixture formula.
      4. Evaluate and apply one Stein update to $x_t$.
      5. Apply the off-manifold correction by predicting $\bar x_{0|t}$ from the Stein-refined latent.
   3. Compute $x_{t-1}$ from $\bar x_{0|t}$ and the scheduler noise term.
3. Continue until $t=0$.
4. Decode final particles and choose the output policy, e.g. best reward particle per prompt or return all particles for analysis.

## 9. Recommended Defaults

- `num_particles (K) = 4`
- `predicted_samples (L) = 1` to `4` per particle
- `stein_step = 0.002` to `0.005`
- `stein_kernel = "rbf"` or `"imq"` (`"rbf"` default)
- `alpha = kl_coeff`
- `prediction_model = "dpm"` for the fast solver, or `prediction_model = "default"` for the base $\hat x_{0|t}$ estimate
