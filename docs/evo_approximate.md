# EVO Approximation: Relaxed Good/Bad Stein Guidance

This document specifies the approximate EVO sampler that uses Good/Bad particle splitting and closed-form good-score estimation for Stein transport.

## 1. Objective

Given a pretrained conditional diffusion score model

$$
s_\theta(x_t, t \mid c) \approx \nabla_{x_t}\log p_t(x_t \mid c),
$$

we steer sampling toward a reward-tilted distribution using particles and Stein updates, without SMC weighting/resampling.

## 2. Setup and Notation

- $K$: number of particles per prompt.
- $N$: number of denoising timesteps.
- $M$: inner Stein loops per steered timestep.
- $(\text{start}, \text{end})$: steering window in inference-step index space, with $\text{start} \le i \le \text{end}$.
- $r(x_0)$: reward/verifier score on decoded predictions.
- `relaxed`: non-negative relaxation margin for Good/Bad split.

At each steered step, particles are split by reward:

$$
\mu_r = \frac{1}{K}\sum_{j=1}^{K} r_j,\qquad
\tau_{\text{relaxed}} = \mu_r - \text{relaxed}.
$$

Good and Bad sets:

$$
G_t = \{j: r_j > \tau_{\text{relaxed}}\},\qquad
B_t = \{j: r_j \le \tau_{\text{relaxed}}\}.
$$

## 3. Approximate Good-Score Probability

For each particle state $x_t$, approximate the Good-region score with anchors from Good terminal samples $g_0^{(i)} \in G_0$:

$$
\nabla_{x_t}\log q_t(x_t \mid c)
\approx
\sum_{i=1}^{N_G} w_i(x_t)\,\nabla_{x_t}\log p(x_t \mid x_0=g_0^{(i)}),
$$

$$
w_i(x_t)=\frac{p(x_t \mid x_0=g_0^{(i)})}{\sum_j p(x_t \mid x_0=g_0^{(j)})}.
$$

Under standard diffusion forward marginals:

$$
p(x_t \mid x_0=g_0^{(i)})=
\mathcal N\!\left(x_t;\sqrt{\bar\alpha_t}\,g_0^{(i)},(1-\bar\alpha_t)I\right),
$$

$$
\nabla_{x_t}\log p(x_t \mid x_0=g_0^{(i)})
=
-\frac{x_t-\sqrt{\bar\alpha_t}\,g_0^{(i)}}{1-\bar\alpha_t}.
$$

This gives a closed-form approximation for the Good score at each steered timestep.

## 4. Stein Transport with AdaGrad

Use an SVGD field on current particles $\{x_t^{(j)}\}_{j=1}^{K}$:

$$
\hat\phi_t(x)=\frac{1}{K}\sum_{j=1}^{K}
\left[
k(x_t^{(j)},x)\,\nabla_{x_t^{(j)}}\log q_t(x_t^{(j)}\mid c)
+\nabla_{x_t^{(j)}}k(x_t^{(j)},x)
\right].
$$

Default kernel (required):

$$
k(x,x')=\exp\!\left(-\frac{\lVert x-x'\rVert_2^2}{\sigma_t}\right)
\quad\text{(RBF)}.
$$

Inner-loop update for `M` iterations:

$$
G \leftarrow G+\hat\phi_t(x_t)^2,\qquad
\eta_{\text{adapt}}=\frac{\eta_0}{\sqrt{G}+\epsilon},
$$

$$
x_t \leftarrow x_t+\eta_{\text{adapt}}\odot\hat\phi_t(x_t).
$$

Where $\eta_0=$ `stein_step` and $\epsilon=$ `stein_adagrad_eps`.

## 5. Compute $x_{t-1}$ from Steered $x_t$

After $M$ Stein updates, let the refined latent be

$$
\tilde{x}_t \equiv x_t^{(M)}.
$$

Then compute $x_{t-1}$ directly from $\tilde{x}_t$:

$$
x_{t-1}
=
\sqrt{\bar\alpha_{t-1}}\,\tilde{x}_t
+\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\,\epsilon_\theta(\tilde{x}_t,t)
+\sigma_t z,\quad z\sim\mathcal N(0,I).
$$

$x_{t-1}$ is computed directly from the Stein-refined latent $\tilde{x}_t$.

## 6. Full Algorithm

1. Initialize $K$ particles and run diffusion for $N$ timesteps.
2. For each timestep $i$ (with $t=\text{timesteps}[i]$):
   1. If $i$ is outside the steering window $[\text{start}, \text{end}]$, run the base scheduler transition only.
   2. If $i$ is inside the steering window:
      1. Evaluate rewards, compute $\mu_r$, and split particles into Good/Bad using $\tau_{\text{relaxed}}=\mu_r-\text{relaxed}$.
      2. Build the approximate Good score $\nabla\log q_t$ from Good anchors using the equations above.
      3. Run $M$ Stein updates on $x_t$ with AdaGrad-preconditioned step size using only the Stein vector field.
   3. Compute $x_{t-1}$ from the refined latent $\tilde{x}_t$ via the base scheduler transition.
3. Continue until $t=0$.

## 7. Recommended Defaults

- `num_particles (K) = 4`
- `stein_loop (M) = 1`
- `stein_step = 0.02` to `0.05`
- `stein_adagrad_eps = 1e-8`
- `stein_kernel = "rbf"` (default and currently required)
- `relaxed = 0.0` (set `> 0` to widen Good set)
