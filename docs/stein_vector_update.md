## 11. Stein Vector Normalization and Adaptive Step Size

### Motivation

The magnitude of the Stein transport vector $\|\hat\phi_t\|$ varies substantially
across the diffusion trajectory. The score norm scales approximately as

$$
\|\nabla_{x_t}\log p\| \sim \frac{\|\epsilon\|}{\sqrt{1-\bar\alpha_t}}
$$

which introduces a factor of roughly $3\times$ between early and late timesteps
in a typical noise schedule. With the default $M=1$ inner loop, the AdaGrad
accumulator $G$ receives only a single gradient update before reset and provides
no meaningful adaptation. The effective update is therefore:

$$
x_t \leftarrow x_t + \eta_0\,\hat\phi_t(x_t)
$$

with no implicit magnitude control. A fixed `stein_step` $\eta_0$ will
consequently be too conservative early in the steering window and potentially
too large late in the trajectory.

### Combined Normalization and Schedule Correction

The recommended approach combines a noise-schedule correction with per-step
normalization:

$$
\boxed{
x_t \leftarrow x_t
+ \eta_0
\cdot \sqrt{1-\bar\alpha_t}
\cdot \frac{\hat\phi_t(x_t)}{\|\hat\phi_t(x_t)\| + \epsilon}
}
$$

The two factors serve distinct roles:

- $\sqrt{1-\bar\alpha_t}$ anchors the update magnitude to the diffusion manifold
  scale at timestep $t$, counteracting score norm growth as $t\to 0$.
- $\|\hat\phi_t\|^{-1}$ normalizes out reward-gradient magnitude variance,
  making `stein_step` $\eta_0$ interpretable as a literal fraction of the
  latent scale moved per update regardless of timestep or reward strength.

### Softer Variant

If preserving relative magnitude information across particles is desirable,
a partial normalization compresses rather than removes the scale:

$$
x_t \leftarrow x_t
+ \eta_0
\cdot \sqrt{1-\bar\alpha_t}
\cdot \frac{\hat\phi_t(x_t)}{\|\hat\phi_t(x_t)\|^{1/2} + \epsilon}
$$

This retains directional and relative-strength information while preventing
extreme updates when the score or reward gradient is large.

### Behavior at $M > 1$

When $M > 1$, AdaGrad accumulation becomes meaningful and provides implicit
per-coordinate magnitude control. In this regime the schedule correction
$\sqrt{1-\bar\alpha_t}$ remains useful but the full normalization may interact
with AdaGrad redundantly. A reasonable rule of thumb:

- $M = 1$: use full normalization (combined formula above)
- $M \geq 3$: use schedule correction only, let AdaGrad handle magnitude

### Recommended Defaults

- `stein_step` $\eta_0 = 0.01$ to $0.05$ (with normalization active;
  note this range differs from the unnormalized recommendation in Section 9)
- `stein_normalize = "full"` (options: `full`, `soft`, `none`)
- `stein_schedule_correction = true`

### Note on $\epsilon$

The $\epsilon$ in the denominator is the same `stein_adagrad_eps` defined in
Section 6 and serves only as a numerical guard against division by zero.
It does not require separate tuning.