from collections import namedtuple
import math
import os
from torch import nn
import torch
from gflower.config.flow_matching import FlowMatchingEvaluationConfig
from gflower import utils
from gflower.models_flow.flow_matcher import apply_conditioning, apply_conditioning_from_conditioned_x
from gflower.models_flow.optimal_transport import OTPlanSampler
from gflower.sampling.guides import ValueGuide
from gflower.utils.arrays import to_device, to_torch

Trajectories = namedtuple('Trajectories', 'actions observations values')

def to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x

class ConditionedODESolver(nn.Module):
    def __init__(self, model, conditions, action_dim, guide_fn=None, ode_method='euler'):
        super().__init__()
        self.model = model
        self.conditions = conditions
        self.guide_fn = guide_fn
        self.action_dim = action_dim
        assert ode_method in ['euler'], "Only Euler is supported for now"

    def forward(self, x, t_span, *args, **kwargs):
        """
        Args:
            x (B, C, T), t (T)
        """
        assert len(t_span) > 1, "t_span must have at least 2 elements"
        x0 = x.clone()
        dt = t_span[1] - t_span[0]
        for t in t_span:
            if self.guide_fn is None:
                # model forward pass
                dx_dt = self.model(x, t)
            # add gradient guidance
            else:
                x = x.requires_grad_()
                dx_dt = self.model(x, t)
                dx_dt = dx_dt + self.guide_fn(x, t, dx_dt, self.model)
            # fill in the condition
            dx_dt = apply_conditioning_from_conditioned_x(
                dx_dt, torch.zeros_like(x), self.conditions, self.action_dim
            )
            x = x + dx_dt * dt
            x = x.detach()
        return x


class FlowPolicy(nn.Module):
    """
    This class is a wrapper around a flow model that generates actions from ONE step of 
    the observed state. 

    The generation is guided with the value model using different guidance methods.

    Normalization:
        Input observation and output action are denormalized; Models' input and output 
        are normalized.
    """
    def __init__(
        self, 
        flow_model, value_model, normalizer, action_dim, state_dim, horizon, 
        cfg: FlowMatchingEvaluationConfig,
        guide_model=None,
    ):
        super().__init__()
        self.flow_model = flow_model
        self.normalizer = normalizer
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.horizon = horizon

        self.cfg = cfg
        self.value_model = value_model # we need this to return value
        self.guide_model = guide_model

        self.flow_model.eval()
        if self.value_model is not None:
            self.value_model.eval()
        if self.guide_model is not None:
            self.guide_model.eval()

    def __call__(self, conditions, batch_size=1, verbose=True):
        # assert batch_size == 1, "batch_size must be 1 for now"

        if self.cfg.guidance_method == 'smc':
            return self.smc_forward(conditions, batch_size)
        elif self.cfg.guidance_method in ['gradient']:
            return self.gradient_forward(conditions, batch_size)
        elif self.cfg.guidance_method in ['stein']:
            return self.stein_forward(conditions, batch_size)
        elif self.cfg.guidance_method in ['mc']:
            return self.mc_forward(conditions, batch_size)
        elif self.cfg.guidance_method in ['no']:
            pass
        elif self.cfg.guidance_method in ['guidance_matching']:
            return self.learned_guidance_forward(conditions, batch_size)
        elif self.cfg.guidance_method in ['sim_mc']:
            return self.sim_mc_guidance_forward(conditions, batch_size)
        else:
            raise ValueError(f"Unsupported guidance method: {self.cfg.guidance_method}")

        # Only normalize the observation
        conditions = utils.apply_dict(self.normalizer.normalize, conditions, 'observations')

        # Generate actions
        solver = ConditionedODESolver(
            self.flow_model, 
            conditions, 
            guide_fn=None, 
            ode_method=self.cfg.ode_solver,
            action_dim=self.action_dim,
        )        
    
        x = torch.randn(batch_size, self.horizon, self.action_dim + self.state_dim, device=self.cfg.device) # (B, T, C)
        x = apply_conditioning(x, to_torch(conditions, device=x.device), self.action_dim)

        x = solver(
            x, 
            t_span=torch.linspace(
                *self.cfg.ode_t_span, self.cfg.ode_t_steps, device=x.device
            ),
        ) # (B, T, C)

        normed_actions = x[:, :, :self.action_dim]
        actions = self.normalizer.unnormalize(normed_actions, 'actions')

        normed_observations = x[:, :, self.action_dim:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')
        
        if self.cfg.guidance_method != 'no':
            values = self.value_model(normed_observations, normed_actions)
        else:
            values = None

        trajectories = Trajectories(actions, observations, values)
        
        # TODO: Add more "guidance" methods, including sample and selection-based MPC
        actions = actions[0, 0] # simply get the first action in the first sample in the batch
        
        return actions, trajectories


    ### Sequential Monte Carlo ###

    def _systematic_resample(self, weights):
        """Return systematic-resampling ancestors for weights shaped (B, N)."""
        batch_size, num_particles = weights.shape
        offsets = torch.rand(
            batch_size, 1, device=weights.device, dtype=weights.dtype
        ) / num_particles
        positions = offsets + torch.arange(
            num_particles, device=weights.device, dtype=weights.dtype
        ).unsqueeze(0) / num_particles
        cdf = weights.cumsum(dim=1)
        cdf[:, -1] = 1.0
        return torch.searchsorted(
            cdf.contiguous(), positions.contiguous(), right=False
        )

    def _smc_endpoint_value(self, x, t, conditions, is_final):
        """Score a first-order prediction of the completed trajectory."""
        if not is_final:
            dx_dt = self.flow_model(x, t)
            x_eval = x + (1 - t) * dx_dt
            x_eval = apply_conditioning(
                x_eval,
                conditions,
                self.action_dim,
            )
        else:
            x_eval = x
        return self.value_model(x_eval)[:, -1, 0]

    def smc_forward(self, conditions, batch_size=1):
        """Annealed SMC using predicted terminal value as the target potential."""
        assert self.cfg.guidance_method == 'smc'
        assert self.value_model is not None, "value_model is required for SMC"
        if self.cfg.smc_particles < 1:
            raise ValueError("smc_particles must be >= 1")
        if self.cfg.smc_scale < 0:
            raise ValueError("smc_scale must be >= 0")
        if not 0 < self.cfg.smc_ess_threshold <= 1:
            raise ValueError("smc_ess_threshold must be in (0, 1]")
        if self.cfg.smc_resample_every < 1:
            raise ValueError("smc_resample_every must be >= 1")

        num_particles = self.cfg.smc_particles
        total_batch = batch_size * num_particles
        conditions = utils.apply_dict(
            self.normalizer.normalize, conditions, 'observations'
        )
        conditions = to_torch(conditions, device=self.cfg.device)
        conditions = self._repeat_conditions_for_particles(
            conditions, batch_size, num_particles
        )

        x = torch.randn(
            total_batch,
            self.horizon,
            self.action_dim + self.state_dim,
            device=self.cfg.device,
        )
        x = apply_conditioning(x, conditions, self.action_dim)
        t_span = torch.linspace(
            *self.cfg.ode_t_span, self.cfg.ode_t_steps, device=x.device
        )
        if len(t_span) < 2:
            raise ValueError("ode_t_steps must be >= 2 for SMC")
        if t_span[-1] <= t_span[0]:
            raise ValueError("ode_t_span must be increasing for SMC")

        log_weights = torch.zeros(
            batch_size, num_particles, device=x.device, dtype=x.dtype
        )
        previous_potential = torch.zeros_like(log_weights)
        batch_idx = torch.arange(batch_size, device=x.device).unsqueeze(1)

        with torch.no_grad():
            for step_idx in range(len(t_span) - 1):
                t = t_span[step_idx]
                t_next = t_span[step_idx + 1]
                dt = t_next - t

                dx_dt = self.flow_model(x, t)
                dx_dt = apply_conditioning_from_conditioned_x(
                    dx_dt, torch.zeros_like(x), conditions, self.action_dim
                )
                x = apply_conditioning(
                    x + dx_dt * dt, conditions, self.action_dim
                )

                is_final_step = step_idx == len(t_span) - 2
                values = self._smc_endpoint_value(
                    x, t_next, conditions, is_final_step
                ).reshape(batch_size, num_particles)
                values = torch.nan_to_num(
                    values, nan=0.0, posinf=1e6, neginf=-1e6
                )
                annealing = (t_next - t_span[0]) / (t_span[-1] - t_span[0])
                potential = self.cfg.smc_scale * annealing * values
                log_weights = log_weights + potential - previous_potential
                previous_potential = potential

                is_resampling_step = (
                    (step_idx + 1) % self.cfg.smc_resample_every == 0
                )
                if is_final_step or not is_resampling_step:
                    continue

                weights = torch.softmax(log_weights, dim=1)
                ess = 1.0 / weights.square().sum(dim=1)
                resample_groups = ess <= (
                    self.cfg.smc_ess_threshold * num_particles
                )
                if not resample_groups.any():
                    continue

                ancestors = self._systematic_resample(weights)
                identity = torch.arange(num_particles, device=x.device)
                identity = identity.unsqueeze(0).expand(batch_size, -1)
                ancestors = torch.where(
                    resample_groups.unsqueeze(1), ancestors, identity
                )

                x_grouped = x.reshape(
                    batch_size,
                    num_particles,
                    self.horizon,
                    self.action_dim + self.state_dim,
                )
                x = x_grouped[batch_idx, ancestors].reshape_as(x)
                previous_potential = previous_potential[batch_idx, ancestors]
                log_weights = torch.where(
                    resample_groups.unsqueeze(1),
                    torch.zeros_like(log_weights),
                    log_weights,
                )

        normed_actions = x[:, :, :self.action_dim]
        normed_observations = x[:, :, self.action_dim:]
        actions = self._as_device_tensor(
            self.normalizer.unnormalize(normed_actions, 'actions')
        ).reshape(batch_size, num_particles, self.horizon, self.action_dim)
        observations = self._as_device_tensor(
            self.normalizer.unnormalize(normed_observations, 'observations')
        ).reshape(batch_size, num_particles, self.horizon, self.state_dim)

        with torch.no_grad():
            final_values = self.value_model(x)[:, -1, 0].reshape(
                batch_size, num_particles
            )
            final_values = torch.nan_to_num(
                final_values, nan=-1e6, posinf=1e6, neginf=-1e6
            )
        best_idx = final_values.argmax(dim=1)
        flat_batch_idx = torch.arange(batch_size, device=x.device)
        best_values = final_values[flat_batch_idx, best_idx]
        best_actions = actions[flat_batch_idx, best_idx]
        best_observations = observations[flat_batch_idx, best_idx]

        trajectories = Trajectories(
            to_np(best_actions), to_np(best_observations), to_np(best_values)
        )
        return to_np(best_actions[0, 0]), trajectories


    ### Taylor Expansion Approximate Gradient Guidance ###

    def _canonical_grad_location(self, location):
        aliases = {
            'x1': 'x_1',
            'x_1': 'x_1',
            'xt': 'x_t',
            'x_t': 'x_t',
        }
        if location not in aliases:
            raise ValueError(f"Unsupported gradient location: {location}")
        return aliases[location]

    def _compute_value_gradient(self, x, t, dx_dt, value_model, schedule_fn, scale, grad_at, grad_to):
        grad_at = self._canonical_grad_location(grad_at)
        grad_to = self._canonical_grad_location(grad_to)

        if grad_at == 'x_t':
            x1_pred = None
            value = value_model(x)[:, -1, 0] # (B, T, 1) -> (B,)
        elif grad_at == 'x_1':
            x1_pred = x + (1 - t) * dx_dt
            value = value_model(x1_pred)[:, -1, 0] # (B, T, 1) -> (B,)
        else:
            raise ValueError(f"Unsupported gradient compute at: {grad_at}")

        if grad_to == 'x_t':
            grad = torch.autograd.grad([value.sum()], [x])[0]
        elif grad_to == 'x_1':
            if grad_at != 'x_1':
                raise ValueError("cannot compute gradient wrt x_1 when grad_at is x_t")
            grad = torch.autograd.grad([value.sum()], [x1_pred])[0]
        else:
            raise ValueError(f"Unsupported gradient compute wrt: {grad_to}")

        return grad * scale * schedule_fn(t)

    def get_gradient_guidance_model(self, value_model, schedule_fn, scale, grad_at='x_1', grad_to='x_1'):
        """
        Return the guidance model for the flow model.
        """
        assert self.cfg.guidance_method in ['gradient'], f"Unsupported guidance method: {self.cfg.guidance_method}"

        def guide_fn(x, t, dx_dt, flow_model):
            return self._compute_value_gradient(
                x=x,
                t=t,
                dx_dt=dx_dt,
                value_model=value_model,
                schedule_fn=schedule_fn,
                scale=scale,
                grad_at=grad_at,
                grad_to=grad_to,
            )
        return guide_fn

    def get_scheduler(self, schedule_fn):
        """
        Return the scheduler for the gradient guidance.
        """
        if schedule_fn == 'const':
            return lambda x: x
        elif schedule_fn == 'linear_decay':
            return lambda x: 1 - x
        elif schedule_fn == 'cosine_decay':
            return lambda x: 0.5 * (1 + torch.cos(x * math.pi))
        elif schedule_fn == 'exp_decay':
            return lambda x: (torch.exp(-x) - math.exp(-1)) / (1 - math.exp(-1))
        else:
            raise ValueError(f"Unsupported gradient schedule: {schedule_fn}")

    def gradient_forward(self, conditions, batch_size=1):
        """
        Use gradient guidance to generate actions.
        """
        # assert batch_size == 1, "batch_size must be 1 for now"
        assert self.cfg.guidance_method == 'gradient', f"guidance_method must be gradient, but got {self.cfg.guidance_method}"

        # Only normalize the observation
        conditions = utils.apply_dict(self.normalizer.normalize, conditions, 'observations')

        # Generate actions
        solver = ConditionedODESolver(
            self.flow_model, 
            conditions, 
            guide_fn=self.get_gradient_guidance_model(
                self.value_model, 
                schedule_fn=self.get_scheduler(self.cfg.grad_schedule), 
                scale=self.cfg.grad_scale, 
                grad_at=self.cfg.grad_compute_at, 
                grad_to=self.cfg.grad_wrt
            ), 
            ode_method=self.cfg.ode_solver,
            action_dim=self.action_dim,
        )        
    
        x = torch.randn(batch_size, self.horizon, self.action_dim + self.state_dim, device=self.cfg.device) # (B, T, C)
        x = apply_conditioning(x, to_torch(conditions, device=x.device), self.action_dim)

        x = solver(x, t_span=torch.linspace(
            *self.cfg.ode_t_span, self.cfg.ode_t_steps, device=x.device
        )) # (B, T, C)

        normed_actions = x[:, :, :self.action_dim]
        actions = self.normalizer.unnormalize(normed_actions, 'actions')

        normed_observations = x[:, :, self.action_dim:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')
        
        values = self.value_model(torch.cat([normed_actions, normed_observations], dim=-1))
        final_values = values[:, -1, 0]
        best_idx = final_values.argmax()

        trajectories = Trajectories(
            to_np(actions[best_idx]),
            to_np(observations[best_idx]),
            to_np(final_values[best_idx]),
        )
        actions = to_np(actions[best_idx, 0])
        
        return actions, trajectories

    ### Stein Particle Steering ###

    def _repeat_conditions_for_particles(self, conditions, batch_size, num_particles):
        target_batch = batch_size * num_particles
        repeated_conditions = {}
        for t, val in conditions.items():
            if val.dim() == 1:
                repeated_conditions[t] = val.unsqueeze(0).repeat(target_batch, 1)
            elif val.shape[0] == target_batch:
                repeated_conditions[t] = val
            elif val.shape[0] == batch_size:
                repeated_conditions[t] = val.repeat_interleave(num_particles, dim=0)
            elif val.shape[0] == 1:
                repeats = [target_batch] + [1] * (val.dim() - 1)
                repeated_conditions[t] = val.repeat(*repeats)
            else:
                raise ValueError(
                    f"Condition batch {val.shape[0]} does not match batch_size={batch_size} "
                    f"or batch_size * stein_particles={target_batch}."
                )
        return repeated_conditions

    def _rbf_stein_vector_field(
        self,
        particles,
        score,
        batch_size,
        num_particles,
        repulsion_strength=1.0,
        eps=1e-8,
    ):
        """
        Compute an RBF SVGD field independently for each environment batch.

        Args:
            particles: Tensor, shape (batch_size * num_particles, T, C)
            score: Tensor, shape (batch_size * num_particles, T, C)
        """
        if num_particles == 1:
            return score

        if particles.shape[0] != batch_size * num_particles:
            raise ValueError("Particle batch does not match batch_size * num_particles.")

        horizon = particles.shape[1]
        dim = particles.shape[2]
        particles_grouped = particles.view(batch_size, num_particles, horizon, dim)
        score_grouped = score.view(batch_size, num_particles, horizon, dim)
        out_grouped = torch.zeros_like(score_grouped)

        for group_idx in range(batch_size):
            x = particles_grouped[group_idx].reshape(num_particles, -1)
            s = score_grouped[group_idx].reshape(num_particles, -1)

            dist2 = torch.cdist(x, x) ** 2
            positive_dist2 = dist2[dist2 > 0]
            if positive_dist2.numel() == 0:
                h_bandwidth = torch.tensor(1.0, device=particles.device, dtype=particles.dtype)
            else:
                h_bandwidth = positive_dist2.median() / (math.log(num_particles + 1.0) + eps)
                h_bandwidth = torch.clamp(h_bandwidth, min=eps)

            kernel = torch.exp(-dist2 / h_bandwidth)
            attraction = (kernel.t() @ s) / float(num_particles)

            weighted_sum = kernel.t() @ x
            kernel_sum = kernel.sum(dim=0, keepdim=True).t()
            repulsion = (2.0 / h_bandwidth) * (weighted_sum - x * kernel_sum) / float(num_particles)

            out_grouped[group_idx] = (
                attraction + repulsion_strength * repulsion
            ).view(num_particles, horizon, dim)

        return out_grouped.view(batch_size * num_particles, horizon, dim)

    def _as_device_tensor(self, x):
        if isinstance(x, torch.Tensor):
            return x.to(self.cfg.device)
        return torch.tensor(x, device=self.cfg.device)

    def stein_forward(self, conditions, batch_size=1):
        """
        Use grouped RBF Stein steering over trajectory particles.
        """
        assert self.cfg.guidance_method == 'stein', f"guidance_method must be stein, but got {self.cfg.guidance_method}"
        assert self.value_model is not None, "value_model is required for Stein steering"
        if self.cfg.stein_particles < 1:
            raise ValueError("stein_particles must be >= 1")
        if self.cfg.stein_loop < 0:
            raise ValueError("stein_loop must be >= 0")
        if self.cfg.stein_step < 0:
            raise ValueError("stein_step must be >= 0")
        if self.cfg.stein_repulsion < 0:
            raise ValueError("stein_repulsion must be >= 0")
        if self.cfg.stein_kernel != 'rbf':
            raise ValueError(f"Unsupported stein_kernel: {self.cfg.stein_kernel}. Only 'rbf' is supported.")

        num_particles = self.cfg.stein_particles
        total_batch = batch_size * num_particles
        schedule_fn = self.get_scheduler(self.cfg.grad_schedule)

        conditions = utils.apply_dict(self.normalizer.normalize, conditions, 'observations')
        conditions = to_torch(conditions, device=self.cfg.device)
        conditions = self._repeat_conditions_for_particles(conditions, batch_size, num_particles)

        x = torch.randn(total_batch, self.horizon, self.action_dim + self.state_dim, device=self.cfg.device)
        x = apply_conditioning(x, conditions, self.action_dim)

        t_span = torch.linspace(*self.cfg.ode_t_span, self.cfg.ode_t_steps, device=x.device)
        assert len(t_span) > 1, "t_span must have at least 2 elements"
        dt = t_span[1] - t_span[0]

        for t in t_span:
            grad_accumulator = torch.zeros_like(x, dtype=torch.float32)

            for _ in range(self.cfg.stein_loop):
                x = x.detach().requires_grad_()
                dx_dt = self.flow_model(x, t)
                score = self._compute_value_gradient(
                    x=x,
                    t=t,
                    dx_dt=dx_dt,
                    value_model=self.value_model,
                    schedule_fn=schedule_fn,
                    scale=self.cfg.grad_scale,
                    grad_at=self.cfg.grad_compute_at,
                    grad_to=self.cfg.grad_wrt,
                )
                stein_direction = self._rbf_stein_vector_field(
                    particles=x.detach().float(),
                    score=score.detach().float(),
                    batch_size=batch_size,
                    num_particles=num_particles,
                    repulsion_strength=self.cfg.stein_repulsion,
                )
                stein_direction = torch.nan_to_num(stein_direction)

                grad_accumulator = grad_accumulator + stein_direction * stein_direction
                adaptive_step = self.cfg.stein_step / (
                    torch.sqrt(grad_accumulator) + self.cfg.stein_adagrad_eps
                )
                if self.cfg.stein_adagrad_clip is not None:
                    adaptive_step = adaptive_step.clamp(
                        min=self.cfg.stein_adagrad_clip[0],
                        max=self.cfg.stein_adagrad_clip[1],
                    )

                x = x.detach() + (adaptive_step * stein_direction).to(x.dtype)
                x = apply_conditioning(x, conditions, self.action_dim)

            x = x.detach()
            with torch.no_grad():
                dx_dt = self.flow_model(x, t)
                dx_dt = apply_conditioning_from_conditioned_x(
                    dx_dt, torch.zeros_like(x), conditions, self.action_dim
                )
                x = x + dx_dt * dt
                x = apply_conditioning(x, conditions, self.action_dim)

        normed_actions = x[:, :, :self.action_dim]
        normed_observations = x[:, :, self.action_dim:]

        actions = self._as_device_tensor(self.normalizer.unnormalize(normed_actions, 'actions'))
        observations = self._as_device_tensor(self.normalizer.unnormalize(normed_observations, 'observations'))

        values = self.value_model(torch.cat([normed_actions, normed_observations], dim=-1))[:, -1, 0]
        values = values.reshape(batch_size, num_particles)
        best_idx = values.argmax(dim=1)
        batch_idx = torch.arange(batch_size, device=x.device)

        best_values = values[batch_idx, best_idx]
        best_observations = observations.reshape(batch_size, num_particles, self.horizon, self.state_dim)[
            batch_idx, best_idx
        ]
        best_actions = actions.reshape(batch_size, num_particles, self.horizon, self.action_dim)[
            batch_idx, best_idx
        ]

        trajectories = Trajectories(to_np(best_actions), to_np(best_observations), to_np(best_values))
        actions = to_np(best_actions[0, 0])

        return actions, trajectories
    

    ### Monte-Carlo Approximate Guidance ###

    def _get_cached_ot_cfm_plan(self):
        if self.cached_ot_cfm_plan is None:
            raise ValueError("No cached OT-CFM plan found")
        return self.cached_ot_cfm_plan

    def _save_cached_ot_cfm_plan(self, x0_, x1_):
        self.cached_ot_cfm_plan = (x0_, x1_)

    def _remove_cached_ot_cfm_plan(self):
        self.cached_ot_cfm_plan = None

    def gaussian_prob(self, x, mean=0, std=1):
        """
        x: (B, T, C)
        """
        return torch.exp(-(x - mean).square().sum((1, 2)) / 2 / std.pow(x.shape[1] * x.shape[2])) / (2 * math.pi * std.pow(2)).pow(x.shape[1] * x.shape[2] / 2)

    def get_mc_guide_fn(self, x1, cached_v=None):
        """
        Compute the gradient guidance for the flow model.
        I think we only need to implement CFM and OT-CFM with Gaussian paths

        Args:
            x1_: Tensor, shape (B, T, C)
        """

        def cfm_log_p_t1(x1, xt, t, epsilon):
            # xt = t x1 + (1 - t) x0 -> x0 = xt / (1 - t) - t / (1 - t) x1
            x1 = x1.flatten(1) # (B, T * C)
            xt = xt.flatten(1) # (B, T * C)
            mu_t = t * x1 # (B, T * C)
            sigma_t = (1 - t + epsilon)
            log_p1t = torch.distributions.MultivariateNormal(
                mu_t, torch.eye(mu_t.shape[1], device=mu_t.device) * sigma_t
            ).log_prob(xt) # (B, T * C)
            return log_p1t
        
        def ot_cfm_log_p_tz(x0, x1, xt, t, std):
            """ 
            Args:
                std: float, g.t.: 0. Too small: requires large mc_batch_size; Too large: inaccurate
            """
            # xt = t x1 + (1 - t) x0 -> x0 = xt / (1 - t) - t / (1 - t) x1
            x0 = x0.flatten(1) # (B, T * C)
            x1 = x1.flatten(1) # (B, T * C)
            xt = xt.flatten(1) # (B, T * C)
            mean = t * x1 + (1 - t) * x0 # (B, T * C)
            log_p1t = torch.distributions.MultivariateNormal(
                mean, torch.eye(mean.shape[1], device=mean.device) * std
            ).log_prob(xt)
            return log_p1t

        def guide_fn(x, t, dx_dt, model):
            """
            Args:
                t: float
                x: Tensor, shape (b, T, C)
                dx_dt: Tensor, shape (b, T, C)
            """
            # estimate E (e^{-J} / Z - 1) * u
            MC_EP = self.cfg.mc_ep
            MC_B = self.cfg.mc_batch_size
            assert MC_B == x1.shape[0], "MC_B must be the same as the number of samples in x1"
            SCALE = self.cfg.mc_scale
            OT_STD = self.cfg.mc_ot_std
            b = x.shape[0]
            x_ = x.repeat(MC_B, 1, 1) # (MC_B * b, T, C)
            x1_ = x1.unsqueeze(0).repeat(b, 1, 1, 1).permute(1, 0, 2, 3).reshape(-1, *x1.shape[1:]) # (MC_B * b, T, C)
            
            if self.cfg.flow_matching_type == 'cfm':
                log_p_t1_x = cfm_log_p_t1(x1_, x_, t, epsilon=MC_EP) # (MC_B * b)
                
                if cached_v is None:
                    v_ = self.value_model(x1_)[:, -1, 0]
                else:
                    v_ = cached_v.clone()
                
                if self.cfg.mc_linear_J:
                    J_ = SCALE * v_ # value model output is (B, T, 1) but only the last step is used. J_: (MC_B * b)
                    if self.cfg.mc_self_normalize:
                        J_ = ((J_ - J_.mean()) / (J_.std() + 1e-8)).clamp(0)
                else:
                    # self normalize
                    if self.cfg.mc_self_normalize:
                        v_ = (v_ - v_.mean()) / (v_.std() + 1e-8)
                    J_ = torch.exp(SCALE * v_) # value model output is (B, T, 1) but only the last step is used. J_: (MC_B * b)
                
                log_p_t1_x = log_p_t1_x.reshape(MC_B, b, 1, 1)
                log_p_t_x = log_p_t1_x.logsumexp(0) - torch.log(torch.tensor(MC_B)) # (MC_B, B, 1, 1) -> (B, 1, 1)
                # Z = (p_t1_x * J_).reshape(MC_B, b, 1, 1).mean(0) / (p_t_x + 1e-8) # (MC_B, B, 1, 1) -> (B, 1, 1)
                log_p_t1_x_times_J_ = log_p_t1_x + torch.log(J_).reshape(MC_B, b, 1, 1) # (MC_B, b, 1, 1)
                log_Z = log_p_t1_x_times_J_.logsumexp(0) - torch.log(torch.tensor(MC_B)) - log_p_t_x # (B, 1, 1)
                Z = torch.exp(log_Z) # (B, 1, 1)

                u = (x1_ - x_) / (1 - t + MC_EP) # (MC_B * b, T, C)
                g = torch.exp(log_p_t1_x - log_p_t_x) \
                    * (J_.reshape(MC_B, b, 1, 1) / (Z + 1e-8) - 1) \
                    * u.reshape(MC_B, b, *x_.shape[1:]) # (MC_B, b, T, C)
                return g.mean(0) # (MC_B, B, T, C) -> (B, T, C)

            elif self.cfg.flow_matching_type == 'ot_cfm':
                try:
                    x0_, x1_ = self._get_cached_ot_cfm_plan()
                except:
                    x0_ = torch.randn(MC_B, *x.shape[1:], device=x.device) # (MC_B, T, C)
                    x0_, x1_ = OTPlanSampler(method='exact').sample_plan(x0_, x1_)
                    x0_ = x0_.unsqueeze(0).repeat(b, 1, 1, 1).permute(1, 0, 2, 3).reshape(-1, *x.shape[1:]) # (MC_B * b, T, C)
                    x1_ = x1_.unsqueeze(0).repeat(b, 1, 1, 1).permute(1, 0, 2, 3).reshape(-1, *x.shape[1:]) # (MC_B * b, T, C)
                    self._save_cached_ot_cfm_plan(x0_, x1_)
                log_p_t1_x = ot_cfm_log_p_tz(x0_, x1_, x_, t, std=OT_STD) # (MC_B * b)
                
                if cached_v is None:
                    v_ = self.value_model(x1_)[:, -1, 0]
                else:
                    v_ = cached_v.clone()
                
                if self.cfg.mc_linear_J:
                    J_ = SCALE * v_ # value model output is (B, T, 1) but only the last step is used. J_: (MC_B * b)
                    if self.cfg.mc_self_normalize:
                        J_ = ((J_ - J_.mean()) / (J_.std() + 1e-8)).clamp(0)
                else:
                    # self normalize
                    if self.cfg.mc_self_normalize:
                        v_ = (v_ - v_.mean()) / (v_.std() + 1e-8)
                    J_ = torch.exp(SCALE * v_) # value model output is (B, T, 1) but only the last step is used. J_: (MC_B * b)
                
                log_p_t1_x = log_p_t1_x.reshape(MC_B, b, 1, 1)
                log_p_t_x = log_p_t1_x.logsumexp(0) - torch.log(torch.tensor(MC_B)) # (MC_B, B) -> (B, 1, 1)
                # Z = (p_t1_x * J_).reshape(MC_B, b, 1, 1).mean(0) / (p_t_x + 1e-8) # (MC_B, B) -> (B, 1, 1)
                log_p_t1_x_times_J_ = log_p_t1_x + torch.log(J_).reshape(MC_B, b, 1, 1) # (MC_B, b, 1, 1)
                log_Z = log_p_t1_x_times_J_.logsumexp(0) - torch.log(torch.tensor(MC_B)) - log_p_t_x # (B, 1, 1)
                Z = torch.exp(log_Z) # (B, 1, 1)

                u = x1_ - x0_ # (MC_B * b, T, C)
                g = torch.exp(log_p_t1_x - log_p_t_x) \
                    * (J_.reshape(MC_B, b, 1, 1) / (Z + 1e-8) - 1) \
                    * u.reshape(MC_B, b, *x_.shape[1:]) # (MC_B, b, T, C)
                return g.mean(0) # (MC_B, B, T, C) -> (B, T, C)
            else:
                raise ValueError(f"Unsupported flow matching type: {self.cfg.flow_matching_type}")
        return guide_fn
    
    def mc_forward(self, conditions, batch_size=1):
        # assert batch_size == 1, "env batch_size must be 1 for now" # but SS_B can be > 1
        if batch_size > 1:
            print("WARNING: batch_size > 1 for MC, this is not tested")
        assert self.cfg.guidance_method == 'mc', f"guidance_method must be mc, but got {self.cfg.guidance_method}"
        
        b = self.cfg.mc_ss

        # Only normalize the observation
        conditions = utils.apply_dict(self.normalizer.normalize, conditions, 'observations')

        # first, sample support set x1 ~ p_1(x)
        solver = ConditionedODESolver(
            self.flow_model, 
            conditions, 
            guide_fn=None, 
            ode_method=self.cfg.ode_solver,
            action_dim=self.action_dim,
        )
        x = torch.randn(self.cfg.mc_batch_size, self.horizon, self.action_dim + self.state_dim, device=self.cfg.device) # (B, T, C)
        x = apply_conditioning(x, to_torch(conditions, device=x.device), self.action_dim)
        with torch.no_grad():
            x1_support = solver(x, t_span=torch.linspace(
                *self.cfg.ode_t_span, self.cfg.ode_t_steps, device=x.device
            )) # (MC_B, T, C)
        
        # Then sample guided x1 ~ p_1(x) e^{R(x)} / Z 
        # precompute the value model output for the support set
        x1_support_rep = x1_support.unsqueeze(0).repeat(b * batch_size, 1, 1, 1).permute(1, 0, 2, 3).reshape(-1, *x1_support.shape[1:]) # (MC_B * b, T, C)
        v_support = self.value_model(x1_support_rep)[:, -1, 0].detach() # (MC_B * b)
        
        solver = ConditionedODESolver(
            self.flow_model, 
            conditions, 
            guide_fn=self.get_mc_guide_fn(x1_support, cached_v=v_support), 
            ode_method=self.cfg.ode_solver,
            action_dim=self.action_dim,
        )
        x = torch.randn(b * batch_size, self.horizon, self.action_dim + self.state_dim, device=self.cfg.device) # (B, T, C)
        x = apply_conditioning(x, to_torch(conditions, device=x.device), self.action_dim)
        with torch.no_grad():
            x = solver(x, t_span=torch.linspace(
                *self.cfg.ode_t_span, self.cfg.ode_t_steps, device=x.device
            )) # (B, T, C)

        self._remove_cached_ot_cfm_plan()

        normed_actions = x[:, :, :self.action_dim]
        actions = torch.tensor(self.normalizer.unnormalize(normed_actions, 'actions'), device=self.cfg.device)

        normed_observations = x[:, :, self.action_dim:]
        observations = torch.tensor(self.normalizer.unnormalize(normed_observations, 'observations'), device=self.cfg.device) # NOTE: we do need to make torch tensor, otherwise the indexing later will be wrong
        
        values = self.value_model(torch.cat([normed_actions, normed_observations], dim=-1)) # (B_ss * B, T, 1)
        values = values[:, -1, 0] # (B_ss * B)
        values = values.reshape(b, batch_size) # (B_ss, B)
        best_idx = values.argmax(dim=0).to(self.cfg.device) # (B,)
        
        # to construct trajectories
        best_values = values[best_idx, torch.arange(batch_size, device=self.cfg.device)] # (B,)
        best_observations = observations.reshape(b, batch_size, self.horizon, self.state_dim)[best_idx, torch.arange(batch_size, device=self.cfg.device)] # (B, T, C)
        best_actions = actions.reshape(b, batch_size, self.horizon, self.action_dim)[best_idx, torch.arange(batch_size, device=self.cfg.device)] # (B, T, C)
        
        trajectories = Trajectories(to_np(best_actions), to_np(best_observations), to_np(best_values))

        # output actions
        actions = actions.reshape(b, batch_size, self.horizon, self.action_dim)[best_idx, torch.arange(batch_size, device=self.cfg.device)] # (B, T, C)
        actions = to_np(actions[0, 0]) # (C,), simply get the first action in the first sample in the 
        
        return actions, trajectories


    ### Learned Guidance ###

    def get_learned_guidance_model(self, conditions):
        """
        Return the guidance model for the flow model.
        """
        def guide_fn(x, t, dx_dt, flow_model):
            if self.cfg.guide_matching_type != 'grad_z':
                with torch.no_grad():
                    guidance = self.guide_model(x, t) # input like flow model. x(B, T, C), t (,) or (B, )
            else:
                logz = self.guide_model(x, t)[:, -1, 0] # output is model_z output
                guidance = torch.autograd.grad([logz.sum()], [x])[0].detach()

            return guidance * self.cfg.guide_inference_scale
        return guide_fn

    def learned_guidance_forward(self, conditions, batch_size=1):
        """
        Use learned guidance to generate actions.
        """
        # assert batch_size == 1, "batch_size must be 1 for now"
        assert self.cfg.guidance_method == 'guidance_matching', f"guidance_method must be learned, but got {self.cfg.guidance_method}"
        assert self.guide_model is not None, "guide_model is not provided"

        # Only normalize the observation
        conditions = utils.apply_dict(self.normalizer.normalize, conditions, 'observations')

        # Generate actions
        solver = ConditionedODESolver(
            self.flow_model, 
            conditions, 
            guide_fn=self.get_learned_guidance_model(conditions), 
            ode_method=self.cfg.ode_solver,
            action_dim=self.action_dim,
        )        
    
        x = torch.randn(batch_size, self.horizon, self.action_dim + self.state_dim, device=self.cfg.device) # (B, T, C)
        x = apply_conditioning(x, to_torch(conditions, device=x.device), self.action_dim)

        x = solver(x, t_span=torch.linspace(
            *self.cfg.ode_t_span, self.cfg.ode_t_steps, device=x.device
        )) # (B, T, C)

        normed_actions = x[:, :, :self.action_dim]
        actions = self.normalizer.unnormalize(normed_actions, 'actions')

        normed_observations = x[:, :, self.action_dim:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')
        
        values = self.value_model(torch.cat([normed_actions, normed_observations], dim=-1))
        trajectories = Trajectories(actions, observations, values)
        
        # TODO: Add more "guidance" methods, including sample and selection-based MPC
        actions = actions[0, 0] # simply get the first action in the first sample in the batch
        
        return actions, trajectories
    
    ### Simple p(z|x_1) MC guidance ###
    
    def get_sim_mc_guidance_model(self, value_model, schedule_fn, scale):
        """
        Return the guidance model for the flow model.
        """

        def guide_fn(x, t, dx_dt, flow_model):
            """
            Implements guidance following Eq. 12
            Args:
                t: flow time. float
                x: current sample x_t. Tensor, shape (b, dim)
                dx_dt: current predicted VF. Tensor, shape (b, dim)
                model: flow model. MLP
            """
            x1_pred = x + dx_dt * (1 - t) # (B, 2)

            x1 = torch.randn_like(
                x1_pred.unsqueeze(0).repeat(self.cfg.sim_mc_n, 1, 1, 1)
            ) * self.cfg.sim_mc_std + x1_pred # (cfg.sim_mc_n, B, C, T)
            values = value_model(x1.reshape(-1, *x1.shape[2:]))[:, -1, 0] # (cfg.sim_mc_n * B)
            if self.cfg.sim_mc_self_normalize:
                values = (values - values.mean()) / (values.std() + 1e-8) # (cfg.sim_mc_n * B)
            Jx1_ = torch.exp(
                self.cfg.sim_mc_J_scale * values
            ).reshape(self.cfg.sim_mc_n, -1) # (cfg.sim_mc_n, B)
            v = (x1 - x) / (1 - t + self.cfg.sim_mc_eps)  # Conditional VF v_{t|z} in Eq. 12 (cfg.sim_mc_n, B, C, T)
            Z = Jx1_.mean(0) + 1e-8  # Z in Eq. 12 (B,)
            g = (Jx1_ / Z - 1).reshape(self.cfg.sim_mc_n, -1, 1, 1) * v  # g in Eq. 12 (cfg.sim_mc_n, B, C, T)
            g = g.mean(0) # (B, C, T)
            return g * scale * schedule_fn(t)
        return guide_fn
    
    def sim_mc_guidance_forward(self, conditions, batch_size):
        """
        Use g^{\text{sim-MC}} guidance to generate actions.
        """
        # assert batch_size == 1, "batch_size must be 1 for now"

        # Only normalize the observation
        conditions = utils.apply_dict(self.normalizer.normalize, conditions, 'observations')

        # Generate actions
        solver = ConditionedODESolver(
            self.flow_model, 
            conditions, 
            guide_fn=self.get_sim_mc_guidance_model(
                self.value_model, 
                schedule_fn=self.get_scheduler(self.cfg.sim_mc_schedule), 
                scale=self.cfg.sim_mc_scale
            ), 
            ode_method=self.cfg.ode_solver,
            action_dim=self.action_dim,
        )
    
        x = torch.randn(batch_size, self.horizon, self.action_dim + self.state_dim, device=self.cfg.device) # (B, T, C)
        x = apply_conditioning(x, to_torch(conditions, device=x.device), self.action_dim)

        x = solver(x, t_span=torch.linspace(
            *self.cfg.ode_t_span, self.cfg.ode_t_steps, device=x.device
        )) # (B, T, C)

        normed_actions = x[:, :, :self.action_dim]
        actions = self.normalizer.unnormalize(normed_actions, 'actions')

        normed_observations = x[:, :, self.action_dim:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')
        
        values = self.value_model(torch.cat([normed_actions, normed_observations], dim=-1))
        trajectories = Trajectories(actions, observations, values)
        
        # TODO: Add more "guidance" methods, including sample and selection-based MPC
        actions = actions[0, 0] # simply get the first action in the first sample in the batch
        
        return actions, trajectories
