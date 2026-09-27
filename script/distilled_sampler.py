"""Distilled DDIM sampler for CG-guided posterior sampling.

Exposes the inference interfaces:
    p_sample_loop(model, x_start, measurement, measurement_cond_fn, fixed_condition, ...)
    p_sample(model, x, t, fixed_condition)
"""

import math
import numpy as np
import torch
from tqdm.auto import tqdm


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """Return the DDPM beta schedule used by the distilled teacher."""
    if schedule_name == 'linear':
        scale = 1000 / num_diffusion_timesteps
        return np.linspace(
            scale * 0.0001,
            scale * 0.02,
            num_diffusion_timesteps,
            dtype=np.float64,
        )
    if schedule_name == 'cosine':
        return _betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    raise ValueError(f"Unknown beta schedule: {schedule_name}")


def _betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for index in range(num_diffusion_timesteps):
        t1 = index / num_diffusion_timesteps
        t2 = (index + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas, dtype=np.float64)


def _extract(array, time_idx, target):
    """Select values by index and broadcast them to the target shape."""
    if not torch.is_tensor(array):
        array = torch.tensor(array, dtype=torch.float32, device=target.device)
    else:
        array = array.to(device=target.device, dtype=torch.float32)

    time_idx = time_idx.to(target.device).long()
    val = array[time_idx]
    while val.ndim < target.ndim:
        val = val.unsqueeze(-1)
    return val.expand_as(target)


class DistilledDDIM:
    """DDIM sampler using a student model and its distilled timestep sequence.

    The student typically uses far fewer steps than the original DDIM process.

    Args:
        alphas_cumprod: Full alpha-cumprod schedule, typically length 1000.
        student_timesteps: Ascending original schedule indices recorded during distillation.
        rescale_timesteps: Whether to rescale model timestep inputs.
    """

    def __init__(self, alphas_cumprod, student_timesteps, rescale_timesteps=False):
        self.alphas_cumprod_full = np.array(alphas_cumprod, dtype=np.float64)

        # Original schedule indices in ascending order.
        self.timestep_map = np.array(sorted(student_timesteps), dtype=np.int64)
        self.num_timesteps = len(self.timestep_map)
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = len(alphas_cumprod)

        # Build the local alpha schedule for the distilled timesteps.
        self.alphas_cumprod = np.array([alphas_cumprod[t] for t in self.timestep_map])

        # The predecessor of the first local timestep has alpha equal to one.
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])

        # Precompute coefficients used in sampling.
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

    def q_sample(self, x_start, t):
        """Sample q(x_t | x_0) using local timestep indices."""
        noise = torch.randn_like(x_start)
        coef1 = _extract(self.sqrt_alphas_cumprod, t, x_start)
        coef2 = _extract(self.sqrt_one_minus_alphas_cumprod, t, x_start)
        return coef1 * x_start + coef2 * noise

    def predict_eps_from_x_start(self, x_t, t, pred_xstart):
        """Recover epsilon from x_t and a predicted clean sample."""
        coef1 = _extract(self.sqrt_recip_alphas_cumprod, t, x_t)
        coef2 = _extract(self.sqrt_recipm1_alphas_cumprod, t, x_t)
        return (coef1 * x_t - pred_xstart) / coef2

    def _model_forward(self, model, x, t_local, fixed_condition=None):
        """Run the model after mapping local indices to original timesteps.

        Args:
            model: DiffusersUNetWrapper
            x: (B, C, H, W)
            t_local: tensor of local indices (0..num_timesteps-1)
            fixed_condition: (B, C_cond, H, W)

        Returns:
            eps_pred: (B, C, H, W)
            model_var_values: (B, C, H, W) or None
        """
        # Map local indices to original diffusion timesteps.
        map_tensor = torch.tensor(self.timestep_map, device=t_local.device, dtype=t_local.dtype)
        original_t = map_tensor[t_local]

        if self.rescale_timesteps:
            original_t = original_t.float() * (1000.0 / self.original_num_steps)

        model_output = model(x, original_t, fixed_condition=fixed_condition)

        C = x.shape[1]
        if model_output.shape[1] == 2 * C:
            eps_pred, model_var_values = torch.split(model_output, C, dim=1)
        else:
            eps_pred = model_output
            model_var_values = None

        return eps_pred, model_var_values

    def p_mean_variance(self, model, x, t, fixed_condition=None):
        """Compute the clean prediction and noise estimate for one DDIM step.

        Args:
            t: Tensor of local indices in ``[0, num_timesteps - 1]``.
        """
        eps_pred, _ = self._model_forward(model, x, t, fixed_condition)

        # Predict x_0 from epsilon.
        coef1 = _extract(self.sqrt_recip_alphas_cumprod, t, x)
        coef2 = _extract(self.sqrt_recipm1_alphas_cumprod, t, x)
        pred_xstart = coef1 * x - coef2 * eps_pred

        return {
            'pred_xstart': pred_xstart,
            'eps': eps_pred,
        }

    def p_sample(self, model, x, t, fixed_condition=None, eta=0.0):
        """Perform one DDIM sampling step.

        Args:
            t: Tensor of local timestep indices.
            eta: Stochasticity control; zero is deterministic.
        """
        out = self.p_mean_variance(model, x, t, fixed_condition)
        eps = out['eps']
        pred_xstart = out['pred_xstart']

        alpha_bar = _extract(self.alphas_cumprod, t, x)
        alpha_bar_prev = _extract(self.alphas_cumprod_prev, t, x)

        # DDIM variance
        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )

        # DDIM mean
        mean_pred = (
            pred_xstart * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )

        sample = mean_pred
        if t[0].item() != 0:
            noise = torch.randn_like(x)
            sample = sample + sigma * noise

        return {'sample': sample, 'pred_xstart': pred_xstart}

    def p_sample_loop(self, model, x_start, measurement, measurement_cond_fn,
                      record=False, save_root=None, fixed_condition=None,
                      sample_ids_batch=None):
        """Run the complete distilled posterior-sampling loop.

        Args:
            model: Distilled student model.
            x_start: Initial noise with shape ``(B, C, H, W)``.
            measurement: Low-resolution observation represented on the target grid.
            measurement_cond_fn: Data-consistency conditioning function.
            record: bool
            save_root: str
            fixed_condition: (B, C_cond, H, W)
        """
        img = x_start
        device = x_start.device

        # Iterate from the largest local index down to zero.
        pbar = tqdm(list(range(self.num_timesteps))[::-1], desc=f"Distilled DPS ({self.num_timesteps} steps)")
        conditioner = getattr(measurement_cond_fn, '__self__', None)
        uses_gradient_guidance = getattr(conditioner, 'mode', 'gradient') == 'gradient'

        for local_idx in pbar:
            time = torch.tensor([local_idx] * img.shape[0], device=device)

            if uses_gradient_guidance:
                img = img.requires_grad_()
                out = self.p_sample(x=img, t=time, model=model, fixed_condition=fixed_condition)
            else:
                img = img.detach()
                with torch.no_grad():
                    out = self.p_sample(x=img, t=time, model=model, fixed_condition=fixed_condition)

            # Construct the noisy measurement required by the sampler interface.
            time_measurement = time[:measurement.shape[0]]
            noisy_measurement = self.q_sample(measurement, t=time_measurement)

            # Supply the current local alpha_bar to the CG corrector.
            alpha_bar_t = torch.tensor(
                self.alphas_cumprod[local_idx],
                device=device,
                dtype=img.dtype
            )
            
            img, distance, norm_grad = measurement_cond_fn(
                x_t=out['sample'],
                measurement=measurement,
                noisy_measurement=noisy_measurement,
                x_prev=img,
                x_0_hat=out['pred_xstart'],
                alpha_bar_t=alpha_bar_t,
            )

            img = img.detach_()
            pbar.set_postfix({'distance': distance.item()}, refresh=False)
            # Release per-step tensors before the next denoising step.
            del out, noisy_measurement, distance, norm_grad
            torch.cuda.empty_cache()

        return img


def create_distilled_sampler(alphas_cumprod, student_timesteps, rescale_timesteps=False):
    """Create a distilled DDIM sampler.

    Args:
        alphas_cumprod: Full alpha-cumprod schedule.
        student_timesteps: Ascending distilled timestep indices.
        rescale_timesteps: bool

    Returns:
        A ``DistilledDDIM`` instance.
    """
    return DistilledDDIM(alphas_cumprod, student_timesteps, rescale_timesteps)
