"""Conjugate-gradient data consistency for distilled diffusion inference."""

import torch


def get_conditioning_method(name, operator, noiser, **kwargs):
    """Build the sole conditioning method supported by this minimal package."""
    if name != 'ps_cg':
        raise ValueError(f"Only 'ps_cg' is included in this package, got {name!r}.")
    return ConjugateGradientPosteriorSampling(operator=operator, noiser=noiser, **kwargs)


class ConjugateGradientPosteriorSampling:
    """Correct a diffusion clean prediction with a few CG iterations.

    The corrector solves, independently for every item in a batch,

        (tau^-1 I + scale / sigma_y^2 A^T A) x =
            tau^-1 x0_hat + scale / sigma_y^2 A^T y.

    It then maps the clean-space update back to the current diffusion state.
    """

    mode = 'cg'

    def __init__(self, operator, noiser, **kwargs):
        mode = kwargs.get('mode', 'cg')
        if mode != 'cg':
            raise ValueError("This inference package only supports conditioning mode 'cg'.")

        self.operator = operator
        self.noiser = noiser
        self.scale = float(kwargs.get('scale', 1.0))
        self.tau = float(kwargs.get('tau', 1.0))
        self.sigma_y = float(kwargs.get('sigma_y', 0.05))
        self.cg_iters = int(kwargs.get('cg_iters', 3))
        self.cg_tol = float(kwargs.get('cg_tol', 1e-6))

        if self.scale < 0:
            raise ValueError('scale must be non-negative.')
        if self.tau <= 0 or self.sigma_y <= 0:
            raise ValueError('tau and sigma_y must be positive.')
        if self.cg_iters < 1:
            raise ValueError('cg_iters must be at least one.')

    @staticmethod
    def _batch_dot(left, right):
        """Inner product per batch item, broadcastable over a field tensor."""
        return torch.sum(left * right, dim=tuple(range(1, left.ndim)), keepdim=True)

    def conjugate_gradient(self, matvec, right_hand_side, initial_guess):
        """Solve ``H x = b`` with a fixed, small number of batched CG steps."""
        solution = initial_guess.clone()
        residual = right_hand_side - matvec(solution)
        direction = residual.clone()
        residual_norm_sq = self._batch_dot(residual, residual)

        for _ in range(self.cg_iters):
            h_direction = matvec(direction)
            step_size = residual_norm_sq / (self._batch_dot(direction, h_direction) + 1e-12)
            solution = solution + step_size * direction
            residual = residual - step_size * h_direction
            next_residual_norm_sq = self._batch_dot(residual, residual)

            if torch.all(torch.sqrt(next_residual_norm_sq) < self.cg_tol):
                break

            direction = residual + (next_residual_norm_sq / (residual_norm_sq + 1e-12)) * direction
            residual_norm_sq = next_residual_norm_sq

        return solution

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, alpha_bar_t, **kwargs):
        """Apply CG in x0-space and return the corrected current sample.

        ``x_prev`` is accepted to keep the sampler interface explicit; it is
        not required by the closed-form quadratic correction.
        """
        del x_prev, kwargs
        with torch.no_grad():
            alpha_bar_t = torch.as_tensor(alpha_bar_t, device=x_t.device, dtype=x_t.dtype)

            def hessian_vector_product(vector):
                return (
                    vector / self.tau
                    + (self.scale / self.sigma_y ** 2)
                    * self.operator.transpose(self.operator.forward(vector))
                )

            right_hand_side = (
                x_0_hat / self.tau
                + (self.scale / self.sigma_y ** 2) * self.operator.transpose(measurement)
            )
            x0_cg = self.conjugate_gradient(
                hessian_vector_product, right_hand_side, initial_guess=x_0_hat
            )

            residual_norm = torch.linalg.vector_norm(
                self.operator.forward(x0_cg) - measurement
            )
            clean_update = x0_cg - x_0_hat
            corrected_x_t = x_t + torch.sqrt(alpha_bar_t) * clean_update

        return corrected_x_t, residual_norm, clean_update
