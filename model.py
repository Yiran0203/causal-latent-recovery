# ==== VAE, VAE+u and iVAE models ====

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


class MLP(nn.Module):
    """A small two-hidden-layer multilayer perceptron."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class GaussianLatentModel(nn.Module):
    """
    Shared implementation for the three G2 model conditions.

    All models reconstruct x = [x2, x3, x4, x5].

    Differences
    -----------
    VAE:
        q(z | x), p(z) = Normal(0, I)

    VAEWithU:
        q(z | x, u), p(z) = Normal(0, I)
        Here u is supplied to the encoder, but the prior is not conditional.

    iVAE:
        q(z | x, u), p(z | u)
    """

    def __init__(
        self,
        x_dim: int,
        u_dim: int,
        z_dim: int,
        hidden_dim: int,
        encoder_uses_u: bool,
        conditional_prior: bool,
        decoder_std: float = 0.1,
    ) -> None:
        super().__init__()

        if decoder_std <= 0:
            raise ValueError("decoder_std must be positive.")

        self.x_dim = x_dim
        self.u_dim = u_dim
        self.z_dim = z_dim
        self.encoder_uses_u = encoder_uses_u
        self.conditional_prior = conditional_prior

        encoder_input_dim = x_dim + (u_dim if encoder_uses_u else 0)

        self.encoder_mean = MLP(
            encoder_input_dim,
            hidden_dim,
            z_dim,
        )
        self.encoder_logvar = MLP(
            encoder_input_dim,
            hidden_dim,
            z_dim,
        )

        if conditional_prior:
            self.prior_mean_net: MLP | None = MLP(
                u_dim,
                hidden_dim,
                z_dim,
            )
            self.prior_logvar_net: MLP | None = MLP(
                u_dim,
                hidden_dim,
                z_dim,
            )
        else:
            self.prior_mean_net = None
            self.prior_logvar_net = None

        self.decoder_mean_net = MLP(
            z_dim,
            hidden_dim,
            x_dim,
        )

        # The original iVAE-style fixed variance 0.01 corresponds
        # to a Normal standard deviation of sqrt(0.01) = 0.1.
        self.register_buffer(
            "decoder_std_buffer",
            torch.tensor(float(decoder_std)),
        )

    def _encoder_input(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
    ) -> torch.Tensor:
        if self.encoder_uses_u:
            return torch.cat([x, u], dim=1)
        return x

    def encoder_params(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the mean and log variance of q(z | input)."""
        encoder_input = self._encoder_input(x, u)
        posterior_mean = self.encoder_mean(encoder_input)
        posterior_logvar = self.encoder_logvar(encoder_input).clamp(
            min=-10.0,
            max=10.0,
        )
        return posterior_mean, posterior_logvar

    def prior_params(
        self,
        u: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return parameters of p(z) or p(z | u)."""
        batch_size = u.shape[0]

        if self.conditional_prior:
            if self.prior_mean_net is None or self.prior_logvar_net is None:
                raise RuntimeError("Conditional prior networks are missing.")

            prior_mean = self.prior_mean_net(u)
            prior_logvar = self.prior_logvar_net(u).clamp(
                min=-10.0,
                max=10.0,
            )
            return prior_mean, prior_logvar

        # Standard VAE prior p(z) = Normal(0, I).
        prior_mean = torch.zeros(
            batch_size,
            self.z_dim,
            device=u.device,
            dtype=u.dtype,
        )
        prior_logvar = torch.zeros_like(prior_mean)
        return prior_mean, prior_logvar

    def decoder_params(
        self,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reconstruction_mean = self.decoder_mean_net(z)
        decoder_std = self.decoder_std_buffer.expand_as(
            reconstruction_mean
        )
        return reconstruction_mean, decoder_std

    @staticmethod
    def logvar_to_std(logvar: torch.Tensor) -> torch.Tensor:
        return torch.exp(0.5 * logvar)

    def forward(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        posterior_mean, posterior_logvar = self.encoder_params(x, u)
        posterior_std = self.logvar_to_std(posterior_logvar)
        posterior_dist = Normal(posterior_mean, posterior_std)

        # rsample() uses the reparameterisation trick.
        z = posterior_dist.rsample()

        prior_mean, prior_logvar = self.prior_params(u)
        prior_std = self.logvar_to_std(prior_logvar)

        reconstruction_mean, decoder_std = self.decoder_params(z)

        return {
            "z": z,
            "posterior_mean": posterior_mean,
            "posterior_logvar": posterior_logvar,
            "posterior_std": posterior_std,
            "prior_mean": prior_mean,
            "prior_logvar": prior_logvar,
            "prior_std": prior_std,
            "reconstruction_mean": reconstruction_mean,
            "decoder_std": decoder_std,
        }

    def elbo(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
    ) -> torch.Tensor:
        """Return one ELBO value per observation, shape [batch_size]."""
        output = self.forward(x, u)
        z = output["z"]

        posterior_dist = Normal(
            output["posterior_mean"],
            output["posterior_std"],
        )
        prior_dist = Normal(
            output["prior_mean"],
            output["prior_std"],
        )
        decoder_dist = Normal(
            output["reconstruction_mean"],
            output["decoder_std"],
        )

        log_px_z = decoder_dist.log_prob(x).sum(dim=1)
        log_qz_input = posterior_dist.log_prob(z).sum(dim=1)
        log_pz = prior_dist.log_prob(z).sum(dim=1)

        return log_px_z + log_pz - log_qz_input


class VAE(GaussianLatentModel):
    """Standard VAE: q(z | x), p(z) = Normal(0, I)."""

    def __init__(
        self,
        x_dim: int = 4,
        u_dim: int = 3,
        z_dim: int = 1,
        hidden_dim: int = 50,
        decoder_std: float = 0.1,
    ) -> None:
        super().__init__(
            x_dim=x_dim,
            u_dim=u_dim,
            z_dim=z_dim,
            hidden_dim=hidden_dim,
            encoder_uses_u=False,
            conditional_prior=False,
            decoder_std=decoder_std,
        )


class VAEWithU(GaussianLatentModel):
    """
    VAE with auxiliary input u in the encoder:

        q(z | x, u), p(z) = Normal(0, I), p(x | z).

    u is not reconstructed. This isolates the effect of supplying u to
    the encoder from the effect of using iVAE's conditional prior.
    """

    def __init__(
        self,
        x_dim: int = 4,
        u_dim: int = 3,
        z_dim: int = 1,
        hidden_dim: int = 50,
        decoder_std: float = 0.1,
    ) -> None:
        super().__init__(
            x_dim=x_dim,
            u_dim=u_dim,
            z_dim=z_dim,
            hidden_dim=hidden_dim,
            encoder_uses_u=True,
            conditional_prior=False,
            decoder_std=decoder_std,
        )


class iVAE(GaussianLatentModel):
    """Identifiable VAE: q(z | x, u), p(z | u), p(x | z)."""

    def __init__(
        self,
        x_dim: int = 4,
        u_dim: int = 3,
        z_dim: int = 1,
        hidden_dim: int = 50,
        decoder_std: float = 0.1,
    ) -> None:
        super().__init__(
            x_dim=x_dim,
            u_dim=u_dim,
            z_dim=z_dim,
            hidden_dim=hidden_dim,
            encoder_uses_u=True,
            conditional_prior=True,
            decoder_std=decoder_std,
        )
