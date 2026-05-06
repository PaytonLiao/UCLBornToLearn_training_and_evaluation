from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_mlp(in_dim: int, hidden_dims: List[int], out_dim: int) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


@dataclass
class VAELossBreakdown:
    total_loss: torch.Tensor
    recon_loss: torch.Tensor
    kl_div: torch.Tensor
    elbo: torch.Tensor


class VanillaVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        encoder_hidden_dims: List[int],
        decoder_hidden_dims: List[int],
    ) -> None:
        super().__init__()
        if not encoder_hidden_dims:
            raise ValueError("encoder_hidden_dims must be non-empty.")
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder_backbone = build_mlp(input_dim, encoder_hidden_dims, encoder_hidden_dims[-1])
        self.mu_head = nn.Linear(encoder_hidden_dims[-1], latent_dim)
        self.logvar_head = nn.Linear(encoder_hidden_dims[-1], latent_dim)
        self.decoder = build_mlp(latent_dim, decoder_hidden_dims, input_dim)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_backbone(x)
        return self.mu_head(h), self.logvar_head(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # std = torch.exp(0.5 * logvar)
        # eps = torch.randn_like(std)
        # return mu + eps * std
        logvar = torch.clamp(logvar, -10, 10)  # 🔴 MOVE CLAMP HERE
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # return self.decoder(z)
        return torch.tanh(self.decoder(z))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return {"recon": recon, "mu": mu, "logvar": logvar, "z": z}

    @staticmethod
    def loss_function(x: torch.Tensor, recon: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> VAELossBreakdown:
        # recon_loss = F.mse_loss(recon, x, reduction="mean")
        # kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        # elbo = recon_loss + kl_div
        # total_loss = elbo
        # return VAELossBreakdown(
        #     total_loss=total_loss,
        #     recon_loss=recon_loss,
        #     kl_div=kl_div,
        #     elbo=elbo,
        # )
        logvar = torch.clamp(logvar, -10, 10)

        recon_loss = F.mse_loss(recon, x, reduction="mean")

        kl_div = -0.5 * torch.mean(
            1 + logvar - mu.pow(2) - torch.exp(logvar)
        )

        beta = 0.001  # tune this
        total_loss = recon_loss + beta * kl_div

        return VAELossBreakdown(
            total_loss=total_loss,
            recon_loss=recon_loss,
            kl_div=kl_div,
            elbo=total_loss,
        )
