import os
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Optional
from argparse import Namespace
from tqdm import tqdm
import numpy as np
import wandb

from scaler_gan.network_topology.networks import DiffusionPostNet
from scaler_gan.trainer.scalerGAN import ScalerGANTrainer
from scaler_gan.scalergan_utils.global_logger import logger


class PostNetTrainer:
    """
    Trainer for DiffATSM's Diffusion-based PostNet.

    DiffATSM Paper Section 3.2 & Algorithm 1:
    Trains the PostNet to predict noise in the diffusion process,
    conditioned on the reconstructed mel from the adaptive generator.
    """

    def __init__(self, conf: Namespace, generator_checkpoint: str):
        """
        Initialize PostNet trainer.

        Args:
            conf: Configuration namespace
            generator_checkpoint: Path to trained adaptive generator checkpoint
        """
        self.conf = conf
        self.device = conf.device

        # Paper Section 4.2: Diffusion hyperparameters
        self.T = 100
        self.beta_start = 1e-4
        self.beta_end = 0.05

        self._setup_diffusion_schedule()

        # Load frozen adaptive generator
        logger.info("Loading trained adaptive generator...")

        # Force PPG architecture to match checkpoint and set correct path
        conf.checkpoint_path = generator_checkpoint
        conf.use_ppg = True
        conf.ppg_input_dim = getattr(conf, 'ppg_input_dim', 768)
        conf.ppg_hidden_dim = getattr(conf, 'ppg_hidden_dim', 256)

        self.generator_trainer = ScalerGANTrainer(conf, inference=True)
        self.generator_trainer.G.eval()

        for param in self.generator_trainer.G.parameters():
            param.requires_grad = False

        logger.info("✅ Generator loaded and frozen")

        # Initialize PostNet
        self.postnet = DiffusionPostNet(
            mel_bins=conf.mel_params['num_mels'],
            residual_channels=getattr(conf, 'postnet_channels', 256),
            n_residual_blocks=getattr(conf, 'postnet_blocks', 20),
            time_emb_dim=128,
        ).to(self.device)

        # Paper: Adam with lr=0.001
        self.optimizer = torch.optim.Adam(
            self.postnet.parameters(),
            lr=getattr(conf, 'postnet_lr', 0.001)
        )

        self.current_iter = 0
        self.current_epoch = 0

        logger.info(
            f"PostNet initialized with "
            f"{sum(p.numel() for p in self.postnet.parameters()):,} parameters"
        )

    # =========================================================================
    # Diffusion Schedule
    # =========================================================================

    def _setup_diffusion_schedule(self):
        """
        Setup linear diffusion schedule.

        Paper Section 4.2:
        "linear noise schedule ranging from 1x10^-4 to 0.05"
        """
        self.betas = torch.linspace(
            self.beta_start, self.beta_end, self.T, device=self.device
        )
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        alphas_cumprod_prev = torch.cat([
            torch.ones(1, device=self.device),
            self.alphas_cumprod[:-1]
        ])
        self.posterior_variance = (
            self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    # =========================================================================
    # Generator Utilities
    # =========================================================================

    def generate_xrecon_batch(self, mel_batch: torch.Tensor) -> torch.Tensor:
        """
        Generate reconstructed mels using trained generator.

        Paper Figure 1:
        "The scaled mel spectrogram is inversely scaled by r^-1.
         The reconstructed output serves as the conditioning."

        Args:
            mel_batch: (B, 1, F, T)
        Returns:
            xrecon: (B, F, T)
        """
        with torch.no_grad():
            B, C, F, T = mel_batch.shape

            from scaler_gan.scalergan_utils.scalergan_utils import random_size
            output_size, random_affine = random_size(
                orig_size=(F, T),
                curriculum=self.conf.curriculum,
                i=self.current_iter,
                epoch_for_max_range=self.conf.epoch_for_max_range,
                must_divide=self.conf.must_divide,
                min_scale=self.conf.min_scale,
                max_scale=self.conf.max_scale,
                max_transform_magnitude=self.conf.max_transform_magnitude,
            )

            xr = self.generator_trainer.G(
                mel_batch,
                output_size=output_size,
                random_affine=random_affine,
                ppg=None,
            )

            xrecon = self.generator_trainer.G(
                xr,
                output_size=(F, T),
                random_affine=-random_affine if random_affine is not None else None,
                ppg=None,
            )

            xrecon = xrecon.squeeze(1)  # (B, F, T)

        return xrecon

    # =========================================================================
    # Training
    # =========================================================================

    def train_one_step(self, mel_batch: torch.Tensor) -> float:
        """
        Single training step (Paper Algorithm 1).

        Args:
            mel_batch: (B, 1, F, T)
        Returns:
            loss value
        """
        self.optimizer.zero_grad()

        B, C, F, T = mel_batch.shape

        xrecon = self.generate_xrecon_batch(mel_batch)  # (B, F, T)
        x = mel_batch.squeeze(1)                         # (B, F, T)

        epsilon = torch.randn_like(x)
        t = torch.randint(0, self.T, (B,), device=self.device).long()

        sqrt_alpha_t = self.sqrt_alphas_cumprod[t].view(B, 1, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1)

        xt = sqrt_alpha_t * x + sqrt_one_minus_alpha_t * epsilon

        epsilon_pred = self.postnet(xt, t, xrecon)

        loss = nn.functional.mse_loss(epsilon_pred, epsilon)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train_one_epoch(self, train_loader):
        """
        Train PostNet for one epoch. Saves checkpoint once per epoch.

        Args:
            train_loader: DataLoader for training data
        """
        self.postnet.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {self.current_epoch + 1} | Iter {self.current_iter}",
            ncols=110
        )

        for batch in pbar:
            if isinstance(batch, (list, tuple)):
                mel_batch = batch[0].to(self.device)
            else:
                mel_batch = batch.to(self.device)

            loss = self.train_one_step(mel_batch)

            epoch_loss += loss
            num_batches += 1
            self.current_iter += 1

            pbar.set_postfix({
                'loss': f'{loss:.4f}',
                'iter': self.current_iter
            })

            if self.current_iter >= self.conf.postnet_iterations:
                break

        avg_loss = epoch_loss / max(num_batches, 1)
        self.current_epoch += 1

        # Log to wandb
        if self.conf.wandb:
            wandb.log({
                'postnet/loss': avg_loss,
                'postnet/epoch': self.current_epoch,
                'postnet/iter': self.current_iter,
            }, step=self.current_iter)

        # Save checkpoint once per epoch
        self.save_checkpoint()

        logger.info(
            f"Epoch {self.current_epoch} | "
            f"Iter {self.current_iter:,} | "
            f"Avg Loss: {avg_loss:.6f}"
        )

        # Visualize every 2 epochs
        if self.current_epoch % 2 == 0 and self.conf.wandb:
            sample_batch = next(iter(train_loader))
            if isinstance(sample_batch, (list, tuple)):
                sample_mel = sample_batch[0][:1].to(self.device)
            else:
                sample_mel = sample_batch[:1].to(self.device)
            self.visualize(sample_mel)
            logger.info(f"📊 Logged mel visualizations for epoch {self.current_epoch}")

        return avg_loss

    # =========================================================================
    # Visualization
    # =========================================================================

    def _mel_to_figure(self, mels: dict) -> plt.Figure:
        """
        Plot multiple mel spectrograms side by side.

        Args:
            mels: dict of {title: (F, T) tensor}
        Returns:
            matplotlib Figure
        """
        n = len(mels)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
        if n == 1:
            axes = [axes]
        for ax, (title, mel) in zip(axes, mels.items()):
            mel_np = mel.detach().cpu().float().numpy()
            im = ax.imshow(
                mel_np, aspect='auto', origin='lower',
                interpolation='none', vmin=-1, vmax=1, cmap='magma'
            )
            ax.set_title(title, fontsize=9)
            ax.set_xlabel('Time')
            ax.set_ylabel('Mel bins')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        return fig

    @torch.no_grad()
    def _run_reverse_diffusion(
        self,
        xr: torch.Tensor,
        xrecon: torch.Tensor
    ) -> torch.Tensor:
        """
        Full reverse diffusion (Algorithm 2) for visualization.

        Args:
            xr: scaled mel (B, F, T) — defines output size
            xrecon: conditioning mel (B, F, T)
        Returns:
            x0: denoised mel (B, F, T)
        """
        B, F, T = xr.shape
        xt = torch.randn_like(xr)

        for t_idx in reversed(range(self.T)):
            t_tensor = torch.full((B,), t_idx, device=self.device, dtype=torch.long)

            eps_pred = self.postnet(xt, t_tensor, xrecon)

            alpha_t = self.alphas[t_idx]
            sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t_idx]

            xt = (1.0 / torch.sqrt(alpha_t)) * (
                xt - ((1 - alpha_t) / sqrt_one_minus) * eps_pred
            )

            if t_idx > 0:
                z = torch.randn_like(xt)
                sigma_t = torch.sqrt(self.posterior_variance[t_idx])
                xt = xt + sigma_t * z

        return xt

    @torch.no_grad()
    def visualize(self, mel_batch: torch.Tensor):
        """
        Log mel visualizations to wandb every 2 epochs.

        Panels:
          - Original x
          - xrecon (generator conditioning)
          - x_t at t=50 (mid-noise)
          - x_T (pure noise)
          - ε actual vs ε predicted (at t=50)
          - Denoised x₀ (full reverse diffusion)

        Args:
            mel_batch: (1, 1, F, T) single sample
        """
        self.postnet.eval()

        x = mel_batch[:1]           # (1, 1, F, T)
        B, C, F, T = x.shape

        # Generate xrecon
        from scaler_gan.scalergan_utils.scalergan_utils import random_size
        output_size, random_affine = random_size(
            orig_size=(F, T),
            curriculum=self.conf.curriculum,
            i=self.current_iter,
            epoch_for_max_range=self.conf.epoch_for_max_range,
            must_divide=self.conf.must_divide,
            min_scale=self.conf.min_scale,
            max_scale=self.conf.max_scale,
            max_transform_magnitude=self.conf.max_transform_magnitude,
        )
        xr = self.generator_trainer.G(
            x, output_size=output_size,
            random_affine=random_affine, ppg=None
        )
        xrecon = self.generator_trainer.G(
            xr, output_size=(F, T),
            random_affine=-random_affine, ppg=None
        )

        x_2d      = x.squeeze(1).squeeze(0)        # (F, T)
        xrecon_2d = xrecon.squeeze(1).squeeze(0)   # (F, T)
        eps       = torch.randn_like(x.squeeze(1)) # (1, F, T)

        # x_t at t=50
        x_t50 = (
            self.sqrt_alphas_cumprod[50] * x.squeeze(1)
            + self.sqrt_one_minus_alphas_cumprod[50] * eps
        ).squeeze(0)  # (F, T)

        # x_T — fully noised
        x_T = (
            self.sqrt_alphas_cumprod[-1] * x.squeeze(1)
            + self.sqrt_one_minus_alphas_cumprod[-1] * eps
        ).squeeze(0)  # (F, T)

        # ε predicted at t=50
        t50 = torch.tensor([50], device=self.device, dtype=torch.long)
        xt50_input = (
            self.sqrt_alphas_cumprod[50] * x.squeeze(1)
            + self.sqrt_one_minus_alphas_cumprod[50] * eps
        )
        eps_pred = self.postnet(
            xt50_input, t50, xrecon.squeeze(1)
        ).squeeze(0)  # (F, T)

        # Full denoised output
        x0_denoised = self._run_reverse_diffusion(
            xr.squeeze(1), xrecon.squeeze(1)
        ).squeeze(0)  # (F, T)

        fig = self._mel_to_figure({
            'Original x':       x_2d,
            'xrecon (cond)':    xrecon_2d,
            'x_t (t=50)':       x_t50,
            'x_T (pure noise)': x_T,
            'ε actual':         eps.squeeze(0),
            'ε predicted':      eps_pred,
            'Denoised x₀':      x0_denoised,
        })

        wandb.log({
            'postnet/mel_panels': wandb.Image(fig),
            'postnet/epoch': self.current_epoch,
        }, step=self.current_iter)

        plt.close(fig)
        self.postnet.train()

    # =========================================================================
    # Checkpointing
    # =========================================================================

    def save_checkpoint(self):
        """Save PostNet checkpoint (called once per epoch)."""
        checkpoint_dir = os.path.join(self.conf.artifacts_dir, 'postnet_checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_path = os.path.join(
            checkpoint_dir,
            f'postnet_epoch{self.current_epoch:04d}_iter{self.current_iter:06d}.pth.tar'
        )

        torch.save({
            'postnet': self.postnet.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iteration': self.current_iter,
            'epoch': self.current_epoch,
            'betas': self.betas,
            'alphas': self.alphas,
            'alphas_cumprod': self.alphas_cumprod,
        }, checkpoint_path)

        logger.info(
            f"💾 Checkpoint saved: epoch {self.current_epoch}, "
            f"iter {self.current_iter:,} → {checkpoint_path}"
        )

    def load_checkpoint(self, checkpoint_path: str):
        """Load PostNet checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.postnet.load_state_dict(checkpoint['postnet'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.current_iter = checkpoint['iteration']
        self.current_epoch = checkpoint.get('epoch', 0)

        logger.info(
            f"✅ Loaded PostNet checkpoint: "
            f"epoch {self.current_epoch}, iter {self.current_iter:,}"
        )




