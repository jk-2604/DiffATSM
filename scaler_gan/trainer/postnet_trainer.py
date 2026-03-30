#import os
#import torch
#import torch.nn as nn
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
#from typing import Optional, Tuple
#from argparse import Namespace
#from tqdm import tqdm
#import numpy as np
#import wandb
#
#from scaler_gan.network_topology.networks import DiffusionPostNet
#from scaler_gan.trainer.scalerGAN import ScalerGANTrainer
#from scaler_gan.scalergan_utils.global_logger import logger
#
#
#class PostNetTrainer:
#    """
#    Trainer for DiffATSM's Diffusion-based PostNet.
#
#    DiffATSM Paper Section 3.2 & Algorithm 1:
#    Trains the PostNet to predict noise in the diffusion process,
#    conditioned on the reconstructed mel from the adaptive generator.
#    """
#
#    def __init__(self, conf: Namespace, generator_checkpoint: str):
#        """
#        Initialize PostNet trainer.
#
#        Args:
#            conf: Configuration namespace
#            generator_checkpoint: Path to trained adaptive generator checkpoint
#        """
#        self.conf = conf
#        self.device = conf.device
#
#        # Paper Section 4.2: Diffusion hyperparameters
#        self.T = 100
#        self.beta_start = 1e-4
#        self.beta_end = 0.05
#
#        self._setup_diffusion_schedule()
#
#        # Load frozen adaptive generator
#        logger.info("Loading trained adaptive generator...")
#
#        # Force PPG architecture to match checkpoint and override checkpoint path
#        conf.checkpoint_path = generator_checkpoint
#        conf.use_ppg = True
#        conf.ppg_input_dim = getattr(conf, 'ppg_input_dim', 768)
#        conf.ppg_hidden_dim = getattr(conf, 'ppg_hidden_dim', 256)
#
#        self.generator_trainer = ScalerGANTrainer(conf, inference=True)
#        self.generator_trainer.G.eval()
#
#        for param in self.generator_trainer.G.parameters():
#            param.requires_grad = False
#
#        logger.info("✅ Generator loaded and frozen")
#
#        # Initialize PostNet
#        self.postnet = DiffusionPostNet(
#            mel_bins=conf.mel_params['num_mels'],
#            residual_channels=getattr(conf, 'postnet_channels', 256),
#            n_residual_blocks=getattr(conf, 'postnet_blocks', 20),
#            time_emb_dim=128,
#        ).to(self.device)
#
#        # Paper: Adam with lr=0.001
#        self.optimizer = torch.optim.Adam(
#            self.postnet.parameters(),
#            lr=getattr(conf, 'postnet_lr', 0.001)
#        )
#
#        self.current_iter = 0
#        self.current_epoch = 0
#
#        logger.info(
#            f"PostNet initialized with "
#            f"{sum(p.numel() for p in self.postnet.parameters()):,} parameters"
#        )
#
#    # =========================================================================
#    # Diffusion Schedule
#    # =========================================================================
#
#    def _setup_diffusion_schedule(self):
#        """
#        Setup linear diffusion schedule.
#
#        Paper Section 4.2:
#        "linear noise schedule ranging from 1x10^-4 to 0.05"
#        """
#        self.betas = torch.linspace(
#            self.beta_start, self.beta_end, self.T, device=self.device
#        )
#        self.alphas = 1.0 - self.betas
#        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
#        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
#        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
#
#        alphas_cumprod_prev = torch.cat([
#            torch.ones(1, device=self.device),
#            self.alphas_cumprod[:-1]
#        ])
#        self.posterior_variance = (
#            self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
#        )
#
#    # =========================================================================
#    # Generator Utilities
#    # =========================================================================
#
#    def generate_xrecon_batch(self, mel_batch: torch.Tensor) -> torch.Tensor:
#        """
#        Generate reconstructed mels using trained generator.
#
#        Paper Figure 1:
#        "The scaled mel spectrogram is inversely scaled by r^-1.
#         The reconstructed output serves as the conditioning."
#
#        Args:
#            mel_batch: (B, 1, F, T)
#        Returns:
#            xrecon: (B, F, T)
#        """
#        with torch.no_grad():
#            B, C, F, T = mel_batch.shape
#
#            from scaler_gan.scalergan_utils.scalergan_utils import random_size
#            output_size, random_affine = random_size(
#                orig_size=(F, T),
#                curriculum=self.conf.curriculum,
#                i=self.current_iter,
#                epoch_for_max_range=self.conf.epoch_for_max_range,
#                must_divide=self.conf.must_divide,
#                min_scale=self.conf.min_scale,
#                max_scale=self.conf.max_scale,
#                max_transform_magnitude=self.conf.max_transform_magnitude,
#            )
#
#            xr = self.generator_trainer.G(
#                mel_batch,
#                output_size=output_size,
#                random_affine=random_affine,
#                ppg=None,
#            )
#
#            xrecon = self.generator_trainer.G(
#                xr,
#                output_size=(F, T),
#                random_affine=-random_affine if random_affine is not None else None,
#                ppg=None,
#            )
#
#            xrecon = xrecon.squeeze(1)  # (B, F, T)
#
#        return xrecon
#
#    # =========================================================================
#    # Training
#    # =========================================================================
#
#    def train_one_step(self, mel_batch: torch.Tensor) -> float:
#        """
#        Single training step (Paper Algorithm 1).
#
#        Algorithm 1:
#        1. Sample (x, xrecon) from training set
#        2. ε ~ N(0, I)
#        3. Sample t ~ Uniform({1,...,T})
#        4. Gradient descent on ||ε - εθ(√ᾱt·x + √(1-ᾱt)·ε, t, x, xrecon)||²
#
#        Args:
#            mel_batch: (B, 1, F, T)
#        Returns:
#            loss value
#        """
#        self.optimizer.zero_grad()
#
#        B, C, F, T = mel_batch.shape
#
#        xrecon = self.generate_xrecon_batch(mel_batch)  # (B, F, T)
#        x = mel_batch.squeeze(1)                         # (B, F, T)
#
#        epsilon = torch.randn_like(x)
#        t = torch.randint(0, self.T, (B,), device=self.device).long()
#
#        sqrt_alpha_t = self.sqrt_alphas_cumprod[t].view(B, 1, 1)
#        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1)
#
#        xt = sqrt_alpha_t * x + sqrt_one_minus_alpha_t * epsilon
#
#        epsilon_pred = self.postnet(xt, t, xrecon)
#
#        loss = nn.functional.mse_loss(epsilon_pred, epsilon)
#        loss.backward()
#        self.optimizer.step()
#
#        return loss.item()
#
#    def train_one_epoch(self, train_loader) -> float:
#        """
#        Train PostNet for one epoch.
#        Saves checkpoint once per epoch. Visualizes every 2 epochs.
#
#        Args:
#            train_loader: DataLoader for training data
#        Returns:
#            avg_loss
#        """
#        self.postnet.train()
#        epoch_loss = 0.0
#        num_batches = 0
#
#        pbar = tqdm(
#            train_loader,
#            desc=f"Epoch {self.current_epoch + 1} | Iter {self.current_iter}",
#            ncols=110
#        )
#
#        for batch in pbar:
#            if isinstance(batch, (list, tuple)):
#                mel_batch = batch[0].to(self.device)
#            else:
#                mel_batch = batch.to(self.device)
#
#            loss = self.train_one_step(mel_batch)
#
#            epoch_loss += loss
#            num_batches += 1
#            self.current_iter += 1
#
#            pbar.set_postfix({
#                'loss': f'{loss:.4f}',
#                'iter': self.current_iter
#            })
#
#            if self.current_iter >= self.conf.postnet_iterations:
#                break
#
#        avg_loss = epoch_loss / max(num_batches, 1)
#        self.current_epoch += 1
#
#        # Log scalars to wandb
#        if self.conf.wandb:
#            wandb.log({
#                'postnet/loss': avg_loss,
#                'postnet/epoch': self.current_epoch,
#                'postnet/iter': self.current_iter,
#            }, step=self.current_iter)
#
#        # Save checkpoint once per epoch
#        self.save_checkpoint()
#
#        logger.info(
#            f"Epoch {self.current_epoch} | "
#            f"Iter {self.current_iter:,} | "
#            f"Avg Loss: {avg_loss:.6f}"
#        )
#
#        # Visualize every 2 epochs
#        if self.current_epoch % 2 == 0 and self.conf.wandb:
#            sample_batch = next(iter(train_loader))
#            if isinstance(sample_batch, (list, tuple)):
#                sample_mel = sample_batch[0][:1].to(self.device)
#            else:
#                sample_mel = sample_batch[:1].to(self.device)
#            self.visualize(sample_mel)
#            logger.info(f"📊 Logged mel visualizations for epoch {self.current_epoch}")
#
#        return avg_loss
#
#    # =========================================================================
#    # Visualization
#    # =========================================================================
#
#    def _mel_to_figure(self, mels: dict) -> plt.Figure:
#        """
#        Plot multiple mel spectrograms side by side.
#
#        Args:
#            mels: dict of {title: (F, T) tensor}
#        Returns:
#            matplotlib Figure
#        """
#        n = len(mels)
#        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
#        if n == 1:
#            axes = [axes]
#        for ax, (title, mel) in zip(axes, mels.items()):
#            mel_np = mel.detach().cpu().float().numpy()
#            im = ax.imshow(
#                mel_np, aspect='auto', origin='lower',
#                interpolation='none', vmin=-1, vmax=1, cmap='magma'
#            )
#            ax.set_title(title, fontsize=9)
#            ax.set_xlabel('Time')
#            ax.set_ylabel('Mel bins')
#            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
#        plt.tight_layout()
#        return fig
#
#    @torch.no_grad()
#    def _run_reverse_diffusion(self, xr: torch.Tensor) -> torch.Tensor:
#        """
#        Full reverse diffusion (Algorithm 2) for visualization.
#
#        Algorithm 2: condition on xr (scaled mel) directly —
#        same shape as xt throughout, no shape mismatch.
#
#        Args:
#            xr: scaled mel (B, F, T_scaled) — starting noise size AND condition
#        Returns:
#            x0: denoised mel (B, F, T_scaled)
#        """
#        B, F, T = xr.shape
#        xt = torch.randn_like(xr)  # yT ~ N(0, I)
#
#        for t_idx in reversed(range(self.T)):
#            t_tensor = torch.full((B,), t_idx, device=self.device, dtype=torch.long)
#
#            # Condition on xr — same shape as xt, no mismatch
#            eps_pred = self.postnet(xt, t_tensor, xr)
#
#            alpha_t = self.alphas[t_idx]
#            sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t_idx]
#
#            xt = (1.0 / torch.sqrt(alpha_t)) * (
#                xt - ((1 - alpha_t) / sqrt_one_minus) * eps_pred
#            )
#
#            if t_idx > 0:
#                z = torch.randn_like(xt)
#                sigma_t = torch.sqrt(self.posterior_variance[t_idx])
#                xt = xt + sigma_t * z
#
#        return xt
#
#    @torch.no_grad()
#    def visualize(self, mel_batch: torch.Tensor):
#        """
#        Log mel visualizations to wandb every 2 epochs.
#
#        Panels:
#          - Original x
#          - xrecon (generator reconstruction — conditioning during training)
#          - xr (scaled mel — conditioning during inference)
#          - x_t at t=50 (mid-noise)
#          - x_T (pure noise)
#          - ε actual vs ε predicted at t=50
#          - Denoised x₀ (full reverse diffusion from xr)
#
#        Args:
#            mel_batch: (1, 1, F, T) single sample
#        """
#        self.postnet.eval()
#
#        x = mel_batch[:1]       # (1, 1, F, T)
#        B, C, F, T = x.shape
#
#        # Generate xr and xrecon
#        from scaler_gan.scalergan_utils.scalergan_utils import random_size
#        output_size, random_affine = random_size(
#            orig_size=(F, T),
#            curriculum=self.conf.curriculum,
#            i=self.current_iter,
#            epoch_for_max_range=self.conf.epoch_for_max_range,
#            must_divide=self.conf.must_divide,
#            min_scale=self.conf.min_scale,
#            max_scale=self.conf.max_scale,
#            max_transform_magnitude=self.conf.max_transform_magnitude,
#        )
#
#        xr = self.generator_trainer.G(
#            x, output_size=output_size,
#            random_affine=random_affine, ppg=None
#        )  # (1, 1, F, T_scaled)
#
#        xrecon = self.generator_trainer.G(
#            xr, output_size=(F, T),
#            random_affine=-random_affine, ppg=None
#        )  # (1, 1, F, T)
#
#        x_2d      = x.squeeze(1).squeeze(0)       # (F, T)
#        xrecon_2d = xrecon.squeeze(1).squeeze(0)  # (F, T)
#        xr_2d     = xr.squeeze(1).squeeze(0)      # (F, T_scaled)
#        eps       = torch.randn_like(x.squeeze(1))  # (1, F, T)
#
#        # x_t at t=50
#        x_t50 = (
#            self.sqrt_alphas_cumprod[50] * x.squeeze(1)
#            + self.sqrt_one_minus_alphas_cumprod[50] * eps
#        ).squeeze(0)  # (F, T)
#
#        # x_T — fully noised
#        x_T = (
#            self.sqrt_alphas_cumprod[-1] * x.squeeze(1)
#            + self.sqrt_one_minus_alphas_cumprod[-1] * eps
#        ).squeeze(0)  # (F, T)
#
#        # ε predicted at t=50 — conditioned on xrecon (training conditioning)
#        t50 = torch.tensor([50], device=self.device, dtype=torch.long)
#        xt50_input = (
#            self.sqrt_alphas_cumprod[50] * x.squeeze(1)
#            + self.sqrt_one_minus_alphas_cumprod[50] * eps
#        )
#        eps_pred = self.postnet(
#            xt50_input, t50, xrecon.squeeze(1)
#        ).squeeze(0)  # (F, T)
#
#        # Full denoised output — conditioned on xr (inference mode, Algorithm 2)
#        x0_denoised = self._run_reverse_diffusion(
#            xr.squeeze(1)   # (1, F, T_scaled)
#        ).squeeze(0)        # (F, T_scaled)
#
#        fig = self._mel_to_figure({
#            'Original x':       x_2d,
#            'xrecon (cond)':    xrecon_2d,
#            'xr (scaled)':      xr_2d,
#            'x_t (t=50)':       x_t50,
#            'x_T (pure noise)': x_T,
#            'ε actual':         eps.squeeze(0),
#            'ε predicted':      eps_pred,
#            'Denoised x₀':      x0_denoised,
#        })
#
#        wandb.log({
#            'postnet/mel_panels': wandb.Image(fig),
#            'postnet/epoch': self.current_epoch,
#        }, step=self.current_iter)
#
#        plt.close(fig)
#        self.postnet.train()
#
#    # =========================================================================
#    # Checkpointing
#    # =========================================================================
#
#    def save_checkpoint(self):
#        """Save PostNet checkpoint (called once per epoch)."""
#        checkpoint_dir = os.path.join(self.conf.artifacts_dir, 'postnet_checkpoints')
#        os.makedirs(checkpoint_dir, exist_ok=True)
#
#        checkpoint_path = os.path.join(
#            checkpoint_dir,
#            f'postnet_epoch{self.current_epoch:04d}_iter{self.current_iter:06d}.pth.tar'
#        )
#
#        torch.save({
#            'postnet': self.postnet.state_dict(),
#            'optimizer': self.optimizer.state_dict(),
#            'iteration': self.current_iter,
#            'epoch': self.current_epoch,
#            'betas': self.betas,
#            'alphas': self.alphas,
#            'alphas_cumprod': self.alphas_cumprod,
#        }, checkpoint_path)
#
#        logger.info(
#            f"💾 Checkpoint saved: epoch {self.current_epoch}, "
#            f"iter {self.current_iter:,} → {checkpoint_path}"
#        )
#
#    def load_checkpoint(self, checkpoint_path: str):
#        """Load PostNet checkpoint."""
#        checkpoint = torch.load(checkpoint_path, map_location=self.device)
#
#        self.postnet.load_state_dict(checkpoint['postnet'])
#        self.optimizer.load_state_dict(checkpoint['optimizer'])
#        self.current_iter = checkpoint['iteration']
#        self.current_epoch = checkpoint.get('epoch', 0)
#
#        logger.info(
#            f"✅ Loaded PostNet checkpoint: "
#            f"epoch {self.current_epoch}, iter {self.current_iter:,}"
#        )







#
#
#import os
#import torch
#import torch.nn as nn
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
#from argparse import Namespace
#from tqdm import tqdm
#import wandb
#from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
#
#from scaler_gan.network_topology.networks import DiffusionPostNet
#from scaler_gan.trainer.scalerGAN import ScalerGANTrainer
#from scaler_gan.scalergan_utils.global_logger import logger
#
#
#class PostNetTrainer:
#    """
#    Trainer for DiffATSM's Diffusion-based PostNet.
#
#    DiffATSM Paper Section 3.2 & Algorithm 1:
#    Trains the PostNet to predict noise in the diffusion process,
#    conditioned on the reconstructed mel from the adaptive generator.
#    """
#
#    def __init__(self, conf: Namespace, generator_checkpoint: str):
#        self.conf = conf
#        self.device = conf.device
#
#        # Paper Section 4.2: Diffusion hyperparameters
#        self.T = 100
#        self.beta_start = 1e-4
#        self.beta_end = 0.05
#
#        self._setup_diffusion_schedule()
#
#        # Load frozen adaptive generator
#        logger.info("Loading trained adaptive generator...")
#
#        conf.checkpoint_path = generator_checkpoint
#        conf.use_ppg = True
#        conf.ppg_input_dim = getattr(conf, 'ppg_input_dim', 768)
#        conf.ppg_hidden_dim = getattr(conf, 'ppg_hidden_dim', 256)
#
#        self.generator_trainer = ScalerGANTrainer(conf, inference=True)
#        self.generator_trainer.G.eval()
#
#        for param in self.generator_trainer.G.parameters():
#            param.requires_grad = False
#
#        logger.info("✅ Generator loaded and frozen")
#
#        # Initialize PostNet
#        self.postnet = DiffusionPostNet(
#            mel_bins=conf.mel_params['num_mels'],
#            residual_channels=getattr(conf, 'postnet_channels', 256),
#            n_residual_blocks=getattr(conf, 'postnet_blocks', 20),
#            time_emb_dim=128,
#        ).to(self.device)
#
#        # FIX 1: Lower LR — 1e-3 caused divergence.
#        # DiffWave/WaveGrad use 2e-4; diffusion models are LR-sensitive.
#        self.optimizer = torch.optim.Adam(
#            self.postnet.parameters(),
#            lr=2e-4
#        )
#
#        # FIX 2: Warmup (0 → 2e-4 over 1000 steps) + cosine decay.
#        # Warmup prevents early gradient explosion before weights stabilize.
#        # Scheduler is stepped every iteration in train_one_step.
#        warmup = LinearLR(
#            self.optimizer,
#            start_factor=1e-6 / 2e-4,
#            end_factor=1.0,
#            total_iters=1000
#        )
#        cosine = CosineAnnealingLR(
#            self.optimizer,
#            T_max=max(conf.postnet_iterations - 1000, 1),
#            eta_min=1e-6
#        )
#        self.scheduler = SequentialLR(
#            self.optimizer,
#            schedulers=[warmup, cosine],
#            milestones=[1000]
#        )
#
#        # Fixed curriculum iter for generate_xrecon_batch — see method docstring
#        iters_per_epoch = getattr(conf, 'iters_per_epoch', 1637)
#        self.fixed_curriculum_iter = int(conf.epoch_for_max_range * iters_per_epoch)
#
#        self.current_iter = 0
#        self.current_epoch = 0
#
#        logger.info(
#            f"PostNet initialized with "
#            f"{sum(p.numel() for p in self.postnet.parameters()):,} parameters"
#        )
#        logger.info(
#            f"Fixed curriculum iter: {self.fixed_curriculum_iter} "
#            f"(epoch_for_max_range={conf.epoch_for_max_range}, "
#            f"iters_per_epoch={iters_per_epoch})"
#        )
#
#    # =========================================================================
#    # Diffusion Schedule
#    # =========================================================================
#
#    def _setup_diffusion_schedule(self):
#        """
#        Setup linear diffusion schedule.
#        Paper Section 4.2: "linear noise schedule ranging from 1x10^-4 to 0.05"
#        """
#        self.betas = torch.linspace(
#            self.beta_start, self.beta_end, self.T, device=self.device
#        )
#        self.alphas = 1.0 - self.betas
#        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
#        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
#        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
#
#        alphas_cumprod_prev = torch.cat([
#            torch.ones(1, device=self.device),
#            self.alphas_cumprod[:-1]
#        ])
#        self.posterior_variance = (
#            self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
#        )
#
#    # =========================================================================
#    # Normalization
#    # =========================================================================
#
#    def normalize_mel(self, mel: torch.Tensor) -> torch.Tensor:
#        """
#        Per-sample min-max normalization to [-1, 1].
#
#        Paper Section 4.1:
#        "min-max normalization to scale the data range to [-1, 1]"
#
#        Works for any shape: (B, 1, F, T), (B, F, T), etc.
#        Normalizes each sample in the batch independently.
#
#        Args:
#            mel: (...) tensor with batch on dim=0
#        Returns:
#            normalized mel, same shape, values in [-1, 1]
#        """
#        B = mel.shape[0]
#        flat = mel.view(B, -1)
#        mel_min = flat.min(dim=1)[0].view(B, *([1] * (mel.dim() - 1)))
#        mel_max = flat.max(dim=1)[0].view(B, *([1] * (mel.dim() - 1)))
#        return 2.0 * (mel - mel_min) / (mel_max - mel_min + 1e-8) - 1.0
#
#    # =========================================================================
#    # Generator Utilities
#    # =========================================================================
#
#    def generate_xrecon_batch(self, mel_batch: torch.Tensor) -> torch.Tensor:
#        """
#        Generate reconstructed mels via forward scale → inverse scale.
#
#        Paper Figure 1:
#        "The scaled mel spectrogram is inversely scaled by r^-1.
#         The reconstructed output serves as the conditioning."
#
#        Key design decisions:
#        - Caller normalizes mel_batch; this method never normalizes internally
#          to prevent double normalization.
#        - Uses self.fixed_curriculum_iter (not self.current_iter) so the
#          scale distribution stays constant throughout all of PostNet training.
#          Using current_iter causes the conditioning distribution to shift as
#          training progresses (easy scales → hard scales), which destabilizes
#          the PostNet and causes loss explosions at later epochs.
#        - xr is clamped to [-1, 1] before the inverse pass since the generator
#          was trained on normalized inputs.
#        - xrecon is normalized post-generation to correct for generator bias
#          introduced by running without PPG.
#
#        Args:
#            mel_batch: (B, 1, F, T) — MUST already be normalized to [-1, 1] by caller
#        Returns:
#            xrecon: (B, F, T) — normalized to [-1, 1]
#        """
#        with torch.no_grad():
#            B, C, F, T = mel_batch.shape
#
#            from scaler_gan.scalergan_utils.scalergan_utils import random_size
#
#            # FIX 3: Fixed curriculum iter — conditioning distribution stays
#            # constant throughout training, preventing distribution shift explosion
#            output_size, random_affine = random_size(
#                orig_size=(F, T),
#                curriculum=self.conf.curriculum,
#                i=self.fixed_curriculum_iter,       # ← fixed, not self.current_iter
#                epoch_for_max_range=self.conf.epoch_for_max_range,
#                must_divide=self.conf.must_divide,
#                min_scale=self.conf.min_scale,
#                max_scale=self.conf.max_scale,
#                max_transform_magnitude=self.conf.max_transform_magnitude,
#            )
#
#            # Forward scale: x → xr
#            xr = self.generator_trainer.G(
#                mel_batch,
#                output_size=output_size,
#                random_affine=random_affine,
#                ppg=None,
#            )
#
#            # Clamp before inverse pass — generator expects [-1, 1] input
#            xr_clamped = xr.clamp(-1.0, 1.0)
#
#            # Inverse scale: xr → xrecon
#            xrecon = self.generator_trainer.G(
#                xr_clamped,
#                output_size=(F, T),
#                random_affine=-random_affine if random_affine is not None else None,
#                ppg=None,
#            )
#
#            xrecon = xrecon.squeeze(1)          # (B, F, T)
#
#            # Normalize: corrects generator bias from running without PPG
#            xrecon = self.normalize_mel(xrecon)
#
#        return xrecon
#
#    # =========================================================================
#    # Training
#    # =========================================================================
#
#    def train_one_step(self, mel_batch: torch.Tensor) -> float:
#        """
#        Single training step (Paper Algorithm 1).
#
#        Algorithm 1:
#        1. Sample (x, xrecon) from training set
#        2. ε ~ N(0, I)
#        3. Sample t ~ Uniform({1,...,T})
#        4. Gradient descent on ||ε - εθ(√ᾱt·x + √(1-ᾱt)·ε, t, x, xrecon)||²
#
#        Args:
#            mel_batch: (B, 1, F, T) — raw un-normalized mel
#        Returns:
#            loss value
#        """
#        self.optimizer.zero_grad()
#
#        # Normalize once here — generate_xrecon_batch expects normalized input
#        mel_batch = self.normalize_mel(mel_batch)       # (B, 1, F, T)
#
#        B, C, F, T = mel_batch.shape
#
#        xrecon = self.generate_xrecon_batch(mel_batch)  # (B, F, T)
#        x = mel_batch.squeeze(1)                         # (B, F, T)
#
#        epsilon = torch.randn_like(x)
#        t = torch.randint(0, self.T, (B,), device=self.device).long()
#
#        sqrt_alpha_t = self.sqrt_alphas_cumprod[t].view(B, 1, 1)
#        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1)
#
#        xt = sqrt_alpha_t * x + sqrt_one_minus_alpha_t * epsilon
#
#        epsilon_pred = self.postnet(xt, t, xrecon)
#
#        loss = nn.functional.mse_loss(epsilon_pred, epsilon)
#        loss.backward()
#
#        # FIX 4: Tighter gradient clipping — 0.5 instead of 1.0 to suppress
#        # gradient spikes when conditioning occasionally hits hard scale pairs
#        torch.nn.utils.clip_grad_norm_(self.postnet.parameters(), max_norm=0.5)
#
#        self.optimizer.step()
#
#        # FIX 5: Step scheduler every iteration (not every epoch) — finer
#        # LR control prevents the LR from staying high during high-loss phases
#        self.scheduler.step()
#
#        return loss.item()
#
#    def train_one_epoch(self, train_loader) -> float:
#        """
#        Train PostNet for one epoch.
#        Saves checkpoint once per epoch. Visualizes every 2 epochs.
#        """
#        self.postnet.train()
#        epoch_loss = 0.0
#        num_batches = 0
#
#        pbar = tqdm(
#            train_loader,
#            desc=f"Epoch {self.current_epoch + 1} | Iter {self.current_iter}",
#            ncols=110
#        )
#
#        for batch in pbar:
#            if isinstance(batch, (list, tuple)):
#                mel_batch = batch[0].to(self.device)
#            else:
#                mel_batch = batch.to(self.device)
#
#            loss = self.train_one_step(mel_batch)
#
#            epoch_loss += loss
#            num_batches += 1
#            self.current_iter += 1
#
#            pbar.set_postfix({
#                'loss': f'{loss:.4f}',
#                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}',
#                'iter': self.current_iter
#            })
#
#            if self.current_iter >= self.conf.postnet_iterations:
#                break
#
#        avg_loss = epoch_loss / max(num_batches, 1)
#        self.current_epoch += 1
#
#        if self.conf.wandb:
#            wandb.log({
#                'postnet/loss': avg_loss,
#                'postnet/lr': self.scheduler.get_last_lr()[0],
#                'postnet/epoch': self.current_epoch,
#                'postnet/iter': self.current_iter,
#            }, step=self.current_iter)
#
#        # Save checkpoint once per epoch
#        self.save_checkpoint()
#
#        logger.info(
#            f"Epoch {self.current_epoch} | "
#            f"Iter {self.current_iter:,} | "
#            f"Avg Loss: {avg_loss:.6f} | "
#            f"LR: {self.scheduler.get_last_lr()[0]:.2e}"
#        )
#
#        # Visualize every 2 epochs — wrapped in try/except so viz bugs
#        # never interrupt training
#        if self.current_epoch % 2 == 0 and self.conf.wandb:
#            try:
#                sample_batch = next(iter(train_loader))
#                if isinstance(sample_batch, (list, tuple)):
#                    sample_mel = sample_batch[0][:1].to(self.device)
#                else:
#                    sample_mel = sample_batch[:1].to(self.device)
#                self.visualize(sample_mel)
#                logger.info(f"📊 Logged mel visualizations for epoch {self.current_epoch}")
#            except Exception as e:
#                logger.warning(f"⚠️ Visualization failed (non-fatal): {e}")
#
#        return avg_loss
#
#    # =========================================================================
#    # Visualization
#    # =========================================================================
#
#    def _mel_to_figure(self, mels: dict) -> plt.Figure:
#        """
#        Plot multiple mel spectrograms side by side.
#        Uses adaptive vmin/vmax per panel for correct contrast.
#
#        Args:
#            mels: dict of {title: (F, T) tensor}
#        Returns:
#            matplotlib Figure
#        """
#        n = len(mels)
#        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
#        if n == 1:
#            axes = [axes]
#        for ax, (title, mel) in zip(axes, mels.items()):
#            mel_np = mel.detach().cpu().float().numpy()
#            vmin, vmax = mel_np.min(), mel_np.max()
#            im = ax.imshow(
#                mel_np, aspect='auto', origin='lower',
#                interpolation='none', vmin=vmin, vmax=vmax, cmap='magma'
#            )
#            ax.set_title(f"{title}\n[{vmin:.2f}, {vmax:.2f}]", fontsize=9)
#            ax.set_xlabel('Time')
#            ax.set_ylabel('Mel bins')
#            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
#        plt.tight_layout()
#        return fig
#
#    @torch.no_grad()
#    def _run_reverse_diffusion(self, xr: torch.Tensor) -> torch.Tensor:
#        """
#        Full reverse diffusion (Algorithm 2) for visualization.
#
#        Args:
#            xr: (B, F, T_scaled) — normalized to [-1, 1]
#        Returns:
#            x0: (B, F, T_scaled) — denoised mel
#        """
#        B, F, T = xr.shape
#        xt = torch.randn_like(xr)   # yT ~ N(0, I)
#
#        for t_idx in reversed(range(self.T)):
#            t_tensor = torch.full((B,), t_idx, device=self.device, dtype=torch.long)
#
#            eps_pred = self.postnet(xt, t_tensor, xr)
#
#            alpha_t = self.alphas[t_idx]
#            sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t_idx]
#
#            xt = (1.0 / torch.sqrt(alpha_t)) * (
#                xt - ((1 - alpha_t) / sqrt_one_minus) * eps_pred
#            )
#
#            if t_idx > 0:
#                z = torch.randn_like(xt)
#                sigma_t = torch.sqrt(self.posterior_variance[t_idx])
#                xt = xt + sigma_t * z
#
#        return xt
#
#    @torch.no_grad()
#    def visualize(self, mel_batch: torch.Tensor):
#        """
#        Log mel visualizations to wandb every 2 epochs.
#
#        Panels:
#          - Original x (normalized)
#          - xrecon (generator reconstruction — training conditioning, normalized)
#          - xr (scaled mel — inference conditioning)
#          - x_t at t=50 (mid-noise)
#          - x_T (pure noise)
#          - ε actual vs ε predicted at t=50
#          - Denoised x₀ (full reverse diffusion)
#
#        Args:
#            mel_batch: (1, 1, F, T) — raw un-normalized mel
#        """
#        self.postnet.eval()
#
#        # Normalize x — single source of truth for normalization
#        x = self.normalize_mel(mel_batch[:1])       # (1, 1, F, T)
#        B, C, F, T = x.shape
#
#        # generate_xrecon_batch receives normalized x — returns normalized xrecon
#        xrecon_3d = self.generate_xrecon_batch(x)   # (1, F, T)
#        xrecon = xrecon_3d.unsqueeze(1)              # (1, 1, F, T)
#
#        # Get xr for inference visualization
#        from scaler_gan.scalergan_utils.scalergan_utils import random_size
#        output_size, random_affine = random_size(
#            orig_size=(F, T),
#            curriculum=self.conf.curriculum,
#            i=self.fixed_curriculum_iter,
#            epoch_for_max_range=self.conf.epoch_for_max_range,
#            must_divide=self.conf.must_divide,
#            min_scale=self.conf.min_scale,
#            max_scale=self.conf.max_scale,
#            max_transform_magnitude=self.conf.max_transform_magnitude,
#        )
#
#        xr = self.generator_trainer.G(
#            x, output_size=output_size,
#            random_affine=random_affine, ppg=None
#        )  # (1, 1, F, T_scaled)
#
#        x_2d      = x.squeeze(1).squeeze(0)        # (F, T)
#        xrecon_2d = xrecon_3d.squeeze(0)            # (F, T)
#        xr_2d     = xr.squeeze(1).squeeze(0)        # (F, T_scaled)
#
#        # Debug stats
#        print(f"[VIZ] x      : min={x.min():.3f}  max={x.max():.3f}  mean={x.mean():.3f}")
#        print(f"[VIZ] xr     : min={xr.min():.3f}  max={xr.max():.3f}  mean={xr.mean():.3f}")
#        print(f"[VIZ] xrecon : min={xrecon.min():.3f}  max={xrecon.max():.3f}  mean={xrecon.mean():.3f}")
#
#        eps = torch.randn_like(x.squeeze(1))        # (1, F, T)
#
#        # x_t at t=50
#        x_t50 = (
#            self.sqrt_alphas_cumprod[50] * x.squeeze(1)
#            + self.sqrt_one_minus_alphas_cumprod[50] * eps
#        ).squeeze(0)  # (F, T)
#
#        # x_T — fully noised
#        x_T = (
#            self.sqrt_alphas_cumprod[-1] * x.squeeze(1)
#            + self.sqrt_one_minus_alphas_cumprod[-1] * eps
#        ).squeeze(0)  # (F, T)
#
#        # ε predicted at t=50 — conditioned on xrecon
#        t50 = torch.tensor([50], device=self.device, dtype=torch.long)
#        xt50_input = (
#            self.sqrt_alphas_cumprod[50] * x.squeeze(1)
#            + self.sqrt_one_minus_alphas_cumprod[50] * eps
#        )
#        eps_pred = self.postnet(
#            xt50_input, t50, xrecon.squeeze(1)
#        ).squeeze(0)  # (F, T)
#
#        # Full denoised output — conditioned on xr (Algorithm 2, inference mode)
#        x0_denoised = self._run_reverse_diffusion(
#            xr.squeeze(1)   # (1, F, T_scaled)
#        ).squeeze(0)        # (F, T_scaled)
#
#        fig = self._mel_to_figure({
#            'Original x':       x_2d,
#            'xrecon (cond)':    xrecon_2d,
#            'xr (scaled)':      xr_2d,
#            'x_t (t=50)':       x_t50,
#            'x_T (pure noise)': x_T,
#            'ε actual':         eps.squeeze(0),
#            'ε predicted':      eps_pred,
#            'Denoised x₀':      x0_denoised,
#        })
#
#        wandb.log({
#            'postnet/mel_panels': wandb.Image(fig),
#            'postnet/epoch': self.current_epoch,
#        }, step=self.current_iter)
#
#        plt.close(fig)
#        self.postnet.train()
#
#    # =========================================================================
#    # Checkpointing
#    # =========================================================================
#
#    def save_checkpoint(self):
#        """Save PostNet checkpoint (called once per epoch)."""
#        checkpoint_dir = os.path.join(self.conf.artifacts_dir, 'postnet_checkpoints')
#        os.makedirs(checkpoint_dir, exist_ok=True)
#
#        checkpoint_path = os.path.join(
#            checkpoint_dir,
#            f'postnet_epoch{self.current_epoch:04d}_iter{self.current_iter:06d}.pth.tar'
#        )
#
#        torch.save({
#            'postnet': self.postnet.state_dict(),
#            'optimizer': self.optimizer.state_dict(),
#            'scheduler': self.scheduler.state_dict(),
#            'iteration': self.current_iter,
#            'epoch': self.current_epoch,
#            'betas': self.betas,
#            'alphas': self.alphas,
#            'alphas_cumprod': self.alphas_cumprod,
#        }, checkpoint_path)
#
#        logger.info(
#            f"💾 Checkpoint saved: epoch {self.current_epoch}, "
#            f"iter {self.current_iter:,} → {checkpoint_path}"
#        )
#
#    def load_checkpoint(self, checkpoint_path: str):
#        """Load PostNet checkpoint."""
#        checkpoint = torch.load(checkpoint_path, map_location=self.device)
#
#        self.postnet.load_state_dict(checkpoint['postnet'])
#        self.optimizer.load_state_dict(checkpoint['optimizer'])
#        if 'scheduler' in checkpoint:
#            self.scheduler.load_state_dict(checkpoint['scheduler'])
#        self.current_iter = checkpoint['iteration']
#        self.current_epoch = checkpoint.get('epoch', 0)
#
#        logger.info(
#            f"✅ Loaded PostNet checkpoint: "
#            f"epoch {self.current_epoch}, iter {self.current_iter:,}"
#        )
#



























import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from argparse import Namespace
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

try:
    from torch_ema import ExponentialMovingAverage
    HAS_EMA = True
except ImportError:
    HAS_EMA = False
    print("⚠️  torch-ema not installed. Install with: pip install torch-ema")

try:
    import wandb
except ImportError:
    wandb = None

from scaler_gan.network_topology.networks import DiffusionPostNet
from scaler_gan.trainer.scalerGAN import ScalerGANTrainer
from scaler_gan.scalergan_utils.global_logger import logger


class PostNetTrainer:
    """
    Trainer for DiffATSM's Diffusion-based PostNet.
    
    DiffATSM Paper Section 3.2 & Algorithm 1:
    Trains the PostNet to predict noise in the diffusion process,
    conditioned on the reconstructed mel from the adaptive generator.
    
    CRITICAL: This implementation matches Figure 1 exactly by using PPG
    in both forward and inverse generator passes.
    """

    def __init__(self, conf: Namespace, generator_checkpoint: str):
        self.conf = conf
        self.device = conf.device

        # Paper Section 4.2: Diffusion hyperparameters
        self.T = 100
        self.beta_start = 1e-4
        self.beta_end = 0.05

        self._setup_diffusion_schedule()

        # Load frozen adaptive generator
        logger.info("Loading trained adaptive generator...")

        conf.checkpoint_path = generator_checkpoint
        conf.use_ppg = True  # ✅ Generator was trained with PPG
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

        # Optimizer with lower LR
        self.optimizer = torch.optim.Adam(
            self.postnet.parameters(),
            lr=2e-4
        )

        # LR scheduler with warmup
        warmup = LinearLR(
            self.optimizer,
            start_factor=1e-6 / 2e-4,
            end_factor=1.0,
            total_iters=1000
        )
        cosine = CosineAnnealingLR(
            self.optimizer,
            T_max=max(conf.postnet_iterations - 1000, 1),
            eta_min=1e-6
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup, cosine],
            milestones=[1000]
        )

        # EMA for stable inference
        if HAS_EMA:
            self.ema = ExponentialMovingAverage(
                self.postnet.parameters(),
                decay=0.9999
            )
            logger.info("✅ EMA initialized with decay=0.9999")
        else:
            self.ema = None
            logger.warning("⚠️  EMA not available (torch-ema not installed)")

        # Fixed curriculum iter
        iters_per_epoch = getattr(conf, 'iters_per_epoch', 1637)
        self.fixed_curriculum_iter = int(conf.epoch_for_max_range * iters_per_epoch)

        self.current_iter = 0
        self.current_epoch = 0

        logger.info(
            f"PostNet initialized with "
            f"{sum(p.numel() for p in self.postnet.parameters()):,} parameters"
        )
        logger.info(
            f"Fixed curriculum iter: {self.fixed_curriculum_iter}"
        )

    # =========================================================================
    # Diffusion Schedule (Cosine - Better than Linear)
    # =========================================================================

    def _setup_diffusion_schedule(self):
        """
        Cosine noise schedule (improved over linear for mel spectrograms).
        
        Reference: "Improved Denoising Diffusion Probabilistic Models"
                   (Nichol & Dhariwal, 2021)
        """
        def cosine_beta_schedule(timesteps, s=0.008):
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0.0001, 0.9999)
        
        self.betas = cosine_beta_schedule(self.T).to(self.device)
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
    # Normalization
    # =========================================================================

    def normalize_mel(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Per-sample min-max normalization to [-1, 1].
        Paper Section 4.1: "min-max normalization to scale the data range to [-1, 1]"
        """
        B = mel.shape[0]
        flat = mel.view(B, -1)
        mel_min = flat.min(dim=1)[0].view(B, *([1] * (mel.dim() - 1)))
        mel_max = flat.max(dim=1)[0].view(B, *([1] * (mel.dim() - 1)))
        return 2.0 * (mel - mel_min) / (mel_max - mel_min + 1e-8) - 1.0

    # =========================================================================
    # Generator Utilities (WITH PPG - MATCHES FIGURE 1)
    # =========================================================================

    def generate_xrecon_batch_with_ppg(
        self,
        mel_batch: torch.Tensor,
        ppg: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate xrecon WITH PPG (matches Paper Figure 1 exactly).
        
        Paper Figure 1 (Training Stage):
        ┌─────────────────────────────────────────┐
        │ x + PPG → Generator(r) → xr             │
        │ xr + PPG → Generator(r⁻¹) → xrecon      │  ← PPG used in BOTH
        │ xrecon → PostNet (as conditioning)      │
        └─────────────────────────────────────────┘
        
        Args:
            mel_batch: (B, 1, F, T) - normalized mel
            ppg: (B, T, 768) - PPG features from HuBERT
        Returns:
            xrecon: (B, F, T) - reconstructed mel, normalized
        """
        with torch.no_grad():
            B, C, F, T = mel_batch.shape
            
            from scaler_gan.scalergan_utils.scalergan_utils import random_size
            
            # Get scale factor
            output_size, random_affine = random_size(
                orig_size=(F, T),
                curriculum=self.conf.curriculum,
                i=self.fixed_curriculum_iter,
                epoch_for_max_range=self.conf.epoch_for_max_range,
                must_divide=self.conf.must_divide,
                min_scale=self.conf.min_scale,
                max_scale=self.conf.max_scale,
                max_transform_magnitude=self.conf.max_transform_magnitude,
            )
            
            # ✅ Forward: x + PPG → xr
            xr = self.generator_trainer.G(
                mel_batch,
                output_size=output_size,
                random_affine=random_affine,
                ppg=ppg,  # ✅ Pass PPG (matches Figure 1)
            )
            
            xr_clamped = xr.clamp(-1.0, 1.0)
            
            # Align PPG to original size T for inverse pass
            # PPG: (B, T_in, 768) → need (B, T, 768)
            if ppg.shape[1] != T:
                ppg_aligned = F.interpolate(
                    ppg.permute(0, 2, 1).unsqueeze(1),  # (B, 1, 768, T_in)
                    size=(ppg.shape[-1], T),
                    mode='linear',
                    align_corners=False
                ).squeeze(1).permute(0, 2, 1)  # (B, T, 768)
            else:
                ppg_aligned = ppg
            
            # ✅ Inverse: xr + PPG → xrecon
            xrecon = self.generator_trainer.G(
                xr_clamped,
                output_size=(F, T),
                random_affine=-random_affine if random_affine is not None else None,
                ppg=ppg_aligned,  # ✅ Pass PPG (matches Figure 1)
            )
            
            xrecon = xrecon.squeeze(1)  # (B, F, T)
            xrecon = self.normalize_mel(xrecon)
        
        return xrecon

    # =========================================================================
    # Training
    # =========================================================================

    def train_one_step(
        self,
        mel_batch: torch.Tensor,
        ppg: torch.Tensor
    ) -> float:
        """
        Single training step (Paper Algorithm 1) WITH PPG.
        
        Algorithm 1:
        1. Sample (x, xrecon) where xrecon uses PPG ← KEY DIFFERENCE
        2. ε ~ N(0, I)
        3. Sample t ~ Uniform({1,...,T})
        4. Gradient descent on ||ε - εθ(√ᾱt·x + √(1-ᾱt)·ε, t, xrecon)||²
        
        Args:
            mel_batch: (B, 1, F, T) - raw un-normalized mel
            ppg: (B, T, 768) - PPG features from dataloader
        Returns:
            loss value
        """
        self.optimizer.zero_grad()

        # Normalize
        mel_batch = self.normalize_mel(mel_batch)
        B, C, F, T = mel_batch.shape

        # ✅ Generate xrecon WITH PPG (matches Figure 1)
        xrecon = self.generate_xrecon_batch_with_ppg(mel_batch, ppg)
        
        x = mel_batch.squeeze(1)  # (B, F, T)

        # Sample noise and time step
        epsilon = torch.randn_like(x)
        t = torch.randint(0, self.T, (B,), device=self.device).long()

        # Create noisy mel
        sqrt_alpha_t = self.sqrt_alphas_cumprod[t].view(B, 1, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1)
        xt = sqrt_alpha_t * x + sqrt_one_minus_alpha_t * epsilon

        # Predict noise
        epsilon_pred = self.postnet(xt, t, xrecon)

        # Compute loss
        loss = nn.functional.mse_loss(epsilon_pred, epsilon)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.postnet.parameters(), max_norm=0.5)

        self.optimizer.step()
        self.scheduler.step()
        
        # Update EMA
        if self.ema is not None:
            self.ema.update()

        return loss.item()

    def train_one_epoch(self, train_loader) -> float:
        """Train PostNet for one epoch."""
        self.postnet.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {self.current_epoch + 1} | Iter {self.current_iter}",
            ncols=110
        )

        for batch in pbar:
            # Unpack (mel, ppg) from dataloader
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                mel_batch = batch[0].to(self.device)
                ppg = batch[1].to(self.device)
            else:
                raise ValueError(
                    "Dataloader must return (mel, ppg) tuple. "
                    "Set use_ppg=True in MelDataset."
                )

            loss = self.train_one_step(mel_batch, ppg)

            epoch_loss += loss
            num_batches += 1
            self.current_iter += 1

            pbar.set_postfix({
                'loss': f'{loss:.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}',
                'iter': self.current_iter
            })

            if self.current_iter >= self.conf.postnet_iterations:
                break

        avg_loss = epoch_loss / max(num_batches, 1)
        self.current_epoch += 1

        if self.conf.wandb and wandb is not None:
            wandb.log({
                'postnet/loss': avg_loss,
                'postnet/lr': self.scheduler.get_last_lr()[0],
                'postnet/epoch': self.current_epoch,
                'postnet/iter': self.current_iter,
            }, step=self.current_iter)

        # Save checkpoint
        self.save_checkpoint()

        logger.info(
            f"Epoch {self.current_epoch} | "
            f"Iter {self.current_iter:,} | "
            f"Avg Loss: {avg_loss:.6f} | "
            f"LR: {self.scheduler.get_last_lr()[0]:.2e}"
        )

        # Visualize every 2 epochs
        if self.current_epoch % 2 == 0 and self.conf.wandb and wandb is not None:
            try:
                sample_batch = next(iter(train_loader))
                if isinstance(sample_batch, (list, tuple)):
                    sample_mel = sample_batch[0][:1].to(self.device)
                    sample_ppg = sample_batch[1][:1].to(self.device)
                else:
                    raise ValueError("Expected (mel, ppg) tuple")
                
                self.visualize(sample_mel, sample_ppg)
                logger.info(f"✅ Logged visualizations for epoch {self.current_epoch}")
            except Exception as e:
                logger.warning(f"⚠️  Visualization failed: {e}")

        return avg_loss

    # =========================================================================
    # Visualization
    # =========================================================================

    def _mel_to_figure(self, mels: dict) -> plt.Figure:
        """Plot multiple mel spectrograms side by side."""
        n = len(mels)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
        if n == 1:
            axes = [axes]
        for ax, (title, mel) in zip(axes, mels.items()):
            mel_np = mel.detach().cpu().float().numpy()
            vmin, vmax = mel_np.min(), mel_np.max()
            im = ax.imshow(
                mel_np, aspect='auto', origin='lower',
                interpolation='none', vmin=vmin, vmax=vmax, cmap='magma'
            )
            ax.set_title(f"{title}\n[{vmin:.2f}, {vmax:.2f}]", fontsize=9)
            ax.set_xlabel('Time')
            ax.set_ylabel('Mel bins')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        return fig

    @torch.no_grad()
    def _run_reverse_diffusion(self, xr: torch.Tensor) -> torch.Tensor:
        """
        Full reverse diffusion (Algorithm 2) for visualization.
        Uses EMA weights if available.
        """
        B, F, T = xr.shape
        xt = torch.randn_like(xr)

        # Use EMA weights for better quality
        if self.ema is not None:
            with self.ema.average_parameters():
                for t_idx in reversed(range(self.T)):
                    t_tensor = torch.full((B,), t_idx, device=self.device, dtype=torch.long)
                    eps_pred = self.postnet(xt, t_tensor, xr)
                    
                    alpha_t = self.alphas[t_idx]
                    sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t_idx]
                    
                    xt = (1.0 / torch.sqrt(alpha_t)) * (
                        xt - ((1 - alpha_t) / sqrt_one_minus) * eps_pred
                    )
                    
                    if t_idx > 0:
                        z = torch.randn_like(xt)
                        sigma_t = torch.sqrt(self.posterior_variance[t_idx])
                        xt = xt + sigma_t * z
        else:
            # Fallback without EMA
            for t_idx in reversed(range(self.T)):
                t_tensor = torch.full((B,), t_idx, device=self.device, dtype=torch.long)
                eps_pred = self.postnet(xt, t_tensor, xr)
                
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
    def visualize(self, mel_batch: torch.Tensor, ppg: torch.Tensor):
        """
        Log mel visualizations to wandb.
        
        Args:
            mel_batch: (1, 1, F, T) - raw mel
            ppg: (1, T, 768) - PPG features
        """
        self.postnet.eval()

        x = self.normalize_mel(mel_batch[:1])
        B, C, F, T = x.shape

        # Generate xrecon WITH PPG
        xrecon_3d = self.generate_xrecon_batch_with_ppg(x, ppg)
        xrecon = xrecon_3d.unsqueeze(1)

        # Get xr for visualization
        from scaler_gan.scalergan_utils.scalergan_utils import random_size
        output_size, random_affine = random_size(
            orig_size=(F, T),
            curriculum=self.conf.curriculum,
            i=self.fixed_curriculum_iter,
            epoch_for_max_range=self.conf.epoch_for_max_range,
            must_divide=self.conf.must_divide,
            min_scale=self.conf.min_scale,
            max_scale=self.conf.max_scale,
            max_transform_magnitude=self.conf.max_transform_magnitude,
        )

        xr = self.generator_trainer.G(
            x, output_size=output_size,
            random_affine=random_affine, ppg=ppg
        )

        x_2d = x.squeeze(1).squeeze(0)
        xrecon_2d = xrecon_3d.squeeze(0)
        xr_2d = xr.squeeze(1).squeeze(0)

        # Sample noise
        eps = torch.randn_like(x.squeeze(1))

        # x_t at t=50
        x_t50 = (
            self.sqrt_alphas_cumprod[50] * x.squeeze(1)
            + self.sqrt_one_minus_alphas_cumprod[50] * eps
        ).squeeze(0)

        # Full denoised output
        x0_denoised = self._run_reverse_diffusion(
            xr.squeeze(1)
        ).squeeze(0)

        fig = self._mel_to_figure({
            'Original x': x_2d,
            'xrecon (with PPG)': xrecon_2d,
            'xr (scaled)': xr_2d,
            'x_t (t=50)': x_t50,
            'Denoised x₀': x0_denoised,
        })

        if wandb is not None:
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
        """Save PostNet checkpoint."""
        checkpoint_dir = os.path.join(self.conf.artifacts_dir, 'postnet_checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_path = os.path.join(
            checkpoint_dir,
            f'postnet_epoch{self.current_epoch:04d}_iter{self.current_iter:06d}.pth.tar'
        )

        checkpoint = {
            'postnet': self.postnet.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'iteration': self.current_iter,
            'epoch': self.current_epoch,
            'betas': self.betas,
            'alphas': self.alphas,
            'alphas_cumprod': self.alphas_cumprod,
        }
        
        if self.ema is not None:
            checkpoint['ema'] = self.ema.state_dict()

        torch.save(checkpoint, checkpoint_path)

        logger.info(
            f"💾 Checkpoint saved: epoch {self.current_epoch}, "
            f"iter {self.current_iter:,} → {checkpoint_path}"
        )

    def load_checkpoint(self, checkpoint_path: str):
        """Load PostNet checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.postnet.load_state_dict(checkpoint['postnet'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        if 'ema' in checkpoint and self.ema is not None:
            self.ema.load_state_dict(checkpoint['ema'])
        self.current_iter = checkpoint['iteration']
        self.current_epoch = checkpoint.get('epoch', 0)

        logger.info(
            f"✅ Loaded PostNet checkpoint: "
            f"epoch {self.current_epoch}, iter {self.current_iter:,}"
        )
