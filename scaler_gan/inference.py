#!/usr/bin/env python3
"""
Complete DiffATSM Inference Script
Uses BOTH Adaptive Generator AND Diffusion-based PostNet


Paper Figure 1 (Inference Stage):
┌──────────────────────────────────────────────┐
│ x + PPG → Adaptive Generator(r) → xr         │
│ εT ~ N(0,I) → Diffusion PostNet → xr'        │  ← Final output
└──────────────────────────────────────────────┘


Usage:
    python -m scaler_gan.inference \
        --generator_checkpoint path/to/generator.pth.tar \
        --postnet_checkpoint path/to/postnet.pth.tar \
        --inference_file data/inference.txt \
        --infer_scales 0.5 0.7 0.9 1.1 1.3 1.5 \
        --infer_hifi \
        --device cuda
"""


import os
import sys
import json
from argparse import Namespace


import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


# Add parent directory to path
import git
wt_dir = git.Repo(".", search_parent_directories=True).working_tree_dir
hifi_abs_dir = os.path.join(wt_dir, "hifi_gan")
try:
    sys.path.index(wt_dir)
except ValueError:
    sys.path.append(wt_dir)
try:
    sys.path.index(hifi_abs_dir)
except ValueError:
    sys.path.append(hifi_abs_dir)


import hifi_gan.inference_e2e as inference_e2e
from hifi_gan.env import AttrDict


from scaler_gan.configs.configs import Config
from scaler_gan.network_topology.networks import DiffusionPostNet
from scaler_gan.trainer.scalerGAN import ScalerGANTrainer
from scaler_gan.scalergan_utils.global_logger import logger
from scaler_gan.scalergan_utils.scalergan_utils import (
    files_to_list,
    init_logger,
    load_and_norm_audio,
    create_mel_from_audio,
)


matplotlib.use("agg")


# Directory constants
MELS_NPY = "mels_npy"
MELS_IMGS_DIR = "mels_imgs"
G_PRED_DIR = "g_pred"
WAVS = "wavs"
INFERENCE_LOG = "inference.log"



class PPGExtractor:
    """
    PPG extraction for DiffATSM inference.
    Extracts HuBERT 12th-layer features as phonetic posteriorgrams.
    """

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.hubert_model = None
        self.hubert_processor = None
        self._init_hubert()

    def _init_hubert(self):
        try:
            from transformers import HubertModel, Wav2Vec2FeatureExtractor
        except ImportError:
            raise ImportError(
                "transformers library required for PPG extraction.\n"
                "Install with: pip install transformers"
            )

        logger.info("=" * 70)
        logger.info("Loading HuBERT model for PPG extraction...")
        logger.info("=" * 70)

        self.hubert_processor = Wav2Vec2FeatureExtractor.from_pretrained(
            "facebook/hubert-base-ls960"
        )
        self.hubert_model = HubertModel.from_pretrained(
            "facebook/hubert-base-ls960"
        )
        self.hubert_model.eval()

        for param in self.hubert_model.parameters():
            param.requires_grad = False

        self.hubert_model = self.hubert_model.to(self.device)
        logger.info(f"HuBERT loaded on {self.device}")

    def extract_ppg(self, audio: torch.Tensor, sampling_rate: int = 22050) -> torch.Tensor:
        if audio.dim() > 1:
            audio = audio.squeeze()

        if sampling_rate != 16000:
            import torchaudio
            resampler = torchaudio.transforms.Resample(
                orig_freq=sampling_rate,
                new_freq=16000
            )
            audio_16k = resampler(audio)
        else:
            audio_16k = audio

        inputs = self.hubert_processor(
            audio_16k.cpu().numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.hubert_model(**inputs, output_hidden_states=True)
            ppg = outputs.hidden_states[12]

        return ppg.squeeze(0).cpu()  # (T_ppg, 768)

    def align_ppg_to_mel(self, ppg: torch.Tensor, mel_length: int) -> torch.Tensor:
        if ppg.shape[0] == mel_length:
            return ppg

        ppg_aligned = F.interpolate(
            ppg.T.unsqueeze(0),
            size=mel_length,
            mode='linear',
            align_corners=False
        ).squeeze(0).T

        return ppg_aligned



class DiffATSMInference:
    """
    Complete DiffATSM inference system.
    """

    def __init__(
        self,
        conf: Namespace,
        generator_checkpoint: str,
        postnet_checkpoint: str,
    ):
        self.conf = conf
        self.device = conf.device

        logger.info("=" * 70)
        logger.info("COMPLETE DiffATSM INFERENCE")
        logger.info("=" * 70)
        logger.info(f"Device: {self.device}")

        # Load Adaptive Generator
        logger.info("\n[1/3] Loading Adaptive Generator...")
        logger.info(f"Checkpoint: {generator_checkpoint}")

        conf.checkpoint_path = generator_checkpoint
        conf.use_ppg = True
        conf.ppg_input_dim = 768
        conf.ppg_hidden_dim = 256

        self.generator_trainer = ScalerGANTrainer(conf, inference=True)
        self.generator_trainer.G.eval()
        logger.info("Generator loaded")

        # Load Diffusion PostNet
        logger.info("\n[2/3] Loading Diffusion-based PostNet...")
        logger.info(f"Checkpoint: {postnet_checkpoint}")

        postnet_ckpt = torch.load(postnet_checkpoint, map_location=self.device)

        # Auto-detect architecture from checkpoint
        state_dict = postnet_ckpt['postnet']
        n_blocks = max(
            int(k.split('.')[1])
            for k in state_dict.keys()
            if k.startswith('residual_blocks.')
        ) + 1
        residual_channels = state_dict['input_projection.0.weight'].shape[0]

        logger.info(f"Detected PostNet: {n_blocks} blocks, {residual_channels} channels")

        self.postnet = DiffusionPostNet(
            mel_bins=conf.mel_params.num_mels,
            residual_channels=residual_channels,
            n_residual_blocks=n_blocks,
            time_emb_dim=128,
        ).to(self.device)

        self.postnet.load_state_dict(state_dict)
        self.postnet.eval()

        # Load EMA if available
        if 'ema' in postnet_ckpt:
            try:
                from torch_ema import ExponentialMovingAverage
                self.ema = ExponentialMovingAverage(
                    self.postnet.parameters(),
                    decay=0.9999
                )
                self.ema.load_state_dict(postnet_ckpt['ema'])
                logger.info("PostNet loaded with EMA weights")
            except ImportError:
                self.ema = None
                logger.info("PostNet loaded (EMA not available)")
        else:
            self.ema = None
            logger.info("PostNet loaded")

        # Load diffusion schedule from checkpoint
        self.T = 100
        self.betas = postnet_ckpt.get('betas', torch.linspace(1e-4, 0.05, 100)).to(self.device)
        self.alphas = postnet_ckpt.get('alphas', 1.0 - self.betas).to(self.device)
        self.alphas_cumprod = postnet_ckpt.get('alphas_cumprod', torch.cumprod(self.alphas, dim=0)).to(self.device)

        # Precompute sampling coefficients
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        alphas_cumprod_prev = torch.cat([
            torch.ones(1, device=self.device),
            self.alphas_cumprod[:-1]
        ])
        self.posterior_variance = (
            self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_std = torch.sqrt(self.posterior_variance)

        # Initialize PPG extractor
        logger.info("\n[3/3] Initializing PPG extractor...")
        self.ppg_extractor = PPGExtractor(device=self.device)

        logger.info("\n" + "=" * 70)
        logger.info("Complete DiffATSM system ready for inference")
        logger.info("=" * 70 + "\n")

    # =========================================================================
    # Normalization — identical to PostNetTrainer.normalize_mel in training
    # =========================================================================

    def normalize_mel(self, mel: torch.Tensor) -> torch.Tensor:
        """Per-sample min-max normalization to [-1, 1]. Matches training."""
        B = mel.shape[0]
        flat = mel.view(B, -1)
        mel_min = flat.min(dim=1)[0].view(B, *([1] * (mel.dim() - 1)))
        mel_max = flat.max(dim=1)[0].view(B, *([1] * (mel.dim() - 1)))
        return 2.0 * (mel - mel_min) / (mel_max - mel_min + 1e-8) - 1.0

    # =========================================================================
    # xrecon — matches training conditioning (inverse pass + interpolate)
    # =========================================================================

    def get_xrecon(self, input_mel_norm: torch.Tensor, xr: torch.Tensor) -> torch.Tensor:
        """
        Reproduce training conditioning: xr → inverse back to original size,
        then force-interpolate to T_scaled to avoid off-by-one generator errors.
        PostNet was conditioned on xrecon during training, NOT on xr directly.
        """
        import torch.nn.functional as _F
        with torch.no_grad():
            B, C, n_mels, T = input_mel_norm.shape
            _, _, _, T_scaled = xr.shape

            # Inverse scale: xr → xrecon at original size
            xrecon = self.generator_trainer.G(
                xr.clamp(-1.0, 1.0),
                output_size=(n_mels, T),
                random_affine=None,
                ppg=None,
            )  # (1, 1, n_mels, T_approx) — may be off by 1

            # Force to T_scaled via interpolation — avoids generator rounding mismatch
            xrecon_scaled = _F.interpolate(
                xrecon,
                size=(n_mels, T_scaled),
                mode='bilinear',
                align_corners=False,
            )  # (1, 1, n_mels, T_scaled)

            xrecon_out = xrecon_scaled.squeeze(1)        # (1, n_mels, T_scaled)
            xrecon_out = self.normalize_mel(xrecon_out)  # normalize like training
        return xrecon_out

    # =========================================================================
    # Generator
    # =========================================================================

    def generate_scaled_mel(
        self,
        input_mel: torch.Tensor,
        ppg: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        with torch.no_grad():
            B, C, F, T_in = input_mel.shape

            T_out = int(T_in * scale)
            # Guard: ensure T_out is valid and divisible by must_divide
            T_out = max(T_out, self.conf.must_divide)
            T_out = (T_out // self.conf.must_divide) * self.conf.must_divide

            output_size = (F, T_out)

            ppg_aligned = self.ppg_extractor.align_ppg_to_mel(ppg, T_out)
            ppg_batch = ppg_aligned.unsqueeze(0).to(self.device)  # (1, T_out, 768)

            xr = self.generator_trainer.G(
                input_mel,
                output_size=output_size,
                random_affine=None,
                ppg=ppg_batch,
            )

            # Force exact output size in case generator is off by 1
            if xr.shape[-1] != T_out or xr.shape[-2] != F:
                xr = F.interpolate(
                    xr,
                    size=(F, T_out),
                    mode='bilinear',
                    align_corners=False,
                )

            return xr

    # =========================================================================
    # PostNet denoising
    # =========================================================================

    def denoise_with_postnet(
        self,
        xr: torch.Tensor,
        xrecon: torch.Tensor,
    ) -> torch.Tensor:
        """
        Denoise using Diffusion PostNet conditioned on xrecon (matches training).

        Args:
            xr: (1, 1, F, T_out) scaled mel — used only for shape/noise init
            xrecon: (1, F, T_out) conditioning — matches training conditioning
        Returns:
            x0: (1, 1, F, T_out) denoised mel
        """
        B, C, F, T_out = xr.shape
        xrecon_cond = xrecon  # (1, F, T_out)

        xt = torch.randn(B, F, T_out, device=self.device)

        def _denoise_loop(xt):
            for t_idx in reversed(range(self.T)):
                t_tensor = torch.full((B,), t_idx, device=self.device, dtype=torch.long)
                epsilon_pred = self.postnet(xt, t_tensor, xrecon_cond)
                coef = (1.0 - self.alphas[t_idx]) / self.sqrt_one_minus_alphas_cumprod[t_idx]
                xt = (1.0 / self.sqrt_alphas[t_idx]) * (xt - coef * epsilon_pred)
                if t_idx > 0:
                    xt = xt + self.posterior_std[t_idx] * torch.randn_like(xt)
            return xt

        if self.ema is not None:
            with self.ema.average_parameters():
                with torch.no_grad():
                    xt = _denoise_loop(xt)
        else:
            with torch.no_grad():
                xt = _denoise_loop(xt)

        return xt.unsqueeze(1)  # (1, 1, F, T_out)



def inference_one_audio(
    audio_file: str,
    scale: float,
    diffatsm: DiffATSMInference,
    conf: Namespace,
    genonly_mel_dir: str,
    full_mel_dir: str,
    output_mel_plt_dir: str,
):
    file_name_no_ext = os.path.splitext(os.path.basename(audio_file))[0]

    # Step 1: Load audio
    audio = load_and_norm_audio(audio_file, conf.mel_params["sampling_rate"])

    # Step 2: Extract mel
    input_mel = create_mel_from_audio(audio, conf.mel_params, conf.must_divide)
    input_mel = input_mel.to(diffatsm.device)

    # Save original mel stats for denormalization after PostNet
    B = input_mel.shape[0]
    flat = input_mel.view(B, -1)
    mel_min = flat.min(dim=1)[0].view(B, 1, 1, 1)
    mel_max = flat.max(dim=1)[0].view(B, 1, 1, 1)

    # Normalize to [-1, 1] — PostNet was trained on normalized mels
    input_mel_norm = diffatsm.normalize_mel(input_mel)  # (1, 1, F, T)

    # Step 3: Extract PPG
    if isinstance(audio, np.ndarray):
        audio_tensor = torch.from_numpy(audio).float()
    else:
        audio_tensor = audio.float()
    ppg = diffatsm.ppg_extractor.extract_ppg(
        audio_tensor,
        sampling_rate=conf.mel_params["sampling_rate"]
    )

    # Step 4: Generate scaled mel — feed RAW mel to generator
    # Generator (ScalerGAN) was trained on raw log-mel [-11, 2], not normalized
    xr = diffatsm.generate_scaled_mel(input_mel, ppg, scale)  # (1, 1, F, T_scaled)

    # Normalize xr for PostNet conditioning
    xr_norm = diffatsm.normalize_mel(xr)  # (1, 1, F, T_scaled) in [-1, 1]

    # Step 5: Get xrecon conditioning using raw mel and normalized xr
    xrecon = diffatsm.get_xrecon(input_mel_norm, xr_norm)  # (1, F, T_scaled)

    # Step 6: Denoise with PostNet conditioned on xrecon — use normalized xr
    final_mel = diffatsm.denoise_with_postnet(xr_norm, xrecon)  # (1, 1, F, T_scaled) in [-1, 1]

    # Clamp PostNet output to [-1, 1] before denorm — reverse diffusion can drift outside this
    final_mel_clamped = final_mel.clamp(-1.0, 1.0)

    # Step 7: Denormalize back to raw log-mel scale — HiFi-GAN expects [-11, 2] not [-1, 1]
    final_mel_denorm = (final_mel_clamped + 1.0) / 2.0 * (mel_max - mel_min) + mel_min

    # Save generator-only output — xr is raw log-mel (generator received raw input_mel)
    np.save(os.path.join(genonly_mel_dir, file_name_no_ext), xr.squeeze(0).cpu().data.numpy())

    # Save full model output (PostNet denoised + denormalized)
    np.save(os.path.join(full_mel_dir, file_name_no_ext), final_mel_denorm.squeeze(0).cpu().data.numpy())

    # Plot if requested
    if conf.infer_plt:
        plot_comparison(input_mel, xr, final_mel_denorm, file_name_no_ext, output_mel_plt_dir)



def plot_comparison(
    input_mel: torch.Tensor,
    xr: torch.Tensor,
    final_mel: torch.Tensor,
    filename: str,
    output_dir: str,
):
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    plt_path = os.path.join(output_dir, f"{filename}.jpg")

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    axes[0].imshow(
        input_mel.squeeze().cpu().numpy(),
        aspect='auto', origin='lower', cmap='viridis'
    )
    axes[0].set_title('Input Mel')
    axes[0].set_ylabel('Mel Bin')
    axes[0].set_xlabel('Time')

    axes[1].imshow(
        xr.squeeze().cpu().numpy(),
        aspect='auto', origin='lower', cmap='viridis'
    )
    axes[1].set_title('Generator Output (xr)')
    axes[1].set_ylabel('Mel Bin')
    axes[1].set_xlabel('Time')

    axes[2].imshow(
        final_mel.squeeze().cpu().numpy(),
        aspect='auto', origin='lower', cmap='viridis'
    )
    axes[2].set_title('Final Output (PostNet Denoised)')
    axes[2].set_ylabel('Mel Bin')
    axes[2].set_xlabel('Time')

    plt.tight_layout()
    plt.savefig(plt_path, dpi=150, bbox_inches='tight')
    plt.close()



def hifi_inference(
    hifi_checkpoint: str,
    hifi_config: str,
    output_mel_dir: str,
    output_wav_dir: str
):
    """Convert mels to audio using HiFi-GAN."""
    hifi_e2e_dict = {
        "input_mels_dir": os.path.abspath(output_mel_dir),
        "output_dir": os.path.abspath(output_wav_dir),
        "checkpoint_file": os.path.abspath(hifi_checkpoint),
    }
    hifi_e2e_conf = Namespace(**hifi_e2e_dict)

    with open(hifi_config) as f:
        data = f.read()
    json_config = json.loads(data)
    inference_e2e.h = AttrDict(json_config)
    inference_e2e.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"HiFi-GAN checkpoint: {hifi_checkpoint}")
    logger.info(f"Input mels: {output_mel_dir}")
    logger.info(f"Output audio: {output_wav_dir}")
    inference_e2e.inference(hifi_e2e_conf)



def inference_diffatsm_complete(diffatsm: DiffATSMInference):
    conf = diffatsm.conf

    logger.info("\n" + "=" * 70)
    logger.info("COMPLETE DiffATSM INFERENCE")
    logger.info("=" * 70)
    logger.info(f"Artifacts: {conf.artifacts_dir}")
    logger.info(f"Scales: {conf.infer_scales}")
    logger.info("=" * 70 + "\n")

    audio_files = files_to_list(conf.inference_file)
    logger.info(f"Found {len(audio_files)} test files")

    for scale in conf.infer_scales:
        scale_str = str(scale)

        # Per-scale directory structure
        # generator_only/{scale}/
        genonly_mel_dir = os.path.join(conf.artifacts_dir, "generator_only", scale_str, MELS_NPY)
        genonly_wav_dir = os.path.join(conf.artifacts_dir, "generator_only", scale_str, WAVS)
        # full_pipeline/{scale}/
        full_mel_dir    = os.path.join(conf.artifacts_dir, "full_pipeline",  scale_str, MELS_NPY)
        full_wav_dir    = os.path.join(conf.artifacts_dir, "full_pipeline",  scale_str, WAVS)
        # plots shared per scale
        plt_dir = os.path.join(conf.artifacts_dir, MELS_IMGS_DIR, G_PRED_DIR, scale_str)

        for d in [genonly_mel_dir, genonly_wav_dir, full_mel_dir, full_wav_dir, plt_dir]:
            os.makedirs(d, exist_ok=True)

        logger.info(f"Processing scale: {scale}")
        logger.info("-" * 70)

        for audio_file in tqdm(audio_files, unit="file", desc=f"Scale {scale}", ncols=80):
            try:
                inference_one_audio(
                    audio_file=audio_file,
                    scale=scale,
                    diffatsm=diffatsm,
                    conf=conf,
                    genonly_mel_dir=genonly_mel_dir,
                    full_mel_dir=full_mel_dir,
                    output_mel_plt_dir=plt_dir,
                )
            except Exception as e:
                logger.error(f"Error processing {audio_file}: {e}")
                import traceback
                traceback.print_exc()
                continue

        if conf.infer_hifi:
            logger.info(f"Running HiFi-GAN for scale {scale} — generator only")
            hifi_inference(conf.hifi_checkpoint, conf.hifi_config, genonly_mel_dir, genonly_wav_dir)
            logger.info(f"Running HiFi-GAN for scale {scale} — full pipeline")
            hifi_inference(conf.hifi_checkpoint, conf.hifi_config, full_mel_dir, full_wav_dir)

    logger.info("\n" + "=" * 70)
    logger.info("COMPLETE DiffATSM INFERENCE FINISHED!")
    logger.info("=" * 70)
    logger.info(f"Results: {conf.artifacts_dir}")
    logger.info("=" * 70 + "\n")



def main():
    """Main entry point."""
    config_obj = Config()

    config_obj.parser.add_argument(
        '--generator_checkpoint',
        type=str,
        required=True,
        help='Path to trained Adaptive Generator checkpoint'
    )
    config_obj.parser.add_argument(
        '--postnet_checkpoint',
        type=str,
        required=True,
        help='Path to trained Diffusion PostNet checkpoint'
    )

    conf = config_obj.parse(inference_mode=True)

    conf.checkpoint_path = conf.generator_checkpoint

    if not os.path.exists(conf.checkpoint_path):
        raise FileNotFoundError(f"Generator checkpoint not found: {conf.checkpoint_path}")

    if not os.path.exists(conf.postnet_checkpoint):
        raise FileNotFoundError(f"PostNet checkpoint not found: {conf.postnet_checkpoint}")

    os.makedirs(conf.artifacts_dir, exist_ok=True)
    log_level = "DEBUG" if conf.verbose or conf.debug else "INFO"
    log_file = os.path.join(conf.artifacts_dir, INFERENCE_LOG)
    init_logger(log_file=log_file, log_level=log_level)

    diffatsm = DiffATSMInference(
        conf=conf,
        generator_checkpoint=conf.checkpoint_path,
        postnet_checkpoint=conf.postnet_checkpoint,
    )

    inference_diffatsm_complete(diffatsm)


if __name__ == "__main__":
    main()
