# #!/usr/bin/env python3
# """
# Train DiffATSM Diffusion-based PostNet.

# Paper Section 4.2:
# "We defined the diffusion time step to 100 with a linear noise schedule
#  ranging from 1x10^-4 to 0.05. The model training was conducted for over
#  500,000 iterations using an Adam optimizer with an initial learning rate
#  of 0.001."

# Usage:
#     python -m scaler_gan.train_postnet \
#         --generator_checkpoint path/to/checkpoint_00500.pth.tar \
#         --postnet_iterations 500000 \
#         --device cuda
# """

# import os
# import sys
# from argparse import Namespace

# import torch
# from torch.utils.data import DataLoader

# from scaler_gan.configs.configs import Config
# from scaler_gan.data_generator.dataloader import MelDataset
# from scaler_gan.trainer.postnet_trainer import PostNetTrainer
# from scaler_gan.scalergan_utils.global_logger import logger
# from scaler_gan.scalergan_utils.scalergan_utils import init_logger


# def train_postnet(conf: Namespace):
#     """
#     Main PostNet training function.
    
#     Args:
#         conf: Configuration namespace
#     """
#     # Validate generator checkpoint
#     if not hasattr(conf, 'generator_checkpoint') or not conf.generator_checkpoint:
#         raise ValueError(
#             "Must specify --generator_checkpoint pointing to trained adaptive generator"
#         )
    
#     if not os.path.exists(conf.generator_checkpoint):
#         raise FileNotFoundError(
#             f"Generator checkpoint not found: {conf.generator_checkpoint}"
#         )
    
#     logger.info("=" * 70)
#     logger.info("DiffATSM: Training Diffusion-based PostNet")
#     logger.info("=" * 70)
#     logger.info(f"Generator checkpoint: {conf.generator_checkpoint}")
#     logger.info(f"Target iterations: {conf.postnet_iterations:,}")
#     logger.info(f"Batch size: {conf.batch_size}")
#     logger.info(f"Device: {conf.device}")
#     logger.info("=" * 70)
    
#     # Initialize PostNet trainer
#     trainer = PostNetTrainer(conf, conf.generator_checkpoint)
    
#     # Resume from checkpoint if specified
#     if conf.resume:
#         trainer.load_checkpoint(conf.resume)
    
#     # Setup dataloader (same as generator training)
#     train_dataset = MelDataset(
#         training_files=conf.input_file,
#         must_divide=conf.must_divide,
#         segment_size=conf.mel_params.segment_size,
#         n_fft=conf.mel_params.n_fft,
#         num_mels=conf.mel_params.num_mels,
#         hop_size=conf.mel_params.hop_size,
#         win_size=conf.mel_params.win_size,
#         sampling_rate=conf.mel_params.sampling_rate,
#         fmin=conf.mel_params.fmin,
#         fmax=conf.mel_params.fmax,
#         shuffle=True,
#         use_ppg=False,  # PostNet doesn't need PPG during training
#     )
    
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=conf.batch_size,
#         shuffle=True,
#         num_workers=conf.num_workers,
#         pin_memory=True,
#         drop_last=True,
#     )
    
#     logger.info(f"Training dataset: {len(train_dataset)} samples")
#     logger.info(f"Batches per epoch: {len(train_loader)}")
    
#     # Training loop
#     logger.info("\n Starting PostNet training...\n")
    
#     while trainer.current_iter < conf.postnet_iterations:
#         trainer.train_one_epoch(train_loader)
        
#         if trainer.current_iter >= conf.postnet_iterations:
#             break
    
#     # Final checkpoint
#     trainer.save_checkpoint()
    
#     logger.info("\n" + "=" * 70)
#     logger.info(" PostNet training complete!")
#     logger.info(f"Final iteration: {trainer.current_iter:,}")
#     logger.info(f"Checkpoints saved to: {conf.artifacts_dir}/postnet_checkpoints/")
#     logger.info("=" * 70)


# def main():
#     """Main entry point."""
#     # Parse config
#     conf = Config().parse()
    
#     # Add PostNet-specific parameters
#     parser = conf.parser if hasattr(conf, 'parser') else Config().parser
    
#     # PostNet arguments (Paper Section 4.2)
#     parser.add_argument(
#         '--generator_checkpoint',
#         type=str,
#         default='PLACEHOLDER_PATH_TO_GENERATOR_CHECKPOINT.pth.tar',
#         help='Path to trained adaptive generator checkpoint (REQUIRED)'
#     )
#     parser.add_argument(
#         '--postnet_iterations',
#         type=int,
#         default=500000,
#         help='Number of training iterations (paper: 500,000)'
#     )
#     parser.add_argument(
#         '--postnet_lr',
#         type=float,
#         default=0.001,
#         help='PostNet learning rate (paper: 0.001)'
#     )
#     parser.add_argument(
#         '--postnet_channels',
#         type=int,
#         default=256,
#         help='PostNet residual channels (default: 256)'
#     )
#     parser.add_argument(
#         '--postnet_blocks',
#         type=int,
#         default=20,
#         help='Number of residual blocks (default: 20)'
#     )
    
#     # Re-parse with PostNet args
#     conf = parser.parse_args()
    
#     # Setup logging
#     log_file = os.path.join(conf.artifacts_dir, 'postnet_train.log')
#     init_logger(log_file=log_file, log_level='INFO')
    
#     # Train
#     train_postnet(conf)


# if __name__ == '__main__':
#     main()




#
#!/usr/bin/env python3
#"""
#Train DiffATSM Diffusion-based PostNet.
#
#Paper Section 4.2:
#"We defined the diffusion time step to 100 with a linear noise schedule
# ranging from 1x10^-4 to 0.05. The model training was conducted for over
# 500,000 iterations using an Adam optimizer with an initial learning rate
# of 0.001."
#
#Usage:
#    python -m scaler_gan.train_postnet \
#        --generator_checkpoint path/to/checkpoint_00500.pth.tar \
#        --postnet_iterations 500000 \
#        --device cuda
#"""
#
#import os
#import sys
#from argparse import Namespace
#import wandb
#import torch
#from torch.utils.data import DataLoader
#
#from scaler_gan.configs.configs import Config
#from scaler_gan.data_generator.dataloader import MelDataset
#from scaler_gan.trainer.postnet_trainer import PostNetTrainer
#from scaler_gan.scalergan_utils.global_logger import logger
#from scaler_gan.scalergan_utils.scalergan_utils import init_logger
#
#
#def train_postnet(conf: Namespace):
#    """
#    Main PostNet training function.
#
#    Args:
#        conf: Configuration namespace
#    """
#    # Validate generator checkpoint
#    if not hasattr(conf, 'generator_checkpoint') or not conf.generator_checkpoint:
#        raise ValueError(
#            "Must specify --generator_checkpoint pointing to trained adaptive generator"
#        )
#
#    if not os.path.exists(conf.generator_checkpoint):
#        raise FileNotFoundError(
#            f"Generator checkpoint not found: {conf.generator_checkpoint}"
#        )
#
#    logger.info("=" * 70)
#    logger.info("DiffATSM: Training Diffusion-based PostNet")
#    logger.info("=" * 70)
#    logger.info(f"Generator checkpoint: {conf.generator_checkpoint}")
#    logger.info(f"Target iterations: {conf.postnet_iterations:,}")
#    logger.info(f"Batch size: {conf.batch_size}")
#    logger.info(f"Device: {conf.device}")
#    logger.info("=" * 70)
#
#    # Initialize wandb if enabled
#    if conf.wandb:
#        wandb.init(
#            project="DiffATSM",
#            name=f"postnet_{os.path.basename(conf.output_dir)}",
#            config={
#                "postnet_iterations": conf.postnet_iterations,
#                "postnet_lr": getattr(conf, 'postnet_lr', 0.001),
#                "postnet_channels": getattr(conf, 'postnet_channels', 256),
#                "postnet_blocks": getattr(conf, 'postnet_blocks', 20),
#                "batch_size": conf.batch_size,
#                "diffusion_T": 100,
#                "beta_start": 1e-4,
#                "beta_end": 0.05,
#                "generator_checkpoint": conf.generator_checkpoint,
#            },
#            dir=conf.artifacts_dir,
#        )
#        logger.info(" wandb initialized")
#
#    # Initialize PostNet trainer
#    trainer = PostNetTrainer(conf, conf.generator_checkpoint)
#
#    # Resume from checkpoint if specified
#    if conf.resume:
#        trainer.load_checkpoint(conf.resume)
#
#    # Setup dataloader
#    train_dataset = MelDataset(
#        training_files=conf.input_file,
#        must_divide=conf.must_divide,
#        segment_size=conf.mel_params['segment_size'],
#        n_fft=conf.mel_params['n_fft'],
#        num_mels=conf.mel_params['num_mels'],
#        hop_size=conf.mel_params['hop_size'],
#        win_size=conf.mel_params['win_size'],
#        sampling_rate=conf.mel_params['sampling_rate'],
#        fmin=conf.mel_params['fmin'],
#        fmax=conf.mel_params['fmax'],
#        shuffle=True,
#        use_ppg=False,
#    )
#
#    train_loader = DataLoader(
#        train_dataset,
#        batch_size=conf.batch_size,
#        shuffle=True,
#        num_workers=conf.num_workers,
#        pin_memory=True,
#        drop_last=True,
#    )
#
#    logger.info(f"Training dataset: {len(train_dataset)} samples")
#    logger.info(f"Batches per epoch: {len(train_loader)}")
#    logger.info("\n Starting PostNet training...\n")
#
#    while trainer.current_iter < conf.postnet_iterations:
#        trainer.train_one_epoch(train_loader)
#
#        if trainer.current_iter >= conf.postnet_iterations:
#            break
#
#    # Final checkpoint
#    trainer.save_checkpoint()
#
#    logger.info("\n" + "=" * 70)
#    logger.info(" PostNet training complete!")
#    logger.info(f"Final iteration: {trainer.current_iter:,}")
#    logger.info(f"Checkpoints saved to: {conf.artifacts_dir}/postnet_checkpoints/")
#    logger.info("=" * 70)
#
#    if conf.wandb:
#        wandb.finish()
#
#
#def main():
#    """Main entry point."""
#    # Build Config object but don't parse yet
#    config_obj = Config()
#
#    # Add PostNet-specific args directly to Config's parser before any parsing
#    config_obj.parser.add_argument(
#        '--generator_checkpoint',
#        type=str,
#        required=True,
#        help='Path to trained adaptive generator checkpoint (REQUIRED)'
#    )
#    config_obj.parser.add_argument(
#        '--postnet_iterations',
#        type=int,
#        default=500000,
#        help='Number of training iterations (paper: 500,000)'
#    )
#    config_obj.parser.add_argument(
#        '--postnet_lr',
#        type=float,
#        default=0.001,
#        help='PostNet learning rate (paper: 0.001)'
#    )
#    config_obj.parser.add_argument(
#        '--postnet_channels',
#        type=int,
#        default=256,
#        help='PostNet residual channels'
#    )
#    config_obj.parser.add_argument(
#        '--postnet_blocks',
#        type=int,
#        default=20,
#        help='Number of residual blocks'
#    )
#    config_obj.parser.add_argument(
#        '--postnet_save_freq',
#        type=int,
#        default=5000,
#        help='Save checkpoint every N iterations'
#    )
#
#    # Single unified parse  all args known, no conflicts
#    conf = config_obj.parse()
#
#    # Create artifacts dir before logging (parse() creates output_dir but not artifacts_dir)
#    os.makedirs(conf.artifacts_dir, exist_ok=True)
#
#    # Setup logging
#    log_file = os.path.join(conf.artifacts_dir, 'postnet_train.log')
#    init_logger(log_file=log_file, log_level='INFO')
#
#    # Train
#    train_postnet(conf)
#
#
#if __name__ == '__main__':
#    main()
#
#
#
#














#!/usr/bin/env python3
"""
Train DiffATSM Diffusion-based PostNet WITH PPG (matches Figure 1).

Usage:
    python -m scaler_gan.train_postnet \\
        --generator_checkpoint path/to/checkpoint.pth.tar \\
        --postnet_iterations 600000 \\
        --postnet_blocks 30 \\
        --batch_size 16 \\
        --device cuda
"""

import os
from argparse import Namespace

import torch
from torch.utils.data import DataLoader

try:
    import wandb
except ImportError:
    wandb = None

from scaler_gan.configs.configs import Config
from scaler_gan.data_generator.dataloader import MelDataset
from scaler_gan.trainer.postnet_trainer import PostNetTrainer
from scaler_gan.scalergan_utils.global_logger import logger
from scaler_gan.scalergan_utils.scalergan_utils import init_logger


def train_postnet(conf: Namespace):
    """Main PostNet training function WITH PPG."""
    
    if not hasattr(conf, 'generator_checkpoint') or not conf.generator_checkpoint:
        raise ValueError("Must specify --generator_checkpoint")
    
    if not os.path.exists(conf.generator_checkpoint):
        raise FileNotFoundError(f"Generator checkpoint not found: {conf.generator_checkpoint}")
    
    logger.info("=" * 70)
    logger.info("DiffATSM: Training Diffusion-based PostNet WITH PPG")
    logger.info("=" * 70)
    logger.info(f"Generator checkpoint: {conf.generator_checkpoint}")
    logger.info(f"Target iterations: {conf.postnet_iterations:,}")
    logger.info(f"Batch size: {conf.batch_size}")
    logger.info(f"Device: {conf.device}")
    logger.info(f"PPG extraction: ENABLED (matches Figure 1)")
    logger.info("=" * 70)

    # Initialize wandb BEFORE trainer
    if conf.wandb and wandb is not None:
        wandb.init(
            project="scaler-gan-postnet",
            name=f"postnet_b{conf.postnet_blocks}_ch{conf.postnet_channels}_bs{conf.batch_size}",
            config=vars(conf)
        )
        logger.info("wandb initialized")
    elif conf.wandb and wandb is None:
        logger.warning("wandb flag set but wandb not installed -- skipping")

    # Initialize PostNet trainer
    trainer = PostNetTrainer(conf, conf.generator_checkpoint)
    
    # Resume if specified
    if conf.resume:
        trainer.load_checkpoint(conf.resume)
    
    logger.info("Setting up dataloader with PPG extraction...")
    train_dataset = MelDataset(
        training_files=conf.input_file,
        must_divide=conf.must_divide,
        segment_size=conf.mel_params.segment_size,
        n_fft=conf.mel_params.n_fft,
        num_mels=conf.mel_params.num_mels,
        hop_size=conf.mel_params.hop_size,
        win_size=conf.mel_params.win_size,
        sampling_rate=conf.mel_params.sampling_rate,
        fmin=conf.mel_params.fmin,
        fmax=conf.mel_params.fmax,
        shuffle=True,
        use_ppg=True,
        ppg_input_dim=768,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=conf.batch_size,
        shuffle=True,
        num_workers=conf.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    logger.info(f"Training dataset: {len(train_dataset)} samples")
    logger.info(f"Batches per epoch: {len(train_loader)}")
    logger.info("Dataloader returns (mel, ppg) tuples")
    
    # Training loop
    while trainer.current_iter < conf.postnet_iterations:
        trainer.train_one_epoch(train_loader)
        if trainer.current_iter >= conf.postnet_iterations:
            break
    
    # Final checkpoint
    trainer.save_checkpoint()

    if conf.wandb and wandb is not None:
        wandb.finish()
    
    logger.info("=" * 70)
    logger.info("PostNet training complete!")
    logger.info(f"Final iteration: {trainer.current_iter:,}")
    logger.info(f"Checkpoints: {conf.artifacts_dir}/postnet_checkpoints/")
    logger.info("=" * 70)


def main():
    """Main entry point."""
    config_obj = Config()

    # Add PostNet-specific args BEFORE any parse() call
    config_obj.parser.add_argument(
        '--generator_checkpoint',
        type=str,
        required=True,
        help='Path to trained adaptive generator checkpoint'
    )
    config_obj.parser.add_argument(
        '--postnet_iterations',
        type=int,
        default=600000,
        help='Number of training iterations'
    )
    config_obj.parser.add_argument(
        '--postnet_lr',
        type=float,
        default=2e-4,
        help='PostNet learning rate'
    )
    config_obj.parser.add_argument(
        '--postnet_channels',
        type=int,
        default=256,
        help='PostNet residual channels'
    )
    config_obj.parser.add_argument(
        '--postnet_blocks',
        type=int,
        default=30,
        help='Number of residual blocks'
    )

    conf = config_obj.parse()  # single parse, knows ALL args

    os.makedirs(conf.artifacts_dir, exist_ok=True)

    # Setup logging
    log_file = os.path.join(conf.artifacts_dir, 'postnet_train.log')
    init_logger(log_file=log_file, log_level='INFO')

    # Train
    train_postnet(conf)


if __name__ == '__main__':
    main()
