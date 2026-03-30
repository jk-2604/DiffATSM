#import logging
#from typing import Optional, List, Tuple, Union
#
#import numpy as np
#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#from torch.autograd import Variable
#
#from scaler_gan.scalergan_utils.scalergan_utils import (
#    homography_based_on_top_corners_x_shift,
#    homography_grid,
#)
#
#
#def weights_init(module: nn.Module):
#    """
#    This is used to initialize weights of any network
#    :param module: Module object that its weight should be initialized
#    :return:
#    """
#    class_name = module.__class__.__name__
#    if class_name.find("Conv") != -1:
#        nn.init.xavier_normal_(module.weight, 0.01)
#        if hasattr(module.bias, "data"):
#            module.bias.data.fill_(0)
#    elif class_name.find("nn.BatchNorm2d") != -1:
#        module.weight.data.normal_(1.0, 0.02)
#        module.bias.data.fill_(0)
#    elif class_name.find("LocalNorm") != -1:
#        module.weight.data.normal_(1.0, 0.02)
#        module.bias.data.fill_(0)
#
#
#class LocalNorm(nn.Module):
#    """Local Normalization class"""
#
#    def __init__(self, num_features: int):
#        super(LocalNorm, self).__init__()
#        self.weight = nn.Parameter(torch.Tensor(num_features))
#        self.bias = nn.Parameter(torch.Tensor(num_features))
#        self.get_local_mean = nn.AvgPool2d(33, 1, 16, count_include_pad=False)
#        self.get_var = nn.AvgPool2d(33, 1, 16, count_include_pad=False)
#
#    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
#        local_mean = self.get_local_mean(input_tensor)
#        centered_input_tensor = input_tensor - local_mean
#        squared_diff = centered_input_tensor ** 2
#        local_std = self.get_var(squared_diff) ** 0.5
#        normalized_tensor = centered_input_tensor / (local_std + 1e-8)
#        return normalized_tensor
#
#
#class GANLoss(nn.Module):
#    """GAN Loss module"""
#
#    def __init__(self):
#        super(GANLoss, self).__init__()
#        self.label_tensor = None
#        self.loss = nn.MSELoss()
#
#    def forward(self, d_last_layer: torch.Tensor, is_d_input_real: bool) -> torch.Tensor:
#        self.label_tensor = (
#            Variable(torch.ones_like(d_last_layer), requires_grad=False) * is_d_input_real
#        )
#        return self.loss(d_last_layer, self.label_tensor)
#
#
#class WeightedMSELoss(nn.Module):
#    """Weighted MSE Loss"""
#
#    def __init__(self, use_l1: Optional[bool] = False):
#        super(WeightedMSELoss, self).__init__()
#        self.unweighted_loss = nn.L1Loss() if use_l1 else nn.MSELoss()
#
#    def forward(
#        self,
#        input_tensor: torch.Tensor,
#        target_tensor: torch.Tensor,
#        loss_mask: torch.Tensor,
#    ) -> torch.Tensor:
#        if loss_mask is not None:
#            e = (target_tensor.detach() - input_tensor) ** 2
#            e *= loss_mask
#            return torch.sum(e) / torch.sum(loss_mask)
#        return self.unweighted_loss(input_tensor, target_tensor)
#
#
#class MultiScaleLoss(nn.Module):
#    """Multiscale Loss"""
#
#    def __init__(self):
#        super(MultiScaleLoss, self).__init__()
#        self.mse = nn.MSELoss()
#
#    def forward(
#        self,
#        input_tensor: torch.Tensor,
#        target_tensor: torch.Tensor,
#        scale_weights: torch.Tensor,
#    ) -> torch.Tensor:
#        loss = torch.tensor(0)
#        for i, scale_weight in enumerate(scale_weights):
#            input_tensor_scaled = F.interpolate(
#                input_tensor,
#                scale_factor=self.scale_factor ** (-i),
#                mode="bilinear",
#                align_corners=False,
#            )
#            loss += scale_weight * self.mse(input_tensor_scaled, target_tensor)
#        return loss
#
#
## ===========================================================================
## DiffATSM: PPGPreNet (Paper Section 3.1.2)
## ===========================================================================
#class PPGPreNet(nn.Module):
#    """
#    PPG Pre-processing Network.
#    
#    DiffATSM Paper Section 3.1.2:
#    "PPG features are then projected through a PPG pre-processing network (PreNet),
#     which consists of simple fully connected (FC) layers, and are used as auxiliary
#     conditioning."
#    
#    Section 4.2:
#    "PreNet output dimension set to 256"
#    
#    Architecture: Linear(768) -> ReLU -> Linear(256) -> ReLU -> Linear(80)
#    """
#
#    def __init__(
#        self,
#        ppg_input_dim: int = 768,
#        hidden_dim: int = 256,
#        mel_bins: int = 80,
#    ):
#        super(PPGPreNet, self).__init__()
#        self.net = nn.Sequential(
#            nn.Linear(ppg_input_dim, hidden_dim),
#            nn.ReLU(inplace=True),
#            nn.Linear(hidden_dim, mel_bins),
#        )
#
#    def forward(self, ppg: torch.Tensor) -> torch.Tensor:
#        """
#        Args:
#            ppg: (B, T, ppg_input_dim) - HuBERT 12th-layer features
#        Returns:
#            (B, 1, mel_bins, T) - projected features ready to add to mel
#        """
#        out = self.net(ppg)           # (B, T, mel_bins)
#        out = out.permute(0, 2, 1)    # (B, mel_bins, T)
#        return out.unsqueeze(1)       # (B, 1, mel_bins, T)
#
#
## ===========================================================================
## DiffATSM: AdaptiveTransformation (Paper Section 3.1.1)
## ===========================================================================
#class AdaptiveTransformation(nn.Module):
#    """
#    Adaptive Time-Scale Modification with Voiced/Unvoiced Differential Scaling.
#    
#    DiffATSM Paper Section 3.1.1:
#    "The mel spectrogram, in conjunction with PPG, is adaptively transformed according
#     to the desired scale factor r for both voiced and unvoiced sections along the
#     time-scale... by applying a faster speed ratio than desired scale ratio (ruv < r)
#     to the unvoiced sections, and conversely, a slower speed ratio (rv > r) to the
#     voiced sections, greater flexibility in adjusting the speed of the speech achieved
#     without compromising intelligibility."
#    
#    Section 4.2:
#    "scale ratio for voiced and unvoiced sections to 7:3 during the adaptive transformation"
#    
#    This means:
#    - Total output frames: T_out = r * T_in
#    - Voiced budget: 70% of output frames (rv > r)
#    - Unvoiced budget: 30% of output frames (ruv < r)
#    - Constraint: rv * len_voiced + ruv * len_unvoiced = T_out
#    """
#
#    def __init__(self, voiced_ratio: float = 0.7, energy_threshold: float = 0.5):
#        """
#        Args:
#            voiced_ratio: Fraction of output budget for voiced content (paper: 0.7)
#            energy_threshold: Energy threshold multiplier for V/UV detection (paper: not specified)
#        """
#        super(AdaptiveTransformation, self).__init__()
#        self.voiced_ratio = voiced_ratio
#        self.unvoiced_ratio = 1.0 - voiced_ratio
#        self.energy_threshold = energy_threshold
#
#    def _energy_based_vu_detection(self, mel: torch.Tensor) -> torch.Tensor:
#        """
#        Voiced/Unvoiced detection using mel energy threshold.
#        
#        Paper does not specify exact method; this is the most common approach
#        when text labels are unavailable (which is the paper's key constraint).
#        
#        Args:
#            mel: (B, 1, F, T)
#        Returns:
#            vu_mask: (B, T) where 1=voiced, 0=unvoiced
#        """
#        # Average energy across frequency bins
#        energy = mel.squeeze(1).mean(dim=1)  # (B, T)
#        
#        # Threshold = energy_threshold * mean_energy per sample
#        threshold = energy.mean(dim=-1, keepdim=True) * self.energy_threshold
#        
#        return (energy > threshold).float()
#
#    @staticmethod
#    def _interpolate_segment(seg: torch.Tensor, target_len: int) -> torch.Tensor:
#        """
#        Bilinear interpolation of a segment along time axis.
#        
#        Args:
#            seg: (C, F, seg_len)
#            target_len: desired output length
#        Returns:
#            (C, F, target_len)
#        """
#        if target_len == seg.shape[-1]:
#            return seg
#        return F.interpolate(
#            seg.unsqueeze(0),
#            size=(seg.shape[-2], target_len),
#            mode='bilinear',
#            align_corners=False
#        ).squeeze(0)
#
#    def forward(
#        self,
#        mel: torch.Tensor,
#        output_size: List[int],
#        vu_mask: Optional[torch.Tensor] = None,
#    ) -> torch.Tensor:
#        """
#        Adaptive transformation with voiced/unvoiced differential scaling.
#        
#        Args:
#            mel: (B, C, F, T_in)
#            output_size: [F_out, T_out] - target spatial dimensions
#            vu_mask: Optional (B, T_in) binary mask, 1=voiced. If None, uses energy-based detection.
#        Returns:
#            (B, C, F_out, T_out) adaptively scaled mel spectrogram
#        """
#        B, C, F, T_in = mel.shape
#        F_out, T_out = output_size
#
#        # V/UV detection
#        if vu_mask is None:
#            vu_mask = self._energy_based_vu_detection(mel)
#
#        outputs = []
#        for b in range(B):
#            mask_b = vu_mask[b].cpu()  # (T_in,)
#            mel_b = mel[b]              # (C, F, T_in)
#
#            # ---- Find contiguous voiced/unvoiced segments ----
#            segments = []  # List[(start, end, is_voiced)]
#            i = 0
#            while i < T_in:
#                is_voiced = int(mask_b[i].item())
#                j = i + 1
#                while j < T_in and int(mask_b[j].item()) == is_voiced:
#                    j += 1
#                segments.append((i, j, is_voiced))
#                i = j
#
#            # ---- Compute total voiced/unvoiced lengths ----
#            len_voiced = sum(end - start for start, end, v in segments if v == 1)
#            len_unvoiced = sum(end - start for start, end, v in segments if v == 0)
#
#            # ---- Solve for rv and ruv ----
#            # Constraints:
#            #   rv * len_voiced + ruv * len_unvoiced = T_out
#            #   rv / ruv = voiced_ratio / unvoiced_ratio  (e.g., 7/3)
#            #
#            # Solution:
#            #   ratio = voiced_ratio / unvoiced_ratio
#            #   rv * len_voiced + (rv / ratio) * len_unvoiced = T_out
#            #   rv = T_out / (len_voiced + len_unvoiced / ratio)
#            #   ruv = rv / ratio
#
#            if len_voiced > 0 and len_unvoiced > 0:
#                ratio = self.voiced_ratio / self.unvoiced_ratio
#                rv = T_out / (len_voiced + len_unvoiced / ratio)
#                ruv = rv / ratio
#            elif len_voiced > 0:
#                # All voiced
#                rv = T_out / len_voiced
#                ruv = rv  # Not used
#            else:
#                # All unvoiced
#                ruv = T_out / len_unvoiced
#                rv = ruv  # Not used
#
#            # ---- Scale each segment and concatenate ----
#            scaled_segments = []
#            total_frames_allocated = 0
#
#            for idx, (start, end, is_voiced) in enumerate(segments):
#                seg = mel_b[:, :, start:end]  # (C, F, seg_len)
#                seg_len = end - start
#
#                # Compute target length for this segment
#                scale_factor = rv if is_voiced else ruv
#                target_len_float = scale_factor * seg_len
#
#                # Last segment absorbs rounding residual to guarantee exact T_out
#                if idx == len(segments) - 1:
#                    target_len = max(1, T_out - total_frames_allocated)
#                else:
#                    target_len = max(1, round(target_len_float))
#
#                total_frames_allocated += target_len
#                scaled_segments.append(self._interpolate_segment(seg, target_len))
#
#            # Concatenate all scaled segments
#            mel_scaled = torch.cat(scaled_segments, dim=-1)  # (C, F, ~T_out)
#
#            # Safety: ensure exact output dimensions (handles rounding drift)
#            if mel_scaled.shape[-2] != F_out or mel_scaled.shape[-1] != T_out:
#                mel_scaled = F.interpolate(
#                    mel_scaled.unsqueeze(0),
#                    size=(F_out, T_out),
#                    mode='bilinear',
#                    align_corners=False
#                ).squeeze(0)
#
#            outputs.append(mel_scaled)
#
#        return torch.stack(outputs, dim=0)
#
#
## ===========================================================================
## DiffATSM: Generator (Adaptive Neural Generator, Paper Section 3.1)
## ===========================================================================
#class Generator(nn.Module):
#    """
#    Adaptive Neural Generator for DiffATSM.
#    
#    Paper Section 3.1:
#    "The adaptive neural generator applies variable scaling to different speech segments
#     by conditioning the phonetic posteriorgrams (PPG) derived from a self-supervised
#     speech model, taking pronunciation characteristics into account."
#    
#    Key Modifications from ScalerGAN:
#    1. AdaptiveTransformation replaces bilinear interpolation (Section 3.1.1)
#    2. PPGPreNet adds phonetic conditioning (Section 3.1.2)
#    3. PPG features are ADDED (not concatenated) to mel before entry_block
#    4. Backward compatible: when use_ppg=False, degrades to ScalerGAN
#    """
#
#    def __init__(
#        self,
#        base_channels: Optional[int] = 64,
#        n_blocks: Optional[int] = 6,
#        n_downsampling: Optional[int] = 3,
#        use_bias: Optional[bool] = True,
#        skip_flag: Optional[bool] = True,
#        # DiffATSM parameters
#        use_ppg: Optional[bool] = False,
#        ppg_input_dim: Optional[int] = 768,
#        ppg_hidden_dim: Optional[int] = 256,
#        mel_bins: Optional[int] = 80,
#        voiced_ratio: Optional[float] = 0.7,
#        energy_threshold: Optional[float] = 0.5,
#    ):
#        super(Generator, self).__init__()
#        self.skip = skip_flag
#        self.use_ppg = use_ppg
#
#        # ---- DiffATSM Components ----
#        self.adaptive_transform = AdaptiveTransformation(
#            voiced_ratio=voiced_ratio,
#            energy_threshold=energy_threshold
#        )
#
#        if self.use_ppg:
#            self.ppg_prenet = PPGPreNet(
#                ppg_input_dim=ppg_input_dim,
#                hidden_dim=ppg_hidden_dim,
#                mel_bins=mel_bins,
#            )
#
#        # ---- U-Net Backbone (unchanged from ScalerGAN) ----
#        self.entry_block = nn.Sequential(
#            nn.ReflectionPad2d(3),
#            nn.utils.spectral_norm(
#                nn.Conv2d(1, base_channels, kernel_size=7, bias=use_bias)
#            ),
#            nn.BatchNorm2d(base_channels),
#            nn.LeakyReLU(0.2, True),
#        )
#
#        self.geo_transform = GeoTransform()
#        self.downscale_block = RescaleBlock(n_downsampling, 0.5, base_channels, True)
#
#        bottleneck_block = []
#        for _ in range(n_blocks):
#            bottleneck_block += [
#                ResnetBlock(base_channels * 2 ** n_downsampling, use_bias=use_bias)
#            ]
#        self.bottleneck_block = nn.Sequential(*bottleneck_block)
#
#        self.upscale_block = RescaleBlock(n_downsampling, 2.0, base_channels, True)
#
#        self.final_block = nn.Sequential(
#            nn.ReflectionPad2d(3),
#            nn.Conv2d(base_channels, 1, kernel_size=7),
#        )
#
#    def forward(
#        self,
#        input_tensor: torch.Tensor,
#        output_size: List[int],
#        random_affine: Optional[List[float]],
#        ppg: Optional[torch.Tensor] = None,
#        vu_mask: Optional[torch.Tensor] = None,
#    ) -> torch.Tensor:
#        """
#        Forward pass with optional PPG conditioning and adaptive transformation.
#        
#        Args:
#            input_tensor: (B, 1, F, T_in) - mel spectrogram
#            output_size: [F_out, T_out] - target dimensions
#            random_affine: Geometric transformation params (curriculum training).
#                          If None, uses adaptive transformation (DiffATSM).
#                          If not None, uses geo_transform (ScalerGAN curriculum).
#            ppg: Optional (B, T_in, 768) - HuBERT PPG features
#            vu_mask: Optional (B, T_in) - V/UV binary mask
#        Returns:
#            (B, 1, F_out, T_out) - time-scale modified mel spectrogram
#        """
#        # ---- Step 1: Scale input to output_size ----
#        if random_affine is None:
#            # DiffATSM: Adaptive transformation (voiced/unvoiced differential)
#            input_tensor = self.adaptive_transform(
#                input_tensor,
#                output_size,
#                vu_mask=vu_mask
#            )
#        else:
#            # ScalerGAN: Geometric transformation (curriculum training)
#            input_tensor = self.geo_transform(
#                input_tensor,
#                output_size,
#                random_affine
#            )
#
#        # ---- Step 2: PPG Conditioning (DiffATSM Section 3.1.2) ----
#        if self.use_ppg and ppg is not None:
#            T_out = input_tensor.shape[-1]
#
#             # ppg: (B, T_in, 768) → (B, 768, T_in) for interpolate
#            ppg_t = ppg.permute(0, 2, 1)  # (B, 768, T_in)
#
#
#            # Align PPG temporal axis to scaled mel
#            # ppg_aligned = F.interpolate(
#            #     ppg.permute(0, 2, 1).unsqueeze(1),  # (B, 1, 768, T_in)
#            #     size=(ppg.shape[-1], T_out),
#            #     mode='linear',
#            #     align_corners=False
#            # ).squeeze(1).permute(0, 2, 1)  # (B, T_out, 768)
#            ppg_aligned = F.interpolate(
#                ppg_t,
#                size=T_out,           # scalar — align T_in → T_out
#                mode='linear',
#                align_corners=False
#            )  # (B, 768, T_out)
#
#            ppg_aligned = ppg_aligned.permute(0, 2, 1)  # (B, T_out, 768)
#
#
#            # Project PPG and add to mel as conditioning
#            ppg_cond = self.ppg_prenet(ppg_aligned)  # (B, 1, F, T_out)
#            input_tensor = input_tensor + ppg_cond
#
#        # ---- Step 3: U-Net Forward Pass ----
#        feature_map = self.entry_block(input_tensor)
#
#        feature_map, downscales = self.downscale_block(
#            feature_map, return_all_scales=self.skip
#        )
#
#        feature_map = self.bottleneck_block(feature_map)
#
#        feature_map, _ = self.upscale_block(
#            feature_map, pyramid=downscales, skip=self.skip
#        )
#
#        return self.final_block(feature_map)
#
#
## ===========================================================================
## Unchanged ScalerGAN Components
## ===========================================================================
#
#class ResnetBlock(nn.Module):
#    """A single Res-Block module"""
#
#    def __init__(self, dim: int, use_bias: bool):
#        super(ResnetBlock, self).__init__()
#        self.conv_block = nn.Sequential(
#            nn.utils.spectral_norm(nn.Conv2d(dim, dim // 4, kernel_size=1, bias=use_bias)),
#            nn.BatchNorm2d(dim // 4),
#            nn.LeakyReLU(0.2, True),
#            nn.ReflectionPad2d(1),
#            nn.utils.spectral_norm(nn.Conv2d(dim // 4, dim // 4, kernel_size=3, bias=use_bias)),
#            nn.BatchNorm2d(dim // 4),
#            nn.LeakyReLU(0.2, True),
#            nn.utils.spectral_norm(nn.Conv2d(dim // 4, dim, kernel_size=1, bias=use_bias)),
#            nn.BatchNorm2d(dim),
#        )
#
#    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
#        return input_tensor + self.conv_block(input_tensor)
#
#
#class MultiScaleDiscriminator(nn.Module):
#    """The Multiscale Discriminator class"""
#
#    def __init__(
#        self,
#        real_crop_size: int,
#        max_n_scales: Optional[int] = 9,
#        scale_factor: Optional[int] = 2,
#        base_channels: Optional[int] = 128,
#        extra_conv_layers: Optional[int] = 0,
#    ):
#        super(MultiScaleDiscriminator, self).__init__()
#        self.base_channels = base_channels
#        self.scale_factor = scale_factor
#        self.min_size = 16
#        self.extra_conv_layers = extra_conv_layers
#        self.max_n_scales = np.min([
#            int(np.ceil(np.log(np.min(real_crop_size) * 1.0 / self.min_size) / np.log(self.scale_factor))),
#            max_n_scales,
#        ])
#        self.nets = nn.ModuleList()
#        for _ in range(self.max_n_scales):
#            self.nets.append(self.make_net())
#
#    def make_net(self):
#        base_channels = self.base_channels
#        net = []
#        net += [
#            nn.utils.spectral_norm(nn.Conv2d(1, base_channels, kernel_size=3, stride=1)),
#            nn.BatchNorm2d(base_channels),
#            nn.LeakyReLU(0.2, True),
#        ]
#        net += [
#            nn.utils.spectral_norm(nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2)),
#            nn.BatchNorm2d(base_channels * 2),
#            nn.LeakyReLU(0.2, True),
#        ]
#        net += [
#            nn.utils.spectral_norm(nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, bias=True)),
#            nn.BatchNorm2d(base_channels * 2),
#            nn.LeakyReLU(0.2, True),
#        ]
#        for _ in range(self.extra_conv_layers):
#            net += [
#                nn.utils.spectral_norm(nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, bias=True)),
#                nn.BatchNorm2d(base_channels * 2),
#                nn.LeakyReLU(0.2, True),
#            ]
#        net += nn.Sequential(
#            nn.utils.spectral_norm(nn.Conv2d(base_channels * 2, 1, kernel_size=1)),
#            nn.Sigmoid(),
#        )
#        return nn.Sequential(*net)
#
#    def forward(self, input_tensor: torch.Tensor, scale_weights: torch.Tensor) -> torch.Tensor:
#        aggregated = self.nets[0](input_tensor) * scale_weights[0]
#        map_size = aggregated.shape[2:]
#        logger = logging.getLogger()
#        for net, scale_weight, i in zip(self.nets[1:], scale_weights[1:], range(1, len(scale_weights))):
#            downscaled = F.interpolate(
#                input_tensor,
#                scale_factor=self.scale_factor ** (-i),
#                mode="bilinear",
#                align_corners=False
#            )
#            try:
#                result = net(downscaled)
#            except KeyboardInterrupt:
#                raise
#            except Exception:
#                print(f"Something went wrong in epoch {i}, While training.")
#                print(f"epoch in net: {i}, downscaled_image shape: {downscaled.shape}")
#                raise
#            try:
#                upscaled = F.interpolate(result, size=map_size, mode="bilinear", align_corners=False)
#            except:
#                logger.error(f"-------- ERROR --------\n input tensor shape: {input_tensor.shape}")
#                logger.error(f"downscaled shape: {downscaled.shape}\n epoch: {i}")
#                logger.error(f"result map shape: {result.shape}")
#                raise
#            aggregated += upscaled * scale_weight
#        return aggregated
#
#
#class RescaleBlock(nn.Module):
#    """Rescale Block class"""
#
#    def __init__(
#        self,
#        n_layers: int,
#        scale: Optional[float] = 0.5,
#        base_channels: Optional[int] = 64,
#        use_bias: Optional[bool] = True,
#    ):
#        super(RescaleBlock, self).__init__()
#        self.scale = scale
#        self.conv_layers = [None] * n_layers
#        in_channel_power = scale > 1
#        out_channel_power = scale < 1
#        i_range = range(n_layers) if scale < 1 else range(n_layers - 1, -1, -1)
#        for i in i_range:
#            self.conv_layers[i] = nn.Sequential(
#                nn.ReflectionPad2d(1),
#                nn.utils.spectral_norm(
#                    nn.Conv2d(
#                        in_channels=base_channels * 2 ** (i + in_channel_power),
#                        out_channels=base_channels * 2 ** (i + out_channel_power),
#                        kernel_size=3,
#                        stride=1,
#                        bias=use_bias,
#                    )
#                ),
#                nn.BatchNorm2d(base_channels * 2 ** (i + out_channel_power)),
#                nn.LeakyReLU(0.2, True),
#            )
#            self.add_module("conv_%d" % i, self.conv_layers[i])
#        if scale > 1:
#            self.conv_layers = self.conv_layers[::-1]
#        self.max_pool = nn.MaxPool2d(2, 2)
#
#    def forward(
#        self,
#        input_tensor: torch.Tensor,
#        pyramid: Optional[torch.Tensor] = None,
#        return_all_scales: Optional[bool] = False,
#        skip: Optional[bool] = False,
#    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
#        feature_map = input_tensor
#        all_scales = []
#        if return_all_scales:
#            all_scales.append(feature_map)
#        for i, conv_layer in enumerate(self.conv_layers):
#            if self.scale > 1.0:
#                feature_map = F.interpolate(feature_map, scale_factor=self.scale, mode="nearest")
#            feature_map = conv_layer(feature_map)
#            if skip:
#                feature_map = feature_map + pyramid[-i - 2]
#            if self.scale < 1.0:
#                feature_map = self.max_pool(feature_map)
#            if return_all_scales:
#                all_scales.append(feature_map)
#        return (feature_map, all_scales) if return_all_scales else (feature_map, None)
#
#
#class RandomCrop(nn.Module):
#    """Random Crop class"""
#
#    def __init__(
#        self,
#        crop_size: Optional[List[int]] = None,
#        return_pos: Optional[bool] = False,
#        must_divide: Optional[float] = 4.0,
#    ):
#        super(RandomCrop, self).__init__()
#        self.crop_size = crop_size
#        self.must_divide = must_divide
#        self.return_pos = return_pos
#
#    # def forward(
#    #     self,
#    #     input_tensor: torch.Tensor,
#    #     crop_size: Optional[List[int]] = None,
#    # ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[int, int]]]:
#    #     im_v_sz, im_h_sz = input_tensor.shape[2:]
#    #     if crop_size is None:
#    #         cr_v_sz, cr_h_sz = np.clip(self.crop_size, [0, 0], [im_v_sz - 1, im_h_sz - 1])
#    #         cr_v_sz, cr_h_sz = np.uint32(
#    #             np.floor(np.array([cr_v_sz, cr_h_sz]) * 1.0 / self.must_divide) * self.must_divide
#    #         )
#    #     else:
#    #         cr_v_sz, cr_h_sz = crop_size
#    #     top_left_v = np.random.randint(0, im_v_sz - cr_v_sz)
#    #     top_left_h = np.random.randint(0, im_h_sz - cr_h_sz)
#    #     out_tensor = input_tensor[:, :, top_left_v : top_left_v + cr_v_sz, top_left_h : top_left_h + cr_h_sz]
#    #     return (out_tensor, (top_left_v, top_left_h)) if self.return_pos else out_tensor
#
#
#    def forward(self, input_tensor, crop_size=None):
#        im_v_sz, im_h_sz = input_tensor.shape[2:]
#        if crop_size is None:
#            cr_v_sz = int(np.floor(min(self.crop_size[0], im_v_sz) / self.must_divide) * self.must_divide)
#            cr_h_sz = int(np.floor(min(self.crop_size[1], im_h_sz) / self.must_divide) * self.must_divide)
#        else:
#            cr_v_sz, cr_h_sz = crop_size
#
#        # If crop equals full dim, start must be 0
#        top_left_v = np.random.randint(0, max(1, im_v_sz - cr_v_sz))
#        top_left_h = np.random.randint(0, max(1, im_h_sz - cr_h_sz))
#
#        out_tensor = input_tensor[:, :, top_left_v:top_left_v + cr_v_sz, top_left_h:top_left_h + cr_h_sz]
#        return (out_tensor, (top_left_v, top_left_h)) if self.return_pos else out_tensor
#
#
#class GeoTransform(nn.Module):
#    """Geo Transform class"""
#
#    def __init__(self):
#        super(GeoTransform, self).__init__()
#
#    def forward(
#        self,
#        input_tensor: torch.Tensor,
#        target_size: List[int],
#        shifts: List[float],
#    ) -> torch.Tensor:
#        sz = input_tensor.shape
#        theta = homography_based_on_top_corners_x_shift(shifts).to(input_tensor.device)
#        pad = F.pad(
#            input_tensor,
#            (
#                np.abs(int(np.ceil(sz[3] * shifts[0]))),
#                np.abs(int(np.ceil(-sz[3] * shifts[1]))),
#                0,
#                0,
#            ),
#            "reflect",
#        )
#        target_size4d = torch.Size([pad.shape[0], pad.shape[1], target_size[0], target_size[1]])
#        grid = homography_grid(theta.expand(pad.shape[0], -1, -1), target_size4d)
#        return F.grid_sample(pad, grid, mode="bilinear", padding_mode="border", align_corners=True)























import logging
from typing import Optional, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from scaler_gan.scalergan_utils.scalergan_utils import (
    homography_based_on_top_corners_x_shift,
    homography_grid,
)


def weights_init(module: nn.Module):
    """
    This is used to initialize weights of any network
    :param module: Module object that its weight should be initialized
    :return:
    """
    class_name = module.__class__.__name__
    if class_name.find("Conv") != -1:
        nn.init.xavier_normal_(module.weight, 0.01)
        if hasattr(module.bias, "data"):
            module.bias.data.fill_(0)
    elif class_name.find("nn.BatchNorm2d") != -1:
        module.weight.data.normal_(1.0, 0.02)
        module.bias.data.fill_(0)
    elif class_name.find("LocalNorm") != -1:
        module.weight.data.normal_(1.0, 0.02)
        module.bias.data.fill_(0)


class LocalNorm(nn.Module):
    """Local Normalization class"""

    def __init__(self, num_features: int):
        super(LocalNorm, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_features))
        self.bias = nn.Parameter(torch.Tensor(num_features))
        self.get_local_mean = nn.AvgPool2d(33, 1, 16, count_include_pad=False)
        self.get_var = nn.AvgPool2d(33, 1, 16, count_include_pad=False)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        local_mean = self.get_local_mean(input_tensor)
        centered_input_tensor = input_tensor - local_mean
        squared_diff = centered_input_tensor ** 2
        local_std = self.get_var(squared_diff) ** 0.5
        normalized_tensor = centered_input_tensor / (local_std + 1e-8)
        return normalized_tensor


class GANLoss(nn.Module):
    """GAN Loss module"""

    def __init__(self):
        super(GANLoss, self).__init__()
        self.label_tensor = None
        self.loss = nn.MSELoss()

    def forward(self, d_last_layer: torch.Tensor, is_d_input_real: bool) -> torch.Tensor:
        self.label_tensor = (
            Variable(torch.ones_like(d_last_layer), requires_grad=False) * is_d_input_real
        )
        return self.loss(d_last_layer, self.label_tensor)


class WeightedMSELoss(nn.Module):
    """Weighted MSE Loss"""

    def __init__(self, use_l1: Optional[bool] = False):
        super(WeightedMSELoss, self).__init__()
        self.unweighted_loss = nn.L1Loss() if use_l1 else nn.MSELoss()

    def forward(
        self,
        input_tensor: torch.Tensor,
        target_tensor: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        if loss_mask is not None:
            e = (target_tensor.detach() - input_tensor) ** 2
            e *= loss_mask
            return torch.sum(e) / torch.sum(loss_mask)
        return self.unweighted_loss(input_tensor, target_tensor)


class MultiScaleLoss(nn.Module):
    """Multiscale Loss"""

    def __init__(self):
        super(MultiScaleLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(
        self,
        input_tensor: torch.Tensor,
        target_tensor: torch.Tensor,
        scale_weights: torch.Tensor,
    ) -> torch.Tensor:
        loss = torch.tensor(0)
        for i, scale_weight in enumerate(scale_weights):
            input_tensor_scaled = F.interpolate(
                input_tensor,
                scale_factor=self.scale_factor ** (-i),
                mode="bilinear",
                align_corners=False,
            )
            loss += scale_weight * self.mse(input_tensor_scaled, target_tensor)
        return loss


# ===========================================================================
# DiffATSM: PPGPreNet (Paper Section 3.1.2)
# ===========================================================================
class PPGPreNet(nn.Module):
    """
    PPG Pre-processing Network.
    
    DiffATSM Paper Section 3.1.2:
    "PPG features are then projected through a PPG pre-processing network (PreNet),
     which consists of simple fully connected (FC) layers, and are used as auxiliary
     conditioning."
    
    Section 4.2:
    "PreNet output dimension set to 256"
    
    Architecture: Linear(768) -> ReLU -> Linear(256) -> ReLU -> Linear(80)
    """

    def __init__(
        self,
        ppg_input_dim: int = 768,
        hidden_dim: int = 256,
        mel_bins: int = 80,
    ):
        super(PPGPreNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(ppg_input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, mel_bins),
        )

    def forward(self, ppg: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ppg: (B, T, ppg_input_dim) - HuBERT 12th-layer features
        Returns:
            (B, 1, mel_bins, T) - projected features ready to add to mel
        """
        out = self.net(ppg)           # (B, T, mel_bins)
        out = out.permute(0, 2, 1)    # (B, mel_bins, T)
        return out.unsqueeze(1)       # (B, 1, mel_bins, T)


# ===========================================================================
# DiffATSM: AdaptiveTransformation (Paper Section 3.1.1)
# ===========================================================================
class AdaptiveTransformation(nn.Module):
    """
    Adaptive Time-Scale Modification with Voiced/Unvoiced Differential Scaling.
    
    DiffATSM Paper Section 3.1.1:
    "The mel spectrogram, in conjunction with PPG, is adaptively transformed according
     to the desired scale factor r for both voiced and unvoiced sections along the
     time-scale... by applying a faster speed ratio than desired scale ratio (ruv < r)
     to the unvoiced sections, and conversely, a slower speed ratio (rv > r) to the
     voiced sections, greater flexibility in adjusting the speed of the speech achieved
     without compromising intelligibility."
    
    Section 4.2:
    "scale ratio for voiced and unvoiced sections to 7:3 during the adaptive transformation"
    
    This means:
    - Total output frames: T_out = r * T_in
    - Voiced budget: 70% of output frames (rv > r)
    - Unvoiced budget: 30% of output frames (ruv < r)
    - Constraint: rv * len_voiced + ruv * len_unvoiced = T_out
    """
    def __init__(self, voiced_ratio: float = 0.7, energy_threshold: float = 0.5):
        """
        Args:
            voiced_ratio: Fraction of output budget for voiced content (paper: 0.7)
            energy_threshold: Energy threshold multiplier for V/UV detection (paper: not specified)
        """
        super(AdaptiveTransformation, self).__init__()
        self.voiced_ratio = voiced_ratio
        self.unvoiced_ratio = 1.0 - voiced_ratio
        self.energy_threshold = energy_threshold

    def _energy_based_vu_detection(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Voiced/Unvoiced detection using mel energy threshold.
        
        Paper does not specify exact method; this is the most common approach
        when text labels are unavailable (which is the paper's key constraint).
        
        Args:
            mel: (B, 1, F, T)
        Returns:
            vu_mask: (B, T) where 1=voiced, 0=unvoiced
        """
        # Average energy across frequency bins
        energy = mel.squeeze(1).mean(dim=1)  # (B, T)
        
        # Threshold = energy_threshold * mean_energy per sample
        threshold = energy.mean(dim=-1, keepdim=True) * self.energy_threshold
        
        return (energy > threshold).float()

    @staticmethod
    def _interpolate_segment(seg: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        Bilinear interpolation of a segment along time axis.
        
        Args:
            seg: (C, F, seg_len)
            target_len: desired output length
        Returns:
            (C, F, target_len)
        """
        if target_len == seg.shape[-1]:
            return seg
        return F.interpolate(
            seg.unsqueeze(0),
            size=(seg.shape[-2], target_len),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

    def forward(
        self,
        mel: torch.Tensor,
        output_size: List[int],
        vu_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Adaptive transformation with voiced/unvoiced differential scaling.
        
        Args:
            mel: (B, C, F, T_in)
            output_size: [F_out, T_out] - target spatial dimensions
            vu_mask: Optional (B, T_in) binary mask, 1=voiced. If None, uses energy-based detection.
        Returns:
            (B, C, F_out, T_out) adaptively scaled mel spectrogram
        """
        B, C, F, T_in = mel.shape
        F_out, T_out = output_size

        # V/UV detection
        if vu_mask is None:
            vu_mask = self._energy_based_vu_detection(mel)

        outputs = []
        for b in range(B):
            mask_b = vu_mask[b].cpu()  # (T_in,)
            mel_b = mel[b]              # (C, F, T_in)

            # ---- Find contiguous voiced/unvoiced segments ----
            segments = []  # List[(start, end, is_voiced)]
            i = 0
            while i < T_in:
                is_voiced = int(mask_b[i].item())
                j = i + 1
                while j < T_in and int(mask_b[j].item()) == is_voiced:
                    j += 1
                segments.append((i, j, is_voiced))
                i = j

            # ---- Compute total voiced/unvoiced lengths ----
            len_voiced = sum(end - start for start, end, v in segments if v == 1)
            len_unvoiced = sum(end - start for start, end, v in segments if v == 0)

            # ---- Solve for rv and ruv ----
            # Constraints:
            #   rv * len_voiced + ruv * len_unvoiced = T_out
            #   rv / ruv = voiced_ratio / unvoiced_ratio  (e.g., 7/3)
            #
            # Solution:
            #   ratio = voiced_ratio / unvoiced_ratio
            #   rv * len_voiced + (rv / ratio) * len_unvoiced = T_out
            #   rv = T_out / (len_voiced + len_unvoiced / ratio)
            #   ruv = rv / ratio

            if len_voiced > 0 and len_unvoiced > 0:
                ratio = self.voiced_ratio / self.unvoiced_ratio
                rv = T_out / (len_voiced + len_unvoiced / ratio)
                ruv = rv / ratio
            elif len_voiced > 0:
                # All voiced
                rv = T_out / len_voiced
                ruv = rv  # Not used
            else:
                # All unvoiced
                ruv = T_out / len_unvoiced
                rv = ruv  # Not used

            # ---- Scale each segment and concatenate ----
            scaled_segments = []
            total_frames_allocated = 0

            for idx, (start, end, is_voiced) in enumerate(segments):
                seg = mel_b[:, :, start:end]  # (C, F, seg_len)
                seg_len = end - start

                # Compute target length for this segment
                scale_factor = rv if is_voiced else ruv
                target_len_float = scale_factor * seg_len

                # Last segment absorbs rounding residual to guarantee exact T_out
                if idx == len(segments) - 1:
                    target_len = max(1, T_out - total_frames_allocated)
                else:
                    target_len = max(1, round(target_len_float))

                total_frames_allocated += target_len
                scaled_segments.append(self._interpolate_segment(seg, target_len))

            # Concatenate all scaled segments
            mel_scaled = torch.cat(scaled_segments, dim=-1)  # (C, F, ~T_out)

            # Safety: ensure exact output dimensions (handles rounding drift)
            if mel_scaled.shape[-2] != F_out or mel_scaled.shape[-1] != T_out:
                import torch.nn.functional as _F
                mel_scaled = _F.interpolate(
                    mel_scaled.unsqueeze(0),
                    size=(F_out, T_out),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)

            outputs.append(mel_scaled)

        return torch.stack(outputs, dim=0)

# ===========================================================================
# DiffATSM: Generator (Adaptive Neural Generator, Paper Section 3.1)
# ===========================================================================
class Generator(nn.Module):
    """
    Adaptive Neural Generator for DiffATSM.
    
    Paper Section 3.1:
    "The adaptive neural generator applies variable scaling to different speech segments
     by conditioning the phonetic posteriorgrams (PPG) derived from a self-supervised
     speech model, taking pronunciation characteristics into account."
    
    Key Modifications from ScalerGAN:
    1. AdaptiveTransformation replaces bilinear interpolation (Section 3.1.1)
    2. PPGPreNet adds phonetic conditioning (Section 3.1.2)
    3. PPG features are ADDED (not concatenated) to mel before entry_block
    4. Backward compatible: when use_ppg=False, degrades to ScalerGAN
    """

    def __init__(
        self,
        base_channels: Optional[int] = 64,
        n_blocks: Optional[int] = 6,
        n_downsampling: Optional[int] = 3,
        use_bias: Optional[bool] = True,
        skip_flag: Optional[bool] = True,
        # DiffATSM parameters
        use_ppg: Optional[bool] = False,
        ppg_input_dim: Optional[int] = 768,
        ppg_hidden_dim: Optional[int] = 256,
        mel_bins: Optional[int] = 80,
        voiced_ratio: Optional[float] = 0.7,
        energy_threshold: Optional[float] = 0.5,
    ):
        super(Generator, self).__init__()
        self.skip = skip_flag
        self.use_ppg = use_ppg

        # ---- DiffATSM Components ----
        self.adaptive_transform = AdaptiveTransformation(
            voiced_ratio=voiced_ratio,
            energy_threshold=energy_threshold
        )

        if self.use_ppg:
            self.ppg_prenet = PPGPreNet(
                ppg_input_dim=ppg_input_dim,
                hidden_dim=ppg_hidden_dim,
                mel_bins=mel_bins,
            )


        # ---- U-Net Backbone (unchanged from ScalerGAN) ----
        self.entry_block = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.utils.spectral_norm(
                nn.Conv2d(1, base_channels, kernel_size=7, bias=use_bias)
            ),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, True),
        )

        self.geo_transform = GeoTransform()
        self.downscale_block = RescaleBlock(n_downsampling, 0.5, base_channels, True)

        bottleneck_block = []
        for _ in range(n_blocks):
            bottleneck_block += [
                ResnetBlock(base_channels * 2 ** n_downsampling, use_bias=use_bias)
            ]
        self.bottleneck_block = nn.Sequential(*bottleneck_block)

        self.upscale_block = RescaleBlock(n_downsampling, 2.0, base_channels, True)

        self.final_block = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(base_channels, 1, kernel_size=7),
        )

    def forward(
        self,
        input_tensor: torch.Tensor,
        output_size: List[int],
        random_affine: Optional[List[float]],
        ppg: Optional[torch.Tensor] = None,
        vu_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional PPG conditioning and adaptive transformation.
        
        Args:
            input_tensor: (B, 1, F, T_in) - mel spectrogram
            output_size: [F_out, T_out] - target dimensions
            random_affine: Geometric transformation params (curriculum training).
                          If None, uses adaptive transformation (DiffATSM).
                          If not None, uses geo_transform (ScalerGAN curriculum).
            ppg: Optional (B, T_in, 768) - HuBERT PPG features
            vu_mask: Optional (B, T_in) - V/UV binary mask
        Returns:
            (B, 1, F_out, T_out) - time-scale modified mel spectrogram
        """
        # ---- Step 1: Scale input to output_size ----
        if random_affine is None:
            # DiffATSM: Adaptive transformation (voiced/unvoiced differential)
            input_tensor = self.adaptive_transform(
                input_tensor,
                output_size,
                vu_mask=vu_mask
            )
        else:
            # ScalerGAN: Geometric transformation (curriculum training)
            input_tensor = self.geo_transform(
                input_tensor,
                output_size,
                random_affine
            )


        # ---- Step 2: PPG Conditioning (DiffATSM Section 3.1.2) ----
        if self.use_ppg and ppg is not None:
            T_out = input_tensor.shape[-1]

             # ppg: (B, T_in, 768) → (B, 768, T_in) for interpolate
            ppg_t = ppg.permute(0, 2, 1)  # (B, 768, T_in)


            # Align PPG temporal axis to scaled mel
            # ppg_aligned = F.interpolate(
            #     ppg.permute(0, 2, 1).unsqueeze(1),  # (B, 1, 768, T_in)
            #     size=(ppg.shape[-1], T_out),
            #     mode='linear',
            #     align_corners=False
            # ).squeeze(1).permute(0, 2, 1)  # (B, T_out, 768)
            ppg_aligned = F.interpolate(
                ppg_t,
                size=T_out,           # scalar — align T_in → T_out
                mode='linear',
                align_corners=False
            )  # (B, 768, T_out)

            ppg_aligned = ppg_aligned.permute(0, 2, 1)  # (B, T_out, 768)


            # Project PPG and add to mel as conditioning
            ppg_cond = self.ppg_prenet(ppg_aligned)  # (B, 1, F, T_out)
            input_tensor = input_tensor + ppg_cond

        # ---- Step 3: U-Net Forward Pass ----
        feature_map = self.entry_block(input_tensor)

        feature_map, downscales = self.downscale_block(
            feature_map, return_all_scales=self.skip
        )

        feature_map = self.bottleneck_block(feature_map)

        feature_map, _ = self.upscale_block(
            feature_map, pyramid=downscales, skip=self.skip
        )

        return self.final_block(feature_map)


# ===========================================================================
# Unchanged ScalerGAN Components
# ===========================================================================

class ResnetBlock(nn.Module):
    """A single Res-Block module"""

    def __init__(self, dim: int, use_bias: bool):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(dim, dim // 4, kernel_size=1, bias=use_bias)),
            nn.BatchNorm2d(dim // 4),
            nn.LeakyReLU(0.2, True),
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(nn.Conv2d(dim // 4, dim // 4, kernel_size=3, bias=use_bias)),
            nn.BatchNorm2d(dim // 4),
            nn.LeakyReLU(0.2, True),
            nn.utils.spectral_norm(nn.Conv2d(dim // 4, dim, kernel_size=1, bias=use_bias)),
            nn.BatchNorm2d(dim),
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return input_tensor + self.conv_block(input_tensor)


class MultiScaleDiscriminator(nn.Module):
    """The Multiscale Discriminator class"""

    def __init__(
        self,
        real_crop_size: int,
        max_n_scales: Optional[int] = 9,
        scale_factor: Optional[int] = 2,
        base_channels: Optional[int] = 128,
        extra_conv_layers: Optional[int] = 0,
    ):
        super(MultiScaleDiscriminator, self).__init__()
        self.base_channels = base_channels
        self.scale_factor = scale_factor
        self.min_size = 16
        self.extra_conv_layers = extra_conv_layers
        self.max_n_scales = np.min([
            int(np.ceil(np.log(np.min(real_crop_size) * 1.0 / self.min_size) / np.log(self.scale_factor))),
            max_n_scales,
        ])
        self.nets = nn.ModuleList()
        for _ in range(self.max_n_scales):
            self.nets.append(self.make_net())


    def make_net(self):
        base_channels = self.base_channels
        net = []
        net += [
            nn.utils.spectral_norm(nn.Conv2d(1, base_channels, kernel_size=3, stride=1)),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, True),
        ]
        net += [
            nn.utils.spectral_norm(nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2)),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, True),
        ]
        net += [
            nn.utils.spectral_norm(nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, bias=True)),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, True),
        ]
        for _ in range(self.extra_conv_layers):
            net += [
                nn.utils.spectral_norm(nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, bias=True)),
                nn.BatchNorm2d(base_channels * 2),
                nn.LeakyReLU(0.2, True),
            ]
        net += nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(base_channels * 2, 1, kernel_size=1)),
            nn.Sigmoid(),
        )
        return nn.Sequential(*net)

    def forward(self, input_tensor: torch.Tensor, scale_weights: torch.Tensor) -> torch.Tensor:
        aggregated = self.nets[0](input_tensor) * scale_weights[0]
        map_size = aggregated.shape[2:]
        logger = logging.getLogger()
        for net, scale_weight, i in zip(self.nets[1:], scale_weights[1:], range(1, len(scale_weights))):
            downscaled = F.interpolate(
                input_tensor,
                scale_factor=self.scale_factor ** (-i),
                mode="bilinear",
                align_corners=False
            )
            try:
                result = net(downscaled)
            except KeyboardInterrupt:
                raise
            except Exception:
                print(f"Something went wrong in epoch {i}, While training.")
                print(f"epoch in net: {i}, downscaled_image shape: {downscaled.shape}")
                raise
            try:
                upscaled = F.interpolate(result, size=map_size, mode="bilinear", align_corners=False)
            except:
                logger.error(f"-------- ERROR --------\n input tensor shape: {input_tensor.shape}")
                logger.error(f"downscaled shape: {downscaled.shape}\n epoch: {i}")
                logger.error(f"result map shape: {result.shape}")
                raise
            aggregated += upscaled * scale_weight
        return aggregated



class RescaleBlock(nn.Module):
    """Rescale Block class"""

    def __init__(
        self,
        n_layers: int,
        scale: Optional[float] = 0.5,
        base_channels: Optional[int] = 64,
        use_bias: Optional[bool] = True,
    ):
        super(RescaleBlock, self).__init__()
        self.scale = scale
        self.conv_layers = [None] * n_layers
        in_channel_power = scale > 1
        out_channel_power = scale < 1
        i_range = range(n_layers) if scale < 1 else range(n_layers - 1, -1, -1)
        for i in i_range:
            self.conv_layers[i] = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.utils.spectral_norm(
                    nn.Conv2d(
                        in_channels=base_channels * 2 ** (i + in_channel_power),
                        out_channels=base_channels * 2 ** (i + out_channel_power),
                        kernel_size=3,
                        stride=1,
                        bias=use_bias,
                    )
                ),
                nn.BatchNorm2d(base_channels * 2 ** (i + out_channel_power)),
                nn.LeakyReLU(0.2, True),
            )
            self.add_module("conv_%d" % i, self.conv_layers[i])
        if scale > 1:
            self.conv_layers = self.conv_layers[::-1]
        self.max_pool = nn.MaxPool2d(2, 2)

    def forward(
        self,
        input_tensor: torch.Tensor,
        pyramid: Optional[torch.Tensor] = None,
        return_all_scales: Optional[bool] = False,
        skip: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        feature_map = input_tensor
        all_scales = []
        if return_all_scales:
            all_scales.append(feature_map)
        for i, conv_layer in enumerate(self.conv_layers):
            if self.scale > 1.0:
                feature_map = F.interpolate(feature_map, scale_factor=self.scale, mode="nearest")
            feature_map = conv_layer(feature_map)
            if skip:
                p = pyramid[-i - 2]
                min_t = min(feature_map.shape[-1], p.shape[-1])
                feature_map = feature_map[..., :min_t] + p[..., :min_t]
            if self.scale < 1.0:
                feature_map = self.max_pool(feature_map)
            if return_all_scales:
                all_scales.append(feature_map)
        return (feature_map, all_scales) if return_all_scales else (feature_map, None)



class RandomCrop(nn.Module):
    """Random Crop class"""

    def __init__(
        self,
        crop_size: Optional[List[int]] = None,
        return_pos: Optional[bool] = False,
        must_divide: Optional[float] = 4.0,
    ):
        super(RandomCrop, self).__init__()
        self.crop_size = crop_size
        self.must_divide = must_divide
        self.return_pos = return_pos

    # def forward(
    #     self,
    #     input_tensor: torch.Tensor,
    #     crop_size: Optional[List[int]] = None,
    # ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[int, int]]]:
    #     im_v_sz, im_h_sz = input_tensor.shape[2:]
    #     if crop_size is None:
    #         cr_v_sz, cr_h_sz = np.clip(self.crop_size, [0, 0], [im_v_sz - 1, im_h_sz - 1])
    #         cr_v_sz, cr_h_sz = np.uint32(
    #             np.floor(np.array([cr_v_sz, cr_h_sz]) * 1.0 / self.must_divide) * self.must_divide
    #         )
    #     else:
    #         cr_v_sz, cr_h_sz = crop_size
    #     top_left_v = np.random.randint(0, im_v_sz - cr_v_sz)
    #     top_left_h = np.random.randint(0, im_h_sz - cr_h_sz)
    #     out_tensor = input_tensor[:, :, top_left_v : top_left_v + cr_v_sz, top_left_h : top_left_h + cr_h_sz]
    #     return (out_tensor, (top_left_v, top_left_h)) if self.return_pos else out_tensor


    def forward(self, input_tensor, crop_size=None):
        im_v_sz, im_h_sz = input_tensor.shape[2:]
        if crop_size is None:
            cr_v_sz = int(np.floor(min(self.crop_size[0], im_v_sz) / self.must_divide) * self.must_divide)
            cr_h_sz = int(np.floor(min(self.crop_size[1], im_h_sz) / self.must_divide) * self.must_divide)
        else:
            cr_v_sz, cr_h_sz = crop_size

        # If crop equals full dim, start must be 0
        top_left_v = np.random.randint(0, max(1, im_v_sz - cr_v_sz))
        top_left_h = np.random.randint(0, max(1, im_h_sz - cr_h_sz))

        out_tensor = input_tensor[:, :, top_left_v:top_left_v + cr_v_sz, top_left_h:top_left_h + cr_h_sz]
        return (out_tensor, (top_left_v, top_left_h)) if self.return_pos else out_tensor




class GeoTransform(nn.Module):
    """Geo Transform class"""

    def __init__(self):
        super(GeoTransform, self).__init__()

    def forward(
        self,
        input_tensor: torch.Tensor,
        target_size: List[int],
        shifts: List[float],
    ) -> torch.Tensor:
        sz = input_tensor.shape
        theta = homography_based_on_top_corners_x_shift(shifts).to(input_tensor.device)
        pad = F.pad(
            input_tensor,
            (
                np.abs(int(np.ceil(sz[3] * shifts[0]))),
                np.abs(int(np.ceil(-sz[3] * shifts[1]))),
                0,
                0,
            ),
            "reflect",
        )
        target_size4d = torch.Size([pad.shape[0], pad.shape[1], target_size[0], target_size[1]])
        grid = homography_grid(theta.expand(pad.shape[0], -1, -1), target_size4d)
        return F.grid_sample(pad, grid, mode="bilinear", padding_mode="border", align_corners=True)

# ===========================================================================
# DiffATSM: Diffusion-based PostNet (Paper Section 3.2)
# ===========================================================================

class DiffusionPostNet(nn.Module):
    """
    Diffusion-based post-processing network for DiffATSM.
    
    DiffATSM Paper Section 3.2:
    "We integrate a diffusion probabilistic model as a post-processing network
     (PostNet) to enhance the fidelity of the mel spectrogram produced by the
     adaptive generator."
    
    Architecture: Modified non-causal WaveNet with:
    - Dilation = 1 (for mel spectrograms, not raw audio)
    - N residual blocks with gated mechanisms
    - Time step embedding via sinusoidal position encoding
    - Conditioning on reconstructed mel xrecon
    
    Section 4.2: "We set the diffusion time step to 100 with a linear noise
                  schedule ranging from 1x10^-4 to 0.05"
    """
    
    def __init__(
        self,
        mel_bins: int = 80,
        residual_channels: int = 256,
        n_residual_blocks: int = 20,
        time_emb_dim: int = 128,
    ):
        """
        Initialize Diffusion PostNet.
        
        Args:
            mel_bins: Number of mel frequency bins (default: 80 for LJSpeech)
            residual_channels: Hidden dimension for residual blocks (default: 256)
            n_residual_blocks: Number of residual blocks (default: 20)
            time_emb_dim: Time step embedding dimension (paper: 128)
        """
        super(DiffusionPostNet, self).__init__()
        
        self.mel_bins = mel_bins
        self.residual_channels = residual_channels
        self.n_residual_blocks = n_residual_blocks
        self.time_emb_dim = time_emb_dim
        
        # Initial projection: noisy mel xt -> residual_channels
        self.input_projection = nn.Sequential(
            nn.Conv1d(mel_bins, residual_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        
        # Time step encoder (Paper Equation 7 + FC layers)
        # "We transform the time step t into a 128-dimensional embedding vector
        #  using sinusoidal position encoding"
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, residual_channels),
            nn.SiLU(),  # Swish activation (Paper: Fatima and Pethe 2021)
            nn.Linear(residual_channels, residual_channels),
        )
        
        # Residual blocks (Paper Figure 4)
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(
                residual_channels=residual_channels,
                mel_bins=mel_bins,
            )
            for _ in range(n_residual_blocks)
        ])
        
        # Skip connection aggregation
        self.skip_projection = nn.Conv1d(
            residual_channels, residual_channels, kernel_size=1
        )
        
        # Output layers (Paper: "two Conv1x1 layers and ReLU")
        self.output_projection = nn.Sequential(
            nn.Conv1d(residual_channels, residual_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(residual_channels, mel_bins, kernel_size=1),
        )
    
    def forward(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        xrecon: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass to predict noise epsilon.
        
        Paper Algorithm 1:
        ϵθ(√ᾱt·x + √(1-ᾱt)·ϵ, t, x, xrecon)
        
        Args:
            xt: Noisy mel at diffusion step t, shape (B, mel_bins, T)
            t: Diffusion time step, shape (B,)
            xrecon: Reconstructed mel (condition), shape (B, mel_bins, T)
        Returns:
            Predicted noise epsilon, shape (B, mel_bins, T)
        """
        B, F, T_len = xt.shape
        
        # Step 1: Project noisy input
        x = self.input_projection(xt)  # (B, residual_channels, T)
        
        # Step 2: Time step embedding (Paper Equation 7)
        t_emb = self.get_time_embedding(t)  # (B, time_emb_dim)
        t_emb = self.time_mlp(t_emb)        # (B, residual_channels)
        
        # Add time embedding (broadcast across time dimension)
        # Paper: "time step embedding temb is added to it"
        x = x + t_emb.unsqueeze(-1)  # (B, residual_channels, T)
        
        # Step 3: Process through residual blocks with skip connections
        skip_sum = 0
        for block in self.residual_blocks:
            x, skip = block(x, xrecon)
            skip_sum = skip_sum + skip
        
        # Step 4: Aggregate skip connections
        # Paper: "aggregation of skip connections from all N residual layers"
        x = self.skip_projection(skip_sum)
        
        # Step 5: Output projection
        # Paper: "two Conv1x1 layers and ReLU activation functions"
        epsilon_pred = self.output_projection(x)
        
        return epsilon_pred
    
    def get_time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """
        Sinusoidal time step embedding (Paper Equation 7).
        
        Paper Equation 7:
        temb = [sin(10^(0×4/63)·t), ..., sin(10^(63×4/63)·t),
                cos(10^(0×4/63)·t), ..., cos(10^(63×4/63)·t)]
        
        Args:
            t: Time steps, shape (B,)
        Returns:
            Time embeddings, shape (B, time_emb_dim=128)
        """
        device = t.device
        half_dim = self.time_emb_dim // 2
        
        # Compute frequencies: 10^(k×4/63) for k=0,1,...,63
        exponent = torch.arange(half_dim, device=device).float()
        exponent = exponent * (4.0 / (half_dim - 1))
        freqs = torch.pow(10.0, exponent)
        
        # Compute embeddings
        t = t.float().unsqueeze(-1)  # (B, 1)
        args = t * freqs.unsqueeze(0)  # (B, half_dim)
        
        # Concatenate sin and cos
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        
        return embedding


class ResidualBlock(nn.Module):
    """
    Single residual block for DiffusionPostNet.
    
    Paper Figure 4: Each block contains:
    - Conv1x1 layers
    - Conv1d layer with dilation=1
    - Tanh and Sigmoid activation (gated mechanism)
    - Conditioning from xrecon via Conv1x1
    - Skip connection output
    """
    
    def __init__(self, residual_channels: int, mel_bins: int):
        super(ResidualBlock, self).__init__()
        
        # Main path
        self.conv1x1_pre = nn.Conv1d(residual_channels, residual_channels, 1)
        
        # Conv1d with dilation=1 (Paper: "We set the dilation to 1")
        self.conv1d = nn.Conv1d(
            residual_channels,
            residual_channels * 2,  # For Tanh and Sigmoid split
            kernel_size=3,
            padding=1,
            dilation=1
        )
        
        # Conditioning path for xrecon
        # Paper: "The reconstructed mel spectrogram xrecon passes through
        #         the Conv1x1 layers within all residual blocks"
        self.cond_conv = nn.Conv1d(mel_bins, residual_channels * 2, 1)
        
        # Output projections
        self.conv1x1_post = nn.Conv1d(residual_channels, residual_channels, 1)
        self.conv1x1_skip = nn.Conv1d(residual_channels, residual_channels, 1)
    
    def forward(
        self,
        x: torch.Tensor,
        xrecon: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through residual block.
        
        Args:
            x: Input features, shape (B, residual_channels, T)
            xrecon: Conditioning mel, shape (B, mel_bins, T)
        Returns:
            (residual_out, skip_out): Both shape (B, residual_channels, T)
        """
        residual = x
        
        # Main path
        x = self.conv1x1_pre(x)
        x = self.conv1d(x)
        
        # Conditioning
        # Paper: "a gated mechanism is employed"
        cond = self.cond_conv(xrecon)
        x = x + cond
        
        # Gated activation (Paper: "Tanh and Sigmoid activation functions")
        tanh_out, sigmoid_out = torch.chunk(x, 2, dim=1)
        x = torch.tanh(tanh_out) * torch.sigmoid(sigmoid_out)
        
        # Output projections
        skip = self.conv1x1_skip(x)
        x = self.conv1x1_post(x)
        
        # Residual connection
        residual_out = (x + residual) / np.sqrt(2.0)  # Normalize for stability
        
        return residual_out, skip

