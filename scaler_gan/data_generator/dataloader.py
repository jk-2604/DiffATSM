import glob
import os.path
import random
from typing import Optional, Tuple
import torchaudio
import torch
import torch.nn.functional as F

from scaler_gan.scalergan_utils.scalergan_utils import (
    files_to_list,
    load_audio_to_np,
    mel_spectrogram,
    norm_audio_like_hifi_gan,
    sample_segment,
    crop_mel,
)


class MelDataset(torch.utils.data.Dataset):
    """
    MelDataset with optional PPG (Phonetic PosteriorGram) extraction from HuBERT.
    
    DiffATSM Paper (Section 3.1.2):
    "PPGs are content features extracted from a pre-trained self-supervised speech
     representation model, HuBERT. The PPG features are then projected through a PPG
     pre-processing network (PreNet)."
    """

    def __init__(
        self,
        training_files: str,
        must_divide: float,
        segment_size: Optional[int] = 8192,
        n_fft: Optional[int] = 1024,
        num_mels: Optional[int] = 80,
        hop_size: Optional[int] = 256,
        win_size: Optional[int] = 1024,
        sampling_rate: Optional[int] = 22050,
        fmin: Optional[int] = 0,
        fmax: Optional[int] = 8000,
        shuffle: Optional[bool] = True,
        n_cache_reuse: Optional[int] = 1,
        fmax_loss: Optional[float] = None,
        seed: Optional[int] = 1234,
        # DiffATSM additions
        use_ppg: Optional[bool] = False,
        ppg_input_dim: Optional[int] = 768,
    ):
        """
        Init
        :param training_files: The training files, can be directory with wav files or text file with list of wav files
        :param must_divide: Division factor
        :param segment_size: The size of the output segment
        :param n_fft: The number of FFT coefficients
        :param num_mels: The number of mel coefficients
        :param hop_size: The hop size in samples
        :param win_size: The frame (window) size in samples
        :param sampling_rate: The wav file sampling rate
        :param fmin: The lower frequency boundary
        :param fmax: The higher frequency boundary
        :param shuffle: Flag for shuffling or not
        :param n_cache_reuse: Number of item to reuse
        :param fmax_loss: The higher frequency boundary loss
        :param seed: Seed factor
        :param use_ppg: Enable PPG extraction from HuBERT (DiffATSM)
        :param ppg_input_dim: HuBERT hidden dimension (768 for base, 1024 for large)
        """
        # Load list of files
        if os.path.isdir(training_files):
            self.audio_files = sorted(glob.glob(os.path.join(training_files, "*")))
        else:
            self.audio_files = files_to_list(training_files)
        
        if shuffle:
            random.shuffle(self.audio_files)
        
        self.must_divide = must_divide
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        
        # DiffATSM: PPG extraction
        self.use_ppg = use_ppg
        self.ppg_input_dim = ppg_input_dim
        
        if self.use_ppg:
            self._init_hubert()

    def _init_hubert(self):
        """
        Initialize HuBERT model for PPG extraction (DiffATSM Section 4.2).
        
        Paper: "PPG features were extracted from the 12th layer of a pre-trained
                Hsu et al. (2021) transformer encoder"
        """
        try:
            from transformers import HubertModel, Wav2Vec2FeatureExtractor
        except ImportError:
            raise ImportError(
                "transformers library required for PPG extraction. "
                "Install with: pip install transformers"
            )
        
        print("Loading HuBERT model for PPG extraction...")
        self.hubert_processor = Wav2Vec2FeatureExtractor.from_pretrained(
            "facebook/hubert-base-ls960"
        )
        self.hubert_model = HubertModel.from_pretrained(
            "facebook/hubert-base-ls960"
        )
        self.hubert_model.eval()
        
        # Freeze HuBERT (never trained, only used for feature extraction)
        for param in self.hubert_model.parameters():
            param.requires_grad = False
        
        # Move to CPU initially (will move to GPU in __getitem__ if available)
        self.hubert_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hubert_model = self.hubert_model.to(self.hubert_device)
        print(f"HuBERT loaded on {self.hubert_device}")

    def _extract_ppg(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract PPG features from audio using HuBERT 12th layer.
        
        Args:
            audio: (1, T_audio) waveform at 22050Hz
        Returns:
            ppg: (T_ppg, 768) PPG features from HuBERT 12th layer
        """
        # HuBERT expects 16kHz audio
        if self.sampling_rate != 16000:
            resampler = torchaudio.transforms.Resample(
                orig_freq=self.sampling_rate,
                new_freq=16000
            )
            audio_16k = resampler(audio)
        else:
            audio_16k = audio
        
        # Prepare input for HuBERT
        inputs = self.hubert_processor(
            audio_16k.squeeze(0).cpu().numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.hubert_device) for k, v in inputs.items()}
        
        # Extract 12th-layer features
        with torch.no_grad():
            outputs = self.hubert_model(**inputs, output_hidden_states=True)
            # output_hidden_states: tuple of 13 tensors (embedding + 12 transformer layers)
            ppg = outputs.hidden_states[12]  # (1, T_ppg, 768)
        
        return ppg.squeeze(0).cpu()  # (T_ppg, 768)

    def _align_ppg_to_mel(self, ppg: torch.Tensor, mel: torch.Tensor) -> torch.Tensor:
        """
        Align PPG temporal axis to mel temporal axis via interpolation.
        
        PPG from HuBERT: ~50 Hz (20ms frames at 16kHz)
        Mel spectrogram: hop_size=256 at 22050Hz = ~86 Hz (11.6ms frames)
        
        Args:
            ppg: (T_ppg, 768)
            mel: (1, F, T_mel)
        Returns:
            ppg_aligned: (T_mel, 768)
        """
        T_mel = mel.shape[-1]
        T_ppg = ppg.shape[0]
        
        if T_ppg == T_mel:
            return ppg
        
        # Interpolate PPG to match mel's temporal resolution
        # ppg: (T_ppg, 768) -> (1, 768, T_ppg) -> interpolate -> (1, 768, T_mel) -> (T_mel, 768)
        ppg_aligned = F.interpolate(
            ppg.T.unsqueeze(0),  # (1, 768, T_ppg)
            size=T_mel,
            mode='linear',
            align_corners=False
        ).squeeze(0).T  # (T_mel, 768)
        
        return ppg_aligned

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Get mel spectrogram and optionally PPG features.
        
        Returns:
            If use_ppg=False: mel  (1, F, T)
            If use_ppg=True:  (mel, ppg) where ppg is (T, 768)
        """
        filename = self.audio_files[index]
        
        # Load audio with caching
        if self._cache_ref_count == 0:
            audio, sampling_rate = load_audio_to_np(filename)

            if sampling_rate != self.sampling_rate:
                audio_tensor = torch.FloatTensor(audio).unsqueeze(0)
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sampling_rate,
                    new_freq=self.sampling_rate
                )
                audio_tensor = resampler(audio_tensor)
                audio = audio_tensor.squeeze(0).numpy()
                sampling_rate = self.sampling_rate

            audio = norm_audio_like_hifi_gan(audio)
            self.cached_wav = audio
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1

        # Convert to tensor and segment
        audio = torch.FloatTensor(audio).unsqueeze(0)
        audio = sample_segment(audio, self.segment_size)

        # Extract mel spectrogram
        mel = mel_spectrogram(
            audio,
            self.n_fft,
            self.num_mels,
            self.sampling_rate,
            self.hop_size,
            self.win_size,
            self.fmin,
            self.fmax,
            center=False,
        )
        mel = crop_mel(mel, self.must_divide)

        # Extract PPG if enabled
        if self.use_ppg:
            ppg = self._extract_ppg(audio)           # (T_ppg, 768)
            ppg = self._align_ppg_to_mel(ppg, mel)  # (T_mel, 768)
            return mel, ppg
        else:
            return mel

    def __len__(self) -> int:
        """
        The size of the dataset
        :return: The total samples in the training dataset
        """
        return len(self.audio_files)