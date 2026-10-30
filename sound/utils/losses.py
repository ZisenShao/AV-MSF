"""Library of loss functions for audio synthesis optimization."""

import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa

# Constants from spectral_ops
LD_RANGE = 120.0  # dB


def safe_log(x, eps=1e-5):
    """Avoid taking the log of a non-positive number."""
    safe_x = torch.where(x <= eps, torch.tensor(eps, device=x.device, dtype=x.dtype), x)
    return torch.log(safe_x)

def diff(x, axis=-1):
  """Take the finite difference of a tensor along an axis.

  Args:
    x: Input tensor of any dimension.
    axis: Axis on which to take the finite difference.

  Returns:
    d: Tensor with size less than x by 1 along the difference dimension.

  Raises:
    ValueError: Axis out of range for tensor.
  """
  if axis >= len(x.shape):
    raise ValueError('Invalid axis index: %d for tensor with only %d axes.' %
                     (axis, len(x.shape)))
  return torch.diff(x, n=1, dim=axis)

def stft(audio, frame_size=2048, hop_length=256, pad_end=True):
    """Differentiable stft in tensorflow, computed in batch."""
    # Remove channel dim if present
    audio = audio.float()
    if len(audio.shape) == 3:
        audio = audio.squeeze(-1)
    
    if pad_end:
        n_samples = audio.shape[-1]
        n_frames = (n_samples + hop_length - 1) // hop_length
        target_length = (n_frames - 1) * hop_length + frame_size
        pad_len = target_length - n_samples
        if pad_len > 0:
            audio = F.pad(audio, (0, pad_len))
    window = torch.hann_window(frame_size, device=audio.device, dtype=audio.dtype)
    
    s = torch.stft(
        audio,
        n_fft=frame_size,
        hop_length=hop_length,
        win_length=frame_size,
        window=window,
        return_complex=True,
        center=False
    )
    
    s = s.transpose(1, 2)
    return s


def compute_mag(audio, size=2048, hop_length=256, pad_end=True):
    """Compute magnitude spectrogram."""
    mag = torch.abs(stft(audio, frame_size=size, hop_length=hop_length, pad_end=pad_end))
    return mag.float()

def compute_mel_from_mag(mag, 
                         sample_rate=16000, 
                         lo_hz=0.0, 
                         hi_hz=8000.0, 
                         bins=64, 
                         fft_size=2048):
    if not bins:
        bins = int(fft_size / 4) + 1
    mel_filter = librosa.filters.mel(
        sr=sample_rate,
        n_fft=fft_size,
        n_mels=bins,
        fmin=lo_hz,
        fmax=hi_hz,
        htk=True,
        norm=None
    )
    mel_filter = torch.from_numpy(mel_filter.T).to(mag.device, dtype=mag.dtype)
    mel = torch.matmul(mag, mel_filter)
    return mel

def amplitude_to_db(amplitude, amin=1e-20):
    """Converts amplitude to decibels."""
    log10 = lambda x: torch.log(x) / torch.log(torch.tensor(10.0, device=x.device, dtype=x.dtype))
    db = log10(torch.maximum(torch.tensor(amin, device=amplitude.device, dtype=amplitude.dtype), amplitude))
    db *= 20.0
    return db


def pad_or_trim_to_expected_length(vector,
                                   expected_len,
                                   pad_value=0,
                                   len_tolerance=20):
    """Make vector equal to the expected length.
    
    Args:
        vector: PyTorch tensor. Shape [batch_size, vector_length] or [vector_length,]
        expected_len: Expected length of vector.
        pad_value: Value to pad at end of vector.
        len_tolerance: Tolerance of difference between original and desired vector length.
    
    Returns:
        Vector with corrected length.
    """
    expected_len = int(expected_len)
    vector_len = int(vector.shape[-1])
    
    if abs(vector_len - expected_len) > len_tolerance:
        raise ValueError(f'Vector length: {vector_len} differs from expected length: {expected_len} '
                         f'beyond tolerance of: {len_tolerance}')
    
    is_1d = (len(vector.shape) == 1)
    vector = vector.unsqueeze(0) if is_1d else vector
    
    # Pad missing samples
    if vector_len < expected_len:
        n_padding = expected_len - vector_len
        vector = F.pad(vector, (0, n_padding), mode='constant', value=pad_value)
    # Trim samples
    elif vector_len > expected_len:
        vector = vector[..., :expected_len]
    
    # Remove temporary batch dimension
    vector = vector[0] if is_1d else vector
    return vector


def compute_loudness(audio,
                     sample_rate=16000,
                     frame_rate=250,
                     n_fft=2048,
                     range_db=LD_RANGE,
                     ref_db=20.7):
    """Perceptual loudness in dB, relative to white noise, amplitude=1.
    
    Differentiable PyTorch version.
    
    Args:
        audio: Tensor of shape [batch_size, audio_length] or [audio_length,].
        sample_rate: Audio sample rate in Hz.
        frame_rate: Rate of loudness frames in Hz.
        n_fft: FFT window size.
        range_db: Sets the dynamic range of loudness in decibels.
        ref_db: Sets the reference maximum perceptual loudness.
    
    Returns:
        Loudness in decibels. Shape [batch_size, n_frames] or [n_frames,].
    """
    if sample_rate % frame_rate != 0:
        raise ValueError(
            f'frame_rate: {frame_rate} must evenly divide sample_rate: {sample_rate}. '
            'For default frame_rate: 250Hz, suggested sample_rate: 16kHz or 48kHz'
        )
    
    # Make inputs tensors for PyTorch.
    audio = torch.tensor(audio).float()
    
    # Temporarily add batch dimension for single examples
    is_1d = (len(audio.shape) == 1)
    audio = audio.unsqueeze(0) if is_1d else audio
    
    # Take STFT
    hop_size = sample_rate // frame_rate
    s = stft(audio, frame_size=n_fft, hop_length=hop_size, pad_end=True)
    
    # Compute power.
    amplitude = torch.abs(s)
    power_db = amplitude_to_db(amplitude)
    
    # Perceptual weighting.
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    a_weighting = librosa.A_weighting(frequencies)
    a_weighting = torch.from_numpy(a_weighting).to(audio.device, dtype=audio.dtype)
    a_weighting = a_weighting.unsqueeze(0).unsqueeze(0)  # [1, 1, freq_bins]
    loudness = power_db + a_weighting
    
    # Set dynamic range.
    loudness = loudness - ref_db
    loudness = torch.maximum(loudness, torch.tensor(-range_db, device=audio.device, dtype=audio.dtype))
    
    # Average over frequency bins
    loudness = torch.mean(loudness, dim=-1)
    
    # Remove temporary batch dimension
    loudness = loudness[0] if is_1d else loudness
    
    # Compute expected length of loudness vector
    n_secs = audio.shape[-1] / float(sample_rate)
    expected_len = int(n_secs * frame_rate)
    
    # Pad with -range_db noise floor or trim vector
    loudness = pad_or_trim_to_expected_length(loudness, expected_len, -range_db)
    
    return loudness


def mean_difference(target, value, loss_type='L1', weights=None):
    """Common loss functions.
    
    Args:
        target: Target tensor.
        value: Value tensor.
        loss_type: One of 'L1', 'L2', or 'COSINE'.
        weights: A weighting mask for the per-element differences.
    
    Returns:
        The average loss.
    """
    difference = target - value
    weights = 1.0 if weights is None else weights
    loss_type = loss_type.upper()
    
    if loss_type == 'L1':
        return torch.mean(torch.abs(difference * weights))
    elif loss_type == 'L2':
        return torch.mean(difference ** 2 * weights)
    elif loss_type == 'COSINE':
        target_norm = F.normalize(target, dim=-1)
        value_norm = F.normalize(value, dim=-1)
        return 1.0 - torch.mean(torch.sum(target_norm * value_norm, dim=-1))
    else:
        raise ValueError(f'Loss type ({loss_type}) must be "L1", "L2", or "COSINE"')


class SpectralLoss(nn.Module):
    """Multi-scale spectrogram loss.
    
    This loss is the bread-and-butter of comparing two audio signals. It offers
    a range of options to compare spectrograms, many of which are redundant, but
    emphasize different aspects of the signal. By far, the most common comparisons
    are magnitudes (mag_weight) and log magnitudes (logmag_weight).
    """
    
    def __init__(self,
                 fft_sizes=(2048, 1024, 512, 256, 128, 64),
                 hop_length=256,
                 loss_type='L1',
                 mag_weight=1.0,
                 mel_weight=0.0,
                 delta_time_weight=0.0,
                 delta_freq_weight=0.0,
                 cumsum_freq_weight=0.0,
                 bin_time_weight=0.0,
                 max_power_weight=0.0,
                 logmag_weight=0.0,
                 logmel_weight=0.0,
                 loudness_weight=0.0,
                 cutoff=None):
        """Constructor, set loss weights of various components.
        
        Args:
            fft_sizes: Compare spectrograms at each of this list of fft sizes. Each
                spectrogram has a time-frequency resolution trade-off based on fft size,
                so comparing multiple scales allows multiple resolutions.
            loss_type: One of 'L1', 'L2', or 'COSINE'.
            mag_weight: Weight to compare linear magnitudes of spectrograms. Core
                audio similarity loss. More sensitive to peak magnitudes than log
                magnitudes.
            mel_weight: Weight to compare mel spectrograms.
            delta_time_weight: Weight to compare the first finite difference of
                spectrograms in time. Emphasizes changes of magnitude in time, such as
                at transients.
            delta_freq_weight: Weight to compare the first finite difference of
                spectrograms in frequency. Emphasizes changes of magnitude in frequency,
                such as at the boundaries of a stack of harmonics.
            cumsum_freq_weight: Weight to compare the cumulative sum of spectrograms
                across frequency for each slice in time. Similar to a 1-D Wasserstein
                loss, this hopefully provides a non-vanishing gradient to push two
                non-overlapping sinusoids towards each other.
            bin_time_weight: Weight to compare the sum of spectrograms across frequency.
            max_power_weight: Weight to compare the max power of spectrograms across frequency.
            logmag_weight: Weight to compare log magnitudes of spectrograms. Core
                audio similarity loss. More sensitive to quiet magnitudes than linear
                magnitudes.
            logmel_weight: Weight to compare log mel spectrograms.
            loudness_weight: Weight to compare the overall perceptual loudness of two
                signals. Very high-level loss signal that is a subset of mag and
                logmag losses.
            cutoff: Optional number of samples to use from the start of the audio.
                If None (default), use the full audio. If set, only the first `cutoff`
                samples are used for loss computation.
        """
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_length = hop_length
        self.loss_type = loss_type
        self.mag_weight = mag_weight
        self.mel_weight = mel_weight
        self.delta_time_weight = delta_time_weight
        self.delta_freq_weight = delta_freq_weight
        self.cumsum_freq_weight = cumsum_freq_weight
        self.bin_time_weight = bin_time_weight
        self.max_power_weight = max_power_weight
        self.logmag_weight = logmag_weight
        self.logmel_weight = logmel_weight
        self.loudness_weight = loudness_weight
        self.cutoff = cutoff
        
        # Create spectrogram operations for each fft size
        self.spectrogram_ops = []
        for size in self.fft_sizes:
            spectrogram_op = functools.partial(compute_mag, size=size, hop_length=self.hop_length)
            self.spectrogram_ops.append(spectrogram_op)
    
    def forward(self, target_audio, audio, weights=None):
        """Compute spectral loss.
        
        Args:
            target_audio: Target audio tensor of shape [batch, samples].
            audio: Synthesized audio tensor of shape [batch, samples].
            weights: Optional weights for the loss.
        
        Returns:
            Scalar loss value.
        """
        # Apply cutoff if specified
        if self.cutoff is not None:
            target_audio = target_audio[..., :self.cutoff]
            audio = audio[..., :self.cutoff]
        
        loss = 0.0
        
        # Compute loss for each fft size
        for i, loss_op in enumerate(self.spectrogram_ops):
            fft_size = self.fft_sizes[i]
            target_mag = loss_op(target_audio)
            value_mag = loss_op(audio)
            
            # Add magnitude loss
            if self.mag_weight > 0:
                loss += self.mag_weight * mean_difference(
                    target_mag, value_mag, self.loss_type, weights=weights)
            
            if self.delta_time_weight > 0:
                target = diff(target_mag, axis=1)
                value = diff(value_mag, axis=1)
                loss += self.delta_time_weight * mean_difference(
                    target, value, self.loss_type, weights=weights)
            
            if self.delta_freq_weight > 0:
                target = diff(target_mag, axis=2)
                value = diff(value_mag, axis=2)
                loss += self.delta_freq_weight * mean_difference(
                    target, value, self.loss_type, weights=weights)
            
            if self.cumsum_freq_weight > 0:
                target = torch.cumsum(target_mag, dim=-1)
                value = torch.cumsum(value_mag, dim=-1)
                loss += self.cumsum_freq_weight * mean_difference(
                    target, value, self.loss_type, weights=weights)
            
            if self.bin_time_weight > 0:
                target = torch.sum(target_mag, dim=-1)
                value = torch.sum(value_mag, dim=-1)
                loss += self.bin_time_weight * mean_difference(
                    target, value, self.loss_type, weights=weights)
            
            if self.max_power_weight > 0:
                target = safe_log(torch.max(target_mag, dim=2)[0])
                value = safe_log(torch.max(value_mag, dim=2)[0])
                loss += self.max_power_weight * mean_difference(
                    target, value, self.loss_type, weights=weights)
            
            # Add logmagnitude loss, reusing spectrogram
            if self.logmag_weight > 0:
                target = safe_log(target_mag)
                value = safe_log(value_mag)
                loss += self.logmag_weight * mean_difference(
                    target, value, self.loss_type, weights=weights)
            
            # Mel spectrogram losses
            if self.mel_weight > 0 or self.logmel_weight > 0:
                target_mel = compute_mel_from_mag(target_mag, lo_hz=2.0, bins=None, fft_size=fft_size)
                value_mel = compute_mel_from_mag(value_mag, lo_hz=2.0, bins=None, fft_size=fft_size)
                if self.mel_weight > 0:
                    loss += self.mel_weight * mean_difference(
                        target_mel, value_mel, self.loss_type, weights=weights)
                if self.logmel_weight > 0:
                    target_logmel = safe_log(target_mel)
                    value_logmel = safe_log(value_mel)
                    loss += self.logmel_weight * mean_difference(
                        target_logmel, value_logmel, self.loss_type, weights=weights)
        
        # Loudness loss (computed once, not per fft size)
        if self.loudness_weight > 0:
            target = compute_loudness(target_audio, n_fft=2048)
            value = compute_loudness(audio, n_fft=2048)
            loss += self.loudness_weight * mean_difference(
                target, value, self.loss_type, weights=weights)
        
        return loss