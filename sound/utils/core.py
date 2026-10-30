"""Library of functions for differentiable digital signal processing (DDSP)."""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import hilbert, sosfiltfilt, butter
import pandas as pd
import scipy.linalg
from scipy import signal


def torch_float32(x):
    """Ensure array/tensor is a float32 torch.Tensor."""
    if isinstance(x, torch.Tensor):
        return x.float()
    else:
        return torch.tensor(x, dtype=torch.float32)


# Math -------------------------------------------------------------------------
def safe_divide(numerator, denominator, eps=1e-7):
    """Avoid dividing by zero by adding a small epsilon."""
    safe_denominator = torch.where(denominator == 0.0, torch.tensor(eps, device=denominator.device, dtype=denominator.dtype), denominator)
    return numerator / safe_denominator


def safe_log(x, eps=1e-5):
    """Avoid taking the log of a non-positive number."""
    safe_x = torch.where(x <= eps, torch.tensor(eps, device=x.device, dtype=x.dtype), x)
    return torch.log(safe_x)


def soft_limit(x, x_min=0.0, x_max=1.0):
    """Softly limits inputs to the range [x_min, x_max]."""
    return F.softplus(x) + x_min - F.softplus(x - (x_max - x_min))


# Unit Conversions -------------------------------------------------------------
def hz_to_bark(hz):
    """From Tranmuller (1990, https://asa.scitation.org/doi/10.1121/1.399849)."""
    hz = torch_float32(hz)
    return 26.81 / (1.0 + (1960.0 / hz)) - 0.53


def bark_to_hz(bark):
    """From Tranmuller (1990, https://asa.scitation.org/doi/10.1121/1.399849)."""
    bark = torch_float32(bark)
    return 1960.0 / (26.81 / (bark + 0.53) - 1.0)


def hz_to_mel(hz):
    """From Young et al. 'The HTK book', Chapter 5.4."""
    hz = torch_float32(hz)
    return 2595.0 * torch.log10(1.0 + hz / 700.0)


def mel_to_hz(mel):
    """From Young et al. 'The HTK book', Chapter 5.4."""
    mel = torch_float32(mel)
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def hz_to_erb(hz):
    """Equivalent Rectangular Bandwidths (ERB) from Moore & Glasberg (1996)."""
    hz = torch_float32(hz)
    return 0.108 * hz + 24.7


# Scaling functions ------------------------------------------------------------
def exp_sigmoid(x, exponent=10.0, max_value=2.0, threshold=1e-7):
    """Exponentiated Sigmoid pointwise nonlinearity.

    Bounds input to [threshold, max_value] with slope given by exponent.

    Args:
        x: Input tensor.
        exponent: In nonlinear regime (away from x=0), the output varies by this
            factor for every change of x by 1.0.
        max_value: Limiting value at x=inf.
        threshold: Limiting value at x=-inf. Stablizes training when outputs are
            pushed to 0.

    Returns:
        A tensor with pointwise nonlinearity applied.
    """
    x = torch_float32(x)
    return max_value * torch.sigmoid(x) ** np.log(exponent) + threshold


def _add_depth_axis(freqs, depth=1):
    """Turns [batch, time, sinusoids*depth] to [batch, time, sinusoids, depth]."""
    freqs = freqs.unsqueeze(-1)
    # Unpack sinusoids dimension.
    n_batch, n_time, n_combined, _ = freqs.shape
    n_sinusoids = n_combined // depth
    return freqs.view(n_batch, n_time, n_sinusoids, depth)


def frequencies_critical_bands(freqs,
                               depth=1,
                               depth_scale=10.0,
                               bandwidth_scale=1.0,
                               hz_min=20.0,
                               hz_max=8000.0,
                               scale='bark'):
    """Center frequencies scaled on mel or bark scale, with ranges given by erb.

    Args:
        freqs: Neural network outputs, [batch, time, n_sinusoids * depth] or
            [batch, time, n_sinusoids, depth]. It is essentially an offset to 
            scale ERB bandwidths of mel or bark frequencies centers.
        depth: If freqs is 3-D, the number of sigmoid components per a sinusoid to
            unroll from the last dimension.
        depth_scale: The degree by which to reduce the influence of each subsequent
            dimension of depth.
        bandwidth_scale: Multiplier (to ERB) for the range of each sinusoid.
        hz_min: Lowest frequency to consider.
        hz_max: Highest frequency to consider.
        scale: Critical frequency scaling, must be either 'mel' or 'bark'.

    Returns:
        A tensor of frequencies in hertz [batch, time, n_sinusoids].
    """
    device = freqs.device
    
    if len(freqs.shape) == 3:
        # Add depth: [B, T, N*D] -> [B, T, N, D]
        freqs = _add_depth_axis(freqs, depth)
    else:
        depth = int(freqs.shape[-1])

    # Figure out the number of sinusoids.
    n_sinusoids = freqs.shape[-2]

    # Initialize the critical frequencies and bandwidths.
    if scale == 'bark':
        # Bark.
        bark_min = hz_to_bark(hz_min)
        bark_max = hz_to_bark(hz_max)
        linear_bark = torch.linspace(bark_min, bark_max, n_sinusoids, device=device)
        f_center = bark_to_hz(linear_bark)
        f_center = torch.tensor(f_center, dtype=torch.float32, device=device)
    else:
        # Mel.
        mel_min = hz_to_mel(hz_min)
        mel_max = hz_to_mel(hz_max)
        linear_mel = torch.linspace(mel_min, mel_max, n_sinusoids, device=device)
        f_center = mel_to_hz(linear_mel)
        f_center = torch.tensor(f_center, dtype=torch.float32, device=device)

    # Bandwidths given by equivalent rectangular bandwidth (ERB).
    bw = hz_to_erb(f_center)

    # Probs: [B, T, N, D]
    modifier = torch.tanh(freqs)
    # [D] 10 ** (-0) = 1, 10 ** (-1) = 0.1, 10 ** (-2) = 0.01, ...
    depth_modifier = depth_scale ** (-torch.arange(depth, dtype=torch.float32, device=device))
    # print(depth, depth_modifier, modifier)

    # weight freqs by depth. [B, T, N, D] -> [B, T, N].
    modifier = torch.sum(modifier * depth_modifier.view(1, 1, 1, -1), dim=-1)

    f_modifier = bandwidth_scale * bw.view(1, 1, -1) * modifier
    return soft_limit(f_center + f_modifier, hz_min, hz_max) # [B, T, N]


# Harmonic Synthesizer ---------------------------------------------------------
def remove_above_nyquist(frequency_envelopes, amplitude_envelopes, sample_rate=44100):
    """Set amplitudes for oscillators above nyquist to 0.

    Args:
        frequency_envelopes: Sample-wise oscillator frequencies (Hz). Shape
            [batch_size, n_samples, n_sinusoids]. The real frequencies are in after frequencies_critical_bands.
        amplitude_envelopes: Sample-wise oscillator amplitude. Shape [batch_size,
            n_samples, n_sinusoids].
        sample_rate: Sample rate in samples per a second.

    Returns:
        amplitude_envelopes: Sample-wise filtered oscillator amplitude.
            Shape [batch_size, n_samples, n_sinusoids].
    """
    frequency_envelopes = torch_float32(frequency_envelopes)
    amplitude_envelopes = torch_float32(amplitude_envelopes)

    amplitude_envelopes = torch.where(
        frequency_envelopes >= sample_rate / 2.0,
        torch.zeros_like(amplitude_envelopes), amplitude_envelopes)
    return amplitude_envelopes


def oscillator_bank(frequency_envelopes,
                    amplitude_envelopes,
                    sample_rate=44100,
                    sum_sinusoids=True,
                    init_phases=None):
    """Generates audio from sample-wise frequencies for a bank of oscillators.

    Args:
        frequency_envelopes: Sample-wise oscillator frequencies (Hz). Shape
            [batch_size, n_samples, n_sinusoids].
        amplitude_envelopes: Sample-wise oscillator amplitude. Shape [batch_size,
            n_samples, n_sinusoids].
        sample_rate: Sample rate in samples per a second.
        sum_sinusoids: Add up audio from all the sinusoids.
        init_phases: Optional initial phases for each oscillator. Shape 
            [batch_size, 1, n_sinusoids] or [batch_size, n_sinusoids].
            If provided, shifts all accumulated phases by this offset.

    Returns:
        wav: Sample-wise audio. Shape [batch_size, n_samples, n_sinusoids] if
            sum_sinusoids=False, else shape is [batch_size, n_samples].
    """
    frequency_envelopes = torch_float32(frequency_envelopes) # [B, T, N]
    amplitude_envelopes = torch_float32(amplitude_envelopes) # [B, T, N]

    # Don't exceed Nyquist.
    amplitude_envelopes = remove_above_nyquist(frequency_envelopes,
                                               amplitude_envelopes,
                                               sample_rate)

    # Angular frequency, Hz -> radians per sample.
    omegas = frequency_envelopes * (2.0 * np.pi)  # rad / sec [B, T, N]
    omegas = omegas / float(sample_rate)  # rad / sample [B, T, N], [B, 0, N] is the start phase of each modal sinusoid at first sample's time

    # Accumulate phase and synthesize.
    # Shift cumsum so phases[0] = 0 (not omega[0]), aligning with extract_params
    # where the phase at t=0 is init_phase, not init_phase + omega[0]
    phases = torch.cumsum(omegas, dim=1) - omegas  # [B, T, N], phases[0] = 0
    
    # Add initial phases if provided.
    if init_phases is not None:
        init_phases = torch_float32(init_phases)
        if len(init_phases.shape) == 2:
            init_phases = init_phases.unsqueeze(1)  # [B, N] -> [B, 1, N]
        phases = phases + init_phases  # broadcast [B, 1, N] to [B, T, N]

    # Convert to waveforms.
    wavs = torch.cos(phases)
    audio = amplitude_envelopes * wavs  # [mb, n_samples, n_sinusoids]
    if sum_sinusoids:
        audio = torch.sum(audio, dim=-1)  # [mb, n_samples]
    return audio

def get_fft_size(frame_size: int, ir_size: int, power_of_2: bool = True) -> int:
    """Calculate final size for efficient FFT.

    Args:
    frame_size: Size of the audio frame.
    ir_size: Size of the convolving impulse response.
    power_of_2: Constrain to be a power of 2. If False, allow other 5-smooth
      numbers. TPU requires power of 2, while GPU is more flexible.

    Returns:
    fft_size: Size for efficient FFT.
    """
    convolved_frame_size = ir_size + frame_size - 1
    if power_of_2:
        # Next power of 2.
        fft_size = int(2**np.ceil(np.log2(convolved_frame_size)))
    else:
        fft_size = int(fftpack.helper.next_fast_len(convolved_frame_size))
    return fft_size


def fft_convolve(audio: torch.Tensor,
                 impulse_response: torch.Tensor,
                 padding: str = 'same',
                 delay_compensation: int = -1) -> torch.Tensor:
    """Filter audio with impulse responses using FFT convolution.

    Args:
        audio: Input audio. Tensor of shape [batch, audio_timesteps].
        impulse_response: Finite impulse response to convolve. Can either be a 2-D
            Tensor of shape [batch, ir_size], or a 3-D Tensor of shape [batch,
            ir_frames, ir_size]. A 2-D tensor will apply a single linear
            time-invariant filter to the audio.
        padding: Either 'valid' or 'same'. For 'same' the final output to be the
            same size as the input audio (audio_timesteps). For 'valid' the audio is
            extended to include the tail of the impulse response.
        delay_compensation: Samples to crop from start of output audio to compensate
            for group delay of the impulse response. If delay_compensation is less
            than 0 it defaults to automatically calculating a constant group delay.

    Returns:
        audio_out: Convolved audio. Tensor of shape [batch, audio_timesteps].
    """
    audio = torch_float32(audio)
    impulse_response = torch_float32(impulse_response)
    
    # Handle 2-D case (time-invariant filter) - most common for noise filtering
    ir_shape = impulse_response.shape
    if len(ir_shape) == 2:
        # Simple FFT convolution for time-invariant filter
        batch_size, audio_size = audio.shape
        ir_size = ir_shape[-1]
        
        # Compute FFT size
        fft_size = get_fft_size(audio_size, ir_size, power_of_2=True)
        
        # Pad and FFT
        audio_fft = torch.fft.rfft(audio, n=fft_size)
        ir_fft = torch.fft.rfft(impulse_response, n=fft_size)
        
        # Multiply in frequency domain
        convolved_fft = audio_fft * ir_fft
        convolved = torch.fft.irfft(convolved_fft, n=fft_size)
        
        # Crop output based on padding mode
        if padding == 'same':
            # Compensate for group delay
            if delay_compensation < 0:
                delay_compensation = (ir_size - 1) // 2
            return convolved[:, delay_compensation:delay_compensation + audio_size]
        else:  # 'valid'
            return convolved[:, :audio_size + ir_size - 1]
    
    # Handle 3-D case (time-varying filter) - frame-based processing
    elif len(ir_shape) == 3:
        batch_size_ir, n_ir_frames, ir_size = ir_shape
        batch_size, audio_size = audio.shape
        
        # Validate batch sizes match
        if batch_size != batch_size_ir:
            raise ValueError('Batch size of audio ({}) and impulse response ({}) must '
                             'be the same.'.format(batch_size, batch_size_ir))
        
        # Cut audio into frames with padding
        frame_size = int(np.ceil(audio_size / n_ir_frames))
        hop_size = frame_size
        
        # Pad audio to ensure it can be evenly split
        pad_size = frame_size * n_ir_frames - audio_size
        if pad_size > 0:
            audio = F.pad(audio, (0, pad_size))
        
        # Reshape to frames [batch, n_frames, frame_size]
        audio_frames = audio.view(batch_size, n_ir_frames, frame_size)
        
        # Pad and FFT
        fft_size = get_fft_size(frame_size, ir_size, power_of_2=True)
        audio_fft = torch.fft.rfft(audio_frames, n=fft_size)
        ir_fft = torch.fft.rfft(impulse_response, n=fft_size)
        
        # Multiply FFTs
        audio_ir_fft = audio_fft * ir_fft
        
        # Take IFFT
        audio_frames_out = torch.fft.irfft(audio_ir_fft, n=fft_size)
        
        # Overlap-add 
        output_size = hop_size * n_ir_frames + fft_size - hop_size
        audio_out = torch.zeros(batch_size, output_size, device=audio.device)
        for i in range(n_ir_frames):
            start = i * hop_size
            end = start + fft_size
            audio_out[:, start:end] += audio_frames_out[:, i, :]
        
        # Crop and compensate for delay
        return crop_and_compensate_delay(audio_out, audio_size, ir_size, padding,
                                         delay_compensation)
    else:
        raise ValueError(f"Impulse response must be 2-D or 3-D, got {len(ir_shape)}-D")

def crop_and_compensate_delay(audio: torch.Tensor, audio_size: int, ir_size: int,
                              padding: str,
                              delay_compensation: int) -> torch.Tensor:
    """Crop audio output from convolution to compensate for group delay.

    Args:
    audio: Audio after convolution. Tensor of shape [batch, time_steps].
    audio_size: Initial size of the audio before convolution.
    ir_size: Size of the convolving impulse response.
    padding: Either 'valid' or 'same'. For 'same' the final output to be the
      same size as the input audio (audio_timesteps). For 'valid' the audio is
      extended to include the tail of the impulse response (audio_timesteps +
      ir_timesteps - 1).
    delay_compensation: Samples to crop from start of output audio to compensate
      for group delay of the impulse response. If delay_compensation < 0 it
      defaults to automatically calculating a constant group delay of the
      windowed linear phase filter from frequency_impulse_response().

    Returns:
    Tensor of cropped and shifted audio.

    Raises:
    ValueError: If padding is not either 'valid' or 'same'.
    """
    # Crop the output.
    if padding == 'valid':
        crop_size = ir_size + audio_size - 1
    elif padding == 'same':
        crop_size = audio_size
    else:
        raise ValueError('Padding must be \'valid\' or \'same\', instead '
                         'of {}.'.format(padding))

    # Compensate for the group delay of the filter by trimming the front.
    # For an impulse response produced by frequency_impulse_response(),
    # the group delay is constant because the filter is linear phase.
    total_size = int(audio.shape[-1])
    crop = total_size - crop_size
    start = ((ir_size - 1) // 2 -
           1 if delay_compensation < 0 else delay_compensation)
    end = crop - start
    return audio[:, start:-end]

def frequency_impulse_response(magnitudes: torch.Tensor, 
                                window_size: int = 0) -> torch.Tensor:
    """Get windowed impulse responses using the frequency sampling method.

    Follows the approach in:
    https://ccrma.stanford.edu/~jos/sasp/Windowing_Desired_Impulse_Response.html

    Args:
        magnitudes: Frequency transfer curve. Float32 Tensor of shape [batch,
            n_frames, n_frequencies] or [batch, n_frequencies]. The frequencies of the
            last dimension are ordered as [0, f_nyqist / (n_frequencies -1), ...,
            f_nyquist], where f_nyquist is (sample_rate / 2). Automatically splits the
            audio into equally sized frames to match frames in magnitudes.
        window_size: Size of the window to apply in the time domain. If window_size
            is less than 1, it defaults to the impulse_response size.

    Returns:
        impulse_response: Time-domain FIR filter of shape [batch, window_size].

    Raises:
        ValueError: If window size is larger than fft size.
    """
    # Get the IR (zero-phase form).
    magnitudes = torch_float32(magnitudes)
    magnitudes_complex = torch.complex(magnitudes, torch.zeros_like(magnitudes))
    impulse_response = torch.fft.irfft(magnitudes_complex)

    # Window and put in causal form.
    impulse_response = apply_window_to_impulse_response(impulse_response,
                                                        window_size)

    return impulse_response

def apply_window_to_impulse_response(impulse_response: torch.Tensor,
                                     window_size: int = 0,
                                     causal: bool = False) -> torch.Tensor:
    """Apply a window to an impulse response and put in causal form.

    Args:
        impulse_response: A series of impulse responses frames to window [batch, ir_size].
        window_size: Size of the window to apply in the time domain. If window_size
            is less than 1, it defaults to the impulse_response size.
        causal: Impulse response input is in causal form (peak in the middle).

    Returns:
        impulse_response: Windowed impulse response in causal form, with last
            dimension cropped to window_size if window_size is greater than 0 and less
            than ir_size.
    """
    impulse_response = torch_float32(impulse_response)
    device = impulse_response.device

    # If IR is in causal form, put it in zero-phase form.
    if causal:
        impulse_response = torch.fft.fftshift(impulse_response, dim=-1)

    # Get a window for better time/frequency resolution than rectangular.
    # Window defaults to IR size, cannot be bigger.
    ir_size = int(impulse_response.shape[-1])
    if (window_size <= 0) or (window_size > ir_size):
        window_size = ir_size
    window = torch.hann_window(window_size, device=device)

    # Zero pad the window and put in in zero-phase form.
    padding = ir_size - window_size
    if padding > 0:
        half_idx = (window_size + 1) // 2
        window = torch.cat([window[half_idx:],
                            torch.zeros(padding, device=device),
                            window[:half_idx]], dim=0)
    else:
        window = torch.fft.fftshift(window, dim=-1)

    # Apply the window, to get new IR (both in zero-phase form).
    # Broadcast window to match impulse_response shape
    window = window.expand_as(impulse_response)
    impulse_response = window * impulse_response.real if impulse_response.is_complex() else window * impulse_response

    # Put IR in causal form and trim zero padding.
    if padding > 0:
        first_half_start = (ir_size - (half_idx - 1)) + 1
        second_half_end = half_idx + 1
        impulse_response = torch.cat([impulse_response[..., first_half_start:],
                                      impulse_response[..., :second_half_end]],
                                     dim=-1)
    else:
        impulse_response = torch.fft.fftshift(impulse_response, dim=-1)

    return impulse_response

def frequency_filter(audio: torch.Tensor, 
                     magnitudes: torch.Tensor, 
                     window_size: int = 0,
                     padding: str = 'same') -> torch.Tensor:
    """Filter audio with a finite impulse response filter.

    Args:
        audio: Input audio. Tensor of shape [batch, audio_timesteps].
        magnitudes: Frequency transfer curve. Float32 Tensor of shape
            [batch, n_frequencies]. The frequencies of the last dimension are ordered
            as [0, f_nyquist / (n_frequencies -1), ..., f_nyquist], where f_nyquist is
            (sample_rate / 2). Automatically splits the audio into equally sized frames
            to match frames in magnitudes.
        window_size: Size of the window to apply in the time domain. If window_size
            is less than 1, it is set as the default (n_frequencies).
        padding: Either 'valid' or 'same'. For 'same' the final output to be the
            same size as the input audio (audio_timesteps). For 'valid' the audio is
            extended to include the tail of the impulse response (audio_timesteps +
            window_size - 1).

    Returns:
        Filtered audio. Tensor of shape
            [batch, audio_timesteps + window_size - 1] ('valid' padding) or shape
            [batch, audio_timesteps] ('same' padding).
    """
    impulse_response = frequency_impulse_response(magnitudes, window_size=window_size)
    return fft_convolve(audio, impulse_response, padding=padding)



# Noise Modeling ----------------------------------------------------------------
def time_varying_noise(magnitudes: torch.Tensor,
                       n_samples: int,
                       n_fft: int = 1024,
                       hop_length: int = 256,
                       amp_scale_fn=None,
                       phases: torch.Tensor = None) -> torch.Tensor:
    """Synthesize noise with time-varying spectral envelope via STFT overlap-add.
    
    This provides much higher expressiveness than time-invariant frequency_filter
    by allowing learnable magnitude per time-frequency bin, and optionally
    learnable phase for capturing structured phase relationships.
    
    Args:
        magnitudes: Learnable spectral magnitudes. Float32 Tensor of shape
            [batch, n_frames, n_freqs] where n_freqs = n_fft // 2 + 1.
            These are raw network outputs that will be scaled by amp_scale_fn.
        n_samples: Desired output length in samples.
        n_fft: FFT size for STFT synthesis.
        hop_length: Hop size between frames.
        amp_scale_fn: Optional function to scale magnitudes (e.g., exp_sigmoid).
            If None, uses exp_sigmoid with default parameters.
        phases: Optional learnable phase tensor of shape [batch, n_frames, n_freqs].
            If provided, these raw values are wrapped to [-pi, pi] and used as phase.
            If None, random phase is generated.
    
    Returns:
        Synthesized noise tensor of shape [batch, n_samples].
    """
    magnitudes = torch_float32(magnitudes)
    device = magnitudes.device
    batch, n_frames, n_freqs = magnitudes.shape
    
    # Apply amplitude scaling (ensure positive magnitudes)
    if amp_scale_fn is None:
        amp_scale_fn = exp_sigmoid
    mag_scaled = amp_scale_fn(magnitudes)  # [batch, n_frames, n_freqs]
    
    # Determine phase: learnable or random
    if phases is not None:
        # Wrap learnable phase to [-pi, pi] using atan2-based wrapping
        # This wrapping preserves gradients better than modulo
        phases = torch_float32(phases)
        phase = torch.atan2(torch.sin(phases), torch.cos(phases))  # Wrap to [-pi, pi]
    else:
        # Generate random phase per time-frequency bin (uniform [0, 2*pi])
        phase = torch.rand(batch, n_frames, n_freqs, device=device) * 2.0 * np.pi
    
    # Construct complex STFT: magnitude * exp(i * phase)
    # Shape: [batch, n_freqs, n_frames] for torch.istft
    real = mag_scaled * torch.cos(phase)
    imag = mag_scaled * torch.sin(phase)
    stft_complex = torch.complex(real, imag)
    stft_complex = stft_complex.permute(0, 2, 1)  # [batch, n_freqs, n_frames]
    
    # Create window for ISTFT
    window = torch.hann_window(n_fft, device=device)
    
    # Inverse STFT with overlap-add
    # length parameter ensures output is at least n_samples
    audio = torch.istft(
        stft_complex,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=True,
        length=n_samples
    )
    
    return audio  # [batch, n_samples]


def compute_noise_frames(n_samples: int, n_fft: int = 1024, hop_length: int = 256) -> int:
    """Compute the number of STFT frames needed for a given audio length.
    
    Args:
        n_samples: Desired output audio length in samples.
        n_fft: FFT size.
        hop_length: Hop size between frames.
    
    Returns:
        Number of frames required.
    """
    # For center=True STFT, the formula is:
    return (n_samples + hop_length - 1) // hop_length + 1


# Metrics ----------------------------------------------------------------
def compute_envelope_dist(pred, gt):
    """Compute envelope distance using Hilbert transform.
    
    Args:
        pred: Predicted audio, 1D numpy array [n_samples]
        gt: Ground truth audio, 1D numpy array [n_samples]
    
    Returns:
        RMSE distance between amplitude envelopes (scalar)
    """
    pred_env = np.abs(hilbert(pred))
    gt_env = np.abs(hilbert(gt))
    return np.sqrt(np.mean((gt_env - pred_env)**2))


# Modal Extraction -----------------------------------------------------------
def extract_params(x, n=128, sr=44100, n_fft=1024, hop_length=64):
    """
    Extract n modal parameters (frequency, damping, gain, phase) from one audio input.
    
    Algorithm:
    1. Use linear scale STFT to get real physical spectrogram.
    2. For each frequency bin, find peak gain across time and apply noise filter.
    3. Select top n modes by energy (sum of squared spectrogram values).
    4. Fit envelope for each mode.

    Args:
        x: Input audio, 1D numpy array [n_samples]
        n: Number of modes to extract
        sr: Sample rate
        n_fft: FFT size
        hop_length: Hop size between frames

    Returns:
        freqs: Modal frequencies [n_modes]
        dampings: Modal damping ratios [n_modes]
        gains: Modal gains [n_modes]
        phases: Modal phases [n_modes]
    """
    import librosa
    from scipy.optimize import curve_fit
    
    n_samples = len(x)
    
    # Step 1: Compute STFT and get real-valued linear spectrogram
    stft_data = librosa.stft(x, n_fft=n_fft, hop_length=hop_length, window="hann", center=True)
    n_bins, n_frames = stft_data.shape
    spectrogram = np.zeros((n_bins, n_samples), dtype=np.float32)
    times = np.arange(n_samples) / sr

    for k in range(n_bins):
        stft_k = np.zeros_like(stft_data)
        stft_k[k, :] = stft_data[k, :]
        y_k = librosa.istft(
            stft_k,
            hop_length=hop_length,
            window="hann",
            center=True,
            length=n_samples,
        )
        spectrogram[k, :] = y_k

    # Step 2: For each frequency bin, find peak gain across time and apply noise filter.
    freq_bins = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    bin_freqs = []
    bin_energies = []
    bin_gains = []
    bin_peak_times = []
    bin_indices = []
    
    for bin_idx in range(n_bins):
        time_idx = np.argmax(np.abs(spectrogram[bin_idx, :]))
        energy = np.sum(spectrogram[bin_idx, :] ** 2)
        spec_vals = spectrogram[bin_idx, :]
        envelope = np.abs(hilbert(spec_vals))
        gain = np.max(envelope)
        bin_freqs.append(freq_bins[bin_idx])
        bin_energies.append(energy)
        bin_gains.append(gain)
        bin_peak_times.append(times[time_idx])
        bin_indices.append(bin_idx)

    bin_freqs = np.array(bin_freqs)
    bin_energies = np.array(bin_energies)
    bin_gains = np.array(bin_gains)
    bin_peak_times = np.array(bin_peak_times)
    bin_indices = np.array(bin_indices, dtype=int)

    # Noise Filter 1: peak time > 0.1s is likely noise
    time_mask = bin_peak_times <= 0.1
    
    # Noise Filter 2: EDC based filter
    # if EDC at 1 second is still >= 10% of EDC at 0 second, it's noise
    decay_mask = np.ones(len(bin_energies), dtype=bool)
    for idx in range(len(bin_energies)):
        if not time_mask[idx]:
            continue
        
        spec_vals = spectrogram[bin_indices[idx], :]
        analytic = hilbert(spec_vals)
        envelope = np.abs(analytic)
        
        # EDC from 0 to end
        edc = np.cumsum(envelope[::-1]**2)[::-1]
        edc_0 = edc[0]
        
        # Check EDC at 1 second
        edc_1s = edc[int(1.0 * sr)]
        if edc_1s >= edc_0 / 10:
            decay_mask[idx] = False
    
    combined_mask = time_mask & decay_mask
    bin_freqs = bin_freqs[combined_mask]
    bin_energies = bin_energies[combined_mask]
    bin_gains = bin_gains[combined_mask]
    bin_peak_times = bin_peak_times[combined_mask]
    bin_indices = bin_indices[combined_mask]

    # Step 3: Select top n modes by energy
    n = min(n, len(bin_energies))
    if n == 0:
        raise ValueError("No modes found")
    top_indices = np.argsort(bin_energies)[-n:][::-1]
    selected_freqs = bin_freqs[top_indices]
    selected_energies = bin_energies[top_indices]
    selected_gains = bin_gains[top_indices]
    selected_times = bin_peak_times[top_indices]
    selected_bins = bin_indices[top_indices]
    
    # Step 4: Fit envelope for each mode
    final_freqs = np.zeros(n)
    final_dampings = np.zeros(n)
    final_gains = np.zeros(n)
    final_phases = np.zeros(n)
    final_energies = selected_energies.copy()  # Copy energies from selection step
    
    for i in range(n):
        freq = selected_freqs[i]
        gain = selected_gains[i]
        t_peak = selected_times[i]
        bin_idx = selected_bins[i]
        
        spec_vals = spectrogram[bin_idx, :]
        analytic = hilbert(spec_vals)
        envelope = np.abs(analytic)
        
        peak_sample = int(t_peak * sr)
        peak_envelope = envelope[peak_sample]
        threshold = peak_envelope / 10
        
        end_offset = np.where(envelope[peak_sample:] < threshold)[0][0]
        end_sample = peak_sample + end_offset

        # damping init - integral of f(t) from t_peak_env to t_EDC_threshold: d = integral of f(t) dt / T
        # where f(t) = ln(env(t_i) / env(t_{i+1})) / (t_{i+1} - t_i)
        edc_ratio = 200.0
        edc_max_time = 1.0
        edc_end_sample = peak_sample + int(edc_max_time * sr)
        envelope_segment = envelope[peak_sample:edc_end_sample]
        edc = np.cumsum(envelope_segment[::-1]**2)[::-1]
        edc_peak = edc[0]
        edc_threshold = edc_peak / edc_ratio
        edc_end_offset = np.where(edc < edc_threshold)[0][0]
        
        d_integral = 0.0
        for j in range(edc_end_offset - 1):
            env_j = envelope_segment[j]
            env_j1 = envelope_segment[j + 1]
            if env_j > 1e-10 and env_j1 > 1e-10:
                # f(t_j) * dt = ln(env[j]/env[j+1])
                d_integral += np.log(env_j / env_j1)
        
        T_total = edc_end_offset / sr
        damp_init = d_integral / T_total
        damp_init = np.clip(damp_init, 1.0, 500.0)
        
        # The envelope follows: env(t) = g * exp(-d * t)
        # Taking log: log(env) = log(g) - d * t
        start_sample = 0
        y_fit = spec_vals[start_sample:end_sample]
        t_fit = np.arange(len(y_fit)) / sr
        env_fit = np.abs(hilbert(y_fit))
        valid_mask = env_fit > 1e-10
        if np.sum(valid_mask) > 10:
            log_env = np.log(env_fit[valid_mask])
            t_valid = t_fit[valid_mask]
            
            # Linear regression: log(env) = log(g) - d * t
            # Solve: [1, t] @ [log(g), -d]^T = log(env)
            A = np.vstack([np.ones_like(t_valid), t_valid]).T
            coeffs, _, _, _ = np.linalg.lstsq(A, log_env, rcond=None)
            
            log_g_est = coeffs[0]
            d_est = -coeffs[1]
            
            d_est = np.clip(d_est, 1.0, 1.5 * damp_init)
            g_est = np.exp(log_g_est)
        else:
            d_est = damp_init
            g_est = gain
        
        # Phase Estimation via correlation with damped sinusoid
        best_phase = 0.0
        best_corr = -np.inf
        for phi_test in np.linspace(-np.pi, np.pi, 32):
            template = g_est * np.exp(-d_est * t_fit) * np.cos(2 * np.pi * freq * t_fit + phi_test)
            corr = np.dot(y_fit, template)
            if corr > best_corr:
                best_corr = corr
                best_phase = phi_test
        
        final_freqs[i] = freq
        final_gains[i] = g_est
        final_dampings[i] = d_est
        final_phases[i] = best_phase

    # sort by energy descending (mode 0 = highest energy)
    sort_idx = np.argsort(final_energies)[::-1]
    final_freqs = final_freqs[sort_idx]
    final_dampings = final_dampings[sort_idx]
    final_gains = final_gains[sort_idx]
    final_phases = final_phases[sort_idx]
    final_energies = final_energies[sort_idx]
    
    # Reconstruct audio from extracted parameters
    t = np.arange(n_samples) / sr
    reconstructed = np.zeros(n_samples)
    
    for i in range(n):
        if final_gains[i] > 0:
            mode = final_gains[i] * np.exp(-final_dampings[i] * t) * np.cos(
                2 * np.pi * final_freqs[i] * t + final_phases[i]
            )
            reconstructed += mode
    
    return final_freqs, final_dampings, final_gains, final_phases, final_energies, reconstructed

def extract_params_multi_audio(audio_list, n=400, sr=44100, n_fft=1024, hop_length=64):
    """
    Extract modal parameters from multiple audio inputs with shared frequency and damping.
    
    Algorithm:
    1. Call extract_params for each audio to extract n modes per audio.
    2. Find intersection of modes across all audios by exact frequency matching.
    3. Select top n_modes modes sorted by sum of energies across all audios.
    
    For final modes: damping = average across all audios, each audio has its own gain, phase.
    
    Args:
        audio_list: List of audio arrays
        n: Number of modes to extract per audio
        sr: Sample rate
        n_fft: FFT size
        hop_length: Hop length for STFT
    
    Returns:
        final_freqs: Array of shape [n_modes] containing shared frequencies
        final_dampings: Array of shape [n_modes] containing averaged dampings
        final_gains: Array of shape [n_audios, n_modes] containing per-audio gains
        final_phases: Array of shape [n_audios, n_modes] containing per-audio phases
    """
    n_audios = len(audio_list)
    
    # Step 1: Extract params for each audio
    all_params = []
    for i, audio in enumerate(audio_list):
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy().flatten()
        freqs, dampings, gains, phases, energies, _ = extract_params(audio, n=n, sr=sr, n_fft=n_fft, hop_length=hop_length)
        all_params.append({'freqs': freqs, 'dampings': dampings, 'gains': gains, 'phases': phases, 'energies': energies})
    
    # Step 2: Find intersection of modes
    freq_sets = [set(all_params[i]['freqs']) for i in range(n_audios)]
    common_freqs = freq_sets[0]
    for i in range(1, n_audios):
        common_freqs = common_freqs.intersection(freq_sets[i])

    common_freqs = sorted(list(common_freqs))
    m = len(common_freqs)
    if m <= 32:
        raise ValueError(f"Only {m} common modes found across {n_audios} audios, which is less than 32")
    
    # m intersection of modes, each {freq, damping(audio1...audioN), gain(audio1...audioN), phase(audio1...audioN), sum_energy_all_audios}
    mode_matches = []
    for freq in common_freqs:
        dampings = []
        gains = []
        phases = []
        energies = []
        
        for audio_idx in range(n_audios):
            params = all_params[audio_idx]
            idx = np.where(params['freqs'] == freq)[0][0]
            dampings.append(params['dampings'][idx])
            gains.append(params['gains'][idx])
            phases.append(params['phases'][idx])
            energies.append(params['energies'][idx])

        if len(gains) == n_audios:
            mode_matches.append({
                'freq': freq,
                'dampings': np.array(dampings),
                'gains': np.array(gains),
                'phases': np.array(phases),
                'energy_sum': np.sum(energies)
            })
    
    # Step 3: Select top n_modes by energy_sum, then build final arrays
    mode_options = [32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512]
    valid_options = [opt for opt in mode_options if opt < m]
    n_modes = max(valid_options)
    print(f"[extract_params_multi_audio] Selected n_modes={n_modes}")

    mode_matches_sorted = sorted(mode_matches, key=lambda x: x['energy_sum'], reverse=True)
    selected_modes = mode_matches_sorted[:n_modes]
    
    final_freqs = np.array([m['freq'] for m in selected_modes])
    final_dampings = np.array([np.mean(m['dampings']) for m in selected_modes])
    final_gains = np.column_stack([m['gains'] for m in selected_modes])  # [n_audios, n_modes]
    final_phases = np.column_stack([m['phases'] for m in selected_modes])  # [n_audios, n_modes]
    
    sort_idx = np.argsort(final_freqs)
    final_freqs = final_freqs[sort_idx]
    final_dampings = final_dampings[sort_idx]
    final_gains = final_gains[:, sort_idx]
    final_phases = final_phases[:, sort_idx]

    print(f"[extract_params_multi_audio] Final: {n_modes} modes, freq range [{final_freqs.min():.1f}, {final_freqs.max():.1f}] Hz")
    print(f"[extract_params_multi_audio] Damping range [{final_dampings.min():.2f}, {final_dampings.max():.2f}]")
    
    return final_freqs, final_dampings, final_gains, final_phases, n_modes
