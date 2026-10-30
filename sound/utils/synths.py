"""Library of synthesizer functions."""

import torch
import torch.nn as nn

from . import core


class ModalFIR(nn.Module):
    """Synthesize a modal FIR with exponentially decaying sinusoidal oscillators."""

    def __init__(self,
                 n_samples=64000,
                 sample_rate=16000,
                 amp_scale_fn=core.exp_sigmoid,
                 freq_scale_fn=core.frequencies_critical_bands,
                 hz_min=10.0,
                 hz_max=8000.0,
                 freq_scale='bark',
                 initial_bias=-1.5):
        super().__init__()
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.amp_scale_fn = amp_scale_fn
        self.freq_scale_fn = freq_scale_fn
        self.hz_min = hz_min
        self.hz_max = hz_max
        self.freq_scale = freq_scale
        self.initial_bias = initial_bias

    def get_controls(self, gains, frequencies, dampings):
        """Convert network output tensors into a dictionary of synthesizer controls.

        Args:
            gains: 3-D Tensor of synthesizer controls, of shape
                [batch, time, n_sinusoids].
            frequencies: 3-D Tensor of synthesizer controls, of shape
                [batch, time, n_sinusoids].
            dampings: 3-D Tensor of synthesizer controls, of shape
                [batch, time, n_sinusoids].

        Returns:
            controls: Dictionary of tensors of synthesizer controls.
        """
        # Scale the inputs.
        if self.amp_scale_fn is not None:
            # gains = 0.001 * self.amp_scale_fn(gains + self.initial_bias, exponent=3.0)
            gains = torch.softmax(gains, dim=-1)
            # dampings = 10.0 / self.amp_scale_fn(4.0 * dampings + self.initial_bias)
            # dampings = 0.05 * self.freq_scale_fn(dampings, hz_min=0.0, hz_max=100000.0)
            dampings = 100 * self.amp_scale_fn(dampings + self.initial_bias, exponent=4.0)

        if self.freq_scale_fn is not None:
            if len(frequencies.shape) == 2:
                frequencies = frequencies.unsqueeze(1)
            frequencies = self.freq_scale_fn(frequencies, hz_min=self.hz_min, hz_max=self.hz_max, scale=self.freq_scale)
        gains = core.remove_above_nyquist(frequencies,
                                          gains,
                                          self.sample_rate)
        return {'gains': gains,
                'frequencies': frequencies,
                'dampings': dampings}

    def get_signal(self, gains, frequencies, dampings, init_phases=None):
        """Synthesize audio with sinusoidal synthesizer from controls.

        Args:
            gains: Gains tensor of shape [batch, n_frames, n_sinusoids].
                Expects float32 that is strictly positive.
            frequencies: Tensor of shape [batch, n_frames, n_sinusoids].
                Expects float32 in Hertz that is strictly positive.
            dampings: Tensor of shape [batch, n_frames, n_sinusoids].
                Expects float32 in Hertz that is strictly positive.
            init_phases: Optional initial phases tensor of shape [batch, 1, n_sinusoids]
                or [batch, n_sinusoids]. If provided, shifts oscillator phases.

        Returns:
            signal: A tensor of exponentially decaying modal frequencies of shape [batch, n_samples].
        """
        device = gains.device
        # Create sample-wise envelopes.
        t = torch.arange(self.n_samples, dtype=torch.float32, device=device) / self.sample_rate
        t = t.unsqueeze(1)  # [n_samples, 1]
        amplitude_envelopes = gains * torch.exp(-dampings * t)
        frequency_envelopes = frequencies * torch.ones_like(amplitude_envelopes)
        signal = core.oscillator_bank(frequency_envelopes=frequency_envelopes,
                                       amplitude_envelopes=amplitude_envelopes,
                                       sample_rate=self.sample_rate,
                                       init_phases=init_phases)
        return signal