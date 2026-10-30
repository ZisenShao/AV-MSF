"""dB mel-spectrogram and waveform comparison for two audio files."""

import os

import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np


def plot_audio(
    audio_path1: str,
    audio_path2: str,
    output_path: str = None,
    label1: str = "Audio 1",
    label2: str = "Audio 2",
    sr: int = None,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 128,
    dpi: int = 150,
):
    """Plot two audio files as dB mel spectrograms, their difference, and waveforms."""
    y1, sr = librosa.load(audio_path1, sr=sr)
    y2, _ = librosa.load(audio_path2, sr=sr)

    min_len = min(len(y1), len(y2))
    y1 = y1[:min_len]
    y2 = y2[:min_len]

    mel1 = librosa.feature.melspectrogram(
        y=y1,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    mel2 = librosa.feature.melspectrogram(
        y=y2,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    ref_val = max(mel1.max(), mel2.max())
    mel_db1 = librosa.power_to_db(mel1, ref=ref_val)
    mel_db2 = librosa.power_to_db(mel2, ref=ref_val)
    mel_diff = mel_db2 - mel_db1

    vmin = min(mel_db1.min(), mel_db2.min())
    vmax = max(mel_db1.max(), mel_db2.max())

    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    gs = GridSpec(
        2,
        6,
        figure=fig,
        width_ratios=[1, 0.03, 1, 0.03, 1, 0.03],
        height_ratios=[1.5, 1],
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax1_cbar = fig.add_subplot(gs[0, 1])
    img1 = librosa.display.specshow(
        mel_db1,
        x_axis="time",
        y_axis="mel",
        sr=sr,
        hop_length=hop_length,
        ax=ax1,
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
    )
    ax1.set_title(label1, fontweight="bold", fontsize=12)
    ax1.set_ylabel("Frequency (Hz)")
    ax1.set_xlabel("Time (s)")
    fig.colorbar(img1, cax=ax1_cbar, format="%+2.0f dB")

    ax2 = fig.add_subplot(gs[0, 2])
    ax2_cbar = fig.add_subplot(gs[0, 3])
    img2 = librosa.display.specshow(
        mel_db2,
        x_axis="time",
        y_axis="mel",
        sr=sr,
        hop_length=hop_length,
        ax=ax2,
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
    )
    ax2.set_title(label2, fontweight="bold", fontsize=12)
    ax2.set_ylabel("")
    ax2.set_xlabel("Time (s)")
    fig.colorbar(img2, cax=ax2_cbar, format="%+2.0f dB")

    ax3 = fig.add_subplot(gs[0, 4])
    ax3_cbar = fig.add_subplot(gs[0, 5])
    diff_max = max(abs(mel_diff.min()), abs(mel_diff.max()))
    img3 = librosa.display.specshow(
        mel_diff,
        x_axis="time",
        y_axis="mel",
        sr=sr,
        hop_length=hop_length,
        ax=ax3,
        cmap="RdBu_r",
        vmin=-diff_max,
        vmax=diff_max,
    )
    ax3.set_title(f"Diff ({label2} - {label1})", fontweight="bold", fontsize=12)
    ax3.set_ylabel("")
    ax3.set_xlabel("Time (s)")
    cbar = fig.colorbar(img3, cax=ax3_cbar, format="%+2.0f dB")
    cbar.set_label("Red = Missing", fontsize=9)

    ax_wave = fig.add_subplot(gs[1, :])
    times = np.arange(min_len) / sr
    ax_wave.plot(times, y1, alpha=0.7, linewidth=0.6, label=label1, color="#1f77b4")
    ax_wave.plot(times, y2, alpha=0.7, linewidth=0.6, label=label2, color="#ff7f0e")
    ax_wave.set_title("Waveform Comparison", fontweight="bold", fontsize=12)
    ax_wave.set_xlabel("Time (s)")
    ax_wave.set_ylabel("Amplitude")
    ax_wave.set_ylim([-1, 1])
    ax_wave.legend(loc="upper right", fontsize=9)
    ax_wave.grid(True, alpha=0.3)

    fig.suptitle(
        f"{label1} vs {label2} | Duration: {min_len / sr:.2f}s | SR: {sr} Hz",
        fontsize=11,
    )

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {output_path}")

    plt.close()
    return fig
