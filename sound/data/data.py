"""
Sound Dataset for Impact Sound Field Reconstruction.
"""

import json
import os
import argparse
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import librosa
from scipy.signal import butter, sosfilt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class Parser:
    """
    Parser for impact_sound.json.
    
    The JSON file contains:
    - object_id: Identifier for the object
    - entries: List of impact sound entries
        - impact_id: Index of the impact
        - position_3dgs: 3D position in 3DGS coordinate space
        - mic_waveform: Path to recorded impact sound
        - force_waveform: Path to hammer IR waveform
        - force_newtons: Peak force magnitude in Newtons
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize the parser.
        
        Args:
            data_dir: Path to the dataset directory containing impact_sound.json
        """
        self.data_dir = data_dir
        mapping_path = os.path.join(data_dir, "impact_sound.json")
        
        with open(mapping_path, 'r') as f:
            self.data = json.load(f)
        
        self.object_id = self.data.get('object_id')
        self.entries = self.data.get('entries')
        
        print(f"[Parser] Loaded {len(self.entries)} impact entries for object {self.object_id}")
    
    def __len__(self) -> int:
        return len(self.entries)
    
    def get_entry(self, idx: int) -> Dict[str, Any]:
        """Get a single impact entry by index."""
        return self.entries[idx]
    
    def get_all_positions(self) -> np.ndarray:
        """Get all impact positions as a numpy array [N, 3]."""
        positions = [entry['position_3dgs'] for entry in self.entries]
        return np.array(positions, dtype=np.float32)
    
    def get_all_mic_paths(self) -> List[str]:
        """Get all microphone waveform paths."""
        return [entry['mic_waveform'] for entry in self.entries]
    
    def get_all_force_paths(self) -> List[str]:
        """Get all force waveform paths (hammer IR)."""
        return [entry['force_waveform'] for entry in self.entries]

    def get_all_force_newtons(self) -> List[float]:
        """Get all force magnitudes in Newtons."""
        return [entry['force_newtons'] for entry in self.entries]


class SoundDataset(Dataset):
    """
    Dataset for impact sound field reconstruction.
    
    Audio Processing:
    - "none": raw audio
    - "peak": peak normalization (each audio normalized by its own peak)
    - "rms": RMS normalization (each audio normalized to target RMS)
    - "force": force normalization + global peak normalization
      (divide by force, then normalize ALL audio by the maximum peak across all samples)
    
    - tool=False: impact sound as is
    - tool=True: deconvolve impact tool's impulse response
    
    Each sample contains:
    - impact_id: Index of the impact
    - position: 3D position of the impact in 3DGS coordinate space
    - audio: Processed audio waveform
    """
    
    def __init__(
        self,
        parser: Parser,
        sample_rate: int = 44100,
        tool: bool = False,
        norm: str = "force",
        audio_duration: Optional[float] = None,
        split: str = 'train',
        train_ratio: float = 0.8,
        hz_min: float = 20.0,
        hz_max: float = 20000.0,
    ):
        """
        Initialize the sound dataset.
        
        Args:
            parser: Parser instance for the dataset
            sample_rate: Target sample rate for audio
            tool: If True, deconvolve hammer IR (Force.wav) from mic.wav
            norm: Normalization mode - "none", "peak", "rms", or "force"
            audio_duration: Duration of audio to load in seconds (None for full audio)
            split: 'train' or 'val'/'test'
            train_ratio: Ratio of data to use for training (used only as a fallback
                when partition.json is missing)
            hz_min: Minimum frequency for bandpass filter (Hz)
            hz_max: Maximum frequency for bandpass filter (Hz)
        """
        self.parser = parser
        self.sample_rate = sample_rate
        self.tool = tool
        self.norm = norm.lower()
        self.audio_duration = audio_duration
        self.split = split
        self.train_ratio = train_ratio
        self.hz_min = hz_min
        self.hz_max = hz_max

        n_total = len(self.parser)

        partition_path = os.path.join(self.parser.data_dir, "partition.json")
        if os.path.exists(partition_path):
            with open(partition_path, 'r') as f:
                partition = json.load(f)
            train_indices = list(partition['train_indices'])
            val_indices = list(partition['test_indices'])
            print(f"[SoundDataset] Loaded partition.json: "
                  f"{len(train_indices)} train, {len(val_indices)} test")
        else:
            n_train = int(n_total * train_ratio)
            all_positions = self.parser.get_all_positions()  # [N, 3]
            train_indices = self._farthest_point_sampling(all_positions, n_train)
            val_indices = list(set(range(n_total)) - set(train_indices))
            print(f"[SoundDataset] partition.json not found at {partition_path}, "
                  f"using FPS split: {n_train} train, {n_total - n_train} test")

        if split == 'train':
            self.indices = np.array(train_indices)
        else:
            self.indices = np.array(val_indices)
        
        print(f"[SoundDataset] {split} split: {len(self.indices)} samples: {self.indices}")
        print(f"[SoundDataset] norm={self.norm}, tool={tool}")
                
        if self.norm == "force":
            self._global_peak = self._force_global_peak()
                
        # Cache for loaded audio
        self._audio_cache: Dict[int, torch.Tensor] = {}
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def _farthest_point_sampling(self, positions: np.ndarray, n_train: int) -> List[int]:
        """
        Select n_train points using Farthest Point Sampling (FPS).
        """
        n_total = len(positions)
        centroid = positions.mean(axis=0)
        distances_to_centroid = np.linalg.norm(positions - centroid, axis=1)
        first_idx = int(np.argmin(distances_to_centroid))
        selected = [first_idx]
        min_distances = np.full(n_total, np.inf)
        
        for _ in range(n_train - 1):
            last_selected = selected[-1]
            dist_to_last = np.linalg.norm(positions - positions[last_selected], axis=1)
            min_distances = np.minimum(min_distances, dist_to_last)
            min_distances[selected] = -np.inf
            next_idx = int(np.argmax(min_distances))
            selected.append(next_idx)
        
        return selected
    
    def _load_audio_raw(self, path: str) -> np.ndarray:
        """Load raw audio file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Audio file not found: {path}")
        
        waveform, sr = librosa.load(
            path, 
            sr=self.sample_rate,
            mono=True
        )
        return waveform
    
    def _peak_normalize(self, waveform: np.ndarray) -> np.ndarray:
        """Apply peak normalization (normalize by the audio's own peak amplitude)."""
        peak = np.abs(waveform).max()
        if peak > 0:
            waveform = waveform / peak
        return waveform

    def _rms_normalize(self, waveform: np.ndarray, target_rms: float = 0.01) -> np.ndarray:
        """Apply RMS (Root Mean Square) normalization."""
        rms = np.sqrt(np.mean(waveform ** 2))
        if rms > 0:
            waveform = waveform * (target_rms / rms)
        return waveform
    
    def _force_normalize(self, waveform: np.ndarray, force_newtons: float) -> np.ndarray:
        """Normalize waveform by force magnitude."""
        if force_newtons > 0:
            waveform = waveform / force_newtons
        return waveform
    
    def _bandpass_filter(self, waveform: np.ndarray, low_hz: float = 20.0, high_hz: float = 20000.0, order: int = 4) -> np.ndarray:
        """Apply bandpass filter to keep only audible frequencies (20Hz - 20kHz)."""
        nyquist = self.sample_rate / 2
        # Ensure high_hz doesn't exceed Nyquist
        high_hz = min(high_hz, nyquist - 1)
        if low_hz >= high_hz:
            return waveform
        sos = butter(order, [low_hz, high_hz], btype='band', fs=self.sample_rate, output='sos')
        return sosfilt(sos, waveform)
    
    def _force_global_peak(self) -> float:
        """
        Preload all audio samples (across ALL entries, not just current split),
        apply force normalization, and compute the global max peak amplitude.
        
        This is used for force + global peak normalization.
        """
        all_peaks = []
        n_total = len(self.parser)
        
        for idx in range(n_total):
            entry = self.parser.get_entry(idx)
            mic_path = entry['mic_waveform']
            force_newtons = entry['force_newtons']
            
            waveform = self._load_audio_raw(mic_path)
            waveform = self._force_normalize(waveform, force_newtons)
            
            peak = np.abs(waveform).max()
            all_peaks.append(peak)
        
        return max(all_peaks)
    
    def _deconvolve_hammer_ir(
        self, 
        mic_waveform: np.ndarray, 
        hammer_ir: np.ndarray,
        regularization: float = 1e-5
    ) -> np.ndarray:
        """
        Deconvolve hammer IR from recorded sound to get clean object IR.
        
        Uses Wiener deconvolution in frequency domain.
        After deconvolution, re-aligns the output so that the impact time
        matches the original mic_waveform impact time.
        """
        
        # Transform from time domain to frequency domain
        L = len(mic_waveform) + len(hammer_ir)
        fft_len = int(2 ** np.ceil(np.log2(L)))
        
        mic_fft = np.fft.rfft(mic_waveform, n=fft_len)
        hammer_fft = np.fft.rfft(hammer_ir, n=fft_len)
        
        hammer_power = np.abs(hammer_fft) ** 2
        reg = regularization * np.max(hammer_power)
        
        object_ir_fft = mic_fft * np.conj(hammer_fft) / (hammer_power + reg)
        object_ir = np.fft.irfft(object_ir_fft, n=fft_len)
        object_ir = object_ir[:len(mic_waveform)]
        
        return object_ir
    
    def _process_audio(
        self, 
        mic_path: str, 
        force_path: str,
        force_newtons: float
    ) -> torch.Tensor:
        """
        Load and process audio based on configuration.
        
        Processing logic:
        1. Align peak (impact) to the beginning of the audio
        2. Truncate or pad to specified duration
        3. Apply normalization/deconvolution
        """
        # Load mic waveform
        waveform = self._load_audio_raw(mic_path)
        hammer_ir = self._load_audio_raw(force_path)
        
        waveform = self._bandpass_filter(waveform, low_hz=self.hz_min, high_hz=self.hz_max)
        hammer_ir = self._bandpass_filter(hammer_ir, low_hz=self.hz_min, high_hz=self.hz_max)
        
        # Find peak position (impact time) and align to beginning
        mic_peak_idx = np.argmax(np.abs(waveform))
        hammer_peak_idx = np.argmax(np.abs(hammer_ir))
        waveform = waveform[mic_peak_idx:]
        hammer_ir = hammer_ir[hammer_peak_idx:]

        # Truncate or pad to specified duration
        if self.audio_duration is not None:
            target_samples = int(self.audio_duration * self.sample_rate)
            if len(waveform) > target_samples:
                waveform = waveform[:target_samples]
            elif len(waveform) < target_samples:
                padding = np.zeros(target_samples - len(waveform))
                waveform = np.concatenate([waveform, padding])
            if len(hammer_ir) > target_samples:
                hammer_ir = hammer_ir[:target_samples]
            elif len(hammer_ir) < target_samples:
                padding = np.zeros(target_samples - len(hammer_ir))
                hammer_ir = np.concatenate([hammer_ir, padding])
        
        # Deconvolve hammer IR if requested
        if self.tool:
            waveform = self._deconvolve_hammer_ir(waveform, hammer_ir)

        # Normalization
        if self.norm == "peak":
            waveform = self._peak_normalize(waveform)
        elif self.norm == "rms":
            waveform = self._rms_normalize(waveform)
        elif self.norm == "force":
            waveform = self._force_normalize(waveform, force_newtons)
            waveform = waveform / self._global_peak
        else:
            pass
        
        return torch.from_numpy(waveform).float()
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single impact sound sample.
        
        Returns:
            Dictionary containing:
            - impact_id: Original impact ID
            - position: 3D position tensor [3]
            - audio: Processed audio waveform tensor [num_samples]
            - sample_rate: Audio sample rate
        """
        original_idx = self.indices[idx]
        entry = self.parser.get_entry(original_idx)
        # Position
        position = torch.tensor(entry['position_3dgs'], dtype=torch.float32)
        
        # Load and process audio (with caching)
        if original_idx not in self._audio_cache:
            audio = self._process_audio(
                mic_path=entry['mic_waveform'],
                force_path=entry['force_waveform'],
                force_newtons=entry['force_newtons']
            )
            self._audio_cache[original_idx] = audio
        
        audio = self._audio_cache[original_idx]
        
        return {
            'impact_id': entry['impact_id'],
            'position': position,
            'audio': audio,
            'sample_rate': self.sample_rate,
        }
    
    def get_all_positions(self) -> torch.Tensor:
        """Get all impact positions in the dataset split."""
        positions = []
        for idx in self.indices:
            entry = self.parser.get_entry(idx)
            positions.append(entry['position_3dgs'])
        return torch.tensor(positions, dtype=torch.float32)


if __name__ == "__main__":
    """Test the sound dataset."""
    
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--data_dir", type=str, default="sound/data/outputs/80")
    arg_parser.add_argument("--sample_rate", type=int, default=44100)
    arg_parser.add_argument("--tool", action="store_true")
    arg_parser.add_argument("--norm", type=str, default="peak")
    arg_parser.add_argument("--audio_duration", type=float, default=3)

    args = arg_parser.parse_args()
    
    # Create parser and datasets
    parser = Parser(data_dir=args.data_dir)
    
    train_dataset = SoundDataset(
        parser=parser,
        sample_rate=args.sample_rate,
        tool=args.tool,
        norm=args.norm,
        audio_duration=args.audio_duration,
        split='train'
    )
    val_dataset = SoundDataset(
        parser=parser,  
        sample_rate=args.sample_rate,
        tool=args.tool,
        norm=args.norm,
        audio_duration=args.audio_duration,
        split='val'
    )
    
    print(f"\nTrain: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Test sample
    sample = train_dataset[0]
    print(f"\nSample: position={sample['position']}, audio shape={sample['audio'].shape}")
    print(f"Audio range: [{sample['audio'].min():.4f}, {sample['audio'].max():.4f}]")
    print(f"Audio RMS: {torch.sqrt(torch.mean(sample['audio']**2)):.4f}")
    
    # Test batch
    loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    batch = next(iter(loader))
    print(f"\nBatch: positions={batch['positions'].shape}, audio={batch['audio'].shape}")