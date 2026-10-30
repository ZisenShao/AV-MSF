"""
Modal Sound Field Trainer.
"""

import os
import io
import math
import random
import json
import yaml
from dataclasses import asdict, dataclass, field
from typing import List, Optional
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import tyro
import scipy.io.wavfile as wavfile
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
import cdpam

from data.data import Parser, SoundDataset
from utils.synths import ModalFIR
from utils import core
from utils.core import exp_sigmoid, frequency_filter, extract_params_multi_audio
from utils.losses import SpectralLoss
from utils.plot import plot_audio
from utils.contact import (
    extract_gains_at_freqs,
    jet_colormap,
    marker,
    write_ply,
)


@dataclass
class Config:
    """Configuration for Modal Sound Field Trainer."""
    
    # Paths
    config: Optional[str] = None
    object_id: Optional[int] = None
    ckpt_path: str = "../results/feature_3dgs/{object_id}/feature_3dgs.pt"
    sound_dataset_path: str = "data/outputs/{object_id}"
    result_dir: str = "../results/{config_name}/{object_id}"
    device: str = "cuda"

    # Audio data
    sample_rate: int = 44100
    audio_duration: float = 3.0
    tool: bool = False
    norm: str = "force"
    hz_min: float = 20.0
    hz_max: float = 20000.0
    train_ratio: float = 0.2

    # STFT
    n_fft: int = 1024
    hop_length: int = 64

    # Initialization
    n: int = 400                          # modes extracted per training audio
    damp_init_from_global: bool = False
    gain_weight_init_zero: bool = False

    # Sound field
    n_audio_clusters: int = 512
    q_pos_freqs: int = 10
    d_attn: Optional[int] = None          # attention dim; None -> 768
    tau_feat: float = 1.0                 # softmax temperature
    context_norm: str = "l2"              # normalization: 'l2' or 'train'
    spatial_damping: bool = True

    # Loss
    mag_weight: float = 1.0
    logmag_weight: float = 1.0
    cutoff_weight: float = 0.5
    cutoff_samples: int = 4410
    stft_fft_sizes: List[int] = field(default_factory=lambda: [2048, 1024, 512, 256])

    # Training
    seed: int = 42
    max_steps: int = 10000
    warmup_steps: int = 2000

    # Learning rates
    lr: float = 5e-3
    attn_lr_mult: float = 1.0             # LR multiplier for q/k/v/point_feat/gain/damp MLPs
    q_lr: Optional[float] = None
    k_lr: Optional[float] = None
    v_lr: Optional[float] = None
    point_mlp_lr: Optional[float] = None
    gain_bias_lr_scale: float = 1.0
    damp_bias_lr_scale: float = 1.0
    cosine_per_group: bool = False
    cosine_min_ratio: float = 0.1
    cosine_min_lrs: dict[str, float] = field(default_factory=dict)

    # Evaluation
    eval_interval: int = 1000
    eval_seed: Optional[int] = None
    run_contact_loc: bool = True          # run contact localization + PLY at end
    contact_export_topk: int = 3          # export this many best and worst contact PLYs


def build_mlp(in_dim: int, hidden_dims: List[int], out_dim: int) -> nn.Sequential:
    """Build a simple MLP."""
    dims = [in_dim] + list(hidden_dims) + [out_dim]
    n_linears = len(dims) - 1
    layers: List[nn.Module] = []
    for i in range(n_linears):
        linear = nn.Linear(dims[i], dims[i + 1])
        is_last = i == n_linears - 1
        layers.append(linear)
        if not is_last:
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class SoundFieldTrainer(nn.Module):
    """
    Modal Sound Field Trainer.
    """
    # ----- Initialization -----
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.device
        os.makedirs(cfg.result_dir, exist_ok=True)
        
        # Load 3DGS checkpoint
        ckpt = torch.load(cfg.ckpt_path, map_location=self.device, weights_only=True)
        splats = ckpt['splats']
        means = splats['means'].float()
        self.n_gaussians = means.shape[0]
        self.register_buffer('gaussian_means', means.to(self.device))
        print(f"[SoundFieldTrainer] Loaded {self.n_gaussians} Gaussians from checkpoint {cfg.ckpt_path}")

        # Load 3DGS DINO features
        assert "features" in splats, f"Expected splats['features'], got keys={list(splats.keys())}"
        feats = splats["features"].float()
        if feats.dim() == 3:
            feats = feats.squeeze(1)
        assert feats.dim() == 2, f"Expected features [N,D], got {feats.shape}"
        assert feats.shape[0] == self.n_gaussians, f"Mismatch N: features {feats.shape[0]} vs gaussians {self.n_gaussians}"
        assert feats.shape[1] == 768, f"Expected DINO feature dim 768, got {feats.shape[1]}"
        self.register_buffer("gaussian_features", feats.to(self.device))
        self.gaussian_features = F.normalize(self.gaussian_features, dim=-1)
        print(f"[SoundFieldTrainer] gaussian_features: {self.gaussian_features.shape}")
        
        # Build audio clusters from Gaussian positions
        self.register_buffer(
            "gaussian2cluster",  # [N_gaussians]
            torch.empty(self.n_gaussians, dtype=torch.long, device=self.device)
        )
        self._build_audio_clusters(means)
        
        # Load training data for initialization
        parser = Parser(data_dir=cfg.sound_dataset_path)
        init_dataset = SoundDataset(
            parser=parser,
            sample_rate=cfg.sample_rate,
            tool=cfg.tool,
            norm=cfg.norm,
            audio_duration=cfg.audio_duration,
            train_ratio=cfg.train_ratio,
            split='train',
            hz_min=cfg.hz_min,
            hz_max=cfg.hz_max,
        )
        audio_list = [init_dataset[i]['audio'] for i in range(len(init_dataset))]
        print(f"[SoundFieldTrainer] Loaded {len(audio_list)} audio samples for initialization")
        if cfg.context_norm == 'train':
            # Gaussian index of each training impact. context_norm='train' computes its
            # normalization mean and scale on these points only (see _normalize_context).
            reference = self.map_positions_to_gaussians(init_dataset.get_all_positions())
            self.register_buffer(
                "context_reference_indices",
                torch.as_tensor(reference, dtype=torch.long).to(self.device),
                persistent=False,
            )
        
        # Initialize parameters from audio data
        init_freqs, init_damps, init_gains, init_phases, n_modes = extract_params_multi_audio(
            audio_list=audio_list,
            n=cfg.n,
            sr=cfg.sample_rate,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length
        )
        self.n_modes = n_modes

        # For warmup loss
        self.register_buffer('init_gains_target', torch.from_numpy(init_gains).float().to(self.device))
        self.freqs = nn.Parameter(torch.from_numpy(init_freqs).float().to(self.device))
        self.damps = nn.Parameter(torch.from_numpy(init_damps).float().to(self.device))
        
        # Freq/damp clamp bounds per mode: freq +-30 Hz, damp [1, 2*init_damp]
        self.register_buffer('freq_clamp_min', torch.from_numpy(init_freqs - 30.0).float().to(self.device))  # [n_modes]
        self.register_buffer('freq_clamp_max', torch.from_numpy(init_freqs + 30.0).float().to(self.device))  # [n_modes]
        self.register_buffer('damp_clamp_max', torch.from_numpy(init_damps * 2.0).float().to(self.device))   # [n_modes]
        self.register_buffer('init_phases', torch.from_numpy(init_phases).float().to(self.device))  # [n_audios, n_modes]
        self.n_samples = int(cfg.audio_duration * cfg.sample_rate)
        
        # QKV feature attention
        self.q_pos_freqs = cfg.q_pos_freqs
        self.pe_dim_per = 3 + 3 * 2 * self.q_pos_freqs  # per-offset PE dim
        self._build_field_modules()

        # Estimate per-frequency noise spectrum from training audio
        noise_spectrum = estimate_noise_spectrum(
            audio_list=audio_list,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            device=self.device
        )
        self.register_buffer('noise_initial_bias', noise_spectrum)  # [n_freqs]

        n_freqs = cfg.n_fft // 2 + 1
        self.noise_mags = nn.Parameter(
            torch.zeros(1, n_freqs, device=self.device)  # learnable residual offset
        )


        # Initialize sound synthesizer
        self.modal_fir = ModalFIR(
            n_samples=self.n_samples,
            sample_rate=cfg.sample_rate,
            amp_scale_fn=exp_sigmoid,
            freq_scale_fn=core.frequencies_critical_bands,
            hz_min=cfg.hz_min,
            hz_max=cfg.hz_max,
            freq_scale='mel',
            initial_bias=-1.5
        ).to(self.device)

        print(f"[SoundFieldTrainer] Synthesizer: {self.n_samples} samples at {cfg.sample_rate} Hz")
        print(f"[SoundFieldTrainer] Initialized: {self.n_gaussians} Gaussians -> {cfg.n_audio_clusters} clusters, {self.n_modes} modes")
        print(f"[SoundFieldTrainer] Frequencies: {self.freqs.shape}, Dampings: {self.damps.shape}")
        print(f"[SoundFieldTrainer] Noise magnitudes: {self.noise_mags.shape}")
    
    def _build_field_modules(self):
        """Attention encoder, point MLP and gain/damp heads."""
        cfg = self.cfg
        K_clusters = cfg.n_audio_clusters
        D = 768
        d_attn = cfg.d_attn if cfg.d_attn is not None else D
        self.d_attn = d_attn

        # W_q, W_k, W_v for attention
        self.q_proj = nn.Linear(K_clusters * self.pe_dim_per + D, d_attn).to(self.device)
        self.k_proj = nn.Linear(D, d_attn).to(self.device)
        self.v_proj = nn.Linear(D, d_attn).to(self.device)

        # Extra projection of attention output z before the gain/damp heads.
        self.point_feat_mlp = build_mlp(d_attn, [512], D).to(self.device)

        # Gain/damp head input: attention context z concatenated with the local DINO feature f_{i*}.
        head_in = D + D

        # Gain head G_theta(x) = MLP([z; f_{i*}])
        self.gain_mlp = nn.Linear(head_in, self.n_modes).to(self.device)
        # Damping head (optional spatially-varying damping)
        if cfg.spatial_damping:
            self.damp_mlp = nn.Linear(head_in, self.n_modes).to(self.device)
        else:
            self.damp_mlp = None
        self._initialize_damping_head()
        self._initialize_gain_head()

    def _effective_noise_magnitudes(self) -> torch.Tensor:
        return exp_sigmoid(self.noise_mags + self.noise_initial_bias)
    
    def _build_audio_clusters(self, means: torch.Tensor):
        """Cluster Gaussian centers in 3D Euclidean space"""
        cfg = self.cfg
        xyz = means.detach().cpu().numpy().astype(np.float32)  # [N, 3]
        print(
            f"[AudioCluster] Clustering {xyz.shape[0]} Gaussians -> "
            f"{cfg.n_audio_clusters} clusters (3D XYZ Euclidean)"
        )

        kmeans = MiniBatchKMeans(
            n_clusters=cfg.n_audio_clusters,
            batch_size=4096,
            max_iter=100,
            verbose=0,
            random_state=cfg.seed,
        ).fit(xyz)

        cluster_ids = torch.from_numpy(kmeans.labels_).long().to(self.device)
        self.gaussian2cluster.copy_(cluster_ids)

        centers_xyz = np.zeros((cfg.n_audio_clusters, 3), dtype=np.float32)
        counts = np.zeros(cfg.n_audio_clusters, dtype=np.int64)
        np.add.at(centers_xyz, kmeans.labels_, xyz)
        np.add.at(counts, kmeans.labels_, 1)
        counts = np.clip(counts, 1, None)
        centers_xyz = centers_xyz / counts[:, None]

        self.register_buffer(
            "audio_cluster_centers",
            torch.from_numpy(centers_xyz).float().to(self.device),
        )
        print(f"[AudioCluster] Done. Cluster centers (xyz): {self.audio_cluster_centers.shape}")

        self._compute_cluster_features_mean()

    def _compute_cluster_features_mean(self):
        """Compute mean feature vector for each audio cluster (then STD-normalize)."""
        cfg = self.cfg
        K = cfg.n_audio_clusters
        feats = self.gaussian_features  # [N,D]
        D = feats.shape[1]

        cluster_feat = torch.zeros(K, D, device=self.device, dtype=feats.dtype)
        cluster_cnt = torch.zeros(K, device=self.device, dtype=torch.long)

        cid = self.gaussian2cluster  # [N]
        cluster_feat.index_add_(0, cid, feats)
        cluster_cnt.index_add_(0, cid, torch.ones_like(cid, dtype=torch.long))

        cluster_cnt = cluster_cnt.clamp_min(1)
        cluster_feat = cluster_feat / cluster_cnt[:, None]   # [K, D]

        # STD normalize over clusters
        mu = cluster_feat.mean(dim=0, keepdim=True)                      # [1, D]
        std = cluster_feat.std(dim=0, keepdim=True, unbiased=False)      # [1, D]
        cluster_feat = (cluster_feat - mu) / (std + 1e-6)

        if hasattr(self, "audio_cluster_features"):
            self.audio_cluster_features.copy_(cluster_feat)
        else:
            self.register_buffer("audio_cluster_features", cluster_feat)

        print(f"[AudioCluster] audio_cluster_features: {self.audio_cluster_features.shape}")

    def _abs_pe(self, pos: torch.Tensor, num_freqs: int) -> torch.Tensor:
        """
        pos: [3] or [N,3]  (absolute xyz)
        return: concat([pos, sincos(pos)]) => [N, 3 + 3*2F]
        """
        if pos.dim() == 1:
            pos = pos[None, :]
        freqs = (2.0 ** torch.arange(num_freqs, device=pos.device, dtype=pos.dtype)) * math.pi  # [F]
        x = pos[:, :, None] * freqs[None, None, :]  # [N,3,F]
        pe = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)  # [N,3,2F]
        pe = pe.reshape(pos.shape[0], -1)  # [N, 3*2F]
        return torch.cat([pos, pe], dim=-1)  # [N, 3 + 3*2F]
    
    def compute_point_feature(self, gs_idx: int):
        """
        r_k = x - c_k   for all K clusters,    r = [r_1, ..., r_K]
        q   = W_q [PE(r); f_{i*}]
        k_k = W_k f̄_k,  v_k = W_v f̄_k
        α_k = softmax(k_k^T q / sqrt(d))
        z   = Σ_k α_k v_k
        """
        if self.cfg.context_norm != 'l2':
            return self._normalized_point_feats([gs_idx])[0]
        x = self.gaussian_means[gs_idx]         # [3]
        gfeat = self.gaussian_features[gs_idx]  # [D], local feature f_{i*}

        # Offsets to ALL K cluster centers, then per-offset PE, then stacked.
        rel_all = x[None, :] - self.audio_cluster_centers          # [K, 3]
        pe_rel = self._abs_pe(rel_all, num_freqs=self.q_pos_freqs)  # [K, pe_dim_per]
        pe_rel_flat = pe_rel.reshape(1, -1)                        # [1, K * pe_dim_per]

        # Q
        q_in = torch.cat([pe_rel_flat, gfeat[None, :]], dim=-1)    # [1, K*pe + D]
        q = self.q_proj(q_in)[0]  # [d_attn]

        # cluster features (no position)
        cf = self.audio_cluster_features  # [K, D], normalized

        # K/V
        k = self.k_proj(cf)  # [K, d_attn]
        v = self.v_proj(cf)  # [K, d_attn]

        # attention
        scale = 1.0 / math.sqrt(k.shape[-1])
        logits = (k @ q) * scale  # [K]
        attn = torch.softmax(logits / max(self.cfg.tau_feat, 1e-6), dim=0)  # [K]

        agg = (attn[:, None] * v).sum(dim=0, keepdim=True)  # [1, d_attn]
        agg = F.normalize(agg, dim=-1)

        point_feat = self.point_feat_mlp(agg)            # [1, D]
        point_feat = F.normalize(point_feat, dim=-1)[0]  # [D]
        return point_feat

    def _compute_point_feats_batch(self, indices: torch.Tensor) -> torch.Tensor:
        """Returns [B, head_in]."""
        if self.cfg.context_norm != 'l2':
            return self._normalized_point_feats(indices)
        cf = self.audio_cluster_features                     # [K, D]
        k = self.k_proj(cf)                                  # [K, d]
        v = self.v_proj(cf)                                  # [K, d]
        scale = 1.0 / math.sqrt(k.shape[-1])
        cc = self.audio_cluster_centers                      # [K, 3]

        xs = self.gaussian_means[indices]                    # [B, 3]
        fs = self.gaussian_features[indices]                 # [B, D]
        rel = xs[:, None, :] - cc[None, :, :]                # [B, K, 3]
        pe = self._abs_pe(rel.reshape(-1, 3),
                          num_freqs=self.q_pos_freqs).reshape(len(indices), -1)
        q_in = torch.cat([pe, fs], dim=-1)                   # [B, K*pe + D]
        q = self.q_proj(q_in)                                # [B, d]
        logits = (q @ k.T) * scale                           # [B, K]
        attn = torch.softmax(logits / max(self.cfg.tau_feat, 1e-6), dim=-1)
        agg = attn @ v                                       # [B, d]
        agg = F.normalize(agg, dim=-1)
        pf = self.point_feat_mlp(agg)                        # [B, D]
        pf = F.normalize(pf, dim=-1)
        return pf

    def _normalized_point_feats(self, indices) -> torch.Tensor:
        """Point features under 'train' normalization."""
        indices = torch.as_tensor(indices, device=self.device, dtype=torch.long)
        n_query = len(indices)
        indices = torch.cat([indices, self.context_reference_indices])
        xs = self.gaussian_means[indices]
        fs = self.gaussian_features[indices]
        rel = xs[:, None, :] - self.audio_cluster_centers[None, :, :]
        pe = self._abs_pe(rel.reshape(-1, 3), self.q_pos_freqs).reshape(len(indices), -1)
        q = self.q_proj(torch.cat([pe, fs], dim=-1))
        k = self.k_proj(self.audio_cluster_features)
        v = self.v_proj(self.audio_cluster_features)
        logits = (q @ k.T) / math.sqrt(k.shape[-1]) / max(self.cfg.tau_feat, 1e-6)
        attn = logits.softmax(dim=-1)
        context = attn @ v
        projected = self.point_feat_mlp(self._normalize_context(context, n_query))
        return self._normalize_context(projected, n_query)[:n_query]

    def _normalize_context(self, x: torch.Tensor, n_query: int) -> torch.Tensor:
        """Normalization before and after the point MLP."""
        if self.cfg.context_norm == 'l2':
            return F.normalize(x, dim=-1)
        reference = x[n_query:]
        mean = reference.mean(dim=0)
        scale = ((reference - mean).square().sum(dim=-1).mean() + 1e-12).sqrt()
        return (x - mean) / scale

    @torch.no_grad()
    def _compute_all_gains(self, chunk_size: int = 2048) -> torch.Tensor:
        """Gain field G_theta for ALL N Gaussians. Returns [N, M]."""
        N = self.gaussian_means.shape[0]
        out = []
        for s in range(0, N, chunk_size):
            e = min(s + chunk_size, N)
            idx = torch.arange(s, e, device=self.device)
            pf = self._compute_point_feats_batch(idx)
            out.append(self.gain_mlp(self._head_input(pf, idx)))
        return torch.cat(out, dim=0)

    def map_positions_to_gaussians(self, positions: torch.Tensor) -> List[int]:
        positions = positions.to(self.device)
        dists = torch.norm(positions.unsqueeze(1) - self.gaussian_means.unsqueeze(0), dim=-1)
        return dists.argmin(dim=1).tolist()


    # ----- Rendering -----
    def render_impact_sound(
        self, 
        gs_idx: int, 
        modal_fir: Optional[ModalFIR] = None,
        n_samples: Optional[int] = None,
        init_phases: Optional[torch.Tensor] = None,
        precomputed_gains: Optional[torch.Tensor] = None,
        precomputed_damps: Optional[torch.Tensor] = None,
        return_components: bool = False,
        noise_generator: Optional[torch.Generator] = None,
    ):
        cfg = self.cfg
        if precomputed_gains is not None:
            weighted_gains_1d = precomputed_gains
            weighted_damps_1d = (precomputed_damps if precomputed_damps is not None
                                 else (self.compute_weighted_gains_damps(gs_idx)[1]
                                       if cfg.spatial_damping else self.damps))
        else:
            weighted_gains_1d, weighted_damps_1d = self.compute_weighted_gains_damps(gs_idx)  # [M]
        weighted_gains = weighted_gains_1d[None, None, :]  # [1,1,M] for synthesizer
        
        freqs = self.freqs[None, None, :]  # [1, 1, M]
        damps = weighted_damps_1d[None, None, :]  # [1, 1, M]
        
        if modal_fir is None:
            modal_fir = self.modal_fir
        
        # Prepare init_phases for synthesizer: [1, 1, M]
        synth_init_phases = None
        if init_phases is not None:
            synth_init_phases = init_phases.unsqueeze(0).unsqueeze(0)  # [M] -> [1, 1, M]
        
        signal = modal_fir.get_signal(
            weighted_gains,  # raw gains
            freqs,           # raw frequencies (Hz)
            damps,           # raw dampings
            init_phases=synth_init_phases
        )[0]  # [T]

        if n_samples is None:
            n_samples = self.n_samples

        white_noise = torch.randn(1, n_samples, device=signal.device, generator=noise_generator)
        noise_magnitudes = self._effective_noise_magnitudes()  # [1, n_freqs]
        noise = frequency_filter(white_noise, noise_magnitudes, 257)[0]  # [n_samples]
        audio = signal + noise

        if return_components:
            return audio, signal, noise
        return audio

 
    # ----- Training -----
    def _head_input(self, point_feat: torch.Tensor, idx) -> torch.Tensor:
        """Concatenate the attention context z with the local DINO feature
        f_{i*} for the gain/damp head. Single index or batch."""
        lf = self.gaussian_features[idx]  # [D] or [B, D]
        return torch.cat([point_feat, lf], dim=-1)

    def compute_weighted_gains_damps(self, gs_idx: int) -> torch.Tensor:
        point_feat = self.compute_point_feature(gs_idx)  # [D] or [d_attn]
        hi = self._head_input(point_feat, gs_idx)
        gains = self.gain_mlp(hi)  # [M]
        if self.cfg.spatial_damping:
            damps = self.damp_mlp(hi)  # [M], spatially-varying
        else:
            damps = self.damps                 # [M], global
        return gains, damps

    @torch.no_grad()
    def _initialize_damping_head(self) -> None:
        if self.cfg.spatial_damping and self.cfg.damp_init_from_global:
            self.damp_mlp.weight.zero_()
            self.damp_mlp.bias.copy_(self.damps.detach())

    @torch.no_grad()
    def _initialize_gain_head(self) -> None:
        if self.cfg.gain_weight_init_zero:
            self.gain_mlp.weight.zero_()

    def _attention_param_groups(self) -> list:
        default_lr = self.cfg.lr * self.cfg.attn_lr_mult
        overrides = {
            'q_proj': self.cfg.q_lr, 'k_proj': self.cfg.k_lr,
            'v_proj': self.cfg.v_lr, 'point_feat_mlp': self.cfg.point_mlp_lr,
        }
        groups = [
            {
                'name': name,
                'params': list(getattr(self, name).parameters()),
                'lr': default_lr if overrides.get(name) is None else overrides[name],
            }
            for name in self._field_module_names()
        ]
        for name, scale in (('gain_mlp', self.cfg.gain_bias_lr_scale), ('damp_mlp', self.cfg.damp_bias_lr_scale)):
            head = getattr(self, name)
            if scale == 1.0 or head is None:
                continue
            group = next(group for group in groups if group['name'] == name)
            group['params'] = [head.weight]
            groups.append({'name': f'{name}_bias', 'params': [head.bias], 'lr': group['lr'] * scale})
        return groups

    def _make_optimizer(self, warmup: bool = False) -> torch.optim.Optimizer:
        groups = []
        if not warmup:
            names = ['freqs', 'noise_mags']
            if not self.cfg.spatial_damping:
                names.append('damps')
            groups = [
                {'name': name, 'params': [getattr(self, name)], 'lr': self.cfg.lr}
                for name in names
            ]
        return torch.optim.Adam(groups + self._attention_param_groups())

    def _make_scheduler(self, optimizer):
        cfg = self.cfg
        if cfg.cosine_per_group:
            def multiplier(step, minimum_ratio):
                progress = min(step, cfg.max_steps) / cfg.max_steps
                return minimum_ratio + (1.0 - minimum_ratio) * (
                    1.0 + math.cos(math.pi * progress)
                ) / 2.0

            floors = dict(cfg.cosine_min_lrs)
            for name, scale in (('gain_mlp', cfg.gain_bias_lr_scale), ('damp_mlp', cfg.damp_bias_lr_scale)):
                if name in floors:  # the bias group keeps the weights' decay ratio by default
                    floors.setdefault(f'{name}_bias', floors[name] * scale)
            ratios = [
                floors.get(group['name'], group['lr'] * cfg.cosine_min_ratio) / group['lr']
                for group in optimizer.param_groups
            ]
            return torch.optim.lr_scheduler.LambdaLR(
                optimizer, [lambda step, ratio=ratio: multiplier(step, ratio) for ratio in ratios]
            )
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.max_steps, eta_min=cfg.lr * cfg.cosine_min_ratio
        )

    def train(self):
        # Load training data
        cfg = self.cfg
        parser = Parser(data_dir=cfg.sound_dataset_path)
        dataset = SoundDataset(
            parser=parser,
            sample_rate=cfg.sample_rate,
            tool=cfg.tool,
            norm=cfg.norm,
            audio_duration=cfg.audio_duration,
            train_ratio=cfg.train_ratio,
            split='train',
            hz_min=cfg.hz_min,
            hz_max=cfg.hz_max,
        )
        n_train = len(dataset)
        print(f"[Train] Loaded {n_train} training samples")
        
        gs_indices = self.map_positions_to_gaussians(dataset.get_all_positions())

        target_audios = []
        for i in range(n_train):
            target_audios.append(dataset[i]['audio'].to(self.device))

        # Warmup
        optimizer_warmup = self._make_optimizer(warmup=True)

        for step in range(cfg.warmup_steps):
            idx = random.randint(0, n_train - 1)
            gs_idx = gs_indices[idx]
            target_gains = self.init_gains_target[idx].detach()     # [M]

            # Warmup MSE against extracted modal gains.
            pred_gains, pred_damps = self.compute_weighted_gains_damps(gs_idx)
            loss_gain = F.mse_loss(pred_gains, target_gains)
            if cfg.spatial_damping:
                loss_damp = F.mse_loss(pred_damps, self.damps.detach())
                loss = loss_gain + loss_damp
            else:
                loss_damp = torch.zeros((), device=self.device)
                loss = loss_gain

            optimizer_warmup.zero_grad()
            loss.backward()
            optimizer_warmup.step()

            if step % 100 == 0:
                print(f"[Warmup] step {step}  gain={loss_gain.item():.6f}  damp={loss_damp.item():.6f}")

        del optimizer_warmup

        self.freqs.requires_grad = True
        self.damps.requires_grad = not cfg.spatial_damping

        optimizer = self._make_optimizer()
        scheduler = self._make_scheduler(optimizer)
        optimizer.zero_grad()

        # Loss
        fft_sizes = tuple(cfg.stft_fft_sizes)
        loss_fn = SpectralLoss(
            fft_sizes=fft_sizes,
            hop_length=cfg.hop_length,
            loss_type='L1',
            mag_weight=cfg.mag_weight,
            logmag_weight=cfg.logmag_weight
        ).to(self.device)
        
        loss_fn_cutoff = SpectralLoss(
            fft_sizes=fft_sizes,
            hop_length=cfg.hop_length,
            loss_type='L1',
            mag_weight=cfg.mag_weight,
            logmag_weight=0.0,
            cutoff=cfg.cutoff_samples
        ).to(self.device)

        # Track the validation-envelope best checkpoint. Contact localization is
        # computed once from this selected state after training.
        best_env = {'val': float('inf'), 'step': -1, 'state': None}
        ckpt_path = os.path.join(cfg.result_dir, "sound_field.pt")

        pbar = tqdm(range(cfg.max_steps), desc="[Train]")
        for step in pbar:
            idx = random.randint(0, n_train - 1)
            target_audio = target_audios[idx]
            gs_idx = gs_indices[idx]

            point_feat = self.compute_point_feature(gs_idx)
            hi = self._head_input(point_feat, gs_idx)
            weighted_gains = self.gain_mlp(hi)  # [M]
            if cfg.spatial_damping:
                pred_damps = self.damp_mlp(hi)
            else:
                pred_damps = self.damps

            pred_audio = self.render_impact_sound(
                gs_idx,
                init_phases=self.init_phases[idx],
                precomputed_gains=weighted_gains,
                precomputed_damps=pred_damps,
            )

            loss_spec = loss_fn(target_audio.unsqueeze(0), pred_audio.unsqueeze(0))
            loss_cutoff = loss_fn_cutoff(target_audio.unsqueeze(0), pred_audio.unsqueeze(0))
            total_loss = loss_spec + cfg.cutoff_weight * loss_cutoff

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                self.freqs.data = torch.clamp(
                    self.freqs.data,
                    min=self.freq_clamp_min,
                    max=self.freq_clamp_max,
                )
            
            pbar.set_postfix(
                loss=f"{total_loss.item():.4f}",
                spec=f"{loss_spec.item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}"
            )

            if (step + 1) % cfg.eval_interval == 0 or step == cfg.max_steps - 1:
                eval_stats = self.eval(step=step, export_artifacts=True)
                val_stats = eval_stats.get("test")
                ckpt_step = step + 1
                if val_stats:
                    env = val_stats.get('envelope_dist', float('inf'))
                    if env < best_env['val']:
                        best_env = {
                            'val': float(env),
                            'step': ckpt_step,
                            'state': self._field_state_snapshot(),
                        }
                        print(f"[Best] envelope_dist={best_env['val']:.6f}@{best_env['step']}")
                    else:
                        print(f"[Best] envelope_dist={best_env['val']:.6f}@{best_env['step']}")

        best_step = best_env['step']

        if best_env['state'] is not None:
            self._restore_field_state(best_env['state'])

        self._save_sound_field_ckpt(ckpt_path)

        # Re-evaluate the restored best checkpoint for the final metrics.
        test_stats = self.eval()["test"]
        contact_summary = None
        if cfg.run_contact_loc:
            try:
                contact_summary = self.run_contact_localization(export_plys=True)
            except Exception as e:
                print(f"[ContactLoc] FAILED: {type(e).__name__}: {e}")

        metrics = {
            "step": int(best_step),
            "l1_mag": test_stats["l1_mag"],
            "l1_log": test_stats["l1_logmag"],
            "envelope_dist": test_stats["envelope_dist"],
            "cdpam": test_stats["cdpam"],
            "snr_db": test_stats["snr_db"],
            "num_test_samples": test_stats["num_test_samples"],
            "contact_localization": contact_summary,
        }
        metrics_path = os.path.join(cfg.result_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[Final] Metrics saved to {metrics_path}")


    def _field_module_names(self):
        names = ['q_proj', 'k_proj', 'v_proj', 'gain_mlp']
        names.append('point_feat_mlp')
        if self.cfg.spatial_damping:
            names.append('damp_mlp')
        return names

    def _field_state_snapshot(self) -> dict:
        sd: dict = {}
        for name in self._field_module_names():
            for k, v in getattr(self, name).state_dict().items():
                sd[f"{name}.{k}"] = v.detach().cpu().clone()
        for n in ('freqs', 'damps', 'noise_mags'):
            sd[n] = getattr(self, n).detach().cpu().clone()
        return sd

    def _restore_field_state(self, sd: dict) -> None:
        with torch.no_grad():
            for n in ('freqs', 'damps', 'noise_mags'):
                getattr(self, n).copy_(sd[n].to(self.device))
        for name in self._field_module_names():
            sub = {k.split('.', 1)[1]: v for k, v in sd.items()
                   if k.startswith(f"{name}.")}
            getattr(self, name).load_state_dict(sub, strict=True)

    def _save_sound_field_ckpt(self, path: str) -> None:
        cfg = self.cfg
        data = {
            "cfg": {
                "sample_rate": int(cfg.sample_rate),
                "audio_duration": float(cfg.audio_duration),
                "n_fft": int(cfg.n_fft),
                "hop_length": int(cfg.hop_length),
                "n_audio_clusters": cfg.n_audio_clusters,
                "feature_dim": 768,
                "n_modes": int(self.n_modes),
                "q_pos_freqs": int(self.q_pos_freqs),
                "d_attn": int(self.d_attn),
                "tau_feat": float(cfg.tau_feat),
                "context_norm": cfg.context_norm,
                "spatial_damping": bool(cfg.spatial_damping),
            },
            "training_config": asdict(cfg),
            "freqs": self.freqs.detach().cpu(),
            "damps": self.damps.detach().cpu(),
            "noise_mags": self.noise_mags.detach().cpu(),
            "noise_initial_bias": self.noise_initial_bias.detach().cpu(),
            "noise_effective_magnitudes": (self._effective_noise_magnitudes().detach().cpu()),
            "gaussian2cluster": self.gaussian2cluster.detach().cpu(),
            "audio_cluster_centers": self.audio_cluster_centers.detach().cpu(),
        }
        if hasattr(self, "context_reference_indices"):
            data["context_reference_indices"] = self.context_reference_indices.detach().cpu()
        for name in self._field_module_names():
            data[name] = getattr(self, name).state_dict()
        torch.save(data, path)
        print(f"[Train] Saved final sound-field checkpoint to {path}")

    # ----- Contact localization -----
    @torch.no_grad()
    def run_contact_localization(self, export_plys: bool = False) -> Optional[dict]:
        """Localize each TEST impact by matching audio-extracted modal gains to
        the learned sound field, report MED as % of bbox diagonal (lower is better).

        export_plys: if True, write the best and worst localization PLYs. Returns the summary dict."""
        cfg = self.cfg
        out_dir = os.path.join(cfg.result_dir, "contact")
        if export_plys:
            os.makedirs(out_dir, exist_ok=True)

        parser = Parser(data_dir=cfg.sound_dataset_path)
        test_dataset = SoundDataset(
            parser=parser,
            sample_rate=cfg.sample_rate,
            tool=cfg.tool,
            norm=cfg.norm,
            audio_duration=cfg.audio_duration,
            train_ratio=cfg.train_ratio,
            split="test",
            hz_min=cfg.hz_min,
            hz_max=cfg.hz_max,
        )
        print(f"[ContactLoc] {len(test_dataset)} test impacts")

        gains = self._compute_all_gains().cpu().numpy().astype(np.float32)  # [N,M]
        gains = np.abs(gains)
        gain_norms = np.maximum(np.linalg.norm(gains, axis=1, keepdims=True), 1e-10)
        gains_normed = gains / gain_norms
        xyz = self.gaussian_means.cpu().numpy().astype(np.float32)
        freqs_hz = self.freqs.detach().cpu().numpy()

        bbox_diag = float(np.linalg.norm(xyz.max(0) - xyz.min(0)))
        obj_centroid = xyz.mean(axis=0)
        obj_extent = (xyz.max(0) - xyz.min(0)).max()

        results = []
        for i in range(len(test_dataset)):
            sample = test_dataset[i]
            impact_id = int(sample["impact_id"])
            gt_pos = sample["position"].cpu().numpy().astype(np.float32)
            audio = sample["audio"].cpu().numpy().astype(np.float32)
            audio_gains = extract_gains_at_freqs(
                audio,
                freqs_hz,
                sr=cfg.sample_rate,
                n_fft=cfg.n_fft,
                hop_length=cfg.hop_length,
            )
            audio_gain_norm = np.linalg.norm(audio_gains)
            if audio_gain_norm > 1e-10:
                audio_gains = audio_gains / audio_gain_norm
            cos_sim = gains_normed @ audio_gains
            best_idx = int(np.argmax(cos_sim))
            pred_pos = xyz[best_idx]
            results.append(dict(impact_id=impact_id, gt_pos=gt_pos, pred_pos=pred_pos,
                                distance=float(np.linalg.norm(pred_pos - gt_pos)),
                                cos_sim=cos_sim))

        if not results:
            print("[ContactLoc] no test results")
            return None

        pcts = [r["distance"] / bbox_diag * 100.0 for r in results] if bbox_diag > 0 else [0.0]
        med_cm = float(np.mean([r["distance"] * 100.0 for r in results]))
        med_pct = float(np.mean(pcts))

        summary = {
            "bbox_diagonal_cm": float(bbox_diag * 100.0),
            "MED_cm": med_cm,
            "MED_pct_bbox": med_pct,
        }

        if export_plys:
            export_count = max(int(cfg.contact_export_topk), 0)
            ranked = sorted(results, key=lambda r: r["distance"])
            groups = (("best", ranked[:export_count]),
                      ("worst", list(reversed(ranked))[:export_count]))
            for group, selected in groups:
                for rank, r in enumerate(selected, start=1):
                    scores = r["cos_sim"]
                    smin, smax = scores.min(), scores.max()
                    norm = ((scores - smin) / (smax - smin)
                            if smax > smin else np.zeros_like(scores))
                    colors_u8 = (jet_colormap(norm) * 255).clip(0, 255).astype(np.uint8)
                    pct_r = r["distance"] / bbox_diag * 100.0 if bbox_diag > 0 else 0.0
                    ply_tag = (f"{group}_rank{rank:02d}_impact{r['impact_id']:03d}"
                               f"_{r['distance']*100:.2f}cm_{pct_r:.1f}pct")
                    # write_ply colors the first marker green (GT) and the
                    # pred_diamond_verts marker blue (prediction).
                    gt_verts = marker(
                        r["gt_pos"], obj_centroid, obj_extent)
                    pred_verts = marker(
                        r["pred_pos"], obj_centroid, obj_extent)
                    write_ply(
                        os.path.join(out_dir, f"{ply_tag}.ply"),
                        xyz, colors_u8, gt_verts,
                        pred_diamond_verts=pred_verts,
                    )

        return summary

    # ----- Evaluation -----
    @torch.no_grad()
    def eval(
        self, step: Optional[int] = None, export_artifacts: bool = False
    ) -> dict:
        """Optionally isolate evaluation randomness from training and other runs."""
        if self.cfg.eval_seed is None:
            return self._eval(step=step, export_artifacts=export_artifacts)
        python_state, numpy_state = random.getstate(), np.random.get_state()
        try:
            with torch.random.fork_rng():
                random.seed(self.cfg.eval_seed)
                np.random.seed(self.cfg.eval_seed)
                torch.manual_seed(self.cfg.eval_seed)
                return self._eval(step=step, export_artifacts=export_artifacts)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)

    @torch.no_grad()
    def _eval(
        self, step: Optional[int] = None, export_artifacts: bool = False
    ) -> dict:
        """Evaluate the sound field on test splits.
        
        Metrics:
            - L1 Spectral: Spectral L1 loss (mag only)
            - L1 LogMag Spectral: Spectral L1 loss (logmag only)
            - Envelope: Envelope distance
            - CDPAM: Contrastive learning-based Deep Perceptual Audio Metric
        """
        
        cfg = self.cfg
        device = self.device
        parser = Parser(data_dir=cfg.sound_dataset_path)

        step_name = "final" if step is None else str(step)
        render_dir = os.path.join(cfg.result_dir, "renders", step_name)
        wav_dir = os.path.join(render_dir, "wav")
        image_dir = os.path.join(render_dir, "images")
        if export_artifacts:
            os.makedirs(wav_dir, exist_ok=True)
            os.makedirs(image_dir, exist_ok=True)
        
        spectral_mag_loss_fn = SpectralLoss(
            fft_sizes=(1024,),
            hop_length=cfg.hop_length,
            loss_type='L1',
            mag_weight=1.0,
            logmag_weight=0.0
        ).to(device)
        
        spectral_logmag_loss_fn = SpectralLoss(
            fft_sizes=(1024,),
            hop_length=cfg.hop_length,
            loss_type='L1',
            mag_weight=0.0,
            logmag_weight=1.0
        ).to(device)

        if getattr(self, "_cdpam_metric", None) is None:
            original_torch_load = torch.load
            def patched_torch_load(*args, **kwargs):
                kwargs.setdefault('weights_only', False)
                return original_torch_load(*args, **kwargs)
            torch.load = patched_torch_load
            self._cdpam_metric = cdpam.CDPAM()
            torch.load = original_torch_load
        cdpam_metric = self._cdpam_metric

        dataset = SoundDataset(
            parser=parser,
            sample_rate=cfg.sample_rate,
            tool=cfg.tool,
            norm=cfg.norm,
            audio_duration=cfg.audio_duration,
            train_ratio=cfg.train_ratio,
            split="test",
            hz_min=cfg.hz_min,
            hz_max=cfg.hz_max,
        )
        print(f"[Eval] Loaded {len(dataset)} test samples")

        gs_indices = self.map_positions_to_gaussians(dataset.get_all_positions())
        metrics = defaultdict(list)
        noise_generator = None
        if cfg.eval_seed is not None:
            noise_generator = torch.Generator(device=device).manual_seed(cfg.eval_seed)

        for i in tqdm(range(len(dataset)), desc="[Eval] test"):
            sample = dataset[i]
            target_audio = sample['audio'].to(device)  # [n_samples]
            gs_idx = gs_indices[i]

            pred_audio, signal_audio, noise_audio = self.render_impact_sound(
                gs_idx,
                init_phases=None,
                return_components=True,
                noise_generator=noise_generator,
            )

            signal_power = torch.mean(signal_audio.float() ** 2).item()
            noise_power = torch.mean(noise_audio.float() ** 2).item()
            snr_db = 10.0 * math.log10((signal_power + 1e-12) / (noise_power + 1e-12))
            metrics["snr_db"].append(snr_db)

            # Peak normalize both syn and gt to [-1, 1] before metrics
            pred_peak = pred_audio.abs().max()
            if pred_peak > 0:
                pred_audio = pred_audio / pred_peak
            target_peak = target_audio.abs().max()
            if target_peak > 0:
                target_audio = target_audio / target_peak

            target_batch = target_audio.unsqueeze(0)  # [1, n_samples]
            pred_batch = pred_audio.unsqueeze(0)  # [1, n_samples]

            # 1. L1 Spectral Loss (mag only)
            l1_mag = spectral_mag_loss_fn(target_batch, pred_batch)
            metrics["l1_mag"].append(l1_mag.item())

            # 2. L1 Spectral Loss (logmag only)
            l1_logmag = spectral_logmag_loss_fn(target_batch, pred_batch)
            metrics["l1_logmag"].append(l1_logmag.item())

            # 3. Envelope Distance
            pred_np = pred_audio.detach().cpu().numpy()
            target_np = target_audio.detach().cpu().numpy()
            envelope_dist = core.compute_envelope_dist(pred_np, target_np)
            metrics["envelope_dist"].append(envelope_dist)

            # 4. CDPAM
            if cfg.sample_rate != 22050:
                target_22k = librosa.resample(target_np, orig_sr=cfg.sample_rate, target_sr=22050)
                pred_22k = librosa.resample(pred_np, orig_sr=cfg.sample_rate, target_sr=22050)
            else:
                target_22k = target_np
                pred_22k = pred_np

            target_22k = librosa.util.normalize(target_22k)
            pred_22k = librosa.util.normalize(pred_22k)

            target_22k_t = torch.from_numpy(target_22k).float().unsqueeze(0)
            pred_22k_t = torch.from_numpy(pred_22k).float().unsqueeze(0)

            cdpam_score = cdpam_metric.forward(target_22k_t, pred_22k_t)
            metrics["cdpam"].append(float(cdpam_score))

            if export_artifacts:
                impact_id = int(sample["impact_id"])
                stem = f"val_{impact_id}"
                pred_path = os.path.join(wav_dir, f"{stem}.wav")
                pred_pcm = (np.clip(pred_np, -1.0, 1.0) * 32767).astype(np.int16)
                wavfile.write(pred_path, cfg.sample_rate, pred_pcm)
                target_pcm = (np.clip(target_np, -1.0, 1.0) * 32767).astype(np.int16)
                with io.BytesIO() as target_wav:
                    wavfile.write(target_wav, cfg.sample_rate, target_pcm)
                    target_wav.seek(0)
                    plot_audio(
                        audio_path1=pred_path,
                        audio_path2=target_wav,
                        output_path=os.path.join(image_dir, f"{stem}.png"),
                        label1="Synthesized",
                        label2="Ground Truth",
                        sr=cfg.sample_rate,
                    )

        stats = {k: float(np.mean(v)) for k, v in metrics.items()}
        stats["num_test_samples"] = len(dataset)

        print(f"\n[Eval] TEST Results (step={step}):")
        print(f"  L1 Mag:    {stats['l1_mag']:.6f}")
        print(f"  L1 LogMag: {stats['l1_logmag']:.6f}")
        print(f"  Envelope Dist:      {stats['envelope_dist']:.6f}")
        print(f"  CDPAM:              {stats['cdpam']:.6f}")
        print(f"  Signal-to-noise:    {stats['snr_db']:.2f} dB")

        if step is not None:
            eval_dir = os.path.join(cfg.result_dir, "eval")
            os.makedirs(eval_dir, exist_ok=True)
            eval_data = {
                "step": int(step),
                "l1_mag": stats["l1_mag"],
                "l1_log": stats["l1_logmag"],
                "envelope_dist": stats["envelope_dist"],
                "cdpam": stats["cdpam"],
                "snr_db": stats["snr_db"],
                "num_test_samples": stats["num_test_samples"],
            }
            eval_path = os.path.join(eval_dir, f"{step}.json")
            with open(eval_path, "w") as f:
                json.dump(eval_data, f, indent=2)
            print(f"[Eval] Metrics saved to {eval_path}")

        return {"test": stats}
    

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Seed] Set all random seeds to {seed}")

def estimate_noise_spectrum(
    audio_list: List[torch.Tensor],
    n_fft: int = 2048,
    hop_length: int = 256,
    device: str = 'cuda',
) -> torch.Tensor:
    """
    Estimate per-frequency-bin noise spectrum from training audio.

    Returns:
        noise_spectrum: Per-frequency bias tensor of shape [n_freqs]
        In inverse exp_sigmoid space, so exp_sigmoid(bias) gives target magnitude
    """
    
    n_freqs = n_fft // 2 + 1
    all_mags = []  # Collect all STFT magnitudes [n_audio, n_freqs, n_frames]
    window = torch.hann_window(n_fft)
    
    for audio in audio_list:
        if isinstance(audio, torch.Tensor):
            audio_np = audio.detach().cpu().numpy()
        else:
            audio_np = np.array(audio)
        
        audio_t = torch.from_numpy(audio_np).float().unsqueeze(0)  # [1, n_samples]
        stft_result = torch.stft(audio_t, n_fft=n_fft, hop_length=hop_length, 
                                  win_length=n_fft, window=window, return_complex=True)
        mag = torch.abs(stft_result[0])  # [n_freqs, n_frames]
        all_mags.append(mag)
    
    all_mags = torch.cat(all_mags, dim=1)  # [n_freqs, total_frames]

    def lower_tail_mean(magnitudes: torch.Tensor) -> torch.Tensor:
        cutoff = torch.quantile(magnitudes, 0.25, dim=1, keepdim=True)
        mask = magnitudes <= cutoff
        return (magnitudes * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)

    target_spectrum = lower_tail_mean(all_mags)

    # Calibrate STFT magnitudes to transfer magnitudes for unit white noise.
    generator = torch.Generator().manual_seed(0)
    reference_noise = torch.randn(
        4, len(audio_list[0]), generator=generator, dtype=torch.float32
    )
    reference_stft = torch.stft(
        reference_noise,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        return_complex=True,
    )
    reference_mags = reference_stft.abs().permute(1, 0, 2).reshape(n_freqs, -1)
    white_reference = lower_tail_mean(reference_mags)
    target_spectrum = target_spectrum / white_reference.clamp_min(1e-7)
    
    print(f"[estimate_noise_spectrum] Analyzed {len(audio_list)} audio samples")
    print(f"  Total STFT frames: {all_mags.shape[1]}")
    print(f"  Calibrated filter magnitude: [{target_spectrum.min():.6f}, {target_spectrum.max():.6f}]")
    
    # Invert exp_sigmoid per frequency bin
    exponent = 10.0
    max_value = 2.0
    threshold = 1e-7
    
    target_spectrum = torch.clamp(target_spectrum, threshold + 1e-7, max_value + threshold - 1e-7)
    p = (target_spectrum - threshold) / max_value
    log_exp = np.log(exponent)
    sigmoid_x = torch.pow(p, 1.0 / log_exp)
    sigmoid_x = torch.clamp(sigmoid_x, 1e-7, 1.0 - 1e-7)
    noise_spectrum = torch.log(sigmoid_x / (1.0 - sigmoid_x))
    
    return noise_spectrum.to(device)


def main(cfg: Config):
    exp_name = "default"
    if cfg.config is not None:
        with open(cfg.config, 'r') as f:
            yaml_cfg = yaml.safe_load(f) or {}

        exp_name = yaml_cfg.pop('name', os.path.splitext(os.path.basename(cfg.config))[0])
        
        for key, value in yaml_cfg.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        
        print(f"[Config] Loaded from {cfg.config}, exp_name={exp_name}")
    
    if cfg.object_id is not None:
        path_vars = {
            "object_id": cfg.object_id,
            "config_name": exp_name,
        }
        cfg.ckpt_path = cfg.ckpt_path.format(**path_vars)
        cfg.sound_dataset_path = cfg.sound_dataset_path.format(**path_vars)
        cfg.result_dir = cfg.result_dir.format(**path_vars)
        print(f"[Config] object_id={cfg.object_id}")
    else:
        raise ValueError("object_id must be specified via --object-id")
    
    set_seed(cfg.seed)

    damp_tag = "damp_true" if cfg.spatial_damping else "damp_false"
    cfg.result_dir = os.path.join(cfg.result_dir, damp_tag)

    os.makedirs(cfg.result_dir, exist_ok=True)

    trainer = SoundFieldTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
