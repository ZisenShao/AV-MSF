"""Multi-View Visual Feature Extraction and Refinement.

Pipeline per object:
  (A) Load 3DGS ckpt.
  (B) Lift dense DINOv2 features to 3D.
  (C) Auto-detect mirror planes and rotation axes.
  (D) Iteratively pool features over accepted symmetries.
  (E) Export feature_3dgs.pt and feature_3dgs.ply.
"""

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import tyro
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from transformers import AutoImageProcessor, AutoModel
import imageio.v2 as imageio
import open3d as o3d

@dataclass
class Config:
    """Configuration for building one feature 3DGS."""

    # Object and paths
    object_id: int
    image_path: str = "vision/data/outputs/{id}"
    ckpt_path: str = "results/3dgs/{id}/ckpts/ckpt_29999_rank0.pt"
    output_path: str = "results/feature_3dgs/{id}"

    # DINOv2 lifting
    points_chunk: int = 200_000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Symmetry detection
    num_random_axes: int = 24
    detect_subset: int = 5_000
    mirror_err_thr_rel: float = 0.020
    rotation_err_thr_rel: float = 0.020
    rotation_K_list: Tuple[int, ...] = (2, 3, 4, 6, 8, 12)
    keep_top_planes: int = 3
    keep_top_axes: int = 2
    cos_dedup_thr: float = 0.97
    seed: int = 42

    # Feature refinement
    refine_iters: int = 3


# DINOv2 dense feature helpers.
@torch.no_grad()
def _load_dinov2(device: str):
    """Load DINOv2. Returns (processor, model, feat_dim, patch)."""
    repo = "facebook/dinov2-base"
    processor = AutoImageProcessor.from_pretrained(repo)
    model = AutoModel.from_pretrained(repo).eval().to(device)
    return processor, model, int(model.config.hidden_size), 14


@torch.no_grad()
def extract_dense_features_dinov2(
    processor, model, image: torch.Tensor, patch: int = 14,
) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int]]:
    """image: [1,H,W,3] in [0,1] -> (feat_map [1,C,Hf,Wf], orig_hw, padded_hw).

    Runs the HF DINOv2 at native resolution (padded to a multiple of `patch`)
    so the patch grid maps directly to the image and bilinear sampling stays valid.
    """
    assert image.ndim == 4 and image.shape[-1] == 3
    H, W = int(image.shape[1]), int(image.shape[2])

    x = image.permute(0, 3, 1, 2).contiguous()
    mean = torch.tensor(processor.image_mean, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(processor.image_std, device=x.device).view(1, 3, 1, 1)
    x = (x - mean) / std

    Hp = int(np.ceil(H / patch) * patch)
    Wp = int(np.ceil(W / patch) * patch)
    if Hp > H or Wp > W:
        x = F.pad(x, (0, Wp - W, 0, Hp - H), mode="replicate")

    tokens = model(pixel_values=x).last_hidden_state[:, 1:, :]   # drop CLS token
    B, _, C = tokens.shape
    Hf, Wf = Hp // patch, Wp // patch
    feat = tokens.transpose(1, 2).reshape(B, C, Hf, Wf).contiguous()
    return F.normalize(feat, dim=1), (H, W), (Hp, Wp)


def project_world_to_pixel(
    points_world: torch.Tensor, camtoworld: torch.Tensor, K: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project world points into pixels (camtoworld is OpenCV c2w).
    Returns (uv [M,2], valid [M] = in front of camera)."""
    w2c = torch.linalg.inv(camtoworld)
    ones = torch.ones((points_world.shape[0], 1), device=points_world.device,
                      dtype=points_world.dtype)
    Pc = (w2c @ torch.cat([points_world, ones], dim=1).t()).t()
    X, Y, Z = Pc[:, 0], Pc[:, 1], Pc[:, 2]
    valid = Z > 1e-6
    Zc = torch.clamp(Z, min=1e-6)
    u = K[0, 0] * (X / Zc) + K[0, 2]
    v = K[1, 1] * (Y / Zc) + K[1, 2]
    return torch.stack([u, v], dim=-1), valid


@torch.no_grad()
def sample_feat_at_pixels(
    feat_map: torch.Tensor, uv: torch.Tensor,
    padded_hw: Tuple[int, int], patch: int,
) -> torch.Tensor:
    """Bilinearly sample [1,C,Hf,Wf] at pixel coords uv -> [M,C] (L2-normalized)."""
    Hp, Wp = padded_hw
    Hf, Wf = Hp // patch, Wp // patch
    gx = (uv[:, 0] / patch / (Wf - 1)) * 2 - 1
    gy = (uv[:, 1] / patch / (Hf - 1)) * 2 - 1
    grid = torch.stack([gx, gy], dim=-1).view(1, -1, 1, 2)
    samp = F.grid_sample(feat_map, grid, mode="bilinear", align_corners=True)
    samp = samp.squeeze(0).squeeze(-1).transpose(0, 1).contiguous()
    return F.normalize(samp, dim=-1)


class TransformsParser:
    """Loads camera poses + one shared pinhole intrinsic from
    `<data_dir>/transforms.json`. Exposes the attributes used by the lifting
    loop: image_paths, camtoworlds, K."""

    # diag(1,-1,-1,1): flips camera-local Y/Z -> Blender(OpenGL) c2w to OpenCV c2w.
    _GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)

    def __init__(self, data_dir: str):

        with open(os.path.join(data_dir, "transforms.json"), "r") as f:
            meta = json.load(f)
        frames = sorted(meta.get("frames", []),
                        key=lambda fr: fr.get("file_path", ""))
        if not frames:
            raise ValueError(f"No frames in {data_dir}/transforms.json")

        image_paths, camtoworlds = [], []
        for fr in frames:
            path = self._resolve(data_dir, fr["file_path"])
            if path is None:
                print(f"[Parser] missing image: {fr['file_path']}")
                continue
            c2w_gl = np.array(fr["transform_matrix"], dtype=np.float32)
            camtoworlds.append(c2w_gl @ self._GL_TO_CV)   # -> OpenCV c2w
            image_paths.append(path)
        if not image_paths:
            raise RuntimeError(f"No images found under {data_dir}")

        # Single pinhole intrinsic (fx == fy, centered principal point).
        H, W = imageio.imread(image_paths[0]).shape[:2]
        fovx = float(frames[0].get("camera_angle_x", meta.get("camera_angle_x")))
        fx = (W * 0.5) / np.tan(fovx * 0.5)
        K = np.array([[fx, 0.0, (W - 1) * 0.5],
                      [0.0, fx, (H - 1) * 0.5],
                      [0.0, 0.0, 1.0]], dtype=np.float32)

        self.image_paths = image_paths
        self.camtoworlds = np.stack(camtoworlds, axis=0)
        self.K = K

    @staticmethod
    def _resolve(data_dir: str, file_path: str) -> Optional[str]:
        name = os.path.basename(file_path)
        for cand in (os.path.join(data_dir, file_path),
                     os.path.join(data_dir, name),
                     os.path.join(data_dir, "images", name)):
            if os.path.exists(cand):
                return cand
        return None


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# 1. DINOv2 lifting
@torch.no_grad()
def lift_dino_features_to_points(
    means: torch.Tensor,        # [N, 3] world-space Gaussian centers
    parser: "TransformsParser",  # multi-view loader (already does GL->CV)
    cfg: Config,
    log_prefix: str = "",
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Project each point into every view, sample dense DINO feats, average.
    Returns (feat_mean [N,C] L2-normalized, count [N] views-per-point, feat_dim).
    """

    device = cfg.device
    N = means.shape[0]
    means = means.to(device)

    print(f"{log_prefix}Loading DINOv2 [dinov2_vitb14] ...")
    processor, dino_model, feat_dim, patch = _load_dinov2(device)

    feat_sum = torch.zeros((N, feat_dim), device=device, dtype=torch.float32)
    feat_cnt = torch.zeros((N,), device=device, dtype=torch.float32)
    K = torch.from_numpy(parser.K.copy()).float().to(device)
    num_images = len(parser.image_paths)
    print(f"{log_prefix}Lifting features across {num_images} views (N={N}, C={feat_dim})")

    for idx in tqdm(range(num_images), desc=f"{log_prefix}views", leave=False):
        img_np = imageio.imread(parser.image_paths[idx])[..., :3]    # [H, W, 3]
        image = torch.from_numpy(img_np).float().unsqueeze(0).to(device) / 255.0
        H, W = img_np.shape[:2]

        camtoworld = torch.from_numpy(parser.camtoworlds[idx]).float().to(device)

        feat_map, _, padded_hw = extract_dense_features_dinov2(
            processor, dino_model, image, patch=patch
        )  # [1, C, Hf, Wf]

        for start in range(0, N, cfg.points_chunk):
            end = min(start + cfg.points_chunk, N)
            pts = means[start:end]
            uv, valid = project_world_to_pixel(pts, camtoworld, K)
            inside = (
                (uv[:, 0] >= 0) & (uv[:, 0] <= (W - 1)) &
                (uv[:, 1] >= 0) & (uv[:, 1] <= (H - 1))
            )
            mask = valid & inside
            if not mask.any():
                continue
            feat_sel = sample_feat_at_pixels(
                feat_map, uv[mask], padded_hw, patch=patch
            )
            chunk_idx = torch.arange(start, end, device=device)[mask]
            feat_sum[chunk_idx] += feat_sel
            feat_cnt[chunk_idx] += 1.0

    cnt_safe = torch.clamp(feat_cnt, min=1.0).unsqueeze(-1)
    feat_mean = feat_sum / cnt_safe
    feat_mean = F.normalize(feat_mean, dim=-1)
    feat_mean[feat_cnt == 0] = 0.0

    del dino_model, feat_sum
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return feat_mean, feat_cnt, feat_dim


# 2. Symmetry detection
@dataclass
class MirrorPlane:
    normal: np.ndarray            # unit vec (3,)
    error: float                  # mean NN dist normalized by diameter

    def to_dict(self) -> dict:
        return {"normal": self.normal.tolist(), "error": float(self.error)}


@dataclass
class RotationAxis:
    axis: np.ndarray              # unit vec (3,)
    K: int                        # K-fold rotational symmetry
    error: float                  # worst-angle mean NN dist normalized by diameter

    def to_dict(self) -> dict:
        return {"axis": self.axis.tolist(), "K": int(self.K),
                "error": float(self.error)}


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _dedup_unsigned_dirs(items: Sequence, get_vec, thr: float) -> List:
    """Deduplicate items by direction."""
    kept = []
    for it in items:
        v = _unit(get_vec(it))
        is_dup = False
        for k in kept:
            if abs(float(np.dot(v, _unit(get_vec(k))))) > thr:
                is_dup = True
                break
        if not is_dup:
            kept.append(it)
    return kept


def _rotate(points: np.ndarray, axis: np.ndarray, angle: float,
            center: np.ndarray) -> np.ndarray:
    """Rotate points around (center, axis) by angle (Rodrigues)."""
    a = _unit(axis)
    c, s = np.cos(angle), np.sin(angle)
    K = np.array([[0.0, -a[2], a[1]],
                  [a[2], 0.0, -a[0]],
                  [-a[1], a[0], 0.0]], dtype=np.float64)
    R = np.eye(3, dtype=np.float64) * c + (1 - c) * np.outer(a, a) + s * K
    return (points - center) @ R.T + center


def _reflect(points: np.ndarray, normal: np.ndarray,
             center: np.ndarray) -> np.ndarray:
    """Reflect points through plane (center, normal)."""
    n = _unit(normal)
    delta = points - center
    return points - 2.0 * (delta @ n)[:, None] * n[None, :]


def _alignment_error(
    transformed: np.ndarray,   # [M, 3]
    nbrs: NearestNeighbors,    # KDTree on FULL reference cloud
    diameter: float,
) -> float:
    """Mean NN distance / diameter (lower = tighter symmetry; ~0.02 threshold)."""
    d, _ = nbrs.kneighbors(transformed, n_neighbors=1, return_distance=True)
    return float(d.mean() / max(diameter, 1e-9))


def _candidate_axes(P_full: np.ndarray, cfg: Config) -> np.ndarray:
    """Build candidate unit vectors: 3 PCA principal axes + N random."""
    center = P_full.mean(axis=0, keepdims=True)
    P0 = P_full - center
    sub = P0 if P0.shape[0] <= 20_000 else P0[
        np.random.default_rng(cfg.seed).choice(P0.shape[0], 20_000, replace=False)
    ]
    _, _, Vt = np.linalg.svd(sub - sub.mean(0, keepdims=True), full_matrices=False)
    cand = [_unit(Vt[i]) for i in range(3)]
    rng = np.random.default_rng(cfg.seed)
    for _ in range(cfg.num_random_axes):
        v = rng.normal(size=3)
        cand.append(_unit(v))
    cand = [v if v[np.argmax(np.abs(v))] >= 0 else -v for v in cand]
    return np.stack(cand, axis=0)


def detect_symmetries(P_full: np.ndarray,
                      cfg: Config
                      ) -> Tuple[List[MirrorPlane], List[RotationAxis],
                                 np.ndarray, float]:
    """Detect mirror planes and rotation axes via alignment-error validation.

    Returns:
        mirrors  : list[MirrorPlane]
        rotations: list[RotationAxis]
        center   : (3,) centroid used for all transforms
        diameter : float -- max bbox extent (used to normalize errors)
    """
    rng = np.random.default_rng(cfg.seed)

    center = P_full.mean(axis=0).astype(np.float64)
    diameter = float((P_full.max(0) - P_full.min(0)).max())

    if P_full.shape[0] > cfg.detect_subset:
        idx = rng.choice(P_full.shape[0], cfg.detect_subset, replace=False)
        P_query = P_full[idx].astype(np.float64)
    else:
        P_query = P_full.astype(np.float64)

    nbrs = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(
        P_full.astype(np.float64)
    )

    candidates = _candidate_axes(P_full, cfg)

    # Mirror planes
    mirror_scored: List[MirrorPlane] = []
    for v in candidates:
        P_ref = _reflect(P_query, v, center)
        err = _alignment_error(P_ref, nbrs, diameter)
        if err < cfg.mirror_err_thr_rel:
            mirror_scored.append(MirrorPlane(normal=v.astype(np.float64),
                                             error=err))
    mirror_scored.sort(key=lambda m: m.error)
    mirror_scored = _dedup_unsigned_dirs(
        mirror_scored, lambda m: m.normal, thr=cfg.cos_dedup_thr
    )[: cfg.keep_top_planes]

    # Rotation axes
    rot_scored: List[RotationAxis] = []
    for v in candidates:
        best_K, best_err = 0, float("inf")
        for K in cfg.rotation_K_list:
            # Worst-angle error over non-identity rotations of order K (all must
            # be small for a true K-fold axis). Keep the largest passing K.
            worst = 0.0
            for j in range(1, K):
                worst = max(worst, _alignment_error(
                    _rotate(P_query, v, 2 * np.pi * j / K, center), nbrs, diameter))
                if worst >= cfg.rotation_err_thr_rel:
                    break  # early stop
            if worst < cfg.rotation_err_thr_rel and K > best_K:
                best_K, best_err = K, worst
        if best_K >= 2:
            rot_scored.append(RotationAxis(axis=v.astype(np.float64),
                                           K=best_K, error=best_err))
    rot_scored.sort(key=lambda r: (-r.K, r.error))
    rot_scored = _dedup_unsigned_dirs(
        rot_scored, lambda r: r.axis, thr=cfg.cos_dedup_thr
    )[: cfg.keep_top_axes]

    return mirror_scored, rot_scored, center, diameter


# 3. Symmetry-aware feature refinement
def _pool_with_nn(F_full: np.ndarray, P_transformed: np.ndarray,
                  nbrs_full: NearestNeighbors) -> np.ndarray:
    """For each transformed point, pick the feature of its nearest neighbor in
    the FULL reference cloud. Returns [N, C]."""
    _, idx = nbrs_full.kneighbors(P_transformed, n_neighbors=1,
                                  return_distance=True)
    return F_full[idx[:, 0]]


def refine_features(
    P_full: np.ndarray,                # [N, 3]
    F_full: np.ndarray,                # [N, C]
    mirrors: Sequence[MirrorPlane],
    rotations: Sequence[RotationAxis],
    center: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    """Iteratively pool features over detected symmetries (Supp A.1 Eq. 1-3).

    Orbit pooling (Eq. 1): f_i <- mean over theta in {2pi*j/K} of f[NN(R_theta o_i)]
    (j=0 is identity). Mirror pooling (Eq. 2): f_i <- 1/2 (f_i + f[NN(reflect o_i)]).
    Apply orbit then mirror per step, repeat `refine_iters` times.
    """
    if not mirrors and not rotations:
        return F_full.copy()

    nbrs = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(
        P_full.astype(np.float64)
    )

    F_curr = F_full.copy()
    P64 = P_full.astype(np.float64)

    for _ in range(cfg.refine_iters):
        # (a) Orbit pooling over each rotation axis
        for r in rotations:
            angles = [2 * np.pi * j / r.K for j in range(r.K)]
            acc = np.zeros_like(F_curr)
            for ang in angles:
                if ang == 0.0:
                    acc += F_curr
                else:
                    P_rot = _rotate(P64, r.axis, ang, center)
                    acc += _pool_with_nn(F_curr, P_rot, nbrs)
            F_curr = acc / len(angles)

        # (b) Mirror pooling over each plane
        for m in mirrors:
            P_ref = _reflect(P64, m.normal, center)
            F_curr = 0.5 * (F_curr + _pool_with_nn(F_curr, P_ref, nbrs))

    norms = np.linalg.norm(F_curr, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    F_curr = F_curr / norms
    return F_curr

def write_feature_pca_ply(out_path: Path, points: np.ndarray,
                          features: np.ndarray) -> None:

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = features.shape[0]
    if n < 3:
        rgb = np.zeros((n, 3), dtype=np.float32)
        d = min(3, features.shape[1])
        rgb[:, :d] = features[:, :d]
    else:
        k = min(3, n, features.shape[1])
        rgb = np.zeros((n, 3), dtype=np.float32)
        rgb[:, :k] = PCA(n_components=k).fit_transform(features).astype(np.float32)

    lo = np.percentile(rgb, 1, axis=0, keepdims=True)
    hi = np.percentile(rgb, 99, axis=0, keepdims=True)
    rgb = np.clip(rgb, lo, hi)
    rgb -= rgb.min(axis=0, keepdims=True)
    rgb /= rgb.max(axis=0, keepdims=True) + 1e-12

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))
    o3d.io.write_point_cloud(str(out_path), pcd)


# 4. Per-object pipeline
def _load_means(ckpt_path: Path, device: str) -> Tuple[torch.Tensor, dict]:
    print(f"  Loading 3DGS ckpt: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    means = ckpt["splats"]["means"].to(device).float()
    return means, ckpt


def process_object(cfg: Config) -> None:
    """Run the full feature-3DGS pipeline for one object."""
    object_id = str(cfg.object_id)
    data_dir = Path(cfg.image_path.format(id=object_id))
    ckpt_path = Path(cfg.ckpt_path.format(id=object_id))
    out_dir = Path(cfg.output_path.format(id=object_id))
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ckpt_path.exists():
        print(f"[skip {object_id}] missing ckpt: {ckpt_path}")
        return

    if not data_dir.exists():
        print(f"[skip {object_id}] missing data dir: {data_dir}")
        return

    out_feat_ckpt = out_dir / "feature_3dgs.pt"
    means, ckpt = _load_means(ckpt_path, cfg.device)
    means_np = means.cpu().numpy().astype(np.float32)
    N = means_np.shape[0]
    print(f"  Gaussians: N={N}")

    print(f"  [symmetry detection] subset={cfg.detect_subset}, "
          f"thr={cfg.mirror_err_thr_rel}")
    _set_seed(cfg.seed)
    mirrors, rotations, center, diameter = detect_symmetries(means_np, cfg)
    print(f"    diameter: {diameter:.4f}")
    print(f"    accepted mirrors  ({len(mirrors)}): "
          + ", ".join(f"err={m.error:.4f}" for m in mirrors))
    print(f"    accepted rotations ({len(rotations)}): "
          + ", ".join(f"K={r.K},err={r.error:.4f}" for r in rotations))

    print("  [feature lifting]")
    parser = TransformsParser(str(data_dir))
    feat_mean, feat_cnt, feat_dim = lift_dino_features_to_points(
        means, parser, cfg, log_prefix="    "
    )
    feat_np = feat_mean.cpu().numpy().astype(np.float32)
    valid_np = feat_cnt.cpu().numpy() > 0
    print(f"    observed: {valid_np.sum()} / {N} ({100*valid_np.mean():.1f}%)")

    # Symmetry-aware feature refinement
    if mirrors or rotations:
        print(f"  [refinement] iters={cfg.refine_iters}, "
              f"mirrors={len(mirrors)}, rotations={len(rotations)}")
        feat_refined = refine_features(
            means_np.astype(np.float64), feat_np.astype(np.float64),
            mirrors, rotations, center, cfg,
        ).astype(np.float32)
    else:
        print("  [refinement] no symmetry detected -> identity refine")
        feat_refined = feat_np

    feat_refined[~valid_np] = 0.0

    splats = ckpt["splats"]
    splats["features"] = torch.from_numpy(feat_refined)
    out_ckpt = {
        "step": ckpt.get("step", 29999),
        "splats": splats,
        "feature_meta": {
            "dim": int(feat_dim),
            "model": "dinov2_vitb14",
            "refinement": {
                "iters": cfg.refine_iters,
                "mirrors": [m.to_dict() for m in mirrors],
                "rotations": [r.to_dict() for r in rotations],
            },
        },
    }
    torch.save(out_ckpt, out_feat_ckpt)
    print(f"    feature ckpt   -> {out_feat_ckpt}")

    write_feature_pca_ply(out_dir / "feature_3dgs.ply", means_np, feat_refined)


def main(cfg: Config) -> None:
    _set_seed(cfg.seed)
    print(f"[Config] object_id={cfg.object_id}, dino_model=dinov2_vitb14, "
          f"seed={cfg.seed}")
    process_object(cfg)


if __name__ == "__main__":
    main(tyro.cli(Config))