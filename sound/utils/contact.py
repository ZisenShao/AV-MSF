"""Utilities for contact localization and PLY export."""

import librosa
import numpy as np


def jet_colormap(t):
    t = np.atleast_1d(np.clip(t, 0, 1))
    r = np.interp(t, [0, 0.35, 0.66, 0.89, 1.0], [0, 0, 1, 1, 0.5])
    g = np.interp(t, [0, 0.125, 0.375, 0.64, 0.91, 1.0], [0, 0, 1, 1, 0, 0])
    b = np.interp(t, [0, 0.11, 0.34, 0.65, 1.0], [0.5, 1, 1, 0, 0])
    return np.stack([r, g, b], axis=-1)

def extract_gains_at_freqs(
    audio_np, freqs_hz, sr=44100, n_fft=1024, hop_length=64
):
    """Return [M] gains from the nearest FFT bin over the full audio."""
    stft = librosa.stft(audio_np, n_fft=n_fft, hop_length=hop_length)
    mag = np.abs(stft)
    freq_bins = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    peak_by_bin = np.max(mag, axis=1)

    gains = np.zeros(len(freqs_hz), dtype=np.float32)
    for i, f in enumerate(freqs_hz):
        bin_idx = int(np.argmin(np.abs(freq_bins - f)))
        gains[i] = float(peak_by_bin[bin_idx])
    return gains

def marker(center, obj_centroid, obj_extent, scale: float = 0.7):
    """Make a diamond-shaped point."""
    s_long = obj_extent * 0.08 * scale
    s_short = obj_extent * 0.08 * scale

    direction = center - obj_centroid
    d_norm = np.linalg.norm(direction)
    if d_norm < 1e-8:
        direction = np.array([0.0, 0.0, 1.0])
    else:
        direction = direction / d_norm

    up = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(direction, up)) > 0.95:
        up = np.array([1.0, 0.0, 0.0])
    p1 = np.cross(direction, up);  p1 /= np.linalg.norm(p1)
    p2 = np.cross(direction, p1)

    pts = []
    n_grid = 35
    vals = np.linspace(-1.0, 1.0, n_grid)
    for u in vals:
        for v in vals:
            for w in vals:
                if abs(u) + abs(v) + abs(w) <= 1.0:
                    pt = (center + direction * (u * s_long)
                          + p1 * (v * s_short) + p2 * (w * s_short))
                    pts.append(pt)

    verts = np.array(pts, dtype=np.float32)
    return verts

def write_ply(out_path, points, colors_u8, diamond_verts,
              point_dup_radius=0.001, pred_diamond_verts=None):
    """
    Write the heatmap point cloud + a green diamond at GT, and
    a blue diamond at the predicted position.
    """
    points = np.ascontiguousarray(points, dtype=np.float32)
    colors_u8 = np.ascontiguousarray(colors_u8, dtype=np.uint8)
    diamond_verts = np.ascontiguousarray(diamond_verts, dtype=np.float32)

    r = point_dup_radius
    offsets = np.array([
        [r, 0, 0], [-r, 0, 0],
        [0, r, 0], [0, -r, 0],
        [0, 0, r], [0, 0, -r],
    ], dtype=np.float32)
    expanded_pts = np.concatenate(
        [points] + [points + o for o in offsets], axis=0)
    expanded_rgb = np.tile(colors_u8, (1 + len(offsets), 1))

    n_pts = len(expanded_pts)
    n_dia = len(diamond_verts)
    n_pred = 0 if pred_diamond_verts is None else len(pred_diamond_verts)
    total_v = n_pts + n_dia + n_pred

    with open(out_path, "wb") as fp:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {total_v}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )
        fp.write(header.encode("ascii"))

        buf = np.empty(n_pts, dtype=[("xyz", "<f4", 3), ("rgb", "u1", 3)])
        buf["xyz"] = expanded_pts
        buf["rgb"] = expanded_rgb
        fp.write(buf.tobytes())

        # GT diamond -> green
        dbuf = np.empty(n_dia, dtype=[("xyz", "<f4", 3), ("rgb", "u1", 3)])
        dbuf["xyz"] = diamond_verts
        dbuf["rgb"] = np.full((n_dia, 3), [0, 255, 0], dtype=np.uint8)
        fp.write(dbuf.tobytes())

        # Predicted diamond -> blue (same shape as GT diamond)
        if n_pred > 0:
            pred_verts = np.ascontiguousarray(pred_diamond_verts, dtype=np.float32)
            pbuf = np.empty(n_pred, dtype=[("xyz", "<f4", 3), ("rgb", "u1", 3)])
            pbuf["xyz"] = pred_verts
            pbuf["rgb"] = np.full((n_pred, 3), [0, 0, 255], dtype=np.uint8)
            fp.write(pbuf.tobytes())
