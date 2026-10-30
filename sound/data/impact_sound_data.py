"""
Build per-object impact-sound dataset for ObjectFolder-Real.

Reads:
  - data/ObjectFolder-Real/<id>/tactile_data/obj_points.yaml  (raw 3D contact positions)
  - data/ObjectFolder-Real/<id>/audio/<idx>/{mic.wav, Force.wav}  (waveforms)
  - data/ObjectFolder-Real/<id>/audio/<idx>/striking_force.yaml  (peak force)

Writes per object (default sound/data/outputs/<id>/):
  - impact_sound.json     entries: impact_id, position_3dgs,
                          mic_waveform, force_waveform, force_newtons
  - partition.json        FPS-based train / test split over impact positions

Usage:
  python sound/data/impact_sound_data.py --object-id 80
  python sound/data/impact_sound_data.py --object-ids 1 2 4 80 100
  python sound/data/impact_sound_data.py                       # default 1..100
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "ObjectFolder-Real"


# IO helpers
def load_points_to_contact(obj_points_file: Path) -> np.ndarray:
    """Load `points_to_contact` (N, 3) from obj_points.yaml. RAW positions."""
    with obj_points_file.open() as fh:
        data = yaml.safe_load(fh)
    return np.asarray(data["points_to_contact"], dtype=np.float64)


def load_striking_force(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    with path.open() as f:
        data = yaml.safe_load(f)
    if isinstance(data, list):
        for e in data:
            if isinstance(e, dict) and "striking_force_newtons" in e:
                return float(e["striking_force_newtons"])
    elif isinstance(data, dict) and "striking_force_newtons" in data:
        return float(data["striking_force_newtons"])
    return None


def farthest_point_sampling(positions: np.ndarray, n_train: int) -> List[int]:
    n_total = len(positions)
    if n_train >= n_total:
        return list(range(n_total))

    centroid = positions.mean(axis=0)
    distances_to_centroid = np.linalg.norm(positions - centroid, axis=1)
    first_idx = int(np.argmin(distances_to_centroid))

    selected = [first_idx]
    min_distances = np.full(n_total, np.inf)
    for _ in range(n_train - 1):
        last = selected[-1]
        dist_to_last = np.linalg.norm(positions - positions[last], axis=1)
        min_distances = np.minimum(min_distances, dist_to_last)
        min_distances[selected] = -np.inf
        nxt = int(np.argmax(min_distances))
        selected.append(nxt)
    return selected


def build_impact_sound(object_id: int, obj_points_file: Path, audio_root: Path) -> Dict[str, Any]:
    impact_points = load_points_to_contact(obj_points_file)

    entries: List[Dict[str, Any]] = []
    for idx in range(len(impact_points)):
        impact_dir = audio_root / str(idx)
        mic_path = impact_dir / "mic.wav"
        force_path = impact_dir / "Force.wav"

        entries.append({
            "impact_id": idx,
            "position_3dgs": impact_points[idx].tolist(),
            "mic_waveform": str(mic_path),
            "force_waveform": str(force_path) if force_path.exists() else None,
            "force_newtons": load_striking_force(impact_dir / "striking_force.yaml"),
        })
    return {"object_id": object_id, "entries": entries}


def build_partition(payload: Dict[str, Any], train_ratio: float) -> Dict[str, Any]:
    entries = payload["entries"]
    n_total = len(entries)
    n_train = max(1, int(n_total * train_ratio))

    positions = np.asarray([e["position_3dgs"] for e in entries], dtype=np.float64)
    train_indices = sorted(farthest_point_sampling(positions, n_train))
    test_indices = sorted(set(range(n_total)) - set(train_indices))

    return {
        "object_id": payload["object_id"],
        "train_ratio": train_ratio,
        "n_total": n_total,
        "train_indices": train_indices,
        "test_indices": test_indices,
    }


def process_object(args: argparse.Namespace, object_id: int) -> bool:
    obj_points_file = Path(args.obj_points_file.format(object_id=object_id))
    audio_root = Path(args.audio_root.format(object_id=object_id))

    if not obj_points_file.exists():
        print(f"[obj {object_id}] skip: missing {obj_points_file}")
        return False

    out_dir = args.output_dir / str(object_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = build_impact_sound(object_id, obj_points_file, audio_root)
    impact_path = out_dir / "impact_sound.json"
    with impact_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"[obj {object_id}] {len(payload['entries'])} entries -> {impact_path}")

    partition = build_partition(payload, args.train_ratio)
    partition_path = out_dir / "partition.json"
    with partition_path.open("w") as f:
        json.dump(partition, f, indent=2)
    print(
        f"[obj {object_id}] {len(partition['train_indices'])} train / "
        f"{len(partition['test_indices'])} test -> {partition_path}"
    )
    return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--object-id", type=int, default=None,
                   help="Single object id; takes precedence over --object-ids")
    p.add_argument("--object-ids", type=int, nargs="*", default=None,
                   help="Explicit list of object ids (default: 1..100)")
    p.add_argument("--obj-points-file", type=str,
                   default=str(DATA_ROOT / "{object_id}" / "tactile_data" / "obj_points.yaml"))
    p.add_argument("--audio-root", type=str,
                   default=str(DATA_ROOT / "{object_id}" / "audio"))
    p.add_argument("--output-dir", type=Path,
                   default=REPO_ROOT / "sound" / "data" / "outputs")
    p.add_argument("--train-ratio", type=float, default=0.2,
                   help="Fraction of impacts kept for training via FPS (rest is test)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.object_id is not None:
        ids = [args.object_id]
    elif args.object_ids:
        ids = args.object_ids
    else:
        ids = list(range(1, 101))

    ok = 0
    for oid in ids:
        if process_object(args, oid):
            ok += 1
    print(f"done: ok={ok}/{len(ids)}")


if __name__ == "__main__":
    main()
