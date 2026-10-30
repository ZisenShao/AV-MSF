# Objects as Audio-Visual Modal Sound Fields (ECCV 2026)

[![Paper](https://img.shields.io/badge/arXiv-2608.05145-b31b1b.svg)](https://arxiv.org/pdf/2608.05145)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://zisenshao.github.io/AV-MSF/)
[![Video](https://img.shields.io/badge/YouTube-Video-red.svg?logo=youtube)](https://www.youtube.com/watch?v=LaNWn3mkHkY)
[![BibTeX](https://img.shields.io/badge/Cite-BibTeX-lightgrey.svg)](#citation)

## Environment

Python 3.10, CUDA 12.x.

```bash
git clone --single-branch --branch main https://github.com/ZisenShao/AV-MSF.git
cd AV-MSF
bash setup.sh
conda activate avmsf
```

## Dataset Preparation

Download [ObjectFolder-Real](https://objectfolder.stanford.edu/) to
`data/ObjectFolder-Real`:

```
data/ObjectFolder-Real/
├── meshes/
│   ├── 16k/<id>/
│   ├── 64k/<id>/
│   └── full/<id>/
│       ├── model.obj
│       ├── model.mtl
│       └── texture.png
└── <id>/
    ├── audio/
    │   └── <idx>/
    │       ├── mic.wav
    │       ├── Force.wav
    │       ├── striking_force.yaml
    │       ├── metadata.yaml
    │       ├── force_audio.png
    │       └── cam/
    ├── tactile_data/
    │   ├── obj_points.yaml
    │   ├── obj_mesh_t.ply
    │   ├── obj_transformation.p
    │   └── <idx>/
    └── video.MP4
```

> **Note:** Some objects in the dataset have defects, such as heavy noise or
> contact positions that do not match the recordings. For example, objects 62,
> 64, 85, and 88 in ObjectFolder-Real have a different number of
> `points_to_contact` entries than `audio` recordings, so we recommend not
> evaluating on them.

## Pipeline

To choose which objects to run, update `OBJECT_IDS` in each bash script.

### 1. Render vision data

Render object meshes into multi-view images, camera poses, and a point cloud.

```bash
bash vision/data/render.sh
```

Output: `vision/data/outputs/<id>/` with images, `transforms.json`, and `mesh.ply`.
Objects with both metadata files already present are skipped.

### 2. Train 3D Gaussian Splatting

Train one 3DGS model per object from the stage 1 renders.

```bash
bash run_3dgs.sh
```

Output: `results/3dgs/<id>/ckpts/ckpt_29999_rank0.pt`.

### 3. Build feature 3DGS

Lift DINOv2 features onto the trained Gaussians and refine them using symmetry.
The `facebook/dinov2-base` weights must be cached or downloadable.

```bash
bash run_feature_3dgs.sh
```

Output: `results/feature_3dgs/<id>/feature_3dgs.pt` and `feature_3dgs.ply`.

### 4. Prepare impact-sound data

Read the raw audio and contact points, then create the impact metadata and
train/test split. This stage can run independently of stages 1–3.

```bash
bash sound/data/impact_sound_data.sh
```

Output: `sound/data/outputs/<id>/{impact_sound.json,partition.json}`.

### 5. Train sound fields and localize contacts

Use the stage 3 features and stage 4 impact data. The wrapper trains both spatial
and shared damping variants, then evaluates sound synthesis and contact localization.

```bash
bash run_sound_field_trainer.sh configs/train.yaml
```

Output: `results/train/<id>/{damp_true,damp_false}/`, including `sound_field.pt`, `metrics.json`, per-step `eval/` metrics,
rendered audio and plots under `renders/`, and contact-localization PLYs under
`contact/`.

## Citation

```bibtex
@inproceedings{shao2026avmsf,
  title={Objects as Audio-Visual Modal Sound Fields},
  author={Shao, Zisen and Wei, Zihao and Jin, Derong and Gao, Ruohan},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2026}
}
```

## Acknowledgements

We thank the authors of [DiffImpact](https://github.com/samuel-clarke/diffimpact)
and [gsplat](https://github.com/nerfstudio-project/gsplat) for releasing their code.
