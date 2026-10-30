#!/bin/bash
set -eo pipefail

# Create the conda environment for AV-MSF.
# Requires Linux, an NVIDIA GPU, and CUDA toolkit 12.x (nvcc on PATH): fused-ssim
# is built here and gsplat compiles its CUDA kernels at the end.
#
# Usage:
#   bash setup.sh            # env name: avmsf
#   bash setup.sh myenv

ENV_NAME=${1:-avmsf}

if ! command -v nvcc >/dev/null; then
    echo "ERROR: nvcc not found; install CUDA toolkit 12.x first." >&2
    exit 1
fi

eval "$(conda shell.bash hook)"
conda create -y -n "${ENV_NAME}" -c conda-forge --override-channels python=3.10
conda activate "${ENV_NAME}"

pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128

PKGS=(
    # common
    numpy==1.26.4 scipy==1.15.3 scikit-learn==1.7.2 matplotlib==3.10.7
    pyyaml==6.0.3 tqdm==4.67.1 tyro==0.9.35 setuptools==80.9.0
    # 3DGS (vision/examples)
    gsplat==1.5.3 torchmetrics==1.8.2 tensorboard==2.19.0 viser==1.0.15
    "imageio[ffmpeg]==2.37.2" opencv-python==4.11.0.86 pillow==12.0.0 plyfile==1.1.3
    splines==0.3.3  # imported by nerfview but not declared as its dependency
    git+https://github.com/nerfstudio-project/nerfview@4538024fe0d15fd1a0e4d760f3695fc44ca72787
    git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e
    # feature 3DGS
    transformers==4.57.6 open3d==0.18.0
    # sound
    librosa==0.10.2.post1 pandas==2.3.3 cdpam==0.0.6 resampy==0.4.3
)
pip install "${PKGS[@]}"

# fused-ssim's setup.py imports torch, so it must see the torch installed above.
pip install --no-build-isolation \
    git+https://github.com/rahul-goel/fused-ssim@328dc9836f513d00c4b5bc38fe30478b4435cbb5

# Compile gsplat's CUDA kernels now instead of on the first training run.
python -c "from gsplat.cuda._backend import _C; assert _C is not None, 'gsplat CUDA build failed'"

echo "Done. Run: conda activate ${ENV_NAME}"
