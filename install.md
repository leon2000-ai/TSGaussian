### **Standard Installation**


Clone the repository locally
```
git clone https://github.com/leon2000-ai/TSGaussian.git
cd TSGaussian
```

Our default, provided install method is based on Conda package and environment management:
```bash
conda create -n TSGaussian python=3.8 -y
conda activate TSGaussian

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install plyfile==0.8.1
pip install tqdm scipy wandb opencv-python scikit-learn lpips open3d

pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```

(Optional) If you want to prepare masks on your own dataset, you will also need to prepare [DEVA](https://github.com/hkchengrex/Tracking-Anything-with-DEVA) environment.

```bash
cd Tracking-Anything-with-DEVA
pip install -e .
bash scripts/download_models.sh     # Download the pretrained models

git clone https://github.com/hkchengrex/Grounded-Segment-Anything.git
cd Grounded-Segment-Anything
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
python -m pip install -e segment_anything
python -m pip install -e GroundingDINO

cd ../..
```

