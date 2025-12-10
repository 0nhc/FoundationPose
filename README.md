# Customized FoundationPose with RTX 40 Series GPU and CUDA 12.1

This is my customized repo for FoundationPose running on RTX 40 Series GPU with CUDA 12.1. See detailed original [README](original_readme.md).

## Installation
To install Python environment:
```sh
micromamba create -n foundationpose python=3.10
micromamba activate foundationpose
micromamba install -c conda-forge eigen=3.4.0 boost cuda-toolkit=12.1

python -m pip install -r requirements.txt
python -m pip install --quiet --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git

# See https://github.com/facebookresearch/pytorch3d/discussions/1752
pip install --extra-index-url https://miropsota.github.io/torch_packages_builder pytorch3d==0.7.8+5043d15pt2.1.0cu121 

CMAKE_PREFIX_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/pybind11/share/cmake/pybind11 bash build_all_conda.sh

# Download weights
cd weights
./download_weights.sh
```

Download demo data from [https://drive.google.com/drive/folders/1pRyFmxYXmAnpku7nGRioZaKrVJtIsroP?usp=sharing](https://drive.google.com/drive/folders/1pRyFmxYXmAnpku7nGRioZaKrVJtIsroP?usp=sharing). Unzip and put them into `./demo_data/`.



## Quick Start
To run the official demo:
```sh
python run_demo.py
```

To run with a Realsense camera:
```sh
pip install pyrealsense2
python realsense_demo.py 
```
