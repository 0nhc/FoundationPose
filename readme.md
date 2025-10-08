# Customized FoundationPose with RTX 40 Series GPU and CUDA 12.1

## Installation
To install Python environment:
```sh
conda create -n foundationpose python=3.10 # The same system Python version as Ubuntu 22.04
conda activate foundationpose
conda install conda-forge::eigen=3.4.0
export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:/eigen/path/under/conda"
python -m pip install -r requirements.txt
python -m pip install --quiet --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git

# See https://github.com/facebookresearch/pytorch3d/discussions/1752
pip install --extra-index-url https://miropsota.github.io/torch_packages_builder pytorch3d==0.7.8+5043d15pt2.1.0cu121 

CMAKE_PREFIX_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/pybind11/share/cmake/pybind11 bash build_all_conda.sh
```

Download weights from [https://drive.google.com/drive/folders/1BEQLZH69UO5EOfah-K9bfI3JyP9Hf7wC?usp=drive_link](https://drive.google.com/drive/folders/1BEQLZH69UO5EOfah-K9bfI3JyP9Hf7wC?usp=drive_link). Unzip and put them into `./weights/`. Rename the folder names as `2023-10-28-18-33-37` and `2024-01-11-20-02-45`.

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

## Original README
See [original_readme](original_readme.md).