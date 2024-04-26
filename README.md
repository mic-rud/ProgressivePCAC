# Scalable Coding


## Setup
```
    # Python
    python -m venv .env
    source .env/bin/activate
    python -m pip install -r requirements.txt

    # MinkowskiEngine
    sudo apt install build-essential libopenblas-dev
    python -m pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps

    # Open3D
    sudo apt-get install libosmesa6-dev
    mkdir dependencies && cd dependencies
    git clone https://github.com/isl-org/Open3D
    cd Open3D
    util/install_deps_ubuntu.sh
    mkdir build && cd build
    cmake -DENABLE_HEADLESS_RENDERING=ON \
                    -DBUILD_GUI=OFF \
                    -DBUILD_WEBRTC=OFF \
                    -DUSE_SYSTEM_GLEW=OFF \
                    -DUSE_SYSTEM_GLFW=OFF \
                    ..
    make -j$(nproc)
    make install-pip-package
    cd ../../..
```
## Dataloading
```
    cd data
    python download_raw_pointclouds.py

    #DTD dataset for projecting textures
    cd datasets
    mkdir dtd && cd dtd
    cd ../../..
```

## Rerun evaluation
```
    python evaluate.py
```

## Re-Train models
Sample configurations for training a model can be found in [./configs](./configs). 
For re-training our model, use
```
    python train.py --config=configs/MeanScale_5_lambda400-6400.yaml
```

Configurations for the comparison against a reimplemented version of [https://ieeexplore.ieee.org/document/9874468/citations?tabFilter=papers#citations](SparsePCAC) have the indicator "_1_"


