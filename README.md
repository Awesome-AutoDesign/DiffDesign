# DiffDesign

![Static Badge](https://img.shields.io/badge/to_be_continue-orange)

The official repository for DiffDesign with CODE and DATASET.

## Overview

DiffDesign is a deep learning framework for interior design generation and understanding. It supports data collection, preprocessing, model training, and evaluation.

## Environment

- Python >= 3.8
- CUDA-enabled GPU recommended
- Install dependencies:
  ```sh
  pip install torch torchvision transformers diffusers pandas tqdm tap huggingface_hub pillow pyyaml
  ```
  Additional libraries such as `argtyped`, `lmdb`, `cv2`, and `datasets` may be required for specific scripts.

## Dataset

To evaluate the effectiveness of DiffDesign, quantitative experiments are divided into two parts: a comparative experiment for CLIP performance and an assessment of DiffDesign's generative performance.

- **CLIP Benchmark Datasets**:  
  
  - Flickr [[DATA]](https://www.aclweb.org/anthology/D14-1021/)
  - MSCOCO [[DATA]](https://arxiv.org/abs/1405.0312)
  - DesignHelper [[DATA]](github.com/Awesome-AutoDesign/DiffDesign)
  
- **DesignHelper Construction**:  
  The DesignHelper dataset is built by combining self-collected data with benchmark datasets containing practical interior design cases, including:
  
    - Matterport3D [[DATA]](https://arxiv.org/abs/1709.06158)
    - HM3D [[DATA]](https://arxiv.org/abs/2109.08238)
    - Zillow Indoor Dataset [[DATA]](https://arxiv.org/abs/2109.04916)
  
  Some DesignHelper data is available in this repository. For own dataset construction, you can refer to the scripts in `dataset_gathering/` and follow the instructions to collect and preprocess images and metadata from the above sources. 

## Training

1. Edit the configuration file at `configs/diffdesign.yaml` to set data paths, batch size, learning rate, etc.
2. Start training:
   ```sh
   python train_diffdesign.py
   ```
   The best model will be saved as `best_diffdesign.pt`.

## Testing & Inference

1. Generate results using the trained model:
   ```sh
   python generate_diffdesign.py
   ```
2. For batch generation and sampling, use the `DiffDesignPipeline` in `pipelines/diffdesign_pipeline.py`.

## Directory Structure

- `models/` ！ Model definitions
- `datasets/` ！ Dataset loaders
- `pipelines/` ！ Inference pipelines
- `dataset_gathering/` ！ Data collection and preprocessing scripts
- `get_empty_room/` ！ Room segmentation and furniture removal
- `configs/` ！ Configuration files
- `dataset/` ！ Text description data

## Citation

If you use DiffDesign or DesignHelper in your research, please cite the corresponding papers:

```
@article{10.1371/journal.pone.0331240,
    doi = {10.1371/journal.pone.0331240},
    author = {Geng, Tao AND Yang, Yuxuan},
    journal = {PLOS ONE},
    publisher = {Public Library of Science},
    title = {DiffDesign: Controllable diffusion with meta prior for efficient interior design generation},
    year = {2025},
    month = {09},
    volume = {20},
    url = {https://doi.org/10.1371/journal.pone.0331240},
    pages = {1-26},
    number = {9},
}
```

