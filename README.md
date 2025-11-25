# Animal Intrusion Detection System — SSDLite + MobileNetV3

This repository contains an implementation of an Animal Intrusion Detection System (ADIS) built around the SSDLite object detector with a MobileNetV3 backbone. The project provides training and inference utilities, example notebooks, model checkpoints, and helper modules to run detection experiments on a balanced animal dataset.

Key goals:
- Lightweight detector for on-device or resource-constrained environments using SSDLite + MobileNetV3
- Reproducible training pipelines and hyperparameter tuning (Optuna / BOHB)
- Easy-to-use inference and evaluation notebooks for visualizing detections and computing metrics

## Highlights / Features

- SSDLite MobileNetV3 model implementation (see `ssdlite_mobnetv3_adis/model.py`)
- Cached dataset support for faster I/O using LMDB (see `ssdlite_mobnetv3_adis/dataset.py`)
- Training utilities, schedulers and a BOHB-compatible tuning wrapper (see `trainer.py`, notebooks)
- Inference helpers and plotting utilities for visualizing detections (see `inference.py`, `plot.py`)
- Example Jupyter notebooks for training/tuning and inference:
	- `SSDLite_MobileNetV3_Bohbtune_Training.ipynb` (hyperparameter tuning + training)
	- `SSDLite_MobileNetV3_Inference.ipynb` (load checkpoints, evaluate and visualize)

## Repository structure

```
LICENSE
README.md
requirements.txt
SSDLite_MobileNetV3_Bohbtune_Training.ipynb
SSDLite_MobileNetV3_Inference.ipynb
ckpt/
		ssdlite_mobnetv3_bestparams_ckpt.pth
		ssdlite_mobv3_custom_params_ckpt.pth
ssdlite_mobnetv3_adis/
		__init__.py
		bohbtune.py
		dataset.py
		evaluate.py
		inference.py
		model.py
		plot.py
		trainer.py
		utils.py
study/
```

## Prerequisites

- Python 3.8+ recommended (code tested with Python 3.8–3.11)
- For GPU training: a CUDA-enabled system with a compatible PyTorch build
- Basic packages are declared in `requirements.txt` (torch, torchvision, albumentations, optuna, etc.)

Install dependencies into a virtual environment (PowerShell example):

```powershell
# create & activate venv (Windows PowerShell)
python -m venv .venv; .\.venv\Scripts\Activate.ps1

# upgrade pip and install required packages
python -m pip install --upgrade pip; pip install -r requirements.txt
```

Notes:
- If you plan to use GPU acceleration, install a PyTorch build that matches your CUDA version. See https://pytorch.org for wheel selection.
- On some environments you may prefer conda instead of venv.

## Dataset

This project expects the ADIS dataset (balanced animal dataset) used in the example notebooks. The notebooks use `huggingface_hub.hf_hub_download` to fetch `pt-sk/ADIS` (dataset repo) and extract `balanced_dataset.zip` into the working directory. If you already have the dataset, point the dataset loader to the dataset root folder.

Classes (example):
```
['Cat', 'Cattle', 'Chicken', 'Deer', 'Dog', 'Squirrel', 'Eagle', 'Goat', 'Rodents', 'Snake']
```

## Quick usage

1) Run the inference notebook to verify the model loads and produces detections.

Open `SSDLite_MobileNetV3_Inference.ipynb` in Jupyter / VS Code and update `best_ckpt_path` to the full path of one of the checkpoint files in `ckpt/`, for example:

```python
best_ckpt_path = "./ckpt/ssdlite_mobnetv3_bestparams_ckpt.pth"
```

The notebook contains helper code to build the model, load the checkpoint, prepare dataloaders and then run `draw_detections(...)` for a sample image.

2) Run the training/tuning notebook to reproduce training or hyperparameter tuning (Optuna).

Open `SSDLite_MobileNetV3_Bohbtune_Training.ipynb` and follow the cells. The notebook demonstrates constructing cached datasets, building the model (with EPU activation replacement), and running an Optuna-based BOHB tuning session. After tuning you can train a final model using the best hyperparameters.

Example: run training script components from Python (very small snippet):

```python
from ssdlite_mobnetv3_adis.model import SSDLITE_MOBILENET_V3_Large
import torch

# instantiate model
model = SSDLITE_MOBILENET_V3_Large(num_classes_with_bg=11)

# load checkpoint
ckpt = torch.load("./ckpt/ssdlite_mobnetv3_bestparams_ckpt.pth", map_location="cpu")
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# inference helpers are in ssdlite_mobnetv3_adis.inference
```

## Checkpoints

Two example checkpoints are included in the `ckpt/` directory. Use them as starting points for inference or fine-tuning. The checkpoint format is a PyTorch state-dict style dict; load with `torch.load(..., map_location=...)` and call `model.load_state_dict(...)`.

## Development notes

- The `ssdlite_mobnetv3_adis` package contains modular code for dataset loading, model implementation, training and evaluation.
- If you add new features, please keep the API of key modules stable (model forward should follow SSD-style: forward(images, targets) -> loss_dict during training and forward(images) -> detections during inference where applicable).

## Tests / Validation

The notebooks provide a quick manual validation flow. For automated tests, consider adding a small unit test that:
- Instantiates the model with a small random input tensor and confirms a forward pass returns a dictionary of losses (training mode) or detection tensors (inference mode).

## License

This project is released under the MIT License — see `LICENSE` for details.

## Acknowledgements

Special thanks to the PyTorch team for providing the framework, reference implementations and many helpful example repositories — this project builds on top of those tools and examples. Additional thanks to the maintainers and contributors of the following projects and libraries that were used or referenced while building this repository:

- Hugging Face (huggingface_hub) — for dataset hosting utilities used in the notebooks
- Albumentations — strong image augmentation primitives
- Optuna — for hyperparameter search and the BOHB interface
- OpenCV, Pillow, NumPy, and other open-source libraries used in preprocessing and visualization

If you found the project helpful or used it in your work, attribution and stars are appreciated.

Thank you — and especially thanks to the PyTorch team for the reference code, tools and ecosystem that made this work possible.