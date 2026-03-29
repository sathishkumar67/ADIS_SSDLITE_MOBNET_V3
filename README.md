# Animal Intrusion Detection System — SSDLite + MobileNetV3

This repository contains an implementation of an Animal Intrusion Detection System (ADIS) built around the SSDLite object detector with a MobileNetV3 backbone. The project provides training and inference utilities, example notebooks, model checkpoints, and helper modules to run detection experiments on a balanced animal dataset.

Key goals:
- Lightweight detector for on-device or resource-constrained environments using SSDLite + MobileNetV3
- Reproducible training pipelines and hyperparameter tuning (Optuna / BOHB)
- **Energy-efficient AI**: Built-in GPU power and energy monitoring for training and evaluation.
- **Advanced Performance Analytics**: Detailed evaluation including FPS, latency, and false positive rates.

## Highlights / Features

- **SSDLite MobileNetV3**: Efficient object detection model (see `ssdlite_mobnetv3_adis/model.py`).
- **GPU Energy Monitoring**: Real-time tracking of Average GPU Power (W) and Total Energy Consumption (J) using `pynvml` (see `GPUMonitor` in `utils.py`).
- **Advanced Evaluation**: Comprehensive metrics including FPS, Latency per frame (ms), and False Positive Rates (FPR) per class (see `energy_evaluator.py`).
- **Cached Dataset**: Fast I/O using LMDB (see `ssdlite_mobnetv3_adis/dataset.py`).
- **Training Utilities**: Schedulers and BOHB-compatible tuning wrappers (see `trainer.py`, notebooks).
- **Inference Helpers**: Visualization and plotting tools (see `inference.py`, `plot.py`).

## Repository structure

```
LICENSE
README.md
requirements.txt
SSDLite_MobileNetV3_Bohbtune_Training.ipynb
SSDLite_MobileNetV3_Inference.ipynb
ckpt/
    ssdlite_mobnetv3_bestparams_ckpt.pth
    ...
ssdlite_mobnetv3_adis/
    __init__.py
    bohbtune.py
    dataset.py
    energy_evaluator.py  <-- NEW: Advanced metrics & energy tracking
    evaluate.py
    inference.py
    model.py
    plot.py
    trainer.py           <-- UPDATED: Integrated energy reporting
    utils.py             <-- UPDATED: Added GPUMonitor class
study/
```

## Prerequisites

- **Python 3.8+**
- **NVIDIA GPU**: Required for GPU power monitoring via `pynvml`.
- **Dependencies**: Install from `requirements.txt`.

```bash
# Install dependencies
pip install -r requirements.txt

# Additional dependency for GPU monitoring
pip install pynvml
```

## Energy & Performance Evaluation

The new `energy_evaluator.py` provides a powerful way to measure the efficiency of your model.

### 1. Evaluating on a Dataset
You can compute accuracy (mAP, Precision, Recall, F1, FPR) alongside energy and performance metrics:

```python
from ssdlite_mobnetv3_adis.energy_evaluator import evaluate_with_energy
from ssdlite_mobnetv3_adis.dataset import CachedSSDLITEOBJDET_DATASET

# ... Load model and dataloader ...
metrics_df, energy = evaluate_with_energy(
    model, 
    test_loader, 
    device, 
    class_names=CLASSES,
    conf_thresh=0.2
)
```

### 2. Evaluating on Video
Measure inference speed and power consumption on a video file:

```python
from ssdlite_mobnetv3_adis.energy_evaluator import evaluate_video_energy

energy = evaluate_video_energy(
    model, 
    video_path="path/to/video.mp4", 
    device=device, 
    class_names=CLASSES
)
```

## Training with Energy Tracking

The standard `train()` and `bohb_tunner()` functions in `trainer.py` now include energy monitoring. At the end of a training session, it will output:
- **Average GPU Power (W)**
- **Total GPU Energy Consumption (J)**
- **Total Training Duration (s)**

## Dataset

This project uses the ADIS dataset (Animal Detection and Identification System). The notebooks automatically download it from Hugging Face (`pt-sk/ADIS`).

Classes:
`['Cat', 'Cattle', 'Chicken', 'Deer', 'Dog', 'Squirrel', 'Eagle', 'Goat', 'Rodents', 'Snake']`

## Quick usage

1) **Inference**: Open `SSDLite_MobileNetV3_Inference.ipynb` to visualize detections.
2) **Training/Tuning**: Open `SSDLite_MobileNetV3_Bohbtune_Training.ipynb` to reproduce tuning results.

## License

This project is released under the **MIT License**.

## Acknowledgements

Special thanks to the PyTorch team, and the creators of Albumentations, Optuna, and the `pynvml` library for providing the tools that make this research possible.
s work possible.