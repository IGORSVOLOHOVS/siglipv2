# 🏎️ SigLIPv2 Image Classifier

A professional, zero-shot image classification pipeline powered by **SigLIP 2** (`google/siglip2-base-patch16-224`). This project demonstrates high-performance visual-text alignment for open-vocabulary classification tasks.

## 🌟 Features

- **Zero-Shot Classification**: Classify images using natural language prompts without retraining.
- **SigLIPv2 Core**: Utilizes the latest SigLIPv2 architecture for superior alignment scores.
- **Hardware Optimized**: Automatic detection and utilization of CUDA for high-speed inference.
- **Rich Terminal UI**: Beautifully formatted terminal output for enhanced observability.

## 🎞️ Demo

### Input Image
![Input Image](data/image.png)

### Execution Logs
```powershell
INFO: Loading model google/siglip2-base-patch16-224 on cuda...
INFO: HTTP Request: HEAD https://huggingface.co/google/siglip2-base-patch16-224/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
INFO: HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/google/siglip2-base-patch16-224/75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2/config.json "HTTP/1.1 200 OK"
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 408/408 [00:00<00:00, 9494.32it/s]
INFO: Classifying data/image.png
Predicted index: 0
Predicted label: a car
```

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- PyTorch with CUDA support (optional but recommended)
- `transformers`, `pillow`, `rich`

### Installation
```bash
pip install torch torchvision transformers pillow rich
```

### Usage
Run the classifier on the default image:
```bash
python classify_siglip.py
```

## 🏗️ Architecture

The project follows a modular protocol-based design:
- `SigLIPClassifier`: The core engine handling model loading and image processing.
- `Classifier Protocol`: Ensures future-proofing for alternative classification backends.
