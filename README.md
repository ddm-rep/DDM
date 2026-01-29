# Video Generation Pipeline

A deep learning pipeline for video sequence prediction and high-resolution restoration.

## Architecture

This project consists of two main components:

1. **Planner**: Transformer-based autoregressive video sequence prediction model
2. **Refiner**: Diffusion-based super-resolution model

## Requirements

```bash
pip install -r requirements.txt
```

## Data Preparation

Data should be provided in preprocessed `.pt` file format:
- Low resolution: `[T, 3, 32, 32]` (uint8)
- High resolution: `[T, 3, 128, 128]` (uint8)

Data structure:
```
data/
├── processed_32/    # Low-resolution data
│   ├── train/
│   └── val/
└── processed/       # High-resolution data
    ├── train/
    └── val/
```

## Usage

### Training Planner

```bash
cd planner
python train_planner.py --data_dir /path/to/data/processed_32 --epochs 100
```

### Training Refiner

```bash
python scripts/train.py --data_dir /path/to/data/processed --high_res 128 --latent_res 32
```

## Model Architecture

### Planner

- Transformer Decoder based
- Autoregressive sequence prediction
- Low-resolution video frame generation

### Refiner

- Asymmetric U-Net based
- Progressive resolution growth
- Low-resolution → High-resolution restoration

## Configuration

Main configuration can be modified in `planner/modules/config_planner.py`.

## License

MIT License
