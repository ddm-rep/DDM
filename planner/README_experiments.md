# Experiment Automation Script

A script to quickly test multiple model configurations and find optimal settings.

## Usage

### Basic execution (all experiments, 100 epochs)
```bash
cd planner
python run_experiments.py
```

### Custom number of epochs
```bash
python run_experiments.py --epochs 50
```

### Run specific experiments only
```bash
python run_experiments.py --only baseline_512_12 medium_768_12
```

### Skip specific experiments
```bash
python run_experiments.py --skip large_1024_12
```

### Multi-GPU execution
```bash
python run_experiments.py --gpus 4 --epochs 100
```

### View summary only (without running experiments)
```bash
python run_experiments.py --summary-only
```

## Experiment Configurations

1. **baseline_512_12**: Baseline configuration
   - dim=512, layers=12, heads=8, lr=1e-4

2. **medium_768_12**: Medium-sized model
   - dim=768, layers=12, heads=12, lr=1.5e-4

3. **deep_512_16**: Deeper model
   - dim=512, layers=16, heads=8, lr=1e-4

4. **large_1024_12**: Large model
   - dim=1024, layers=12, heads=16, lr=2e-4

5. **high_lr_512_12**: High learning rate
   - dim=512, layers=12, heads=8, lr=2e-4

## Results

After experiments complete, a summary is automatically printed and saved to `experiments/experiment_summary.json`.

Detailed results for each experiment can be found in `experiments/YYYYMMDD_HHMMSS_<note>/` directories.
