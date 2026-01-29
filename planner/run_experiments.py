#!/usr/bin/env python3

import os
import sys
import subprocess
import time
import json
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

EXPERIMENTS = [
    {
        "name": "baseline_512_12",
        "note": "baseline_512_12",
        "config_overrides": {
            "planner_dim": 512,
            "planner_layers": 12,
            "planner_heads": 8,
            "lr": 1e-4,
        }
    },
    {
        "name": "medium_768_12",
        "note": "medium_768_12",
        "config_overrides": {
            "planner_dim": 768,
            "planner_layers": 12,
            "planner_heads": 12,
            "lr": 1.5e-4,
        }
    },
    {
        "name": "deep_512_16",
        "note": "deep_512_16",
        "config_overrides": {
            "planner_dim": 512,
            "planner_layers": 16,
            "planner_heads": 8,
            "lr": 1e-4,
        }
    },
    {
        "name": "large_1024_12",
        "note": "large_1024_12",
        "config_overrides": {
            "planner_dim": 1024,
            "planner_layers": 12,
            "planner_heads": 16,
            "lr": 2e-4,
        }
    },
    {
        "name": "high_lr_512_12",
        "note": "high_lr_512_12",
        "config_overrides": {
            "planner_dim": 512,
            "planner_layers": 12,
            "planner_heads": 8,
            "lr": 2e-4,
        }
    },
]

def run_experiment(exp_config, epochs=100, gpus=1):
    print(f"\n{'='*80}")
    print(f"Starting experiment: {exp_config['name']}")
    print(f"Config: {exp_config['config_overrides']}")
    print(f"{'='*80}\n")
    
    cmd = [
        "python", "train_planner.py",
        "--note", exp_config['note'],
        "--epochs", str(epochs),
        "--vis_interval", "25",
        "--vis_frames", "50",
    ]
    
    overrides = exp_config['config_overrides']
    if 'planner_dim' in overrides:
        cmd.extend(["--planner_dim", str(overrides['planner_dim'])])
    if 'planner_layers' in overrides:
        cmd.extend(["--planner_layers", str(overrides['planner_layers'])])
    if 'planner_heads' in overrides:
        cmd.extend(["--planner_heads", str(overrides['planner_heads'])])
    if 'lr' in overrides:
        cmd.extend(["--lr", str(overrides['lr'])])
    
    if gpus > 1:
        base_cmd = [
            "torchrun",
            "--nproc_per_node", str(gpus),
            "--master_port", "12356",
            "train_planner.py",
            "--note", exp_config['note'],
            "--epochs", str(epochs),
            "--vis_interval", "25",
            "--vis_frames", "50",
        ]
        overrides = exp_config['config_overrides']
        if 'planner_dim' in overrides:
            base_cmd.extend(["--planner_dim", str(overrides['planner_dim'])])
        if 'planner_layers' in overrides:
            base_cmd.extend(["--planner_layers", str(overrides['planner_layers'])])
        if 'planner_heads' in overrides:
            base_cmd.extend(["--planner_heads", str(overrides['planner_heads'])])
        if 'lr' in overrides:
            base_cmd.extend(["--lr", str(overrides['lr'])])
        cmd = base_cmd
    
    print(f"Command: {' '.join(cmd)}")
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
        elapsed = time.time() - start_time
        print(f"\nExperiment completed: {exp_config['name']} ({elapsed/60:.1f} minutes)")
        return True, elapsed
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\nExperiment failed: {exp_config['name']} ({elapsed/60:.1f} minutes)")
        print(f"Error: {e}")
        return False, elapsed

def collect_results():
    results = []
    exp_dir = os.path.join(PROJECT_ROOT, "experiments")
    
    for exp in EXPERIMENTS:
        pattern = f"*_{exp['note']}"
        matching_dirs = sorted(Path(exp_dir).glob(pattern), key=os.path.getmtime, reverse=True)
        
        if matching_dirs:
            exp_path = matching_dirs[0]
            loss_file = exp_path / "loss_history.tsv"
            config_file = exp_path / "config.json"
            
            if loss_file.exists():
                with open(loss_file, 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        first_line = lines[1].strip().split('\t')
                        last_line = lines[-1].strip().split('\t')
                        
                        results.append({
                            "name": exp['name'],
                            "exp_dir": str(exp_path),
                            "epochs": len(lines) - 1,
                            "initial_train": float(first_line[1]),
                            "initial_val": float(first_line[3]),
                            "final_train": float(last_line[1]),
                            "final_val": float(last_line[3]),
                            "improvement": float(first_line[1]) - float(last_line[1]),
                        })
    
    return results

def print_summary(results):
    print(f"\n{'='*80}")
    print("Experiment Results Summary")
    print(f"{'='*80}\n")
    
    print(f"{'Experiment':<20} {'Epochs':<8} {'Initial Train':<15} {'Final Train':<15} {'Final Val':<15} {'Improvement':<12}")
    print("-" * 80)
    
    for r in sorted(results, key=lambda x: x['final_val']):
        improvement_pct = (r['improvement'] / r['initial_train']) * 100
        print(f"{r['name']:<20} {r['epochs']:<8} {r['initial_train']:<15.2f} {r['final_train']:<15.2f} {r['final_val']:<15.2f} {improvement_pct:<12.1f}%")
    
    print(f"\n{'='*80}")
    print("Best Performance (Final Val Loss):")
    best = min(results, key=lambda x: x['final_val'])
    print(f"  Experiment: {best['name']}")
    print(f"  Final Val Loss: {best['final_val']:.2f}")
    print(f"  Experiment Directory: {best['exp_dir']}")
    print(f"{'='*80}\n")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Quick test multiple model configurations")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs per experiment")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--skip", type=str, nargs="+", help="Experiment names to skip")
    parser.add_argument("--only", type=str, nargs="+", help="Only run these experiments")
    parser.add_argument("--summary-only", action="store_true", help="Only print summary (do not run experiments)")
    args = parser.parse_args()
    
    if args.summary_only:
        results = collect_results()
        print_summary(results)
        return
    
    experiments_to_run = EXPERIMENTS
    if args.only:
        experiments_to_run = [e for e in EXPERIMENTS if e['name'] in args.only]
    elif args.skip:
        experiments_to_run = [e for e in EXPERIMENTS if e['name'] not in args.skip]
    
    print(f"Running {len(experiments_to_run)} experiments")
    print(f"{args.epochs} epochs per experiment")
    print(f"GPUs: {args.gpus}\n")
    
    total_start = time.time()
    success_count = 0
    
    for i, exp in enumerate(experiments_to_run, 1):
        print(f"\n[{i}/{len(experiments_to_run)}] Running experiment...")
        success, elapsed = run_experiment(exp, epochs=args.epochs, gpus=args.gpus)
        if success:
            success_count += 1
        
        if i < len(experiments_to_run):
            print("\nPreparing next experiment... (5 second wait)")
            time.sleep(5)
    
    total_elapsed = time.time() - total_start
    
    print(f"\n{'='*80}")
    print(f"All experiments completed!")
    print(f"Success: {success_count}/{len(experiments_to_run)}")
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    print(f"{'='*80}\n")
    
    results = collect_results()
    if results:
        print_summary(results)
        
        summary_file = os.path.join(PROJECT_ROOT, "experiments", "experiment_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Summary saved: {summary_file}")

if __name__ == "__main__":
    main()
