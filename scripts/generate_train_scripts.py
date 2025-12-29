#!/usr/bin/env python3
"""
Generate SLURM training scripts for each model variant.
Creates train_{model}.sh files from train_tiny.sh template.
"""

from pathlib import Path

# List of models
models = [
    "vitt16-distillationv1_vitb16",
    "vitt16-distillationv1_vits16plus",
    "vitt16-notpretrained_vits16",
    "vitt16_vitb16",
    "vitt16_vits16plus",
    "vitt16-distillationv1_vitb16_200e",
    "vitt16-distillationv1_vits16plus_200e",
    "vitt16-notpretrained_vits16_200e",
    "vitt16_vitb16_200e",
    "vitt16_vits16plus_200e",
    "vitt16-distillationv1_vits16",
    "vitt16-notpretrained_vitb16",
    "vitt16-notpretrained_vits16plus",
    "vitt16_vits16",
    "vitt16-distillationv1_vits16_200e",
    "vitt16-notpretrained_vitb16_200e",
    "vitt16-notpretrained_vits16plus_200e",
    "vitt16_vits16_200e"
]

# Template script
template = """#!/bin/bash
#SBATCH -p mit_preemptable  
#SBATCH --gres=gpu:1 
#SBATCH -t 2-00:00:00
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=40GB
#SBATCH --requeue

module load miniforge/24.3.0-0
mamba activate dacon

cd ..

MODEL='{model}'

python dacon/train.py --config configs/train-$MODEL.yaml --tiny
python dacon/test.py --config configs/test-$MODEL-consecutive.yaml --model checkpoints/$MODEL/dacon_v1.1.pth --tiny
python dacon/test.py --config configs/test-$MODEL-keyframe-ref1.yaml --model checkpoints/$MODEL/dacon_v1.1.pth --tiny
python dacon/test.py --config configs/test-$MODEL-keyframe-ref5.yaml --model checkpoints/$MODEL/dacon_v1.1.pth --tiny
"""

def generate_train_scripts():
    """Generate SLURM training scripts for each model."""
    
    for model in models:
        # Create script content with the model name
        script_content = template.format(model=model)
        
        # Write to new file
        output_file = f"train_{model}.sh"
        with open(output_file, 'w') as f:
            f.write(script_content)
        
        # Make the script executable
        Path(output_file).chmod(0o755)
        
        print(f"Created: {output_file}")
    
    print(f"\nSuccessfully generated {len(models)} SLURM training scripts!")

if __name__ == "__main__":
    generate_train_scripts()
