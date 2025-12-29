#!/bin/bash
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

MODEL='vitt16-distillationv1_vitb16'

python dacon/train.py --config configs/train-$MODEL.yaml --tiny
python dacon/test.py --config configs/test-$MODEL-consecutive.yaml --model checkpoints/$MODEL/dacon_v1.1.pth --tiny
python dacon/test.py --config configs/test-$MODEL-keyframe-ref1.yaml --model checkpoints/$MODEL/dacon_v1.1.pth --tiny
python dacon/test.py --config configs/test-$MODEL-keyframe-ref5.yaml --model checkpoints/$MODEL/dacon_v1.1.pth --tiny
