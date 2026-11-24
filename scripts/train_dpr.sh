#!/bin/bash
#BSUB -J dpr_epoch1
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "select[gpu32gb]"
#BSUB -W 02:30
#BSUB -o logs/dpr_%J.out
#BSUB -e logs/dpr_%J.err

unset PYTHONPATH
source $BLACKHOLE/dpr/venv_cu126/bin/activate || exit 1
export HF_HOME=$BLACKHOLE/dpr/hf_cache
export HF_HUB_OFFLINE=1
cd $BLACKHOLE/dpr/Deep-Learning || exit 1
nvidia-smi
python -u DPR_training/bert.py
