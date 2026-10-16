#!/bin/bash
# Requeueable training with trainLight.py. Submit from the vlm-tuning root:
#   NAME=dual-qformer sbatch slurm/train.sbatch --multiresolution --set model.vision.weights=/path/to/ssl.pth
# CONFIG (default model/configs/dual_default.yaml) picks the config; extra arguments go to trainLight.py.
# 3 minutes before the time limit slurm sends SIGUSR1: lightning saves <output_dir>/<NAME>/hpc_ckpt_<n>.ckpt and
# requeues the job, which resumes from it (--requeue) and logs to the same wandb run. A job that dies without the
# signal (node failure, scancel) is resumed the same way from last.ckpt by submitting it again with the same NAME.
#SBATCH --job-name=vlm-tuning
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4          # one task per GPU, lightning reads the ranks from slurm (must equal --gpus)
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10           # 8 dataloader workers per GPU
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --account=tornado
#SBATCH --output=logs/vlm-%x-%j.out
#SBATCH --open-mode=append           # restarts of a requeued job write to the same log
#SBATCH --requeue
#SBATCH --signal=SIGUSR1@180
##SBATCH --partition=tc

export HTTP_PROXY=http://$USER:$PASSWORD@inet-sys.petrobras.com.br:804
export HTTPS_PROXY=http://$USER:$PASSWORD@inet-sys.petrobras.com.br:804
export NO_PROXY="127.0.0.1, localhost, petrobras.com.br, petrobras.biz"

set -eo pipefail
. ~/.bashrc
conda activate cap
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
set -u

NAME=${NAME:?set NAME to the run name (folder under output_dir, same NAME to resume)}
CONFIG=${CONFIG:-model/configs/dual_default.yaml}
GPUS=${SLURM_NTASKS_PER_NODE:-4}

mkdir -p logs
echo "$(date) job $SLURM_JOB_ID (restart ${SLURM_RESTART_COUNT:-0}) name=$NAME config=$CONFIG nodes=$SLURM_NNODES gpus/node=$GPUS"

srun python trainLight.py --config "$CONFIG" --name "$NAME" --nnodes "$SLURM_NNODES" --gpus "$GPUS" --requeue "$@"
