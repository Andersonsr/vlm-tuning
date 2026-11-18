#!/bin/bash
# Image retrieval of the off-the-shelf vision encoders (retrieval/vision_encoders.py) on the geo data of trainLight.py,
# single GPU. Submit from the vlm-tuning root:
#   sbatch slurm/vision_encoders.sbatch
#   sbatch slurm/vision_encoders.sbatch --models dinov3 radio --split val
# CONFIG (default model/configs/geo_default.yaml) gives the dataset settings, with the cluster paths filled in as in
# train.sbatch; PE_REPO (default $CLUSTER_HOME/perception_models) is the clone of facebookresearch/perception_models
# for --models pe. Extra arguments go to vision_encoders.py (default: every encoder, test split, random baseline,
# bf16). The results are appended to vision_encoders_<split>.json in the vlm-tuning root.
#SBATCH --job-name=vision-encoders
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10           # 8 dataloader workers
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --account=tornado
#SBATCH --output=logs/%x-%j.out
##SBATCH --partition=tc

set -eo pipefail
. ~/.bashrc
conda activate cap
# proxy host and NO_PROXY are set in the cap env (conda env config vars); USER and PASSWORD come from the submit shell
# (the weights are downloaded from the HuggingFace hub on the first run)
export HTTP_PROXY=http://$USER:$PASSWORD@${PROXY_HOST:?set PROXY_HOST in the cap env}
export HTTPS_PROXY=$HTTP_PROXY
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
set -u

CONFIG=${CONFIG:-model/configs/geo_default.yaml}
# the configs have placeholders for the cluster paths: the job uses a copy holding the real ones, from the cap env
# (conda env config vars)
CLUSTER_HOME=${CLUSTER_HOME:?set CLUSTER_HOME in the cap env}
PROJECT_HOME=${PROJECT_HOME:?set PROJECT_HOME in the cap env}
GEO_IMAGES=${GEO_IMAGES:?set GEO_IMAGES in the cap env}
PE_REPO=${PE_REPO:-$CLUSTER_HOME/perception_models}

mkdir -p logs
JOB_CONFIG=logs/config-$SLURM_JOB_ID.yaml
sed -e "s#/path/to/cluster_home#$CLUSTER_HOME#g" -e "s#/path/to/project_home#$PROJECT_HOME#g" \
    -e "s#/path/to/geo_images#$GEO_IMAGES#g" "$CONFIG" > "$JOB_CONFIG"
echo "$(date) job $SLURM_JOB_ID config=$CONFIG pe_repo=$PE_REPO"

python retrieval/vision_encoders.py --config "$JOB_CONFIG" --pe_repo "$PE_REPO" --random --bf16 "$@"
