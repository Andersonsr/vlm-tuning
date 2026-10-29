#!/bin/bash
# CC3M download (dataset/download_cc3m.py) and shard indexing (dataset/cc3m.py), CPU only. Submit from the vlm-tuning root:
#   OUT=/path/to/cc3m sbatch slurm/download_cc3m.sbatch
#   OUT=/path/to/cc3m SPLITS=validation sbatch slurm/download_cc3m.sbatch
#   OUT=/path/to/cc3m sbatch slurm/download_cc3m.sbatch --source urls --processes 16 --threads 64
# SPLITS (default "train validation") goes to both steps; extra arguments go to download_cc3m.py.
# 5 minutes before the time limit slurm sends SIGUSR1 and the job requeues itself; both sources resume where they
# stopped (huggingface skips finished shards, img2dataset finished tars), so the restart picks up the download.
# After a failure or scancel, submitting again with the same OUT resumes the same way.
#SBATCH --job-name=cc3m-download
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16           # parallel downloads, img2dataset processes, indexing workers
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=tornado
#SBATCH --output=logs/cc3m-%x-%j.out
#SBATCH --open-mode=append           # restarts of a requeued job write to the same log
#SBATCH --requeue
#SBATCH --signal=B:USR1@300          # B: to this batch script, which requeues the job
##SBATCH --partition=tc

export HTTP_PROXY=http://$USER:$PASSWORD@inet-sys.petrobras.com.br:804
export HTTPS_PROXY=http://$USER:$PASSWORD@inet-sys.petrobras.com.br:804
export NO_PROXY="127.0.0.1, localhost, petrobras.com.br, petrobras.biz"

set -eo pipefail
. ~/.bashrc
conda activate cap
export PYTHONUNBUFFERED=1
set -u

OUT=${OUT:?set OUT to the destination folder}
SPLITS=${SPLITS:-train validation}
CPUS=${SLURM_CPUS_PER_TASK:-16}

mkdir -p logs "$OUT"
echo "$(date) job $SLURM_JOB_ID (restart ${SLURM_RESTART_COUNT:-0}) out=$OUT splits=$SPLITS cpus=$CPUS"

requeue() {
    echo "$(date) time limit close: requeueing job $SLURM_JOB_ID"
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
}
trap requeue USR1

# in the background so the trap runs as soon as the signal arrives (bash waits for a foreground command first)
python dataset/download_cc3m.py --out "$OUT" --splits $SPLITS --workers "$CPUS" "$@" &
wait $!

# the index trainLight.py would otherwise build on every rank at the start of a multi-gpu run
python -m dataset.cc3m --root "$OUT" --splits $SPLITS --workers "$CPUS" &
wait $!

echo "$(date) done"
