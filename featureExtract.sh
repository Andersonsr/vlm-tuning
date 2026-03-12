#!/bin/bash
#SBATCH --nodes=1 #Número de Nós
#SBATCH --ntasks-per-node=4 #Número de tarefas por Nó
#SBATCH -p gpu #Fila (partition) a ser utilizada
#SBATCH -J vlm-test #Nome job
#SBATCH --chdir=/u/fibz/projects/vlm-tuning/ #path a partir do qual será executado o job
#SBATCH --account=tornado #Conta do projeto
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --exclusive
#SBATCH --gres=gpu:4                #Define o número de GPUs visíveis por nó (opcional)

export HTTP_PROXY=http://fibz:Petr*123456@inet-sys.petrobras.com.br:804
export HTTPS_PROXY=http://fibz:Petr*123456@inet-sys.petrobras.com.br:804
export NO_PROXY="127.0.0.1, localhost, petrobras.com.br, petrobras.biz"

. ~/.bashrc
conda activate cap

#export NCCL_P2P_DISABLE=1
#export NCCL_SHM_DISABLE=1

#export NCCL_SOCKET_IFNAME=^lo,docker,br-,virbr,veth,
#export GLOO_SOCKET_IFNAME=^lo,docker,br-,virbr,veth,eno1,enp5s0

# export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_SOCKET_IFNAME=enp4s0
# export NCCL_IB_DISABLE=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=NET

export WANDB_MODE=offline

#srun hostname
#srun ip addr

#srun ls /home/users/adsdrosa/datasets/nwpu/

srun bash -c "
  IFACE=\$(ip -o -4 addr show up scope global \
    | grep -E '192\\.168\\.30\\.' \
    | head -n1 \
    | cut -d' ' -f2)
  export NCCL_SOCKET_IFNAME=\$IFACE
  export GLOO_SOCKET_IFNAME=\$IFACE
  echo \"Using NCCL interface: \$IFACE\"
  python featuresExtraction.py \
          --dataset nwpu \
          
"
