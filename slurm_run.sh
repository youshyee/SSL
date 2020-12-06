#!/bin/bash -login
#SBATCH --job-name=simsiam

#SBATCH --ntasks=8
#SBATCH --time=7-00:00:00

#SBATCH --mem=120G
#SBATCH --mail-type=ALL
#SBATCH --partition gpu
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4

PARTITION=gpu
JOB_NAME=simsiam

TRAINFILE=/mnt/storage/scratch/rn18510/codespace/cyclepre/train.py
CONFIG=/mnt/storage/scratch/rn18510/codespace/cyclepre/configs/simsiam.py
GPUS=8

GPUS_PER_NODE=2
CPUS_PER_TASK=4
SRUN_ARGS=${SRUN_ARGS:-""}

while true;
do
srun -p ${PARTITION}\
	--job-name=${JOB_NAME}\
	--gres=gpu:${GPUS_PER_NODE}\
	--ntasks=${GPUS}\
	--ntasks-per-node=${GPUS_PER_NODE}\
	--cpus-per-task=${CPUS_PER_TASK}\
	--kill-on-bad-exit=1\
	${SRUN_ARGS}\
	/mnt/storage/scratch/rn18510/anaconda3/envs/pytorch/bin/python -u $TRAINFILE ${CONFIG}  --launcher="slurm" && break \
	|| echo 0 > "$SLURM_SUBMIT_DIR"/control_$SLURM_JOBID; /mnt/storage/scratch/rn18510/.local/bin/composemail $SLURM_JOBID "$SLURM_SUBMIT_DIR"/slurm-"$SLURM_JOBID".out; sendmail rn18510@bristol.ac.uk < ~/.tmp/mail_"$SLURM_JOBID".txt; while [ $(cat "$SLURM_SUBMIT_DIR"/control_$SLURM_JOBID) == 0 ];
do
	sleep 1
done
done
[ -f  "$SLURM_SUBMIT_DIR"/control_$SLURM_JOBID ] && rm "$SLURM_SUBMIT_DIR"/control_$SLURM_JOBID

