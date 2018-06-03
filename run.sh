#!/bin/bash

#SBATCH --time=00:10:00
#SBATCH --mem=20G
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --gres=gpu:4
#SBATCH --job-name=cme213
#SBATCH --output=cme213-%j.out
#SBATCH --error=cme213-%j.err
#SBATCH --nodelist=gpu-200-2

#echo "In file run.sh, update the line below before running the script"
#echo "WORKDIR='<directory with your code>'"
#exit 0

# Comment the 3 lines above after setting WORKDIR

WORKDIR='/home/mbauer94/final'
export WORKDIR

### ---------------------------------------
### BEGINNING OF EXECUTION
### ---------------------------------------

echo The master node of this job is `hostname`
echo This job runs on the following nodes:
echo `scontrol show hostname $SLURM_JOB_NODELIST`
echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
echo "Current working directory is `echo $WORKDIR`"
echo
echo Output from code
echo ----------------

# mpirun -np 4 ./main -s -n 1000 -b 800 -l 0.01 -e 20
#mpirun -np 4 ./main -n 20 -e 3 -sd
#mpirun -np 4 ./main -n 23 -r 0.0077 -l 0.0007 -e 1 -b 799 -sd
#MV2_USE_CUDA=1 mpirun -np 4 nvprof --output-profile profile.%p.nvprof ./main -e 1
#MV2_USE_CUDA=1 mpirun -np 1 nvprof --kernels "myGEMM_kernel" --analysis-metrics --output-profile GEMMmetrics.out.%p.nvprof ./main -g 4
#./a.out
#cuda-memcheck --leak-check full ./main -e 1
mpirun -np 4 ./main
