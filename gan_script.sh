#!/bin/bash

# Cuda paths
CUDA_HOME=/scratch_net/googolplex/rheiki/cuda-8.0/
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# Pyenv path
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"


## Pass environment variables of workstation to GPU node 
#$ -V

## stderr and stdout are merged together to stdout
#$ -j y

# logging directory. preferrably on your scratch
#$ -o /srv/glusterfs/rheiki/logs

export PATH="/scratch_net/googolplex/rheiki/cuda-8.0/bin:$PATH"
export LD_LIBRARY_PATH="/scratch_net/googolplex/rheiki$ cd cuda-8.0/lib64:$LD_LIBRARY_PATH"

export SGE_GPU_ALL="$(ls -rt /tmp/lock-gpu*/info.txt | xargs grep -h  $(whoami) | awk '{print $2}' | paste -sd "," -)"
export SGE_GPU=$(echo $SGE_GPU_ALL |rev|cut -d, -f1|rev) # USE LAST GPU by request time

export CUDA_VISIBLE_DEVICES=$SGE_GPU

now="$(date)"
echo "--------------------------------------------------"
echo "starting batch with GPUs: $SGE_GPU of $SGE_GPU_ALL on machine `hostname`"
echo "starting time: $now"
echo "--------------------------------------------------"

# here starts the main execution of our job
cd /srv/glusterfs/rheiki/splitting_gan/

# start training script
python gan_cifar_resnet_kmeans.py


then="$(date)"
echo "--------------------------------------------------"
echo "finishing time: $then"
echo "--------------------------------------------------"
exit 0
