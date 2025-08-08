#!/bin/bash
#PBS -P rp06
#PBS -q dgxa100
#PBS -l ngpus=1            
#PBS -l ncpus=16            
#PBS -l mem=16GB           
#PBS -l walltime=00:02:00  
#PBS -l wd                  
#PBS -l storage=scratch/rp06

module load cuda/12.6.2
# module load python3/3.10.4

nvidia-smi >> gpu-info.txt

cd ..

source /scratch/rp06/sl5952/IHA-and-BMCN/.venv/bin/activate
python3 test_implementation.py >> out_test_implementation.txt

