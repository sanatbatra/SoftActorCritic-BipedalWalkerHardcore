#!/bin/bash
#SBATCH --mem=20000
#SBATCH --time=40:00:00
#SBATCH --job-name="bwh"

module purge
module load anaconda3/5.3.1
module load cuda/10.0.130
module load gcc/6.3.0



# Replace with your NetID
NETID=sab1086

cd /scratch/${NETID}/BWH-SAC
source activate env
pip install torch==1.4.0


python3 -u main.py