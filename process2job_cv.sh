#!/bin/bash
#SBATCH -n 4 # Request 4 CPU' s cores . Maximum 10 CPU ’ s cores.
#SBATCH -N 1 # Ensure that all cores are on one machine.
#SBATCH -D /fhome/vlia01/MED-GIA # Working directory. Change to your user homer folder.
#SBATCH -t 4-00:05 # Runtime in D - HH : MM
#SBATCH -p tfg # Partition to submit to.
#SBATCH --mem 12288 # Request 12 GB of RAM memory. Maximum 60 GB.
#SBATCH -o %x_%u_%j.out # File to which STDOUT will be written.
#SBATCH -e %x_%u_%j.err # File to which STDERR will be written.
#SBATCH --gres gpu:1 # Request 1 GPU. Maximum 8 GPUs.

sleep 3

# Remove all .out and .err files
# rm /fhome/vlia01/MED-GIA/*.out
# rm /fhome/vlia01/MED-GIA/*.err

# Run the train script
python /fhome/vlia01/MED-GIA/AnomalyDetection/cross_val.py
