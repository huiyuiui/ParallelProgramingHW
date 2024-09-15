#! /bin/bash
#SBATCH -N 2
#SBATCH -n 12
#SBATCH -c 2
#SBATCH -J mpi_test

module load mpi

srun $HOME/test