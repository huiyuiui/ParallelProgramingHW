# Pixels in circle with different parallel program interface

### Introduction

The problem definition is the same as Lab1, but we will use Pthread, OpenMP, and Hybrid MPI with OpenMP methods to solve it this time.

### Linking Pthread & Openmp

- Linking pthread
    - gcc your_program.c -o your_program -pthread
- Linking openmp
    - gcc your_program.c -o your_program -fopenmp
- Linking both
    - gcc your_program.c -o your_program -pthread -fopenmp

### Approximate pixels using pthread and OpenMP

- Modify the sequential code lab2_pthread.cc with pthread
    - g++ lab2_pthread.cc -o lab2_pthread -pthread -lm 
    - srun -c4 -n1 ./lab2_pthread r k
- Modify the sequential code lab2_openmp.cc with openmp
    - g++ lab2_openmp.cc -o lab2_openmp -fopenmp -lm 
    - srun -c4 -n1 ./lab2_openmp r k

### Approximate pixels using Hybrid MPI with OpenMP

- Modify the sequential code lab2_hybrid.cc with MPI and OpenMP
    - mpicxx lab2_hybrid.cc -o lab2_hybrid -fopenmp -lm 
    - srun -n6 -c4 ./lab2_hybrid ? ?




