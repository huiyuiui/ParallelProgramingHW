#include "mpi.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}

    int i, rank, size, namelen;
    char name[MPI_MAX_PROCESSOR_NAME];
    MPI_Status stat;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &size); // total number of processor
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // now processor number
    MPI_Get_processor_name(name, &namelen);

    unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long pixels = 0;
    unsigned long long y;

    // equally split part of computation to every rank
    unsigned long long range = r / size;
    unsigned long long start = rank * range;
    unsigned long long end = (rank != size - 1) ? (rank + 1) * range : r;
    // in case r isn't divisible by size, let the end of final rank to compute rest of part

    if(rank == 0){
        for(unsigned long long x = start; x < end; x++){
            y = ceil(sqrtl(r*r - x*x));
            pixels += y;
		    pixels %= k;
        }
        // Receive every part computed by other rank, then sum up
        for(unsigned long long i = 1; i < size; i++){
            MPI_Recv(&y, 1, MPI_UNSIGNED_LONG_LONG, i, 1, MPI_COMM_WORLD, &stat);
            pixels += y;
            pixels %= k;
        }
        printf("%llu\n", (4 * pixels) % k);
    }
    else{
        for(unsigned long long x = start; x < end; x++){
            y = ceil(sqrtl(r*r - x*x));
            pixels += y;
		    pixels %= k;
        }
        MPI_Send(&pixels, 1, MPI_UNSIGNED_LONG_LONG, 0, 1, MPI_COMM_WORLD);
    }
	
    MPI_Finalize();
    return (0);
}
