#include <assert.h>
#include <stdio.h>
#include <math.h>

#include <mpi.h>
#include <omp.h>

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	int mpi_rank, mpi_rank_size, omp_threads;
	MPI_Status stat;

	MPI_Init(&argc, &argv);
	
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_rank_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	omp_threads = omp_get_max_threads();

	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long pixels = 0;
	unsigned long long y;

	// equally split part of computation to every rank
	unsigned long long final_rank, range, start, end, chunk;
	if(r >= mpi_rank_size){ // normal case, r >= p
		range = r / mpi_rank_size;
		final_rank = mpi_rank_size - 1;
	}
	else{ // rare case r < p
		range = 1;
		final_rank = r - 1;
	}
    start = mpi_rank * range;
    end = (mpi_rank != final_rank) ? (mpi_rank + 1) * range : r;
	chunk = (range >= omp_threads) ? range / omp_threads : 1; // normal case: range >= threads, rare case: range < threads

	if(mpi_rank == 0){
        #pragma omp parallel num_threads(omp_threads) shared(r, k) private(y) reduction(+:pixels)
		{
			#pragma omp for schedule(static, chunk)
			for (unsigned long long x = start; x < end; x++) {
				y = ceil(sqrtl(r*r - x*x));
				pixels += y;
				pixels %= k;
			}
		}
        // Receive every part computed by other rank, then sum up
        for(unsigned long long i = 1; i <= final_rank; i++){
			unsigned long long part_pixels;
            MPI_Recv(&part_pixels, 1, MPI_UNSIGNED_LONG_LONG, i, 1, MPI_COMM_WORLD, &stat);
            pixels += part_pixels;
            pixels %= k;
        }
        printf("%llu\n", (4 * pixels) % k);
    }
    else if(mpi_rank != 0 && mpi_rank <= final_rank){ // make sure that now rank will not exceed final rank
        #pragma omp parallel num_threads(omp_threads) shared(r, k) private(y) reduction(+:pixels)
		{
			#pragma omp for schedule(static, chunk)
			for (unsigned long long x = start; x < end; x++) {
				y = ceil(sqrtl(r*r - x*x));
				pixels += y;
				pixels %= k;
			}
		}
        MPI_Send(&pixels, 1, MPI_UNSIGNED_LONG_LONG, 0, 1, MPI_COMM_WORLD);
    }

	MPI_Finalize();
	return 0;
}
