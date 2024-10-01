#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}	

	int omp_threads = omp_get_max_threads();

	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long pixels = 0;
	unsigned long long y;
	unsigned long long chunk = r / omp_threads;

	#pragma omp parallel num_threads(omp_threads) shared(r, k) private(y) reduction(+:pixels)
	{
		#pragma omp for schedule(static, chunk)
		for (unsigned long long x = 0; x < r; x++) {
			y = ceil(sqrtl(r*r - x*x));
			pixels += y;
			pixels %= k;
		}
	}
	
	printf("%llu\n", (4 * pixels) % k);

	return 0;
}
