#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>

unsigned long long r;
unsigned long long k;

typedef struct data{
	unsigned long long start;
	unsigned long long end;
	unsigned long long result;
} data;

void* calculate(void* arg){
	data* thread_data = (data*)arg;
	unsigned long long y;
	for(unsigned long long x = thread_data->start; x < thread_data->end; x++){
		y = ceil(sqrtl(r*r - x*x));
		thread_data->result += y;
		thread_data->result %= k;
	}

	pthread_exit(NULL);
}

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	r = atoll(argv[1]);
	k = atoll(argv[2]);
	unsigned long long pixels = 0;

	cpu_set_t cpuset;
	sched_getaffinity(0, sizeof(cpuset), &cpuset);
	unsigned long long ncpus = CPU_COUNT(&cpuset);
	
	pthread_t threads[ncpus];
	data thread_data[ncpus];
	int rc;
	unsigned long long range = r / ncpus;

	// threads calculation
	for(int t = 0; t < ncpus; t++){
		thread_data[t].start = t * range;
		thread_data[t].end = (t != ncpus - 1) ? (t + 1) * range : r;
		thread_data[t].result = 0;
		rc = pthread_create(&threads[t], NULL, calculate, (void*)&thread_data[t]);
		if(rc){
			printf("ERROR; return code from pthread_create() is %d\n", rc);
            exit(-1);
		}
	}

	// join result
	for(int t = 0; t < ncpus; t++){
		pthread_join(threads[t], NULL);
		pixels += thread_data[t].result;
		pixels %= k;
	}
	printf("%llu\n", (4 * pixels) % k);

	pthread_exit(NULL);
}
