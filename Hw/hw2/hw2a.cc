#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

int iters, width, height;
double left, right, lower, upper;
int* image;

typedef struct data{
    int start, end;
    int thread_num;
} data;

void* mandelbrot_set(void* arg){
    data* thread_data = (data*)arg;

    /* calculation */
    for(int j = thread_data->start; j < thread_data->end; j++){
        double y0 = j * ((upper - lower) / height) + lower;
        for(int i = 0; i < width; i++){
            double x0 = i * ((right - left) / width) + left;
            int repeats = 0;
            double x = 0;
            double y = 0;
            double length_squared = 0;
            while (repeats < iters && length_squared < 4) {
                double temp = x * x - y * y + x0;
                y = 2 * x * y + y0;
                x = temp;
                length_squared = x * x + y * y;
                ++repeats;
            }
            image[j * width + i] = repeats;
        }
    }

    pthread_exit(NULL);
}


void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

int main(int argc, char** argv) {
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    int ncpus = CPU_COUNT(&cpu_set);
    printf("%d cpus available\n", ncpus);

    /* argument parsing */
    assert(argc == 9);
    const char* filename = argv[1];
    iters = strtol(argv[2], 0, 10);
    left = strtod(argv[3], 0); // real lower boundary
    right = strtod(argv[4], 0); // real upper boundary
    lower = strtod(argv[5], 0); // imaginary lower bound
    upper = strtod(argv[6], 0); // imaginary upper bound
    width = strtol(argv[7], 0, 10);  // image width
    height = strtol(argv[8], 0, 10); // image height

    /* initialization */
    pthread_t threads[ncpus];
    data thread_data[ncpus];
    int chunk_size = height / ncpus; // row partition
    int remainder = height % ncpus;
    int now_start = 0, now_end = 0;
    
    /* allocate memory for image */
    image = new int[height * width];
    assert(image);
    
    /* create threads */
    for(int t = 0; t < ncpus; t++){
        now_start = now_end; // end is non-inclusive
        if(remainder > 0){
            now_end = now_start + chunk_size + 1;
            remainder--;
        }
        else now_end = now_start + chunk_size;

        thread_data[t].start = now_start;
        thread_data[t].end = now_end;
        thread_data[t].thread_num = t;

        int rc = pthread_create(&threads[t], NULL, mandelbrot_set, (void*)&thread_data[t]);
        if(rc){
			printf("ERROR; return code from pthread_create() is %d\n", rc);
            exit(-1);
		}
    }

    /* joint threads*/
    for(int t = 0; t < ncpus; t++)
        pthread_join(threads[t], NULL);

    /* draw and cleanup */
    write_png(filename, iters, width, height, image);
    free(image);
}
