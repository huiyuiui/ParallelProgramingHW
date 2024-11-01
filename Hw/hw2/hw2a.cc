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
#include <immintrin.h>

int ncpus;
int iters, width, height;
double left, right, lower, upper;
int* image;

void* mandelbrot_set(void* thread_Id){
    int thread_id = *(int*)thread_Id;

    int vec_size = 8; // avx512: 8 double per register
    double y_offset = ((upper - lower) / height);
    double x_offset = ((right - left) / width);

    /* calculation */
    for(int j = thread_id; j < height; j += ncpus){
        double y0 = j * y_offset + lower;
        __m512d y0_vec = _mm512_set1_pd(y0);
        int i;

        for(i = 0; i + vec_size <= width; i += vec_size){
            double x0_vals[vec_size] = {
                (i + 0) * x_offset + left,
                (i + 1) * x_offset + left,
                (i + 2) * x_offset + left,
                (i + 3) * x_offset + left,
                (i + 4) * x_offset + left,
                (i + 5) * x_offset + left,
                (i + 6) * x_offset + left,
                (i + 7) * x_offset + left
            };

            // initialize avx512 to compute
            __m512d x0_vec = _mm512_loadu_pd(x0_vals);   
            __m512i repeats_vec = _mm512_setzero_si512();
            __m512d x_vec = _mm512_setzero_pd();
            __m512d y_vec = _mm512_setzero_pd();
            __m512d length_squared = _mm512_setzero_pd();
            __m512d two_vec = _mm512_set1_pd(2.0);
            __m512d four_vec = _mm512_set1_pd(4.0);
            __m512d x_square = _mm512_setzero_pd();
            __m512d y_square = _mm512_setzero_pd();

            // manderbrot set calculate
            for(int k = 0; k < iters; k++) {
                // termination check
                __mmask8 mask = _mm512_cmplt_pd_mask(length_squared, four_vec); // decide which element will be calculate
                if(mask == 0) break; // if length_squared all greater or equal than four, mask will all be zero
                
                /* vectorize computation */
                __m512d x_temp = _mm512_add_pd(_mm512_sub_pd(x_square, y_square), x0_vec);

                // y = 2 * x * y + y0;
                y_vec = _mm512_add_pd(_mm512_mul_pd(_mm512_mul_pd(two_vec, x_vec), y_vec), y0_vec);

                x_vec = x_temp;

                x_square = _mm512_mul_pd(x_vec, x_vec);
                y_square = _mm512_mul_pd(y_vec, y_vec);

                // length_squared = x * x + y * y
                length_squared = _mm512_add_pd(x_square, y_square);
                // only update those elements less than 4
                repeats_vec = _mm512_mask_add_epi64(repeats_vec, mask, repeats_vec, _mm512_set1_epi64(1));
            }

            int64_t repeats[8];
            _mm512_storeu_si512((__m512i*)repeats, repeats_vec);
            for(int vk = 0; vk < vec_size; vk++)
                image[j * width + i + vk] = repeats[vk];
        }

        // remaining elements
        for(; i < width; i++){
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
    png_set_compression_level(png_ptr, 0);
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
    ncpus = CPU_COUNT(&cpu_set);
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
    int threads_id[ncpus];

    /* allocate memory for image */
    image = new int[height * width];
    assert(image);
    
    /* create threads */
    for(int t = 0; t < ncpus; t++){
        threads_id[t] = t;
        int rc = pthread_create(&threads[t], NULL, mandelbrot_set, (void *)&threads_id[t]);
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
    delete[] image;
}
