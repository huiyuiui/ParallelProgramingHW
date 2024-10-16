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
#include <mpi.h>
#include <omp.h>

int iters, width, height;
double left, right, lower, upper;

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
    int mpi_rank, mpi_rank_size, omp_threads;
    MPI_Status stat;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &mpi_rank_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    omp_threads = omp_get_max_threads();

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
    int chunk_size = height / mpi_rank_size;
    int remainder = height % mpi_rank_size;
    int start, end;
    int thread_chunk = width / omp_threads;
    if(mpi_rank < remainder){
        start = mpi_rank * chunk_size + mpi_rank;
        end = start + chunk_size + 1;
    }
    else{
        start = mpi_rank * chunk_size + remainder;
        end = start + chunk_size;
    }

    /* allocate memory for image */
    // image = new int[height * width];
    int* image = new int[height * width]();
    assert(image);

    /* mandelbrot set */
    for (int j = start; j < end; j++) {
        double y0 = j * ((upper - lower) / height) + lower;
        #pragma omp parallel num_threads(omp_threads) shared(image, y0)
        {
            #pragma omp for schedule(static, thread_chunk)
            for (int i = 0; i < width; ++i) {
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
    }

    if(mpi_rank == 0){
        int* final_image = new int[height * width]();
        assert(final_image);
        MPI_Reduce(image, final_image, height * width, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        /* draw and cleanup */
        write_png(filename, iters, width, height, final_image);
        free(image);
        free(final_image);
    }
    else{
        MPI_Reduce(image, NULL, height * width, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        free(image);
    }

    MPI_Finalize();
    return 0;
}
