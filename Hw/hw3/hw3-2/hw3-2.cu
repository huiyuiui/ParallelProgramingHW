#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <algorithm>

using namespace std;


//======================
#define DEV_NO 0
#define BLOCK_SIZE 32
#define SHARED_BLOCK 64
cudaDeviceProp prop;

const int INF = ((1 << 30) - 1);
int n, m, n_padded;
unsigned int* host_dist_s;
unsigned int* host_dist_t;
unsigned int* device_dist;

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    // pad array size to multiples of 64
    n_padded = ((n + 63) / 64) * 64;
    host_dist_s = (unsigned int*) malloc(n_padded * n_padded * sizeof(unsigned int));

    for (int i = 0; i < n_padded; ++i) {
        for (int j = 0; j < n_padded; ++j) {
            if (i < n && j < n && i == j) {
                host_dist_s[i * n_padded + j] = 0;
            } else {
                host_dist_s[i * n_padded + j] = INF;
            }
        }
    }

    int* edges = (int*)malloc(m * 3 * sizeof(int));
    fread(edges, sizeof(int), m * 3, file);
    for (int i = 0; i < m; ++i) {
        host_dist_s[edges[i * 3 + 0] * n_padded + edges[i * 3 + 1]] = edges[i * 3 + 2];
    }
    fclose(file);
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    for (int i = 0; i < n; ++i) {
        fwrite(&host_dist_t[i * n_padded], sizeof(unsigned int), n, outfile);
    }
    fclose(outfile);
}

int ceil(int a, int b) { return (a + b - 1) / b; }

__global__ void calKernelPhase1(unsigned int* Dist, int n, int B, int Round, int block_start_y, int block_start_x){

    // block index
    int block_x = block_start_x + blockIdx.x;
    int block_y = block_start_y + blockIdx.y;

    // thread index
    int x = threadIdx.x;
    int y = threadIdx.y;

    // global index
    int i = block_y * SHARED_BLOCK + y;
    int j = block_x * SHARED_BLOCK + x;

    // move to share memory
    __shared__ unsigned int shared_dist[SHARED_BLOCK][SHARED_BLOCK];
    shared_dist[y][x] = Dist[i * n + j];
    shared_dist[y][x + B] = Dist[i * n + (j + B)];
    shared_dist[y + B][x] = Dist[(i + B) * n + j];
    shared_dist[y + B][x + B] = Dist[(i + B) * n + (j + B)];

    __syncthreads();

    // For each block, it need to compute B times
    for (int k = 0; k < SHARED_BLOCK; ++k) { // each phase will perform B iterations
        shared_dist[y][x] = min(shared_dist[y][x], shared_dist[y][k] + shared_dist[k][x]);
        shared_dist[y][x + B] = min(shared_dist[y][x + B], shared_dist[y][k] + shared_dist[k][x + B]);
        shared_dist[y + B][x] = min(shared_dist[y + B][x], shared_dist[y + B][k] + shared_dist[k][x]);
        shared_dist[y + B][x + B] = min(shared_dist[y + B][x + B] ,shared_dist[y + B][k] + shared_dist[k][x + B]);
        __syncthreads();
    }

    // move back to GPU memory
    Dist[i * n + j] = shared_dist[y][x];
    Dist[i * n + (j + B)] = shared_dist[y][x + B];
    Dist[(i + B) * n + j] = shared_dist[y + B][x];
    Dist[(i + B) * n + (j + B)] = shared_dist[y + B][x + B];
}


__global__ void calKernelPhase2(unsigned int* Dist, int n, int row_col, int B, int Round, int block_start_y, int block_start_x){
    
    // block index
    int block_x = block_start_x + blockIdx.x;
    int block_y = block_start_y + blockIdx.y;

    // thread index
    int x = threadIdx.x;
    int y = threadIdx.y;

    // global index
    int i = block_y * SHARED_BLOCK + y;
    int j = block_x * SHARED_BLOCK + x;

    // pivot index
    int pivot_start = Round * SHARED_BLOCK;

    __shared__ unsigned int shared_pivot[SHARED_BLOCK][SHARED_BLOCK];
    __shared__ unsigned int shared_dist[SHARED_BLOCK][SHARED_BLOCK];

    // use registers to cache the points to be calculated
    unsigned int vertex_0 = Dist[i * n + j];
    unsigned int vertex_1 = Dist[i * n + (j + B)];
    unsigned int vertex_2 = Dist[(i + B) * n + j];
    unsigned int vertex_3 = Dist[(i + B) * n + (j + B)];

    // init shared memory
    shared_dist[y][x] = vertex_0;
    shared_dist[y][x + B] = vertex_1;
    shared_dist[y + B][x] = vertex_2;
    shared_dist[y + B][x + B] = vertex_3;
    if(row_col == 0){
        shared_pivot[y][x] = Dist[i * n + (pivot_start + x)];
        shared_pivot[y][x + B] = Dist[i * n + (pivot_start + x + B)];
        shared_pivot[y + B][x] = Dist[(i + B) * n + (pivot_start + x)];
        shared_pivot[y + B][x + B] = Dist[(i + B) * n + (pivot_start + x + B)];
    }
        
    else if(row_col == 1){
        shared_pivot[y][x] = Dist[(pivot_start + y) * n + j];
        shared_pivot[y][x + B] = Dist[(pivot_start + y) * n + (j + B)];
        shared_pivot[y + B][x] = Dist[(pivot_start + y + B) * n + (j)];
        shared_pivot[y + B][x + B] = Dist[(pivot_start + y + B) * n + (j + B)];
    }

    __syncthreads();

    // Computation
    if(row_col == 0){ // pivot row
        // For each block, it need to compute B times
        #pragma unroll 64
        for (int k = 0; k < SHARED_BLOCK; ++k) {
            vertex_0 = min(vertex_0, shared_pivot[y][k] + shared_dist[k][x]);
            vertex_1 = min(vertex_1, shared_pivot[y][k] + shared_dist[k][x + B]);
            vertex_2 = min(vertex_2, shared_pivot[y + B][k] + shared_dist[k][x]);
            vertex_3 = min(vertex_3 ,shared_pivot[y + B][k] + shared_dist[k][x + B]);
        }
    }
    else if(row_col == 1){ // pivot col
        // For each block, it need to compute B times
        #pragma unroll 64
        for (int k = 0; k < SHARED_BLOCK; ++k) { 
            vertex_0 = min(vertex_0, shared_dist[y][k] + shared_pivot[k][x]);
            vertex_1 = min(vertex_1, shared_dist[y][k] + shared_pivot[k][x + B]);
            vertex_2 = min(vertex_2, shared_dist[y + B][k] + shared_pivot[k][x]);
            vertex_3 = min(vertex_3 ,shared_dist[y + B][k] + shared_pivot[k][x + B]);
        }
    }

    // write back to GPU memory
    Dist[i * n + j] = vertex_0;
    Dist[i * n + (j + B)] = vertex_1;
    Dist[(i + B) * n + j] = vertex_2;
    Dist[(i + B) * n + (j + B)] = vertex_3;
}


__global__ void calKernelPhase3(unsigned int* Dist, int n, int B, int Round, int block_start_y, int block_start_x){

    // block index
    int block_x = block_start_x + blockIdx.x;
    int block_y = block_start_y + blockIdx.y;

    // thread index
    int x = threadIdx.x;
    int y = threadIdx.y;

    // global index
    int i = block_y * SHARED_BLOCK + y;
    int j = block_x * SHARED_BLOCK + x;

    // pivot index
    int pivot_start = Round * SHARED_BLOCK;

    __shared__ unsigned int shared_row[SHARED_BLOCK][SHARED_BLOCK];
    __shared__ unsigned int shared_col[SHARED_BLOCK][SHARED_BLOCK];

    // use registers to cache the points to be calculated
    unsigned int vertex_0 = Dist[i * n + j];
    unsigned int vertex_1 = Dist[i * n + (j + B)];
    unsigned int vertex_2 = Dist[(i + B) * n + j];
    unsigned int vertex_3 = Dist[(i + B) * n + (j + B)];

    // init shared memory
    shared_row[y][x] = Dist[i * n + (pivot_start + x)];
    shared_row[y][x + B] = Dist[i * n + (pivot_start + x + B)];
    shared_row[y + B][x] = Dist[(i + B) * n + (pivot_start + x)];
    shared_row[y + B][x + B] = Dist[(i + B) * n + (pivot_start + x + B)];

    shared_col[y][x] = Dist[(pivot_start + y) * n + j];
    shared_col[y][x + B] = Dist[(pivot_start + y) * n + (j + B)];
    shared_col[y + B][x] = Dist[(pivot_start + y + B) * n + (j)];
    shared_col[y + B][x + B] = Dist[(pivot_start + y + B) * n + (j + B)];

    __syncthreads();

    // Computation
    // For each block, it need to compute B times
    #pragma unroll 64
    for (int k = 0; k < SHARED_BLOCK; ++k) {
        vertex_0 = min(vertex_0, shared_row[y][k] + shared_col[k][x]);
        vertex_1 = min(vertex_1, shared_row[y][k] + shared_col[k][x + B]);
        vertex_2 = min(vertex_2, shared_row[y + B][k] + shared_col[k][x]);
        vertex_3 = min(vertex_3 ,shared_row[y + B][k] + shared_col[k][x + B]);
    }
    
    // write back to GPU memory
    Dist[i * n + j] = vertex_0;
    Dist[i * n + (j + B)] = vertex_1;
    Dist[(i + B) * n + j] = vertex_2;
    Dist[(i + B) * n + (j + B)] = vertex_3;
}

void calPhase1(int n, int B, int Round, int block_start_y, int block_start_x){
    dim3 blockSize(B, B);
    dim3 gridSize(1, 1);
    calKernelPhase1<<<gridSize, blockSize>>>(device_dist, n, B, Round, block_start_y, block_start_x);
}

void calPhase2(int n, int row_col, int B, int Round, int block_start_y, int block_start_x, int height_num_blocks, int width_num_blocks){
    dim3 blockSize(B, B);
    dim3 gridSize(width_num_blocks, height_num_blocks);
    calKernelPhase2<<<gridSize, blockSize>>>(device_dist, n, row_col, B, Round, block_start_y, block_start_x);
}

void calPhase3(int n, int B, int Round, int block_start_y, int block_start_x, int height_num_blocks, int width_num_blocks){
    dim3 blockSize(B, B);
    dim3 gridSize(width_num_blocks, height_num_blocks);
    calKernelPhase3<<<gridSize, blockSize>>>(device_dist, n, B, Round, block_start_y, block_start_x);
}

void block_FW(int B) {
    int round = ceil(n_padded, SHARED_BLOCK);
    for (int r = 0; r < round; ++r) {
        /* Phase 1*/
        // pivot block
        calPhase1(n_padded, B, r, r, r);

        /* Phase 2*/
        calPhase2(n_padded, 0, B, r, r, 0, 1, r); // pivot row: from 0 to now index
        calPhase2(n_padded, 0, B, r, r, r + 1, 1, round - r - 1); // pivot row: from now index + 1 to end
        calPhase2(n_padded, 1, B, r, 0, r, r, 1);  // pivot col: from 0 to now index
        calPhase2(n_padded, 1, B, r, r + 1, r, round - r - 1, 1); // pivot col: from now index + 1 to end

        /* Phase 3*/
        calPhase3(n_padded, B, r, 0, 0, r, r); // other: left upper
        calPhase3(n_padded, B, r, 0, r + 1, r, round - r - 1); // other: right upper
        calPhase3(n_padded, B, r, r + 1, 0, round - r - 1, r); // other: left lower
        calPhase3(n_padded, B, r, r + 1, r + 1, round - r - 1, round - r - 1); // ohter: right lower
    }
}


int main(int argc, char* argv[]) {
    input(argv[1]);
    host_dist_t = (unsigned int*) malloc(n_padded * n_padded * sizeof(unsigned int));

    cudaGetDeviceProperties(&prop, DEV_NO);
    printf("maxThreasPerBlock = %d, sharedMemPerBlock = %d\n", prop.maxThreadsPerBlock, prop.sharedMemPerBlock);
    
    // cuda malloc memory
    cudaMalloc((void**)&device_dist, n_padded * n_padded * sizeof(unsigned int));

    // copy host memory to cuda
    cudaMemcpy(device_dist, host_dist_s, n_padded * n_padded * sizeof(unsigned int), cudaMemcpyHostToDevice);

    // Blocked Floyd-Warshall
    block_FW(BLOCK_SIZE);

    // copy cuda memory back to host
    cudaMemcpy(host_dist_t, device_dist, n_padded * n_padded * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // output
    output(argv[2]);

    // free memory
    free(host_dist_s);
    free(host_dist_t);
    cudaFree(device_dist);

    return 0;
}