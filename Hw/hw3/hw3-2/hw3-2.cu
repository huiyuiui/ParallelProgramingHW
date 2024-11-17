#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <algorithm>

using namespace std;


//======================
#define DEV_NO 0
#define BLOCK_SIZE 32
cudaDeviceProp prop;

const int INF = ((1 << 30) - 1);
int n, m;
unsigned int* host_dist_s;
unsigned int* host_dist_t;
unsigned int* device_dist;

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);
    host_dist_s = (unsigned int*) malloc(n * n * sizeof(unsigned int));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                host_dist_s[i * n + j] = 0;
            } else {
                host_dist_s[i * n + j] = INF;
            }
        }
    }

    int pair[3];
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, file);
        host_dist_s[pair[0] * n + pair[1]] = pair[2];
    }
    fclose(file);
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (host_dist_t[i * n + j] >= INF) host_dist_t[i * n + j] = INF;
        }
        fwrite(&host_dist_t[i * n], sizeof(unsigned int), n, outfile);
    }
    fclose(outfile);
}

int ceil(int a, int b) { return (a + b - 1) / b; }

__global__ void calKernelPhase1(unsigned int* Dist, int n, int B, int Round, int block_start_y, int block_start_x){

    // block index
    int block_x = block_start_x + blockIdx.x;
    int block_y = block_start_y + blockIdx.y;

    // global index
    int i = block_y * B + threadIdx.y;
    int j = block_x * B + threadIdx.x;

    if(i >= n || j >= n) return;

    // pivot index
    int pivot_start = Round * B;
    int pivot_end = min((Round + 1) * B, n);

    // move to share memory
    __shared__ unsigned int shared_dist[BLOCK_SIZE][BLOCK_SIZE];
    shared_dist[threadIdx.y][threadIdx.x] = Dist[i * n + j];

    __syncthreads();

    // For each block, it need to compute B times
    for (int k = 0; k < pivot_end - pivot_start; ++k) { // each phase will perform B iterations
        if(shared_dist[threadIdx.y][k] + shared_dist[k][threadIdx.x] < shared_dist[threadIdx.y][threadIdx.x])
            shared_dist[threadIdx.y][threadIdx.x] = shared_dist[threadIdx.y][k] + shared_dist[k][threadIdx.x];
        __syncthreads();
    }

    // move back to GPU memory
    Dist[i * n + j] = shared_dist[threadIdx.y][threadIdx.x];
}


__global__ void calKernelPhase2(unsigned int* Dist, int n, int row_col, int B, int Round, int block_start_y, int block_start_x){
    
    // block index
    int block_x = block_start_x + blockIdx.x;
    int block_y = block_start_y + blockIdx.y;

    // global index
    int i = block_y * B + threadIdx.y;
    int j = block_x * B + threadIdx.x;

    // pivot index
    int pivot_start = Round * B;
    int pivot_end = min((Round + 1) * B, n);

    __shared__ unsigned int shared_pivot[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int shared_dist[BLOCK_SIZE][BLOCK_SIZE];

    // init shared memory
    {
        if (i < n && j < n) 
            shared_dist[threadIdx.y][threadIdx.x] = Dist[i * n + j];
        else 
            shared_dist[threadIdx.y][threadIdx.x] = INF;

        if(row_col == 0){ // pivot row: now block and pivot block are in the same row
            if (i < n && (pivot_start + threadIdx.x) < n) 
                shared_pivot[threadIdx.y][threadIdx.x] = Dist[i * n + (pivot_start + threadIdx.x)];
            else 
                shared_pivot[threadIdx.y][threadIdx.x] = INF;
        } 
        else if(row_col == 1){ // pivot col: now block and pivot block are in the same col
            if ((pivot_start + threadIdx.y) < n && j < n) 
                shared_pivot[threadIdx.y][threadIdx.x] = Dist[(pivot_start + threadIdx.y) * n + j];
            else 
                shared_pivot[threadIdx.y][threadIdx.x] = INF;
        }
    }

    __syncthreads();

    // Computation
    if(i < n && j < n){
        if(row_col == 0){ // pivot row
            // For each block, it need to compute B times
            for (int k = 0; k < pivot_end - pivot_start; ++k) {
                if(shared_pivot[threadIdx.y][k] + shared_dist[k][threadIdx.x] < shared_dist[threadIdx.y][threadIdx.x])
                    shared_dist[threadIdx.y][threadIdx.x] = shared_pivot[threadIdx.y][k] + shared_dist[k][threadIdx.x];
                __syncthreads();
            }
       }
       else if(row_col == 1){ // pivot col
            // For each block, it need to compute B times
            for (int k = 0; k < pivot_end - pivot_start; ++k) { 
                if(shared_dist[threadIdx.y][k] + shared_pivot[k][threadIdx.x] < shared_dist[threadIdx.y][threadIdx.x])
                    shared_dist[threadIdx.y][threadIdx.x] = shared_dist[threadIdx.y][k] + shared_pivot[k][threadIdx.x];
                __syncthreads();
            }
       }

       // write back to GPU memory
       Dist[i * n + j] = shared_dist[threadIdx.y][threadIdx.x];
    }
}


__global__ void calKernelPhase3(unsigned int* Dist, int n, int B, int Round, int block_start_y, int block_start_x){

    // block index
    int block_x = block_start_x + blockIdx.x;
    int block_y = block_start_y + blockIdx.y;

    // global index
    int i = block_y * B + threadIdx.y;
    int j = block_x * B + threadIdx.x;

    // pivot index
    int pivot_start = Round * B;
    int pivot_end = min((Round + 1) * B, n);

    __shared__ unsigned int shared_dist[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int shared_row[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int shared_col[BLOCK_SIZE][BLOCK_SIZE];

    // init shared memory
    {
        if (i < n && j < n) 
            shared_dist[threadIdx.y][threadIdx.x] = Dist[i * n + j];
        else 
            shared_dist[threadIdx.y][threadIdx.x] = INF;
        
        if (i < n && (pivot_start + threadIdx.x) < n) 
            shared_row[threadIdx.y][threadIdx.x] = Dist[i * n + (pivot_start + threadIdx.x)];
        else 
            shared_row[threadIdx.y][threadIdx.x] = INF;
        
        if ((pivot_start + threadIdx.y) < n && j < n) 
            shared_col[threadIdx.y][threadIdx.x] = Dist[(pivot_start + threadIdx.y) * n + j];
        else 
            shared_col[threadIdx.y][threadIdx.x] = INF;
    }

    __syncthreads();

    // Computation
    if(i < n && j < n){
        // For each block, it need to compute B times
        for (int k = 0; k < pivot_end - pivot_start; ++k) {
            if (shared_row[threadIdx.y][k] + shared_col[k][threadIdx.x] < shared_dist[threadIdx.y][threadIdx.x]) 
                shared_dist[threadIdx.y][threadIdx.x] = shared_row[threadIdx.y][k] + shared_col[k][threadIdx.x];
            __syncthreads();
        }
        
        // write back to GPU memory
        Dist[i * n + j] = shared_dist[threadIdx.y][threadIdx.x];
    }
}

void calPhase1(int n, int B, int Round, int block_start_y, int block_start_x){
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(1, 1);
    calKernelPhase1<<<gridSize, blockSize>>>(device_dist, n, B, Round, block_start_y, block_start_x);
}

void calPhase2(int n, int row_col, int B, int Round, int block_start_y, int block_start_x, int height_num_blocks, int width_num_blocks){
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(width_num_blocks, height_num_blocks);
    calKernelPhase2<<<gridSize, blockSize>>>(device_dist, n, row_col, B, Round, block_start_y, block_start_x);
}

void calPhase3(int n, int B, int Round, int block_start_y, int block_start_x, int height_num_blocks, int width_num_blocks){
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(width_num_blocks, height_num_blocks);
    calKernelPhase3<<<gridSize, blockSize>>>(device_dist, n, B, Round, block_start_y, block_start_x);
}

void block_FW(int B) {
    int round = ceil(n, B);
    for (int r = 0; r < round; ++r) {
        /* Phase 1*/
        // pivot block
        calPhase1(n, B, r, r, r);

        /* Phase 2*/
        calPhase2(n, 0, B, r, r, 0, 1, r); // pivot row: from 0 to now index
        calPhase2(n, 0, B, r, r, r + 1, 1, round - r - 1); // pivot row: from now index + 1 to end
        calPhase2(n, 1, B, r, 0, r, r, 1);  // pivot col: from 0 to now index
        calPhase2(n, 1, B, r, r + 1, r, round - r - 1, 1); // pivot col: from now index + 1 to end

        /* Phase 3*/
        calPhase3(n, B, r, 0, 0, r, r); // other: left upper
        calPhase3(n, B, r, 0, r + 1, r, round - r - 1); // other: right upper
        calPhase3(n, B, r, r + 1, 0, round - r - 1, r); // other: left lower
        calPhase3(n, B, r, r + 1, r + 1, round - r - 1, round - r - 1); // ohter: right lower
    }
}


int main(int argc, char* argv[]) {
    input(argv[1]);
    host_dist_t = (unsigned int*) malloc(n * n * sizeof(unsigned int));

    cudaGetDeviceProperties(&prop, DEV_NO);
    printf("maxThreasPerBlock = %d, sharedMemPerBlock = %d\n", prop.maxThreadsPerBlock, prop.sharedMemPerBlock);
    
    // cuda malloc memory
    cudaMalloc((void**)&device_dist, n * n * sizeof(unsigned int));

    // copy host memory to cuda
    cudaMemcpy(device_dist, host_dist_s, n * n * sizeof(unsigned int), cudaMemcpyHostToDevice);

    // Blocked Floyd-Warshall
    block_FW(BLOCK_SIZE);

    // copy cuda memory back to host
    cudaMemcpy(host_dist_t, device_dist, n * n * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // output
    output(argv[2]);

    // free memory
    free(host_dist_s);
    free(host_dist_t);
    cudaFree(device_dist);

    return 0;
}