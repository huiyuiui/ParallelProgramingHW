#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>


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

__global__ void calKernel(unsigned int* Dist, int n, int B, int Round, int block_start_x, int block_start_y) {

    // block index
    int block_x = block_start_x + blockIdx.x;
    int block_y = block_start_y + blockIdx.y;

    // global index
    int i = block_y * B + threadIdx.y;
    int j = block_x * B + threadIdx.x;

    // init pivot index
    int pivot_start = Round * B;
    int pivot_end = min((Round + 1) * B, n);
    int pivot_ki = pivot_start + threadIdx.y; 
    int pivot_kj = pivot_start + threadIdx.x;

    // move to share memory
    __shared__ unsigned int shared_dist[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int shared_row[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int shared_col[BLOCK_SIZE][BLOCK_SIZE];

    // boundary
    {
        if(i < n && j < n)
            shared_dist[threadIdx.y][threadIdx.x] = Dist[i * n + j];
        else
            shared_dist[threadIdx.y][threadIdx.x] = INF;

        if(pivot_kj < pivot_end && i < n)
            shared_row[threadIdx.y][threadIdx.x] = Dist[i * n + pivot_kj];
        else
            shared_row[threadIdx.y][threadIdx.x] = INF;
        
        if(pivot_ki < pivot_end && j < n)
            shared_col[threadIdx.y][threadIdx.x] = Dist[pivot_ki * n + j];
        else
            shared_col[threadIdx.y][threadIdx.x] = INF;
    }

    __syncthreads(); // to ensure that all elements in shared memory are initialized

    // For each block, it need to compute B times
    for (int k = 0; k < pivot_end - pivot_start; ++k) { // each phase will perform B iterations
        if(shared_row[threadIdx.y][k] + shared_col[k][threadIdx.x] < shared_dist[threadIdx.y][threadIdx.x])
            shared_dist[threadIdx.y][threadIdx.x] = shared_row[threadIdx.y][k] + shared_col[k][threadIdx.x];
        __syncthreads(); // wait every threads finish this round
    }
    
    if(i < n && j < n)
        Dist[i * n + j] = shared_dist[threadIdx.y][threadIdx.x];
}

void cal(int n, int B, int Round, int block_start_x, int block_start_y, int width_num_blocks, int height_num_blocks){
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(width_num_blocks, height_num_blocks);
    calKernel<<<gridSize, blockSize>>>(device_dist, n, B, Round, block_start_x, block_start_y);
}

void block_FW(int B) {
    int round = ceil(n, B);
    for (int r = 0; r < round; ++r) {
        /* Phase 1*/
        // pivot block
        dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridSize(1, 1); // only one block need to be calculated
        calKernel<<<gridSize, blockSize>>>(device_dist, n, BLOCK_SIZE, r, r, r);
        
        /* Phase 2*/
        cal(n, B, r, r, 0, r, 1); // pivot row: from 0 to now index
        cal(n, B, r, r, r + 1, round - r - 1, 1); // pivot row: from now index + 1 to end
        cal(n, B, r, 0, r, 1, r);  // pivot col: from 0 to now index
        cal(n, B, r, r + 1, r, 1, round - r - 1); // pivot col: from now index + 1 to end

        /* Phase 3*/
        cal(n, B, r, 0, 0, r, r); // other: left upper
        cal(n, B, r, 0, r + 1, round - r - 1, r); // other: left lower
        cal(n, B, r, r + 1, 0, r, round - r - 1); // other: right upper
        cal(n, B, r, r + 1, r + 1, round - r - 1, round - r - 1); // ohter: right lower
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