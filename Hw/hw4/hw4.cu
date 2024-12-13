#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>
#include <cuda.h>

using namespace std;

//======================
#define DEV_NO 0
#define BLOCK_SIZE 32
cudaDeviceProp prop;

int B, N, d;
float *Q, *K, *V, *O;
float *dev_Q, *dev_K, *dev_V, *dev_O;

void input(char *input_filename);

void output(char *output_filename);

double getTimeStamp() {
    struct timeval tv;
    gettimeofday( &tv, NULL );
    return (double) tv.tv_usec/1000000 + tv.tv_sec;
}

__global__ void flash_attention_kernel(float* q, float* k, float* v, float* o, float* l, float* m, int d);

int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("Usage: %s <input_filename> <output_filename>\n", argv[0]);
        return 1;
    }

    input(argv[1]);

    cudaGetDeviceProperties(&prop, DEV_NO);
    printf("maxThreasPerBlock = %d, sharedMemPerBlock = %d\n", prop.maxThreadsPerBlock, prop.sharedMemPerBlock);
    
    // cuda malloc memory
    cudaMalloc((void**)&dev_Q, B * N * d * sizeof(float));
    cudaMalloc((void**)&dev_K, B * N * d * sizeof(float));
    cudaMalloc((void**)&dev_V, B * N * d * sizeof(float));
    cudaMalloc((void**)&dev_O, B * N * d * sizeof(float));

    // copy host memory to cuda
    cudaMemcpy(dev_Q, Q, B * N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_K, K, B * N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_V, V, B * N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_O, O, B * N * d * sizeof(float), cudaMemcpyHostToDevice);

    // kernel parameters
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blockNum(N / BLOCK_SIZE, N / BLOCK_SIZE);

    double start, end;
    start = getTimeStamp();

    // flash attention
    for (int i = 0; i < B; i++){
        // init l and m
        float *l = (float *)malloc(N * sizeof(float));
        float *m = (float *)malloc(N * sizeof(float));
        memset(l, 0x00, N * sizeof(float));
        for (int j = 0; j < N; j++) {
            m[j] = -FLT_MAX;
        }

        // copy l and m to device
        float *dev_l, *dev_m;
        cudaMalloc((void**)&dev_l, N * sizeof(float));
        cudaMalloc((void**)&dev_m, N * sizeof(float));
        cudaMemcpy(dev_l, l, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_m, m, N * sizeof(float), cudaMemcpyHostToDevice);

        // kernel call
        flash_attention_kernel<<<blockNum, blockSize>>>(
            dev_Q + i * N * d, 
            dev_K + i * N * d, 
            dev_V + i * N * d, 
            dev_O + i * N * d,
            dev_l, dev_m, d
        );

        //free memory of l and m 
        free(l);
        free(m);
        cudaFree(dev_l);
        cudaFree(dev_m);
    }

    end = getTimeStamp();
    printf("(B, N, d): (%d, %d, %d)\n", B, N, d);
    printf("Time: %.3f seconds\n", end - start);

    // copy cuda memory back to host
    cudaMemcpy(O, dev_O, B * N * d * sizeof(float), cudaMemcpyDeviceToHost);

    output(argv[2]);

    // free memory
    cudaFree(dev_Q);
    cudaFree(dev_K);
    cudaFree(dev_V);
    cudaFree(dev_O);

    return 0;
}

void input(char *input_filename) {
    FILE *file = fopen(input_filename, "rb");

    fread(&B, sizeof(int), 1, file);
    fread(&N, sizeof(int), 1, file);
    fread(&d, sizeof(int), 1, file);

    Q = (float *)malloc(B * N * d * sizeof(float));
    K = (float *)malloc(B * N * d * sizeof(float));
    V = (float *)malloc(B * N * d * sizeof(float));
    O = (float *)malloc(B * N * d * sizeof(float));

    for (int i = 0; i < B; i++) {
        fread(Q + (i * N * d), sizeof(float), N * d, file);
        fread(K + (i * N * d), sizeof(float), N * d, file);
        fread(V + (i * N * d), sizeof(float), N * d, file);
    }
    memset(O, 0x00, B * N * d * sizeof(float));

    fclose(file);
}

void output(char *output_filename) {
    FILE *file = fopen(output_filename, "wb");

    fwrite(O, sizeof(float), B * N * d, file);

    free(Q);
    free(K);
    free(V);
    free(O);

    fclose(file);
}

__global__ void flash_attention_kernel(float* q, float* k, float* v, float* o, float* l, float* m, int d){
    int thread_x = threadIdx.x;
    int thread_y = threadIdx.y;
    int block_j = blockIdx.x;
    int block_i = blockIdx.y;
    int dim = d;
    int br = BLOCK_SIZE, bc = BLOCK_SIZE;
    
    // shared memory declaration 
    __shared__ float qi[BLOCK_SIZE * 64];
    __shared__ float kj[BLOCK_SIZE * 64];
    __shared__ float vj[BLOCK_SIZE * 64];
    __shared__ float oi[BLOCK_SIZE * 64];
    __shared__ float li[BLOCK_SIZE];
    __shared__ float mi[BLOCK_SIZE];

    __shared__ float sij[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float pij[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float mij[BLOCK_SIZE];
    __shared__ float lij[BLOCK_SIZE];

    __shared__ float mi_new[BLOCK_SIZE];
    __shared__ float li_new[BLOCK_SIZE];

    // init shared memory
    kj[thread_x * dim + thread_y] = k[(block_j * bc + thread_x) * dim + thread_y];
    vj[thread_x * dim + thread_y] = v[(block_j * bc + thread_x) * dim + thread_y];
    qi[thread_y * dim + thread_x] = q[(block_i * br + thread_y) * dim + thread_x];
    oi[thread_y * dim + thread_x] = o[(block_i * br + thread_y) * dim + thread_x];
    if(dim == 64){
        kj[thread_x * dim + (bc + thread_y)] = k[(block_j * bc + thread_x) * dim + (bc + thread_y)];
        vj[thread_x * dim + (bc + thread_y)] = v[(block_j * bc + thread_x) * dim + (bc + thread_y)];
        qi[thread_y * dim + (br + thread_x)] = q[(block_i * br + thread_y) * dim + (br + thread_x)];
        oi[thread_y * dim + (br + thread_x)] = o[(block_i * br + thread_y) * dim + (br + thread_x)];
    }
    if(thread_x == 0){ // only need one thread for each row to initial li and mi
        li[thread_y] = l[block_i * br + thread_y];
        mi[thread_y] = m[block_i * br + thread_y];
    }
    __syncthreads();

    /* QKDotAndScalar(sij, qi, kj, br, bc, 1.0/sqrt(d)) */
    sij[thread_y * bc + thread_x] = 0.0F;
    for (int t = 0; t < dim; t++)
    {
        sij[thread_y * bc + thread_x] += qi[thread_y * dim + t] * kj[thread_x * dim + t];
    }
    sij[thread_y * bc + thread_x] *= (1.0 / sqrtf(dim));
    __syncthreads();

    /* RowMax(mij, sij, br, bc) */
    if(thread_x == 0){ // only need one thread for each row to compute row max
        mij[thread_y] = sij[thread_y * bc];
        for (int j = 0; j < bc; j++)
        {
            mij[thread_y] = max(mij[thread_y], sij[thread_y * bc + j]);
        }
    }
    __syncthreads();

    /*  MinusMaxAndExp(pij, sij, mij, br, bc) */
    pij[thread_y * bc + thread_x] = expf(sij[thread_y * bc + thread_x] - mij[thread_y]);
    __syncthreads();

    /* RowSum(lij, pij, br, bc) */
    if(thread_x == 0){ // only need one thread for each row to sum up row
        lij[thread_y] = 0.0F;
        for (int j = 0; j < bc; j++)
        {
            lij[thread_y] += pij[thread_y * bc + j];
        }
    }
    __syncthreads();
    
    /* UpdateMiLiOi(mi, li, oi, mij, lij, pij, vj, br, bc) */
    if(thread_x == 0){ // only need one thread for each row to do this
        mi_new[thread_y] = max(mi[thread_y], mij[thread_y]);
        li_new[thread_y] = expf(mi[thread_y] - mi_new[thread_y]) * li[thread_y] 
                         + expf(mij[thread_y] - mi_new[thread_y]) * lij[thread_y];

        __syncthreads();

        for (int j = 0; j < dim; j++)
        {
            float pv = 0.0F;
            for (int t = 0; t < bc; t++)
            {
                pv += pij[thread_y * bc + t] * vj[t * dim + j];
            }
            oi[thread_y * dim + j] = (li[thread_y] * expf(mi[thread_y] - mi_new[thread_y]) * oi[thread_y * dim + j]
                                    + expf(mij[thread_y] - mi_new[thread_y]) * pv) / li_new[thread_y];
        }
    }
    __syncthreads();

    // copy memory back
    o[(block_i * br + thread_y) * dim + thread_x] = oi[thread_y * dim + thread_x];
    if(dim == 64) 
       o[(block_i * br + thread_y) * dim + (br + thread_x)] = oi[thread_y * dim + (br + thread_x)];
    if(thread_x == 0){
        l[block_i * br + thread_y] = li_new[thread_y];
        m[block_i * br + thread_y] = mi_new[thread_y];
    }
}