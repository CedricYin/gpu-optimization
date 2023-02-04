#include <cuda.h>
#include <stdio.h>

#define FLOAT_SIZE sizeof(float)

const unsigned int THREAD_PER_BLOCK = 256;
const unsigned int N = 32 * 1024 * 1024;
const unsigned int NUM_PER_BLOCK = 2 * THREAD_PER_BLOCK;
const unsigned int BLOCK_NUM = N / NUM_PER_BLOCK;

__device__ void warpReduce(volatile float* sdata, unsigned int tid) {
    sdata[tid] += sdata[tid + 32]; 
    sdata[tid] += sdata[tid + 16]; 
    sdata[tid] += sdata[tid + 8]; 
    sdata[tid] += sdata[tid + 4]; 
    sdata[tid] += sdata[tid + 2]; 
    sdata[tid] += sdata[tid + 1];
}

__global__ void reduce5(float *d_in, float *d_out) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    unsigned int i = 2 * blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int tid = threadIdx.x;
    sdata[tid] = d_in[i] + d_in[i + blockDim.x];
    __syncthreads();

    /*
        完全展开下面的（上个版本的）for循环，节省循环的开销
        for (unsigned int s = blockDim.x / 2; s > 32; s /= 2) {
            if (tid < s) {
                sdata[tid] = sdata[tid + s];
            }
            __syncthreads();
        }
    */

   if (tid < 128) {
        sdata[tid] += sdata[tid + 128];
   }
   __syncthreads();

   if (tid < 64) {
        sdata[tid] += sdata[tid + 64];
   }
    __syncthreads();

    if (tid < 32)
        warpReduce(sdata, tid);

    if (tid == 0)
        d_out[blockIdx.x] = sdata[tid];
}

void check(float *out, float *res, int n) {
    for (int i = 0; i < n; i++) {
        if (out[i] != res[i]) {
            puts("the result is wrong");
            return ;
        }
    }
    puts("the result is true");
}

int main() {
    float *in, *out, *d_in, *d_out, *res;
    in = (float*)malloc(N * FLOAT_SIZE);
    out = (float*)malloc(BLOCK_NUM * FLOAT_SIZE);
    cudaMalloc((void**)&d_in, N * FLOAT_SIZE);
    cudaMalloc((void**)&d_out, BLOCK_NUM * FLOAT_SIZE);
    for (int i = 0; i < N; i++) {
        in[i] = 1;
    }

    res = (float*)malloc(BLOCK_NUM * FLOAT_SIZE);
    for (int i = 0; i < BLOCK_NUM; i++) {
        float cur = 0;
        for (int j = 0; j < NUM_PER_BLOCK; j++) {
            if (i * NUM_PER_BLOCK + j < N)
                cur += in[i * NUM_PER_BLOCK + j];
        }
        res[i] = cur;
    }

    cudaMemcpy(d_in, in, N * FLOAT_SIZE, cudaMemcpyHostToDevice);

    dim3 grid(BLOCK_NUM);
    dim3 block(THREAD_PER_BLOCK);
    reduce5<<<grid, block>>>(d_in, d_out);
    
    cudaMemcpy(out, d_out, BLOCK_NUM * FLOAT_SIZE, cudaMemcpyDeviceToHost);

    check(out, res, BLOCK_NUM);

    free(in);
    free(out);
    free(res);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}