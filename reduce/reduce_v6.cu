#include <cuda.h>
#include <stdio.h>

#define FLOAT_SIZE sizeof(float)

const unsigned int THREAD_PER_BLOCK = 256;
const unsigned int N = 32 * 1024 * 1024;
const unsigned int BLOCK_NUM = 1024;
const unsigned int NUM_PER_BLOCK = N / BLOCK_NUM;                     // 一个block要处理的元素数量
const unsigned int NUM_PER_THREAD = NUM_PER_BLOCK / THREAD_PER_BLOCK; // 一个thread要处理的元素数量

__device__ void warpReduce(volatile float *sdata, unsigned int tid)
{
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

// 模板参数，方便调试
template <unsigned int block_size, unsigned int num_per_thread>
__global__ void reduce6(float *d_in, float *d_out)
{
    __shared__ float sdata[block_size];
    unsigned int tid = threadIdx.x;
    // 步幅stride = num_per_thread * block_size
    unsigned int i = threadIdx.x + num_per_thread * block_size * blockIdx.x;

    sdata[tid] = 0; // init
// 根据步幅，累加当前线程要处理的所有元素，且完全展开循环
#pragma unroll
    for (unsigned int j = 0; j < num_per_thread; j++)
    {
        sdata[tid] += d_in[j * block_size + i];
    }
    __syncthreads();

    if (tid < 128)
        sdata[tid] += sdata[tid + 128];
    __syncthreads();

    if (tid < 64)
        sdata[tid] += sdata[tid + 64];
    __syncthreads();

    if (tid < 32)
        warpReduce(sdata, tid);

    if (tid == 0)
        atomicAdd(d_out, sdata[tid]);
}

int main()
{
    float *in, *out, *d_in, *d_out;
    in = (float *)malloc(N * FLOAT_SIZE);
    out = (float *)malloc(BLOCK_NUM * FLOAT_SIZE);
    cudaMalloc((void **)&d_in, N * FLOAT_SIZE);
    cudaMalloc((void **)&d_out, BLOCK_NUM * FLOAT_SIZE);
    for (int i = 0; i < N; i++)
    {
        in[i] = 1;
    }

    cudaMemcpy(d_in, in, N * FLOAT_SIZE, cudaMemcpyHostToDevice);

    dim3 grid(BLOCK_NUM);
    dim3 block(THREAD_PER_BLOCK);
    reduce6<THREAD_PER_BLOCK, NUM_PER_THREAD><<<grid, block>>>(d_in, d_out);

    cudaMemcpy(out, d_out, BLOCK_NUM * FLOAT_SIZE, cudaMemcpyDeviceToHost);

    if (out[0] == 33554432.0)
        puts("the result is right");
    else
        puts("the result is wrong");

    free(in);
    free(out);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}