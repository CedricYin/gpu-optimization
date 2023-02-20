#include <cuda.h>
#include <stdio.h>

#define FLOAT_SIZE sizeof(float)

const unsigned int THREAD_PER_BLOCK = 256;
const unsigned int N = 32 * 1024 * 1024;
const unsigned int NUM_PER_BLOCK = THREAD_PER_BLOCK; // 每个block要处理的元素数量 = 每个block的线程数
const unsigned int BLOCK_NUM = N / NUM_PER_BLOCK;

__global__ void reduce0(float *d_in, float *d_out)
{
    // 将 256 个元素存在共享内存中，每个线程负责一个元素
    __shared__ float sdata[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = d_in[i];
    __syncthreads();

    // baseline
    for (unsigned int s = 1; s < blockDim.x; s *= 2)
    {
        if (tid % (2 * s) == 0)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 将本 block 最终 reduce 后的结果存在 d_out 对应位置
    if (tid == 0)
        d_out[blockIdx.x] = sdata[tid];
}

void check(float *out, float *res, int n)
{
    for (int i = 0; i < n; i++)
    {
        if (out[i] != res[i])
        {
            puts("the result is wrong");
            return;
        }
    }
    puts("the result is true");
}

int main()
{
    float *in, *out, *d_in, *d_out, *res;
    in = (float *)malloc(N * FLOAT_SIZE);
    out = (float *)malloc(BLOCK_NUM * FLOAT_SIZE);
    cudaMalloc((void **)&d_in, N * FLOAT_SIZE);
    cudaMalloc((void **)&d_out, BLOCK_NUM * FLOAT_SIZE);
    for (int i = 0; i < N; i++)
    {
        in[i] = 1;
    }

    res = (float *)malloc(BLOCK_NUM * FLOAT_SIZE);
    for (int i = 0; i < BLOCK_NUM; i++)
    {
        float cur = 0;
        for (int j = 0; j < NUM_PER_BLOCK; j++)
        {
            if (i * NUM_PER_BLOCK + j < N)
                cur += in[i * NUM_PER_BLOCK + j];
        }
        res[i] = cur;
    }

    cudaMemcpy(d_in, in, N * FLOAT_SIZE, cudaMemcpyHostToDevice);

    dim3 grid(BLOCK_NUM);
    dim3 block(THREAD_PER_BLOCK);
    reduce0<<<grid, block>>>(d_in, d_out);

    cudaMemcpy(out, d_out, BLOCK_NUM * FLOAT_SIZE, cudaMemcpyDeviceToHost);

    check(out, res, BLOCK_NUM);

    free(in);
    free(out);
    free(res);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}