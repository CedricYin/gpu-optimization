#include <cuda.h>
#include <stdio.h>

// #define THREAD_PER_BLOCK 256
#define FLOAT_SIZE sizeof(float)

const unsigned int THREAD_PER_BLOCK = 256;
const unsigned int N = 32 * 1024 * 1024;
const unsigned int NUM_PER_BLOCK = 2 * THREAD_PER_BLOCK; // 每个block要处理的元素数量：每个block的线程数的两倍（512个数）
const unsigned int BLOCK_NUM = N / NUM_PER_BLOCK;

/*
    每个block要处理的元素数量加倍（block数量减半，每个block的线程数不变），让原来那一半空闲的线程工作起来

    阶段一：
        0 256，1 257，2 258 ... 255 511（处理512个元素，将后256个元素先加到前256个元素上）
        原先一个线程处理一个元素，现在一个线程处理两个元素
    阶段二：
        八轮迭代，同上一个版本

    分析：
        原先：
            一个block处理256个元素，有256个线程

        现在：
            一个block处理512个元素，有256个线程
            每个线程进入迭代阶段前，先进行一次加法，将后一半对应位置的元素加进来
*/
__global__ void reduce3(float *d_in, float *d_out)
{
    __shared__ float sdata[THREAD_PER_BLOCK];
    unsigned int i = 2 * blockDim.x * blockIdx.x + threadIdx.x; // 现在一个block要处理双倍的数，所以这里要*2
    unsigned int tid = threadIdx.x;
    sdata[tid] = d_in[i] + d_in[i + blockDim.x];
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s /= 2)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

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
    reduce3<<<grid, block>>>(d_in, d_out);

    cudaMemcpy(out, d_out, BLOCK_NUM * FLOAT_SIZE, cudaMemcpyDeviceToHost);

    check(out, res, BLOCK_NUM);

    free(in);
    free(out);
    free(res);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}