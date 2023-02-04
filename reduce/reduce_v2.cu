#include <cuda.h>
#include <stdio.h>

#define FLOAT_SIZE sizeof(float)

const unsigned int THREAD_PER_BLOCK = 256;
const unsigned int N = 32 * 1024 * 1024;
const unsigned int NUM_PER_BLOCK = THREAD_PER_BLOCK;  // 每个block要处理的元素数量 = 每个block的线程数
const unsigned int BLOCK_NUM = N / NUM_PER_BLOCK;

__global__ void reduce2(float *d_in, float *d_out) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    sdata[tid] = d_in[i];
    __syncthreads();

    /*
        第一轮迭代（s=128）：0 128，1 129，2 130 ... 127 255（前一半线程在工作）
            同一个warp的不同线程刚好访问了一行共享内存（不同bank）
        第二轮迭代（s=64）：0 64，1 65，2 66 ... 63 127（前四分之一线程在工作）
            同一个warp的不同线程刚好访问了一行共享内存（不同bank）
        第三轮迭代（s=32）：0 32，1 33，2 34 ... 31 63（前八分之一线程在工作）
            同一个warp的不同线程刚好访问了一行共享内存（不同bank）
        第四轮迭代（s=16）：0 16，1 17，2 18 ... 15 31（前十六分之一线程在工作）
            同一个warp的不同线程刚好访问了半行共享内存（不同bank）
        第五轮s=8
        第六轮s=4
        第七轮s=2
        第八轮s=1，结束

        但有个问题，每轮的工作线程都减半，其余线程阻塞在 __syncthreads()，导致线程的浪费
    */
    for (unsigned int s = blockDim.x / 2; s > 0; s /= 2) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

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
    reduce2<<<grid, block>>>(d_in, d_out);
    
    cudaMemcpy(out, d_out, BLOCK_NUM * FLOAT_SIZE, cudaMemcpyDeviceToHost);

    check(out, res, BLOCK_NUM);

    free(in);
    free(out);
    free(res);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}