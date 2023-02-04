#include <cuda.h>
#include <stdio.h>

#define FLOAT_SIZE sizeof(float)

const unsigned int THREAD_PER_BLOCK = 256;
const unsigned int N = 32 * 1024 * 1024;
const unsigned int NUM_PER_BLOCK = THREAD_PER_BLOCK;  // 每个block要处理的元素数量 = 每个block的线程数
const unsigned int BLOCK_NUM = N / NUM_PER_BLOCK;

__global__ void reduce1(float *d_in, float *d_out) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int tid = threadIdx.x;
    sdata[tid] = d_in[i];
    __syncthreads();

    /*
        第一轮迭代：0 1，2 3，4 5，6 7 ... 254 255（0~3号warp进入if分支）
            第128号线程：idx=256 >= THREAD_PER_BLOCK，所以不会进入if分支
            所以128~255号线程（4~7号warp）不会进入if分支
        第二轮迭代：0 2，4 6，8 10 ... 252 254（0~1号warp进入if分支）
            第64号线程：idx=256 >= THREAD_PER_BLOCK，所以不会进入if分支
            所以64~255号线程（2~7号warp）不会进入if分支
        第三轮迭代：0 4，8 12，16 20 ... 248 252（0号warp进入if分支）
            第32号线程：idx=256 >= THREAD_PER_BLOCK，所以不会进入if分支
            所以32~255号线程（1~7号warp）不会进入if分支       
        第四轮迭代：0 8，16 24，32 40 ... 240 248（0号warp的前一半线程进入if分支）
            第16号线程：idx=256 >= THREAD_PER_BLOCK，所以不会进入if分支
            所以16~255号线程（0号warp的后一半线程和1~7号warp）不会进入if分支 
        
        如以此来，就减少了三轮迭代的warp分歧，但存在bank conflict（比如0号和16号线程同时访问到0号bank的不同地址（0和32））
    */
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        unsigned int idx = 2 * s * tid;
        if (idx < blockDim.x) {
            sdata[idx] += sdata[idx + s];
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
    reduce1<<<grid, block>>>(d_in, d_out);
    
    cudaMemcpy(out, d_out, BLOCK_NUM * FLOAT_SIZE, cudaMemcpyDeviceToHost);

    check(out, res, BLOCK_NUM);

    free(in);
    free(out);
    free(res);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}