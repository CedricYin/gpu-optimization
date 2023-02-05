// #include <cuda.h>  // 如果使用nvcc编译，则无需添加该头文件
#include <stdio.h>

#define FLOAT_SIZE sizeof(float)

const unsigned int THREAD_PER_BLOCK = 256;
const unsigned int N = 32 * 1024 * 1024;
const unsigned int BLOCK_NUM = 1024;
const unsigned int NUM_PER_BLOCK = N / BLOCK_NUM;  // 一个block要处理的元素数量
const unsigned int NUM_PER_THREAD = NUM_PER_BLOCK / THREAD_PER_BLOCK;  // 一个thread要处理的元素数量

// shuffle函数可以让同一个warp内的线程互相访问寄存器，利用这一点可以减少访问共享内存的开销
__device__ float warpReduce(volatile float *sdata, unsigned int tid) {
    float t = sdata[tid];
    // 假设当前线程号为tid，__shfl_down_sync返回线程号为tid+i的变量t的值（读取寄存器）
    #pragma unroll
    for (unsigned int i = 16; i >= 1; i /= 2) {
        t += __shfl_down_sync(0xffffffff, t, i);
    }
    return t;
}

// 模板参数，方便调试
template<unsigned int block_size, unsigned int num_per_thread>
__global__ void reduce6(float *d_in, float *d_out) {
    __shared__ float sdata[block_size];
    unsigned int tid = threadIdx.x;
    // 步幅stride = num_per_thread * block_size
    unsigned int i = threadIdx.x + num_per_thread * block_size * blockIdx.x;
    
    sdata[tid] = 0;  // init
    // 累加当前线程要处理的所有元素，且完全展开循环
    #pragma unroll
    for (unsigned int j = 0; j < num_per_thread; j++) {
        sdata[tid] += d_in[j * block_size + i];
    }
    __syncthreads();
    
    #pragma unroll
    for (int i = block_size / 2; i > 32; i /= 2) {
        if (tid < i) sdata[tid] += sdata[tid + i];
        __syncthreads();
    }
    
    if (tid < 32) 
        sdata[tid] += sdata[tid + 32];
    __syncwarp();  // 同一个warp内，使用 __syncwarp进行线程同步，比__syncthreads开销更小

    float t = warpReduce(sdata, tid);

    if (tid == 0) atomicAdd(d_out, t);  // 原子操作会丢失一点性能
}

int main() {
    float *in, *out, *d_in, *d_out;
    in = (float*)malloc(N * FLOAT_SIZE);
    out = (float*)malloc(BLOCK_NUM * FLOAT_SIZE);
    cudaMalloc((void**)&d_in, N * FLOAT_SIZE);
    cudaMalloc((void**)&d_out, BLOCK_NUM * FLOAT_SIZE);

    for (int i = 0; i < N; i++) {
        in[i] = 1;
    }

    cudaMemcpy(d_in, in, N * FLOAT_SIZE, cudaMemcpyHostToDevice);

    dim3 grid(BLOCK_NUM);
    dim3 block(THREAD_PER_BLOCK);
    reduce6<THREAD_PER_BLOCK, NUM_PER_THREAD><<<grid, block>>>(d_in, d_out);
    
    cudaMemcpy(out, d_out, BLOCK_NUM * FLOAT_SIZE, cudaMemcpyDeviceToHost);

    if (out[0] == 33554432.0) puts("the result is right");
    else puts("the result is wrong");

    free(in);
    free(out);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}