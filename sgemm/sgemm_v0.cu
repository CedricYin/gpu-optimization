#include "utils.cuh"
#include "common.cuh"

/*
    baseline
*/
__global__ void sgemm0(int M, int N, int K, float *A, float *B, float *C)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    float tmp = 0;
#pragma unroll
    for (int i = 0; i < K; i++)
    {
        tmp += A[y * K + i] * B[i * N + x];
    }
    C[y * N + x] = tmp;
}

int main()
{
    float *A = (float *)malloc(sizeof(float) * M * K);
    float *B = (float *)malloc(sizeof(float) * K * N);
    float *C = (float *)malloc(sizeof(float) * M * N);
    float *C_base = (float *)malloc(sizeof(float) * M * N);
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, sizeof(float) * M * K);
    cudaMalloc((void **)&d_B, sizeof(float) * K * N);
    cudaMalloc((void **)&d_C, sizeof(float) * M * N);

    init(A, B, C_base);

    cudaMemcpy(d_A, A, sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(float) * K * N, cudaMemcpyHostToDevice);
    dim3 block(BLOCK_N, BLOCK_M); // 注意：一个block最多有1024个线程，定义dim时，按顺序书写一维x，二维y，三维z ...
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    TIMING(10, sgemm0, grid, block, M, N, K, d_A, d_B, d_C);
    cudaMemcpy(C, d_C, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    check_ans(C, C_base);

    free(A);
    free(B);
    free(C);
    free(C_base);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}