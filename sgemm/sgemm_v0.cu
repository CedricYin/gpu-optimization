#include <stdio.h>

const int M = 512;
const int N = 512;
const int K = 512;
const int BLOCK_M = 16;
const int BLOCK_N = 16;

/*
    baseline
*/
__global__ void sgemm0(int M, int N, int K, float *A, float *B, float *C) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    float tmp = 0;
    #pragma unroll
    for (int i = 0; i < K; i++) {
        tmp += A[y * K + i] * B[i * N + x];
    }
    C[y * N + x] = tmp;
}

void init(float *A, float *B, float *C) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < K; j++)
            A[i * K + j] = 1.0 * (i + j);
    
    for (int i = 0; i < K; i++)
        for (int j = 0; j < N; j++)
            B[i * N + j] = 1.0 * (i + j);
    
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float tmp = 0;
            for (int k = 0; k < K; k++)
                tmp += A[i * K + k] * B[k * N + j];
            C[i * N + j] = tmp;
        }

}

void check(float *A, float *B) {
    double diff = 0.0;
    for (int i = 0; i < M * N; i++) {
        diff = fabs((double)A[i] - B[i]);
        if (diff > 1e-2) {
            printf("Ai: %lf; Bi: %lf\n", A[i], B[i]);
            puts("wrong");
            return ;
        }
    }
    puts("right");
}

int main() {
    float *A = (float*)malloc(sizeof(float) * M * K);
    float *B = (float*)malloc(sizeof(float) * K * N);
    float *C = (float*)malloc(sizeof(float) * M * N);
    float *C_base = (float*)malloc(sizeof(float) * M * N);
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, sizeof(float) * M * K);
    cudaMalloc((void**)&d_B, sizeof(float) * K * N);
    cudaMalloc((void**)&d_C, sizeof(float) * M * N);

    init(A, B, C_base);

    cudaMemcpy(d_A, A, sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(float) * K * N, cudaMemcpyHostToDevice);
    dim3 block(BLOCK_N, BLOCK_M);  // 注意：一个block最多有1024个线程，定义dim时，按顺序书写一维x，二维y，三维z ...
    dim3 grid(N / BLOCK_N, M / BLOCK_M);
    sgemm0<<<grid, block>>>(M, N, K, d_A, d_B, d_C);
    cudaMemcpy(C, d_C, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    check(C, C_base);

    free(A);
    free(B);
    free(C);
    free(C_base);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}