#include <stdio.h>

const int M = 512;
const int N = 512;
const int K = 512;
const int BLOCK_M = 16;
const int BLOCK_N = 16;
const int BLOCK_K = 16;

/*
    using shared

    一个线程计算一个C的元素
    使用共享内存，减少对全局内存的访问次数
    global访存次数：(K / bk) * (bm * bk + bk * bn) * (M / bm) * (N / bn)，为原来的 1/2 * (1 / BN + 1 / BM)
    BN和BM不能无限增大，因为BM和BN太大会导致SM的利用率不足，且需要更大的共享内存
    单线程内共享内存占用越多，活跃线程束越少，不利于隐藏指令延迟。因此BN和BM的值要适当
    可以通过减小BK的值，减少共享内存的占用

    共享内存比global访问速度快，因此性能得到提升
    计算访存比：还是1/2
*/
template <const int BM, const int BN, const int BK>
__global__ void sgemm1(int M, int N, int K, float *A, float *B, float *C)
{
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    __shared__ float s_A[BM * BK];
    __shared__ float s_B[BK * BN];

    // move to begin pointer
    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    float tmp = 0;
#pragma unroll
    for (int k = 0; k < K; k += BK)
    {
        // global to shared
        s_A[ty * BK + tx] = A[ty * K + tx];
        s_B[ty * BN + tx] = B[ty * N + tx];
        __syncthreads();

// compute
#pragma unroll
        for (int i = 0; i < BK; i++)
        {
            tmp += s_A[ty * BK + i] * s_B[i * BN + tx];
        }
        __syncthreads();

        // move pointer for next compute
        A += BK;
        B += BK * N;
    }

    // write result to global
    C[ty * N + tx] = tmp;
}

void init(float *A, float *B, float *C)
{
    for (int i = 0; i < M; i++)
        for (int j = 0; j < K; j++)
            A[i * K + j] = 1.0 * (i + j);

    for (int i = 0; i < K; i++)
        for (int j = 0; j < N; j++)
            B[i * N + j] = 1.0 * (i + j);

    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
        {
            float tmp = 0;
            for (int k = 0; k < K; k++)
                tmp += A[i * K + k] * B[k * N + j];
            C[i * N + j] = tmp;
        }
}

void check(float *A, float *B)
{
    double diff = 0.0;
    for (int i = 0; i < M * N; i++)
    {
        diff = fabs((double)A[i] - B[i]);
        if (diff > 1e-2)
        {
            printf("Ai: %lf; Bi: %lf\n", A[i], B[i]);
            puts("wrong");
            return;
        }
    }
    puts("right");
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
    dim3 block(BLOCK_N, BLOCK_M);
    dim3 grid(N / BLOCK_N, M / BLOCK_M);
    sgemm1<BLOCK_M, BLOCK_N, BLOCK_K><<<grid, block>>>(M, N, K, d_A, d_B, d_C);
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