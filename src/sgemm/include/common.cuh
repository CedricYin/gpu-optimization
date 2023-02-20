#include <cstdlib>
#include <cstdio>

const int M = 512;
const int N = 512;
const int K = 512;
const int BLOCK_M = 32; // v0 v1 要注意，1 个 block 不能超过 1024 个线程
const int BLOCK_N = 32;
const int BLOCK_K = 8;
const int THREAD_M = 8;
const int THREAD_N = 8;

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

void check_ans(float *A, float *B)
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