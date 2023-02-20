#include <stdio.h>
#include "utils.cuh"
#include "common.cuh"

#define OFFSET(row, col, ld) ((row)*ld + col)
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

/*
    benchmark of prefetch

    分离了global to shared以及小迭代
    在gpu中，只有寄存器和global间、寄存器和shared间的指令，所以即使代码写的是global to shared，底层指令中也会经过寄存器进行中转
    因此，这里显式地分离global to shared，使得增加访存和计算重叠的机会
    分离小迭代为7+1也是一样的道理，增加访存和计算重叠的机会
    如此一来，可以隐藏延迟

    所以大迭代的逻辑为：
        global -> register (中转)
        BK-1次小迭代
        register -> shared (预取)
        最后1次小迭代

    v5大迭代的逻辑：（这里放出来做对比）
        BK次小迭代
        global -> shared (预取。这里底层是global->register->shared)
*/

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm6(int M, int N, int K, float *A, float *B, float *C)
{
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    // in this kernel, tx and ty is the left top position of a tile
    const int tx = threadIdx.x * TN;
    const int ty = threadIdx.y * TM;

    const int block_dim_x = BN / TN; // 为了赋予原生的 blockDim.x和blockDim.y const属性
    const int block_dim_y = BM / TM;
    const int thread_num_per_block = block_dim_x * block_dim_y;
    const int tid = threadIdx.y * block_dim_x + threadIdx.x; // number of current thread

    const int move_num = thread_num_per_block * 4; // 一个block的线程，一次可以搬运的元素数量
    // move times
    const int ldg_a_num = BM * BK / move_num;
    const int ldg_b_num = BK * BN / move_num;

    float ldg_a_reg[4 * ldg_a_num] = {0.0}; // 用于转置s_A，以及暂存A数据
    float ldg_b_reg[4 * ldg_b_num] = {0.0}; // 用于暂存B数据

    // paramters about moving a data block (cause threads is less than beofre, one threads shall move more elements)
    const int a_tile_row = tid / (BK / 4);
    const int a_tile_col = tid % (BK / 4) * 4;
    const int a_tile_stride = BM / ldg_a_num; // how many rows
    const int b_tile_row = tid / (BN / 4);
    const int b_tile_col = tid % (BN / 4) * 4;
    const int b_tile_stride = BK / ldg_b_num;

    // shared 和 register 都开双倍，用于实现数据预取
    __shared__ float s_A[2][BM * BK];
    __shared__ float s_B[2][BK * BN];
    float a_frag[2][TM] = {0.0};
    float b_frag[2][TN] = {0.0};

    // move to begin pointer
    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

// 先预取一次（global to shared & shared to register），为第一次迭代做准备
// global to shared
#pragma unroll
    for (int i = 0; i < BM; i += a_tile_stride)
    {
        int ldg_index = i / a_tile_stride * 4; // 第ldg_index轮
        FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(a_tile_row + i, a_tile_col, K)]);
        // 转置（按行取，按列存）
        s_A[0][OFFSET(a_tile_col, i + a_tile_row, BM)] = ldg_a_reg[ldg_index];
        s_A[0][OFFSET(a_tile_col + 1, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 1];
        s_A[0][OFFSET(a_tile_col + 2, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 2];
        s_A[0][OFFSET(a_tile_col + 3, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 3];
    }
#pragma unroll
    for (int i = 0; i < BK; i += b_tile_stride)
    {
        FETCH_FLOAT4(s_B[0][OFFSET(b_tile_row + i, b_tile_col, BN)]) = FETCH_FLOAT4(B[OFFSET(b_tile_row + i, b_tile_col, N)]);
    }
    __syncthreads();
// shared to register
#pragma unroll
    for (int r = 0; r < TM; r += 4)
    {
        FETCH_FLOAT4(a_frag[0][r]) = FETCH_FLOAT4(s_A[0][OFFSET(0, ty + r, BM)]);
    }
#pragma unroll
    for (int c = 0; c < TN; c += 4)
    {
        FETCH_FLOAT4(b_frag[0][c]) = FETCH_FLOAT4(s_B[0][OFFSET(0, tx + c, BN)]);
    }

    // 大迭代
    int write_index = 1;
    int read_index;
    float tmp[TM][TN] = {0.0};
    int k = 0;
    do
    {
        k += BK;
        read_index = write_index ^ 1;

        // 如果还有下一个迭代，则将下一个迭代的数据块，搬运到寄存器上暂存（这里对A不用转置）
        if (k < K)
        {
#pragma unroll
            for (int i = 0; i < BM; i += a_tile_stride)
            {
                int ldg_index = i / a_tile_stride * 4; // 第ldg_index轮
                FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(a_tile_row + i, k + a_tile_col, K)]);
            }
#pragma unroll
            for (int i = 0; i < BK; i += b_tile_stride)
            {
                int ldg_index = i / b_tile_stride * 4; // 第ldg_index轮
                FETCH_FLOAT4(ldg_b_reg[ldg_index]) = FETCH_FLOAT4(B[OFFSET(k + b_tile_row + i, b_tile_col, N)]);
            }
        }

// BK -1 次小迭代
#pragma unroll
        for (int i = 0; i < BK - 1; i++)
        {
// 将下一个小迭代的数据块，搬运到寄存器上
#pragma unroll
            for (int r = 0; r < TM; r += 4)
            {
                FETCH_FLOAT4(a_frag[(i + 1) % 2][r]) = FETCH_FLOAT4(s_A[read_index][OFFSET(i + 1, ty + r, BM)]);
            }
#pragma unroll
            for (int c = 0; c < TN; c += 4)
            {
                FETCH_FLOAT4(b_frag[(i + 1) % 2][c]) = FETCH_FLOAT4(s_B[read_index][OFFSET(i + 1, tx + c, BN)]);
            }
// 计算tile
#pragma unroll
            for (int r = 0; r < TM; r++)
            {
#pragma unroll
                for (int c = 0; c < TN; c++)
                {
                    tmp[r][c] += a_frag[i % 2][r] * b_frag[i % 2][c];
                }
            }
        }
        // __syncthreads() 不需要了，所以先完成小迭代的线程，可以直接往下执行，进行数据的预取。由此可以实现读写并行

        // prefetch
        if (k < K)
        {
// 将存储在临时寄存器的数据搬运到shared memory中，完成shared memory的预取
#pragma unroll
            for (int i = 0; i < BM; i += a_tile_stride)
            {
                int ldg_index = i / a_tile_stride * 4;
                s_A[write_index][OFFSET(a_tile_col, i + a_tile_row, BM)] = ldg_a_reg[ldg_index];
                s_A[write_index][OFFSET(a_tile_col + 1, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 1];
                s_A[write_index][OFFSET(a_tile_col + 2, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 2];
                s_A[write_index][OFFSET(a_tile_col + 3, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 3];
            }
#pragma unroll
            for (int i = 0; i < BK; i += b_tile_stride)
            {
                int ldg_index = i / b_tile_stride * 4;
                FETCH_FLOAT4(s_B[write_index][OFFSET(b_tile_row + i, b_tile_col, BN)]) = FETCH_FLOAT4(ldg_b_reg[ldg_index]);
            }
            // use double buffer, only need one sync
            __syncthreads();

// 完成寄存器的预取
#pragma unroll
            for (int r = 0; r < TM; r += 4)
            {
                FETCH_FLOAT4(a_frag[0][r]) = FETCH_FLOAT4(s_A[write_index][OFFSET(0, ty + r, BM)]);
            }
#pragma unroll
            for (int c = 0; c < TN; c += 4)
            {
                FETCH_FLOAT4(b_frag[0][c]) = FETCH_FLOAT4(s_B[write_index][OFFSET(0, tx + c, BN)]);
            }
        }

// 将最后一个小迭代完成
#pragma unroll
        for (int r = 0; r < TM; r++)
        {
#pragma unroll
            for (int c = 0; c < TN; c++)
            {
                tmp[r][c] += a_frag[(BK - 1) % 2][r] * b_frag[(BK - 1) % 2][c];
            }
        }

        write_index ^= 1;

    } while (k < K);

// write result to global using float4
#pragma unroll
    for (int r = 0; r < TM; r++)
    {
#pragma unroll
        for (int c = 0; c < TN; c += 4)
        {
            FETCH_FLOAT4(C[OFFSET(ty + r, tx + c, N)]) = FETCH_FLOAT4(tmp[r][c]);
        }
    }
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
    dim3 block(BLOCK_N / THREAD_N, BLOCK_M / THREAD_M);
    dim3 grid(N / BLOCK_N, M / BLOCK_M);
    sgemm6<BLOCK_M, BLOCK_N, BLOCK_K, THREAD_M, THREAD_N><<<grid, block>>>(M, N, K, d_A, d_B, d_C);
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