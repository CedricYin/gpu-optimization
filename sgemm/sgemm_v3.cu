#include <stdio.h>

const int M = 512;
const int N = 512;
const int K = 512;
const int BLOCK_M = 16;
const int BLOCK_N = 16;
const int BLOCK_K = 16;
const int THREAD_M = 2;
const int THREAD_N = 2;

/*
    using register to decreasing the times of shared access
*/

template<const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm3(int M, int N, int K, float *A, float *B, float *C) {
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    // in this kernel, tx and ty is the left top position of a tile
    const int tx = threadIdx.x * TN;
    const int ty = threadIdx.y * TM;
    const int thread_num_per_block = blockDim.x * blockDim.y;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;  // number of current thread

    // paramters about moving a data block (cause threads is less than beofre, one threads shall move more elements)
    const int a_tile_row = tid / BK;
    const int a_tile_col = tid % BK;
    const int a_tile_stride = thread_num_per_block / BK;  // how many rows
    const int b_tile_row = tid / BN;
    const int b_tile_col = tid % BN;
    const int b_tile_stride = thread_num_per_block / BN;

    __shared__ float s_A[BM * BK];
    __shared__ float s_B[BK * BN];

    // register
    float a_frag[TM] = {0};
    float b_frag[TN] = {0};

    // move to begin pointer
    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    float tmp[TM][TN] = {0.0};
    #pragma unroll
    for (int k = 0; k < K; k += BK) {
        // global to shared
        #pragma unroll
        for (int i = 0; i < BM; i += a_tile_stride) {
            s_A[(a_tile_row + i) * BK + a_tile_col] = A[(a_tile_row + i) * K + a_tile_col];
        }
        #pragma unroll
        for (int i = 0; i < BK; i += b_tile_stride) {
            s_B[(b_tile_row + i) * BN + b_tile_col] = B[(b_tile_row + i) * N + b_tile_col];
        }
        __syncthreads();

        // compute
        #pragma unroll
        for (int i = 0; i < BK; i++) {
            // shared to register
            #pragma unroll
            for (int r = 0; r < TM; r++) {
                a_frag[r] = s_A[(ty + r) * BK + i];
            }
            #pragma unroll
            for (int c = 0; c < TN; c++) {
                b_frag[c] = s_B[i * BN + (tx + c)];
            }
            
            // real compute
            #pragma unroll
            for (int r = 0; r < TM; r++) {
                #pragma unroll
                for (int c = 0; c < TN; c++) {
                    tmp[r][c] += a_frag[r] * b_frag[c];
                }
            }
        }
        __syncthreads();

        // move pointer for next compute
        A += BK;
        B += BK * N;
    }

    // write result to global
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            C[(ty + i) * N + (tx + j)] = tmp[i][j];
        }
    }
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
    dim3 block(BLOCK_N / THREAD_N, BLOCK_M / THREAD_M);  
    dim3 grid(N / BLOCK_N, M / BLOCK_M);
    sgemm3<BLOCK_M, BLOCK_N, BLOCK_K, THREAD_M, THREAD_N><<<grid, block>>>(M, N, K, d_A, d_B, d_C);
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