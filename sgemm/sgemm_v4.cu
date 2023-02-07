#include <stdio.h>

#define OFFSET(row, col, ld) ((row) * ld + col)
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

const int M = 512;
const int N = 512;
const int K = 512;
const int BLOCK_M = 64;
const int BLOCK_N = 64;
const int BLOCK_K = 8;
const int THREAD_M = 8;
const int THREAD_N = 8;

/*
    using float4
*/

template<const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm4(int M, int N, int K, float *A, float *B, float *C) {
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    // in this kernel, tx and ty is the left top position of a tile
    const int tx = threadIdx.x * TN;
    const int ty = threadIdx.y * TM;

    const int block_dim_x = BN / TN;  // 为了赋予原生的 blockDim.x和blockDim.y const属性
    const int block_dim_y = BM / TM;
    const int thread_num_per_block = block_dim_x * block_dim_y;
    const int tid = threadIdx.y * block_dim_x + threadIdx.x;  // number of current thread

    const int move_num = thread_num_per_block * 4;  // 一个block的线程，一次可以搬运的元素数量
    // move times
    const int ldg_a_num = BM * BK / move_num;
    const int ldg_b_num = BK * BN / move_num;

    // 计算ldg_a_num的所有参数必须全部是const，否则不能用来申明数组大小
    // 然而blockDim.x 和 blockDim.y非const，所以这里将blockDim.x 和 blockDim.y用BN/TN 和 BM/TM表示
    float ldg_a_reg[4 * ldg_a_num] = {0.0};  // 每个线程搬运ldg_a_num轮，寄存器缓存ldg_a_num个float4元素，用于转置s_A

    // paramters about moving a data block (cause threads is less than beofre, one threads shall move more elements)
    const int a_tile_row = tid / (BK / 4);
    const int a_tile_col = tid % (BK / 4) * 4;
    const int a_tile_stride = BM / ldg_a_num;  // how many rows
    const int b_tile_row = tid / (BN / 4);
    const int b_tile_col = tid % (BN / 4) * 4;
    const int b_tile_stride = BK / ldg_b_num;

    __shared__ float s_A[BM * BK];
    __shared__ float s_B[BK * BN];

    // register
    float a_frag[TM] = {0.0};
    float b_frag[TN] = {0.0};

    // move to begin pointer
    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    float tmp[TM][TN] = {0.0};
    #pragma unroll
    for (int k = 0; k < K; k += BK) {
        // global to shared using float4
        #pragma unroll
        for (int i = 0; i < BM; i += a_tile_stride) {
            int ldg_index = i / a_tile_stride * 4;  // 第ldg_index轮
            FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(a_tile_row + i, a_tile_col, K)]);
            // 转置（按行取，按列存）
            s_A[OFFSET(a_tile_col, i + a_tile_row, BM)] = ldg_a_reg[ldg_index];
            s_A[OFFSET(a_tile_col + 1, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 1];
            s_A[OFFSET(a_tile_col + 2, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 2];
            s_A[OFFSET(a_tile_col + 3, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 3];
        }
        #pragma unroll
        for (int i = 0; i < BK; i += b_tile_stride) {
            FETCH_FLOAT4(s_B[OFFSET(b_tile_row + i, b_tile_col, BN)]) = FETCH_FLOAT4(B[OFFSET(b_tile_row + i, b_tile_col, N)]);
        }
        __syncthreads();

        // compute
        #pragma unroll
        for (int i = 0; i < BK; i++) {
            // shared to register using float4
            #pragma unroll
            for (int r = 0; r < TM; r += 4) {
                FETCH_FLOAT4(a_frag[r]) = FETCH_FLOAT4(s_A[OFFSET(i, ty + r, BM)]);
            }
            #pragma unroll
            for (int c = 0; c < TN; c += 4) {
                FETCH_FLOAT4(b_frag[c]) = FETCH_FLOAT4(s_B[OFFSET(i, tx + c, BN)]);
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

    // write result to global using float4
    #pragma unroll
    for (int r = 0; r < TM; r++) {
        #pragma unroll
        for (int c = 0; c < TN; c += 4) {
            FETCH_FLOAT4(C[OFFSET(ty + r, tx + c, N)]) = FETCH_FLOAT4(tmp[r][c]);
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
    sgemm4<BLOCK_M, BLOCK_N, BLOCK_K, THREAD_M, THREAD_N><<<grid, block>>>(M, N, K, d_A, d_B, d_C);
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