#include "utils.cuh"
#include "kernel.cuh"
#include "common.cuh"

#define SIZE 1024 * 1024 * 50
#define BLOCK_SIZE 256

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        puts("usage: ./main [kernel id]");
        exit(1);
    }
    int kernel_id = atoi(argv[1]);

    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C, *C;
    h_A = (float *)malloc(sizeof(float) * SIZE);
    h_B = (float *)malloc(sizeof(float) * SIZE);
    h_C = (float *)malloc(sizeof(float) * SIZE);
    C = (float *)malloc(sizeof(float) * SIZE);
    CHECK(cudaMalloc(&d_A, sizeof(float) * SIZE));
    CHECK(cudaMalloc(&d_B, sizeof(float) * SIZE));
    CHECK(cudaMalloc(&d_C, sizeof(float) * SIZE));

    init(h_A, h_B, C, SIZE);

    CHECK(cudaMemcpy(d_A, h_A, sizeof(float) * SIZE, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, sizeof(float) * SIZE, cudaMemcpyHostToDevice));

    dim3 block0(BLOCK_SIZE);
    dim3 grid0((SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block1(BLOCK_SIZE);
    dim3 grid1((SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE / 4);
    switch (kernel_id)
    {
    case 0:
        puts("running kernel elementwise_v0 ...");
        TIMING(100, elementwise0, grid0, block0, d_A, d_B, d_C, SIZE);
        break;
    case 1:
        puts("running kernel elementwise_v1 ...");
        TIMING(100, elementwise1, grid1, block1, d_A, d_B, d_C, SIZE);
        break;
    default:
        break;
    }

    CHECK(cudaMemcpy(h_C, d_C, sizeof(float) * SIZE, cudaMemcpyDeviceToHost));
    check_ans(C, h_C, SIZE);

    free(h_A);
    free(h_B);
    free(h_C);
    free(C);
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
}