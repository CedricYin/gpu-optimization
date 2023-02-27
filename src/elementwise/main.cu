#include "utils.cuh"
#include "kernel.cuh"
#include "common.cuh"
#include "config.cuh"

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        puts("usage: ./main [kernel id]");
        exit(1);
    }
    int kernel_id = atoi(argv[1]);

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

    switch (kernel_id)
    {
    case 0:
        puts("running kernel elementwise_v0 ...");
        TIMING(0, NUM_REPEATS, elementwise0, grid0, block0, t_ave, -1, -1, -1, -1, -1, d_A, d_B, d_C, SIZE);
        break;
    case 1:
        puts("running kernel elementwise_v1 ...");
        TIMING(0, NUM_REPEATS, elementwise1, grid1, block1, t_ave, -1, -1, -1, -1, -1, d_A, d_B, d_C, SIZE);
        break;
    case 2:
        puts("running kernel elementwise_v2 ...");
        TIMING(0, NUM_REPEATS, elementwise2, grid2, block2, t_ave, -1, -1, -1, -1, -1, d_A, d_B, d_C, SIZE);
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