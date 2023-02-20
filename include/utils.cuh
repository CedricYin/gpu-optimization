#pragma once
#include <cstdio>
#include <cstdlib>

// check runtime API
#define CHECK(call)                                                        \
    do                                                                     \
    {                                                                      \
        const cudaError_t error_code = call;                               \
        if (error_code != cudaSuccess)                                     \
        {                                                                  \
            printf("CUDA Error: \n");                                      \
            printf("\tFile:\t%s\n", __FILE__);                             \
            printf("\tLine:\t%d\n", __LINE__);                             \
            printf("\tError Code:\t%d\n", error_code);                     \
            printf("\tError Text:\t%s\n", cudaGetErrorString(error_code)); \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

// 计时 11 次取平均值（去掉第 1 次启动预热）
#define TIMING(NUM_REPEATS, kernel, grid, block, args...)                      \
    do                                                                         \
    {                                                                          \
        float t_sum = 0;                                                       \
        for (int i = 0; i <= NUM_REPEATS; i++)                                 \
        {                                                                      \
            cudaEvent_t start, stop;                                           \
            CHECK(cudaEventCreate(&start));                                    \
            CHECK(cudaEventCreate(&stop));                                     \
            CHECK(cudaEventRecord(start));                                     \
            cudaEventQuery(start);                                             \
            kernel<<<grid, block>>>(args);                                     \
            CHECK(cudaEventRecord(stop));                                      \
            CHECK(cudaEventSynchronize(stop));                                 \
            float elapsed_time;                                                \
            CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));           \
            if (i > 0)                                                         \
            {                                                                  \
                t_sum += elapsed_time;                                         \
            }                                                                  \
            CHECK(cudaEventDestroy(start));                                    \
            CHECK(cudaEventDestroy(stop));                                     \
        }                                                                      \
        const float t_ave = t_sum / NUM_REPEATS;                               \
        printf("Time ave = %g ms\n", t_ave);                                   \
        printf("Effective Bandwidth = %g GB/s\n", SIZE * 4 * 3 / t_ave / 1e6); \
    } while (0)

/*
NVIDIA GeForce MX450:
理论内存带宽：80 GB/s
有效带宽：每个项目的计算方式不同，这里是 elementwise 的计算方式
*/
