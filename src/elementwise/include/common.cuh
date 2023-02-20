#include <cstdio>
#include <cstdlib>

void init(float *A, float *B, float *C, int SIZE)
{
    for (int i = 0; i < SIZE; i++)
    {
        A[i] = 1.0 * i;
        B[i] = 2.0 * i;
    }

    for (int i = 0; i < SIZE; i++)
    {
        C[i] = A[i] + B[i];
    }
}

void check_ans(float *A, float *B, int SIZE)
{
    for (int i = 0; i < SIZE; i++)
    {
        if (A[i] != B[i])
        {
            puts("answer is wrong");
            exit(1);
        }
    }
    puts("answer is right");
}