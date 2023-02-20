__global__ void elementwise0(float *A, float *B, float *C, int SIZE)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= SIZE)
    {
        return;
    }
    C[tid] = A[tid] + B[tid];
}