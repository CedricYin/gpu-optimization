#define FETCH_FLOAT2(ptr) (reinterpret_cast<float2 *>(&(ptr))[0])

__global__ void elementwise1(float *A, float *B, float *C, int SIZE)
{
    int tid = (threadIdx.x + blockDim.x * blockIdx.x) * 2;
    if (tid >= SIZE)
    {
        return;
    }

    float2 reg_A = FETCH_FLOAT2(A[tid]);
    float2 reg_B = FETCH_FLOAT2(B[tid]);

    float2 reg_C;
    reg_C.x = reg_A.x + reg_B.x;
    reg_C.y = reg_A.y + reg_B.y;

    FETCH_FLOAT2(C[tid]) = reg_C;
}
