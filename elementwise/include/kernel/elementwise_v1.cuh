#define FETCH_FLOAT4(ptr) (reinterpret_cast<float4 *>(&(ptr))[0])

__global__ void elementwise1(float *A, float *B, float *C, int SIZE)
{
    int tid = (threadIdx.x + blockDim.x * blockIdx.x) * 4;
    if (tid >= SIZE)
    {
        return;
    }

    float4 reg_A = FETCH_FLOAT4(A[tid]);
    float4 reg_B = FETCH_FLOAT4(B[tid]);

    float4 reg_C;
    reg_C.x = reg_A.x + reg_B.x;
    reg_C.y = reg_A.y + reg_B.y;
    reg_C.z = reg_A.z + reg_B.z;
    reg_C.w = reg_A.w + reg_B.w;

    FETCH_FLOAT4(C[tid]) = reg_C;
}
