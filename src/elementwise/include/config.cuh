#define SIZE 1024 * 1024 * 32
#define BLOCK_SIZE 256

float t_ave = 0; // time average

const int NUM_REPEATS = 2000;

float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C, *C; // C: for comparing answer

dim3 block0(BLOCK_SIZE);
dim3 grid0((SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);

dim3 block1(BLOCK_SIZE);
dim3 grid1((SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE / 2); // 除以 2 是因为向量访存，一个线程处理 2 个元素，所以总线程数量降为二分之一

dim3 block2(BLOCK_SIZE);
dim3 grid2((SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE / 4); // 除以 4 是因为向量访存，一个线程处理 4 个元素，所以总线程数量降为四分之一
