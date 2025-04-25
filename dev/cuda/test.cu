#include <stdio.h>
#include <cuda_runtime.h>

// 错误检查宏（建议每个CUDA API调用后都使用）
#define checkCudaError(err) __checkCudaError(err, __FILE__, __LINE__)
inline void __checkCudaError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        printf("CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// 向量加法核函数
__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int N = 128;
    size_t size = N * sizeof(float);

    // 分配主机内存
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);

    // 初始化数据
    for (int i = 0; i < N; ++i) {
        h_a[i] = 1.0f;  // 每个元素都是1.0
        h_b[i] = 2.0f;  // 每个元素都是2.0
    }

    // 分配设备内存
    float *d_a, *d_b, *d_c;
    checkCudaError(cudaMalloc(&d_a, size));
    checkCudaError(cudaMalloc(&d_b, size));
    checkCudaError(cudaMalloc(&d_c, size));

    // 拷贝数据到设备
    checkCudaError(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // 启动核函数
    int blockSize = 128;
    int gridSize = (N + blockSize - 1) / blockSize;
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);

    // 同步设备并检查执行错误（mentor强调的步骤）
    auto err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // 拷贝结果回主机
    checkCudaError(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    // 验证结果
    bool success = true;
    for (int i = 0; i < N; ++i) {
        if (fabs(h_c[i] - 3.0f) > 1e-5) {  // 1+2=3
            printf("Error at index %d: %f != 3.0\n", i, h_c[i]);
            success = false;
            break;  // 发现第一个错误即停止
        }
        printf("%f",h_c[i]);
    }
    if (success) {
        printf("Vector add completed successfully!\n");
    }

    // 释放内存
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}