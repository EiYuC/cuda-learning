#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>

// CUDA 出错时打印原因并退出，方便定位问题。
void check_cuda(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA 错误: " << cudaGetErrorString(result) << '\n';
        std::exit(1);
    }
}

// __global__ 表示这个函数在 GPU 上运行，称为 kernel。
// 每个线程只计算一个位置：c[i] = a[i] + b[i]。
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    // 1. 在 CPU 上准备 8 个数。h 表示 host（CPU），d 表示 device（GPU）。
    constexpr int n = 8;
    const size_t bytes = n * sizeof(float);
    float h_a[n], h_b[n], h_c[n];
    for (int i = 0; i < n; ++i) {
        h_a[i] = i + 1;
        h_b[i] = (i + 1) * 10;
    }

    // 2. 在 GPU 上为输入和输出分配显存。
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    check_cuda(cudaMalloc(&d_a, bytes));
    check_cuda(cudaMalloc(&d_b, bytes));
    check_cuda(cudaMalloc(&d_c, bytes));

    // 3. 把两个输入数组从 CPU 复制到 GPU。
    check_cuda(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // 4. 启动 GPU kernel。一个 block 中放 256 个线程。
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    check_cuda(cudaGetLastError());          // 检查启动是否成功。
    check_cuda(cudaDeviceSynchronize());     // 等待 GPU 计算完成，并检查执行错误。

    // 5. 把结果复制回 CPU，然后打印。
    check_cuda(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) {
        std::cout << h_a[i] << " + " << h_b[i] << " = " << h_c[i] << '\n';
    }

    // 6. 释放自己申请的 GPU 显存。
    check_cuda(cudaFree(d_a));
    check_cuda(cudaFree(d_b));
    check_cuda(cudaFree(d_c));
    return 0;
}
