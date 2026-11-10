# 编译和运行

在服务器的 SSH 终端中执行：

```bash
cd ~/cuda-learning
source ~/.miniconda3/etc/profile.d/conda.sh
conda activate aiinfra

nvcc -arch=sm_80 vector_add.cu -o vector_add
./vector_add
```

- `nvcc`：CUDA 编译器。
- `-arch=sm_80`：指定这台服务器的 A800 GPU 架构。
- `vector_add.cu`：源代码文件。
- `-o vector_add`：指定生成的可执行文件名。
- `./vector_add`：运行生成的程序。

编译自己新建的文件时，替换文件名即可：

```bash
nvcc -arch=sm_80 my_kernel.cu -o my_kernel
./my_kernel
```

# 注意事项

- 每次打开新终端，先激活 `aiinfra` 环境。
- 修改 `.cu` 文件后，需要重新编译，再运行。
- 多个文件各自包含 `main()` 时，分别编译运行。
- 先使用小数组练习；这台 GPU 是共享资源，避免大量显存占用和长时间循环。
- 保留 `if (i < n)` 等边界检查，防止访问数组范围之外的内存。
- 申请显存和复制数据时，传入字节数，例如 `n * sizeof(float)`。
- 分清 CPU 和 GPU 内存，使用 `cudaMemcpy` 复制数据，用完显存后调用 `cudaFree`。
- 保留 CUDA 错误检查；入门阶段使用 `cudaDeviceSynchronize()` 等待 GPU 完成并检查执行错误。
- 不修改系统驱动、CUDA Toolkit 或 `/usr/local/cuda` 软链接，不重置 GPU。
