# CUDA 学习环境

本机 GPU 为 GTX 1650 Ti，计算能力 7.5，对应 `sm_75`。
Nix flake 固定 CUDA 12.9 和兼容的 GCC，驱动由 NixOS 管理。
direnv 和 nix-direnv 已通过 Home Manager 配置，打开新终端后：

```bash
cd ~/cuda-learning
direnv allow
```

进入目录会自动加载开发环境，离开目录会恢复原环境。
首次加载需要等待，nix-direnv 会缓存环境并保留依赖。
修改 `.envrc` 后需再次执行 `direnv allow`。
也可手动运行 `nix develop`，用 `exit` 退出。

## 编译和运行

```bash
nvidia-smi
nvcc --version

nvcc -arch=sm_75 vector_add.cu -o vector_add
./vector_add
compute-sanitizer --error-exitcode 1 ./vector_add
```

- `nvcc`：CUDA 编译器。
- `-arch=sm_75`：指定本机 GTX 1650 Ti GPU 架构。
- `vector_add.cu`：源代码文件。
- `-o vector_add`：指定生成的可执行文件名。
- `./vector_add`：运行生成的程序。

编译自己新建的文件时，替换文件名即可：

```bash
nvcc -arch=sm_75 my_kernel.cu -o my_kernel
./my_kernel
```

# 注意事项

- 无需 Conda，进入目录后等待 direnv 加载完成即可。
- 修改 `.cu` 文件后，需要重新编译，再运行。
- 多个文件各自包含 `main()` 时，分别编译运行。
- 先使用小数组练习，避免大量显存占用和长时间循环。
- 保留 `if (i < n)` 等边界检查，防止访问数组范围之外的内存。
- 申请显存和复制数据时，传入字节数，例如 `n * sizeof(float)`。
- 分清 CPU 和 GPU 内存，使用 `cudaMemcpy` 复制数据，用完显存后调用 `cudaFree`。
- 保留 CUDA 错误检查；入门阶段使用 `cudaDeviceSynchronize()` 等待 GPU 完成并检查执行错误。
- 不修改系统驱动、CUDA Toolkit 或 `/usr/local/cuda` 软链接，不重置 GPU。

## 包含的工具和库

NVCC、CUDA Runtime、CCCL（Thrust/CUB）、cuBLAS、cuFFT、cuRAND、cuSOLVER、
cuSPARSE、CUDA-GDB、Compute Sanitizer、CMake、Ninja、GDB。
驱动库从 `/run/opengl-driver/lib` 加载。
请将 `flake.lock` 纳入版本控制；`.direnv/` 为本地缓存，已忽略。
