# CUDA Matrix Multiplication — CPU vs GPU Benchmark

A head-to-head benchmark of matrix multiplication implemented in plain C (CPU) and CUDA (GPU). Results are averaged over 20 timed runs, preceded by 3 warm-up runs, and the achieved speedup is printed to stdout.

---

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [File Structure](#file-structure)
- [Build](#build)
- [Configuration](#configuration)
- [Implementation Details](#implementation-details)
- [Example Output](#example-output)
- [Known Limitations](#known-limitations)
- [License](#license)

---

## Overview

Two implementations are benchmarked for multiplying matrices **A** (M×K) and **B** (K×N) to produce **C** (M×N):

| Implementation | Description |
|---|---|
| `matmul_cpu` | Triple-nested loop in plain C |
| `matmul_gpu` | CUDA kernel — each thread computes one output element in parallel |

---

## Requirements

**Hardware**
- NVIDIA GPU with CUDA Compute Capability ≥ 3.0
- ≥ 16 MB VRAM (three 256×256 float matrices ≈ 768 KB)

**Software**
- CUDA Toolkit ≥ 10.0 (`nvcc`, `cuda_runtime.h`)
- GCC / Clang ≥ 7
- Linux or WSL2 (`clock_gettime(CLOCK_MONOTONIC)` required)
- GNU Make *(optional)*

---

## File Structure

```
cuda-matmul/
├── matmul_cuda.cu     # CPU kernel, GPU kernel, benchmark harness
├── Makefile           # Optional build helper
├── README.md
└── .gitignore
```

**Recommended `.gitignore`:**
```
matmul
matmul.exe
*.o
*.ptx
*.cubin
*.fatbin
```

---

## Build

### Direct `nvcc`

```bash
nvcc -O2 -arch=sm_61 -o matmul matmul_cuda.cu
```

> Replace `sm_61` with your GPU's compute capability — e.g. `sm_75` (Turing), `sm_86` (Ampere).

### Using the Makefile

```makefile
NVCC    = nvcc
CFLAGS  = -O2 -arch=sm_61
TARGET  = matmul
SRC     = matmul_cuda.cu

all: $(TARGET)
$(TARGET): $(SRC)
	$(NVCC) $(CFLAGS) -o $@ $<
clean:
	rm -f $(TARGET)
```

```bash
make        # build
make clean  # remove binary
```

---

## Configuration

All tuneable constants are `#define` macros at the top of `matmul_cuda.cu`:

| Macro | Default | Description |
|---|---|---|
| `M` | 256 | Rows in A and C |
| `N` | 256 | Columns in B and C |
| `K` | 256 | Inner dimension (cols of A = rows of B) |
| `BLOCK_SIZE` | 32 | CUDA thread-block side length (32×32 = 1024 threads/block) |

Override at compile time without editing the source:

```bash
nvcc -O2 -arch=sm_61 -DM=1024 -DN=1024 -DK=1024 -o matmul matmul_cuda.cu
```

---

## Implementation Details

### CPU (`matmul_cpu`)

Triple-nested loop over rows `i`, columns `j`, and inner dimension `l`. Matrix A is accessed row-sequentially (cache-friendly); B is accessed column-strided (cache-unfriendly for large K).

### GPU (`matmul_gpu`)

Each CUDA thread computes a single element `C[row][col]`. The launch configuration is:

```
blockDim = (BLOCK_SIZE, BLOCK_SIZE)
gridDim  = (⌈N/BLOCK_SIZE⌉, ⌈M/BLOCK_SIZE⌉)
```

A boundary guard (`row < m && col < n`) handles matrix sizes that are not multiples of `BLOCK_SIZE`. This version reads directly from global memory — no shared-memory tiling.

### Timing

`clock_gettime(CLOCK_MONOTONIC)` provides nanosecond resolution. GPU timing wraps `cudaDeviceSynchronize()` to ensure kernel completion before the clock stops.

---

## Example Output

On an NVIDIA GTX 1050 Ti (`sm_61`) with 256×256 matrices:

```
Performing warm-up runs...
Benchmarking CPU implementation...
Benchmarking GPU implementation...
CPU average time: 8312.45 microseconds
GPU average time:  215.83 microseconds
Speedup: 38.51x
```

*Actual values depend on hardware, driver version, and system load.*

---

## Known Limitations

| Limitation | Suggested Fix |
|---|---|
| No correctness check — CPU and GPU outputs are never compared | Add element-wise max-abs-diff assertion after benchmarking |
| No shared-memory tiling | Tile A and B into `__shared__` blocks of size `BLOCK_SIZE×BLOCK_SIZE` |
| No CUDA error checking | Wrap every CUDA call with a `checkCuda()` macro |
| No cuBLAS baseline | Add `cublasSgemm()` as a third benchmark column |
| `get_time()` has mangled field access (`ts.tv_sec` / `ts.tv_nsec` rendered as markdown links in some editors) | Ensure source reads `ts.tv_sec` and `ts.tv_nsec` as plain identifiers before compiling |

---

## License

No license is currently specified. To publish on GitHub, add a `LICENSE` file — GitHub's license picker is available at repository creation. Recommended choices:

- **MIT** — permissive, minimal restrictions
- **Apache 2.0** — permissive with explicit patent grant
- **GPL-3.0** — copyleft; derivatives must also be open source
