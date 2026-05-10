#include <stdio.h>

#define TILEDIM 32
#define BLOCK_ROW 8

void trans_cpu(int* A, int* B, int n){

    for(int i = 0; i < n; i++){

        for(int j = 0; j < n; j++){

            B[j*n + i] = A[i*n + j];

        }
    }
}

__global__ void Navie_tran(int* A, int *B, int n){

    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if(row < n && col < n){

        B[col*n + row] = A[row*n + col];

    }
}

__global__ void Tile_trans(int* A, int *B, int n){

    int row = blockIdx.y * TILEDIM + threadIdx.y;
    int col = blockIdx.x * TILEDIM + threadIdx.x;

    // +1 avoids bank conflicts
    __shared__ int tile[TILEDIM][TILEDIM + 1];

    // Load from global memory to shared memory
    for(int j = 0; j < TILEDIM; j += BLOCK_ROW){

        if(col < n && (row + j) < n){

            tile[threadIdx.y + j][threadIdx.x] =
                A[(row + j) * n + col];
        }
    }

    __syncthreads();

    // Transposed coordinates
    row = blockIdx.x * TILEDIM + threadIdx.y;
    col = blockIdx.y * TILEDIM + threadIdx.x;

    // Store transposed data
    for(int j = 0; j < TILEDIM; j += BLOCK_ROW){

        if(col < n && (row + j) < n){

            B[(row + j) * n + col] =
                tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

int main(){

    int n = 4096;

    int size = n * n * sizeof(int);

    int *A, *B, *h_B;

    int *d_A, *d_B;
    int *d_A1, *d_B1;

    A = (int *)malloc(size);
    B = (int *)malloc(size);
    h_B = (int *)malloc(size);

    for(int i = 0; i < n * n; i++){

        A[i] = i;
    }

    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);

    cudaMalloc((void **)&d_A1, size);
    cudaMalloc((void **)&d_B1, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A1, A, size, cudaMemcpyHostToDevice);

    dim3 threadPerBlock(TILEDIM, BLOCK_ROW);

    dim3 blockPerGrid(
        (n + TILEDIM - 1) / TILEDIM,
        (n + TILEDIM - 1) / TILEDIM
    );

    // CUDA events
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float tile_time = 0;
    float navie_time = 0;

    // ---------------- TILE TRANSPOSE ----------------

    cudaEventRecord(start);

    Tile_trans<<<blockPerGrid, threadPerBlock>>>(d_A, d_B, n);

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&tile_time, start, stop);

    cudaError_t err = cudaGetLastError();

    if(err != cudaSuccess){

        printf("Tile kernel launch error: %s\n",
               cudaGetErrorString(err));
    }

    err = cudaDeviceSynchronize();

    if(err != cudaSuccess){

        printf("Tile runtime error: %s\n",
               cudaGetErrorString(err));
    }

    float eff_bandwidth1 =
        (2.0 * n * n * sizeof(int))
        /
        (tile_time / 1000.0)
        /
        1e9;

    printf("Bandwidth = %f GB/s\n", eff_bandwidth1);
    // ---------------- NAVIE TRANSPOSE ----------------
    dim3 threadPerBlock1(32, 8);
    dim3 blockPerGrid1(
        (n + threadPerBlock1.x - 1) / threadPerBlock1.x,
        (n + threadPerBlock1.y - 1) / threadPerBlock1.y
    );
    cudaEventRecord(start);

    Navie_tran<<<blockPerGrid1, threadPerBlock1>>>(d_A1, d_B1, n);

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&navie_time, start, stop);

    err = cudaGetLastError();

    if(err != cudaSuccess){

        printf("Navie kernel launch error: %s\n",
               cudaGetErrorString(err));
    }

    err = cudaDeviceSynchronize();

    if(err != cudaSuccess){

        printf("Navie runtime error: %s\n",
               cudaGetErrorString(err));
    }
    float eff_bandwidth2 =
        (2.0 * n * n * sizeof(int))
        /
        (navie_time / 1000.0)
        /
        1e9;

    printf("Bandwidth = %f GB/s\n", eff_bandwidth2);

    // Copy result back
    cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);

    // CPU transpose
    trans_cpu(A, B, n);

    // Validation
    for(int i = 0; i < n * n; i++){

        if(h_B[i] != B[i]){

            printf("Some error occured\n");

            return 0;
        }
    }

    printf("Both code work fined\n");

    printf("Tile transpose time  = %f ms\n", tile_time);

    printf("Navie transpose time = %f ms\n", navie_time);

    cudaFree(d_A);
    cudaFree(d_B);

    cudaFree(d_A1);
    cudaFree(d_B1);

    free(A);
    free(B);
    free(h_B);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
