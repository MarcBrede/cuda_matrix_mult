#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <ctime>
#include <thread>
#include <chrono>

#define N 1024

void multiplyMatricesCPU(float *a, float *b, float *c) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            c[i * N + j] = 0;
            for (int k = 0; k < N; k++) {
                c[i * N + j] += a[i * N + k] * b[k * N + j];
            }
        }
    }
}

__global__ void multiplyMatricesCUDA(float *a, float *b, float *c) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;

    if (row < N && col < N) {
        for (int k = 0; k < N; k++) {
            sum += a[row * N + k] * b[k * N + col];
        }
        c[row * N + col] = sum;
    }
}

#define TILE_WIDTH 16

__global__ void multiplyMatricesSharedMemory(float *a, float *b, float *c) {
    // declare two shared memory arrays
    __shared__ float A_tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ float B_tile[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;

    // load data into shared memory
    for (int k = 0; k < N / TILE_WIDTH; ++k) {
        A_tile[threadIdx.y][threadIdx.x] = a[row * N + k * TILE_WIDTH + threadIdx.x];
        B_tile[threadIdx.y][threadIdx.x] = b[(k * TILE_WIDTH + threadIdx.y) * N + col];

        __syncthreads();

        // each part is computed
        for (int m = 0; m < TILE_WIDTH; ++m) {
            sum += A_tile[threadIdx.y][m] * B_tile[m][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        c[row * N + col] = sum;
    }
}


int main() {
    float *a = (float*) malloc(N * N * sizeof(float));
    float *b = (float*) malloc(N * N * sizeof(float));
    float *c_cuda = (float*) malloc(N * N * sizeof(float));
    float *c_cpu = (float*) malloc(N * N * sizeof(float));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i * N + j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX); // Random floats between 0 and 1
            b[i * N + j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        }
    }

    // CPU
    clock_t startCPU = clock();
    multiplyMatricesCPU(a, b, c_cpu);
    clock_t endCPU = clock();
    double cpuTime = 1000.0 * (endCPU - startCPU) / CLOCKS_PER_SEC;
    printf("CPU Time: %f ms\n", cpuTime);

    // CUDA
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * N * sizeof(float));
    cudaMalloc(&d_b, N * N * sizeof(float));
    cudaMalloc(&d_c, N * N * sizeof(float));

    cudaMemcpy(d_a, a, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaEventRecord(start);
    multiplyMatricesCUDA<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c);
    cudaEventRecord(stop); 
    cudaEventSynchronize(stop); 

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("CUDA Time: %f ms\n", milliseconds); 

    cudaMemcpy(c_cuda, d_c, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // CUDA shared memory
    dim3 threadsPerBlock_sharedMemory(TILE_WIDTH, TILE_WIDTH); 
    dim3 numBlocks_sharedMemory((N + threadsPerBlock_sharedMemory.x - 1) / threadsPerBlock_sharedMemory.x, (N + threadsPerBlock_sharedMemory.y - 1) / threadsPerBlock_sharedMemory.y);
    cudaEventRecord(start);
    multiplyMatricesSharedMemory<<<numBlocks_sharedMemory, threadsPerBlock_sharedMemory>>>(d_a, d_b, d_c);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("CUDA Time with Shared Memory: %f ms\n", milliseconds);
    
    // Occupancy optimization yields no speed-up
    // int minGridSize;
    // int blockSize;   
    // cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, multiplyMatricesSharedMemory, 0, N * N);
    // int gridSize = (N * N + blockSize - 1) / blockSize;
    // cudaEventRecord(start);
    // multiplyMatricesSharedMemory<<<gridSize, blockSize>>>(d_a, d_b, d_c);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("CUDA Time with Shared Memory: %f ms\n", milliseconds);

    // Prefetching data to GPU
    int deviceID = 0;
    cudaMemPrefetchAsync(a, N * N * sizeof(float), deviceID);
    cudaMemPrefetchAsync(b, N * N * sizeof(float), deviceID);
    cudaEventRecord(start);
    multiplyMatricesSharedMemory<<<numBlocks_sharedMemory, threadsPerBlock_sharedMemory>>>(d_a, d_b, d_c);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("CUDA Time with Shared Memory and Prefetching to GPU: %f ms\n", milliseconds);

    // clean-up
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(a);
    free(b);
    free(c_cuda); 
    free(c_cpu);

    return 0;
}
