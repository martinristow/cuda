#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define RANGE 10

__global__ void distribute(int* input, int* bucket, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        int value = input[tid];
        atomicAdd(&bucket[value], 1);
    }
}

__global__ void gather(int* input, int* bucket, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        int value = threadIdx.x;
        int count = bucket[value];

        for (int i = 0; i < count; ++i) {
            input[tid] = value;
            tid += blockDim.x * gridDim.x;
        }
    }
}

void distributionSort(int* d_input, int n) {
    int* d_bucket;
    cudaMalloc((void**)&d_bucket, RANGE * sizeof(int));
    cudaMemset(d_bucket, 0, RANGE * sizeof(int));

    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    distribute << <grid_size, block_size >> > (d_input, d_bucket, n);
    cudaDeviceSynchronize();

    gather << <grid_size, block_size >> > (d_input, d_bucket, n);
    cudaDeviceSynchronize();

    cudaFree(d_bucket);
}

int main() {
    const int n = 10;
    int arr[n] = { 5, 2, 9, 1, 5, 6, 3, 8, 7, 4 };
    int* d_arr;

    // Allocate device memory
    cudaMalloc((void**)&d_arr, n * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);

    // Perform distribution sort on GPU
    distributionSort(d_arr, n);

    // Copy data from device to host
    cudaMemcpy(arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_arr);

    // Print sorted array
    printf("Sorted Array: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");

    return 0;
}
