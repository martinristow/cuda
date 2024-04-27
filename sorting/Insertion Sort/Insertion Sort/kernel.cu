#include <stdio.h>

__global__ void insertionSort(int* arr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int key = arr[i];
        int j = i - 1;

        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j = j - 1;
        }

        arr[j + 1] = key;
    }
}

int main() {
    const int n = 10;
    int arr[n] = { 5, 2, 9, 1, 5, 6, 3, 8, 7, 4 };
    int* d_arr;

    // Allocate device memory
    cudaMalloc((void**)&d_arr, n * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    // Launch kernel for insertion sort
    insertionSort << <grid_size, block_size >> > (d_arr, n);

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
