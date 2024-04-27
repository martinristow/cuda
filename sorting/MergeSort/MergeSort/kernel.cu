#include <stdio.h>
#include <stdlib.h>

__device__ void swap(int* arr, int i, int j) {
    int temp = arr[i];
    arr[i] = arr[j];
    arr[j] = temp;
}

__device__ int partition(int* arr, int low, int high) {
    int pivot = arr[high];
    int i = low - 1;

    for (int j = low; j <= high - 1; j++) {
        if (arr[j] <= pivot) {
            i++;
            swap(arr, i, j);
        }
    }

    swap(arr, i + 1, high);
    return i + 1;
}

__global__ void quicksort(int* arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);

        // Рекурзивни повици за левата и десната страна од пивотот
        quicksort << <1, 1 >> > (arr, low, pi - 1);
        quicksort << <1, 1 >> > (arr, pi + 1, high);
    }
}

void quicksortCUDA(int* arr, int n) {
    int* d_arr;

    cudaMalloc((void**)&d_arr, n * sizeof(int));
    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);

    quicksort << <1, 1 >> > (d_arr, 0, n - 1);
    cudaDeviceSynchronize();

    cudaMemcpy(arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_arr);
}

int main() {
    const int n = 10;
    int arr[n] = { 5, 2, 9, 1, 5, 6, 3, 8, 7, 4 };

    quicksortCUDA(arr, n);

    // Печатење на сортираната листа
    printf("Sorted Array: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");

    return 0;
}
