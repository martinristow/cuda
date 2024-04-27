#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "kernel.h"

__global__ void bubbleSort(int* arr, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = 0; i < n - 1; i++) {
        if (tid < n - i - 1) {
            if (arr[tid] > arr[tid + 1]) {
                // Замена на соседни елементи
                int temp = arr[tid];
                arr[tid] = arr[tid + 1];
                arr[tid + 1] = temp;
            }
        }
       // __syncthreads();  // Синхронизација на нитките пред да продолжат со следната итерација
    }
}

