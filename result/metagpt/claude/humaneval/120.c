#include <stdio.h>
#include <stdlib.h>

// Function to compare two integers (used for qsort)
int compare(const void *a, const void *b) {
    return (*(int*)a - *(int*)b);
}

// Function to find the maximum k numbers in the array and return them in a sorted array
int* maximum(int* arr, int arrSize, int k, int* returnSize) {
    // Sort the array in ascending order
    qsort(arr, arrSize, sizeof(int), compare);
    
    // Allocate memory for the output array
    int* out = (int*)malloc(k * sizeof(int));
    
    // Copy the last k elements from the sorted array to the output array
    for (int i = 0; i < k; i++) {
        out[i] = arr[arrSize - k + i];
    }
    
    // Set the return size
    *returnSize = k;
    
    return out;
}