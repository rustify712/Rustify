#include <stdio.h>
#include <stdlib.h>

// Function to compare two integers (used by qsort)
int compare(const void *a, const void *b) {
    return (*(int *)a - *(int *)b);
}

// Function to return sorted unique elements in an array
int* unique(int *arr, int size, int *result_size) {
    if (size == 0) {
        *result_size = 0;
        return NULL;
    }

    // Sort the array
    qsort(arr, size, sizeof(int), compare);

    // Allocate memory for the result array
    int *result = (int *)malloc(size * sizeof(int));
    if (result == NULL) {
        *result_size = 0;
        return NULL;
    }

    // Copy the first element
    result[0] = arr[0];
    int j = 1;

    // Remove duplicates
    for (int i = 1; i < size; i++) {
        if (arr[i] != arr[i - 1]) {
            result[j++] = arr[i];
        }
    }

    // Resize the result array to the actual number of unique elements
    result = (int *)realloc(result, j * sizeof(int));
    *result_size = j;

    return result;
}