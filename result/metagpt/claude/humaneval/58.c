#include <stdio.h>
#include <stdlib.h>

// Function to find if an element exists in an array
int find(int *arr, int size, int element) {
    for (int i = 0; i < size; i++) {
        if (arr[i] == element) {
            return 1;
        }
    }
    return 0;
}

// Function to sort an array using bubble sort
void sort(int *arr, int size) {
    for (int i = 0; i < size - 1; i++) {
        for (int j = 0; j < size - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

// Function to find common elements between two arrays
int* common(int *l1, int size1, int *l2, int size2, int *outSize) {
    int *out = (int *)malloc(size1 * sizeof(int));
    int count = 0;

    for (int i = 0; i < size1; i++) {
        if (!find(out, count, l1[i])) {
            if (find(l2, size2, l1[i])) {
                out[count++] = l1[i];
            }
        }
    }

    sort(out, count);
    *outSize = count;
    return out;
}