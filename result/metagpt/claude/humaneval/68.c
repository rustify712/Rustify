#include <stdio.h>
#include <stdlib.h>

int* pluck(int* arr, int size, int* returnSize) {
    int* out = (int*)malloc(2 * sizeof(int));
    *returnSize = 0;

    for (int i = 0; i < size; i++) {
        if (arr[i] % 2 == 0 && (*returnSize == 0 || arr[i] < out[0])) {
            out[0] = arr[i];
            out[1] = i;
            *returnSize = 2;
        }
    }

    if (*returnSize == 0) {
        free(out);
        return NULL;
    }

    return out;
}