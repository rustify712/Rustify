#include <stdio.h>
#include <stdlib.h>

int* rolling_max(int* numbers, int size, int* out_size) {
    int* out = (int*)malloc(size * sizeof(int));
    if (out == NULL) {
        *out_size = 0;
        return NULL;
    }

    int max = 0;
    for (int i = 0; i < size; i++) {
        if (numbers[i] > max) {
            max = numbers[i];
        }
        out[i] = max;
    }

    *out_size = size;
    return out;
}