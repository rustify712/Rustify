#include <stdio.h>
#include <stdlib.h>

int* unique_digits(int* x, int size, int* result_size) {
    int* out = (int*)malloc(size * sizeof(int));
    int out_index = 0;

    for (int i = 0; i < size; i++) {
        int num = x[i];
        int u = 1; // Assume the number has no even digits initially

        if (num == 0) {
            u = 0;
        }

        while (num > 0 && u) {
            if (num % 2 == 0) {
                u = 0;
            }
            num = num / 10;
        }

        if (u) {
            out[out_index++] = x[i];
        }
    }

    // Sort the output array
    for (int i = 0; i < out_index - 1; i++) {
        for (int j = i + 1; j < out_index; j++) {
            if (out[i] > out[j]) {
                int temp = out[i];
                out[i] = out[j];
                out[j] = temp;
            }
        }
    }

    *result_size = out_index;
    return out;
}