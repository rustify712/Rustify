#include <stdio.h>
#include <stdlib.h>

// Function to sort even indices of the array
float* sort_even(float* l, int size, int* out_size) {
    // Allocate memory for the output array
    float* out = (float*)malloc(size * sizeof(float));
    if (out == NULL) {
        *out_size = 0;
        return NULL;
    }

    // Allocate memory for the even indices array
    int even_size = (size + 1) / 2;
    float* even = (float*)malloc(even_size * sizeof(float));
    if (even == NULL) {
        free(out);
        *out_size = 0;
        return NULL;
    }

    // Extract even indices
    for (int i = 0; i * 2 < size; i++) {
        even[i] = l[i * 2];
    }

    // Sort the even indices array
    for (int i = 0; i < even_size - 1; i++) {
        for (int j = i + 1; j < even_size; j++) {
            if (even[i] > even[j]) {
                float temp = even[i];
                even[i] = even[j];
                even[j] = temp;
            }
        }
    }

    // Construct the output array
    for (int i = 0; i < size; i++) {
        if (i % 2 == 0) {
            out[i] = even[i / 2];
        } else {
            out[i] = l[i];
        }
    }

    // Free the even array
    free(even);

    // Set the output size
    *out_size = size;

    return out;
}