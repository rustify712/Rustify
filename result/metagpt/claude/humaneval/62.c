#include <stdlib.h>

float* derivative(float* xs, int size, int* out_size) {
    *out_size = size - 1;
    float* out = (float*)malloc((*out_size) * sizeof(float));
    for (int i = 1; i < size; i++) {
        out[i - 1] = i * xs[i];
    }
    return out;
}