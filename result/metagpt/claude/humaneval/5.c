#include <stdio.h>
#include <stdlib.h>

int* intersperse(int* numbers, int size, int delimeter, int* out_size) {
    *out_size = size == 0 ? 0 : 2 * size - 1;
    int* out = (int*)malloc(*out_size * sizeof(int));
    
    if (size > 0) {
        out[0] = numbers[0];
        for (int i = 1; i < size; i++) {
            out[2 * i - 1] = delimeter;
            out[2 * i] = numbers[i];
        }
    }
    
    return out;
}