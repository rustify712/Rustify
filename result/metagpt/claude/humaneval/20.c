#include <stdio.h>
#include <math.h>
#include <stdlib.h>

void find_closest_elements(float *numbers, int size, float *out) {
    float min_diff = INFINITY;
    int min_i = 0, min_j = 1;

    for (int i = 0; i < size; i++) {
        for (int j = i + 1; j < size; j++) {
            float diff = fabs(numbers[i] - numbers[j]);
            if (diff < min_diff) {
                min_diff = diff;
                min_i = i;
                min_j = j;
            }
        }
    }

    if (numbers[min_i] > numbers[min_j]) {
        out[0] = numbers[min_j];
        out[1] = numbers[min_i];
    } else {
        out[0] = numbers[min_i];
        out[1] = numbers[min_j];
    }
}