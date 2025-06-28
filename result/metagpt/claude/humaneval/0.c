#include <stdio.h>
#include <math.h>
#include <stdbool.h>

bool has_close_elements(float* numbers, int size, float threshold) {
    int i, j;
    
    for (i = 0; i < size; i++) {
        for (j = i + 1; j < size; j++) {
            if (fabs(numbers[i] - numbers[j]) < threshold) {
                return true;
            }
        }
    }
    
    return false;
}