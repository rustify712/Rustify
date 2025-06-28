#include <stdio.h>
#include <stdlib.h>

float* rescale_to_unit(float* numbers, int size) {
    float min = 100000, max = -100000;
    
    // Find the minimum and maximum values in the array
    for (int i = 0; i < size; i++) {
        if (numbers[i] < min) min = numbers[i];
        if (numbers[i] > max) max = numbers[i];
    }
    
    // Rescale the numbers to the range [0, 1]
    for (int i = 0; i < size; i++) {
        numbers[i] = (numbers[i] - min) / (max - min);
    }
    
    return numbers;
}