#include <stdio.h>
#include <stdlib.h>

// Function to compare two floats for qsort
int compare_floats(const void* a, const void* b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
}

// Function to calculate the median of an array of floats
float median(float* l, int size) {
    // Sort the array
    qsort(l, size, sizeof(float), compare_floats);
    
    // Calculate the median
    if (size % 2 == 1) {
        return l[size / 2];
    } else {
        return 0.5 * (l[size / 2] + l[size / 2 - 1]);
    }
}