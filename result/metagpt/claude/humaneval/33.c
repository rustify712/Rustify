#include <stdio.h>
#include <stdlib.h>

int* sort_third(int* l, int size) {
    int* third = (int*)malloc(size * sizeof(int));
    int third_size = 0;
    
    // Extract elements at indices divisible by 3
    for (int i = 0; i * 3 < size; i++) {
        third[third_size++] = l[i * 3];
    }
    
    // Sort the extracted elements
    for (int i = 0; i < third_size - 1; i++) {
        for (int j = i + 1; j < third_size; j++) {
            if (third[i] > third[j]) {
                int temp = third[i];
                third[i] = third[j];
                third[j] = temp;
            }
        }
    }
    
    // Create the output array
    int* out = (int*)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        if (i % 3 == 0) {
            out[i] = third[i / 3];
        } else {
            out[i] = l[i];
        }
    }
    
    free(third);
    return out;
}