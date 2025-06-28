#include <stdio.h>
#include <stdlib.h>

// Function to increment each element of the array by 1
int* incr_list(int* l, int size) {
    for (int i = 0; i < size; i++) {
        l[i] += 1;
    }
    return l;
}