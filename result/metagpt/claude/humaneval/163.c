#include <stdio.h>
#include <stdlib.h>

int* generate_integers(int a, int b, int* returnSize) {
    int m;
    if (b < a) {
        m = a;
        a = b;
        b = m;
    }

    // Allocate memory for the output array
    int* out = (int*)malloc((b - a + 1) * sizeof(int));
    int count = 0;

    for (int i = a; i <= b; i++) {
        if (i < 10 && i % 2 == 0) {
            out[count++] = i;
        }
    }

    *returnSize = count;
    return out;
}