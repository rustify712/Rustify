#include <stdio.h>
#include <stdlib.h>

int* make_a_pile(int n, int* returnSize) {
    int* out = (int*)malloc(n * sizeof(int));
    out[0] = n;
    for (int i = 1; i < n; i++) {
        out[i] = out[i - 1] + 2;
    }
    *returnSize = n;
    return out;
}