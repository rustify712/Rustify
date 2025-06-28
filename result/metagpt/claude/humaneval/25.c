#include <stdio.h>
#include <stdlib.h>

int* factorize(int n, int* returnSize) {
    int* out = (int*)malloc(sizeof(int) * n); // Allocate memory for the output array
    int count = 0; // Counter for the number of factors

    for (int i = 2; i * i <= n; i++) {
        while (n % i == 0) {
            out[count++] = i;
            n /= i;
        }
    }

    if (n > 1) {
        out[count++] = n;
    }

    *returnSize = count; // Set the size of the output array
    return out; // Return the array of factors
}