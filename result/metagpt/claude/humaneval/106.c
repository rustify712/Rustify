#include <stdio.h>
#include <stdlib.h>

int* f(int n) {
    int sum = 0, prod = 1;
    int* out = (int*)malloc(n * sizeof(int));
    
    for (int i = 1; i <= n; i++) {
        sum += i;
        prod *= i;
        if (i % 2 == 0) {
            out[i - 1] = prod;
        } else {
            out[i - 1] = sum;
        }
    }
    
    return out;
}