#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int prod_signs(int* arr, int size) {
    if (size == 0) return -32768;
    int i, sum = 0, prods = 1;
    for (i = 0; i < size; i++) {
        sum += abs(arr[i]);
        if (arr[i] == 0) prods = 0;
        if (arr[i] < 0) prods = -prods;
    }
    return sum * prods;
}