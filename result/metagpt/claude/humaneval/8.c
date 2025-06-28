#include <stdio.h>

void sum_product(int numbers[], int size, int *sum, int *product) {
    *sum = 0;
    *product = 1;
    for (int i = 0; i < size; i++) {
        *sum += numbers[i];
        *product *= numbers[i];
    }
}