#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int* sort_array(int* arr, int size) {
    int* bin = (int*)malloc(size * sizeof(int));
    int m;

    for (int i = 0; i < size; i++) {
        int b = 0, n = abs(arr[i]);
        while (n > 0) {
            b += n % 2;
            n = n / 2;
        }
        bin[i] = b;
    }

    for (int i = 0; i < size; i++) {
        for (int j = 1; j < size; j++) {
            if (bin[j] < bin[j - 1] || (bin[j] == bin[j - 1] && arr[j] < arr[j - 1])) {
                m = arr[j];
                arr[j] = arr[j - 1];
                arr[j - 1] = m;

                m = bin[j];
                bin[j] = bin[j - 1];
                bin[j - 1] = m;
            }
        }
    }

    free(bin);
    return arr;
}