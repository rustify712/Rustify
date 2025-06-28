#include <stdio.h>

int smallest_change(int arr[], int size) {
    int out = 0;
    for (int i = 0; i < size - 1 - i; i++) {
        if (arr[i] != arr[size - 1 - i]) {
            out += 1;
        }
    }
    return out;
}