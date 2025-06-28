#include <stdio.h>

int can_arrange(int arr[], int size) {
    int max = -1;
    for (int i = 0; i < size; i++) {
        if (arr[i] <= i) {
            max = i;
        }
    }
    return max;
}