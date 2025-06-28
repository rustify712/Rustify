#include <stdio.h>

int add_elements(int arr[], int k, int len) {
    int sum = 0;
    for (int i = 0; i < k && i < len; i++) {
        if (arr[i] >= -99 && arr[i] <= 99) {
            sum += arr[i];
        }
    }
    return sum;
}