#include <stdbool.h>

bool move_one_ball(int arr[], int size) {
    int num = 0;
    if (size == 0) return true;
    for (int i = 1; i < size; i++) {
        if (arr[i] < arr[i - 1]) num++;
    }
    if (arr[size - 1] > arr[0]) num++;
    if (num < 2) return true;
    return false;
}