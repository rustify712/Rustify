#include <stdbool.h>

bool below_zero(int* operations, int size) {
    int num = 0;
    for (int i = 0; i < size; i++) {
        num += operations[i];
        if (num < 0) return true;
    }
    return false;
}