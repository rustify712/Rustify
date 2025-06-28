#include <stdbool.h>

bool will_it_fly(int q[], int size, int w) {
    int sum = 0;
    for (int i = 0; i < size; i++) {
        if (q[i] != q[size - 1 - i]) return false;
        sum += q[i];
    }
    if (sum > w) return false;
    return true;
}