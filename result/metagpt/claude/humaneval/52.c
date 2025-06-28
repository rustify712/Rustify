#include <stdbool.h>

bool below_threshold(int* l, int size, int t) {
    for (int i = 0; i < size; i++) {
        if (l[i] >= t) {
            return false;
        }
    }
    return true;
}