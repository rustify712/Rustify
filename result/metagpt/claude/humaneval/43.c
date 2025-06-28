#include <stdbool.h>

bool pairs_sum_to_zero(int* l, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = i + 1; j < size; j++) {
            if (l[i] + l[j] == 0) {
                return true;
            }
        }
    }
    return false;
}