#include <stdbool.h>

bool is_sorted(int lst[], int size) {
    for (int i = 1; i < size; i++) {
        if (lst[i] < lst[i - 1]) return false;
        if (i >= 2 && lst[i] == lst[i - 1] && lst[i] == lst[i - 2]) return false;
    }
    return true;
}