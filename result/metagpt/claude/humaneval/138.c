#include <stdbool.h>

bool is_equal_to_sum_even(int n) {
    if (n % 2 == 0 && n >= 8) return true;
    return false;
}