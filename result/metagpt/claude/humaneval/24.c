#include <stdio.h>

int largest_divisor(int n) {
    for (int i = 2; i * i <= n; i++) {
        if (n % i == 0) {
            return n / i;
        }
    }
    return 1;
}