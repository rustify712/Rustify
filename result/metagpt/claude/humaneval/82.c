#include <stdbool.h>
#include <string.h>

bool prime_length(const char* str) {
    int l, i;
    l = strlen(str);
    if (l < 2) return false;
    for (i = 2; i * i <= l; i++) {
        if (l % i == 0) return false;
    }
    return true;
}