#include <stdbool.h>
#include <string.h>

bool is_happy(const char* s) {
    int length = strlen(s);
    if (length < 3) return false;
    for (int i = 2; i < length; i++) {
        if (s[i] == s[i-1] || s[i] == s[i-2]) return false;
    }
    return true;
}