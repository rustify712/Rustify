#include <stdbool.h>
#include <string.h>

bool correct_bracketing(const char* brackets) {
    int level = 0;
    for (int i = 0; i < strlen(brackets); i++) {
        if (brackets[i] == '(') level += 1;
        if (brackets[i] == ')') level -= 1;
        if (level < 0) return false;
    }
    if (level != 0) return false;
    return true;
}