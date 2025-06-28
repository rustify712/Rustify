#include <stdio.h>
#include <string.h>

int same_chars(const char* s0, const char* s1) {
    int len0 = strlen(s0);
    int len1 = strlen(s1);

    for (int i = 0; i < len0; i++) {
        if (strchr(s1, s0[i]) == NULL) {
            return 0;
        }
    }

    for (int i = 0; i < len1; i++) {
        if (strchr(s0, s1[i]) == NULL) {
            return 0;
        }
    }

    return 1;
}