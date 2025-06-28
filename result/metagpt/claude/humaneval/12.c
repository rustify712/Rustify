#include <stdio.h>
#include <string.h>

char* longest(char** strings, int size) {
    if (size == 0) {
        return NULL;
    }

    char* out = strings[0];
    for (int i = 1; i < size; i++) {
        if (strlen(strings[i]) > strlen(out)) {
            out = strings[i];
        }
    }
    return out;
}