#include <stdio.h>
#include <string.h>
#include <stdlib.h>

char* encode_shift(const char* s) {
    // returns encoded string by shifting every character by 5 in the alphabet.
    int length = strlen(s);
    char* out = (char*)malloc(length + 1);
    if (out == NULL) {
        return NULL; // Handle memory allocation failure
    }
    for (int i = 0; i < length; i++) {
        int w = ((int)s[i] + 5 - (int)'a') % 26 + (int)'a';
        out[i] = (char)w;
    }
    out[length] = '\0';
    return out;
}

char* decode_shift(const char* s) {
    // takes as input string encoded with encode_shift function. Returns decoded string.
    int length = strlen(s);
    char* out = (char*)malloc(length + 1);
    if (out == NULL) {
        return NULL; // Handle memory allocation failure
    }
    for (int i = 0; i < length; i++) {
        int w = ((int)s[i] + 21 - (int)'a') % 26 + (int)'a';
        out[i] = (char)w;
    }
    out[length] = '\0';
    return out;
}