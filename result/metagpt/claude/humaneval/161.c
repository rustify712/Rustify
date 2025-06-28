#include <stdio.h>
#include <string.h>
#include <ctype.h>

char* solve(const char* s) {
    int nletter = 0;
    int len = strlen(s);
    char* out = (char*)malloc(len + 1); // Allocate memory for the output string
    if (out == NULL) {
        return NULL; // Handle memory allocation failure
    }

    for (int i = 0; i < len; i++) {
        char w = s[i];
        if (isupper(w)) {
            w = tolower(w);
        } else if (islower(w)) {
            w = toupper(w);
        } else {
            nletter++;
        }
        out[i] = w;
    }
    out[len] = '\0'; // Null-terminate the string

    if (nletter == len) {
        // Reverse the string
        for (int i = 0; i < len / 2; i++) {
            char temp = out[i];
            out[i] = out[len - i - 1];
            out[len - i - 1] = temp;
        }
    }

    return out;
}