#include <stdio.h>
#include <string.h>
#include <stdlib.h>

char* flip_case(const char* str) {
    int length = strlen(str);
    char* out = (char*)malloc(length + 1); // Allocate memory for the output string
    if (out == NULL) {
        return NULL; // Handle memory allocation failure
    }

    for (int i = 0; i < length; i++) {
        char w = str[i];
        if (w >= 'a' && w <= 'z') {
            w -= 32; // Convert lowercase to uppercase
        } else if (w >= 'A' && w <= 'Z') {
            w += 32; // Convert uppercase to lowercase
        }
        out[i] = w;
    }
    out[length] = '\0'; // Null-terminate the string
    return out;
}