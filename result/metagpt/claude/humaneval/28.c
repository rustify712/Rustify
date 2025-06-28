#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char* concatenate(char** strings, int count) {
    // Calculate the total length of the concatenated string
    int total_length = 0;
    for (int i = 0; i < count; i++) {
        total_length += strlen(strings[i]);
    }

    // Allocate memory for the concatenated string
    char* out = (char*)malloc(total_length + 1);
    if (out == NULL) {
        return NULL; // Handle memory allocation failure
    }

    // Concatenate the strings
    out[0] = '\0'; // Initialize the output string
    for (int i = 0; i < count; i++) {
        strcat(out, strings[i]);
    }

    return out;
}