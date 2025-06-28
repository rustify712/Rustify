#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Function to return all prefixes from shortest to longest of the input string
char** all_prefixes(const char* str, int* out_size) {
    int len = strlen(str);
    char** out = (char**)malloc(len * sizeof(char*));
    char* current = (char*)malloc((len + 1) * sizeof(char));
    current[0] = '\0'; // Initialize current as an empty string

    for (int i = 0; i < len; i++) {
        current[i] = str[i];
        current[i + 1] = '\0';
        out[i] = (char*)malloc((i + 2) * sizeof(char));
        strcpy(out[i], current);
    }

    *out_size = len;
    free(current);
    return out;
}