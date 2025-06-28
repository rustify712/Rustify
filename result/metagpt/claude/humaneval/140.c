#include <stdio.h>
#include <string.h>
#include <stdlib.h>

char* fix_spaces(const char* text) {
    int len = strlen(text);
    char* out = (char*)malloc(2 * len * sizeof(char)); // Allocate enough space for the output
    if (out == NULL) {
        return NULL; // Handle memory allocation failure
    }
    
    int out_index = 0;
    int spacelen = 0;
    
    for (int i = 0; i < len; i++) {
        if (text[i] == ' ') {
            spacelen++;
        } else {
            if (spacelen == 1) {
                out[out_index++] = '_';
            } else if (spacelen == 2) {
                out[out_index++] = '_';
                out[out_index++] = '_';
            } else if (spacelen > 2) {
                out[out_index++] = '-';
            }
            spacelen = 0;
            out[out_index++] = text[i];
        }
    }
    
    // Handle trailing spaces
    if (spacelen == 1) {
        out[out_index++] = '_';
    } else if (spacelen == 2) {
        out[out_index++] = '_';
        out[out_index++] = '_';
    } else if (spacelen > 2) {
        out[out_index++] = '-';
    }
    
    out[out_index] = '\0'; // Null-terminate the string
    return out;
}