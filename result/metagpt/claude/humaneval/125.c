#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Function to split words based on whitespace or commas
char** split_words(const char* txt, int* out_size) {
    int i;
    char* current = (char*)malloc(strlen(txt) + 1);
    strcpy(current, "");
    char** out = (char**)malloc(0);
    *out_size = 0;

    // Check if the text contains whitespace
    if (strchr(txt, ' ') != NULL) {
        char* temp = (char*)malloc(strlen(txt) + 2);
        strcpy(temp, txt);
        strcat(temp, " ");
        for (i = 0; i < strlen(temp); i++) {
            if (temp[i] == ' ') {
                if (strlen(current) > 0) {
                    out = (char**)realloc(out, (*out_size + 1) * sizeof(char*));
                    out[*out_size] = (char*)malloc(strlen(current) + 1);
                    strcpy(out[*out_size], current);
                    (*out_size)++;
                }
                strcpy(current, "");
            } else {
                strncat(current, &temp[i], 1);
            }
        }
        free(temp);
        free(current);
        return out;
    }

    // Check if the text contains commas
    if (strchr(txt, ',') != NULL) {
        char* temp = (char*)malloc(strlen(txt) + 2);
        strcpy(temp, txt);
        strcat(temp, ",");
        for (i = 0; i < strlen(temp); i++) {
            if (temp[i] == ',') {
                if (strlen(current) > 0) {
                    out = (char**)realloc(out, (*out_size + 1) * sizeof(char*));
                    out[*out_size] = (char*)malloc(strlen(current) + 1);
                    strcpy(out[*out_size], current);
                    (*out_size)++;
                }
                strcpy(current, "");
            } else {
                strncat(current, &temp[i], 1);
            }
        }
        free(temp);
        free(current);
        return out;
    }

    // If no whitespace or commas, count lowercase letters with odd order
    int num = 0;
    for (i = 0; i < strlen(txt); i++) {
        if (txt[i] >= 'a' && txt[i] <= 'z' && (txt[i] - 'a') % 2 == 0) {
            num++;
        }
    }

    // Return the count as a single element in the output array
    out = (char**)realloc(out, (*out_size + 1) * sizeof(char*));
    out[*out_size] = (char*)malloc(12); // Enough space for a 32-bit integer
    sprintf(out[*out_size], "%d", num);
    (*out_size)++;

    free(current);
    return out;
}