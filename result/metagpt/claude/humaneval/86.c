#include <stdio.h>
#include <string.h>
#include <stdlib.h>

void sort_string(char *str) {
    int n = strlen(str);
    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            if (str[i] > str[j]) {
                char temp = str[i];
                str[i] = str[j];
                str[j] = temp;
            }
        }
    }
}

char* anti_shuffle(const char* s) {
    int len = strlen(s);
    char* out = (char*)malloc((len + 1) * sizeof(char));
    char* current = (char*)malloc((len + 1) * sizeof(char));
    int out_index = 0;
    int current_index = 0;

    for (int i = 0; i <= len; i++) {
        if (s[i] == ' ' || s[i] == '\0') {
            current[current_index] = '\0';
            sort_string(current);
            if (out_index > 0) {
                out[out_index++] = ' ';
            }
            strcpy(&out[out_index], current);
            out_index += strlen(current);
            current_index = 0;
        } else {
            current[current_index++] = s[i];
        }
    }

    out[out_index] = '\0';
    free(current);
    return out;
}