#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char** words_string(const char* s, int* word_count) {
    int len = strlen(s);
    char* str = (char*)malloc((len + 1) * sizeof(char));
    strcpy(str, s);
    strcat(str, " ");  // Add a space at the end to handle the last word

    int count = 0;
    for (int i = 0; i < len + 1; i++) {
        if (str[i] == ' ' || str[i] == ',') {
            count++;
        }
    }

    char** words = (char**)malloc(count * sizeof(char*));
    int word_index = 0;
    char* current = (char*)malloc((len + 1) * sizeof(char));
    int current_index = 0;

    for (int i = 0; i < len + 1; i++) {
        if (str[i] == ' ' || str[i] == ',') {
            if (current_index > 0) {
                current[current_index] = '\0';
                words[word_index] = (char*)malloc((current_index + 1) * sizeof(char));
                strcpy(words[word_index], current);
                word_index++;
                current_index = 0;
            }
        } else {
            current[current_index++] = str[i];
        }
    }

    *word_count = word_index;
    free(current);
    free(str);
    return words;
}