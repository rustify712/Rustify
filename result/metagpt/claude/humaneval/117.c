#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Function to check if a character is a consonant
int is_consonant(char c) {
    char vowels[] = "aeiouAEIOU";
    for (int i = 0; i < strlen(vowels); i++) {
        if (c == vowels[i]) {
            return 0;
        }
    }
    return 1;
}

// Function to select words with exactly n consonants
char** select_words(const char* s, int n, int* result_size) {
    int capacity = 10;
    char** out = (char**)malloc(capacity * sizeof(char*));
    *result_size = 0;

    char current[100]; // Assuming max word length is 100
    int current_len = 0;
    int numc = 0;

    for (int i = 0; i <= strlen(s); i++) {
        if (s[i] == ' ' || s[i] == '\0') {
            if (numc == n) {
                current[current_len] = '\0';
                if (*result_size >= capacity) {
                    capacity *= 2;
                    out = (char**)realloc(out, capacity * sizeof(char*));
                }
                out[*result_size] = (char*)malloc((current_len + 1) * sizeof(char));
                strcpy(out[*result_size], current);
                (*result_size)++;
            }
            current_len = 0;
            numc = 0;
        } else {
            current[current_len++] = s[i];
            if ((s[i] >= 'A' && s[i] <= 'Z') || (s[i] >= 'a' && s[i] <= 'z')) {
                if (is_consonant(s[i])) {
                    numc++;
                }
            }
        }
    }

    return out;
}