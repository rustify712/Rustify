#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// Function to find the word with the maximum number of unique characters
char* find_max(char** words, int num_words) {
    char* max_word = "";
    int max_unique = 0;

    for (int i = 0; i < num_words; i++) {
        char unique[256] = {0}; // Assuming ASCII characters
        int unique_count = 0;

        for (int j = 0; j < strlen(words[i]); j++) {
            if (unique[(unsigned char)words[i][j]] == 0) {
                unique[(unsigned char)words[i][j]] = 1;
                unique_count++;
            }
        }

        if (unique_count > max_unique || 
            (unique_count == max_unique && strcmp(words[i], max_word) < 0)) {
            max_word = words[i];
            max_unique = unique_count;
        }
    }

    return max_word;
}