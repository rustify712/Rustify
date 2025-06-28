#include <stdio.h>
#include <string.h>
#include <stdbool.h>

char* words_in_sentence(const char* sentence) {
    char* out = (char*)malloc(100 * sizeof(char)); // Allocate memory for the output string
    out[0] = '\0'; // Initialize the output string as empty
    char current[100]; // Temporary buffer to hold the current word
    int out_index = 0; // Index for the output string

    int sentence_len = strlen(sentence);
    int current_index = 0;

    for (int i = 0; i <= sentence_len; i++) {
        if (sentence[i] != ' ' && sentence[i] != '\0') {
            current[current_index++] = sentence[i];
        } else {
            current[current_index] = '\0'; // Null-terminate the current word

            bool isp = true;
            int l = current_index;
            if (l < 2) isp = false;
            for (int j = 2; j * j <= l; j++) {
                if (l % j == 0) {
                    isp = false;
                    break;
                }
            }
            if (isp) {
                strcat(out, current);
                strcat(out, " ");
            }
            current_index = 0; // Reset the current word buffer
        }
    }

    // Remove the trailing space if any
    if (strlen(out) > 0) {
        out[strlen(out) - 1] = '\0';
    }

    return out;
}