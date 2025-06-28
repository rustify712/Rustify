#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define MAX_CHAR 256

typedef struct {
    char key;
    int value;
} CharCount;

CharCount* histogram(const char* test) {
    int count[MAX_CHAR] = {0};
    int max = 0;
    int len = strlen(test);

    // Count the frequency of each character
    for (int i = 0; i < len; i++) {
        if (test[i] != ' ') {
            count[(unsigned char)test[i]]++;
            if (count[(unsigned char)test[i]] > max) {
                max = count[(unsigned char)test[i]];
            }
        }
    }

    // Count how many characters have the maximum frequency
    int max_count = 0;
    for (int i = 0; i < MAX_CHAR; i++) {
        if (count[i] == max) {
            max_count++;
        }
    }

    // Allocate memory for the result
    CharCount* out = (CharCount*)malloc((max_count + 1) * sizeof(CharCount));
    if (out == NULL) {
        return NULL;
    }

    // Fill the result array
    int index = 0;
    for (int i = 0; i < MAX_CHAR; i++) {
        if (count[i] == max) {
            out[index].key = (char)i;
            out[index].value = count[i];
            index++;
        }
    }

    // Mark the end of the array
    out[index].key = '\0';
    out[index].value = 0;

    return out;
}