#include <stdio.h>
#include <string.h>
#include <ctype.h>

int count_distinct_characters(const char* str) {
    char distinct[256] = {0};  // Assuming ASCII characters
    int count = 0;
    int len = strlen(str);

    for (int i = 0; i < len; i++) {
        char lower_char = tolower(str[i]);
        if (!distinct[(unsigned char)lower_char]) {
            distinct[(unsigned char)lower_char] = 1;
            count++;
        }
    }

    return count;
}