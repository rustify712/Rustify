#include <stdio.h>
#include <string.h>
#include <stdbool.h>

bool is_palindrome(const char* text) {
    int length = strlen(text);
    char reversed[length + 1];
    
    for (int i = 0; i < length; i++) {
        reversed[i] = text[length - 1 - i];
    }
    reversed[length] = '\0';
    
    return strcmp(reversed, text) == 0;
}