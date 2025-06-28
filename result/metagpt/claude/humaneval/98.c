#include <stdio.h>
#include <string.h>

int count_upper(const char* s) {
    const char* uvowel = "AEIOU";
    int count = 0;
    int length = strlen(s);
    
    for (int i = 0; i * 2 < length; i++) {
        if (strchr(uvowel, s[i * 2]) != NULL) {
            count++;
        }
    }
    
    return count;
}