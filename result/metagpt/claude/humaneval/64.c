#include <stdio.h>
#include <string.h>

int vowels_count(const char *s) {
    const char *vowels = "aeiouAEIOU";
    int count = 0;
    int len = strlen(s);

    for (int i = 0; i < len; i++) {
        if (strchr(vowels, s[i]) != NULL) {
            count++;
        }
    }

    if (len > 0 && (s[len - 1] == 'y' || s[len - 1] == 'Y')) {
        count++;
    }

    return count;
}