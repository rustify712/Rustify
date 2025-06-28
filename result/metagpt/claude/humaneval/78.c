#include <stdio.h>
#include <string.h>

int hex_key(const char* num) {
    const char* key = "2357BD";
    int out = 0;
    for (int i = 0; i < strlen(num); i++) {
        if (strchr(key, num[i]) != NULL) {
            out++;
        }
    }
    return out;
}