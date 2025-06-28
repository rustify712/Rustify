#include <stdio.h>
#include <string.h>

int digitSum(const char* s) {
    int sum = 0;
    for (int i = 0; i < strlen(s); i++) {
        if (s[i] >= 65 && s[i] <= 90) {
            sum += s[i];
        }
    }
    return sum;
}