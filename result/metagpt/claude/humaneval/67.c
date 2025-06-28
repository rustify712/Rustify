#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int fruit_distribution(const char *s, int n) {
    char num1[32] = {0};
    char num2[32] = {0};
    int is12 = 0;
    int j = 0;

    for (int i = 0; i < strlen(s); i++) {
        if (s[i] >= '0' && s[i] <= '9') {
            if (is12 == 0) {
                num1[j++] = s[i];
            } else {
                num2[j++] = s[i];
            }
        } else {
            if (is12 == 0 && j > 0) {
                is12 = 1;
                j = 0;
            }
        }
    }

    return n - atoi(num1) - atoi(num2);
}