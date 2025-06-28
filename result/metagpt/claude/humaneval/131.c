#include <stdio.h>
#include <string.h>

int digits(int n) {
    int prod = 1, has = 0;
    char s[20]; // Assuming the maximum number of digits is 20
    sprintf(s, "%d", n); // Convert integer to string

    for (int i = 0; i < strlen(s); i++) {
        if ((s[i] - '0') % 2 == 1) {
            has = 1;
            prod = prod * (s[i] - '0');
        }
    }

    if (has == 0) return 0;
    return prod;
}