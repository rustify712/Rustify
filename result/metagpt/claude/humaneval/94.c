#include <stdio.h>
#include <stdbool.h>
#include <string.h>

int skjkasdkd(int* lst, int size) {
    int largest = 0;
    for (int i = 0; i < size; i++) {
        if (lst[i] > largest) {
            bool prime = true;
            for (int j = 2; j * j <= lst[i]; j++) {
                if (lst[i] % j == 0) {
                    prime = false;
                    break;
                }
            }
            if (prime) {
                largest = lst[i];
            }
        }
    }
    int sum = 0;
    char s[20]; // Assuming the largest prime won't exceed 20 digits
    sprintf(s, "%d", largest);
    for (int i = 0; i < strlen(s); i++) {
        sum += s[i] - '0';
    }
    return sum;
}