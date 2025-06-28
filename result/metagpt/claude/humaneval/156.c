#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char* int_to_mini_roman(int number) {
    char* current = (char*)malloc(100 * sizeof(char)); // Allocate memory for the result string
    current[0] = '\0'; // Initialize the string as empty

    const char* rep[] = {"m", "cm", "d", "cd", "c", "xc", "l", "xl", "x", "ix", "v", "iv", "i"};
    int num[] = {1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1};
    int pos = 0;

    while (number > 0) {
        while (number >= num[pos]) {
            strcat(current, rep[pos]); // Append the Roman numeral to the result string
            number -= num[pos];
        }
        if (number > 0) pos += 1;
    }

    return current;
}