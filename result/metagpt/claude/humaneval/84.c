#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char* solve(int N) {
    char str[6]; // Since N can be up to 10000, the maximum number of digits is 5
    sprintf(str, "%d", N); // Convert integer to string
    int sum = 0;
    for (int i = 0; i < strlen(str); i++) {
        sum += str[i] - '0'; // Sum the digits
    }

    // Calculate the binary representation
    char* bi = (char*)malloc(20 * sizeof(char)); // Allocate enough space for the binary string
    bi[0] = '\0'; // Initialize the string
    if (sum == 0) {
        strcat(bi, "0");
    } else {
        while (sum > 0) {
            char temp[2];
            sprintf(temp, "%d", sum % 2);
            strcat(temp, bi);
            strcpy(bi, temp);
            sum /= 2;
        }
    }

    return bi;
}