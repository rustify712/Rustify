#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char* rounded_avg(int n, int m) {
    if (n > m) {
        char* result = (char*)malloc(3 * sizeof(char));
        strcpy(result, "-1");
        return result;
    }
    
    int num = (m + n) / 2;
    char* out = (char*)malloc(32 * sizeof(char)); // Assuming 32-bit integer
    out[0] = '\0'; // Initialize the string
    
    if (num == 0) {
        strcpy(out, "0");
        return out;
    }
    
    while (num > 0) {
        char temp[2];
        sprintf(temp, "%d", num % 2);
        strcat(temp, out);
        strcpy(out, temp);
        num = num / 2;
    }
    
    return out;
}