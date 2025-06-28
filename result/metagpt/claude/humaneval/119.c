#include <stdio.h>
#include <string.h>

char* match_parens(char* lst[2]) {
    char l1[1000]; // Assuming a maximum length for the concatenated string
    int i, count = 0;
    int can = 1;

    // Concatenate the two strings
    strcpy(l1, lst[0]);
    strcat(l1, lst[1]);

    // Check if the concatenated string is balanced
    for (i = 0; i < strlen(l1); i++) {
        if (l1[i] == '(') count += 1;
        if (l1[i] == ')') count -= 1;
        if (count < 0) can = 0;
    }

    if (count != 0) return "No";
    if (can == 1) return "Yes";

    // Try the other order
    strcpy(l1, lst[1]);
    strcat(l1, lst[0]);

    count = 0;
    can = 1;

    for (i = 0; i < strlen(l1); i++) {
        if (l1[i] == '(') count += 1;
        if (l1[i] == ')') count -= 1;
        if (count < 0) can = 0;
    }

    if (can == 1) return "Yes";
    return "No";
}