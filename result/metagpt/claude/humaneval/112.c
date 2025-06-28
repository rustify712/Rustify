#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// Function to check if a character exists in a string
int char_in_string(char ch, const char *str) {
    for (int i = 0; str[i] != '\0'; i++) {
        if (str[i] == ch) {
            return 1;
        }
    }
    return 0;
}

// Function to reverse a string
void reverse_string(char *str) {
    int len = strlen(str);
    for (int i = 0; i < len / 2; i++) {
        char temp = str[i];
        str[i] = str[len - i - 1];
        str[len - i - 1] = temp;
    }
}

// Function to perform the reverse delete operation
char** reverse_delete(const char *s, const char *c) {
    char *n = (char *)malloc(strlen(s) + 1);
    int j = 0;
    for (int i = 0; s[i] != '\0'; i++) {
        if (!char_in_string(s[i], c)) {
            n[j++] = s[i];
        }
    }
    n[j] = '\0';

    char **result = (char **)malloc(2 * sizeof(char *));
    result[0] = (char *)malloc(strlen(n) + 1);
    strcpy(result[0], n);

    if (strlen(n) == 0) {
        result[1] = (char *)malloc(6);
        strcpy(result[1], "True");
        free(n);
        return result;
    }

    char *w = (char *)malloc(strlen(n) + 1);
    strcpy(w, n);
    reverse_string(w);

    if (strcmp(w, n) == 0) {
        result[1] = (char *)malloc(6);
        strcpy(result[1], "True");
    } else {
        result[1] = (char *)malloc(6);
        strcpy(result[1], "False");
    }

    free(n);
    free(w);
    return result;
}