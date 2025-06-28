#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char** odd_count(char** lst, int lst_size) {
    char** out = (char**)malloc(lst_size * sizeof(char*));
    for (int i = 0; i < lst_size; i++) {
        int sum = 0;
        for (int j = 0; j < strlen(lst[i]); j++) {
            if (lst[i][j] >= '0' && lst[i][j] <= '9' && (lst[i][j] - '0') % 2 == 1) {
                sum += 1;
            }
        }
        char* s = "the number of odd elements in the string i of the input.";
        char* s2 = (char*)malloc((strlen(s) + 20) * sizeof(char)); // Allocate enough space
        int k = 0;
        for (int j = 0; j < strlen(s); j++) {
            if (s[j] == 'i') {
                char num_str[20];
                sprintf(num_str, "%d", sum);
                strcat(s2, num_str);
                k += strlen(num_str);
            } else {
                s2[k++] = s[j];
            }
        }
        s2[k] = '\0';
        out[i] = s2;
    }
    return out;
}