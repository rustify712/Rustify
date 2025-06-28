#include <stdio.h>
#include <string.h>

const char* file_name_check(const char* file_name) {
    int numdigit = 0, numdot = 0;
    int length = strlen(file_name);
    
    if (length < 5) return "No";
    
    char w = file_name[0];
    if (!((w >= 'A' && w <= 'Z') || (w >= 'a' && w <= 'z'))) return "No";
    
    const char* last = file_name + length - 4;
    if (strcmp(last, ".txt") != 0 && strcmp(last, ".exe") != 0 && strcmp(last, ".dll") != 0) return "No";
    
    for (int i = 0; i < length; i++) {
        if (file_name[i] >= '0' && file_name[i] <= '9') numdigit++;
        if (file_name[i] == '.') numdot++;
    }
    
    if (numdigit > 3 || numdot != 1) return "No";
    
    return "Yes";
}