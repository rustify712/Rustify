#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

bool valid_date(const char* date) {
    int mm, dd, yy;
    if (strlen(date) != 10) return false;
    for (int i = 0; i < 10; i++) {
        if (i == 2 || i == 5) {
            if (date[i] != '-') return false;
        } else {
            if (date[i] < '0' || date[i] > '9') return false;
        }
    }

    char mm_str[3], dd_str[3], yy_str[5];
    strncpy(mm_str, date, 2);
    mm_str[2] = '\0';
    strncpy(dd_str, date + 3, 2);
    dd_str[2] = '\0';
    strncpy(yy_str, date + 6, 4);
    yy_str[4] = '\0';

    mm = atoi(mm_str);
    dd = atoi(dd_str);
    yy = atoi(yy_str);

    if (mm < 1 || mm > 12) return false;
    if (dd < 1 || dd > 31) return false;
    if (dd == 31 && (mm == 4 || mm == 6 || mm == 9 || mm == 11 || mm == 2)) return false;
    if (dd == 30 && mm == 2) return false;

    return true;
}