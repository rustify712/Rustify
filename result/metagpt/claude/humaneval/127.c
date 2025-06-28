#include <stdio.h>
#include <string.h>

char* intersection(int interval1[2], int interval2[2]) {
    int inter1, inter2, l, i;
    inter1 = (interval1[0] > interval2[0]) ? interval1[0] : interval2[0];
    inter2 = (interval1[1] < interval2[1]) ? interval1[1] : interval2[1];
    l = inter2 - inter1;
    if (l < 2) return "NO";
    for (i = 2; i * i <= l; i++)
        if (l % i == 0) return "NO";
    return "YES";
}