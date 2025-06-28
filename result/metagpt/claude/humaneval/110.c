#include <stdio.h>
#include <string.h>

const char* exchange(int* lst1, int lst1_size, int* lst2, int lst2_size) {
    int num = 0;
    for (int i = 0; i < lst1_size; i++) {
        if (lst1[i] % 2 == 0) num++;
    }
    for (int i = 0; i < lst2_size; i++) {
        if (lst2[i] % 2 == 0) num++;
    }
    if (num >= lst1_size) return "YES";
    return "NO";
}