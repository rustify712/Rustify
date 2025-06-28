#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int count_nums(int* n, int size) {
    int num = 0;
    for (int i = 0; i < size; i++) {
        if (n[i] > 0) {
            num += 1;
        } else {
            int sum = 0;
            int w;
            w = abs(n[i]);
            while (w >= 10) {
                sum += w % 10;
                w = w / 10;
            }
            sum -= w;
            if (sum > 0) num += 1;
        }
    }
    return num;
}