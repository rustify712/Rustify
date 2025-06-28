#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int specialFilter(int* nums, int size) {
    int num = 0;
    for (int i = 0; i < size; i++) {
        if (nums[i] > 10) {
            char buffer[20];
            snprintf(buffer, sizeof(buffer), "%d", nums[i]);
            int len = strlen(buffer);
            if ((buffer[0] - '0') % 2 == 1 && (buffer[len - 1] - '0') % 2 == 1) {
                num++;
            }
        }
    }
    return num;
}