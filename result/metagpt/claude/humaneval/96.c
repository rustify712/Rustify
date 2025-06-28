#include <stdio.h>
#include <stdlib.h>

int* count_up_to(int n, int* returnSize) {
    int* out = (int*)malloc(n * sizeof(int));
    int count = 0;
    int i, j;

    for (i = 2; i < n; i++) {
        if (count == 0) {
            out[count++] = i;
        } else {
            int isp = 1;
            for (j = 0; out[j] * out[j] <= i; j++) {
                if (i % out[j] == 0) {
                    isp = 0;
                    break;
                }
            }
            if (isp) {
                out[count++] = i;
            }
        }
    }

    *returnSize = count;
    return out;
}