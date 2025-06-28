#include <stdio.h>
#include <stdlib.h>

int* remove_duplicates(int* numbers, int size, int* result_size) {
    int* out = (int*)malloc(size * sizeof(int));
    int* has1 = (int*)malloc(size * sizeof(int));
    int* has2 = (int*)malloc(size * sizeof(int));
    int out_count = 0, has1_count = 0, has2_count = 0;

    for (int i = 0; i < size; i++) {
        int found_in_has2 = 0;
        for (int j = 0; j < has2_count; j++) {
            if (has2[j] == numbers[i]) {
                found_in_has2 = 1;
                break;
            }
        }
        if (found_in_has2) continue;

        int found_in_has1 = 0;
        for (int j = 0; j < has1_count; j++) {
            if (has1[j] == numbers[i]) {
                found_in_has1 = 1;
                break;
            }
        }
        if (found_in_has1) {
            has2[has2_count++] = numbers[i];
        } else {
            has1[has1_count++] = numbers[i];
        }
    }

    for (int i = 0; i < size; i++) {
        int found_in_has2 = 0;
        for (int j = 0; j < has2_count; j++) {
            if (has2[j] == numbers[i]) {
                found_in_has2 = 1;
                break;
            }
        }
        if (!found_in_has2) {
            out[out_count++] = numbers[i];
        }
    }

    *result_size = out_count;
    free(has1);
    free(has2);
    return out;
}