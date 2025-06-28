#include <stdio.h>
#include <stdlib.h>

int search(int* lst, int lst_size) {
    int max = -1;
    int* freq_values = (int*)malloc(lst_size * sizeof(int));
    int* freq_counts = (int*)malloc(lst_size * sizeof(int));
    int freq_size = 0;

    for (int i = 0; i < lst_size; i++) {
        bool has = false;
        for (int j = 0; j < freq_size; j++) {
            if (lst[i] == freq_values[j]) {
                freq_counts[j]++;
                has = true;
                if (freq_counts[j] >= freq_values[j] && freq_values[j] > max) {
                    max = freq_values[j];
                }
                break;
            }
        }
        if (!has) {
            freq_values[freq_size] = lst[i];
            freq_counts[freq_size] = 1;
            if (max == -1 && lst[i] == 1) {
                max = 1;
            }
            freq_size++;
        }
    }

    free(freq_values);
    free(freq_counts);

    return max;
}