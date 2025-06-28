#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int* compare(int* game, int* guess, int size) {
    int* out = (int*)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        out[i] = abs(game[i] - guess[i]);
    }
    return out;
}