#include <stdio.h>

int max_fill(int** grid, int gridSize, int* gridColSize, int capacity) {
    int out = 0;
    for (int i = 0; i < gridSize; i++) {
        int sum = 0;
        for (int j = 0; j < gridColSize[i]; j++) {
            sum += grid[i][j];
        }
        if (sum > 0) {
            out += (sum - 1) / capacity + 1;
        }
    }
    return out;
}