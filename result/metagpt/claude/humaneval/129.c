#include <stdio.h>
#include <stdlib.h>

int* minPath(int** grid, int gridSize, int* gridColSize, int k, int* returnSize) {
    int i, j, x, y, min;
    for (i = 0; i < gridSize; i++) {
        for (j = 0; j < gridColSize[i]; j++) {
            if (grid[i][j] == 1) {
                x = i;
                y = j;
            }
        }
    }
    
    min = gridSize * gridSize;
    if (x > 0 && grid[x-1][y] < min) min = grid[x-1][y];
    if (x < gridSize - 1 && grid[x+1][y] < min) min = grid[x+1][y];
    if (y > 0 && grid[x][y-1] < min) min = grid[x][y-1];
    if (y < gridSize - 1 && grid[x][y+1] < min) min = grid[x][y+1];
    
    int* out = (int*)malloc(k * sizeof(int));
    *returnSize = k;
    
    for (i = 0; i < k; i++) {
        if (i % 2 == 0) {
            out[i] = 1;
        } else {
            out[i] = min;
        }
    }
    
    return out;
}