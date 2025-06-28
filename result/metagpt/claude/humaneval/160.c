#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

int do_algebra(char** operato, int* operand, int operato_size, int operand_size) {
    int* posto = (int*)malloc(operand_size * sizeof(int));
    for (int i = 0; i < operand_size; i++) {
        posto[i] = i;
    }

    for (int i = 0; i < operato_size; i++) {
        if (strcmp(operato[i], "**") == 0) {
            while (posto[posto[i]] != posto[i]) posto[i] = posto[posto[i]];
            while (posto[posto[i + 1]] != posto[i + 1]) posto[i + 1] = posto[posto[i + 1]];
            operand[posto[i]] = (int)pow(operand[posto[i]], operand[posto[i + 1]]);
            posto[i + 1] = posto[i];
        }
    }

    for (int i = 0; i < operato_size; i++) {
        if (strcmp(operato[i], "*") == 0 || strcmp(operato[i], "//") == 0) {
            while (posto[posto[i]] != posto[i]) posto[i] = posto[posto[i]];
            while (posto[posto[i + 1]] != posto[i + 1]) posto[i + 1] = posto[posto[i + 1]];
            if (strcmp(operato[i], "*") == 0) {
                operand[posto[i]] = operand[posto[i]] * operand[posto[i + 1]];
            } else {
                operand[posto[i]] = operand[posto[i]] / operand[posto[i + 1]];
            }
            posto[i + 1] = posto[i];
        }
    }

    for (int i = 0; i < operato_size; i++) {
        if (strcmp(operato[i], "+") == 0 || strcmp(operato[i], "-") == 0) {
            while (posto[posto[i]] != posto[i]) posto[i] = posto[posto[i]];
            while (posto[posto[i + 1]] != posto[i + 1]) posto[i + 1] = posto[posto[i + 1]];
            if (strcmp(operato[i], "+") == 0) {
                operand[posto[i]] = operand[posto[i]] + operand[posto[i + 1]];
            } else {
                operand[posto[i]] = operand[posto[i]] - operand[posto[i + 1]];
            }
            posto[i + 1] = posto[i];
        }
    }

    int result = operand[0];
    free(posto);
    return result;
}