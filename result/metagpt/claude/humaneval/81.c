#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char** numerical_letter_grade(float* grades, int size) {
    char** out = (char**)malloc(size * sizeof(char*));
    for (int i = 0; i < size; i++) {
        if (grades[i] >= 3.9999) {
            out[i] = strdup("A+");
        } else if (grades[i] > 3.7001 && grades[i] < 3.9999) {
            out[i] = strdup("A");
        } else if (grades[i] > 3.3001 && grades[i] <= 3.7001) {
            out[i] = strdup("A-");
        } else if (grades[i] > 3.0001 && grades[i] <= 3.3001) {
            out[i] = strdup("B+");
        } else if (grades[i] > 2.7001 && grades[i] <= 3.0001) {
            out[i] = strdup("B");
        } else if (grades[i] > 2.3001 && grades[i] <= 2.7001) {
            out[i] = strdup("B-");
        } else if (grades[i] > 2.0001 && grades[i] <= 2.3001) {
            out[i] = strdup("C+");
        } else if (grades[i] > 1.7001 && grades[i] <= 2.0001) {
            out[i] = strdup("C");
        } else if (grades[i] > 1.3001 && grades[i] <= 1.7001) {
            out[i] = strdup("C-");
        } else if (grades[i] > 1.0001 && grades[i] <= 1.3001) {
            out[i] = strdup("D+");
        } else if (grades[i] > 0.7001 && grades[i] <= 1.0001) {
            out[i] = strdup("D");
        } else if (grades[i] > 0.0001 && grades[i] <= 0.7001) {
            out[i] = strdup("D-");
        } else if (grades[i] <= 0.0001) {
            out[i] = strdup("E");
        }
    }
    return out;
}