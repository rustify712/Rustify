#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// 定义行星数组
const char *planets[] = {"Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"};
const int num_planets = sizeof(planets) / sizeof(planets[0]);

// 查找行星在数组中的位置
int find_planet_index(const char *planet) {
    for (int i = 0; i < num_planets; i++) {
        if (strcmp(planets[i], planet) == 0) {
            return i;
        }
    }
    return -1;
}

// 返回两个行星之间的行星数组
char** bf(const char *planet1, const char *planet2, int *out_size) {
    int pos1 = find_planet_index(planet1);
    int pos2 = find_planet_index(planet2);

    // 如果输入的行星名称无效，返回空数组
    if (pos1 == -1 || pos2 == -1) {
        *out_size = 0;
        return NULL;
    }

    // 确保 pos1 < pos2
    if (pos1 > pos2) {
        int temp = pos1;
        pos1 = pos2;
        pos2 = temp;
    }

    // 计算输出数组的大小
    *out_size = pos2 - pos1 - 1;
    if (*out_size <= 0) {
        *out_size = 0;
        return NULL;
    }

    // 分配内存并填充输出数组
    char **out = (char **)malloc(*out_size * sizeof(char *));
    for (int i = 0; i < *out_size; i++) {
        out[i] = strdup(planets[pos1 + 1 + i]);
    }

    return out;
}

// 释放动态分配的内存
void free_bf_result(char **result, int size) {
    for (int i = 0; i < size; i++) {
        free(result[i]);
    }
    free(result);
}