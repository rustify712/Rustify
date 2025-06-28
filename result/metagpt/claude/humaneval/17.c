#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int* parse_music(const char* music_string, int* out_size) {
    int capacity = 10;
    int* out = (int*)malloc(capacity * sizeof(int));
    int count = 0;
    char current[3] = {0}; // 用于存储当前音符，最大长度为2（"o|"或".|"）
    int i = 0;

    while (music_string[i] != '\0') {
        if (music_string[i] == ' ') {
            if (strcmp(current, "o") == 0) {
                if (count >= capacity) {
                    capacity *= 2;
                    out = (int*)realloc(out, capacity * sizeof(int));
                }
                out[count++] = 4;
            } else if (strcmp(current, "o|") == 0) {
                if (count >= capacity) {
                    capacity *= 2;
                    out = (int*)realloc(out, capacity * sizeof(int));
                }
                out[count++] = 2;
            } else if (strcmp(current, ".|") == 0) {
                if (count >= capacity) {
                    capacity *= 2;
                    out = (int*)realloc(out, capacity * sizeof(int));
                }
                out[count++] = 1;
            }
            current[0] = '\0'; // 重置current
        } else {
            strncat(current, &music_string[i], 1);
        }
        i++;
    }

    *out_size = count;
    return out;
}