#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int* parse_nested_parens(const char* paren_string, int* result_size) {
    int* all_levels = NULL;
    int capacity = 0;
    int size = 0;
    int level = 0, max_level = 0;
    int i;

    for (i = 0; paren_string[i] != '\0'; i++) {
        char chr = paren_string[i];
        if (chr == '(') {
            level += 1;
            if (level > max_level) max_level = level;
        } else if (chr == ')') {
            level -= 1;
            if (level == 0) {
                if (size >= capacity) {
                    capacity = (capacity == 0) ? 1 : capacity * 2;
                    all_levels = (int*)realloc(all_levels, capacity * sizeof(int));
                }
                all_levels[size++] = max_level;
                max_level = 0;
            }
        }
    }

    *result_size = size;
    return all_levels;
}