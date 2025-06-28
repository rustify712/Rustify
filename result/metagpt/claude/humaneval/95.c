#include <stdio.h>
#include <string.h>
#include <stdbool.h>

typedef struct {
    char *key;
    char *value;
} KeyValuePair;

typedef struct {
    KeyValuePair *pairs;
    size_t size;
} Map;

bool check_dict_case(Map dict) {
    if (dict.size == 0) return false;

    int islower = 0, isupper = 0;

    for (size_t i = 0; i < dict.size; i++) {
        char *key = dict.pairs[i].key;

        for (int j = 0; j < strlen(key); j++) {
            if (key[j] < 65 || (key[j] > 90 && key[j] < 97) || key[j] > 122) return false;
            if (key[j] >= 65 && key[j] <= 90) isupper = 1;
            if (key[j] >= 97 && key[j] <= 122) islower = 1;
            if (isupper + islower == 2) return false;
        }
    }

    return true;
}