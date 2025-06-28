#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Define a structure to hold any type of value
typedef struct {
    enum { INT, DOUBLE, STRING, OTHER } type;
    union {
        int int_value;
        double double_value;
        char* string_value;
        void* other_value;
    };
} any;

// Define a structure to hold a list of any values
typedef struct {
    any* items;
    size_t size;
    size_t capacity;
} list_any;

// Function to filter integers from a list of any values
int* filter_integers(list_any values, size_t* out_size) {
    int* out = NULL;
    size_t count = 0;

    for (size_t i = 0; i < values.size; i++) {
        if (values.items[i].type == INT) {
            out = realloc(out, (count + 1) * sizeof(int));
            out[count] = values.items[i].int_value;
            count++;
        }
    }

    *out_size = count;
    return out;
}