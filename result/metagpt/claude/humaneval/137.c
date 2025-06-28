#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

typedef enum {
    TYPE_INT,
    TYPE_DOUBLE,
    TYPE_STRING
} ValueType;

typedef struct {
    ValueType type;
    union {
        int int_value;
        double double_value;
        char* string_value;
    };
} AnyValue;

AnyValue compare_one(AnyValue a, AnyValue b) {
    double numa, numb;
    AnyValue out;

    // Convert a to double
    if (a.type == TYPE_STRING) {
        char* s = a.string_value;
        char* comma = strchr(s, ',');
        if (comma) {
            *comma = '.';
        }
        numa = atof(s);
    } else if (a.type == TYPE_INT) {
        numa = (double)a.int_value;
    } else if (a.type == TYPE_DOUBLE) {
        numa = a.double_value;
    }

    // Convert b to double
    if (b.type == TYPE_STRING) {
        char* s = b.string_value;
        char* comma = strchr(s, ',');
        if (comma) {
            *comma = '.';
        }
        numb = atof(s);
    } else if (b.type == TYPE_INT) {
        numb = (double)b.int_value;
    } else if (b.type == TYPE_DOUBLE) {
        numb = b.double_value;
    }

    // Compare and return the larger value
    if (numa == numb) {
        out.type = TYPE_STRING;
        out.string_value = strdup("None");
        return out;
    } else if (numa < numb) {
        return b;
    } else {
        return a;
    }
}