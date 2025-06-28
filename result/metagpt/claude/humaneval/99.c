#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int closest_integer(const char* value) {
    double w = atof(value);
    return round(w);
}