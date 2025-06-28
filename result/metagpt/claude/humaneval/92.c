#include <stdio.h>
#include <math.h>

int any_int(float a, float b, float c) {
    if (round(a) != a) return 0;
    if (round(b) != b) return 0;
    if (round(c) != c) return 0;
    if (a + b == c || a + c == b || b + c == a) return 1;
    return 0;
}