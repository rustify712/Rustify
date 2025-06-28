#include <stdio.h>
#include <math.h>

float triangle_area(float a, float b, float c) {
    if (a + b <= c || a + c <= b || b + c <= a) return -1;
    float h = (a + b + c) / 2;
    float area = sqrt(h * (h - a) * (h - b) * (h - c));
    return area;
}