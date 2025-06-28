#include <stdio.h>
#include <stdlib.h>
#include <string.h>

bool simplify(const char* x, const char* n) {
    int a, b, c, d;
    char* x_slash = strchr(x, '/');
    char* n_slash = strchr(n, '/');
    
    a = atoi(strndup(x, x_slash - x));
    b = atoi(strdup(x_slash + 1));
    
    c = atoi(strndup(n, n_slash - n));
    d = atoi(strdup(n_slash + 1));
    
    if ((a * c) % (b * d) == 0) return true;
    return false;
}