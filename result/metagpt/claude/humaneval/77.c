#include <stdio.h>
#include <math.h>

int iscuber(int a) {
    for (int i = 0; i * i * i <= abs(a); i++) {
        if (i * i * i == abs(a)) {
            return 1;
        }
    }
    return 0;
}