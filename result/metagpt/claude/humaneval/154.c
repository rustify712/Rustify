#include <stdio.h>
#include <string.h>

int cycpattern_check(const char *a, const char *b) {
    int len_a = strlen(a);
    int len_b = strlen(b);
    char rotate[len_b + 1];

    for (int i = 0; i < len_b; i++) {
        // 构造旋转后的字符串
        strncpy(rotate, b + i, len_b - i);
        strncpy(rotate + len_b - i, b, i);
        rotate[len_b] = '\0';

        // 检查旋转后的字符串是否是a的子串
        if (strstr(a, rotate) != NULL) {
            return 1;
        }
    }
    return 0;
}