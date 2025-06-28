#include <stdio.h>
#include <string.h>
#include <ctype.h>

char* encode(const char* message) {
    static char out[256];  // 假设消息长度不超过255个字符
    const char* vowels = "aeiouAEIOU";
    int len = strlen(message);
    int out_index = 0;

    for (int i = 0; i < len; i++) {
        char w = message[i];
        if (islower(w)) {
            w = toupper(w);
        } else if (isupper(w)) {
            w = tolower(w);
        }

        if (strchr(vowels, w) != NULL) {
            w = w + 2;
        }

        out[out_index++] = w;
    }
    out[out_index] = '\0';  // 添加字符串结束符
    return out;
}