#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <openssl/md5.h>

char* string_to_md5(const char* text) {
    unsigned char md[MD5_DIGEST_LENGTH];
    if (strlen(text) == 0) {
        return "None";
    }

    MD5_CTX c;
    MD5_Init(&c);
    MD5_Update(&c, (unsigned char*)text, strlen(text));
    MD5_Final(md, &c);

    char* out_str = (char*)malloc(33); // MD5 hash is 32 characters long + 1 for null terminator
    if (out_str == NULL) {
        return NULL; // Handle memory allocation failure
    }

    for (int i = 0; i < MD5_DIGEST_LENGTH; i++) {
        char w;
        if (md[i] < 160) {
            w = 48 + md[i] / 16;
        } else {
            w = 87 + md[i] / 16;
        }
        out_str[2 * i] = w;

        if (md[i] % 16 < 10) {
            w = 48 + md[i] % 16;
        } else {
            w = 87 + md[i] % 16;
        }
        out_str[2 * i + 1] = w;
    }

    out_str[32] = '\0'; // Null-terminate the string
    return out_str;
}