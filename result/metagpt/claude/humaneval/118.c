#include <stdio.h>
#include <string.h>

char* get_closest_vowel(const char* word) {
    static char out[2] = "";  // 用于返回结果的字符串
    const char* vowels = "AEIOUaeiou";
    int len = strlen(word);

    for (int i = len - 2; i >= 1; i--) {
        if (strchr(vowels, word[i]) != NULL) {
            if (strchr(vowels, word[i + 1]) == NULL && strchr(vowels, word[i - 1]) == NULL) {
                out[0] = word[i];
                out[1] = '\0';
                return out;
            }
        }
    }

    return out;  // 如果没有找到符合条件的元音，返回空字符串
}