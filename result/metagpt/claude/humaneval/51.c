#include <stdio.h>
#include <string.h>

char* remove_vowels(const char* text) {
    int len = strlen(text);
    char* out = (char*)malloc(len + 1); // Allocate memory for the output string
    if (out == NULL) {
        return NULL; // Handle memory allocation failure
    }

    const char* vowels = "AEIOUaeiou";
    int out_index = 0;

    for (int i = 0; i < len; i++) {
        if (strchr(vowels, text[i]) == NULL) {
            out[out_index++] = text[i];
        }
    }

    out[out_index] = '\0'; // Null-terminate the output string
    return out;
}
```

### 说明：
1. **内存管理**：C语言中没有`string`类，因此需要使用`char*`来表示字符串，并且需要手动管理内存。`remove_vowels`函数返回的字符串是动态分配的，调用者需要在使用完后释放内存。
2. **字符串操作**：使用`strchr`函数来检查字符是否在元音字符串中。
3. **字符串长度**：使用`strlen`函数获取输入字符串的长度。
4. **输出字符串**：使用`malloc`动态分配内存来存储输出字符串，并在最后添加`\0`来终止字符串。

### 注意：
- 调用者在使用完`remove_vowels`返回的字符串后，需要使用`free`函数释放内存，以避免内存泄漏。

例如：
```c
char* result = remove_vowels("abcdef");
printf("%s\n", result); // 输出: bcdf
free(result); // 释放内存