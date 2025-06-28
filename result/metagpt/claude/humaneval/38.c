#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char* encode_cyclic(const char* s) {
    // returns encoded string by cycling groups of three characters.
    // split string to groups. Each of length 3.
    int l = strlen(s);
    int num = (l + 2) / 3;
    char* output = (char*)malloc(l + 1);
    output[0] = '\0'; // Initialize the output string

    for (int i = 0; i * 3 < l; i++) {
        char x[4] = {0}; // Temporary buffer to hold the group of 3 characters
        strncpy(x, s + i * 3, 3); // Copy the group of 3 characters

        if (strlen(x) == 3) {
            // Cycle elements in each group. Unless group has fewer elements than 3.
            char temp = x[0];
            x[0] = x[1];
            x[1] = x[2];
            x[2] = temp;
        }

        strcat(output, x); // Append the cycled group to the output string
    }

    return output;
}

char* decode_cyclic(const char* s) {
    /*
    takes as input string encoded with encode_cyclic function. Returns decoded string.
    */
    int l = strlen(s);
    int num = (l + 2) / 3;
    char* output = (char*)malloc(l + 1);
    output[0] = '\0'; // Initialize the output string

    for (int i = 0; i * 3 < l; i++) {
        char x[4] = {0}; // Temporary buffer to hold the group of 3 characters
        strncpy(x, s + i * 3, 3); // Copy the group of 3 characters

        if (strlen(x) == 3) {
            // Reverse the cycling operation
            char temp = x[2];
            x[2] = x[1];
            x[1] = x[0];
            x[0] = temp;
        }

        strcat(output, x); // Append the decoded group to the output string
    }

    return output;
}
```

### 说明：
1. **内存管理**：在C语言中，字符串是字符数组，因此需要使用`malloc`动态分配内存来存储结果字符串。调用者在使用完返回的字符串后，需要手动释放内存。
2. **字符串操作**：C语言中没有`string`类，因此使用`strncpy`和`strcat`等函数来操作字符串。
3. **循环移位**：在`encode_cyclic`和`decode_cyclic`函数中，通过交换字符的位置来实现循环移位。

### 注意：
- 调用者在使用完`encode_cyclic`和`decode_cyclic`返回的字符串后，应使用`free`函数释放内存，以避免内存泄漏。

例如：
```c
char* encoded = encode_cyclic("abcdef");
printf("Encoded: %s\n", encoded);
free(encoded);

char* decoded = decode_cyclic("bcaefd");
printf("Decoded: %s\n", decoded);
free(decoded);