#include <stdio.h>
#include <stdlib.h>

int* tri(int n, int* returnSize) {
    *returnSize = n + 1;
    int* out = (int*)malloc((n + 1) * sizeof(int));
    if (out == NULL) {
        *returnSize = 0;
        return NULL;
    }

    out[0] = 1;
    if (n == 0) {
        return out;
    }

    out[1] = 3;
    for (int i = 2; i <= n; i++) {
        if (i % 2 == 0) {
            out[i] = 1 + i / 2;
        } else {
            out[i] = out[i - 1] + out[i - 2] + 1 + (i + 1) / 2;
        }
    }

    return out;
}
```

### 说明：
1. **动态内存分配**：C语言中没有`vector`，因此使用`malloc`动态分配内存来存储结果。
2. **返回数组大小**：通过`returnSize`参数返回数组的大小。
3. **内存释放**：调用者在使用完返回的数组后，需要手动释放内存，以避免内存泄漏。

### 使用示例：
```c
int main() {
    int n = 3;
    int returnSize;
    int* result = tri(n, &returnSize);

    for (int i = 0; i < returnSize; i++) {
        printf("%d ", result[i]);
    }

    free(result);  // 释放内存
    return 0;
}
```

### 输出：
```
1 3 2 8 