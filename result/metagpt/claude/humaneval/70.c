#include <stdio.h>
#include <stdlib.h>

int* strange_sort_list(int* lst, int size, int* out_size) {
    // Sort the input list
    qsort(lst, size, sizeof(int), cmpfunc);

    // Allocate memory for the output list
    int* out = (int*)malloc(size * sizeof(int));
    *out_size = 0;

    int l = 0, r = size - 1;
    while (l < r) {
        out[(*out_size)++] = lst[l++];
        out[(*out_size)++] = lst[r--];
    }
    if (l == r) {
        out[(*out_size)++] = lst[l];
    }

    return out;
}

// Comparison function for qsort
int cmpfunc(const void* a, const void* b) {
    return (*(int*)a - *(int*)b);
}
```

### 说明：
1. **qsort**：C语言中没有`std::sort`，因此使用`qsort`来对数组进行排序。
2. **malloc**：C语言中没有`vector`，因此使用`malloc`动态分配内存来存储输出数组。
3. **cmpfunc**：`qsort`需要一个比较函数，`cmpfunc`用于比较两个整数的大小。
4. **out_size**：由于C语言中没有`vector`，因此需要通过指针返回输出数组的大小。

### 使用示例：
```c
int main() {
    int lst[] = {1, 2, 3, 4};
    int size = sizeof(lst) / sizeof(lst[0]);
    int out_size;
    int* out = strange_sort_list(lst, size, &out_size);

    for (int i = 0; i < out_size; i++) {
        printf("%d ", out[i]);
    }
    printf("\n");

    free(out); // 释放动态分配的内存
    return 0;
}