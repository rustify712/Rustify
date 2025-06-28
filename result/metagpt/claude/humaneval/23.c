#include <string.h>

int strlen(const char *str) {
    return (int)strlen(str);
}
```

### 说明：
1. C语言中没有`string`类，因此需要使用`char *`来表示字符串。
2. C标准库中已经有一个`strlen`函数，用于计算字符串的长度。为了避免冲突，可以将函数名改为其他名称，例如`my_strlen`。
3. 由于C标准库中的`strlen`返回的是`size_t`类型，这里将其强制转换为`int`类型以匹配原C++代码的返回类型。

如果希望避免与标准库函数冲突，可以将函数名改为`my_strlen`：

```c
#include <string.h>

int my_strlen(const char *str) {
    return (int)strlen(str);
}