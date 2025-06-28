#include <stdio.h>

// 定义一个结构体来存储结果
typedef struct {
    int total_eaten;
    int carrots_left;
} Result;

// 函数定义
Result eat(int number, int need, int remaining) {
    Result result;
    if (need > remaining) {
        result.total_eaten = number + remaining;
        result.carrots_left = 0;
    } else {
        result.total_eaten = number + need;
        result.carrots_left = remaining - need;
    }
    return result;
}