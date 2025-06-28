#include <stdbool.h>
#include <string.h>

bool is_nested(const char* str) {
    int count = 0, maxcount = 0;
    int length = strlen(str);
    
    for (int i = 0; i < length; i++) {
        if (str[i] == '[') count += 1;
        if (str[i] == ']') count -= 1;
        if (count < 0) count = 0;
        if (count > maxcount) maxcount = count;
        if (count <= maxcount - 2) return true;
    }
    
    return false;
}