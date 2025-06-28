#include <stdbool.h>
#include <string.h>

bool check_if_last_char_is_a_letter(const char* txt) {
    int length = strlen(txt);
    if (length == 0) return false;
    
    char chr = txt[length - 1];
    if (chr < 65 || (chr > 90 && chr < 97) || chr > 122) return false;
    
    if (length == 1) return true;
    
    chr = txt[length - 2];
    if ((chr >= 65 && chr <= 90) || (chr >= 97 && chr <= 122)) return false;
    
    return true;
}