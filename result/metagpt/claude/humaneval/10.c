#include <stdio.h>
#include <string.h>
#include <stdbool.h>

bool is_palindrome(const char *str) {
    // Test if given string is a palindrome
    int len = strlen(str);
    for (int i = 0; i < len / 2; i++) {
        if (str[i] != str[len - 1 - i]) {
            return false;
        }
    }
    return true;
}

void make_palindrome(const char *str, char *result) {
    /*
    Find the shortest palindrome that begins with a supplied string.
    Algorithm idea is simple: - Find the longest postfix of supplied string that is a palindrome.
    - Append to the end of the string reverse of a string prefix that comes before the palindromic suffix.
    >>> make_palindrome("")
    ""
    >>> make_palindrome("cat")
    "catac"
    >>> make_palindrome("cata")
    "catac"
    */
    int len = strlen(str);
    for (int i = 0; i < len; i++) {
        if (is_palindrome(str + i)) {
            // Copy the original string to the result
            strcpy(result, str);
            // Append the reverse of the prefix before the palindromic suffix
            for (int j = i - 1; j >= 0; j--) {
                result[len + (i - 1 - j)] = str[j];
            }
            result[len + i] = '\0';
            return;
        }
    }
    // If no palindromic suffix is found, append the reverse of the entire string
    strcpy(result, str);
    for (int i = len - 1; i >= 0; i--) {
        result[len + (len - 1 - i)] = str[i];
    }
    result[2 * len] = '\0';
}