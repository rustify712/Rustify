#include <stdio.h>
#include <string.h>

void even_odd_palindrome(int n, int *even_count, int *odd_count) {
    *even_count = 0;
    *odd_count = 0;
    
    for (int i = 1; i <= n; i++) {
        char w[12]; // Assuming n <= 10^3, so the maximum number of digits is 4 (1000)
        sprintf(w, "%d", i);
        
        int len = strlen(w);
        int is_palindrome = 1;
        
        for (int j = 0; j < len / 2; j++) {
            if (w[j] != w[len - j - 1]) {
                is_palindrome = 0;
                break;
            }
        }
        
        if (is_palindrome) {
            if (i % 2 == 0) {
                (*even_count)++;
            } else {
                (*odd_count)++;
            }
        }
    }
}