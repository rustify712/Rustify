#include <stdio.h>
#include <string.h>

int is_bored(const char* S) {
    int isstart = 1;
    int isi = 0;
    int sum = 0;
    int len = strlen(S);

    for (int i = 0; i < len; i++) {
        if (S[i] == ' ' && isi) {
            isi = 0;
            sum += 1;
        }
        if (S[i] == 'I' && isstart) {
            isi = 1;
        } else {
            isi = 0;
        }
        if (S[i] != ' ') {
            isstart = 0;
        }
        if (S[i] == '.' || S[i] == '?' || S[i] == '!') {
            isstart = 1;
        }
    }
    return sum;
}