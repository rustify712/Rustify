#include <stdio.h>
#include <string.h>

char* Strongest_Extension(const char* class_name, const char** extensions, int extensions_count) {
    char* strongest = NULL;
    int max_strength = -1000;

    for (int i = 0; i < extensions_count; i++) {
        int strength = 0;
        const char* ext = extensions[i];
        int len = strlen(ext);

        for (int j = 0; j < len; j++) {
            char chr = ext[j];
            if (chr >= 'A' && chr <= 'Z') strength += 1;
            if (chr >= 'a' && chr <= 'z') strength -= 1;
        }

        if (strength > max_strength) {
            max_strength = strength;
            strongest = (char*)ext;
        }
    }

    // Assuming the result is stored in a static buffer for simplicity
    static char result[256];
    snprintf(result, sizeof(result), "%s.%s", class_name, strongest);
    return result;
}