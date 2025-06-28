#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Function to filter an array of strings based on whether they contain a given substring
char** filter_by_substring(char** strings, int num_strings, const char* substring, int* out_num_strings) {
    // Allocate memory for the output array (maximum size is num_strings)
    char** out = (char**)malloc(num_strings * sizeof(char*));
    int count = 0;

    // Iterate through each string in the input array
    for (int i = 0; i < num_strings; i++) {
        // Check if the substring is present in the current string
        if (strstr(strings[i], substring) != NULL) {
            // If found, add the string to the output array
            out[count] = strings[i];
            count++;
        }
    }

    // Set the number of strings in the output array
    *out_num_strings = count;

    // Return the filtered array
    return out;
}