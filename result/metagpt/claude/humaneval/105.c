#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Function to sort an array of integers in ascending order
void sort(int *arr, int size) {
    for (int i = 0; i < size - 1; i++) {
        for (int j = 0; j < size - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

// Function to reverse an array of integers
void reverse(int *arr, int size) {
    for (int i = 0; i < size / 2; i++) {
        int temp = arr[i];
        arr[i] = arr[size - i - 1];
        arr[size - i - 1] = temp;
    }
}

// Function to convert an integer to its corresponding word
const char* numToWord(int num) {
    switch (num) {
        case 0: return "Zero";
        case 1: return "One";
        case 2: return "Two";
        case 3: return "Three";
        case 4: return "Four";
        case 5: return "Five";
        case 6: return "Six";
        case 7: return "Seven";
        case 8: return "Eight";
        case 9: return "Nine";
        default: return NULL;
    }
}

// Function to process the array and return the result
char** by_length(int *arr, int size, int *result_size) {
    // Sort the array
    sort(arr, size);

    // Reverse the array
    reverse(arr, size);

    // Count the number of valid elements (1-9)
    int count = 0;
    for (int i = 0; i < size; i++) {
        if (arr[i] >= 1 && arr[i] <= 9) {
            count++;
        }
    }

    // Allocate memory for the result array
    char **result = (char **)malloc(count * sizeof(char *));
    if (result == NULL) {
        *result_size = 0;
        return NULL;
    }

    // Fill the result array with the corresponding words
    int index = 0;
    for (int i = 0; i < size; i++) {
        if (arr[i] >= 1 && arr[i] <= 9) {
            const char *word = numToWord(arr[i]);
            result[index] = (char *)malloc((strlen(word) + 1) * sizeof(char));
            if (result[index] == NULL) {
                // Free previously allocated memory in case of failure
                for (int j = 0; j < index; j++) {
                    free(result[j]);
                }
                free(result);
                *result_size = 0;
                return NULL;
            }
            strcpy(result[index], word);
            index++;
        }
    }

    *result_size = count;
    return result;
}