// sortedarray.rs

/// Sorted array implementation in Rust.
///
/// This implementation provides similar functionality to the C version,
/// with operations to manipulate the sorted array.

pub struct SortedArray<T> {
    data: Vec<T>,
    equ_func: fn(&T, &T) -> bool,
    cmp_func: fn(&T, &T) -> std::cmp::Ordering,
}

impl<T> SortedArray<T> {
    /// Create a new sorted array.
    ///
    /// # Arguments
    /// * `equ_func` - The function to determine if two values are equal.
    /// * `cmp_func` - The function to determine the order of two values.
    ///
    /// # Returns
    /// A new `SortedArray` instance.
    pub fn new(equ_func: fn(&T, &T) -> bool, cmp_func: fn(&T, &T) -> std::cmp::Ordering) -> Self {
        SortedArray {
            data: Vec::new(),
            equ_func,
            cmp_func,
        }
    }

    /// Get the length of the sorted array.
    pub fn length(&self) -> usize {
        self.data.len()
    }

    /// Get the value at a specified index.
    ///
    /// # Arguments
    /// * `index` - The index of the value to retrieve.
    ///
    /// # Returns
    /// A reference to the value, or `None` if out of range.
    pub fn get(&self, index: usize) -> Option<&T> {
        self.data.get(index)
    }

    /// Insert a value into the sorted array.
    ///
    /// # Arguments
    /// * `value` - The value to insert.
    pub fn insert(&mut self, value: T) {
        let pos = self.data.binary_search_by(|x| (self.cmp_func)(x, &value)).unwrap_or_else(|e| e);
        self.data.insert(pos, value);
    }

    /// Remove a value at a specified index.
    ///
    /// # Arguments
    /// * `index` - The index of the value to remove.
    pub fn remove(&mut self, index: usize) {
        if index < self.data.len() {
            self.data.remove(index);
        }
    }

    /// Remove a range of values starting at a specified index.
    ///
    /// # Arguments
    /// * `index` - The starting index of the range to remove.
    /// * `length` - The number of elements to remove.
    pub fn remove_range(&mut self, index: usize, length: usize) {
        if index + length <= self.data.len() {
            self.data.drain(index..index + length);
        }
    }

    /// Find the first index of a value in the sorted array.
    ///
    /// # Arguments
    /// * `value` - The value to find.
    ///
    /// # Returns
    /// The index of the first occurrence, or `None` if not found.
    pub fn first_index(&self, value: &T) -> Option<usize> {
        self.data.iter().position(|x| (self.equ_func)(x, value))
    }

    /// Find the last index of a value in the sorted array.
    ///
    /// # Arguments
    /// * `value` - The value to find.
    ///
    /// # Returns
    /// The index of the last occurrence, or `None` if not found.
    pub fn last_index(&self, value: &T) -> Option<usize> {
        self.data.iter().rposition(|x| (self.equ_func)(x, value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn equ_ints(a: &i32, b: &i32) -> bool {
        a == b
    }

    fn cmp_ints(a: &i32, b: &i32) -> std::cmp::Ordering {
        a.cmp(b)
    }

    #[test]
    fn test_sorted_array_operations() {
        let mut array = SortedArray::new(equ_ints, cmp_ints);
        array.insert(3);
        array.insert(1);
        array.insert(2);

        assert_eq!(array.get(0), Some(&1));
        assert_eq!(array.get(1), Some(&2));
        assert_eq!(array.get(2), Some(&3));
        assert_eq!(array.length(), 3);

        array.remove(1);
        assert_eq!(array.get(1), Some(&3));
        assert_eq!(array.length(), 2);

        array.insert(2);
        array.insert(2);
        assert_eq!(array.first_index(&2), Some(1));
        assert_eq!(array.last_index(&2), Some(2));

        array.remove_range(1, 2);
        assert_eq!(array.length(), 1);
        assert_eq!(array.get(0), Some(&1));
    }
}