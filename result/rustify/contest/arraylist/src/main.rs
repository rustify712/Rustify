/// Defines a generic `ArrayList` struct for storing data of any type.
#[derive(Debug, Clone, Default)]
pub struct ArrayList<T> {
    /// A vector to store the data.
    data: Vec<T>,
    /// The current number of elements in the ArrayList.
    length: usize,
}

impl<T: Ord> ArrayList<T> {
    /// Sorts the values in the `ArrayList`.
    pub fn sort(&mut self) {
        self.data.sort();
    }
}

impl<T> Drop for ArrayList<T> {
    fn drop(&mut self) {
        // No need to manually free memory, as `Vec<T>` will handle it.
    }
}

/// Creates a new `ArrayList`.
///
/// If the provided length is zero, a default capacity of 16 is used.
///
/// # Arguments
///
/// * `length` - The initial capacity hint.
///
/// # Returns
///
/// Returns a new `ArrayList`.
impl<T> ArrayList<T> {
    pub fn new(length: usize) -> Self {
        let length = if length == 0 { 16 } else { length };
        ArrayList {
            data: Vec::with_capacity(length),
            length: 0,
        }
    }

    /// Removes a range of entries at the specified location in the `ArrayList`.
    ///
    /// If the range exceeds the bounds of the `ArrayList`, no action is taken.
    pub fn remove_range(&mut self, index: usize, length: usize) {
        if index + length > self.length {
            return;
        }
        self.data.drain(index..index + length);
        self.length -= length;
    }

    /// Removes the entry at the specified location in the `ArrayList`.
    ///
    /// If the index is out of bounds, no action is taken.
    pub fn remove(&mut self, index: usize) {
        if index >= self.length {
            return;
        }
        self.data.remove(index);
        self.length -= 1;
    }

    /// Inserts a value at the specified index in the `ArrayList`.
    ///
    /// # Arguments
    ///
    /// * `index` - The index at which to insert the value.
    /// * `value` - The value to insert.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the value was successfully inserted, or `Err("Index out of bounds")` if the index is invalid.
    pub fn insert(&mut self, index: usize, value: T) -> Result<(), &'static str> {
        if index > self.length {
            return Err("Index out of bounds");
        }
        self.data.insert(index, value);
        self.length += 1;
        Ok(())
    }

    /// Clears all elements from the `ArrayList`.
    pub fn clear(&mut self) {
        self.data.clear();
        self.length = 0;
    }

    /// Finds the index of a particular value in the `ArrayList`.
    ///
    /// # Arguments
    ///
    /// * `data` - The value to search for.
    ///
    /// # Returns
    ///
    /// Returns `Some(index)` if the value is found, or `None` if the value is not present.
    pub fn index_of(&self, data: &T) -> Option<usize>
    where
        T: PartialEq,
    {
        self.data.iter().position(|x| x == data)
    }

    /// Prepends a value to the beginning of the `ArrayList`.
    ///
    /// # Arguments
    ///
    /// * `value` - The value to prepend.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the value was successfully prepended, or `Err("Index out of bounds")` if the index is invalid.
    pub fn prepend(&mut self, value: T) -> Result<(), &'static str> {
        self.insert(0, value)
    }

    /// Returns the number of elements in the `ArrayList`.
    pub fn len(&self) -> usize {
        self.length
    }

    /// Retrieves the element at the specified index in the `ArrayList`.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the element to retrieve.
    ///
    /// # Returns
    ///
    /// Returns `Some(&T)` if the index is valid, or `None` if the index is out of bounds.
    pub fn get(&self, index: usize) -> Option<&T> {
        if index < self.length {
            self.data.get(index)
        } else {
            None
        }
    }
}