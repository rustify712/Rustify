// arraylist.rs

/// A dynamically resizing array list in Rust.
///
/// This implementation provides similar functionality to the C version,
/// with automatic resizing and various utility functions.

pub struct ArrayList<T> {
    data: Vec<T>,
}

impl<T> ArrayList<T> {
    /// Create a new array list with an initial capacity.
    ///
    /// # Arguments
    /// * `capacity` - The initial capacity of the array list.
    ///
    /// # Returns
    /// A new `ArrayList` instance.
    pub fn new(capacity: usize) -> Self {
        ArrayList {
            data: Vec::with_capacity(capacity),
        }
    }

    /// Append an element to the end of the array list.
    ///
    /// # Arguments
    /// * `value` - The value to append.
    pub fn append(&mut self, value: T) {
        self.data.push(value);
    }

    /// Insert an element at a specified index.
    ///
    /// # Arguments
    /// * `index` - The index at which to insert the element.
    /// * `value` - The value to insert.
    ///
    /// # Panics
    /// Panics if `index` is out of bounds.
    pub fn insert(&mut self, index: usize, value: T) {
        if index > self.data.len() {
            panic!("Index out of bounds");
        }
        self.data.insert(index, value);
    }

    /// Remove an element at a specified index.
    ///
    /// # Arguments
    /// * `index` - The index of the element to remove.
    ///
    /// # Panics
    /// Panics if `index` is out of bounds.
    pub fn remove(&mut self, index: usize) {
        if index >= self.data.len() {
            panic!("Index out of bounds");
        }
        self.data.remove(index);
    }

    /// Get the length of the array list.
    ///
    /// # Returns
    /// The number of elements in the array list.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the array list is empty.
    ///
    /// # Returns
    /// `true` if the array list is empty, `false` otherwise.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Clear all elements from the array list.
    pub fn clear(&mut self) {
        self.data.clear();
    }

    /// Get a reference to an element at a specified index.
    ///
    /// # Arguments
    /// * `index` - The index of the element to retrieve.
    ///
    /// # Returns
    /// A reference to the element.
    ///
    /// # Panics
    /// Panics if `index` is out of bounds.
    pub fn get(&self, index: usize) -> &T {
        &self.data[index]
    }

    /// Get a mutable reference to an element at a specified index.
    ///
    /// # Arguments
    /// * `index` - The index of the element to retrieve.
    ///
    /// # Returns
    /// A mutable reference to the element.
    ///
    /// # Panics
    /// Panics if `index` is out of bounds.
    pub fn get_mut(&mut self, index: usize) -> &mut T {
        &mut self.data[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arraylist_operations() {
        let mut list = ArrayList::new(10);
        assert!(list.is_empty());

        list.append(1);
        list.append(2);
        list.append(3);
        assert_eq!(list.len(), 3);
        assert_eq!(*list.get(0), 1);
        assert_eq!(*list.get(1), 2);
        assert_eq!(*list.get(2), 3);

        list.insert(1, 4);
        assert_eq!(*list.get(1), 4);
        assert_eq!(list.len(), 4);

        list.remove(1);
        assert_eq!(*list.get(1), 2);
        assert_eq!(list.len(), 3);

        list.clear();
        assert!(list.is_empty());
    }
}