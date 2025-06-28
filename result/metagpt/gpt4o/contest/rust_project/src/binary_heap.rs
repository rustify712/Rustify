// binary_heap.rs

/// Binary Heap implementation in Rust.
///
/// This implementation provides similar functionality to the C version,
/// supporting both min-heaps and max-heaps.

use std::cmp::Ordering;

pub enum BinaryHeapType {
    Min,
    Max,
}

pub struct BinaryHeap<T> {
    heap_type: BinaryHeapType,
    values: Vec<T>,
    compare_func: fn(&T, &T) -> Ordering,
}

impl<T> BinaryHeap<T> {
    /// Create a new binary heap.
    ///
    /// # Arguments
    /// * `heap_type` - The type of heap (min or max).
    /// * `compare_func` - The function to compare two elements.
    ///
    /// # Returns
    /// A new `BinaryHeap` instance.
    pub fn new(heap_type: BinaryHeapType, compare_func: fn(&T, &T) -> Ordering) -> Self {
        BinaryHeap {
            heap_type,
            values: Vec::new(),
            compare_func,
        }
    }

    /// Insert a value into the heap.
    ///
    /// # Arguments
    /// * `value` - The value to insert.
    pub fn insert(&mut self, value: T) {
        self.values.push(value);
        self.percolate_up(self.values.len() - 1);
    }

    /// Remove and return the top value from the heap.
    ///
    /// # Returns
    /// The top value, or `None` if the heap is empty.
    pub fn pop(&mut self) -> Option<T> {
        if self.values.is_empty() {
            return None;
        }

        let result = self.values.swap_remove(0);
        if !self.values.is_empty() {
            self.percolate_down(0);
        }
        Some(result)
    }

    /// Percolate a value up the heap to maintain heap order.
    ///
    /// # Arguments
    /// * `index` - The index of the value to percolate up.
    fn percolate_up(&mut self, mut index: usize) {
        while index > 0 {
            let parent = (index - 1) / 2;
            if self.compare(index, parent) == Ordering::Less {
                self.values.swap(index, parent);
                index = parent;
            } else {
                break;
            }
        }
    }

    /// Percolate a value down the heap to maintain heap order.
    ///
    /// # Arguments
    /// * `index` - The index of the value to percolate down.
    fn percolate_down(&mut self, mut index: usize) {
        let len = self.values.len();
        loop {
            let left = 2 * index + 1;
            let right = 2 * index + 2;
            let mut smallest = index;

            if left < len && self.compare(left, smallest) == Ordering::Less {
                smallest = left;
            }
            if right < len && self.compare(right, smallest) == Ordering::Less {
                smallest = right;
            }
            if smallest != index {
                self.values.swap(index, smallest);
                index = smallest;
            } else {
                break;
            }
        }
    }

    /// Compare two elements in the heap.
    ///
    /// # Arguments
    /// * `i` - The index of the first element.
    /// * `j` - The index of the second element.
    ///
    /// # Returns
    /// The ordering of the two elements.
    fn compare(&self, i: usize, j: usize) -> Ordering {
        match self.heap_type {
            BinaryHeapType::Min => (self.compare_func)(&self.values[i], &self.values[j]),
            BinaryHeapType::Max => (self.compare_func)(&self.values[j], &self.values[i]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn compare_ints(a: &i32, b: &i32) -> Ordering {
        a.cmp(b)
    }

    #[test]
    fn test_binary_heap_insert_and_pop() {
        let mut min_heap = BinaryHeap::new(BinaryHeapType::Min, compare_ints);
        min_heap.insert(3);
        min_heap.insert(1);
        min_heap.insert(4);
        min_heap.insert(1);
        min_heap.insert(5);

        assert_eq!(min_heap.pop(), Some(1));
        assert_eq!(min_heap.pop(), Some(1));
        assert_eq!(min_heap.pop(), Some(3));
        assert_eq!(min_heap.pop(), Some(4));
        assert_eq!(min_heap.pop(), Some(5));
        assert_eq!(min_heap.pop(), None);

        let mut max_heap = BinaryHeap::new(BinaryHeapType::Max, compare_ints);
        max_heap.insert(3);
        max_heap.insert(1);
        max_heap.insert(4);
        max_heap.insert(1);
        max_heap.insert(5);

        assert_eq!(max_heap.pop(), Some(5));
        assert_eq!(max_heap.pop(), Some(4));
        assert_eq!(max_heap.pop(), Some(3));
        assert_eq!(max_heap.pop(), Some(1));
        assert_eq!(max_heap.pop(), Some(1));
        assert_eq!(max_heap.pop(), None);
    }
}