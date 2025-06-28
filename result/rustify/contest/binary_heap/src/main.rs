/// Heap type. If a heap is a min heap (`Min`), the values with the lowest priority are stored at the top of the heap and will be the first returned.
/// If a heap is a max heap (`Max`), the values with the greatest priority are stored at the top of the heap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryHeapType {
    /// A minimum heap.
    Min,
    /// A maximum heap.
    Max,
}

/// A binary heap data structure.
#[derive(Debug, Clone)]
pub struct BinaryHeap<T> {
    heap_type: BinaryHeapType,
    values: Vec<T>,
}

impl<T: Ord> BinaryHeap<T> {
    /// Create a new binary heap.
    ///
    /// # Arguments
    ///
    /// * `heap_type` - The type of heap: min heap or max heap.
    ///
    /// # Returns
    ///
    /// A new binary heap.
    pub fn new(heap_type: BinaryHeapType) -> Self {
        BinaryHeap { heap_type, values: Vec::new() }
    }

    /// Insert a value into the binary heap.
    ///
    /// # Arguments
    ///
    /// * `value` - The value to insert.
    ///
    /// # Returns
    ///
    /// Returns `true` if the value was successfully inserted.
    pub fn insert(&mut self, value: T) -> bool {
        self.values.push(value);
        let mut index = self.values.len() - 1;

        while index > 0 {
            let parent = (index - 1) / 2;
            if self.values[parent] < self.values[index] {
                break;
            }
            self.values.swap(index, parent);
            index = parent;
        }

        true
    }
}