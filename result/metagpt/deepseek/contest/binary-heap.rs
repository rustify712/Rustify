// Rust implementation of Binary Heap
// Translated from C version

use std::cmp::Ordering;

#[derive(Debug, Clone, Copy)]
pub enum BinaryHeapType {
    Min,
    Max,
}

pub type BinaryHeapValue = i32;

pub struct BinaryHeap {
    heap_type: BinaryHeapType,
    values: Vec<BinaryHeapValue>,
    compare_func: fn(BinaryHeapValue, BinaryHeapValue) -> Ordering,
}

impl BinaryHeap {
    pub fn new(heap_type: BinaryHeapType, compare_func: fn(BinaryHeapValue, BinaryHeapValue) -> Ordering) -> Self {
        BinaryHeap {
            heap_type,
            values: Vec::with_capacity(16),
            compare_func,
        }
    }

    fn compare(&self, a: BinaryHeapValue, b: BinaryHeapValue) -> Ordering {
        match self.heap_type {
            BinaryHeapType::Min => (self.compare_func)(a, b),
            BinaryHeapType::Max => (self.compare_func)(a, b).reverse(),
        }
    }

    pub fn insert(&mut self, value: BinaryHeapValue) -> bool {
        self.values.push(value);
        let mut index = self.values.len() - 1;
        
        while index > 0 {
            let parent = (index - 1) / 2;
            
            match self.compare(self.values[index], self.values[parent]) {
                Ordering::Less => {
                    self.values.swap(index, parent);
                    index = parent;
                }
                _ => break,
            }
        }
        
        true
    }
    
    pub fn extract(&mut self) -> Option<BinaryHeapValue> {
        if self.values.is_empty() {
            return None;
        }
        
        let result = self.values[0];
        let last = self.values.pop().unwrap();
        
        if !self.values.is_empty() {
            self.values[0] = last;
            self.heapify(0);
        }
        
        Some(result)
    }
    
    fn heapify(&mut self, index: usize) {
        let left = 2 * index + 1;
        let right = 2 * index + 2;
        let mut smallest = index;
        
        if left < self.values.len() && 
           self.compare(self.values[left], self.values[smallest]) == Ordering::Less {
            smallest = left;
        }
        
        if right < self.values.len() && 
           self.compare(self.values[right], self.values[smallest]) == Ordering::Less {
            smallest = right;
        }
        
        if smallest != index {
            self.values.swap(index, smallest);
            self.heapify(smallest);
        }
    }
}