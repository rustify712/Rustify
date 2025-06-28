// Rust implementation of Binomial Heap
// Translated from C version

use std::cmp::Ordering;

#[derive(Debug, Clone, Copy)]
pub enum BinomialHeapType {
    Min,
    Max,
}

pub type BinomialHeapValue = i32;

struct BinomialTree {
    value: BinomialHeapValue,
    order: u16,
    refcount: u16,
    subtrees: Vec<Box<BinomialTree>>,
}

pub struct BinomialHeap {
    heap_type: BinomialHeapType,
    compare_func: fn(BinomialHeapValue, BinomialHeapValue) -> Ordering,
    num_values: usize,
    roots: Vec<Option<Box<BinomialTree>>>,
}

impl BinomialHeap {
    pub fn new(heap_type: BinomialHeapType, compare_func: fn(BinomialHeapValue, BinomialHeapValue) -> Ordering) -> Self {
        BinomialHeap {
            heap_type,
            compare_func,
            num_values: 0,
            roots: Vec::new(),
        }
    }

    fn compare(&self, a: BinomialHeapValue, b: BinomialHeapValue) -> Ordering {
        match self.heap_type {
            BinomialHeapType::Min => (self.compare_func)(a, b),
            BinomialHeapType::Max => (self.compare_func)(a, b).reverse(),
        }
    }
}