// Rust实现有序数组
// 从C版本转换而来

use std::cmp::Ordering;

pub type SortedArrayValue = i32;
pub type SortedArrayEqualFunc = fn(&SortedArrayValue, &SortedArrayValue) -> bool;
pub type SortedArrayCompareFunc = fn(&SortedArrayValue, &SortedArrayValue) -> Ordering;

pub struct SortedArray {
    data: Vec<SortedArrayValue>,
    equal_func: SortedArrayEqualFunc,
    compare_func: SortedArrayCompareFunc,
}

impl SortedArray {
    pub fn new(
        equal_func: SortedArrayEqualFunc, 
        compare_func: SortedArrayCompareFunc
    ) -> Self {
        SortedArray {
            data: Vec::new(),
            equal_func,
            compare_func,
        }
    }

    // 查找第一个等于data的索引
    fn first_index(&self, data: &SortedArrayValue) -> Option<usize> {
        self.data.binary_search_by(|x| (self.compare_func)(x, data))
            .ok()
    }

    // 查找最后一个等于data的索引
    fn last_index(&self, data: &SortedArrayValue) -> Option<usize> {
        let mut index = self.first_index(data)?;
        while index + 1 < self.data.len() && (self.equal_func)(&self.data[index + 1], data) {
            index += 1;
        }
        Some(index)
    }

    // 插入元素并保持有序
    pub fn insert(&mut self, value: SortedArrayValue) {
        let index = match self.data.binary_search_by(|x| (self.compare_func)(x, &value)) {
            Ok(i) => i,
            Err(i) => i,
        };
        self.data.insert(index, value);
    }

    // 删除元素
    pub fn remove(&mut self, value: &SortedArrayValue) -> bool {
        if let Some(index) = self.first_index(value) {
            self.data.remove(index);
            true
        } else {
            false
        }
    }
}