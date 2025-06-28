/// 定义一个 `SortedArray` 结构体，用于存储排序后的元素。
///
/// `SortedArray` 是一个自动调整大小的数组，存储的元素保持排序状态。
/// 所有操作都保持数组的排序属性。大多数操作的时间复杂度为 O(n)，
/// 但搜索操作的时间复杂度为 O(log n)。
#[derive(Debug, Clone)]
pub struct SortedArray<T>
where
    T: PartialEq + Ord,
{
    /// 实际存储数据的数组。
    data: Vec<T>,
}

/// 找到值 `data` 在 `SortedArray` 中的最后一个索引位置。
/// 使用二分查找来确定这个位置。
///
/// # 参数
/// * `data` - 要查找的值。
/// * `left` - 查找范围的左边界。
/// * `right` - 查找范围的右边界。
///
/// # 返回值
/// 返回 `data` 在 `SortedArray` 中的最后一个索引位置。
fn last_index<T: Ord>(data: &T, array: &[T], left: usize, right: usize) -> usize {
    let mut left = left;
    let mut right = right;
    let mut index = right;

    while left < right {
        index = (left + right) / 2;

        if data <= &array[index] {
            left = index + 1;
        } else {
            right = index;
        }
    }

    index
}

/// 分配一个新的 `SortedArray` 结构体。
///
/// 该函数用于初始化一个新的 `SortedArray`，并为其分配内存。
///
/// # 参数
///
/// * `length` - 初始容量，如果为 0，则使用默认值 16。
///
/// # 返回值
///
/// 返回一个新的 `SortedArray`，如果分配失败则返回 `None`。
pub fn new<T: PartialEq + Ord>(length: usize) -> Option<SortedArray<T>> {
    let capacity = if length == 0 { 16 } else { length };
    Some(SortedArray { data: Vec::with_capacity(capacity) })
}

impl<T> SortedArray<T>
where
    T: PartialEq + Ord,
{
    /// 从 `SortedArray` 中移除一个范围的元素，同时保持数组的排序属性。
    pub fn remove_range(&mut self, index: usize, length: usize) {
        if index + length > self.data.len() {
            return;
        }
        self.data.drain(index..index + length);
    }

    /// 从 `SortedArray` 中移除指定索引的单个元素。
    pub fn remove(&mut self, index: usize) {
        if index < self.data.len() {
            self.data.remove(index);
        }
    }

    /// Retrieves the element at the specified index from the sorted array.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the element to retrieve.
    ///
    /// # Returns
    ///
    /// An `Option<&T>` containing the element at the specified index, or `None` if the index is out of bounds.
    pub fn get(&self, index: usize) -> Option<&T> {
        self.data.get(index)
    }

    /// 将一个值插入到 `SortedArray` 中，并保持数组的排序属性。
    /// 如果插入成功，返回 `Ok(())`，否则返回 `Err`。
    pub fn insert(&mut self, value: T) -> Result<(), &'static str> {
        match self.data.binary_search(&value) {
            Ok(index) => {
                self.data.insert(index, value);
            }
            Err(index) => {
                self.data.insert(index, value);
            }
        }
        Ok(())
    }

    /// 查找值 `data` 在 `SortedArray` 中的索引。
    /// 使用二分查找来确定这个位置。
    ///
    /// # 参数
    /// * `data` - 要查找的值。
    ///
    /// # 返回值
    /// 返回 `data` 在 `SortedArray` 中的索引，如果未找到则返回 `None`。
    pub fn index_of(&self, data: &T) -> Option<usize> {
        // 使用二分查找来查找值
        self.data.binary_search_by(|probe| probe.cmp(data)).ok()
    }
}