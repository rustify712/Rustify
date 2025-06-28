/// Represents an entry in a doubly-linked list.
#[derive(Debug, PartialEq, Clone)]
pub struct ListEntry<T>
where
    T: Clone + PartialEq, // Add `T: PartialEq` constraint
{
    pub data: T,
    pub prev: Option<Box<ListEntry<T>>>, // Pointer to the previous entry
    pub next: Option<Box<ListEntry<T>>>, // Pointer to the next entry
}

impl<T> ListEntry<T>
where
    T: Clone + PartialEq, // Add `T: PartialEq` constraint
{
    /// Creates a new `ListEntry` with the given data.
    pub fn new(data: T) -> Self {
        ListEntry { data, prev: None, next: None }
    }

    /// Prepend a value to the start of a list.
    ///
    /// # Arguments
    ///
    /// * `data` - The value to prepend.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the value was successfully prepended, or `Err(())` if memory allocation failed.
    pub fn prepend(&mut self, data: T) -> Result<(), ()> {
        let mut new_entry = Box::new(ListEntry::new(data)); // Declare `new_entry` as mutable
        if let Some(mut old_head) = self.next.take() {
            old_head.prev = Some(new_entry.clone());
            new_entry.next = Some(old_head);
        }
        self.next = Some(new_entry);
        Ok(())
    }

    /// Retrieve the previous entry in a list.
    ///
    /// # Arguments
    ///
    /// * `&self` - A reference to the current list entry.
    ///
    /// # Returns
    ///
    /// An `Option<&ListEntry<T>>` representing the previous entry in the list, or `None` if this was the first entry in the list.
    pub fn prev(&self) -> Option<&ListEntry<T>> {
        self.prev.as_deref()
    }

    /// Retrieve the next entry in a list.
    ///
    /// # Arguments
    ///
    /// * `&self` - A reference to the current list entry.
    ///
    /// # Returns
    ///
    /// An `Option<&ListEntry<T>>` representing the next entry in the list, or `None` if this was the last entry in the list.
    pub fn next(&self) -> Option<&ListEntry<T>> {
        self.next.as_deref()
    }

    /// Removes the current entry from the list.
    ///
    /// # Returns
    /// Returns `true` if the entry was successfully removed, otherwise `false`.
    pub fn remove_entry(&mut self) -> bool {
        if let Some(mut prev) = self.prev.take() {
            prev.next = self.next.take();
        } else if let Some(mut next) = self.next.take() {
            next.prev = None;
        } else {
            return false;
        }
        true
    }

    /// Find the entry for a particular value in a list.
    ///
    /// # Arguments
    ///
    /// * `data` - The value to search for.
    ///
    /// # Returns
    ///
    /// The list entry of the item being searched for, or `None` if not found.
    pub fn find_data(&self, data: &T) -> Option<&ListEntry<T>> {
        let mut current = self.next.as_ref();
        while let Some(entry) = current {
            if entry.data == *data {
                return Some(entry);
            }
            current = entry.next.as_ref();
        }
        None
    }

    /// Remove all occurrences of a particular value from the list.
    ///
    /// # Arguments
    ///
    /// * `data` - The value to remove from the list.
    ///
    /// # Returns
    ///
    /// The number of entries removed from the list.
    pub fn remove_data(&mut self, data: &T) -> usize {
        let mut entries_removed = 0;
        let mut current = self.next.take();
        self.next = None;

        while let Some(mut node) = current {
            current = node.next.take();

            if node.data == *data {
                if let Some(mut prev) = node.prev.take() {
                    prev.next = current.take(); // 直接使用 current 的值，避免双重装箱
                }
                if let Some(ref mut next) = current {
                    next.prev = node.prev;
                }
                entries_removed += 1;
            } else {
                node.next = current.take();
                node.prev = None;
                self.next = Some(node);
            }
        }

        entries_removed
    }

    /// Retrieve the entry at a specified index in a list.
    ///
    /// # Arguments
    ///
    /// * `n` - The index into the list.
    ///
    /// # Returns
    ///
    /// The entry at the specified index, or `None` if out of range.
    pub fn nth_entry(&self, n: usize) -> Option<&ListEntry<T>> {
        let mut entry = self;
        for _ in 0..n {
            entry = entry.next.as_deref()?;
        }
        Some(entry)
    }

    /// Retrieve the value at a specified index in the list.
    ///
    /// # Arguments
    ///
    /// * `n` - The index into the list.
    ///
    /// # Returns
    ///
    /// The value at the specified index, or `None` if out of range.
    pub fn nth_data(&self, n: usize) -> Option<&T> {
        self.nth_entry(n).map(|entry| &entry.data)
    }
}

/// Retrieve the value at a list entry.
/// If the list entry is `None`, returns `None`.
/// Otherwise, returns the value stored at the list entry.
impl<T> ListEntry<T>
where
    T: Clone + PartialEq, // Add `T: PartialEq` constraint
{
    pub fn data(&self) -> Option<&T> {
        Some(&self.data)
    }
}

/// Set the value at a list entry. The value provided will be written to the
/// given list entry. If the list entry is `None`, nothing is done.
///
/// # Arguments
///
/// * `listentry` - A mutable reference to the list entry.
/// * `value` - The value to set.
pub fn set_data<T: Clone + PartialEq>(listentry: Option<&mut ListEntry<T>>, value: T) {
    if let Some(entry) = listentry {
        entry.data = value;
    }
}

/// Sorts the list using the quicksort algorithm.
pub fn list_sort<T: PartialOrd + Clone + PartialEq>(list: &mut Option<Box<ListEntry<T>>>) {
    list_sort_internal(list);
}

/// Internal function to perform the quicksort algorithm.
fn list_sort_internal<T: PartialOrd + Clone + PartialEq>(list: &mut Option<Box<ListEntry<T>>>) {
    if let Some(mut pivot) = list.take() {
        let mut less_list = None;
        let mut more_list = None;
        let mut rover = pivot.next.take();

        while let Some(mut entry) = rover {
            rover = entry.next.take();
            if entry.data < pivot.data {
                let entry_clone = entry.clone(); // Clone `entry` to avoid borrow conflicts
                entry.next = less_list.take();
                if let Some(ref mut l) = entry.next {
                    l.prev = Some(entry_clone);
                }
                less_list = Some(entry);
            } else {
                let entry_clone = entry.clone(); // Clone `entry` to avoid borrow conflicts
                entry.next = more_list.take();
                if let Some(ref mut m) = entry.next {
                    m.prev = Some(entry_clone);
                }
                more_list = Some(entry);
            }
        }

        list_sort_internal(&mut less_list);
        list_sort_internal(&mut more_list);

        *list = less_list;
        if let Some(ref mut l) = list {
            l.prev = None;
        }

        let pivot_clone = pivot.clone(); // Clone `pivot` to avoid borrow conflicts
        pivot.next = more_list;
        if let Some(ref mut m) = pivot.next {
            m.prev = Some(pivot_clone);
        }

        *list = Some(pivot);
    }
}

/// Free an entire list.
pub fn list_free<T: Clone + PartialEq>(list: Option<Box<ListEntry<T>>>) {
    let mut current = list;
    while let Some(mut entry) = current {
        current = entry.next.take();
    }
}

/// Structure used to iterate over a list.
pub struct ListIterator<'a, T: Clone + PartialEq> {
    prev_next: Option<&'a mut Option<Box<ListEntry<T>>>>,
    current: Option<&'a mut ListEntry<T>>,
}

impl<'a, T: Clone + PartialEq> ListIterator<'a, T> {
    /// Creates a new `ListIterator` starting from the given list.
    pub fn new(list: &'a mut Option<Box<ListEntry<T>>>) -> Self {
        ListIterator {
            prev_next: Some(list),
            current: None,
        }
    }

    /// Determine if there are more values in the list to iterate over.
    ///
    /// # Returns
    ///
    /// Returns `true` if there are more values in the list to iterate over, otherwise `false`.
    pub fn has_more(&self) -> bool {
        if self.current.is_none() || self.current.as_deref() != self.prev_next.as_deref().and_then(|x| x.as_deref()) {
            self.prev_next.as_deref().is_some()
        } else {
            self.current.as_ref().unwrap().next.is_some()
        }
    }

    /// Delete the current entry in the list (the value last returned from
    /// list_iter_next).
    pub fn remove_current(&mut self) {
        if let Some(current) = self.current.take() {
            if let Some(prev_next) = self.prev_next.take() {
                *prev_next = current.next.take();
                if let Some(next) = current.next.as_mut() {
                    next.prev = current.prev.take();
                }
            }
            self.current = None;
        }
    }
}

/// Initialise a `ListIterator` structure to iterate over a list.
///
/// # Arguments
///
/// * `list` - A mutable reference to the list to iterate over.
pub fn list_iterate<T: Clone + PartialEq>(list: &mut Option<Box<ListEntry<T>>>) -> ListIterator<T> {
    ListIterator::new(list)
}