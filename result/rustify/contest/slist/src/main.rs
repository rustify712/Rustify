/// Value stored in a list.
type SListValue<T> = T;

/// Represents an entry in a singly-linked list.
/// The empty list is represented by `None`.
#[derive(Debug, Clone)]
pub struct SListEntry<T: Clone> {
    pub data: T,
    pub next: Option<Box<SListEntry<T>>>,
}

impl<T: PartialEq + Clone> SListEntry<T> {
    /// Append a value to the end of a list.
    ///
    /// # Arguments
    ///
    /// * `data` - The value to append.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the value was successfully appended, or `Err(())` if memory allocation failed.
    pub fn append(&mut self, data: T) -> Result<(), ()> {
        let new_entry = Box::new(SListEntry { data, next: None });
        match self.next.take() {
            Some(mut current) => {
                while current.next.is_some() {
                    current = current.next.take().unwrap();
                }
                current.next = Some(new_entry);
            }
            None => {
                self.next = Some(new_entry);
            }
        }
        Ok(())
    }

    /// Set the value at a list entry.
    /// If the list entry is `None`, nothing is done.
    pub fn set_data(&mut self, data: T) {
        self.data = data;
    }

    /// Find the entry for a particular value in a list.
    ///
    /// # Arguments
    ///
    /// * `data` - The value to search for.
    ///
    /// # Returns
    ///
    /// The list entry of the value being searched for, or `None` if not found.
    pub fn find_entry(&self, data: &T) -> Option<&SListEntry<T>> {
        let mut current = Some(self);
        while let Some(entry) = current {
            if entry.data == *data {
                return Some(entry);
            }
            current = entry.next.as_ref().map(|boxed| boxed.as_ref());
        }
        None
    }

    /// Remove an entry from the list.
    ///
    /// # Arguments
    ///
    /// * `entry` - The entry to remove.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the entry was successfully removed, or `Err(())` if the entry was not found.
    pub fn remove_entry(&mut self, entry: &mut SListEntry<T>) -> Result<(), ()> {
        let mut current = self;
        while let Some(mut node) = current.next.take() {
            if node.as_mut() as *mut SListEntry<T> == entry as *mut SListEntry<T> {
                current.next = node.next.take();
                return Ok(());
            }
            current.next = Some(node);
            current = current.next.as_mut().unwrap();
        }
        Err(())
    }

    /// Retrieve the value stored at a list entry.
    pub fn data(&self) -> &T {
        &self.data
    }

    /// Retrieve the next entry in a list.
    ///
    /// # Returns
    ///
    /// The next entry in the list, or `None` if this is the last entry.
    pub fn next(&self) -> Option<&SListEntry<T>> {
        self.next.as_deref()
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
    pub fn nth_entry(&self, n: usize) -> Option<&SListEntry<T>> {
        let mut current = self;
        for _ in 0..n {
            if let Some(next) = current.next.as_ref() {
                current = next;
            } else {
                return None;
            }
        }
        Some(current)
    }

    /// Retrieve the value stored at a specified index in the list.
    ///
    /// # Arguments
    ///
    /// * `n` - The index into the list.
    ///
    /// # Returns
    ///
    /// The value stored at the specified index, or `None` if out of range.
    pub fn nth_data(&self, n: usize) -> Option<&T> {
        self.nth_entry(n).map(|entry| &entry.data)
    }
}

/// Structure used to iterate over a list.
pub struct SListIterator<'a, T: Clone> {
    /// Pointer to the previous node's `next` field.
    pub prev_next: Option<&'a mut Option<Box<SListEntry<T>>>>,
    /// Pointer to the current node.
    pub current: Option<&'a SListEntry<T>>,
}

/// Calculate the length of a singly-linked list.
pub fn slist_length<T: Clone>(list: &Option<Box<SListEntry<T>>>) -> usize {
    let mut count = 0;
    let mut current = list;
    while let Some(node) = current {
        count += 1;
        current = &node.next;
    }
    count
}

/// Convert a singly-linked list to a Vec containing all values in the list.
pub fn slist_to_array<T: Clone>(list: &Option<Box<SListEntry<T>>>) -> Vec<T> {
    let mut result = Vec::new();
    let mut current = list;
    while let Some(node) = current {
        result.push(node.data.clone());
        current = &node.next;
    }
    result
}

/// Sort a singly-linked list using the quick sort algorithm.
pub fn sort<T: Ord + Clone>(list: &mut Option<Box<SListEntry<T>>>) {
    if let Some(mut pivot) = list.take() {
        let mut less_list = None;
        let mut more_list = None;
        let mut current = pivot.next.take();

        while let Some(mut node) = current {
            current = node.next.take();
            if node.data < pivot.data {
                node.next = less_list;
                less_list = Some(node);
            } else {
                node.next = more_list;
                more_list = Some(node);
            }
        }

        sort(&mut less_list);
        sort(&mut more_list);

        *list = less_list;
        if let Some(mut end) = list.as_mut() {
            while end.next.is_some() {
                end = end.next.as_mut().unwrap();
            }
            end.next = Some(pivot);
        } else {
            *list = Some(pivot);
        }
        if let Some(mut end) = list.as_mut() {
            while end.next.is_some() {
                end = end.next.as_mut().unwrap();
            }
            end.next = more_list;
        }
    }
}

/// Prepend a value to the start of a list.
///
/// # Arguments
///
/// * `list` - A mutable reference to the list to prepend to.
/// * `data` - The value to prepend.
///
/// # Returns
///
/// The new entry in the list, or `None` if it was not possible to allocate a new entry.
pub fn prepend<T: Clone>(list: &mut Option<Box<SListEntry<T>>>, data: T) -> Option<Box<SListEntry<T>>> {
    let new_entry = Box::new(SListEntry { data, next: list.take() });
    *list = Some(new_entry.clone());
    Some(new_entry)
}

/// Remove all occurrences of a particular value from a list.
///
/// # Arguments
///
/// * `data` - The value to remove from the list.
///
/// # Returns
///
/// The number of entries removed from the list.
pub fn remove_data<T: PartialEq + Clone>(list: &mut Option<Box<SListEntry<T>>>, data: T) -> usize {
    let mut entries_removed = 0;
    let mut prev_next = list;

    while let Some(mut entry) = prev_next.take() {
        if entry.data == data {
            // Data found, so remove this entry
            *prev_next = entry.next.take();
            entries_removed += 1;
        } else {
            // Advance to the next entry
            *prev_next = Some(entry);
            prev_next = &mut prev_next.as_mut().unwrap().next;
        }
    }

    entries_removed
}

impl<T: Clone> Drop for SListEntry<T> {
    fn drop(&mut self) {
        let mut current = self.next.take();
        while let Some(mut node) = current {
            current = node.next.take();
        }
    }
}

/// Initialise a `SListIterator` structure to iterate over a list.
///
/// # Arguments
///
/// * `list` - A mutable reference to the list to iterate over.
pub fn iterate<T: Clone>(list: &mut Option<Box<SListEntry<T>>>) -> SListIterator<T> {
    SListIterator {
        prev_next: Some(list), // 将 list 包装在 Some 中
        current: None,
    }
}