// list.rs

/// A doubly-linked list implementation in Rust.
///
/// This implementation provides similar functionality to the C version,
/// with operations to manipulate the list.

use std::ptr::NonNull;

pub struct ListEntry<T> {
    data: T,
    prev: Option<NonNull<ListEntry<T>>>,
    next: Option<NonNull<ListEntry<T>>>,
}

pub struct List<T> {
    head: Option<NonNull<ListEntry<T>>>,
    tail: Option<NonNull<ListEntry<T>>>,
}

impl<T> List<T> {
    /// Create a new empty list.
    pub fn new() -> Self {
        List {
            head: None,
            tail: None,
        }
    }

    /// Free all entries in the list.
    pub fn free(&mut self) {
        let mut current = self.head;
        while let Some(mut node) = current {
            unsafe {
                current = node.as_ref().next;
                Box::from_raw(node.as_ptr());
            }
        }
        self.head = None;
        self.tail = None;
    }

    /// Prepend a value to the start of the list.
    pub fn prepend(&mut self, data: T) -> NonNull<ListEntry<T>> {
        let mut new_entry = Box::new(ListEntry {
            data,
            prev: None,
            next: self.head,
        });

        let new_entry_ptr = unsafe { NonNull::new_unchecked(Box::into_raw(new_entry)) };

        if let Some(head) = self.head {
            unsafe {
                head.as_mut().prev = Some(new_entry_ptr);
            }
        } else {
            self.tail = Some(new_entry_ptr);
        }

        self.head = Some(new_entry_ptr);
        new_entry_ptr
    }

    /// Append a value to the end of the list.
    pub fn append(&mut self, data: T) -> NonNull<ListEntry<T>> {
        let mut new_entry = Box::new(ListEntry {
            data,
            prev: self.tail,
            next: None,
        });

        let new_entry_ptr = unsafe { NonNull::new_unchecked(Box::into_raw(new_entry)) };

        if let Some(tail) = self.tail {
            unsafe {
                tail.as_mut().next = Some(new_entry_ptr);
            }
        } else {
            self.head = Some(new_entry_ptr);
        }

        self.tail = Some(new_entry_ptr);
        new_entry_ptr
    }

    /// Get the data from a list entry.
    pub fn data(entry: &NonNull<ListEntry<T>>) -> &T {
        unsafe { &entry.as_ref().data }
    }

    /// Set the data for a list entry.
    pub fn set_data(entry: &mut NonNull<ListEntry<T>>, value: T) {
        unsafe {
            entry.as_mut().data = value;
        }
    }

    /// Get the previous entry in the list.
    pub fn prev(entry: &NonNull<ListEntry<T>>) -> Option<NonNull<ListEntry<T>>> {
        unsafe { entry.as_ref().prev }
    }

    /// Get the next entry in the list.
    pub fn next(entry: &NonNull<ListEntry<T>>) -> Option<NonNull<ListEntry<T>>> {
        unsafe { entry.as_ref().next }
    }

    /// Get the nth entry in the list.
    pub fn nth_entry(&self, n: usize) -> Option<NonNull<ListEntry<T>>> {
        let mut current = self.head;
        for _ in 0..n {
            if let Some(node) = current {
                current = unsafe { node.as_ref().next };
            } else {
                return None;
            }
        }
        current
    }

    /// Get the data of the nth entry in the list.
    pub fn nth_data(&self, n: usize) -> Option<&T> {
        self.nth_entry(n).map(|entry| unsafe { &entry.as_ref().data })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_operations() {
        let mut list = List::new();
        let entry1 = list.append(1);
        let entry2 = list.append(2);
        let entry3 = list.prepend(0);

        assert_eq!(*List::data(&entry1), 1);
        assert_eq!(*List::data(&entry2), 2);
        assert_eq!(*List::data(&entry3), 0);

        assert_eq!(List::next(&entry3), Some(entry1));
        assert_eq!(List::prev(&entry1), Some(entry3));
        assert_eq!(List::next(&entry1), Some(entry2));
        assert_eq!(List::prev(&entry2), Some(entry1));

        assert_eq!(list.nth_data(0), Some(&0));
        assert_eq!(list.nth_data(1), Some(&1));
        assert_eq!(list.nth_data(2), Some(&2));
        assert_eq!(list.nth_data(3), None);

        list.free();
        assert_eq!(list.nth_data(0), None);
    }
}