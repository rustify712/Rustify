// slist.rs

/// Singly-linked list implementation in Rust.
///
/// This implementation provides similar functionality to the C version,
/// with operations to manipulate the list.

pub struct SListEntry<T> {
    data: T,
    next: Option<Box<SListEntry<T>>>,
}

pub struct SList<T> {
    head: Option<Box<SListEntry<T>>>,
}

impl<T> SList<T> {
    /// Create a new empty singly-linked list.
    pub fn new() -> Self {
        SList { head: None }
    }

    /// Free all entries in the list.
    pub fn free(&mut self) {
        self.head = None;
    }

    /// Prepend a value to the start of the list.
    pub fn prepend(&mut self, data: T) {
        let new_entry = Box::new(SListEntry {
            data,
            next: self.head.take(),
        });
        self.head = Some(new_entry);
    }

    /// Append a value to the end of the list.
    pub fn append(&mut self, data: T) {
        let new_entry = Box::new(SListEntry { data, next: None });

        match self.head.as_mut() {
            Some(mut current) => {
                while let Some(next) = current.next.as_mut() {
                    current = next;
                }
                current.next = Some(new_entry);
            }
            None => {
                self.head = Some(new_entry);
            }
        }
    }

    /// Get the data from a list entry.
    pub fn data(entry: &SListEntry<T>) -> &T {
        &entry.data
    }

    /// Set the data for a list entry.
    pub fn set_data(entry: &mut SListEntry<T>, data: T) {
        entry.data = data;
    }

    /// Get the next entry in the list.
    pub fn next(entry: &SListEntry<T>) -> Option<&SListEntry<T>> {
        entry.next.as_deref()
    }

    /// Get the nth entry in the list.
    pub fn nth_entry(&self, n: usize) -> Option<&SListEntry<T>> {
        let mut current = self.head.as_deref();
        for _ in 0..n {
            current = current?.next.as_deref();
        }
        current
    }

    /// Get the data of the nth entry in the list.
    pub fn nth_data(&self, n: usize) -> Option<&T> {
        self.nth_entry(n).map(|entry| &entry.data)
    }

    /// Get the length of the list.
    pub fn length(&self) -> usize {
        let mut length = 0;
        let mut current = self.head.as_deref();
        while let Some(entry) = current {
            length += 1;
            current = entry.next.as_deref();
        }
        length
    }

    /// Convert the list to an array.
    pub fn to_array(&self) -> Vec<&T> {
        let mut array = Vec::new();
        let mut current = self.head.as_deref();
        while let Some(entry) = current {
            array.push(&entry.data);
            current = entry.next.as_deref();
        }
        array
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slist_operations() {
        let mut list = SList::new();
        list.append(1);
        list.append(2);
        list.prepend(0);

        assert_eq!(list.nth_data(0), Some(&0));
        assert_eq!(list.nth_data(1), Some(&1));
        assert_eq!(list.nth_data(2), Some(&2));
        assert_eq!(list.nth_data(3), None);

        assert_eq!(list.length(), 3);

        let array = list.to_array();
        assert_eq!(array, vec![&0, &1, &2]);

        list.free();
        assert_eq!(list.length(), 0);
    }
}