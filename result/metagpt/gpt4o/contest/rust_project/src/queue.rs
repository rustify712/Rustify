// queue.rs

/// Queue implementation in Rust.
///
/// This implementation provides similar functionality to the C version,
/// supporting operations on a double-ended queue.

use std::ptr::NonNull;

pub struct QueueEntry<T> {
    data: T,
    prev: Option<NonNull<QueueEntry<T>>>,
    next: Option<NonNull<QueueEntry<T>>>,
}

pub struct Queue<T> {
    head: Option<NonNull<QueueEntry<T>>>,
    tail: Option<NonNull<QueueEntry<T>>>,
}

impl<T> Queue<T> {
    /// Create a new empty queue.
    pub fn new() -> Self {
        Queue {
            head: None,
            tail: None,
        }
    }

    /// Free all entries in the queue.
    pub fn free(&mut self) {
        while !self.is_empty() {
            self.pop_head();
        }
    }

    /// Check if the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.head.is_none()
    }

    /// Push a value onto the head of the queue.
    pub fn push_head(&mut self, data: T) -> NonNull<QueueEntry<T>> {
        let mut new_entry = Box::new(QueueEntry {
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

    /// Pop a value from the head of the queue.
    pub fn pop_head(&mut self) -> Option<T> {
        self.head.map(|node| {
            let node = unsafe { Box::from_raw(node.as_ptr()) };
            self.head = node.next;

            if let Some(head) = self.head {
                unsafe {
                    head.as_mut().prev = None;
                }
            } else {
                self.tail = None;
            }

            node.data
        })
    }

    /// Peek at the value at the head of the queue.
    pub fn peek_head(&self) -> Option<&T> {
        self.head.map(|node| unsafe { &node.as_ref().data })
    }

    /// Push a value onto the tail of the queue.
    pub fn push_tail(&mut self, data: T) -> NonNull<QueueEntry<T>> {
        let mut new_entry = Box::new(QueueEntry {
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

    /// Pop a value from the tail of the queue.
    pub fn pop_tail(&mut self) -> Option<T> {
        self.tail.map(|node| {
            let node = unsafe { Box::from_raw(node.as_ptr()) };
            self.tail = node.prev;

            if let Some(tail) = self.tail {
                unsafe {
                    tail.as_mut().next = None;
                }
            } else {
                self.head = None;
            }

            node.data
        })
    }

    /// Peek at the value at the tail of the queue.
    pub fn peek_tail(&self) -> Option<&T> {
        self.tail.map(|node| unsafe { &node.as_ref().data })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_queue_operations() {
        let mut queue = Queue::new();
        let entry1 = queue.push_head(1);
        let entry2 = queue.push_tail(2);
        let entry3 = queue.push_head(0);

        assert_eq!(queue.peek_head(), Some(&0));
        assert_eq!(queue.peek_tail(), Some(&2));

        assert_eq!(queue.pop_head(), Some(0));
        assert_eq!(queue.pop_tail(), Some(2));
        assert_eq!(queue.pop_head(), Some(1));
        assert_eq!(queue.pop_head(), None);

        queue.free();
        assert!(queue.is_empty());
    }
}