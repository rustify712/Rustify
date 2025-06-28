/// A node in a doubly linked list.
#[derive(Debug, Clone)]
pub struct QueueEntry<T: Clone> {
    pub data: T,
    pub prev: Option<Box<QueueEntry<T>>>,
    pub next: Option<Box<QueueEntry<T>>>,
}

/// A double-ended queue.
#[derive(Debug, Clone)]
pub struct Queue<T: Clone> {
    head: Option<Box<QueueEntry<T>>>, // Pointer to the head of the queue
    tail: Option<Box<QueueEntry<T>>>, // Pointer to the tail of the queue
}

/// Create a new double-ended queue.
///
/// # Returns
/// A new queue.
impl<T: Clone> Queue<T> {
    pub fn new() -> Self {
        Queue {
            head: None,
            tail: None,
        }
    }

    /// Query if any values are currently in a queue.
    ///
    /// # Returns
    /// - `true` if the queue is empty.
    /// - `false` if the queue is not empty.
    pub fn is_empty(&self) -> bool {
        self.head.is_none()
    }

    /// Add a value to the head of the queue.
    pub fn push_head(&mut self, data: T) {
        let mut new_entry = Box::new(QueueEntry {
            data,
            prev: None,
            next: self.head.take(),
        });

        if let Some(mut old_head) = new_entry.next.take() {
            old_head.prev = Some(new_entry.clone());
            new_entry.next = Some(old_head);
        }

        if self.tail.is_none() {
            self.tail = Some(new_entry.clone());
        }

        self.head = Some(new_entry);
    }

    /// Add a value to the tail of the queue.
    pub fn push_tail(&mut self, data: T) {
        let new_entry = Box::new(QueueEntry { data, prev: self.tail.take(), next: None });
        let new_entry = Some(new_entry);

        if let Some(tail) = &mut self.tail {
            tail.next = new_entry.clone();
        } else {
            self.head = new_entry.clone();
        }

        self.tail = new_entry;
    }

    /// Read a value from the tail of a queue, without removing it from the queue.
    ///
    /// # Returns
    /// - `Some(T)` if the queue is not empty.
    /// - `None` if the queue is empty.
    pub fn peek_tail(&self) -> Option<&T> {
        self.tail.as_ref().map(|entry| &entry.data)
    }

    /// Read value from the head of a queue, without removing it from the queue.
    ///
    /// # Returns
    /// - `Some(T)` if the queue is not empty.
    /// - `None` if the queue is empty.
    pub fn peek_head(&self) -> Option<&T> {
        self.head.as_ref().map(|entry| &entry.data)
    }

    /// Remove a value from the tail of the queue.
    ///
    /// # Returns
    /// - `Some(T)` if the value was removed successfully.
    /// - `None` if the queue is empty.
    pub fn pop_tail(&mut self) -> Option<T> {
        let tail = self.tail.take()?;
        self.tail = tail.prev;
        if self.tail.is_none() {
            self.head = None;
        } else {
            self.tail.as_mut().unwrap().next = None;
        }
        Some(tail.data)
    }
}

impl<T: Clone> Drop for Queue<T> {
    fn drop(&mut self) {
        while let Some(node) = self.head.take() {
            self.head = node.next;
        }
    }
}