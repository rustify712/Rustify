/// Heap type. If a heap is a min heap (`BinomialHeapType::Min`), the
/// values with the lowest priority are stored at the top of the heap and
/// will be the first returned. If a heap is a max heap
/// (`BinomialHeapType::Max`), the values with the greatest priority
/// are stored at the top of the heap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinomialHeapType {
    /// A minimum heap.
    Min,
    /// A maximum heap.
    Max,
}

use std::rc::Rc;
use std::cell::RefCell;

/// 表示一个二项树的结构体。
///
/// 每个二项树包含一个值 `value`，树的阶数 `order`，引用计数 `refcount`，
/// 以及一个子树的向量 `subtrees`。
pub struct BinomialTree<T> {
    /// 存储在树中的值。
    pub value: T,
    /// 树的阶数，表示树的深度。
    pub order: u16,
    /// 引用计数，用于管理树的生命周期。
    pub refcount: u16,
    /// 子树的向量，每个子树都是一个 `Rc<RefCell<BinomialTree<T>>>`。
    pub subtrees: Vec<Rc<RefCell<BinomialTree<T>>>>,
}

/// Decrements the reference count of a binomial tree.
/// If the reference count reaches zero, the tree and its subtrees are dropped.
impl<T> BinomialTree<T> {
    pub fn unref(tree: Rc<RefCell<BinomialTree<T>>>) {
        let mut tree_borrow = tree.borrow_mut();
        tree_borrow.refcount -= 1;

        if tree_borrow.refcount == 0 {
            for subtree in tree_borrow.subtrees.iter() {
                Self::unref(subtree.clone());
            }
            tree_borrow.subtrees.clear();
        }
    }
}

/// 表示一个二项堆的结构体。
///
/// 每个二项堆包含一个堆类型 `heap_type`，一个存储二项树根节点的向量 `roots`，
/// 以及堆中存储的值的数量 `num_values`。
pub struct BinomialHeap<T: Ord> {
    /// 堆的类型，表示最小堆或最大堆。
    pub heap_type: BinomialHeapType,
    /// 存储二项树根节点的向量，每个根节点都是一个 `Rc<RefCell<BinomialTree<T>>>`。
    pub roots: Vec<Rc<RefCell<BinomialTree<T>>>>,
    /// 堆中存储的值的数量。
    pub num_values: usize,
}

impl<T: Ord + Clone> BinomialHeap<T> {
    /// Create a new binomial heap.
    ///
    /// # Arguments
    ///
    /// * `heap_type` - The type of heap: min heap or max heap.
    ///
    /// # Returns
    ///
    /// A new binomial heap.
    pub fn new(heap_type: BinomialHeapType) -> Self {
        Self {
            heap_type,
            roots: Vec::new(),
            num_values: 0,
        }
    }

    /// Find the number of values stored in a binomial heap.
    pub fn num_entries(&self) -> usize {
        self.num_values
    }

    /// Merges two binomial heaps into one.
    ///
    /// This function takes two heaps and merges them into a new heap.
    /// The new heap will contain all the trees from both heaps, and the
    /// roots of the new heap will be sorted by the order of the trees.
    ///
    /// # Arguments
    ///
    /// * `other` - The other heap to merge with.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the merge was successful, or `Err(())` if memory allocation failed.
    pub fn merge(&mut self, other: &mut BinomialHeap<T>) -> Result<(), ()> {
        let max_length = self.roots.len().max(other.roots.len()) + 1;
        let mut new_roots = vec![None; max_length];
        let mut carry = None;

        for i in 0..max_length {
            let mut vals = Vec::with_capacity(3);

            if i < self.roots.len() {
                if let Some(tree) = self.roots.get(i) {
                    vals.push(tree.clone());
                }
            }

            if i < other.roots.len() {
                if let Some(tree) = other.roots.get(i) {
                    vals.push(tree.clone());
                }
            }

            if let Some(carry_tree) = carry {
                vals.push(carry_tree);
            }

            let num_vals = vals.len();

            if num_vals & 1 != 0 {
                new_roots[i] = Some(vals.pop().unwrap());
            }

            if num_vals & 2 != 0 {
                carry = binomial_tree_merge(vals[0].clone(), vals[1].clone());
            } else {
                carry = None;
            }
        }

        self.roots = new_roots.into_iter().filter_map(|x| x).collect();
        self.num_values += other.num_values;
        Ok(())
    }

    /// Insert a value into a binomial heap.
    ///
    /// # Arguments
    ///
    /// * `value` - The value to insert.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the entry was added, or `Err(())` if memory allocation failed.
    pub fn insert(&mut self, value: T) -> Result<(), ()> {
        // Allocate an order 0 tree for storing the new value
        let new_tree = Rc::new(RefCell::new(BinomialTree {
            value,
            order: 0,
            refcount: 1,
            subtrees: Vec::new(),
        }));

        // Build a fake heap structure for merging
        let mut fake_heap = BinomialHeap::new(self.heap_type);
        fake_heap.roots.push(new_tree.clone());
        fake_heap.num_values = 1;

        // Perform the merge
        self.merge(&mut fake_heap)?;

        // Update the number of values
        self.num_values += 1;

        // Remove reference to the new tree
        BinomialTree::unref(new_tree);

        Ok(())
    }
}

/// Undoes a merge operation by unreferencing all trees in the new roots array and freeing the array.
pub fn binomial_heap_merge_undo<T>(new_roots: Vec<Rc<RefCell<BinomialTree<T>>>>) {
    for tree in new_roots {
        BinomialTree::unref(tree);
    }
}

/// Merges two binomial trees into a new tree.
///
/// This function takes two trees and merges them into a new tree.
/// The tree with the smaller root value becomes the new root,
/// and the other tree becomes the last subtree of the new tree.
///
/// # Arguments
///
/// * `tree1` - The first binomial tree.
/// * `tree2` - The second binomial tree.
///
/// # Returns
///
/// A new binomial tree resulting from the merge, or `None` if memory allocation failed.
pub fn binomial_tree_merge<T: Ord + Clone>(tree1: Rc<RefCell<BinomialTree<T>>>, tree2: Rc<RefCell<BinomialTree<T>>>) -> Option<Rc<RefCell<BinomialTree<T>>>> {
    let mut tree1_borrow = tree1.borrow_mut();
    let mut tree2_borrow = tree2.borrow_mut();

    // Determine which tree has the smaller root value
    let (smaller_tree, larger_tree) = if tree1_borrow.value <= tree2_borrow.value {
        (tree1.clone(), tree2.clone())
    } else {
        (tree2.clone(), tree1.clone())
    };

    // Create a new tree
    let new_tree = Rc::new(RefCell::new(BinomialTree {
        value: smaller_tree.borrow().value.clone(),
        order: smaller_tree.borrow().order + 1,
        refcount: 0,
        subtrees: Vec::with_capacity(smaller_tree.borrow().order as usize + 1),
    }));

    // Copy subtrees of the smaller tree
    new_tree.borrow_mut().subtrees.extend_from_slice(&smaller_tree.borrow().subtrees);
    new_tree.borrow_mut().subtrees.push(larger_tree.clone());

    // Increment reference count for each subtree
    for subtree in new_tree.borrow().subtrees.iter() {
        subtree.borrow_mut().refcount += 1;
    }

    Some(new_tree)
}

/// Destroy a binomial heap.
impl<T: Ord> BinomialHeap<T> {
    /// Frees all resources associated with the heap.
    pub fn free(&mut self) {
        // Unreference all trees in the heap. This should free all subtrees.
        for root in self.roots.iter() {
            BinomialTree::unref(root.clone());
        }
        // Clear the roots vector.
        self.roots.clear();
    }
}