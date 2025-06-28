/// Represents a node in the quadtree.
#[derive(Debug, Clone)]
pub struct QuadtreeNode {
    pub nw: Option<Box<QuadtreeNode>>,
    pub ne: Option<Box<QuadtreeNode>>,
    pub sw: Option<Box<QuadtreeNode>>,
    pub se: Option<Box<QuadtreeNode>>,
    pub point: Option<QuadtreePoint>,
    pub bounds: Option<QuadtreeBounds>,
    pub key: Option<Box<dyn std::any::Any>>,
}

impl QuadtreeNode {
    /// Creates a new, empty quadtree node.
    pub fn new() -> Option<QuadtreeNode> {
        Some(QuadtreeNode {
            nw: None,
            ne: None,
            sw: None,
            se: None,
            point: None,
            bounds: None,
            key: None,
        })
    }

    /// Creates a new quadtree node with the specified bounds.
    pub fn with_bounds(minx: f64, miny: f64, maxx: f64, maxy: f64) -> Option<QuadtreeNode> {
        let mut node = QuadtreeNode::new()?;
        node.bounds = Some(quadtree_bounds_new());
        quadtree_bounds_extend(node.bounds.as_mut().unwrap(), minx, miny);
        quadtree_bounds_extend(node.bounds.as_mut().unwrap(), maxx, maxy);
        Some(node)
    }

    /// Checks if the node has all four quadrants initialized.
    pub fn is_pointer(&self) -> bool {
        self.nw.is_some() && self.ne.is_some() && self.sw.is_some() && self.se.is_some() && !self.is_leaf()
    }

    /// Checks if the node is empty (i.e., it has no child nodes and no point).
    pub fn is_empty(&self) -> bool {
        self.nw.is_none() && self.ne.is_none() && self.sw.is_none() && self.se.is_none() && !self.is_leaf()
    }

    /// Checks if the node is a leaf (i.e., it contains a point).
    pub fn is_leaf(&self) -> bool {
        self.point.is_some()
    }

    /// Resets the node, freeing its point and key.
    pub fn reset(&mut self, key_free: &dyn Fn(Box<dyn std::any::Any>)) {
        if let Some(point) = self.point.take() {
            quadtree_point_free(point);
        }
        if let Some(key) = self.key.take() {
            key_free(key);
        }
    }
}

/// Frees the memory of a node and all of its children.
pub fn quadtree_node_free(node: &mut QuadtreeNode, key_free: &dyn Fn(Box<dyn std::any::Any>)) {
    if let Some(mut nw) = node.nw.take() {
        quadtree_node_free(&mut nw, &key_free);
    }
    if let Some(mut ne) = node.ne.take() {
        quadtree_node_free(&mut ne, &key_free);
    }
    if let Some(mut sw) = node.sw.take() {
        quadtree_node_free(&mut sw, &key_free);
    }
    if let Some(mut se) = node.se.take() {
        quadtree_node_free(&mut se, &key_free);
    }

    if let Some(bounds) = node.bounds.take() {
        quadtree_bounds_free(&mut bounds);
    }

    node.reset(&key_free);
}

/// Helper function to free a point in the quadtree.
pub fn quadtree_point_free(point: QuadtreePoint) {
    // In Rust, memory is managed automatically, so we don't need to explicitly free memory.
    // The point will be deallocated when it goes out of scope.
    // For educational purposes, we simulate the freeing action by explicitly dropping the point.
    drop(point);
}
