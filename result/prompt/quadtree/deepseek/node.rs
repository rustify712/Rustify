// quadtree.rs

impl QuadtreeNode {
    // Check if the node is a pointer (has children)
    fn is_pointer(&self) -> bool {
        self.nw.is_some()
            && self.ne.is_some()
            && self.sw.is_some()
            && self.se.is_some()
            && !self.is_leaf()
    }

    // Check if the node is empty (no children or point)
    fn is_empty(&self) -> bool {
        self.nw.is_none()
            && self.ne.is_none()
            && self.sw.is_none()
            && self.se.is_none()
            && !self.is_leaf()
    }

    // Check if the node is a leaf (contains a point)
    fn is_leaf(&self) -> bool {
        self.point.is_some()
    }

    // Reset the node by freeing its point and key
    fn reset(&mut self, key_free: Option<fn(&mut Box<dyn std::any::Any>)>) {
        if let Some(point) = self.point.take() {
            // No explicit free needed for point in Rust (ownership handles it)
        }
        if let Some(key) = self.key.take() {
            if let Some(free_fn) = key_free {
                free_fn(Box::new(key));
            }
        }
    }

    // Create a new QuadtreeNode instance
    fn new() -> Self {
        QuadtreeNode {
            ne: None,
            nw: None,
            se: None,
            sw: None,
            bounds: None,
            point: None,
            key: None,
        }
    }

    // Create a new QuadtreeNode with bounds
    fn with_bounds(minx: f64, miny: f64, maxx: f64, maxy: f64) -> Option<Box<Self>> {
        let mut node = QuadtreeNode::new();
        node.bounds = Some(Box::new(QuadtreeBounds::new()));
        if let Some(bounds) = &mut node.bounds {
            bounds.extend(maxx, maxy);
            bounds.extend(minx, miny);
        }
        Some(Box::new(node))
    }

    // Free the node and its children recursively
    fn free(node: Box<Self>, key_free: Option<fn(&mut Box<dyn std::any::Any>)>) {
        if let Some(nw) = node.nw {
            QuadtreeNode::free(nw, key_free);
        }
        if let Some(ne) = node.ne {
            QuadtreeNode::free(ne, key_free);
        }
        if let Some(sw) = node.sw {
            QuadtreeNode::free(sw, key_free);
        }
        if let Some(se) = node.se {
            QuadtreeNode::free(se, key_free);
        }

        // Free bounds and reset the node
        if let Some(bounds) = node.bounds {
            // No explicit free needed for bounds in Rust (ownership handles it)
        }
        node.reset(key_free);
        // No explicit free needed for node in Rust (ownership handles it)
    }
}