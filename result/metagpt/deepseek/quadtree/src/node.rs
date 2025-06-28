use std::fmt;
use super::{Bounds, Point};

#[derive(Debug)]
pub struct Node {
    pub ne: Option<Box<Node>>,
    pub nw: Option<Box<Node>>,
    pub se: Option<Box<Node>>,
    pub sw: Option<Box<Node>>,
    pub bounds: Option<Bounds>,
    pub point: Option<Point>,
    pub key: Option<Box<dyn std::any::Any>>,
}

impl Node {
    pub fn new() -> Self {
        Node {
            ne: None,
            nw: None,
            se: None,
            sw: None,
            bounds: None,
            point: None,
            key: None,
        }
    }

    pub fn with_bounds(minx: f64, miny: f64, maxx: f64, maxy: f64) -> Option<Self> {
        let mut node = Node::new();
        node.bounds = Some(Bounds::new());
        if let Some(bounds) = &mut node.bounds {
            bounds.extend(maxx, maxy);
            bounds.extend(minx, miny);
        }
        Some(node)
    }

    pub fn is_pointer(&self) -> bool {
        self.nw.is_some() && self.ne.is_some() && 
        self.sw.is_some() && self.se.is_some() && 
        !self.is_leaf()
    }

    pub fn is_empty(&self) -> bool {
        self.nw.is_none() && self.ne.is_none() && 
        self.sw.is_none() && self.se.is_none() && 
        !self.is_leaf()
    }

    pub fn is_leaf(&self) -> bool {
        self.point.is_some()
    }

    pub fn reset(&mut self, key_free: Option<fn(&mut Box<dyn std::any::Any>)>) {
        self.point = None;
        if let Some(free_fn) = key_free {
            if let Some(key) = &mut self.key {
                free_fn(key);
            }
        }
        self.key = None;
    }
}

impl Drop for Node {
    fn drop(&mut self) {
        // Rust's ownership system will automatically clean up child nodes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_new() {
        let node = Node::new();
        assert!(node.ne.is_none());
        assert!(node.nw.is_none());
        assert!(node.se.is_none());
        assert!(node.sw.is_none());
        assert!(node.bounds.is_none());
        assert!(node.point.is_none());
        assert!(node.key.is_none());
    }

    #[test]
    fn test_node_with_bounds() {
        let node = Node::with_bounds(0.0, 0.0, 10.0, 10.0).unwrap();
        assert!(node.bounds.is_some());
        let bounds = node.bounds.unwrap();
        assert_eq!(bounds.nw.x, 0.0);
        assert_eq!(bounds.se.x, 10.0);
    }

    #[test]
    fn test_node_checks() {
        let mut node = Node::new();
        assert!(node.is_empty());
        assert!(!node.is_pointer());
        assert!(!node.is_leaf());

        node.point = Some(Point::new(1.0, 2.0));
        assert!(node.is_leaf());
    }
}