use std::any::Any;
use crate::{Point, Bounds};

#[derive(Debug)]
pub struct Node {
    pub ne: Option<Box<Node>>,
    pub nw: Option<Box<Node>>,
    pub se: Option<Box<Node>>,
    pub sw: Option<Box<Node>>,
    pub bounds: Option<Box<Bounds>>,
    pub point: Option<Point>,
    pub key: Option<Box<dyn Any + Send + Sync>>,
}

impl Node {
    /// Creates a new empty node
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

    /// Creates a new node with the specified bounds
    pub fn with_bounds(minx: f64, miny: f64, maxx: f64, maxy: f64) -> Option<Self> {
        let mut node = Self::new();
        let mut bounds = Bounds::new();
        bounds.extend(maxx, maxy);
        bounds.extend(minx, miny);
        node.bounds = Some(Box::new(bounds));
        Some(node)
    }

    /// Checks if the node is a pointer node (has all children but no point)
    pub fn is_pointer(&self) -> bool {
        self.nw.is_some() 
            && self.ne.is_some() 
            && self.sw.is_some() 
            && self.se.is_some() 
            && !self.is_leaf()
    }

    /// Checks if the node is empty (has no children and no point)
    pub fn is_empty(&self) -> bool {
        self.nw.is_none() 
            && self.ne.is_none() 
            && self.sw.is_none() 
            && self.se.is_none() 
            && !self.is_leaf()
    }

    /// Checks if the node is a leaf node (has a point)
    pub fn is_leaf(&self) -> bool {
        self.point.is_some()
    }

    /// Resets the node by clearing its point and key
    pub fn reset<F>(&mut self, key_free: F) 
    where 
        F: FnOnce(Option<Box<dyn Any + Send + Sync>>)
    {
        self.point = None;
        key_free(self.key.take());
    }
}

// Note: We don't need to implement Drop trait explicitly as Rust's ownership system
// will automatically handle the recursive freeing of child nodes and other resources