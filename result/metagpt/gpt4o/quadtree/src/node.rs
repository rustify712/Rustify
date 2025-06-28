use std::rc::Rc;
use crate::quadtree_mod::{QuadtreeBounds, QuadtreePoint};

pub struct QuadtreeNode {
    pub nw: Option<Box<QuadtreeNode>>,
    pub ne: Option<Box<QuadtreeNode>>,
    pub sw: Option<Box<QuadtreeNode>>,
    pub se: Option<Box<QuadtreeNode>>,
    pub point: Option<Box<QuadtreePoint>>,
    pub bounds: Option<Box<QuadtreeBounds>>,
    pub key: Option<Box<dyn FnOnce()>>,
}

impl QuadtreeNode {
    pub fn new() -> Box<Self> {
        Box::new(QuadtreeNode {
            nw: None,
            ne: None,
            sw: None,
            se: None,
            point: None,
            bounds: None,
            key: None,
        })
    }

    pub fn with_bounds(minx: f64, miny: f64, maxx: f64, maxy: f64) -> Option<Box<Self>> {
        let mut node = QuadtreeNode::new();
        node.bounds = Some(Box::new(QuadtreeBounds {
            nw: Rc::new(QuadtreePoint { x: minx, y: miny }),
            se: Rc::new(QuadtreePoint { x: maxx, y: maxy }),
            width: maxx - minx,
            height: maxy - miny,
        }));
        Some(node)
    }

    pub fn is_pointer(&self) -> bool {
        self.nw.is_some() &&
        self.ne.is_some() &&
        self.sw.is_some() &&
        self.se.is_some() &&
        !self.is_leaf()
    }

    pub fn is_empty(&self) -> bool {
        self.nw.is_none() &&
        self.ne.is_none() &&
        self.sw.is_none() &&
        self.se.is_none() &&
        !self.is_leaf()
    }

    pub fn is_leaf(&self) -> bool {
        self.point.is_some()
    }

    pub fn reset(&mut self) {
        self.point = None;
        self.key = None;
    }
}

impl Drop for QuadtreeNode {
    fn drop(&mut self) {
        // Recursively free memory for the child nodes
        if let Some(nw) = self.nw.take() { drop(nw); }
        if let Some(ne) = self.ne.take() { drop(ne); }
        if let Some(sw) = self.sw.take() { drop(sw); }
        if let Some(se) = self.se.take() { drop(se); }

        if let Some(bounds) = self.bounds.take() {
            // Placeholder for bounds freeing logic
            drop(bounds);
        }
        self.reset();
    }
}