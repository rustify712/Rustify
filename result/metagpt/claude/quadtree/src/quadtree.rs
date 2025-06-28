use std::any::Any;
use crate::{Node, Point};

pub struct QuadTree {
    pub root: Option<Box<Node>>,
    pub key_free: Option<Box<dyn Fn(Option<Box<dyn Any + Send + Sync>>)>>,
    pub length: usize,
}

impl QuadTree {
    /// Creates a new quadtree with the specified bounds
    pub fn new(minx: f64, miny: f64, maxx: f64, maxy: f64) -> Option<Self> {
        let root = Node::with_bounds(minx, miny, maxx, maxy)?;
        Some(QuadTree {
            root: Some(Box::new(root)),
            key_free: None,
            length: 0,
        })
    }

    /// Inserts a point with an associated key into the quadtree
    pub fn insert(&mut self, x: f64, y: f64, key: Box<dyn Any + Send + Sync>) -> bool {
        let point = Point::new(x, y);
        
        if let Some(root) = &self.root {
            // Check if point is within bounds
            if !Self::node_contains(root, &point) {
                return false;
            }
            
            if self.insert_internal(&mut self.root, point, key) {
                self.length += 1;
                true
            } else {
                false
            }
        } else {
            false
        }
    }

    /// Searches for a point in the quadtree
    pub fn search(&self, x: f64, y: f64) -> Option<&Point> {
        self.find_internal(self.root.as_ref()?, x, y)
    }

    // Private helper methods
    fn node_contains(outer: &Node, point: &Point) -> bool {
        if let Some(bounds) = &outer.bounds {
            bounds.nw.x < point.x
                && bounds.nw.y > point.y
                && bounds.se.x > point.x
                && bounds.se.y < point.y
        } else {
            false
        }
    }

    fn get_quadrant<'a>(root: &'a Node, point: &Point) -> Option<&'a Box<Node>> {
        if let Some(nw) = &root.nw {
            if Self::node_contains(nw, point) {
                return Some(nw);
            }
        }
        if let Some(ne) = &root.ne {
            if Self::node_contains(ne, point) {
                return Some(ne);
            }
        }
        if let Some(sw) = &root.sw {
            if Self::node_contains(sw, point) {
                return Some(sw);
            }
        }
        if let Some(se) = &root.se {
            if Self::node_contains(se, point) {
                return Some(se);
            }
        }
        None
    }

    fn split_node(&self, node: &mut Node) -> bool {
        if let Some(bounds) = &node.bounds {
            let x = bounds.nw.x;
            let y = bounds.nw.y;
            let hw = bounds.width / 2.0;
            let hh = bounds.height / 2.0;

            let nw = Node::with_bounds(x, y - hh, x + hw, y);
            let ne = Node::with_bounds(x + hw, y - hh, x + hw * 2.0, y);
            let sw = Node::with_bounds(x, y - hh * 2.0, x + hw, y - hh);
            let se = Node::with_bounds(x + hw, y - hh * 2.0, x + hw * 2.0, y - hh);

            if let (Some(nw), Some(ne), Some(sw), Some(se)) = (nw, ne, sw, se) {
                node.nw = Some(Box::new(nw));
                node.ne = Some(Box::new(ne));
                node.sw = Some(Box::new(sw));
                node.se = Some(Box::new(se));

                let old_point = node.point.take();
                let old_key = node.key.take();

                if let Some(point) = old_point {
                    self.insert_internal(&mut Some(Box::new(node.clone())), point, old_key.unwrap());
                }
                true
            } else {
                false
            }
        } else {
            false
        }
    }

    fn insert_internal(
        &self,
        root: &mut Option<Box<Node>>,
        point: Point,
        key: Box<dyn Any + Send + Sync>,
    ) -> bool {
        if let Some(root) = root {
            if root.is_empty() {
                root.point = Some(point);
                root.key = Some(key);
                true
            } else if root.is_leaf() {
                if let Some(existing_point) = &root.point {
                    if existing_point.x == point.x && existing_point.y == point.y {
                        if let Some(key_free) = &self.key_free {
                            root.reset(key_free.as_ref());
                        }
                        root.point = Some(point);
                        root.key = Some(key);
                        false
                    } else {
                        if !self.split_node(root) {
                            return false;
                        }
                        self.insert_internal(root, point, key)
                    }
                } else {
                    false
                }
            } else if root.is_pointer() {
                if let Some(quadrant) = Self::get_quadrant(root, &point) {
                    self.insert_internal(&mut Some(quadrant.clone()), point, key)
                } else {
                    false
                }
            } else {
                false
            }
        } else {
            false
        }
    }

    fn find_internal(&self, node: &Node, x: f64, y: f64) -> Option<&Point> {
        if node.is_leaf() {
            if let Some(point) = &node.point {
                if point.x == x && point.y == y {
                    return Some(point);
                }
            }
        } else {
            let test_point = Point::new(x, y);
            if let Some(quadrant) = Self::get_quadrant(node, &test_point) {
                return self.find_internal(quadrant, x, y);
            }
        }
        None
    }

    /// Walks the quadtree in a depth-first manner, calling the provided functions
    pub fn walk<D, A>(root: &Node, descent: D, ascent: A)
    where
        D: Fn(&Node),
        A: Fn(&Node),
    {
        descent(root);
        if let Some(nw) = &root.nw {
            Self::walk(nw, &descent, &ascent);
        }
        if let Some(ne) = &root.ne {
            Self::walk(ne, &descent, &ascent);
        }
        if let Some(sw) = &root.sw {
            Self::walk(sw, &descent, &ascent);
        }
        if let Some(se) = &root.se {
            Self::walk(se, &descent, &ascent);
        }
        ascent(root);
    }
}

// Note: We don't need to implement Drop explicitly as Rust's ownership system
// will handle memory deallocation automatically