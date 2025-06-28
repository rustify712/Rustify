use std::ptr;
use crate::node::{QuadtreeNode, quadtree_node_with_bounds, quadtree_node_reset, quadtree_node_free};
use crate::point::{QuadtreePoint, quadtree_point_new};

pub struct Quadtree {
    pub root: Option<Box<QuadtreeNode>>,
    pub key_free: Option<Box<dyn FnOnce()>>,
    pub length: usize,
}

impl Quadtree {
    pub fn new(minx: f64, miny: f64, maxx: f64, maxy: f64) -> Option<Box<Self>> {
        let root = quadtree_node_with_bounds(minx, miny, maxx, maxy)?;
        Some(Box::new(Quadtree {
            root: Some(root),
            key_free: None,
            length: 0,
        }))
    }

    pub fn insert(&mut self, x: f64, y: f64, key: Box<dyn FnOnce()>) -> bool {
        let point = quadtree_point_new(x, y)?;
        if !self.node_contains(&self.root, &point) {
            return false;
        }
        if !self.insert_node(&mut self.root, point, key) {
            return false;
        }
        self.length += 1;
        true
    }

    pub fn search(&self, x: f64, y: f64) -> Option<Box<QuadtreePoint>> {
        self.find(&self.root, x, y)
    }

    pub fn free(&mut self) {
        if let Some(root) = self.root.take() {
            if let Some(key_free) = self.key_free.take() {
                quadtree_node_free(root, key_free);
            } else {
                quadtree_node_free(root, Box::new(|| {}));
            }
        }
    }

    pub fn walk<F, G>(&self, root: &Box<QuadtreeNode>, descent: F, ascent: G)
    where
        F: Fn(&Box<QuadtreeNode>),
        G: Fn(&Box<QuadtreeNode>),
    {
        descent(root);
        if let Some(nw) = &root.nw {
            self.walk(nw, &descent, &ascent);
        }
        if let Some(ne) = &root.ne {
            self.walk(ne, &descent, &ascent);
        }
        if let Some(sw) = &root.sw {
            self.walk(sw, &descent, &ascent);
        }
        if let Some(se) = &root.se {
            self.walk(se, &descent, &ascent);
        }
        ascent(root);
    }

    fn node_contains(&self, outer: &Option<Box<QuadtreeNode>>, it: &Box<QuadtreePoint>) -> bool {
        if let Some(outer) = outer {
            if let Some(bounds) = &outer.bounds {
                bounds.nw.x < it.x && bounds.nw.y > it.y && bounds.se.x > it.x && bounds.se.y < it.y
            } else {
                false
            }
        } else {
            false
        }
    }

    fn insert_node(&mut self, root: &mut Option<Box<QuadtreeNode>>, point: Box<QuadtreePoint>, key: Box<dyn FnOnce()>) -> bool {
        if let Some(root) = root {
            if root.is_empty() {
                root.point = Some(point);
                root.key = Some(key);
                return true;
            } else if root.is_leaf() {
                if let Some(existing_point) = &root.point {
                    if existing_point.x == point.x && existing_point.y == point.y {
                        self.reset_node(root);
                        root.point = Some(point);
                        root.key = Some(key);
                        return false;
                    } else {
                        if !self.split_node(root) {
                            return false;
                        }
                        return self.insert_node(root, point, key);
                    }
                }
            } else if root.is_pointer() {
                if let Some(quadrant) = self.get_quadrant(root, &point) {
                    return self.insert_node(&mut Some(quadrant), point, key);
                }
            }
        }
        false
    }

    fn reset_node(&mut self, node: &mut Box<QuadtreeNode>) {
        if let Some(key_free) = self.key_free.take() {
            quadtree_node_reset(node, key_free);
        } else {
            quadtree_node_reset(node, Box::new(|| {}));
        }
    }

    fn split_node(&mut self, node: &mut Box<QuadtreeNode>) -> bool {
        let x = node.bounds.as_ref()?.nw.x;
        let y = node.bounds.as_ref()?.nw.y;
        let hw = node.bounds.as_ref()?.width / 2.0;
        let hh = node.bounds.as_ref()?.height / 2.0;

        let nw = quadtree_node_with_bounds(x, y - hh, x + hw, y)?;
        let ne = quadtree_node_with_bounds(x + hw, y - hh, x + hw * 2.0, y)?;
        let sw = quadtree_node_with_bounds(x, y - hh * 2.0, x + hw, y - hh)?;
        let se = quadtree_node_with_bounds(x + hw, y - hh * 2.0, x + hw * 2.0, y - hh)?;

        node.nw = Some(nw);
        node.ne = Some(ne);
        node.sw = Some(sw);
        node.se = Some(se);

        let old_point = node.point.take();
        let old_key = node.key.take();

        if let Some(old_point) = old_point {
            return self.insert_node(node, old_point, old_key.unwrap());
        }
        false
    }

    fn get_quadrant(&self, root: &Box<QuadtreeNode>, point: &Box<QuadtreePoint>) -> Option<Box<QuadtreeNode>> {
        if self.node_contains(&root.nw, point) {
            return root.nw.clone();
        }
        if self.node_contains(&root.ne, point) {
            return root.ne.clone();
        }
        if self.node_contains(&root.sw, point) {
            return root.sw.clone();
        }
        if self.node_contains(&root.se, point) {
            return root.se.clone();
        }
        None
    }

    fn find(&self, node: &Option<Box<QuadtreeNode>>, x: f64, y: f64) -> Option<Box<QuadtreePoint>> {
        if let Some(node) = node {
            if node.is_leaf() {
                if let Some(point) = &node.point {
                    if point.x == x && point.y == y {
                        return Some(point.clone());
                    }
                }
            } else {
                let test_point = Box::new(QuadtreePoint { x, y });
                return self.find(&self.get_quadrant(node, &test_point), x, y);
            }
        }
        None
    }
}