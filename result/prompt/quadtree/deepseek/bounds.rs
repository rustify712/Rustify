use std::f64::{INFINITY, NEG_INFINITY};

impl QuadtreeBounds {
    // Extend the bounds to include the given point (x, y)
    fn extend(&mut self, x: f64, y: f64) {
        self.nw.x = f64::min(x, self.nw.x);
        self.nw.y = f64::max(y, self.nw.y);
        self.se.x = f64::max(x, self.se.x);
        self.se.y = f64::min(y, self.se.y);
        self.width = f64::abs(self.nw.x - self.se.x);
        self.height = f64::abs(self.nw.y - self.se.y);
    }

    // Create a new QuadtreeBounds instance
    fn new() -> Self {
        QuadtreeBounds {
            nw: Box::new(QuadtreePoint::new(INFINITY, NEG_INFINITY)),
            se: Box::new(QuadtreePoint::new(NEG_INFINITY, INFINITY)),
            width: 0.0,
            height: 0.0,
        }
    }
}

// Free the memory associated with a QuadtreeBounds instance
fn quadtree_bounds_free(bounds: QuadtreeBounds) {
    // No explicit free needed in Rust due to ownership model
    // The bounds and its contents will be dropped automatically when they go out of scope
}