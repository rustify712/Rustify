use std::f64;

/// A point in the 2D space.
#[derive(Debug, Clone)]
pub struct QuadtreePoint {
    pub x: f64,
    pub y: f64,
}

/// A bounding box that defines the area covered by a node.
#[derive(Debug, Clone)]
pub struct QuadtreeBounds {
    pub nw: QuadtreePoint,
    pub se: QuadtreePoint,
    pub width: f64,
    pub height: f64,
}

/// Creates a new bounding box.
pub fn quadtree_bounds_new() -> QuadtreeBounds {
    QuadtreeBounds {
        nw: QuadtreePoint {
            x: f64::INFINITY,
            y: f64::NEG_INFINITY,
        },
        se: QuadtreePoint {
            x: f64::NEG_INFINITY,
            y: f64::INFINITY,
        },
        width: 0.0,
        height: 0.0,
    }
}

/// Extends a bounds object to include a new point.
pub fn quadtree_bounds_extend(bounds: &mut QuadtreeBounds, x: f64, y: f64) {
    bounds.nw.x = bounds.nw.x.min(x);
    bounds.nw.y = bounds.nw.y.max(y);
    bounds.se.x = bounds.se.x.max(x);
    bounds.se.y = bounds.se.y.min(y);
    bounds.width = (bounds.nw.x - bounds.se.x).abs();
    bounds.height = (bounds.nw.y - bounds.se.y).abs();
}

/// Frees the memory used by a bounds object.
pub fn quadtree_bounds_free(bounds: &mut QuadtreeBounds) {
    // In Rust, there is no need to explicitly free memory like in C.
    // Rust will automatically drop the bounds when they go out of scope.
    // We simulate the freeing process for educational purposes:
    // Drop `nw` and `se` points.
    // In a real-world application, this may be where additional cleanup occurs.
    // No need to manually free anything here, Rust handles ownership automatically.
    bounds.nw = QuadtreePoint { x: 0.0, y: 0.0 }; // Simulate free
    bounds.se = QuadtreePoint { x: 0.0, y: 0.0 }; // Simulate free
}
