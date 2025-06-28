/// Represents a point in the 2D space.
#[derive(Debug, Clone)]
pub struct QuadtreePoint {
    pub x: f64,
    pub y: f64,
}

impl QuadtreePoint {
    /// Creates a new point with the given coordinates.
    pub fn new(x: f64, y: f64) -> QuadtreePoint {
        QuadtreePoint { x, y }
    }
}

/// Frees the memory used by a point.
pub fn quadtree_point_free(point: QuadtreePoint) {
    // In Rust, the memory for the point will be automatically freed when it goes out of scope.
    // This function is more of a placeholder to demonstrate the idea of freeing a point.
    // In practice, this line would not be necessary as Rust handles memory management automatically.
    drop(point);  // Explicitly drops the point, though unnecessary in most cases.
}
