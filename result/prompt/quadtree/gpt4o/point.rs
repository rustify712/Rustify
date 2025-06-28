// quadtree.rs

impl QuadtreePoint {
    // Create a new QuadtreePoint instance
    fn new(x: f64, y: f64) -> Self {
        QuadtreePoint { x, y }
    }
}

// No explicit free function is needed in Rust due to ownership model
// The point will be automatically dropped when it goes out of scope