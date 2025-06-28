#[derive(Debug, Clone)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    /// Creates a new point with the given coordinates
    /// 
    /// # Arguments
    /// * `x` - The x coordinate
    /// * `y` - The y coordinate
    /// 
    /// # Returns
    /// A new Point instance
    pub fn new(x: f64, y: f64) -> Self {
        Point { x, y }
    }

    /// Creates a new point and returns it as a boxed value
    /// This is equivalent to the C function quadtree_point_new
    /// 
    /// # Arguments
    /// * `x` - The x coordinate
    /// * `y` - The y coordinate
    /// 
    /// # Returns
    /// A new Point instance wrapped in Some, or None if allocation would fail
    pub fn new_boxed(x: f64, y: f64) -> Option<Box<Self>> {
        Some(Box::new(Self::new(x, y)))
    }
}

// Note: We don't need to implement Drop trait or free function
// as Rust's ownership system handles memory deallocation automatically