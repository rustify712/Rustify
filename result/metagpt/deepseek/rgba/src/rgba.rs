//! RGBA color module
//!
//! Provides color parsing and conversion functionality

use std::fmt;

/// Named color with its RGBA value
struct NamedColor {
    name: &'static str,
    value: u32,
}

/// Predefined named colors
const NAMED_COLORS: &[NamedColor] = &[
    NamedColor { name: "transparent", value: 0xFFFFFF00 },
    NamedColor { name: "aliceblue", value: 0xF0F8FFFF },
    NamedColor { name: "antiquewhite", value: 0xFAEBD7FF },
    // ... (other color definitions from the C code)
    NamedColor { name: "yellowgreen", value: 0x9ACD32FF },
];

/// RGBA color representation
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rgba {
    pub r: f64,
    pub g: f64,
    pub b: f64,
    pub a: f64,
}

impl Rgba {
    /// Create new Rgba from 32-bit RGBA value
    pub fn new(rgba: u32) -> Self {
        Rgba {
            r: ((rgba >> 24) & 0xff) as f64 / 255.0,
            g: ((rgba >> 16) & 0xff) as f64 / 255.0,
            b: ((rgba >> 8) & 0xff) as f64 / 255.0,
            a: (rgba & 0xff) as f64 / 255.0,
        }
    }

    /// Convert Rgba to string representation
    pub fn to_string(&self) -> String {
        if self.a == 1.0 {
            format!(
                "#{:02x}{:02x}{:02x}",
                (self.r * 255.0) as u8,
                (self.g * 255.0) as u8,
                (self.b * 255.0) as u8
            )
        } else {
            format!(
                "rgba({}, {}, {}, {:.2})",
                (self.r * 255.0) as u8,
                (self.g * 255.0) as u8,
                (self.b * 255.0) as u8,
                self.a
            )
        }
    }
}

/// Parse hex digit to integer value
fn hex_digit(c: char) -> Option<u8> {
    match c {
        '0'..='9' => Some((c as u8) - b'0'),
        'a'..='f' => Some((c as u8) - b'a' + 10),
        'A'..='F' => Some((c as u8) - b'A' + 10),
        _ => None,
    }
}

/// Parse color from string representation
pub fn rgba_from_string(s: &str) -> Result<u32, &'static str> {
    if s.starts_with('#') {
        let hex = &s[1..];
        match hex.len() {
            3 => {
                let r = hex_digit(hex.chars().nth(0).ok_or("Invalid hex color")?)?;
                let g = hex_digit(hex.chars().nth(1).ok_or("Invalid hex color")?)?;
                let b = hex_digit(hex.chars().nth(2).ok_or("Invalid hex color")?)?;
                Ok((r << 4 | r) << 24 | (g << 4 | g) << 16 | (b << 4 | b) << 8 | 0xff)
            }
            6 => {
                let r = hex_digit(hex.chars().nth(0).ok_or("Invalid hex color")?)? << 4
                    | hex_digit(hex.chars().nth(1).ok_or("Invalid hex color")?)?;
                let g = hex_digit(hex.chars().nth(2).ok_or("Invalid hex color")?)? << 4
                    | hex_digit(hex.chars().nth(3).ok_or("Invalid hex color")?)?;
                let b = hex_digit(hex.chars().nth(4).ok_or("Invalid hex color")?)? << 4
                    | hex_digit(hex.chars().nth(5).ok_or("Invalid hex color")?)?;
                Ok(r << 24 | g << 16 | b << 8 | 0xff)
            }
            _ => Err("Invalid hex color length"),
        }
    } else if s.starts_with("rgba(") {
        // Parse rgba() format
        Err("rgba() parsing not implemented")
    } else if s.starts_with("rgb(") {
        // Parse rgb() format
        Err("rgb() parsing not implemented")
    } else {
        // Check named colors
        for color in NAMED_COLORS {
            if color.name == s {
                return Ok(color.value);
            }
        }
        Err("Unknown color format")
    }
}

/// Inspect RGBA color components
pub fn rgba_inspect(rgba: u32) {
    println!(
        "rgba({},{},{},{})",
        (rgba >> 24) & 0xff,
        (rgba >> 16) & 0xff,
        (rgba >> 8) & 0xff,
        rgba & 0xff
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgba_new() {
        let color = Rgba::new(0xFF0000FF);
        assert_eq!(color.r, 1.0);
        assert_eq!(color.g, 0.0);
        assert_eq!(color.b, 0.0);
        assert_eq!(color.a, 1.0);
    }

    #[test]
    fn test_hex_color_parsing() {
        assert_eq!(rgba_from_string("#f00"), Ok(0xFF0000FF));
        assert_eq!(rgba_from_string("#ff0000"), Ok(0xFF0000FF));
    }

    #[test]
    fn test_named_color_parsing() {
        assert_eq!(rgba_from_string("red"), Ok(0xFF0000FF));
    }
}