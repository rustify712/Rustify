// rgba.rs

use std::str::FromStr;
use std::fmt;

// RGBA struct.
#[derive(Debug, Clone, Copy)]
pub struct Rgba {
    pub r: f64,
    pub g: f64,
    pub b: f64,
    pub a: f64,
}

// Named color structure.
#[derive(Debug, Clone, Copy)]
struct NamedColor<'a> {
    name: &'a str,
    val: u32,
}

// Define the list of named colors.
const NAMED_COLORS: &[NamedColor] = &[
    NamedColor { name: "transparent", val: 0xFFFFFF00 },
    NamedColor { name: "aliceblue", val: 0xF0F8FFFF },
    NamedColor { name: "antiquewhite", val: 0xFAEBD7FF },
    NamedColor { name: "aqua", val: 0x00FFFFFF },
    NamedColor { name: "aquamarine", val: 0x7FFFD4FF },
    // ... (Include all other named colors as necessary)
];

// Helper function to convert hex characters to integer values.
fn hex_digit(c: char) -> u8 {
    match c {
        '0'..='9' => c as u8 - b'0',
        'a'..='f' => c as u8 - b'a' + 10,
        'A'..='F' => c as u8 - b'A' + 10,
        _ => 0,
    }
}

// Create a new RGBA color from a 32-bit integer.
pub fn rgba_new(rgba: u32) -> Rgba {
    Rgba {
        r: (rgba >> 24) as f64 / 255.0,
        g: ((rgba >> 16) & 0xFF) as f64 / 255.0,
        b: ((rgba >> 8) & 0xFF) as f64 / 255.0,
        a: (rgba & 0xFF) as f64 / 255.0,
    }
}

// Return a string representation of the RGBA color.
pub fn rgba_to_string(rgba: Rgba) -> String {
    if rgba.a == 1.0 {
        format!("#{:02x}{:02x}{:02x}",
            (rgba.r * 255.0) as u8,
            (rgba.g * 255.0) as u8,
            (rgba.b * 255.0) as u8)
    } else {
        format!("rgba({},{},{},{:.2})",
            (rgba.r * 255.0) as u8,
            (rgba.g * 255.0) as u8,
            (rgba.b * 255.0) as u8,
            rgba.a)
    }
}

// Convert from RGB values to a 32-bit integer RGBA value.
fn rgba_from_rgba(r: u8, g: u8, b: u8, a: u8) -> u32 {
    (r as u32) << 24 | (g as u32) << 16 | (b as u32) << 8 | (a as u32)
}

// Convert from RGB values to a 32-bit RGBA value (with full opacity).
fn rgba_from_rgb(r: u8, g: u8, b: u8) -> u32 {
    rgba_from_rgba(r, g, b, 255)
}

// Convert from "#RRGGBB" hex string to a 32-bit RGBA value.
fn rgba_from_hex6_string(s: &str) -> Option<u32> {
    if s.len() == 6 {
        Some(rgba_from_rgb(
            (hex_digit(s[0]) << 4) + hex_digit(s[1]),
            (hex_digit(s[2]) << 4) + hex_digit(s[3]),
            (hex_digit(s[4]) << 4) + hex_digit(s[5]),
        ))
    } else {
        None
    }
}

// Convert from "#RGB" hex string to a 32-bit RGBA value (expand to "#RRGGBB").
fn rgba_from_hex3_string(s: &str) -> Option<u32> {
    if s.len() == 3 {
        Some(rgba_from_rgb(
            (hex_digit(s[0]) << 4) + hex_digit(s[0]),
            (hex_digit(s[1]) << 4) + hex_digit(s[1]),
            (hex_digit(s[2]) << 4) + hex_digit(s[2]),
        ))
    } else {
        None
    }
}

// Parse "rgb(r,g,b)" string to a 32-bit RGBA value.
fn rgba_from_rgb_string(s: &str) -> Option<u32> {
    if s.starts_with("rgb(") {
        let parts: Vec<&str> = s[4..s.len() - 1].split(',').collect();
        if parts.len() == 3 {
            let r: u8 = parts[0].trim().parse().ok()?;
            let g: u8 = parts[1].trim().parse().ok()?;
            let b: u8 = parts[2].trim().parse().ok()?;
            return Some(rgba_from_rgb(r, g, b));
        }
    }
    None
}

// Parse "rgba(r,g,b,a)" string to a 32-bit RGBA value.
fn rgba_from_rgba_string(s: &str) -> Option<u32> {
    if s.starts_with("rgba(") {
        let parts: Vec<&str> = s[5..s.len() - 1].split(',').collect();
        if parts.len() == 4 {
            let r: u8 = parts[0].trim().parse().ok()?;
            let g: u8 = parts[1].trim().parse().ok()?;
            let b: u8 = parts[2].trim().parse().ok()?;
            let a: f64 = parts[3].trim().parse().ok()?;
            return Some(rgba_from_rgba(r, g, b, (a * 255.0) as u8));
        }
    }
    None
}

// Convert a color name string to a 32-bit RGBA value.
fn rgba_from_name_string(s: &str) -> Option<u32> {
    NAMED_COLORS.iter()
        .find(|&color| color.name == s)
        .map(|color| color.val)
}

// Parse any color string format to a 32-bit RGBA value.
pub fn rgba_from_string(s: &str) -> Option<u32> {
    if s.starts_with('#') {
        let hex = &s[1..];
        if hex.len() == 6 {
            return rgba_from_hex6_string(hex);
        } else if hex.len() == 3 {
            return rgba_from_hex3_string(hex);
        }
    }
    rgba_from_rgba_string(s)
        .or_else(|| rgba_from_rgb_string(s))
        .or_else(|| rgba_from_name_string(s))
}

// Inspect and print an RGBA color.
pub fn rgba_inspect(rgba: u32) {
    println!("rgba({}, {}, {}, {})",
        (rgba >> 24) & 0xFF,
        (rgba >> 16) & 0xFF,
        (rgba >> 8) & 0xFF,
        rgba & 0xFF);
}
