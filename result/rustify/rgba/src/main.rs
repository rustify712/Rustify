use std::fmt;

/// RGBA color representation using f64 values between 0.0 and 1.0
#[derive(Debug, Clone, Copy)]
pub struct Rgba {
    pub r: f64,
    pub g: f64,
    pub b: f64,
    pub a: f64,
}

/// Named color definition
struct NamedColor {
    name: &'static str,
    val: u32,
}

/// List of predefined named colors
const NAMED_COLORS: &[NamedColor] = &[
    NamedColor { name: "transparent", val: 0xFFFFFF00 },
    NamedColor { name: "aliceblue", val: 0xF0F8FFFF },
    NamedColor { name: "antiquewhite", val: 0xFAEBD7FF },
    NamedColor { name: "aqua", val: 0x00FFFFFF },
    NamedColor { name: "aquamarine", val: 0x7FFFD4FF },
    NamedColor { name: "azure", val: 0xF0FFFFFF },
    NamedColor { name: "beige", val: 0xF5F5DCFF },
    NamedColor { name: "bisque", val: 0xFFE4C4FF },
    NamedColor { name: "black", val: 0x000000FF },
    NamedColor { name: "blanchedalmond", val: 0xFFEBCDFF },
    NamedColor { name: "blue", val: 0x0000FFFF },
    NamedColor { name: "blueviolet", val: 0x8A2BE2FF },
    NamedColor { name: "brown", val: 0xA52A2AFF },
    NamedColor { name: "burlywood", val: 0xDEB887FF },
    NamedColor { name: "cadetblue", val: 0x5F9EA0FF },
    NamedColor { name: "chartreuse", val: 0x7FFF00FF },
    NamedColor { name: "chocolate", val: 0xD2691EFF },
    NamedColor { name: "coral", val: 0xFF7F50FF },
    NamedColor { name: "cornflowerblue", val: 0x6495EDFF },
    NamedColor { name: "cornsilk", val: 0xFFF8DCFF },
    NamedColor { name: "crimson", val: 0xDC143CFF },
    NamedColor { name: "cyan", val: 0x00FFFFFF },
    NamedColor { name: "darkblue", val: 0x00008BFF },
    NamedColor { name: "darkcyan", val: 0x008B8BFF },
    NamedColor { name: "darkgoldenrod", val: 0xB8860BFF },
    NamedColor { name: "darkgray", val: 0xA9A9A9FF },
    NamedColor { name: "darkgreen", val: 0x006400FF },
    NamedColor { name: "darkgrey", val: 0xA9A9A9FF },
    NamedColor { name: "darkkhaki", val: 0xBDB76BFF },
    NamedColor { name: "darkmagenta", val: 0x8B008BFF },
    NamedColor { name: "darkolivegreen", val: 0x556B2FFF },
    NamedColor { name: "darkorange", val: 0xFF8C00FF },
    NamedColor { name: "darkorchid", val: 0x9932CCFF },
    NamedColor { name: "darkred", val: 0x8B0000FF },
    NamedColor { name: "darksalmon", val: 0xE9967AFF },
    NamedColor { name: "darkseagreen", val: 0x8FBC8FFF },
    NamedColor { name: "darkslateblue", val: 0x483D8BFF },
    NamedColor { name: "darkslategray", val: 0x2F4F4FFF },
    NamedColor { name: "darkslategrey", val: 0x2F4F4FFF },
    NamedColor { name: "darkturquoise", val: 0x00CED1FF },
    NamedColor { name: "darkviolet", val: 0x9400D3FF },
    NamedColor { name: "deeppink", val: 0xFF1493FF },
    NamedColor { name: "deepskyblue", val: 0x00BFFFFF },
    NamedColor { name: "dimgray", val: 0x696969FF },
    NamedColor { name: "dimgrey", val: 0x696969FF },
    NamedColor { name: "dodgerblue", val: 0x1E90FFFF },
    NamedColor { name: "firebrick", val: 0xB22222FF },
    NamedColor { name: "floralwhite", val: 0xFFFAF0FF },
    NamedColor { name: "forestgreen", val: 0x228B22FF },
    NamedColor { name: "fuchsia", val: 0xFF00FFFF },
    NamedColor { name: "gainsboro", val: 0xDCDCDCFF },
    NamedColor { name: "ghostwhite", val: 0xF8F8FFFF },
    NamedColor { name: "gold", val: 0xFFD700FF },
    NamedColor { name: "goldenrod", val: 0xDAA520FF },
    NamedColor { name: "gray", val: 0x808080FF },
    NamedColor { name: "green", val: 0x008000FF },
    NamedColor { name: "greenyellow", val: 0xADFF2FFF },
    NamedColor { name: "grey", val: 0x808080FF },
    NamedColor { name: "honeydew", val: 0xF0FFF0FF },
    NamedColor { name: "hotpink", val: 0xFF69B4FF },
    NamedColor { name: "indianred", val: 0xCD5C5CFF },
    NamedColor { name: "indigo", val: 0x4B0082FF },
    NamedColor { name: "ivory", val: 0xFFFFF0FF },
    NamedColor { name: "khaki", val: 0xF0E68CFF },
    NamedColor { name: "lavender", val: 0xE6E6FAFF },
    NamedColor { name: "lavenderblush", val: 0xFFF0F5FF },
    NamedColor { name: "lawngreen", val: 0x7CFC00FF },
    NamedColor { name: "lemonchiffon", val: 0xFFFACDFF },
    NamedColor { name: "lightblue", val: 0xADD8E6FF },
    NamedColor { name: "lightcoral", val: 0xF08080FF },
    NamedColor { name: "lightcyan", val: 0xE0FFFFFF },
    NamedColor { name: "lightgoldenrodyellow", val: 0xFAFAD2FF },
    NamedColor { name: "lightgray", val: 0xD3D3D3FF },
    NamedColor { name: "lightgreen", val: 0x90EE90FF },
    NamedColor { name: "lightgrey", val: 0xD3D3D3FF },
    NamedColor { name: "lightpink", val: 0xFFB6C1FF },
    NamedColor { name: "lightsalmon", val: 0xFFA07AFF },
    NamedColor { name: "lightseagreen", val: 0x20B2AAFF },
    NamedColor { name: "lightskyblue", val: 0x87CEFAFF },
    NamedColor { name: "lightslategray", val: 0x778899FF },
    NamedColor { name: "lightslategrey", val: 0x778899FF },
    NamedColor { name: "lightsteelblue", val: 0xB0C4DEFF },
    NamedColor { name: "lightyellow", val: 0xFFFFE0FF },
    NamedColor { name: "lime", val: 0x00FF00FF },
    NamedColor { name: "limegreen", val: 0x32CD32FF },
    NamedColor { name: "linen", val: 0xFAF0E6FF },
    NamedColor { name: "magenta", val: 0xFF00FFFF },
    NamedColor { name: "maroon", val: 0x800000FF },
    NamedColor { name: "mediumaquamarine", val: 0x66CDAAFF },
    NamedColor { name: "mediumblue", val: 0x0000CDFF },
    NamedColor { name: "mediumorchid", val: 0xBA55D3FF },
    NamedColor { name: "mediumpurple", val: 0x9370DBFF },
    NamedColor { name: "mediumseagreen", val: 0x3CB371FF },
    NamedColor { name: "mediumslateblue", val: 0x7B68EEFF },
    NamedColor { name: "mediumspringgreen", val: 0x00FA9AFF },
    NamedColor { name: "mediumturquoise", val: 0x48D1CCFF },
    NamedColor { name: "mediumvioletred", val: 0xC71585FF },
    NamedColor { name: "midnightblue", val: 0x191970FF },
    NamedColor { name: "mintcream", val: 0xF5FFFAFF },
    NamedColor { name: "mistyrose", val: 0xFFE4E1FF },
    NamedColor { name: "moccasin", val: 0xFFE4B5FF },
    NamedColor { name: "navajowhite", val: 0xFFDEADFF },
    NamedColor { name: "navy", val: 0x000080FF },
    NamedColor { name: "oldlace", val: 0xFDF5E6FF },
    NamedColor { name: "olive", val: 0x808000FF },
    NamedColor { name: "olivedrab", val: 0x6B8E23FF },
    NamedColor { name: "orange", val: 0xFFA500FF },
    NamedColor { name: "orangered", val: 0xFF4500FF },
    NamedColor { name: "orchid", val: 0xDA70D6FF },
    NamedColor { name: "palegoldenrod", val: 0xEEE8AAFF },
    NamedColor { name: "palegreen", val: 0x98FB98FF },
    NamedColor { name: "paleturquoise", val: 0xAFEEEEFF },
    NamedColor { name: "palevioletred", val: 0xDB7093FF },
    NamedColor { name: "papayawhip", val: 0xFFEFD5FF },
    NamedColor { name: "peachpuff", val: 0xFFDAB9FF },
    NamedColor { name: "peru", val: 0xCD853FFF },
    NamedColor { name: "pink", val: 0xFFC0CBFF },
    NamedColor { name: "plum", val: 0xDDA0DDFF },
    NamedColor { name: "powderblue", val: 0xB0E0E6FF },
    NamedColor { name: "purple", val: 0x800080FF },
    NamedColor { name: "red", val: 0xFF0000FF },
    NamedColor { name: "rosybrown", val: 0xBC8F8FFF },
    NamedColor { name: "royalblue", val: 0x4169E1FF },
    NamedColor { name: "saddlebrown", val: 0x8B4513FF },
    NamedColor { name: "salmon", val: 0xFA8072FF },
    NamedColor { name: "sandybrown", val: 0xF4A460FF },
    NamedColor { name: "seagreen", val: 0x2E8B57FF },
    NamedColor { name: "seashell", val: 0xFFF5EEFF },
    NamedColor { name: "sienna", val: 0xA0522DFF },
    NamedColor { name: "silver", val: 0xC0C0C0FF },
    NamedColor { name: "skyblue", val: 0x87CEEBFF },
    NamedColor { name: "slateblue", val: 0x6A5ACDFF },
    NamedColor { name: "slategray", val: 0x708090FF },
    NamedColor { name: "slategrey", val: 0x708090FF },
    NamedColor { name: "snow", val: 0xFFFAFAFF },
    NamedColor { name: "springgreen", val: 0x00FF7FFF },
    NamedColor { name: "steelblue", val: 0x4682B4FF },
    NamedColor { name: "tan", val: 0xD2B48CFF },
    NamedColor { name: "teal", val: 0x008080FF },
    NamedColor { name: "thistle", val: 0xD8BFD8FF },
    NamedColor { name: "tomato", val: 0xFF6347FF },
    NamedColor { name: "turquoise", val: 0x40E0D0FF },
    NamedColor { name: "violet", val: 0xEE82EEFF },
    NamedColor { name: "wheat", val: 0xF5DEB3FF },
    NamedColor { name: "white", val: 0xFFFFFFFF },
    NamedColor { name: "whitesmoke", val: 0xF5F5F5FF },
    NamedColor { name: "yellow", val: 0xFFFF00FF },
    NamedColor { name: "yellowgreen", val: 0x9ACD32FF },
];

impl Rgba {
    /// Create a new RGBA color from a u32
    pub fn new(rgba: u32) -> Self {
        Self {
            r: ((rgba >> 24) & 0xff) as f64 / 255.0,
            g: ((rgba >> 16) & 0xff) as f64 / 255.0,
            b: ((rgba >> 8) & 0xff) as f64 / 255.0,
            a: (rgba & 0xff) as f64 / 255.0,
        }
    }

    /// Convert RGBA color to string representation
    pub fn to_string(&self) -> String {
        if (self.a - 1.0).abs() < f64::EPSILON {
            format!("#{:02x}{:02x}{:02x}",
                    (self.r * 255.0) as i32,
                    (self.g * 255.0) as i32,
                    (self.b * 255.0) as i32
            )
        } else {
            format!("rgba({}, {}, {}, {:.2})",
                    (self.r * 255.0) as i32,
                    (self.g * 255.0) as i32,
                    (self.b * 255.0) as i32,
                    self.a
            )
        }
    }

    /// Inspect the color value
    pub fn inspect(rgba: u32) {
        println!("rgba({},{},{},{})",
                 (rgba >> 24) & 0xff,
                 (rgba >> 16) & 0xff,
                 (rgba >> 8) & 0xff,
                 rgba & 0xff
        );
    }
}

impl fmt::Display for Rgba {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

/// Convert hex character to integer value
fn hex_to_int(c: char) -> u8 {
    match c {
        '0'..='9' => c as u8 - b'0',
        'a'..='f' => c as u8 - b'a' + 10,
        'A'..='F' => c as u8 - b'A' + 10,
        _ => 0,
    }
}

/// Helper function to create RGBA from components
fn rgba_from_components(r: u8, g: u8, b: u8, a: u8) -> u32 {
    ((r as u32) << 24) | ((g as u32) << 16) | ((b as u32) << 8) | (a as u32)
}

/// Helper function to create RGB with alpha=255
fn rgb_from_components(r: u8, g: u8, b: u8) -> u32 {
    rgba_from_components(r, g, b, 255)
}

/// Parse a 6-digit hex color string
fn rgba_from_hex6_string(str: &str) -> u32 {
    let r = (hex_to_int(str.chars().nth(0).unwrap()) << 4) + hex_to_int(str.chars().nth(1).unwrap());
    let g = (hex_to_int(str.chars().nth(2).unwrap()) << 4) + hex_to_int(str.chars().nth(3).unwrap());
    let b = (hex_to_int(str.chars().nth(4).unwrap()) << 4) + hex_to_int(str.chars().nth(5).unwrap());
    rgb_from_components(r, g, b)
}

/// Parse a 3-digit hex color string
fn rgba_from_hex3_string(str: &str) -> u32 {
    let chars: Vec<char> = str.chars().collect();
    let r = hex_to_int(chars[0]);
    let g = hex_to_int(chars[1]);
    let b = hex_to_int(chars[2]);
    rgb_from_components(
        (r << 4) + r,
        (g << 4) + g,
        (b << 4) + b
    )
}

/// Parse an RGB color string like "rgb(r,g,b)"
fn rgba_from_rgb_string(str: &str) -> Option<u32> {
    if !str.starts_with("rgb(") {
        return None;
    }

    // Remove "rgb(" prefix and ")" suffix
    let content = str[4..].trim();
    if !content.ends_with(')') {
        return None;
    }
    let content = &content[..content.len() - 1];

    let mut parts = content.split(',');

    let r = parts.next()?.trim().parse::<u8>().ok()?;
    let g = parts.next()?.trim().parse::<u8>().ok()?;
    let mut b_str = parts.next()?.trim();

    // Remove any trailing characters after the last number
    if let Some(end_pos) = b_str.find(|c: char| !c.is_ascii_digit()) {
        b_str = &b_str[..end_pos];
    }

    let b = b_str.parse::<u8>().ok()?;

    // Check for values exceeding 255
    if r > 255 || g > 255 || b > 255 {
        return None;
    }

    Some(rgb_from_components(r, g, b))
}


/// Parse an RGBA color string like "rgba(r,g,b,a)"
fn rgba_from_rgba_string(str: &str) -> Option<u32> {
    if !str.starts_with("rgba(") {
        return None;
    }

    // Remove "rgba(" prefix and ")" suffix
    let content = str[5..].trim();
    if !content.ends_with(')') {
        return None;
    }
    let content = &content[..content.len() - 1];

    let mut parts = content.split(',');

    let r = parts.next()?.trim().parse::<u8>().ok()?;
    let g = parts.next()?.trim().parse::<u8>().ok()?;
    let b = parts.next()?.trim().parse::<u8>().ok()?;

    // Check for values exceeding 255
    if r > 255 || g > 255 || b > 255 {
        return None;
    }

    let a_str = parts.next()?.trim();
    let a = if a_str.starts_with('1') {
        255
    } else if a_str.starts_with('0') {
        let mut value = 0.0;
        if let Some(decimal) = a_str.split('.').nth(1) {
            let mut multiplier = 0.1;
            for c in decimal.chars() {
                if !c.is_ascii_digit() {
                    break;
                }
                value += (c as u8 - b'0') as f64 * multiplier;
                multiplier *= 0.1;
            }
        }
        (value * 255.0) as u8
    } else {
        return None;
    };

    Some(rgba_from_components(r, g, b, a))
}

/// Parse a hex color string
fn rgba_from_hex_string(str: &str) -> Option<u32> {
    match str.len() {
        6 => Some(rgba_from_hex6_string(str)),
        3 => Some(rgba_from_hex3_string(str)),
        _ => None
    }
}

/// Find a named color value
fn rgba_from_name_string(str: &str) -> Option<u32> {
    NAMED_COLORS.iter()
        .find(|color| color.name == str)
        .map(|color| color.val)
}

/// Parse a color string in any supported format
pub fn rgba_from_string(str: &str) -> Option<u32> {
    if str.starts_with('#') {
        rgba_from_hex_string(&str[1..])
    } else if str.starts_with("rgba") {
        rgba_from_rgba_string(str)
    } else if str.starts_with("rgb") {
        rgba_from_rgb_string(str)
    } else {
        rgba_from_name_string(str)
    }
}