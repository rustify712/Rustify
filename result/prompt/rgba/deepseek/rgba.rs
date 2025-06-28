// rgba.rs

use std::str;

// RGBA struct
#[derive(Debug, Clone, Copy)]
pub struct Rgba {
    pub r: f64,
    pub g: f64,
    pub b: f64,
    pub a: f64,
}

// Named colors
const NAMED_COLORS: &[(&str, u32)] = &[
    ("transparent", 0xFFFFFF00),
    ("aliceblue", 0xF0F8FFFF),
    ("antiquewhite", 0xFAEBD7FF),
    ("aqua", 0x00FFFFFF),
    ("aquamarine", 0x7FFFD4FF),
    ("azure", 0xF0FFFFFF),
    ("beige", 0xF5F5DCFF),
    ("bisque", 0xFFE4C4FF),
    ("black", 0x000000FF),
    ("blanchedalmond", 0xFFEBCDFF),
    ("blue", 0x0000FFFF),
    ("blueviolet", 0x8A2BE2FF),
    ("brown", 0xA52A2AFF),
    ("burlywood", 0xDEB887FF),
    ("cadetblue", 0x5F9EA0FF),
    ("chartreuse", 0x7FFF00FF),
    ("chocolate", 0xD2691EFF),
    ("coral", 0xFF7F50FF),
    ("cornflowerblue", 0x6495EDFF),
    ("cornsilk", 0xFFF8DCFF),
    ("crimson", 0xDC143CFF),
    ("cyan", 0x00FFFFFF),
    ("darkblue", 0x00008BFF),
    ("darkcyan", 0x008B8BFF),
    ("darkgoldenrod", 0xB8860BFF),
    ("darkgray", 0xA9A9A9FF),
    ("darkgreen", 0x006400FF),
    ("darkgrey", 0xA9A9A9FF),
    ("darkkhaki", 0xBDB76BFF),
    ("darkmagenta", 0x8B008BFF),
    ("darkolivegreen", 0x556B2FFF),
    ("darkorange", 0xFF8C00FF),
    ("darkorchid", 0x9932CCFF),
    ("darkred", 0x8B0000FF),
    ("darksalmon", 0xE9967AFF),
    ("darkseagreen", 0x8FBC8FFF),
    ("darkslateblue", 0x483D8BFF),
    ("darkslategray", 0x2F4F4FFF),
    ("darkslategrey", 0x2F4F4FFF),
    ("darkturquoise", 0x00CED1FF),
    ("darkviolet", 0x9400D3FF),
    ("deeppink", 0xFF1493FF),
    ("deepskyblue", 0x00BFFFFF),
    ("dimgray", 0x696969FF),
    ("dimgrey", 0x696969FF),
    ("dodgerblue", 0x1E90FFFF),
    ("firebrick", 0xB22222FF),
    ("floralwhite", 0xFFFAF0FF),
    ("forestgreen", 0x228B22FF),
    ("fuchsia", 0xFF00FFFF),
    ("gainsboro", 0xDCDCDCFF),
    ("ghostwhite", 0xF8F8FFFF),
    ("gold", 0xFFD700FF),
    ("goldenrod", 0xDAA520FF),
    ("gray", 0x808080FF),
    ("green", 0x008000FF),
    ("greenyellow", 0xADFF2FFF),
    ("grey", 0x808080FF),
    ("honeydew", 0xF0FFF0FF),
    ("hotpink", 0xFF69B4FF),
    ("indianred", 0xCD5C5CFF),
    ("indigo", 0x4B0082FF),
    ("ivory", 0xFFFFF0FF),
    ("khaki", 0xF0E68CFF),
    ("lavender", 0xE6E6FAFF),
    ("lavenderblush", 0xFFF0F5FF),
    ("lawngreen", 0x7CFC00FF),
    ("lemonchiffon", 0xFFFACDFF),
    ("lightblue", 0xADD8E6FF),
    ("lightcoral", 0xF08080FF),
    ("lightcyan", 0xE0FFFFFF),
    ("lightgoldenrodyellow", 0xFAFAD2FF),
    ("lightgray", 0xD3D3D3FF),
    ("lightgreen", 0x90EE90FF),
    ("lightgrey", 0xD3D3D3FF),
    ("lightpink", 0xFFB6C1FF),
    ("lightsalmon", 0xFFA07AFF),
    ("lightseagreen", 0x20B2AAFF),
    ("lightskyblue", 0x87CEFAFF),
    ("lightslategray", 0x778899FF),
    ("lightslategrey", 0x778899FF),
    ("lightsteelblue", 0xB0C4DEFF),
    ("lightyellow", 0xFFFFE0FF),
    ("lime", 0x00FF00FF),
    ("limegreen", 0x32CD32FF),
    ("linen", 0xFAF0E6FF),
    ("magenta", 0xFF00FFFF),
    ("maroon", 0x800000FF),
    ("mediumaquamarine", 0x66CDAAFF),
    ("mediumblue", 0x0000CDFF),
    ("mediumorchid", 0xBA55D3FF),
    ("mediumpurple", 0x9370DBFF),
    ("mediumseagreen", 0x3CB371FF),
    ("mediumslateblue", 0x7B68EEFF),
    ("mediumspringgreen", 0x00FA9AFF),
    ("mediumturquoise", 0x48D1CCFF),
    ("mediumvioletred", 0xC71585FF),
    ("midnightblue", 0x191970FF),
    ("mintcream", 0xF5FFFAFF),
    ("mistyrose", 0xFFE4E1FF),
    ("moccasin", 0xFFE4B5FF),
    ("navajowhite", 0xFFDEADFF),
    ("navy", 0x000080FF),
    ("oldlace", 0xFDF5E6FF),
    ("olive", 0x808000FF),
    ("olivedrab", 0x6B8E23FF),
    ("orange", 0xFFA500FF),
    ("orangered", 0xFF4500FF),
    ("orchid", 0xDA70D6FF),
    ("palegoldenrod", 0xEEE8AAFF),
    ("palegreen", 0x98FB98FF),
    ("paleturquoise", 0xAFEEEEFF),
    ("palevioletred", 0xDB7093FF),
    ("papayawhip", 0xFFEFD5FF),
    ("peachpuff", 0xFFDAB9FF),
    ("peru", 0xCD853FFF),
    ("pink", 0xFFC0CBFF),
    ("plum", 0xDDA0DDFF),
    ("powderblue", 0xB0E0E6FF),
    ("purple", 0x800080FF),
    ("red", 0xFF0000FF),
    ("rosybrown", 0xBC8F8FFF),
    ("royalblue", 0x4169E1FF),
    ("saddlebrown", 0x8B4513FF),
    ("salmon", 0xFA8072FF),
    ("sandybrown", 0xF4A460FF),
    ("seagreen", 0x2E8B57FF),
    ("seashell", 0xFFF5EEFF),
    ("sienna", 0xA0522DFF),
    ("silver", 0xC0C0C0FF),
    ("skyblue", 0x87CEEBFF),
    ("slateblue", 0x6A5ACDFF),
    ("slategray", 0x708090FF),
    ("slategrey", 0x708090FF),
    ("snow", 0xFFFAFAFF),
    ("springgreen", 0x00FF7FFF),
    ("steelblue", 0x4682B4FF),
    ("tan", 0xD2B48CFF),
    ("teal", 0x008080FF),
    ("thistle", 0xD8BFD8FF),
    ("tomato", 0xFF6347FF),
    ("turquoise", 0x40E0D0FF),
    ("violet", 0xEE82EEFF),
    ("wheat", 0xF5DEB3FF),
    ("white", 0xFFFFFFFF),
    ("whitesmoke", 0xF5F5F5FF),
    ("yellow", 0xFFFF00FF),
    ("yellowgreen", 0x9ACD32FF),
];

// Hex digit int value
fn h(c: char) -> u8 {
    match c {
        '0'..='9' => (c as u8) - b'0',
        'a'..='f' => (c as u8) - b'a' + 10,
        'A'..='F' => (c as u8) - b'A' + 10,
        _ => 0,
    }
}

// Return Rgba from rgba
pub fn rgba_new(rgba: u32) -> Rgba {
    Rgba {
        r: ((rgba >> 24) & 0xff) as f64 / 255.0,
        g: ((rgba >> 16) & 0xff) as f64 / 255.0,
        b: ((rgba >> 8) & 0xff) as f64 / 255.0,
        a: (rgba & 0xff) as f64 / 255.0,
    }
}

// Return a string representation of the color
pub fn rgba_to_string(rgba: Rgba) -> String {
    if rgba.a == 1.0 {
        format!(
            "#{:02x}{:02x}{:02x}",
            (rgba.r * 255.0) as u8,
            (rgba.g * 255.0) as u8,
            (rgba.b * 255.0) as u8
        )
    } else {
        format!(
            "rgba({}, {}, {}, {:.2})",
            (rgba.r * 255.0) as u8,
            (rgba.g * 255.0) as u8,
            (rgba.b * 255.0) as u8,
            rgba.a
        )
    }
}

// Return rgba from (r,g,b,a)
fn rgba_from_rgba(r: u8, g: u8, b: u8, a: u8) -> u32 {
    ((r as u32) << 24) | ((g as u32) << 16) | ((b as u32) << 8) | (a as u32)
}

// Return rgba from (r,g,b)
fn rgba_from_rgb(r: u8, g: u8, b: u8) -> u32 {
    rgba_from_rgba(r, g, b, 255)
}

// Return rgb from "#RRGGBB"
fn rgba_from_hex6_string(s: &str) -> u32 {
    let r = (h(s.chars().nth(0).unwrap()) << 4) + h(s.chars().nth(1).unwrap());
    let g = (h(s.chars().nth(2).unwrap()) << 4) + h(s.chars().nth(3).unwrap());
    let b = (h(s.chars().nth(4).unwrap()) << 4) + h(s.chars().nth(5).unwrap());
    rgba_from_rgb(r, g, b)
}

// Return rgb from "#RGB"
fn rgba_from_hex3_string(s: &str) -> u32 {
    let r = (h(s.chars().nth(0).unwrap()) << 4) + h(s.chars().nth(0).unwrap());
    let g = (h(s.chars().nth(1).unwrap()) << 4) + h(s.chars().nth(1).unwrap());
    let b = (h(s.chars().nth(2).unwrap()) << 4) + h(s.chars().nth(2).unwrap());
    rgba_from_rgb(r, g, b)
}

// Return rgb from "rgb()"
fn rgba_from_rgb_string(s: &str) -> Option<u32> {
    if s.starts_with("rgb(") {
        let s = &s[4..];
        let s = s.trim_start_matches(' ');
        let mut parts = s.split(',').map(|p| p.trim().parse::<u8>().ok());
        if let (Some(r), Some(g), Some(b)) = (parts.next(), parts.next(), parts.next()) {
            return Some(rgba_from_rgb(r?, g?, b?));
        }
    }
    None
}

// Return rgb from "rgba()"
fn rgba_from_rgba_string(s: &str) -> Option<u32> {
    if s.starts_with("rgba(") {
        let s = &s[5..];
        let s = s.trim_start_matches(' ');
        let mut parts = s.split(',').map(|p| p.trim());
        if let (Some(r), Some(g), Some(b), Some(a)) = (parts.next(), parts.next(), parts.next(), parts.next()) {
            let r = r.parse::<u8>().ok()?;
            let g = g.parse::<u8>().ok()?;
            let b = b.parse::<u8>().ok()?;
            let a = a.parse::<f32>().ok()?;
            return Some(rgba_from_rgba(r, g, b, (a * 255.0) as u8));
        }
    }
    None
}

// Return rgb from "#RGB" or "#RRGGBB"
fn rgba_from_hex_string(s: &str) -> Option<u32> {
    match s.len() {
        6 => Some(rgba_from_hex6_string(s)),
        3 => Some(rgba_from_hex3_string(s)),
        _ => None,
    }
}

// Return named color value
fn rgba_from_name_string(s: &str) -> Option<u32> {
    NAMED_COLORS
        .iter()
        .find(|(name, _)| *name == s)
        .map(|(_, val)| *val)
}

// Return rgb from various formats
pub fn rgba_from_string(s: &str) -> Option<u32> {
    if s.starts_with('#') {
        rgba_from_hex_string(&s[1..])
    } else if s.starts_with("rgba(") {
        rgba_from_rgba_string(s)
    } else if s.starts_with("rgb(") {
        rgba_from_rgb_string(s)
    } else {
        rgba_from_name_string(s)
    }
}

// Inspect the given rgba color
pub fn rgba_inspect(rgba: u32) {
    println!(
        "rgba({},{},{},{})",
        (rgba >> 24) & 0xff,
        (rgba >> 16) & 0xff,
        (rgba >> 8) & 0xff,
        rgba & 0xff
    );
}