fn parse_music(music_string: &str) -> Vec<i32> {
    let mut out = Vec::new();
    let mut current = String::new();

    for c in music_string.chars() {
        if c == ' ' {
            match current.as_str() {
                "o" => out.push(4),
                "o|" => out.push(2),
                ".|" => out.push(1),
                _ => {},
            }
            current.clear();
        } else {
            current.push(c);
        }
    }

    out
}

fn main() {
    let music_string = "o o| .| o";
    let parsed_music = parse_music(music_string);
    println!("Parsed music: {:?}", parsed_music);
}