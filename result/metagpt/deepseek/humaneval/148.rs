const PLANETS: [&str; 8] = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"];

fn find_planet_index(planet: &str) -> Option<usize> {
    PLANETS.iter().position(|&p| p == planet)
}

pub fn bf(planet1: &str, planet2: &str) -> Vec<String> {
    let pos1 = match find_planet_index(planet1) {
        Some(p) => p,
        None => return vec![],
    };
    
    let pos2 = match find_planet_index(planet2) {
        Some(p) => p,
        None => return vec![],
    };
    
    let (start, end) = if pos1 < pos2 {
        (pos1, pos2)
    } else {
        (pos2, pos1)
    };
    
    if end - start <= 1 {
        return vec![];
    }
    
    PLANETS[start+1..end]
        .iter()
        .map(|s| s.to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bf() {
        assert_eq!(bf("Earth", "Mars"), vec![] as Vec<String>);
        assert_eq!(bf("Earth", "Jupiter"), vec!["Mars".to_string()]);
        assert_eq!(bf("Jupiter", "Earth"), vec!["Mars".to_string()]);
        assert_eq!(bf("Invalid", "Earth"), vec![] as Vec<String>);
    }
}