mod buffer;

fn main() {
    // Create a new buffer with default size
    let mut buf = buffer::Buffer::new();
    println!("New buffer created with size: {}", buf.size());

    // Append a string to the buffer
    if buf.append("Hello, ").is_ok() {
        println!("Buffer after append: {}", String::from_utf8_lossy(&buf.data));
    } else {
        println!("Failed to append to buffer");
    }

    // Append another string
    if buf.append("world!").is_ok() {
        println!("Buffer after second append: {}", String::from_utf8_lossy(&buf.data));
    } else {
        println!("Failed to append to buffer");
    }

    // Print the buffer
    buf.print();

    // Resize the buffer
    if buf.resize(128).is_ok() {
        println!("Buffer resized to: {}", buf.size());
    } else {
        println!("Failed to resize buffer");
    }

    // Compact the buffer
    if let Ok(removed) = buf.compact() {
        println!("Buffer compacted, bytes removed: {}", removed);
    } else {
        println!("Failed to compact buffer");
    }

    // Clear the buffer
    buf.clear();
    println!("Buffer cleared, current size: {}", buf.size());

    // Fill the buffer with a specific byte
    buf.fill(b'x');
    println!("Buffer filled with 'x': {}", String::from_utf8_lossy(&buf.data));

    // Trim the buffer
    buf.trim();
    println!("Buffer after trim: {}", String::from_utf8_lossy(&buf.data));
}