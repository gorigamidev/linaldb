/// Parse de algo como: [1, 3, 224, 224]
pub fn parse_usize_list(text: &str) -> Result<Vec<usize>, String> {
    let inner = text.trim();
    if !inner.starts_with('[') || !inner.ends_with(']') {
        return Err(format!("Expected [d1, d2, ...], got: {}", text));
    }
    let inner = &inner[1..inner.len() - 1]; // sin [ ]
    if inner.trim().is_empty() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for part in inner.split(',') {
        let p = part.trim();
        let n: usize = p.parse().map_err(|_| format!("Invalid dimension: {}", p))?;
        out.push(n);
    }
    Ok(out)
}

/// Parse de algo como: [1, 0, 0] a Vec<f32>
pub fn parse_f32_list(text: &str) -> Result<Vec<f32>, String> {
    let inner = text.trim();
    if !inner.starts_with('[') || !inner.ends_with(']') {
        return Err(format!("Expected [v1, v2, ...], got: {}", text));
    }
    let inner = &inner[1..inner.len() - 1]; // sin [ ]
    if inner.trim().is_empty() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for part in inner.split(',') {
        let p = part.trim();
        let n: f32 = p.parse().map_err(|_| format!("Invalid float: {}", p))?;
        out.push(n);
    }
    Ok(out)
}
