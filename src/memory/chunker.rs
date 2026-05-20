use std::ops::Range;

pub fn chunk_ranges(text: &str, chunk_chars: usize, overlap_chars: usize) -> Vec<Range<usize>> {
    if text.is_empty() {
        return Vec::new();
    }

    let chunk_chars = chunk_chars.max(1);
    let overlap_chars = overlap_chars.min(chunk_chars.saturating_sub(1));

    let char_starts: Vec<usize> = text
        .char_indices()
        .map(|(idx, _)| idx)
        .chain(std::iter::once(text.len()))
        .collect();
    let total_chars = char_starts.len().saturating_sub(1);

    let mut ranges = Vec::new();
    let mut start_char = 0usize;
    while start_char < total_chars {
        let end_char = (start_char + chunk_chars).min(total_chars);
        let start_byte = char_starts[start_char];
        let end_byte = char_starts[end_char];
        if start_byte < end_byte {
            ranges.push(start_byte..end_byte);
        }
        if end_char == total_chars {
            break;
        }
        let next_start = end_char.saturating_sub(overlap_chars);
        start_char = next_start.max(start_char + 1);
    }

    ranges
}

pub fn chunk_text(text: &str, chunk_chars: usize, overlap_chars: usize) -> Vec<&str> {
    chunk_ranges(text, chunk_chars, overlap_chars)
        .into_iter()
        .map(|range| &text[range])
        .collect()
}
