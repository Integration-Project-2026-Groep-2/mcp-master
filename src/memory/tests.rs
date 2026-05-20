use super::chunker::chunk_ranges;

#[test]
fn chunk_ranges_respect_overlap_and_boundaries() {
    let text = "abcdefghijklmnopqrstuvwxyz";
    let ranges = chunk_ranges(text, 10, 2);
    assert!(!ranges.is_empty());
    assert_eq!(&text[ranges[0].clone()], "abcdefghij");
    assert_eq!(&text[ranges[1].clone()], "ijklmnopqr");
}
