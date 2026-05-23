use super::*;

#[test]
fn suggestions_prompt_requires_dutch_and_strict_format() {
    let p = SUGGESTIONS_SYSTEM_PROMPT;
    assert!(p.contains("Nederlands"));
    assert!(p.contains("EXACT 3"));
    assert!(p.contains("\"items\""));
    assert!(p.contains("\"tool\""));
    assert!(p.contains("<UNTRUSTED>"));
}
