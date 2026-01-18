use ocr::{OcrEngine, PlainTextExtractor};
use std::path::PathBuf;
use anyhow::Result;

#[tokio::test]
async fn test_pdf_text_extraction() -> Result<()> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/brainfuck.pdf");
    println!("PDF path: {:?}, exists: {}", path, path.exists());
    if !path.exists() {
        eprintln!("brainfuck.pdf not found, skipping PDF extraction test");
        return Ok(());
    }
    let extractor = PlainTextExtractor;
    let text = extractor.extract_text(&path).await?;
    println!("Extracted PDF text:\n{}", text);
    assert!(!text.trim().is_empty(), "PDF extraction should return some text");
    Ok(())
}
