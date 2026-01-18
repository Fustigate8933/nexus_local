use ocr::{OcrEngine, PlainTextExtractor};
use std::path::PathBuf;
use anyhow::Result;

#[tokio::test]
async fn test_image_ocr() -> Result<()> {
    // This test expects a sample image file with text. You can place a test.png in the project root.
    let path = PathBuf::from("tests/ocr_test.png");
    if !path.exists() {
        eprintln!("ocr_test.png not found, skipping image OCR test");
        return Ok(());
    }
    let extractor = PlainTextExtractor;
    let text = extractor.extract_text(&path).await?;
    let expected = "HOW TO COMBINE\nTEXT AND IMAGE\nIN ELEARNING DESIGN";
    println!("Expected OCR text:\n{}\nExtracted OCR text:\n{}", expected, text);
    assert!(text.to_uppercase().contains("HOW TO COMBINE"), "OCR output should contain the expected phrase");
    assert!(!text.trim().is_empty(), "OCR should extract some text from image");
    Ok(())
}
