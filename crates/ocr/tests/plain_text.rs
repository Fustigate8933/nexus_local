use ocr::{OcrEngine, PlainTextExtractor};
use std::path::PathBuf;
use anyhow::Result;
use async_trait::async_trait;

#[tokio::test]
async fn test_plain_text_extractor_txt() -> Result<()> {
    // Create a temporary .txt file
    let path = PathBuf::from("test_file.txt");
    std::fs::write(&path, "Hello, world!\nThis is a test.")?;
    let extractor = PlainTextExtractor;
    let text = extractor.extract_text(&path).await?;
    assert!(text.contains("Hello, world!"));
    std::fs::remove_file(&path)?;
    Ok(())
}

#[tokio::test]
async fn test_plain_text_extractor_md() -> Result<()> {
    // Create a temporary .md file
    let path = PathBuf::from("test_file.md");
    std::fs::write(&path, "# Title\nSome markdown content.")?;
    let extractor = PlainTextExtractor;
    let text = extractor.extract_text(&path).await?;
    assert!(text.contains("Title"));
    std::fs::remove_file(&path)?;
    Ok(())
}
