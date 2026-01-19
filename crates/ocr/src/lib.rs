//! OCR abstraction for Nexus Local.
//!
//! Provides a trait for extracting text from images and scanned documents using OCR.
//! Images are automatically resized before OCR to limit memory usage.

use std::path::PathBuf;
use std::fs;
use async_trait::async_trait;
use anyhow::Result;

use leptess::LepTess;
use poppler::PopplerDocument;
use image::GenericImageView;
use tempfile::NamedTempFile;

/// Maximum dimension (width or height) for images before OCR.
/// Larger images are downscaled to fit within this limit.
const MAX_IMAGE_DIMENSION: u32 = 2000;

/// Trait for OCR text extraction from images or PDFs.
#[async_trait]
pub trait OcrEngine: Send + Sync {
    async fn extract_text(&self, path: &PathBuf) -> Result<String>;
}

/// Sync trait for parallel text extraction with Rayon.
pub trait SyncOcrEngine: Send + Sync {
    fn extract_text_sync(&self, path: &PathBuf) -> Result<String>;
}

/// Preprocesses an image: loads it, resizes if needed, saves to temp file.
/// Returns the path to use for OCR (either original or temp file).
fn preprocess_image(path: &PathBuf) -> Result<(PathBuf, Option<NamedTempFile>)> {
    let img = image::open(path)?;
    let (width, height) = img.dimensions();
    
    // Check if resizing is needed
    if width <= MAX_IMAGE_DIMENSION && height <= MAX_IMAGE_DIMENSION {
        // Image is small enough, use original
        return Ok((path.clone(), None));
    }
    
    // Calculate new dimensions preserving aspect ratio
    let scale = if width > height {
        MAX_IMAGE_DIMENSION as f64 / width as f64
    } else {
        MAX_IMAGE_DIMENSION as f64 / height as f64
    };
    
    let new_width = (width as f64 * scale) as u32;
    let new_height = (height as f64 * scale) as u32;
    
    eprintln!("  resizing: {}x{} -> {}x{}", width, height, new_width, new_height);
    
    // Resize using Lanczos3 for quality
    let resized = img.resize(new_width, new_height, image::imageops::FilterType::Lanczos3);
    
    // Save to temp file (PNG for lossless)
    let temp_file = NamedTempFile::with_suffix(".png")?;
    resized.save(temp_file.path())?;
    
    Ok((temp_file.path().to_path_buf(), Some(temp_file)))
}

/// Implementation for extracting text from various file types.
pub struct PlainTextExtractor;

impl PlainTextExtractor {
    /// Core sync extraction logic, used by both async and sync traits.
    fn do_extract(&self, path: &PathBuf) -> Result<String> {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
        match ext.as_str() {
            "txt" | "md" => {
                let text = fs::read_to_string(path)?;
                Ok(text)
            }
            "png" | "jpg" | "jpeg" | "webp" | "bmp" | "tiff" | "tif" => {
                // Preprocess image (resize if needed)
                let (ocr_path, _temp_file) = preprocess_image(path)?;
                
                let mut lt = LepTess::new(None, "eng")?;
                lt.set_image(&ocr_path)?;
                let text = lt.get_utf8_text()?;
                
                // _temp_file is dropped here, cleaning up the temp file
                Ok(text)
            }
            "pdf" => {
                let mut data = fs::read(path)?;
                let doc = PopplerDocument::new_from_data(&mut data, None)
                    .map_err(|e| anyhow::anyhow!("Failed to open PDF: {:?}", e))?;
                
                let mut text = String::new();
                for page in doc.pages() {
                    if let Some(page_text) = page.get_text() {
                        text.push_str(page_text);
                        text.push_str("\n");
                    }
                }
                Ok(text)
            }
            _ => Ok(String::new()),
        }
    }
}

#[async_trait]
impl OcrEngine for PlainTextExtractor {
    async fn extract_text(&self, path: &PathBuf) -> Result<String> {
        self.do_extract(path)
    }
}

impl SyncOcrEngine for PlainTextExtractor {
    fn extract_text_sync(&self, path: &PathBuf) -> Result<String> {
        self.do_extract(path)
    }
}

/// Stub for future PDF/image OCR implementation
pub struct StubOcr;

#[async_trait]
impl OcrEngine for StubOcr {
    async fn extract_text(&self, _path: &PathBuf) -> Result<String> {
        Ok("[OCR not implemented]".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_preprocess_small_image_no_resize() {
        // A small image should not be resized
        // This test would need a real small image file
    }

    #[tokio::test]
    async fn test_plain_text_extraction() {
        let extractor = PlainTextExtractor;
        let path = PathBuf::from("src/lib.rs");
        let result = extractor.extract_text(&path).await;
        assert!(result.is_ok());
        assert!(result.unwrap().contains("OcrEngine"));
    }
}
