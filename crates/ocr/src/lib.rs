//! OCR abstraction for Nexus Local.
//
// Provides a trait for extracting text from images and scanned documents using OCR.

use std::path::PathBuf;
use async_trait::async_trait;
use anyhow::Result;

/// Trait for OCR text extraction from images or PDFs.
#[async_trait]
pub trait OcrEngine: Send + Sync {
	async fn extract_text(&self, path: &PathBuf) -> Result<String>;
}

// Example stub implementation (to be replaced with real OCR backend)
pub struct DummyOcr;

#[async_trait]
impl OcrEngine for DummyOcr {
	async fn extract_text(&self, _path: &PathBuf) -> Result<String> {
		Ok("dummy ocr text".to_string())
	}
}
