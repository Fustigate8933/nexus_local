//! OCR abstraction for Nexus Local.
//
// Provides a trait for extracting text from images and scanned documents using OCR.

use std::path::PathBuf;
use std::fs;
use async_trait::async_trait;
use anyhow::Result;

/// Trait for OCR text extraction from images or PDFs.
#[async_trait]
pub trait OcrEngine: Send + Sync {
	async fn extract_text(&self, path: &PathBuf) -> Result<String>;
}

/// Implementation for extracting text from .txt and .md files.
pub struct PlainTextExtractor;

#[async_trait]
impl OcrEngine for PlainTextExtractor {
	async fn extract_text(&self, path: &PathBuf) -> Result<String> {
		let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
		match ext.as_str() {
			"txt" | "md" => {
				let text = fs::read_to_string(path)?;
				Ok(text)
			}
			// TODO: Add PDF/image OCR support here
			_ => Ok(String::new()),
		}
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
