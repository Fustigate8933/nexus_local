//! OCR abstraction for Nexus Local.
//
// Provides a trait for extracting text from images and scanned documents using OCR.

use std::path::PathBuf;
use std::fs;
use async_trait::async_trait;
use anyhow::Result;

use leptess::LepTess;
use poppler::PopplerDocument;

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
			"png" | "jpg" | "jpeg" => {
				let mut lt = LepTess::new(None, "eng")?;
				lt.set_image(path);
				let text = lt.get_utf8_text()?;
				Ok(text)
			}
			"pdf" => {
				let mut data = fs::read(path)?;
				let doc = PopplerDocument::new_from_data(&mut data, None).map_err(|e| anyhow::anyhow!("Failed to open PDF: {:?}", e))?;
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

/// Stub for future PDF/image OCR implementation
pub struct StubOcr;

#[async_trait]
impl OcrEngine for StubOcr {
	async fn extract_text(&self, _path: &PathBuf) -> Result<String> {
		Ok("[OCR not implemented]".to_string())
	}
}
